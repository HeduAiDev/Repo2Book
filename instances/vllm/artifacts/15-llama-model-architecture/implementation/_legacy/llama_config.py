"""
Llama-3.2-1B Architecture — Model Config & Layer Structure.

REFERENCE sources:
    LlamaForCausalLM:    vllm/model_executor/models/llama.py:L501
    LlamaModel:          vllm/model_executor/models/llama.py:L350
    LlamaDecoderLayer:   vllm/model_executor/models/llama.py:L253
    LlamaAttention:      vllm/model_executor/models/llama.py:L124
    LlamaMLP:            vllm/model_executor/models/llama.py:L81
    RMSNorm:             vllm/model_executor/layers/layernorm.py:L103
    Llama3RoPE:          vllm/model_executor/layers/rotary_embedding/llama3_rope.py:L11
    SiluAndMul:          vllm/model_executor/layers/activation.py:L118

Actual Llama-3.2-1B config (verified):
    vocab_size=128256, hidden_size=2048, intermediate_size=8192
    num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8
    head_dim=64, max_position_embeddings=131072
    rms_norm_eps=1e-5, rope_theta=500000.0
    rope_type="llama3" with partial NTK scaling (factor=32.0)

Layer structure (from llama.py:L288-L314):
    input → input_layernorm(RMSNorm) → self_attn(LlamaAttention) → residual
          → post_attention_layernorm(RMSNorm) → mlp(LlamaMLP) → residual → output

LlamaAttention (from llama.py:L164-L221):
    hidden → qkv_proj(QKVParallelLinear) → split[Q,K,V] → RoPE → Attention → o_proj(RowParallelLinear)

LlamaMLP (from llama.py:L81-L121):
    hidden → gate_up_proj(MergedColumnParallelLinear) → SiluAndMul → down_proj(RowParallelLinear)

Fused weights (from llama.py:L436-L460, stacked_params_mapping):
    q_proj + k_proj + v_proj → qkv_proj (saved as one fused tensor)
    gate_proj + up_proj → gate_up_proj (saved as one fused tensor)
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Llama-3.2-1B Config
# REFERENCE: transformers.LlamaConfig (used by vLLM at llama.py:L32)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LlamaConfig:
    """
    Llama-3.2-1B configuration.

    REFERENCE: HuggingFace config + vLLM llama.py parameter usage
    """
    vocab_size: int = 128256
    hidden_size: int = 2048           # d_model
    intermediate_size: int = 8192      # SwiGLU FFN intermediate
    num_hidden_layers: int = 16        # 16 transformer layers (1B model)
    num_attention_heads: int = 32      # Total Q heads
    num_key_value_heads: int = 8       # KV heads (GQA: 4 queries per KV)
    head_dim: int = 64                 # 2048 / 32 = 64
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_type: str = "llama3"
    rope_factor: float = 32.0          # NTK scaling factor
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192

    @property
    def num_queries_per_kv(self) -> int:
        """GQA: how many Q heads share one KV head."""
        return self.num_attention_heads // self.num_key_value_heads  # 32/8 = 4


# ═══════════════════════════════════════════════════════════════════════════
# RMSNorm
# REFERENCE: vllm/model_executor/layers/layernorm.py:L103
# ═══════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    REFERENCE: layernorm.py:L103-L351

    Math:
        RMSNorm(x) = x / RMS(x) * gamma
        RMS(x) = sqrt(mean(x^2) + eps)

    Why RMSNorm instead of LayerNorm?
        LayerNorm centers (subtracts mean) AND scales (divides by std).
        RMSNorm only scales — no centering. The centering step was shown
        to be unnecessary for transformer training stability, and removing
        it saves compute (no mean subtraction). Llama uses RMSNorm throughout.

    vLLM fuses the residual addition into RMSNorm for efficiency:
        fused_add_rms_norm(x, residual) → x_normed, residual_updated
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        # REFERENCE: layernorm.py:L188-L231 forward_static (PyTorch fallback)
        if residual is not None:
            x = x + residual
        # Compute in fp32 for numerical stability
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_f32 * torch.rsqrt(variance + self.eps)
        return (x_normed * self.weight).to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# RoPE (simplified — full Triton version in Chapter 17)
# REFERENCE: vllm/model_executor/layers/rotary_embedding/__init__.py:L33
#            vllm/model_executor/layers/rotary_embedding/llama3_rope.py:L11
# ═══════════════════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (simplified PyTorch version).

    REFERENCE: rotary_embedding/__init__.py → get_rope() factory

    Math:
        RoPE(x_i, pos) = rotate(x_{2i}, x_{2i+1}) by theta_i * pos
        where theta_i = base^(-2i/d)

    For Llama 3 with NTK scaling:
        High frequencies (short wavelengths): unchanged
        Low frequencies (long wavelengths): scaled by factor
        Mid frequencies: smoothly interpolated

    Our simplified version uses the standard RoPE without NTK for clarity.
    Full Llama 3 NTK scaling will be implemented in Chapter 17.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 131072,
                 theta: float = 500000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute cos/sin cache
        # REFERENCE: base.py:L118 — _compute_inv_freq and cos_sin cache
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)  # [seq, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)    # [seq, head_dim]
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to Q and K in-place.

        REFERENCE: base.py:L240-L303 — forward_native (neox-style)
        Llama uses neox-style: rotate pairs (x_0,x_1), (x_2,x_3), ...
        """
        cos = self.cos_cached[positions]  # [seq, head_dim]
        sin = self.sin_cached[positions]

        # Neox-style: half the dimensions are rotated
        q_rot = q.float() * cos + self._rotate_half(q.float()) * sin
        k_rot = k.float() * cos + self._rotate_half(k.float()) * sin
        return q_rot.to(q.dtype), k_rot.to(k.dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Neox-style: swap half-pairs with sign flip."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# SwiGLU Activation
# REFERENCE: vllm/model_executor/layers/activation.py:L118 — SiluAndMul
# ═══════════════════════════════════════════════════════════════════════════

class SiLUActivation(nn.Module):
    """SiLU (Sigmoid Linear Unit) = Swish."""
    def forward(self, x):
        return x * torch.sigmoid(x)


# ═══════════════════════════════════════════════════════════════════════════
# Model Parameter Calculator
# ═══════════════════════════════════════════════════════════════════════════

def count_parameters(config: LlamaConfig, dtype_bytes: int = 2) -> dict:
    """
    Count total and per-component parameters for Llama-3.2-1B.

    REFERENCE: vLLM measures this via DeviceMemoryProfiler during loading.
    We compute analytically.

    The fused weights (qkv_proj, gate_up_proj) combine multiple projections.
    When counting from config, account for the fusion.
    """
    d = config.hidden_size
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv = config.num_key_value_heads
    hd = config.head_dim
    inter = config.intermediate_size
    vocab = config.vocab_size

    # Per-layer parameters
    qkv_params = d * (n_heads * hd + 2 * n_kv * hd)  # Q + K + V projections
    o_params = d * n_heads * hd                        # Output projection
    gate_up_params = d * inter * 2                     # gate + up (fused)
    down_params = inter * d                            # down projection
    norm_params = d * 2                                 # 2 × RMSNorm per layer

    per_layer = qkv_params + o_params + gate_up_params + down_params + norm_params

    # Non-layer parameters
    embed_params = vocab * d                            # token embedding
    lm_head_params = vocab * d                          # output projection

    total = per_layer * n_layers + embed_params + lm_head_params

    return {
        "total_params": total,
        "total_params_M": round(total / 1e6, 1),
        "total_size_gb": round(total * dtype_bytes / (1024**3), 2),
        "per_layer_params_M": round(per_layer / 1e6, 2),
        "embedding_params_M": round(embed_params / 1e6, 1),
        "lm_head_params_M": round(lm_head_params / 1e6, 1),
        "breakdown": {
            "attention": round((qkv_params + o_params) * n_layers / 1e6, 1),
            "mlp": round((gate_up_params + down_params) * n_layers / 1e6, 1),
            "norms": round(norm_params * n_layers / 1e6, 3),
            "embeddings": round((embed_params + lm_head_params) / 1e6, 1),
        },
    }


def demonstrate():
    config = LlamaConfig()
    params = count_parameters(config)

    print("Llama-3.2-1B Architecture")
    print("=" * 60)
    print(f"Layers: {config.num_hidden_layers}")
    print(f"d_model: {config.hidden_size}")
    print(f"Attention: {config.num_attention_heads} Q heads, "
          f"{config.num_key_value_heads} KV heads (GQA {config.num_queries_per_kv}:1)")
    print(f"Head dim: {config.head_dim}")
    print(f"Intermediate: {config.intermediate_size} ({config.intermediate_size/config.hidden_size:.0f}× d_model)")
    print(f"Vocab: {config.vocab_size}")
    print()
    print(f"Total parameters: {params['total_params_M']}M")
    print(f"Model size (bf16): {params['total_size_gb']} GB")
    print(f"Per layer: {params['per_layer_params_M']}M")
    print()
    print("Parameter breakdown:")
    for k, v in params["breakdown"].items():
        print(f"  {k}: {v}M")


if __name__ == "__main__":
    demonstrate()
