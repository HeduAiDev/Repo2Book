"""
Reference Self-Attention — Reimplementation grounded in vLLM source.

Every function references the exact vLLM source file and line numbers.
This is NOT a generic attention tutorial — it mirrors vLLM's architecture.

vLLM Architecture Reference:
    Attention layer:     vllm/model_executor/layers/attention/attention.py:L177
    Backend abstraction: vllm/v1/attention/backend.py (AttentionBackend, AttentionImpl)
    FlashAttention impl: vllm/v1/attention/backends/flash_attn.py
    Backend selector:    vllm/v1/attention/selector.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# 1.1 Scaled Dot-Product Attention — The Core Operator
# ═══════════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Mathematical attention — what vLLM's backends compute under the hood.

    REFERENCE: vllm/v1/attention/backends/flash_attn.py
               → FlashAttentionImpl.forward() calls flash_attn_varlen_func()
               which computes this exact operation, but tiled in SRAM.

    vLLM does NOT have this function as-is. Instead, the Attention layer
    (attention.py:L177) delegates to self.impl.forward() which is a
    backend-specific implementation. All backends compute the same math.

    The formula every backend must satisfy:
        Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    """
    d_k = Q.size(-1)
    if scale is None:
        scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


# ═══════════════════════════════════════════════════════════════════════════
# 1.2 Multi-Head Attention — Mirroring vLLM's Attention class
# ═══════════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Our reimplementation of vLLM's Attention layer.

    REFERENCE: vllm/model_executor/layers/attention/attention.py:L177-L582
               Class: Attention (the main attention layer in vLLM)

    KEY DIFFERENCES from vLLM's Attention:
    1. vLLM uses a BACKEND abstraction. Attention.__init__ selects a backend
       via get_attn_backend() (vllm/v1/attention/selector.py) and creates
       self.impl: AttentionImpl. All computation is delegated to the backend.
       We inline the math for clarity.

    2. vLLM does NOT define QKV projections inside Attention. The projections
       (q_proj, k_proj, v_proj) are defined in model files (e.g.,
       vllm/model_executor/models/llama.py → LlamaAttention).
       We include them here so the chapter is self-contained.

    3. vLLM uses torch.ops.vllm.unified_attention_with_output — an opaque
       custom op that wraps the backend call. This prevents torch.compile
       from graph-breaking on the attention kernel. We skip this optimization.

    4. vLLM supports: FP8 KV cache, sliding window, ALiBi, logits soft-capping,
       cascade attention, KV sharing (Medusa), attention sinks. We skip all
       of these for clarity — they are covered in later chapters.
    """

    def __init__(self, d_model: int, num_heads: int, bias: bool = False):
        super().__init__()
        # REFERENCE: attention.py:L177-L376 — Attention.__init__()
        # vLLM takes (num_heads, head_size, scale, num_kv_heads, ...) — NOT d_model.
        # Check at L278: assert num_heads % num_kv_heads == 0
        # At L344-L357: self.impl = impl_cls(num_heads, head_size, scale, ...)
        #   → ALL computation is delegated to the backend via self.impl
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # REFERENCE: model files (e.g., llama.py → LlamaAttention.__init__)
        # vLLM uses a combined qkv_proj for efficiency (one matmul → split).
        # We use three separate projections for pedagogical clarity.
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        # REFERENCE: attention.py:L193, L345 — scale is passed as a constructor
        # parameter (pre-computed by model config). In the impl: self.scale = float(scale).
        # We compute it here since our class owns the head_dim, unlike vLLM.
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        REFERENCE: attention.py:L410-L450 — Attention.forward()
        vLLM reshapes Q/K/V to [num_tokens, num_heads, head_size] before
        passing to the backend. In our implementation, this is the key
        operation that enables multi-head parallelism.
        """
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _reshape_from_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse of _reshape_for_heads — concatenates heads back."""
        B, _, L, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(
        self, hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        REFERENCE: attention.py:L410-L530 — Attention.forward()

        vLLM's forward does:
        1. Reshape Q/K/V to [num_tokens, num_heads, head_size]
        2. Allocate output tensor (empty_like)
        3. Call torch.ops.vllm.unified_attention_with_output(Q, K, V, ...)
           → which calls self.impl.forward() → FlashAttention/Triton/etc.
        4. Return output

        We compute the attention explicitly for learning purposes.
        """
        Q = self._reshape_for_heads(self.W_q(hidden_states))
        K = self._reshape_for_heads(self.W_k(hidden_states))
        V = self._reshape_for_heads(self.W_v(hidden_states))

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        output = self.W_o(self._reshape_from_heads(attn_output))
        return output, attn_weights


# ═══════════════════════════════════════════════════════════════════════════
# 1.3 GQA — Mirroring vLLM's num_kv_heads parameter
# ═══════════════════════════════════════════════════════════════════════════

class GroupedQueryAttention(nn.Module):
    """
    GQA — Our reimplementation with vLLM's convention.

    REFERENCE: attention.py:L177 — Attention class
               The same Attention class handles GQA by accepting num_kv_heads.
               vLLM's FlashAttention backend handles GQA natively in the kernel,
               without expanding K,V in memory. We expand explicitly for clarity.

    vLLM's KV cache shape (preview of Chapter 2):
        [2, num_blocks, block_size, num_kv_heads, head_size]
    Note: num_kv_heads, NOT num_heads. This is the GQA memory win.
    """

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int,
                 bias: bool = False):
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must divide num_kv_heads ({num_kv_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        # REFERENCE: In vLLM's model files, k_proj and v_proj output
        # num_kv_heads * head_dim, NOT d_model. This is the GQA parameter saving.
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self, hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = hidden_states.shape

        Q = self.W_q(hidden_states)
        K = self.W_k(hidden_states)
        V = self.W_v(hidden_states)

        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # REFERENCE: flash_attn.py — FlashAttentionImpl.forward()
        # vLLM's flash attention kernel handles GQA natively — it reads K,V
        # with a stride of num_kv_heads, without ever expanding in memory.
        # We expand here so you can SEE the sharing pattern.
        if self.num_kv_heads != self.num_heads:
            K = K.repeat_interleave(self.num_queries_per_kv, dim=1)
            V = V.repeat_interleave(self.num_queries_per_kv, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, L, self.d_model)
        return self.W_o(attn_output), attn_weights


# ═══════════════════════════════════════════════════════════════════════════
# 1.5 Attention Masks
# REFERENCE: vllm/v1/attention/backends/flash_attn.py
#            → FlashAttentionMetadataBuilder.build() constructs per-request metadata
#            including seq_lens, block_tables, and mask parameters.
#            The actual masking happens INSIDE the kernel — vLLM does not
#            materialize mask tensors. We create them explicitly for visualization.
# ═══════════════════════════════════════════════════════════════════════════

def create_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """Causal mask — GPT-style decoder attention."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
                     ).unsqueeze(0).unsqueeze(0)


def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Padding mask — variable-length batch support."""
    B = lengths.size(0)
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return (positions < lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(2)


def create_sliding_window_mask(seq_len: int, window_size: int,
                               device=None) -> torch.Tensor:
    """Sliding window mask — Mistral/Gemma attention pattern."""
    positions = torch.arange(seq_len, device=device)
    dist = positions.unsqueeze(1) - positions.unsqueeze(0)
    return ((dist >= 0) & (dist < window_size)).unsqueeze(0).unsqueeze(0)
