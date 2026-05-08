"""DeepSeek-V3 MTP head architecture — pedagogical mirror.

The canonical MTP head is ``DeepSeekMultiTokenPredictorLayer`` at
``vllm/model_executor/models/deepseek_mtp.py:L63-L122``. Each layer is:

    enorm(input_ids_embed) ─┐
                            ├─ concat ─→ eh_proj ─→ mtp_block ─→ shared_head
    hnorm(target_hidden) ───┘                       (DeepseekV2DecoderLayer
                                                     — full transformer block!)

**Trap E** (writer must call out): the ``mtp_block`` is a *full* DeepSeek
decoder layer including the MoE block (`deepseek_mtp.py:L92-L97`). It is
NOT a lightweight MLP. Medusa heads ARE lightweight; MTP heads are heavy.

The ``SharedHead`` (`deepseek_mtp.py:L43-L62`) is RMSNorm + ParallelLMHead;
the LM head weight is shared with the target via `_maybe_share_lm_head`
(`llm_base_proposer.py`). Saves ~`vocab_size × hidden` per MTP layer.

We replace the MoE-laden ``DeepseekV2DecoderLayer`` with a plain dense
transformer block (RMSNorm + multi-head attention + RMSNorm + SwiGLU FFN)
because the MoE machinery lives in Ch09. The ``enorm + hnorm + eh_proj``
fusion is reproduced verbatim — that's the MTP-specific piece.
"""

# REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L43-L122
# REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L124-L184

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm — same impl as Ch01 / Ch16 will deep-dive
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square layer norm.

    Same as ``vllm.model_executor.layers.layernorm.RMSNorm`` algorithmically.
    Pedagogical impl — we don't need the fused Triton path here.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # REFERENCE: vllm/model_executor/layers/layernorm.py — same formula
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return (self.weight * x_norm).to(x.dtype)


# ---------------------------------------------------------------------------
# Attention + MLP — minimal dense transformer block standing in for
# DeepseekV2DecoderLayer.
# ---------------------------------------------------------------------------


class _MultiHeadAttention(nn.Module):
    """Plain MHA. Replaces DeepseekV2's MLA for clarity (Ch27 covers MLA)."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, H = x.shape
        # Reshape qkv → [T, 3, num_heads, head_dim], then permute to put num_heads first
        # so self-attention runs over T tokens within each head.
        qkv = self.qkv(x).view(T, 3, self.num_heads, self.head_dim)
        # [num_heads, T, head_dim] per q/k/v
        q = qkv[:, 0].transpose(0, 1)
        k = qkv[:, 1].transpose(0, 1)
        v = qkv[:, 2].transpose(0, 1)
        scale = self.head_dim**-0.5
        # Causal attention over the T positions in this proposed segment.
        attn = (q @ k.transpose(-1, -2)) * scale  # [num_heads, T, T]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        attn = (attn + mask).softmax(dim=-1)
        out = attn @ v  # [num_heads, T, head_dim]
        # Permute back: [T, num_heads, head_dim] → [T, H]
        return self.o_proj(out.transpose(0, 1).reshape(T, H))


class _DenseFFN(nn.Module):
    """SwiGLU FFN — pedagogical replacement for DeepSeek's MoE FFN.

    Real DeepSeek MTP layer's FFN is a full MoE block with hundreds of
    experts (Ch09). We use a single SwiGLU here because:
      - the parameter count ratio MTP-vs-Medusa demo doesn't depend on FFN
        topology, only on the number of params.
      - the Trap-E "MTP heads are heavy" point is about transformer-block
        machinery, not specifically about MoE.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MTPBlock(nn.Module):
    """Stand-in for ``DeepseekV2DecoderLayer`` inside the MTP head.

    Real layer: input_layernorm → MLA → post_attn_layernorm → MoE FFN.
    Our stand-in: input_layernorm → MHA → post_attn_layernorm → SwiGLU FFN.
    The transformer-block-shaped weight is the WHOLE point — Trap E.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int) -> None:
        super().__init__()
        # REFERENCE: vllm/model_executor/models/deepseek_v2.py DeepseekV2DecoderLayer
        self.input_layernorm = RMSNorm(hidden_size)
        self.attn = _MultiHeadAttention(hidden_size, num_heads)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = _DenseFFN(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        h = self.attn(self.input_layernorm(hidden_states))
        h = h + residual
        residual = h
        h = self.mlp(self.post_attention_layernorm(h))
        return h, residual


# ---------------------------------------------------------------------------
# SharedHead and the MTP layer itself.
# ---------------------------------------------------------------------------


class SharedHead(nn.Module):
    """RMSNorm + LM head, mirror of ``deepseek_mtp.py:L43-L62``.

    Note: in vLLM the LM-head weight is shared with the target's lm_head
    via ``_maybe_share_lm_head`` (``llm_base_proposer.py:L1471+``). We
    expose a ``share_lm_head_with(target)`` method for our mirror.
    """

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        # REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L52-L57
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The DeepSeek source returns the *normalized* hidden states from
        # forward(); the LM-head matmul happens in compute_logits. We follow.
        # REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L59-L60
        return self.norm(hidden_states)

    def compute_logits(self, normed: torch.Tensor) -> torch.Tensor:
        return self.head(normed)

    def share_lm_head_with(self, target_lm_head: nn.Linear) -> None:
        """Tie this MTP head's lm_head weight to the target's.

        # REFERENCE: vllm/v1/spec_decode/llm_base_proposer.py _maybe_share_lm_head
        """
        # Same Parameter object → tied weights → no extra params.
        self.head.weight = target_lm_head.weight


@dataclass
class MTPLayerStats:
    """Parameter accounting for one MTP layer."""

    enorm: int
    hnorm: int
    eh_proj: int
    mtp_block: int
    shared_head_norm: int
    shared_head_lm: int  # excluded if shared with target

    def total(self, share_lm_head: bool = True) -> int:
        s = (
            self.enorm
            + self.hnorm
            + self.eh_proj
            + self.mtp_block
            + self.shared_head_norm
        )
        if not share_lm_head:
            s += self.shared_head_lm
        return s


class DeepSeekMultiTokenPredictorLayer(nn.Module):
    """Pedagogical mirror of ``DeepSeekMultiTokenPredictorLayer``.

    Layout (verbatim from ``deepseek_mtp.py:L63-L122``):

        enorm   = RMSNorm(hidden_size)              # over next-token embed
        hnorm   = RMSNorm(hidden_size)              # over target hidden state
        eh_proj = Linear(2*hidden, hidden, bias=False)
        mtp_block = DeepseekV2DecoderLayer(...)     # full transformer block
        shared_head = SharedHead(...)

    Forward (``deepseek_mtp.py:L99-L121``):
        emb_n = enorm(inputs_embeds)
        h_t   = hnorm(previous_hidden_states)
        x     = eh_proj([emb_n, h_t])
        h, residual = mtp_block(x)
        h = h + residual
        return h
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        # REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L71-L72
        self.enorm = RMSNorm(hidden_size)
        self.hnorm = RMSNorm(hidden_size)
        # REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L73 bias=False
        self.eh_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        # REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L92-L97 mtp_block
        self.mtp_block = MTPBlock(hidden_size, intermediate_size, num_heads)
        self.hidden_size = hidden_size

    def forward(
        self,
        inputs_embeds: torch.Tensor,        # [T, hidden] — embedding of next-token ids
        previous_hidden_states: torch.Tensor,  # [T, hidden] — target's last hidden
        positions: torch.Tensor,            # [T] — position ids; pos 0 masked per source
    ) -> torch.Tensor:
        # REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L107-L121
        # Position 0 has no "next token" yet — the source masks the embedding to 0.
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0.0, inputs_embeds)
        emb_n = self.enorm(inputs_embeds)
        h_t = self.hnorm(previous_hidden_states)
        # eh_proj fuses the two streams.
        # REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L113-L115
        fused = self.eh_proj(torch.cat([emb_n, h_t], dim=-1))
        h, residual = self.mtp_block(fused)
        # REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L120
        return h + residual

    def parameter_stats(self, vocab_size: int) -> MTPLayerStats:
        """Param counts for narrative §10.5 demo.

        Computed analytically — should match `sum(p.numel() for p in module.parameters())`
        modulo bias terms (we use bias=False everywhere per source).
        """
        h = self.hidden_size
        eh = sum(p.numel() for p in self.eh_proj.parameters())
        en = sum(p.numel() for p in self.enorm.parameters())
        hn = sum(p.numel() for p in self.hnorm.parameters())
        mb = sum(p.numel() for p in self.mtp_block.parameters())
        return MTPLayerStats(
            enorm=en,
            hnorm=hn,
            eh_proj=eh,
            mtp_block=mb,
            shared_head_norm=h,  # one SharedHead RMSNorm
            shared_head_lm=vocab_size * h,  # excluded when shared
        )


class DeepSeekMultiTokenPredictor(nn.Module):
    """Stack of ``num_mtp_layers`` MTP heads + shared embedding + LM head.

    Mirrors ``DeepSeekMultiTokenPredictor`` at
    ``deepseek_mtp.py:L124-L184``. Each layer produces ONE next-token
    prediction; calling forward `K` times in sequence produces K drafts.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        vocab_size: int,
        num_mtp_layers: int,
    ) -> None:
        super().__init__()
        self.num_mtp_layers = num_mtp_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L132-L142
        self.layers = nn.ModuleDict(
            {
                str(idx): DeepSeekMultiTokenPredictorLayer(
                    hidden_size, intermediate_size, num_heads
                )
                for idx in range(num_mtp_layers)
            }
        )
        # REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L143-L147
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        # SharedHead is per-layer in source, but for our pedagogical impl we
        # expose ONE SharedHead — saves params if the operator wants per-step
        # heads to share. The source uses one per layer (within the layer
        # struct); both layouts produce the same token output.
        self.shared_head = SharedHead(hidden_size, vocab_size)

    def forward_one_step(
        self,
        input_ids: torch.Tensor,              # [T] — next-token ids
        positions: torch.Tensor,              # [T]
        previous_hidden_states: torch.Tensor,  # [T, hidden]
        spec_step_idx: int,
    ) -> torch.Tensor:
        # REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L160-L170
        layer_idx = spec_step_idx % self.num_mtp_layers
        embeds = self.embed_tokens(input_ids)
        return self.layers[str(layer_idx)](
            inputs_embeds=embeds,
            previous_hidden_states=previous_hidden_states,
            positions=positions,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L172-L182
        normed = self.shared_head(hidden_states)
        return self.shared_head.compute_logits(normed)

    def propose_K(
        self,
        target_last_hidden: torch.Tensor,  # [T, hidden]
        target_next_token_ids: torch.Tensor,  # [T] - sampled next from target
        positions: torch.Tensor,           # [T]
        K: int,
    ) -> torch.Tensor:
        """Generate K draft tokens by running MTP heads sequentially.

        At each step:
          1. embed previous next-token ids
          2. run MTP layer (uses target hidden state as h_t)
          3. compute logits, argmax → next draft id
          4. roll: this step's draft becomes next step's input
        """
        T = target_last_hidden.shape[0]
        device = target_last_hidden.device
        drafts = torch.zeros((T, K), dtype=torch.int64, device=device)
        cur_input = target_next_token_ids
        cur_hidden = target_last_hidden
        for step in range(K):
            cur_hidden = self.forward_one_step(
                cur_input, positions, cur_hidden, spec_step_idx=step
            )
            logits = self.compute_logits(cur_hidden)
            cur_input = logits.argmax(dim=-1)
            drafts[:, step] = cur_input
        return drafts


def parameter_count_mtp(
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
    num_heads: int,
    num_mtp_layers: int,
) -> dict:
    """Closed-form parameter count for the MTP head stack.

    Used by Demo §5 to compare against Medusa. Returns a dict with the
    breakdown the writer can quote verbatim.
    """
    h = hidden_size
    inter = intermediate_size
    # Per-layer:
    enorm = h
    hnorm = h
    eh_proj = 2 * h * h  # bias=False
    # MTPBlock stand-in:
    qkv = 3 * h * h
    o_proj = h * h
    input_norm = h
    post_norm = h
    gate_up = 2 * h * inter
    down = h * inter
    mtp_block = qkv + o_proj + input_norm + post_norm + gate_up + down
    per_layer = enorm + hnorm + eh_proj + mtp_block
    shared_head_norm = h  # one RMSNorm in shared_head
    shared_head_lm = vocab_size * h  # NOT counted when tied to target
    total_layers = per_layer * num_mtp_layers
    embed = vocab_size * h  # MTP module owns its own embedding (NOT tied in source)
    return {
        "per_layer": per_layer,
        "per_layer_breakdown": {
            "enorm": enorm,
            "hnorm": hnorm,
            "eh_proj": eh_proj,
            "mtp_block_attn": qkv + o_proj,
            "mtp_block_ffn": gate_up + down,
            "mtp_block_norms": input_norm + post_norm,
        },
        "num_mtp_layers": num_mtp_layers,
        "total_layers": total_layers,
        "embed_tokens": embed,
        "shared_head_norm": shared_head_norm,
        "shared_head_lm_if_separate": shared_head_lm,
        "total_with_shared_lm": total_layers + embed + shared_head_norm,
        "total_with_separate_lm": (
            total_layers + embed + shared_head_norm + shared_head_lm
        ),
    }


def parameter_count_medusa(
    hidden_size: int,
    vocab_size: int,
    K: int,
) -> dict:
    """K independent Medusa MLP heads + lm_head per head.

    Mirror of ``MedusaProposer`` at ``vllm/v1/spec_decode/medusa.py:L18-L78``.
    Each head is approximately a single MLP block + linear projection to vocab.
    """
    # REFERENCE: vllm/v1/spec_decode/medusa.py — K independent heads on target hidden.
    h = hidden_size
    # Typical Medusa head: 2 hidden→hidden Linear (residual MLP) + LM proj.
    per_head_mlp = 2 * h * h
    per_head_lm = vocab_size * h
    per_head = per_head_mlp + per_head_lm
    return {
        "per_head_mlp": per_head_mlp,
        "per_head_lm": per_head_lm,
        "per_head": per_head,
        "K": K,
        "total_with_separate_lm": per_head * K,
        "total_with_shared_lm": per_head_mlp * K,
    }
