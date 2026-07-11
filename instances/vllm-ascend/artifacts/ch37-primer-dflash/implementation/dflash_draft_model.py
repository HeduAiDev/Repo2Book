"""
A small, paper-faithful reference implementation of a full DFlash draft
model: a stack of KV-injected layers (kv_injection.py) that turns one
forward pass over a block (bonus token + masked positions) into logits for
the whole block at once.

PAPER: arXiv:2602.06036 §3.2 ("Diffusion drafters generate all gamma tokens
in parallel within a single forward pass", Eq.3), §4.1 (KV injection at
every layer), §4.2 ("All masked positions within a block are decoded in
parallel in a single forward pass"). This module assembles the pieces in
kv_injection.py into a runnable multi-layer model so the *structural* form
of Eq.3's claim -- exactly one model call regardless of block size -- can be
observed directly, rather than only asserted arithmetically as in
latency_model.py.

Not modeled here (out of scope for the paper's own equations): tensor
parallelism, quantization, KV-cache paging/eviction, and any Ascend-specific
kernel fusion beyond the one already reproduced in kv_injection.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kv_injection import (
    build_fused_kv_weight,
    dflash_layer_attention,
    fuse_target_context_features,
    precompute_layer_kv_fused,
    rms_norm,
)


# PAPER: §4.1/Appendix A.3 (one draft layer: KV-injected attention + FFN)
class TinyDflashDraftLayer(nn.Module):
    """
    One draft transformer layer. Attention follows Appendix A.3's Q_i=W_i^Q
    H_d / K_i,V_i=[H_t;H_d] split (via kv_injection.dflash_layer_attention);
    the surrounding pre-norm residual structure and MLP are standard
    Transformer-block plumbing that the paper leaves unspecified for the
    draft model's non-attention parts.
    """

    # PAPER: §4.1/Appendix A.3 (per-layer Q/K/V/O + norm weights, plus standard MLP params)
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.eps = eps

        self.input_layernorm_weight = nn.Parameter(torch.ones(hidden_size))
        self.post_attn_layernorm_weight = nn.Parameter(torch.ones(hidden_size))

        self.w_q = nn.Parameter(torch.randn(num_heads * head_dim, hidden_size) * 0.05)
        self.w_k = nn.Parameter(torch.randn(num_kv_heads * head_dim, hidden_size) * 0.05)
        self.w_v = nn.Parameter(torch.randn(num_kv_heads * head_dim, hidden_size) * 0.05)
        self.w_o = nn.Parameter(torch.randn(hidden_size, num_heads * head_dim) * 0.05)
        self.q_norm_weight = nn.Parameter(torch.ones(head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(head_dim))

        self.mlp_up = nn.Parameter(torch.randn(4 * hidden_size, hidden_size) * 0.05)
        self.mlp_down = nn.Parameter(torch.randn(hidden_size, 4 * hidden_size) * 0.05)

    # PAPER: §4.1 (this layer's own K/V weights, for build_fused_kv_weight)
    def kv_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.w_k, self.w_v

    # PAPER: Appendix A.3 (Q_i=W_i^Q H_d; K_i,V_i=[H_t;H_d]) + standard pre-norm residual MLP
    def forward(
        self, h_d: torch.Tensor, context_k: torch.Tensor, context_v: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        residual = h_d
        normed = rms_norm(h_d, self.input_layernorm_weight, self.eps)
        attn_out = dflash_layer_attention(
            normed, context_k, context_v, positions,
            self.w_q, self.w_k, self.w_v, self.w_o,
            self.q_norm_weight, self.k_norm_weight,
            self.num_heads, self.num_kv_heads, self.head_dim, self.eps,
        )
        h_d = residual + attn_out

        residual = h_d
        normed = rms_norm(h_d, self.post_attn_layernorm_weight, self.eps)
        mlp_out = F.linear(F.silu(F.linear(normed, self.mlp_up)), self.mlp_down)
        h_d = residual + mlp_out
        return h_d


# PAPER: §3.2 Eq.(3), §4.1, §4.2 (full draft model: KV injection once + one block forward)
class TinyDflashDraftModel(nn.Module):
    """
    Ties together fuse_target_context_features -> precompute_layer_kv_fused
    -> a stack of TinyDflashDraftLayer -> lm_head, so that a single call to
    `forward` takes the target's selected-layer hidden states plus one block
    of (bonus + masked) input ids and returns logits for every position in
    that block at once -- the runnable form of "T_draft = t_parallel,
    independent of block size" (Eq.3).
    """

    # PAPER: §4.1/§4.2/Appendix A.3 (layer stack + shared W_c + per-layer k_norm weights)
    def __init__(
        self,
        hidden_size: int,
        target_hidden_size: int,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_selected_target_layers: int = 5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.eps = eps

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [TinyDflashDraftLayer(hidden_size, num_heads, num_kv_heads, head_dim, eps) for _ in range(num_layers)]
        )
        self.final_norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.lm_head = nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.05)

        # PAPER: Appendix A.3 shared projection W_c in R^{D x 5D}
        self.w_c = nn.Parameter(torch.randn(hidden_size, num_selected_target_layers * target_hidden_size) * 0.05)
        self.context_norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.k_norm_weights = nn.ParameterList(
            [nn.Parameter(torch.ones(head_dim)) for _ in range(num_layers)]
        )

    # PAPER: §4.1 (precompute_and_store_context_kv equivalent: fuse once, project once for all layers)
    def _precompute_context_kv(
        self, selected_target_layers: list[torch.Tensor], context_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_t = fuse_target_context_features(selected_target_layers, self.w_c, self.context_norm_weight, self.eps)
        k_weights = [layer.w_k for layer in self.layers]
        v_weights = [layer.w_v for layer in self.layers]
        fused_kv_weight = build_fused_kv_weight(k_weights, v_weights)
        all_k, all_v = precompute_layer_kv_fused(
            h_t, fused_kv_weight, list(self.k_norm_weights), context_positions,
            num_layers=self.num_layers, num_kv_heads=self.num_kv_heads, head_dim=self.head_dim,
            rms_eps=self.eps,
        )
        return all_k, all_v  # each [num_layers, num_ctx, num_kv_heads, head_dim]

    # PAPER: §3.2 Eq.(3) + §4.2 (one forward pass -> logits for the whole block)
    def forward(
        self,
        selected_target_layers: list[torch.Tensor],
        context_positions: torch.Tensor,
        block_input_ids: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        selected_target_layers: 5 tensors [num_ctx, target_hidden] from the
            target model's prefill pass.
        context_positions: [num_ctx] positions of those context tokens.
        block_input_ids: [block_size] the query tokens for this round --
            index 0 is the bonus token from the previous verification step,
            the rest are masked (parallel-drafting) placeholder positions.
        query_positions: [block_size] positions for the query tokens.

        Single call: KV injection is precomputed once (context K/V shared
        by every layer), then every layer runs once over the whole block --
        no per-token loop anywhere in this function.
        """
        all_context_k, all_context_v = self._precompute_context_kv(selected_target_layers, context_positions)

        h_d = self.embed_tokens(block_input_ids)
        for i, layer in enumerate(self.layers):
            h_d = layer(h_d, all_context_k[i], all_context_v[i], query_positions)
        h_d = rms_norm(h_d, self.final_norm_weight, self.eps)
        return F.linear(h_d, self.lm_head)


# PAPER: §3.2 Eq.(3) (structural restatement: one forward call regardless of block size)
def count_forward_calls_diffusion(block_size: int) -> int:
    """A block-diffusion drafter needs exactly one forward pass to produce
    the entire block, no matter how large block_size is."""
    del block_size  # intentionally unused: that is precisely Eq.(3)'s point
    return 1


# PAPER: §3.2 Eq.(2) (structural restatement: one forward call per drafted token)
def count_forward_calls_autoregressive(block_size: int) -> int:
    """An autoregressive drafter (e.g. EAGLE-3) needs one forward pass per
    token in the block, i.e. block_size (== gamma) calls."""
    return block_size
