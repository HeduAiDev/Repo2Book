"""
A small, paper-faithful reference implementation of the DeepSeek-V3
Multi-Token Prediction (MTP) module.

PAPER: arXiv:2412.19437 "DeepSeek-V3 Technical Report" §2.2 Multi-Token
Prediction, Eq.21-23 ("MTP Modules"). This module reproduces the paper's
described structure — D sequential modules, each combining the previous
depth's hidden representation with the next token's embedding through a
shared Emb/OutHead, keeping "the complete causal chain at each prediction
depth" (paper-mtp.md §2.2) — on tiny CPU-only tensors, purely to make Eq.21-
23 steppable with a debugger.

This is NOT the production module: vllm_ascend/models/deepseek_v4_mtp.py
additionally handles tensor-parallel sharding, quantization, the DeepSeek-V2
decoder block's full MoE/attention stack as TRM_k, and Ascend-specific
weight-loading remaps (`hc_head`, `spec_step_idx` routing, etc.) — none of
which are part of the paper's Eq.21-23 and are therefore intentionally
absent here (see the landing chapter for those).
"""
from __future__ import annotations

import torch
import torch.nn as nn


# PAPER: paper-mtp.md §2.2 Eq.21 (RMSNorm applied before concatenation)
class RMSNorm(nn.Module):
    """
    The RMSNorm(.) applied to both the previous-depth hidden state and the
    token embedding before concatenation.
    """

    # PAPER: paper-mtp.md §2.2 Eq.21 (RMSNorm parameters)
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    # PAPER: paper-mtp.md §2.2 Eq.21 (RMSNorm(.))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# PAPER: paper-mtp.md §2.2 Eq.21-23 (a single MTP module at depth k)
class MTPModule(nn.Module):
    """
    A single MTP module at prediction depth k.

        h'^k_i = M_k [ RMSNorm(h^{k-1}_i) ; RMSNorm(Emb(t_{i+k})) ]   (Eq.21)
        h^k_{1:T-k} = TRM_k(h'^k_{1:T-k})                             (Eq.22)
        P^k_{i+k+1} = OutHead(h^k_i)                                  (Eq.23)

    `embed` (Emb) and `out_head` (OutHead) are passed in rather than owned,
    because the paper is explicit that "for each MTP module, its embedding
    layer is shared with the main model" and likewise for the output head.
    TRM_k (the Transformer block at depth k) is the one component the paper
    leaves architecture-agnostic ("a Transformer block"); we use a single
    standard encoder layer as a small, concrete stand-in.
    """

    # PAPER: paper-mtp.md §2.2 Eq.21 (M_k projection + shared Emb/OutHead)
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        embed: nn.Embedding,
        out_head: nn.Linear,
    ) -> None:
        super().__init__()
        self.hnorm = RMSNorm(hidden_size)
        self.enorm = RMSNorm(hidden_size)
        # M_k in Eq.21: R^{d x 2d} projection of the concatenated pair.
        self.proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        # TRM_k in Eq.22: "a Transformer block" — one encoder layer here.
        self.trm = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=1,
            dim_feedforward=hidden_size * 2,
            batch_first=True,
        )
        self.embed = embed
        self.out_head = out_head

    # PAPER: paper-mtp.md §2.2 Eq.21-23 (concat -> TRM_k -> OutHead)
    def forward(
        self, h_prev: torch.Tensor, shifted_token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h_prev:           [batch, valid_len, hidden]  h_i^{k-1} for i=1..valid_len
        shifted_token_ids: [batch, valid_len]          t_{i+k} aligned with h_prev

        Returns (h_k, logits_k), both over the same [batch, valid_len, ...] window.
        """
        emb = self.embed(shifted_token_ids)  # Emb(t_{i+k})
        combined = torch.cat([self.hnorm(h_prev), self.enorm(emb)], dim=-1)  # Eq.21 concat
        h_prime = self.proj(combined)  # M_k [...]
        h_k = self.trm(h_prime)  # Eq.22: TRM_k
        logits_k = self.out_head(h_k)  # Eq.23: OutHead -> P^k
        return h_k, logits_k


# PAPER: paper-mtp.md §2.2 (D sequential MTP modules, causal chain across depths)
class DeepSeekMTPPredictor(nn.Module):
    """
    "Our MTP implementation uses D sequential modules to predict D
    additional tokens... we sequentially predict additional tokens and keep
    the complete causal chain at each prediction depth" (contrasted
    explicitly with Gloeckle et al.'s independent parallel output heads).

    Owns the single shared Emb/OutHead and stacks `depth` MTPModule
    instances, feeding depth k's output h_k as depth (k+1)'s h_prev — this
    is the causal chain. The valid window shrinks by one position per depth
    (Eq.22's index range 1:T-k), matching the paper's slicing.
    """

    # PAPER: paper-mtp.md §2.2 (shared Emb/OutHead construction, D modules)
    def __init__(self, depth: int, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.depth = depth
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.out_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.modules_by_depth = nn.ModuleList(
            [MTPModule(hidden_size, vocab_size, self.embed, self.out_head) for _ in range(depth)]
        )

    # PAPER: paper-mtp.md §2.2 Eq.21-23 (sequential depths, causal chain)
    def forward(
        self, h_main: torch.Tensor, token_ids: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        h_main:    [batch, T, hidden]  the main model's representation, used
                   as h^0 (i.e. h_i^{k-1} for k=1, per paper-mtp.md: "when
                   k=1, h_i^{k-1} refers to the representation given by the
                   main model").
        token_ids: [batch, T]          full input token sequence.

        Runs depths k=1..D sequentially. At depth k, the valid window is
        positions i=1..T-k (Eq.22), and the token consumed is t_{i+k}, i.e.
        token_ids sliced with an offset of k.

        Returns a list of (h_k, logits_k) pairs, one per depth.
        """
        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        h_prev = h_main
        seq_len = token_ids.shape[1]
        for k in range(1, self.depth + 1):
            valid_len = seq_len - k
            h_prev_window = h_prev[:, :valid_len, :]
            shifted_tokens = token_ids[:, k : k + valid_len]  # t_{i+k}, i=1..valid_len
            h_k, logits_k = self.modules_by_depth[k - 1](h_prev_window, shifted_tokens)
            outputs.append((h_k, logits_k))
            h_prev = h_k  # causal chain: depth k+1 consumes depth k's h
        return outputs
