"""
A small, paper-faithful reference implementation of DFlash's training-time
position-weighted loss and random anchor-block sampling.

PAPER: arXiv:2602.06036 §4.2 "Training". DFlash trains its draft model to
denoise a masked block in parallel, but not every position in that block is
equally valuable: because acceptance during inference is a longest-prefix
match, an error at an early masked position invalidates every later position
in the same round, while an error at the last position only costs one token.
Eq.(4) reflects this by exponentially down-weighting the cross-entropy loss
at later block positions. The prose right before Eq.(4) also describes how
training blocks are built: random "anchor" tokens from the response become
the clean first position of a block (mirroring the bonus token at inference
time), with the remaining block_size - 1 positions masked and predicted in
parallel; multiple such blocks are then concatenated and trained jointly
under a sparse attention mask (Figure 4) that keeps blocks from attending to
each other while every position still sees the shared injected context
features.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# PAPER: §4.2 Eq.(4)
def position_weights(block_size: int, gamma: int) -> torch.Tensor:
    """
    w_k = exp(-(k-1)/gamma) for block positions k = 1..block_size.

    w_1 = 1 always; later positions decay exponentially, controlled by the
    speculation budget gamma (the same gamma used throughout the latency
    model in latency_model.py).
    """
    k = torch.arange(1, block_size + 1, dtype=torch.float32)
    return torch.exp(-(k - 1) / gamma)


# PAPER: §4.2 Eq.(4) (weighted cross-entropy over one draft block)
def position_weighted_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, gamma: int
) -> torch.Tensor:
    """
    logits: [block_size, vocab] draft-model predictions for one training
        block's masked positions; targets: [block_size] the corresponding
        ground-truth (target-model) tokens.

    Applies Eq.(4)'s per-position weights to the standard cross-entropy
    before averaging, so an error at an early masked position contributes
    more to the loss than the same-magnitude error at a late one.
    """
    block_size = logits.shape[0]
    weights = position_weights(block_size, gamma).to(device=logits.device, dtype=logits.dtype)
    per_token_loss = F.cross_entropy(logits, targets, reduction="none")
    return (weights * per_token_loss).sum() / weights.sum()


# PAPER: §4.2 ("Random sampling of masked blocks", prose preceding Eq.4)
def sample_anchor_blocks(
    response_len: int, block_size: int, num_blocks: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Randomly sample `num_blocks` anchor positions from a response of length
    `response_len`. Each anchor becomes the first, clean position of a
    training block (matching inference, where the draft model always
    conditions on one clean bonus token produced by the target model); the
    following block_size - 1 positions are masked and predicted in parallel.

    Anchors are restricted to [0, response_len - block_size] so that a full
    block_size-length window always fits inside the response.
    """
    max_anchor = response_len - block_size
    if max_anchor < 0:
        raise ValueError(
            f"response_len={response_len} too short to fit a block of size {block_size}"
        )
    return rng.integers(0, max_anchor + 1, size=num_blocks)


# PAPER: §4.2 (sparse attention mask, Figure 4: block-diagonal + shared context visibility)
def build_training_attention_mask(
    block_boundaries: list[tuple[int, int]], num_context: int
) -> torch.Tensor:
    """
    Builds the boolean attention mask used when several sampled blocks are
    concatenated into one training sequence (Figure 4): "Tokens attend
    bidirectionally within the same block and to the corresponding injected
    target context features, while attention across different blocks is
    disallowed."

    block_boundaries: (start, end) exclusive ranges over the concatenated
        block-token axis (query axis).
    num_context: number of shared injected context-feature columns, always
        visible to every query token, conceptually prepended as columns
        0..num_context-1.

    Returns a bool tensor of shape [total_block_tokens, num_context +
    total_block_tokens] where True marks an attendable (query, key) pair.
    """
    total = sum(e - s for s, e in block_boundaries)
    mask = torch.zeros(total, num_context + total, dtype=torch.bool)
    mask[:, :num_context] = True  # every token sees the shared context features
    offset = 0
    for start, end in block_boundaries:
        length = end - start
        mask[offset : offset + length, num_context + offset : num_context + offset + length] = True
        offset += length
    return mask
