"""
PAPER: arXiv:2401.15077 EAGLE §2 Preliminaries (accept/reject rule + residual
distribution, restated from Leviathan et al. 2023's speculative sampling and
reused as-is by EAGLE) and Appendix A.2 Algorithm 1 "Multi-round speculative
sampling" (EAGLE's own addition, needed because EAGLE verifies a *tree* of
candidates instead of a single chain, consistent with SpecInfer -- §3.3).

§2's accept/reject rule and its distribution-preservation proof are already
given in full, with worked numeric examples and Monte Carlo verification, in
the speculative-decode chapter (see ../../ch28-spec-decode/narrative/chapter.md,
§28.5) -- `accept_reject`/`residual_distribution` below are reproduced only
as the two building blocks that Appendix A.2's *recursive* Algorithm 1 calls
at each tree node. This module does not re-derive or re-prove distribution
preservation (that link is to ch28); it only implements the multi-round
recursion that verifying a **tree** of candidates (rather than a chain)
newly requires.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


# PAPER: §2 (acceptance prob of t_hat_{j+i}: min(1, p_{j+i}(t)/p_hat_{j+i}(t)))
def accept_reject(p: np.ndarray, p_hat: np.ndarray, token: int, u: float) -> bool:
    """
    u ~ U(0,1) is drawn by the caller (kept explicit, not sampled inside,
    so tests can pin the exact accept/reject boundary).
    """
    ratio = 1.0 if p_hat[token] <= 0.0 else p[token] / p_hat[token]
    return bool(u < min(1.0, ratio))


# PAPER: §2 (resample from norm(max(0, p_{j+i} - p_hat_{j+i})))
def residual_distribution(p: np.ndarray, p_hat: np.ndarray) -> np.ndarray:
    residual = np.maximum(p - p_hat, 0.0)
    total = residual.sum()
    if total <= 0.0:
        # p == p_hat everywhere: rejection cannot occur under accept_reject,
        # but guard division-by-zero if this is ever called out of context.
        return np.ones_like(p) / len(p)
    return residual / total


# PAPER: Appendix A.2 Algorithm 1 "Multi-round speculative sampling"
def multi_round_speculative_sampling(
    p_target: np.ndarray,
    candidate_tokens: Sequence[int],
    candidate_dists: Sequence[np.ndarray],
    us: Sequence[float],
    rng: np.random.Generator,
) -> tuple[int, int]:
    """
    PAPER: Appendix A.2 Algorithm 1. `candidate_tokens[i]` (sampled from
    `candidate_dists[i]`) are the k sibling candidates at one tree node
    (already proposed by the draft model at that node, §3.1); `p_target` is
    the target LLM's distribution at that node (obtained via one
    tree-attention verification forward pass, §3.3). `us[i]` are the
    U(0,1) draws for each round, explicit for testability.

    Recursively: try candidate i against the *current* p; if accepted,
    return it immediately. Otherwise adjust p <- norm(max(0, p - p_hat_i))
    and try candidate i+1 against the adjusted p. If all k siblings are
    rejected, sample fresh from the final adjusted p (this is the
    "otherwise... sample t ~ p; return t" tail of Algorithm 1).

    Returns (token, rounds_tried): rounds_tried == i (1-indexed) if
    candidate i was accepted, or len(candidate_tokens) if a fresh sample was
    needed from the final residual distribution.
    """
    assert len(candidate_tokens) == len(candidate_dists) == len(us)
    p = p_target
    for i, (t_i, p_hat_i, u_i) in enumerate(zip(candidate_tokens, candidate_dists, us), start=1):
        if accept_reject(p, p_hat_i, t_i, u_i):
            return t_i, i
        p = residual_distribution(p, p_hat_i)
    token = int(rng.choice(len(p), p=p))
    return token, len(candidate_tokens)
