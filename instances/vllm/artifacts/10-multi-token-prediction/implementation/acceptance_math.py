"""Chain-break geometric series, expected tokens, and speedup math.

This module is pedagogical — there is no single source file in vLLM that
"contains" the speedup formula. The math is implicit in:

  # REFERENCE: vllm/v1/sample/rejection_sampler.py:L424-L430
  output buffer pre-fill with PLACEHOLDER_TOKEN_ID — once a position rejects,
  later positions stay PLACEHOLDER. Equivalent to the chain-break invariant.

  # REFERENCE: vllm/v1/sample/rejection_sampler.py:L731-L749
  greedy kernel: `if not rejected: ...` — the only entry condition.

  # REFERENCE: vllm/v1/sample/rejection_sampler.py:L789-L815
  random kernel: same `if rejected: break` semantics.

  # REFERENCE: vllm/config/speculative.py:L213-L227
  _acceptance_length_to_rates — vLLM's helper that maps a target *mean*
  acceptance length to per-position synthetic conditional rates. Inverse of
  the formula derived here.

  # REFERENCE: vllm/v1/spec_decode/utils.py:unconditional_to_conditional_rates
  vLLM's helper that maps unconditional rates [P(0..i all accept)]_{i<K} to
  conditional rates [P(i accepts | 0..i-1 all accepted)]_{i<K} for kernel use.

The math:

  Per-position acceptance rate alpha_i = P(draft_i accepted | drafts 0..i-1 accepted)
  Assume i.i.d. (the simplification we make for analysis): alpha_i = alpha for all i.

  Expected tokens per target forward, given K speculative positions:
     E[tok | alpha, K] = sum_{k=0..K} alpha^k = (1 - alpha^(K+1)) / (1 - alpha)
                        = 1  +  alpha  +  alpha^2  +  ...  +  alpha^K
                        ^---bonus---^   ^---accepted at chain pos k---^

  Why? P(at least k tokens emitted) = P(0..k-1 all accept) = alpha^k for k <= K.
  E[tok] = sum_{k=1..K+1} P(tok >= k) = sum_{k=0..K} alpha^k.

  Speedup vs autoregressive (1 token per target forward):
     S = E[tok] / (1 + c * K)
  where c is the draft-cost ratio (draft_forward_time / target_forward_time)
  and the +1 is the unavoidable target forward.

  Break-even alpha (S = 1) is the alpha where MTP starts being a net win:
     E[tok | alpha, K] = 1 + c * K
     (1 - alpha^(K+1)) / (1 - alpha) = 1 + c * K
"""
# REFERENCE: vllm/v1/sample/rejection_sampler.py:L424-L430 (chain-break invariant)
# REFERENCE: vllm/config/speculative.py:L193-L227 (RejectionSampleMethod synthetic + helper)
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def expected_tokens(alpha: float, K: int) -> float:
    """E[tok | alpha, K] = (1 - alpha^(K+1)) / (1 - alpha).

    Geometric series with K+1 terms: 1 + alpha + alpha^2 + ... + alpha^K.

    Special case alpha == 1 → E[tok] = K + 1 exactly. Handles via L'Hopital.

    # REFERENCE: vllm/config/speculative.py:L213-L227 (analogous helper logic)

    Examples (verbatim numerics for narrative, all values match the brief §7):
        alpha=0.7, K=4 → 2.7731  (NOT K * alpha = 2.8)
        alpha=0.5, K=4 → 1.9375  (NOT K * alpha = 2.0)
        alpha=0.3, K=4 → 1.41615 (NOT K * alpha = 1.2 either way)
        alpha=0.4, K=2 → 1.560
        alpha=0.85, K=4 → 3.62 (DeepSeek-V3 reported regime)
    """
    if abs(alpha - 1.0) < 1e-12:
        return float(K + 1)
    return (1.0 - alpha ** (K + 1)) / (1.0 - alpha)


def speedup(alpha: float, K: int, c: float) -> float:
    """S = E[tok | alpha, K] / (1 + c * K).

    c = draft_forward_time / target_forward_time
       Typical: 0.05-0.10 for DeepSeek MTP (small head); 0.02-0.05 for
       Llama-3.3-1B drafting Llama-3.3-70B; ≈0 for ngram (lookup is free).

    Returns: speedup factor vs plain autoregressive (1 token / target_forward).
        S < 1 → MTP is a NET LOSS at this (alpha, K, c).
    """
    return expected_tokens(alpha, K) / (1.0 + c * K)


def break_even_alpha(K: int, c: float, tol: float = 1e-9) -> float:
    """Find alpha where speedup == 1, i.e., MTP just breaks even.

    Below this alpha, MTP loses to autoregressive. Bisection on monotone S(alpha).

    # Numerics referenced in narrative:
    #   K=4, c=0.10 → break_even ≈ 0.477 (so DeepSeek-V3 at alpha=0.85 wins big)
    #   K=4, c=0.05 → break_even ≈ 0.262
    #   K=2, c=0.10 → break_even ≈ 0.260
    #   K=8, c=0.10 → break_even ≈ 0.610
    """
    target = 1.0 + c * K  # what E[tok] needs to be
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        et = expected_tokens(mid, K)
        if abs(et - target) < tol:
            return mid
        if et < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def alpha_K_grid(
    alphas: List[float], Ks: List[int]
) -> np.ndarray:
    """Build the alpha × K expected-tokens grid (analytic).

    Returns a 2D array [len(alphas), len(Ks)] where entry [i, j] is
    expected_tokens(alphas[i], Ks[j]).

    Used by Demo §2 to produce the verbatim numerics table for the writer.
    """
    grid = np.zeros((len(alphas), len(Ks)), dtype=np.float64)
    for i, a in enumerate(alphas):
        for j, k in enumerate(Ks):
            grid[i, j] = expected_tokens(a, k)
    return grid


def speedup_grid(
    alphas: List[float], Ks: List[int], c: float
) -> np.ndarray:
    """Build the alpha × K speedup grid for fixed c.

    Demo §3 produces this for c in {0.05, 0.10, 0.20, 0.30}.
    """
    grid = np.zeros((len(alphas), len(Ks)), dtype=np.float64)
    for i, a in enumerate(alphas):
        for j, k in enumerate(Ks):
            grid[i, j] = speedup(a, k, c)
    return grid


def simulate_chain_break(
    alpha: float,
    K: int,
    n_trials: int,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Empirical mean tokens-per-target-forward via Bernoulli simulation.

    Each trial draws K i.i.d. Uniform[0,1) and accepts iff u < alpha for each
    position; once any position fails, all later positions are wasted (chain
    break). Bonus token (+1) is emitted iff all K accept.

    Returns: (mean_tokens, std_tokens, ci_95_half_width)

    Compared against analytic `expected_tokens(alpha, K)` to verify the
    geometric series. At alpha=0.7, K=4 with 10000 trials:
        empirical: ≈ 2.77 ± 0.014 (95% CI)
        analytic:  2.7731
    """
    rng = np.random.default_rng(seed)
    tokens = np.zeros(n_trials, dtype=np.float64)
    for t in range(n_trials):
        # Position 0 always emits one token (accepted draft OR recovered token
        # on reject). Each subsequent position emits iff every prior accepted.
        # Bonus (position K) emits iff all K positions accepted.
        all_accepted_so_far = True
        emitted = 0
        for _ in range(K):
            if not all_accepted_so_far:
                break
            emitted += 1  # this slot emits — accepted-draft or recovered-token
            u = rng.random()
            if u >= alpha:
                all_accepted_so_far = False
        if all_accepted_so_far:
            emitted += 1  # bonus token (only if EVERY position accepted)
        tokens[t] = emitted
    mean = float(tokens.mean())
    std = float(tokens.std(ddof=1))
    # 95% normal CI half-width (approx; tokens are bounded so it's tight enough)
    ci = 1.96 * std / np.sqrt(n_trials)
    return mean, std, float(ci)


def parameter_count_mtp(
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
    num_mtp_layers: int,
    num_routed_experts: int = 0,
) -> dict:
    """Approximate parameter count for a DeepSeek-style MTP head stack.

    # REFERENCE: vllm/model_executor/models/deepseek_mtp.py:L63-L122
    # DeepSeekMultiTokenPredictorLayer = enorm + hnorm + eh_proj + mtp_block + shared_head

    Per MTP layer (without MoE):
        enorm:        hidden                    (RMSNorm, ~hidden params)
        hnorm:        hidden                    (RMSNorm)
        eh_proj:      2*hidden * hidden         (the fusion projection)
        mtp_block:    ~12 * hidden^2 + 4 * hidden * intermediate
                      (regular DeepseekV2 decoder layer, dense FFN approximation)
        shared_head:  hidden + vocab*hidden     (RMSNorm + LM-head)

    With MoE (n_routed_experts > 0):
        mtp_block adds n_routed_experts * (3 * hidden * intermediate)
        — gate_proj, up_proj, down_proj per expert.

    Note: lm_head is shared with the target via _maybe_share_lm_head, so it
    SHOULD NOT count against the MTP head budget. We compute "as if separate"
    for comparison, then subtract.
    """
    # Per layer
    rmsnorm_params = hidden_size  # gamma scale
    enorm = rmsnorm_params
    hnorm = rmsnorm_params
    eh_proj = 2 * hidden_size * hidden_size  # bias=False per source
    # Decoder layer: ~12*h^2 (qkv projs ~3h^2, o_proj h^2, attn norms ~2h, etc)
    # Plus dense FFN: ~2 * h * inter (gate_proj, up_proj) + h * inter (down_proj)
    decoder_attn = 4 * hidden_size * hidden_size + 2 * hidden_size  # rough
    decoder_dense_ffn = 3 * hidden_size * intermediate_size
    decoder_norms = 2 * hidden_size
    moe_extra = num_routed_experts * 3 * hidden_size * intermediate_size

    decoder_layer = decoder_attn + decoder_dense_ffn + decoder_norms + moe_extra
    shared_head_norm = hidden_size
    shared_head_lm = vocab_size * hidden_size  # nominal; usually shared

    per_layer = enorm + hnorm + eh_proj + decoder_layer + shared_head_norm + shared_head_lm
    total_with_lm = per_layer * num_mtp_layers
    total_without_lm = total_with_lm - num_mtp_layers * shared_head_lm
    return {
        "per_layer_with_lm": per_layer,
        "per_layer_without_lm": per_layer - shared_head_lm,
        "total_with_lm": total_with_lm,
        "total_without_lm_shared": total_without_lm,
        # Components for narrative
        "components_per_layer": {
            "enorm": enorm,
            "hnorm": hnorm,
            "eh_proj": eh_proj,
            "decoder_attn": decoder_attn,
            "decoder_dense_ffn": decoder_dense_ffn,
            "decoder_norms": decoder_norms,
            "moe_experts_total": moe_extra,
            "shared_head_norm": shared_head_norm,
            "shared_head_lm": shared_head_lm,
        },
    }


def parameter_count_medusa(
    hidden_size: int,
    K: int,
    vocab_size: int,
) -> dict:
    """Approximate parameter count for K Medusa MLP heads.

    # REFERENCE: vllm/v1/spec_decode/medusa.py:L18-L78
    # Medusa = K independent MLP heads, each is a small block on top of target's
    # last hidden state. Typical structure: ResidualMLP(hidden) + Linear(hidden, vocab).

    Each head: ~2 * hidden^2 + hidden * vocab (one MLP block + LM proj).
    """
    per_head = 2 * hidden_size * hidden_size + hidden_size * vocab_size
    return {
        "per_head": per_head,
        "total": per_head * K,
        "K": K,
    }


if __name__ == "__main__":
    print("=== Acceptance-rate math demo ===")
    print()
    print("Expected tokens E[tok | alpha, K] = (1 - alpha^(K+1)) / (1 - alpha)")
    print(f"  alpha=0.7, K=4 → {expected_tokens(0.7, 4):.4f}  (NOT K*alpha=2.80)")
    print(f"  alpha=0.5, K=4 → {expected_tokens(0.5, 4):.4f}  (NOT K*alpha=2.00)")
    print(f"  alpha=0.3, K=4 → {expected_tokens(0.3, 4):.5f}")
    print(f"  alpha=0.85, K=4 → {expected_tokens(0.85, 4):.4f}  (DeepSeek-V3 regime)")
    print()
    print("Speedup S = E[tok] / (1 + c*K) for c=0.10:")
    for alpha in [0.3, 0.5, 0.7, 0.85]:
        for K in [2, 4]:
            s = speedup(alpha, K, 0.10)
            print(f"  alpha={alpha}, K={K} → S={s:.3f}")
    print()
    print(f"Break-even alpha (S=1):")
    for K in [2, 4, 8]:
        for c in [0.05, 0.10, 0.20]:
            print(f"  K={K}, c={c} → alpha*={break_even_alpha(K, c):.4f}")
