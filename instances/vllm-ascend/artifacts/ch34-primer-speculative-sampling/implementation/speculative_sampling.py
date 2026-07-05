"""
A small, paper-faithful reference implementation of speculative sampling and
speculative decoding.

PAPER: arXiv:2211.17192 "Fast Inference from Transformers via Speculative
Decoding" (Leviathan, Kalman, Matias). Every function below reproduces one
named equation/theorem/algorithm from that paper; nothing here is invented
beyond what is needed to make the equations *runnable* on toy categorical
distributions p (target model M_p) and q (approximation model M_q) over a
small finite vocabulary — the same "standardized" distributions the paper
reasons about after argmax/top-k/temperature adjustment (paper.md §2.2).

This module intentionally does NOT batch across a real transformer's KV
cache or vocabulary (that vectorized, batched version is the *landing*
implementation — vllm_ascend/sample/rejection_sampler.py:L919-L1260). Here
the goal is to let a reader step through Algorithm 1 and its theorems with a
debugger and small numbers, and to statistically verify the
distribution-preserving proof (paper.md §A.1) by direct Monte Carlo.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


# PAPER: §2.3 Speculative Sampling / §A.1 (residual distribution p')
def residual_distribution(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    PAPER: §2.3 (Speculative Sampling) — the adjusted distribution used to
    resample after a rejection:
        p'(x) = norm(max(0, p(x) - q(x)))
    Also proved in §A.1 to have normalizing constant (1 - beta), where
    beta = sum_x min(p(x), q(x)) is the acceptance rate of Theorem 3.5.
    """
    residual = np.maximum(p - q, 0.0)
    total = residual.sum()
    if total <= 0.0:
        # p == q everywhere (beta == 1): rejection cannot occur, but guard
        # against division-by-zero if this is ever called out of context.
        return np.ones_like(p) / len(p)
    return residual / total


# PAPER: §2.3 Speculative Sampling (accept-reject rule, min(1, p/q))
def propose_and_check(
    p: np.ndarray, q: np.ndarray, rng: np.random.Generator
) -> tuple[int, bool]:
    """
    PAPER: §2.3 — the accept/reject decision for a single proposed token
    x ~ q(x): "keeping it if q(x) <= p(x), and in case q(x) > p(x) we reject
    the sample with probability 1 - p(x)/q(x)", i.e. accept with probability
    min(1, p(x)/q(x)).

    Returns (x, accepted) where x is the token *drawn from q* (not yet
    corrected) and accepted says whether it should be kept as-is.
    """
    x = int(rng.choice(len(q), p=q))
    accept_prob = 1.0 if q[x] <= p[x] else p[x] / q[x]
    accepted = bool(rng.uniform(0.0, 1.0) < accept_prob)
    return x, accepted


# PAPER: §2.3 Speculative Sampling / §A.1 (full accept-or-resample procedure)
def speculative_sampling_step(
    p: np.ndarray, q: np.ndarray, rng: np.random.Generator
) -> tuple[int, bool]:
    """
    PAPER: §2.3, single-token speculative sampling (full procedure).
    Sample x ~ q(x) and accept/reject it via `propose_and_check`; on
    rejection, resample from the adjusted distribution p'(x) =
    norm(max(0, p(x)-q(x))) instead. Proved in §A.1 to give x ~ p(x) exactly.

    Returns (final_token, was_accepted). Note: on rejection, final_token is
    the *resampled* token, not the originally-drawn (and rejected) one.
    """
    x, accepted = propose_and_check(p, q, rng)
    if accepted:
        return x, True
    p_prime = residual_distribution(p, q)
    x_new = int(rng.choice(len(p), p=p_prime))
    return x_new, False


# PAPER: §2.3 Algorithm 1 SpeculativeDecodingStep
def speculative_decoding_step(
    get_q: Callable[[Sequence[int]], np.ndarray],
    get_p: Callable[[Sequence[int]], np.ndarray],
    prefix: Sequence[int],
    gamma: int,
    rng: np.random.Generator,
) -> tuple[list[int], int]:
    """
    PAPER: §2.3 Algorithm 1 "SpeculativeDecodingStep".

    get_q(prefix) -> q(x) distribution from M_q given prefix
    get_p(prefix) -> p(x) distribution from M_p given prefix

    Steps (mirroring the pseudocode line by line):
      1. sample gamma guesses x_1..x_gamma from M_q, autoregressively
      2. run M_p in parallel to get p_1(x)..p_{gamma+1}(x)
      3. draw r_1..r_gamma ~ U(0,1); n = min({i-1 : r_i > p_i(x_i)/q_i(x_i)} u {gamma})
      4. adjust: p'(x) = p_{n+1}(x) if n == gamma, else norm(max(0, p_{n+1}-q_{n+1}))
      5. sample bonus/recovered token t ~ p'(x)
      6. return prefix + [x_1..x_n, t]

    Returns (prefix + [x_1..x_n, t], n) — n is the number of accepted draft
    tokens (0 <= n <= gamma).
    """
    # Step 1: gamma guesses from M_q, autoregressively.
    draft_tokens: list[int] = []
    draft_dists: list[np.ndarray] = []
    cur_prefix = list(prefix)
    for _ in range(gamma):
        q_i = get_q(cur_prefix)
        x_i = int(rng.choice(len(q_i), p=q_i))
        draft_tokens.append(x_i)
        draft_dists.append(q_i)
        cur_prefix = cur_prefix + [x_i]

    # Step 2: run M_p "in parallel" on prefix, prefix+[x1], ..., prefix+[x1..x_gamma].
    target_dists: list[np.ndarray] = []
    cur_prefix = list(prefix)
    for i in range(gamma + 1):
        target_dists.append(get_p(cur_prefix))
        if i < gamma:
            cur_prefix = cur_prefix + [draft_tokens[i]]

    # Step 3: determine number of accepted guesses n.
    r = rng.uniform(0.0, 1.0, size=gamma)
    n = gamma
    for i in range(gamma):
        p_i = target_dists[i][draft_tokens[i]]
        q_i = draft_dists[i][draft_tokens[i]]
        ratio = p_i / q_i if q_i > 0 else np.inf
        if r[i] > ratio:
            n = i
            break

    # Step 4: adjust the distribution from M_p if needed.
    if n < gamma:
        p_prime = residual_distribution(target_dists[n], draft_dists[n])
    else:
        p_prime = target_dists[n]

    # Step 5/6: sample one extra token and assemble the returned sequence.
    t = int(rng.choice(len(p_prime), p=p_prime))
    return list(prefix) + draft_tokens[:n] + [t], n


# PAPER: §3.2 Theorem 3.5 / Corollary 3.6 (acceptance rate alpha)
def acceptance_rate(p: np.ndarray, q: np.ndarray) -> float:
    """
    PAPER: §3.2 Theorem 3.5 / Corollary 3.6.
    beta = E_{x~q} min(1, p(x)/q(x)) = sum_x min(p(x), q(x)).
    For a single fixed prefix this *is* the acceptance rate beta_{x<t}
    (Definition 3.1); averaging beta over many prefixes gives alpha =
    E(beta) (Corollary 3.6), which callers can compute by averaging this
    function's return value over a set of (p, q) prefixes.
    """
    return float(np.minimum(p, q).sum())


# PAPER: §3.2 Definition 3.2 / Lemma 3.3 (D_LK divergence)
def lukaszyk_karmowski_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    PAPER: §3.2 Definition 3.2 / Lemma 3.3.
    D_LK(p, q) = sum_x |p(x) - M(x)| where M(x) = (p(x)+q(x))/2
               = 1 - sum_x min(p(x), q(x))    (Lemma 3.3)
    """
    return 1.0 - float(np.minimum(p, q).sum())


# PAPER: §3.1 Eq.1 (expected number of generated tokens)
def expected_generated_tokens(alpha: float, gamma: int) -> float:
    """
    PAPER: §3.1 Eq.1.
    Assuming the per-position acceptance probabilities beta are i.i.d. with
    mean alpha, the number of tokens produced by one run of Algorithm 1 is a
    capped geometric variable with success probability (1-alpha) and cap
    gamma+1:
        E[#generated tokens] = (1 - alpha^(gamma+1)) / (1 - alpha)
    """
    if abs(alpha - 1.0) < 1e-12:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


# PAPER: §3.3 Theorem 3.8 / Corollary 3.9 (walltime improvement factor)
def walltime_improvement_factor(alpha: float, gamma: int, c: float) -> float:
    """
    PAPER: §3.3 Theorem 3.8.
    Expected walltime improvement factor of Algorithm 1 over standard
    (fully serial) decoding:
        factor = (1 - alpha^(gamma+1)) / ((1 - alpha) * (gamma*c + 1))
    where c is the cost coefficient (Definition 3.7): time(single M_q run) /
    time(single M_p run).
    """
    return expected_generated_tokens(alpha, gamma) / (gamma * c + 1.0)


# PAPER: §3.5 Choosing gamma (numeric argmax of Theorem 3.8)
def optimal_gamma(alpha: float, c: float, gamma_max: int = 32) -> int:
    """
    PAPER: §3.5 "Choosing gamma".
    Since gamma is an integer, the gamma maximizing Theorem 3.8's walltime
    improvement factor can be found numerically by direct search — exactly
    as the paper describes ("it can be easily found numerically, see
    Figure 3").
    """
    gammas = np.arange(0, gamma_max + 1)
    factors = np.array([walltime_improvement_factor(alpha, int(g), c) for g in gammas])
    return int(gammas[np.argmax(factors)])
