"""
A small, paper-faithful reference implementation of the DFlash speedup /
latency model.

PAPER: arXiv:2602.06036 "DFlash: Block Diffusion for Flash Speculative
Decoding" §3.1-3.2. Speculative decoding's per-token latency (Eq.1) has two
levers: the drafting cost T_draft in the numerator, and the expected accepted
length tau in the denominator. DFlash's two ideas map onto exactly these two
levers -- block-diffusion parallel drafting keeps T_draft flat as the
speculation budget gamma grows (Eq.3, vs. Eq.2's linear growth for
autoregressive drafters), and KV injection (see kv_injection.py) raises tau.

This module only reproduces the arithmetic of Eq.1-3; it takes t_step /
t_parallel / t_verify / tau as given numbers rather than measuring them from a
real model (that measurement is what the paper's Figure 3 and Table 1/9 do).
"""
from __future__ import annotations


# PAPER: §3.1 Eq.(1)
def per_token_latency(t_draft: float, t_verify: float, tau: float) -> float:
    """
    L = (T_draft + T_verify) / tau.

    tau is "the expected number of accepted tokens per cycle, including the
    bonus token produced by the target model" (tau in [1, gamma+1]).
    """
    if tau <= 0:
        raise ValueError("tau (expected accepted tokens per cycle) must be positive")
    return (t_draft + t_verify) / tau


# PAPER: §3.1 (speedup eta = L_target / L, right after Eq.1)
def speedup(l_target: float, l: float) -> float:
    """eta = L_target / L, where L_target is the target model's own
    autoregressive per-token latency."""
    return l_target / l


# PAPER: §3.2 Eq.(2)
def autoregressive_draft_cost(gamma: int, t_step: float) -> float:
    """
    T_draft = gamma * t_step.

    Autoregressive drafters generate tokens sequentially: drafting cost
    grows linearly with the speculation budget gamma, because each of the
    gamma tokens needs its own forward pass of latency t_step.
    """
    return gamma * t_step


# PAPER: §3.2 Eq.(3)
def diffusion_draft_cost(t_parallel: float, gamma: int | None = None) -> float:
    """
    T_draft = t_parallel.

    Diffusion drafters generate all gamma tokens in parallel within a single
    forward pass, so the drafting cost is just the latency of that one block
    forward, t_parallel -- independent of gamma. `gamma` is accepted (and
    ignored) purely so call sites can vary it explicitly and see that the
    returned cost does not change, mirroring the paper's claim that
    "drafting cost no longer scales with the number of generated tokens".
    """
    return t_parallel


# PAPER: §3.1 Eq.(1) + §3.2 Eq.(2)/(3) (combined worked-example helper)
def speedup_for_mode(
    mode: str,
    gamma: int,
    t_step: float,
    t_parallel: float,
    t_verify: float,
    tau: float,
    l_target: float,
) -> float:
    """
    Compute the end-to-end speedup eta for one full cycle under either
    drafting mode, so the two mechanisms (Eq.2's linear-in-gamma drafting
    cost vs. Eq.3's gamma-independent one) can be compared side by side on
    the same tau/t_verify/l_target inputs.
    """
    if mode == "autoregressive":
        t_draft = autoregressive_draft_cost(gamma, t_step)
    elif mode == "diffusion":
        t_draft = diffusion_draft_cost(t_parallel, gamma)
    else:
        raise ValueError(f"unknown drafting mode: {mode!r} (expected 'autoregressive' or 'diffusion')")
    l = per_token_latency(t_draft, t_verify, tau)
    return speedup(l_target, l)
