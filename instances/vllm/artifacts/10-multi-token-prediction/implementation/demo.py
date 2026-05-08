"""Ch10 Multi-Token Prediction — verbatim numerics for the writer.

Five demos, deterministic at seed=42:

§1 Rejection sampling unbiasedness — KL(empirical || p) < 0.01
§2 Geometric chain-break — analytic + empirical α-K grid
§3 Speedup curve under draft cost ratio — break-even α
§4 Greedy fast-path vs random-path numerics
§5 MTP head architecture parameter count vs Medusa

Run:  python implementation/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from implementation.acceptance_math import (  # noqa: E402
    alpha_K_grid,
    break_even_alpha,
    expected_tokens,
    simulate_chain_break,
    speedup,
    speedup_grid,
)
from implementation.mtp_head import (  # noqa: E402
    parameter_count_medusa,
    parameter_count_mtp,
)
from implementation.rejection_sampling import (  # noqa: E402
    rejection_sample,
)
from implementation.spec_metadata import SpecDecodeMetadata  # noqa: E402
from implementation.weight_loading import loader_demo_shapes  # noqa: E402

torch.manual_seed(42)
np.random.seed(42)


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# §1 Rejection sampling unbiasedness — empirical KL to target distribution.
# ---------------------------------------------------------------------------


def demo_unbiasedness() -> None:
    section("§1 Rejection sampling unbiasedness verification (vocab=8, K=4)")

    # Toy distributions chosen so p != q (so the algorithm has work to do).
    p = torch.tensor([0.30, 0.20, 0.15, 0.10, 0.10, 0.07, 0.05, 0.03], dtype=torch.float32)
    q = torch.tensor([0.10, 0.20, 0.20, 0.20, 0.10, 0.10, 0.05, 0.05], dtype=torch.float32)
    # Normalize defensively
    p = p / p.sum()
    q = q / q.sum()

    num_trials = 10000
    counts = torch.zeros(8, dtype=torch.float64)
    gen = torch.Generator().manual_seed(42)
    for _ in range(num_trials):
        d = int(torch.multinomial(q, 1, generator=gen).item())
        u = float(torch.rand(1, dtype=torch.float64, generator=gen).item())
        if q[d] > 0 and (p[d] / q[d]).item() >= u:
            tok = d
        else:
            diff = (p - q).clamp_min(0.0)
            s = float(diff.sum().item())
            residual = diff / s if s > 0 else p
            tok = int(torch.multinomial(residual, 1, generator=gen).item())
        counts[tok] += 1
    empirical = counts / counts.sum()
    safe_emp = empirical.clamp_min(1e-12)
    safe_p = p.clamp_min(1e-12)
    kl = float((safe_emp * (safe_emp.log() - safe_p.log())).sum().item())

    print()
    print(f"  Target distribution p = {[round(x, 4) for x in p.tolist()]}")
    print(f"  Draft distribution  q = {[round(x, 4) for x in q.tolist()]}")
    print(f"  Trials             = {num_trials}")
    print(f"  Empirical p_hat     = {[round(x, 4) for x in empirical.tolist()]}")
    print(f"  KL(empirical || p) = {kl:.6f}  (theorem: should -> 0 as N -> inf)")
    print(f"  Pass threshold     = 0.01")


# ---------------------------------------------------------------------------
# §2 Geometric chain-break — α × K grid (analytic + empirical).
# ---------------------------------------------------------------------------


def demo_chain_break() -> None:
    section("§2 Geometric chain-break: E[tok | alpha, K] = (1 - alpha^(K+1)) / (1 - alpha)")
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    Ks = [1, 2, 3, 4, 5]
    grid = alpha_K_grid(alphas, Ks)

    print()
    header = "alpha\\K     " + "".join(f"{k:>10}" for k in Ks)
    print(header)
    for i, a in enumerate(alphas):
        row = f"  alpha={a:.1f}  " + "".join(f"{grid[i, j]:>10.4f}" for j in range(len(Ks)))
        print(row)

    # Empirical sanity check on a few cells (writer pins one verbatim).
    print()
    print("  Empirical 10000-trial sanity (analytic vs mean ± 95% CI):")
    for a in (0.5, 0.7):
        for K in (2, 4):
            mean, std, ci = simulate_chain_break(a, K, n_trials=10000, seed=42)
            ana = expected_tokens(a, K)
            print(
                f"    alpha={a}, K={K} → empirical {mean:.4f} ± {ci:.4f}  "
                f"vs analytic {ana:.4f}"
            )


# ---------------------------------------------------------------------------
# §3 Speedup curve under draft cost ratio.
# ---------------------------------------------------------------------------


def demo_speedup() -> None:
    section("§3 Speedup S = E[tok] / (1 + c·K) — break-even and net-loss zones")
    K = 4
    cs = [0.05, 0.10, 0.20, 0.30]
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print()
    print(f"  K = {K}")
    print(f"  c\\alpha  " + "".join(f"{a:>8.2f}" for a in alphas))
    for c in cs:
        row = f"  c={c:<5} " + "".join(
            f"{speedup(a, K, c):>8.3f}" for a in alphas
        )
        print(row)

    print()
    print("  Break-even alpha (S = 1):")
    for K_be in (2, 4, 8):
        for c in (0.05, 0.10, 0.20):
            be = break_even_alpha(K_be, c)
            print(f"    K={K_be}, c={c}  →  alpha* = {be:.4f}")


# ---------------------------------------------------------------------------
# §4 Greedy fast-path vs random-path numerics.
# ---------------------------------------------------------------------------


def demo_greedy_vs_random() -> None:
    section("§4 Greedy fast-path vs random-path emit counts (synthetic acceptance)")
    # Build a tiny batch and run rejection sampling in both modes for many trials.
    K = 4
    batch = 1
    vocab = 8

    # Synthetic uniform alpha = 0.7 means each per-position acceptance is
    # independent Bernoulli(0.7). Show that:
    #   greedy: argmax-based, deterministic comparison (no recovery)
    #   random: full algorithm (recovery never wastes a forward — emits
    #           AT LEAST 1 per call because reject still emits a recovered token)
    p = torch.tensor([0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
    p = p / p.sum()
    q_close = torch.tensor([0.35, 0.25, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
    q_close = q_close / q_close.sum()

    n_trials = 1000
    greedy_emit = []
    random_emit = []

    target_logits_one = torch.log(p.clamp_min(1e-12)).unsqueeze(0).expand(K, vocab)

    for trial in range(n_trials):
        # Sample K drafts from q_close
        gen = torch.Generator().manual_seed(1000 + trial)
        drafts = torch.multinomial(q_close, K, replacement=True, generator=gen).to(torch.int32)

        meta = SpecDecodeMetadata.make_dummy([drafts.tolist()], device="cpu")
        bonus = torch.tensor([int(torch.multinomial(p, 1, generator=gen).item())], dtype=torch.int32)
        # Greedy: target logits with sharp argmax → expected to match argmax
        target_logits_g = target_logits_one.clone()

        gen_g = torch.Generator().manual_seed(2000 + trial)
        out_g = rejection_sample(
            metadata=meta,
            draft_probs=None,
            target_logits=target_logits_g,
            bonus_token_ids=bonus,
            all_greedy=True,
            all_random=False,
            generator=gen_g,
        )
        # Count emitted (non-PLACEHOLDER) tokens
        greedy_emit.append(int((out_g[0] != -1).sum().item()))

        # Random: same target_logits, but random path uses full algorithm.
        draft_probs = q_close.unsqueeze(0).expand(K, vocab).contiguous()
        gen_r = torch.Generator().manual_seed(3000 + trial)
        out_r = rejection_sample(
            metadata=meta,
            draft_probs=draft_probs,
            target_logits=target_logits_g,
            bonus_token_ids=bonus,
            all_greedy=False,
            all_random=True,
            generator=gen_r,
        )
        random_emit.append(int((out_r[0] != -1).sum().item()))

    g_mean = float(np.mean(greedy_emit))
    r_mean = float(np.mean(random_emit))
    print()
    print(f"  Trials                  = {n_trials}")
    print(f"  K                       = {K}")
    print(f"  Greedy mean emit        = {g_mean:.4f}")
    print(f"  Random mean emit        = {r_mean:.4f}")
    print(
        f"  ratio random/greedy     = "
        f"{(r_mean / g_mean if g_mean else float('inf')):.4f}"
    )
    print(f"  Greedy emit min/max     = {min(greedy_emit)}/{max(greedy_emit)}")
    print(f"  Random emit min/max     = {min(random_emit)}/{max(random_emit)}")


# ---------------------------------------------------------------------------
# §5 MTP head parameter count vs Medusa.
# ---------------------------------------------------------------------------


def demo_param_count() -> None:
    section("§5 MTP head parameter count vs Medusa (Trap E)")

    hidden = 2048
    intermediate = 8192
    vocab = 32000
    num_heads = 16
    K = 2

    mtp = parameter_count_mtp(
        hidden_size=hidden,
        intermediate_size=intermediate,
        vocab_size=vocab,
        num_heads=num_heads,
        num_mtp_layers=K,
    )
    medusa = parameter_count_medusa(hidden, vocab, K)

    print()
    print(f"  hidden={hidden}, intermediate={intermediate}, vocab={vocab}, K={K}")
    print()
    print(f"  MTP per-layer params       = {mtp['per_layer']:>12,}")
    print(f"     enorm                   = {mtp['per_layer_breakdown']['enorm']:>12,}")
    print(f"     hnorm                   = {mtp['per_layer_breakdown']['hnorm']:>12,}")
    print(f"     eh_proj (2h*h)          = {mtp['per_layer_breakdown']['eh_proj']:>12,}")
    print(f"     mtp_block_attn          = {mtp['per_layer_breakdown']['mtp_block_attn']:>12,}")
    print(f"     mtp_block_ffn           = {mtp['per_layer_breakdown']['mtp_block_ffn']:>12,}")
    print(f"     mtp_block_norms         = {mtp['per_layer_breakdown']['mtp_block_norms']:>12,}")
    print(
        f"  MTP total (shared lm_head) = {mtp['total_with_shared_lm']:>12,}"
    )
    print(
        f"  MTP total (separate lm)    = {mtp['total_with_separate_lm']:>12,}"
    )
    print()
    print(f"  Medusa per-head            = {medusa['per_head']:>12,}")
    print(
        f"     mlp                      = {medusa['per_head_mlp']:>12,}"
    )
    print(f"     lm_head                 = {medusa['per_head_lm']:>12,}")
    print(
        f"  Medusa total (separate lm) = {medusa['total_with_separate_lm']:>12,}"
    )
    print(
        f"  Medusa total (shared lm)   = {medusa['total_with_shared_lm']:>12,}"
    )
    print()
    ratio_with = mtp["total_with_shared_lm"] / max(1, medusa["total_with_shared_lm"])
    ratio_without = mtp["total_with_separate_lm"] / max(1, medusa["total_with_separate_lm"])
    print(f"  Ratio MTP / Medusa (shared lm)   = {ratio_with:.2f}x")
    print(f"  Ratio MTP / Medusa (separate lm) = {ratio_without:.2f}x")
    print()
    print("  Loader demo (HF -> vLLM weight name remap):")
    info = loader_demo_shapes(target_layers=61, mtp_layers=1, hidden=32, vocab=128)
    print(f"    input keys        = {info['input_total_keys']}")
    print(f"    target keys       = {info['target_keys']}")
    print(f"    mtp keys          = {info['mtp_keys']}")
    print(f"    sample rename(s)  = {info['sample_renames']}")


def main() -> None:
    print("Ch10 Multi-Token Prediction — demo numerics (seed=42)")
    demo_unbiasedness()
    demo_chain_break()
    demo_speedup()
    demo_greedy_vs_random()
    demo_param_count()
    print()
    print("All five demos complete.")


if __name__ == "__main__":
    main()
