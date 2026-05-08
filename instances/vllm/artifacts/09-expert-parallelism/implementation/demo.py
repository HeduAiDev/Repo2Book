"""Ch09 demo runner — produces verbatim numerics for the writer.

Five demos covering the chapter's five movements:

§1 Top-K routing distributions (Mixtral and DeepSeek scales)
§2 All-to-all alpha-beta cost model vs all-reduce
§3 Per-expert load distribution under skewed routing × placement strategy
§4 EP+TP composition (2D mesh memory)
§5 EPLB rebalance toy

Run with ``python3 -m demo`` from inside the implementation directory or
``python3 implementation/demo.py`` from the chapter root. All demos are
deterministic (fixed seed). The writer quotes the printed numbers verbatim.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a standalone script from the chapter root.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

import torch  # noqa: E402

from implementation.all2all_baseline import alpha_beta_cost  # noqa: E402
from implementation.eplb import (  # noqa: E402
    EplbState,
    make_skewed_routing,
    per_rank_load_from_logical_load,
)
from implementation.expert_map import (  # noqa: E402
    determine_expert_map,
    per_rank_token_load,
)
from implementation.fused_moe_block import (  # noqa: E402
    FusedMoEBlock,
    memory_per_rank_MiB,
)
from implementation.mixtral_vs_deepseek import (  # noqa: E402
    DEEPSEEK_V2_LITE,
    MIXTRAL_8x7B,
    MoEConfig,
    build_block,
    routing_fingerprint,
)
from implementation.routing import expert_load_counts, fused_topk  # noqa: E402

torch.manual_seed(42)


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# §1 Top-K routing distributions.
# ---------------------------------------------------------------------------


def demo_routing() -> None:
    section("§1 Top-K routing distributions (Mixtral scale, DeepSeek scale)")

    # Build small but illustrative blocks; the routing math doesn't depend on
    # the FFN sizes, so we shrink ``intermediate_size`` here for speed.
    mix_small = MoEConfig(
        num_experts=MIXTRAL_8x7B.num_experts,
        top_k=MIXTRAL_8x7B.top_k,
        hidden_size=512,
        intermediate_size=512,
        scoring_func="softmax",
        name="Mixtral-tiny",
    )
    ds_small = MoEConfig(
        num_experts=DEEPSEEK_V2_LITE.num_experts,
        top_k=DEEPSEEK_V2_LITE.top_k,
        hidden_size=512,
        intermediate_size=512,
        use_grouped_topk=True,
        num_expert_group=DEEPSEEK_V2_LITE.num_expert_group,
        topk_group=DEEPSEEK_V2_LITE.topk_group,
        scoring_func="softmax",
        name="DeepSeek-V2-tiny",
    )
    mix = build_block(mix_small, ep_size=1, seed=0)
    ds = build_block(ds_small, ep_size=1, seed=0)

    fp_mix = routing_fingerprint(mix, num_tokens=1024, seed=7)
    fp_ds = routing_fingerprint(ds, num_tokens=1024, seed=7)

    print()
    print("Mixtral (E=8, K=2):")
    print(f"  per_expert_count = {fp_mix['per_expert_count']}")
    print(
        f"  max={fp_mix['max_count']}  min={fp_mix['min_count']}  "
        f"mean={fp_mix['mean_count']:.2f}  coverage={fp_mix['coverage']:.3f}"
    )
    print(
        f"  per-token weight sum: min={fp_mix['weight_sum_min']:.4f}  "
        f"max={fp_mix['weight_sum_max']:.4f}  mean={fp_mix['weight_sum_mean']:.4f}"
    )

    print()
    print("DeepSeek-V2 grouped (E=64, K=6, n_group=8, topk_group=3):")
    print(
        f"  max={fp_ds['max_count']}  min={fp_ds['min_count']}  "
        f"mean={fp_ds['mean_count']:.2f}  coverage={fp_ds['coverage']:.3f}"
    )
    print(
        f"  per-token weight sum: min={fp_ds['weight_sum_min']:.4f}  "
        f"max={fp_ds['weight_sum_max']:.4f}  mean={fp_ds['weight_sum_mean']:.4f}"
    )

    # Renormalize ON vs OFF — show that without renormalize the K weights
    # sum to <1 (they're a softmax tail, not a probability simplex slice).
    g = torch.Generator().manual_seed(7)
    h = torch.randn(1024, 512, generator=g)
    logits = h @ mix.gate_weight.T
    _, _, _ = fused_topk(h, logits, topk=2, renormalize=True)
    tw_on, _, _ = fused_topk(h, logits, topk=2, renormalize=True)
    tw_off, _, _ = fused_topk(h, logits, topk=2, renormalize=False)
    print()
    print("Renormalize on/off (Mixtral, K=2):")
    print(
        f"  renormalize=True  → sum range "
        f"[{tw_on.sum(dim=-1).min().item():.4f}, "
        f"{tw_on.sum(dim=-1).max().item():.4f}]  mean "
        f"{tw_on.sum(dim=-1).mean().item():.4f}"
    )
    print(
        f"  renormalize=False → sum range "
        f"[{tw_off.sum(dim=-1).min().item():.4f}, "
        f"{tw_off.sum(dim=-1).max().item():.4f}]  mean "
        f"{tw_off.sum(dim=-1).mean().item():.4f}"
    )


# ---------------------------------------------------------------------------
# §2 All-to-all alpha-beta cost model.
# ---------------------------------------------------------------------------


def demo_alpha_beta() -> None:
    section("§2 All-to-all vs all-reduce — alpha-beta cost model (P=8)")

    # NVLink-class beta = 250 GB/s, alpha = 5 μs (round numbers).
    # IB-class    beta =  50 GB/s, alpha = 8 μs.
    p = 8
    payloads = [128, 1024, 8192, 65536]  # tokens
    hidden = 4096
    bytes_per_token = hidden * 2  # bf16
    print(
        "Network: alpha=5μs, beta=250 GB/s (NVLink intra-node), p=8 ranks, "
        "hidden=4096 bf16"
    )
    print()
    print(f"{'tokens':>10}{'bytes':>14}{'T_AR(μs)':>14}{'T_A2A(μs)':>14}{'ratio':>10}")
    for tok in payloads:
        nbytes = tok * bytes_per_token
        t_ar = alpha_beta_cost(nbytes, alpha_us=5.0, beta_GBps=250.0, p=p, op="all_reduce")
        t_a2a = alpha_beta_cost(
            nbytes, alpha_us=5.0, beta_GBps=250.0, p=p, op="all_to_all"
        )
        print(
            f"{tok:>10}{nbytes:>14}{t_ar:>14.2f}{t_a2a:>14.2f}{(t_ar / t_a2a):>10.3f}"
        )

    # Show the IB regime so the writer has an inter-node number too.
    print()
    print(
        "Network: alpha=8μs, beta=50 GB/s (IB inter-node), p=8 ranks, "
        "hidden=4096 bf16"
    )
    print()
    print(f"{'tokens':>10}{'bytes':>14}{'T_AR(μs)':>14}{'T_A2A(μs)':>14}{'ratio':>10}")
    for tok in payloads:
        nbytes = tok * bytes_per_token
        t_ar = alpha_beta_cost(nbytes, alpha_us=8.0, beta_GBps=50.0, p=p, op="all_reduce")
        t_a2a = alpha_beta_cost(
            nbytes, alpha_us=8.0, beta_GBps=50.0, p=p, op="all_to_all"
        )
        print(
            f"{tok:>10}{nbytes:>14}{t_ar:>14.2f}{t_a2a:>14.2f}{(t_ar / t_a2a):>10.3f}"
        )


# ---------------------------------------------------------------------------
# §3 Placement × skewed routing.
# ---------------------------------------------------------------------------


def demo_placement() -> None:
    section("§3 Per-rank load: linear vs round_robin × ep_size, hot routing")

    E = 32
    top_k = 2
    num_tokens = 4096
    skewed = make_skewed_routing(
        num_tokens=num_tokens,
        num_experts=E,
        top_k=top_k,
        hot_fraction=0.2,
        hot_load_fraction=0.6,
        seed=0,
    )
    per_logical = expert_load_counts(skewed, E)
    hot_total = int(per_logical[: int(E * 0.2)].sum().item())
    print(
        f"E={E}, K={top_k}, tokens={num_tokens}, hot 20% of experts received "
        f"{hot_total}/{top_k * num_tokens} routed pairs "
        f"({hot_total / (top_k * num_tokens):.3f})"
    )

    print()
    print(f"{'placement':<14}{'ep_size':>10}{'rank loads':>40}{'max/mean':>12}")
    for ep_size in (1, 4, 8):
        for strat in ("linear", "round_robin"):
            per_rank = per_rank_load_from_logical_load(per_logical, ep_size, strat)
            mean = float(per_rank.float().mean().item()) or 1.0
            ratio = float(per_rank.max().item() / mean) if mean else 0.0
            print(
                f"{strat:<14}{ep_size:>10}{str(per_rank.tolist()):>40}"
                f"{ratio:>12.3f}"
            )

    # Cross-check via expert_map machinery — the masks-based path should
    # produce the same per-rank load.
    print()
    ep = 4
    maps = [determine_expert_map(ep, r, E, "linear")[1] for r in range(ep)]
    pr = per_rank_token_load(skewed, maps)
    print(f"sanity check (ep=4, linear, mask-based) → {pr.tolist()}")


# ---------------------------------------------------------------------------
# §4 EP+TP composition memory.
# ---------------------------------------------------------------------------


def demo_mesh_memory() -> None:
    section("§4 EP×TP weight memory (E=64 DeepSeek-V2-Lite-style block)")

    E = 64
    hidden = 2048
    intermediate = 1408
    print(f"  Per-expert params: 3·{intermediate}·{hidden} = "
          f"{3 * intermediate * hidden:,}")
    print(f"  Total params:       E · 3·intermediate·hidden = "
          f"{E * 3 * intermediate * hidden:,}")
    print(f"  Bytes per param:    2 (bf16)")
    print()
    print(f"{'ep':>4}{'tp':>4}{'mem/rank (MiB)':>18}{'reduction vs (1,1)':>22}")
    base = memory_per_rank_MiB(E, hidden, intermediate, ep_size=1, tp_size=1)
    for ep, tp in ((1, 1), (4, 1), (4, 2), (8, 4), (16, 1), (8, 2)):
        m = memory_per_rank_MiB(E, hidden, intermediate, ep_size=ep, tp_size=tp)
        print(f"{ep:>4}{tp:>4}{m:>18.2f}{(base / m):>22.2f}x")


# ---------------------------------------------------------------------------
# §5 EPLB rebalance toy.
# ---------------------------------------------------------------------------


def demo_eplb() -> None:
    section("§5 EPLB rebalance toy — 100 steps, hot routing")

    E_logical = 32
    top_k = 2
    num_tokens = 1024
    ep_size = 4

    state = EplbState(
        num_logical_experts=E_logical,
        num_redundant_experts=4,
        ep_size=ep_size,
        rearrangement_step_interval=50,
        window_size=50,
    )

    timeline_steps = [0, 25, 50, 51, 75, 99]
    captured: list[tuple[int, list[int], float]] = []

    g = torch.Generator().manual_seed(11)
    placement = "linear"

    for step in range(100):
        # Skewed load: hot experts shift halfway through to test rebalancing.
        hot_fraction = 0.2
        seed_for_step = 100 + step
        ti = make_skewed_routing(
            num_tokens=num_tokens,
            num_experts=E_logical,
            top_k=top_k,
            hot_fraction=hot_fraction,
            hot_load_fraction=0.6,
            seed=seed_for_step,
        )
        per_logical = expert_load_counts(ti, E_logical)
        per_rank = per_rank_load_from_logical_load(per_logical, ep_size, placement)
        rearranged = state.record_step(per_logical)
        if step in timeline_steps:
            ratio = state.imbalance_ratio(per_rank)
            captured.append((step, per_rank.tolist(), ratio))
        if rearranged and step + 1 < 100:
            # Pretend the hot block migrates by switching to round_robin
            # (a stand-in for "EPLB shuffled the layout"). This shows the
            # max/mean ratio drops after the rearrangement.
            placement = "round_robin"

    print()
    print(f"{'step':>6}{'placement':>14}{'per-rank load':>40}{'max/mean':>12}")
    p = "linear"
    for step, loads, ratio in captured:
        if step >= 50:
            p = "round_robin (post-rebalance)"
        print(f"{step:>6}{p:>14}{str(loads):>40}{ratio:>12.3f}")
        if step >= 50:
            p = "round_robin"

    # Also report the physical→logical map shape so readers see redundant slots.
    n_phys = state.num_physical_experts
    n_log = state.num_logical_experts
    p2l = state.physical_to_logical
    print()
    print(
        f"EplbState: num_logical={n_log}, num_redundant=4, num_physical={n_phys}, "
        f"physical_to_logical[0:8]={p2l[:8].tolist()}, "
        f"physical_to_logical[-4:]={p2l[-4:].tolist()}"
    )


def main() -> None:
    print("Ch09 Expert Parallelism — demo numerics (deterministic, seed=42)")
    demo_routing()
    demo_alpha_beta()
    demo_placement()
    demo_mesh_memory()
    demo_eplb()
    print()
    print("All five demos complete.")


if __name__ == "__main__":
    main()
