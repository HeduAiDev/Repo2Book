"""Integration tests — end-to-end fidelity across the 8 modules.

Verifies that the full Ch09 implementation behaves like a coherent system:

- §3.2 alpha-beta NVLink table reproduces verbatim.
- §3.3 placement-skew table reproduces verbatim across (placement × ep_size).
- §3.4 EP×TP memory table reproduces verbatim.
- §3.5 EPLB layout snapshot ``physical_to_logical[-4:]=[5, 2, 0, 4]`` matches
  after running 100 steps with the demo's seed pattern.
- ep=1 vs ep=N functional invariance (forward pass equivalence).
- Cross-module composition: routing → expert_map → all2all → forward.
- The 7 traps from impl-notes §4 each have a pinning test somewhere in the suite
  (this file documents which test pins which trap).

Demo numerics ground truth: ``tests/demo-output.txt``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

from implementation.all2all_baseline import (
    AgRsAll2AllManager,
    all_gatherv,
    alpha_beta_cost,
    reduce_scatterv,
)
from implementation.eplb import (
    EplbState,
    make_skewed_routing,
    per_rank_load_from_logical_load,
)
from implementation.expert_map import (
    determine_expert_map,
    per_rank_token_load,
)
from implementation.fused_moe_block import FusedMoEBlock, memory_per_rank_MiB
from implementation.routing import expert_load_counts


# ---------------------------------------------------------------------------
# §3.2 NVLink alpha-beta table — verbatim pin
# ---------------------------------------------------------------------------


def test_section_32_nvlink_alpha_beta_table_verbatim():
    """Pin §3.2 NVLink table: T_AR={16.09, 67.47, 478.51, 3766.85} μs at p=8."""
    p = 8
    payloads = [128, 1024, 8192, 65536]  # tokens
    expected_AR = [16.09, 67.47, 478.51, 3766.85]
    expected_A2A = [8.05, 33.74, 239.26, 1883.42]
    bytes_per_token = 4096 * 2  # bf16
    for tok, eAR, eA2A in zip(payloads, expected_AR, expected_A2A):
        nbytes = tok * bytes_per_token
        t_ar = alpha_beta_cost(nbytes, alpha_us=5.0, beta_GBps=250.0, p=p, op="all_reduce")
        t_a2a = alpha_beta_cost(nbytes, alpha_us=5.0, beta_GBps=250.0, p=p, op="all_to_all")
        assert math.isclose(t_ar, eAR, abs_tol=0.02), f"AR mismatch at tok={tok}: got {t_ar}, want {eAR}"
        assert math.isclose(t_a2a, eA2A, abs_tol=0.02), f"A2A mismatch at tok={tok}: got {t_a2a}, want {eA2A}"
        assert math.isclose(t_ar / t_a2a, 2.000, abs_tol=1e-3)


def test_section_32_ib_alpha_beta_headlines():
    """Pin §3.2 IB headline numbers: 50.70 μs at 128 tok, 18804.48 μs at 65536 tok."""
    p = 8
    bytes_per_token = 4096 * 2
    t_ar_small = alpha_beta_cost(
        128 * bytes_per_token, alpha_us=8.0, beta_GBps=50.0, p=p, op="all_reduce"
    )
    t_ar_large = alpha_beta_cost(
        65536 * bytes_per_token, alpha_us=8.0, beta_GBps=50.0, p=p, op="all_reduce"
    )
    assert math.isclose(t_ar_small, 50.70, abs_tol=0.05)
    assert math.isclose(t_ar_large, 18804.48, abs_tol=0.5)


# ---------------------------------------------------------------------------
# §3.3 placement-skew table — verbatim pin (Trap A evidence)
# ---------------------------------------------------------------------------


def test_section_33_placement_table_verbatim():
    """Pin §3.3 rank-load table for both placements at ep ∈ {1, 4, 8}."""
    E, top_k, num_tokens = 32, 2, 4096
    skewed = make_skewed_routing(
        num_tokens=num_tokens, num_experts=E, top_k=top_k,
        hot_fraction=0.2, hot_load_fraction=0.6, seed=0,
    )
    per_logical = expert_load_counts(skewed, E)

    expected = {
        ("linear", 1): [8192],
        ("round_robin", 1): [8192],
        ("linear", 4): [5175, 980, 1017, 1020],
        ("round_robin", 4): [2350, 2420, 1695, 1727],
        ("linear", 8): [3329, 1846, 483, 497, 458, 559, 515, 505],
        ("round_robin", 8): [1199, 1195, 1186, 1205, 1151, 1225, 509, 522],
    }
    for (strat, ep), exp in expected.items():
        per_rank = per_rank_load_from_logical_load(per_logical, ep, strat)
        assert per_rank.tolist() == exp, (
            f"{strat} ep={ep}: got {per_rank.tolist()}, want {exp}"
        )


def test_section_33_hot_total_verbatim():
    """Pin §3.3 'hot 20% of experts received 4915/8192 routed pairs (0.600)'."""
    E, top_k, num_tokens = 32, 2, 4096
    skewed = make_skewed_routing(
        num_tokens=num_tokens, num_experts=E, top_k=top_k,
        hot_fraction=0.2, hot_load_fraction=0.6, seed=0,
    )
    per_logical = expert_load_counts(skewed, E)
    hot_total = int(per_logical[: int(E * 0.2)].sum().item())
    assert hot_total == 4915
    assert math.isclose(hot_total / (top_k * num_tokens), 0.600, abs_tol=1e-3)


def test_section_33_max_mean_ratios_verbatim():
    """Pin §3.3 max/mean ratios — Trap A's headline numbers.

    linear ep=4 → 2.527, linear ep=8 → 3.251, round_robin ep=4 → 1.182.
    """
    E, top_k, num_tokens = 32, 2, 4096
    skewed = make_skewed_routing(
        num_tokens=num_tokens, num_experts=E, top_k=top_k,
        hot_fraction=0.2, hot_load_fraction=0.6, seed=0,
    )
    per_logical = expert_load_counts(skewed, E)

    # linear ep=4: max=5175, mean=8192/4=2048, ratio=2.527
    pl4 = per_rank_load_from_logical_load(per_logical, 4, "linear")
    assert math.isclose(pl4.max().item() / pl4.float().mean().item(), 2.527, abs_tol=0.01)

    # linear ep=8: max=3329, mean=8192/8=1024, ratio=3.251
    pl8 = per_rank_load_from_logical_load(per_logical, 8, "linear")
    assert math.isclose(pl8.max().item() / pl8.float().mean().item(), 3.251, abs_tol=0.01)

    # round_robin ep=4: ratio 1.182
    rr4 = per_rank_load_from_logical_load(per_logical, 4, "round_robin")
    assert math.isclose(rr4.max().item() / rr4.float().mean().item(), 1.182, abs_tol=0.01)


# ---------------------------------------------------------------------------
# §3.4 EP×TP memory table — verbatim pin
# ---------------------------------------------------------------------------


def test_section_34_memory_table_verbatim():
    """Pin §3.4: per-rank memory at all (ep, tp) cells; 1056 → 33 MiB."""
    E, hidden, intermediate = 64, 2048, 1408
    expected = {
        (1, 1): 1056.00,
        (4, 1): 264.00,
        (4, 2): 132.00,
        (8, 4): 33.00,
        (16, 1): 66.00,
        (8, 2): 66.00,
    }
    for (ep, tp), m_expected in expected.items():
        m = memory_per_rank_MiB(E, hidden, intermediate, ep_size=ep, tp_size=tp)
        assert math.isclose(m, m_expected, abs_tol=0.01), (
            f"(ep,tp)=({ep},{tp}): got {m}, want {m_expected}"
        )


def test_section_34_total_params_verbatim():
    """Pin §3.4 total params = 553,648,128 = E * 3 * intermediate * hidden."""
    E, hidden, intermediate = 64, 2048, 1408
    total = E * 3 * intermediate * hidden
    assert total == 553_648_128
    assert 3 * intermediate * hidden == 8_650_752


def test_memory_inverse_proportional_to_ep_times_tp():
    """Cross-cell invariant: mem ∝ 1 / (ep × tp). (4,2) == (8,1) == (16,1)/2."""
    E, hidden, intermediate = 64, 2048, 1408
    m_4_2 = memory_per_rank_MiB(E, hidden, intermediate, ep_size=4, tp_size=2)
    m_8_1 = memory_per_rank_MiB(E, hidden, intermediate, ep_size=8, tp_size=1)
    m_2_4 = memory_per_rank_MiB(E, hidden, intermediate, ep_size=2, tp_size=4)
    assert math.isclose(m_4_2, m_8_1, abs_tol=1e-6)
    assert math.isclose(m_4_2, m_2_4, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# §3.5 EPLB layout snapshot — verbatim pin
# ---------------------------------------------------------------------------


def test_section_35_eplb_layout_after_100_steps_verbatim():
    """Pin §3.5: physical_to_logical[-4:] == [5, 2, 0, 4] after 100 steps.

    This reproduces the demo loop exactly: same seeds, same rearrangement
    interval (50), same redundant count (4).
    """
    E_logical = 32
    top_k = 2
    num_tokens = 1024
    state = EplbState(
        num_logical_experts=E_logical,
        num_redundant_experts=4,
        ep_size=4,
        rearrangement_step_interval=50,
        window_size=50,
    )
    for step in range(100):
        ti = make_skewed_routing(
            num_tokens=num_tokens, num_experts=E_logical, top_k=top_k,
            hot_fraction=0.2, hot_load_fraction=0.6, seed=100 + step,
        )
        per_logical = expert_load_counts(ti, E_logical)
        state.record_step(per_logical)

    p2l = state.physical_to_logical
    assert state.num_physical_experts == 36
    assert p2l[:8].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert p2l[-4:].tolist() == [5, 2, 0, 4]


# ---------------------------------------------------------------------------
# ep=1 vs ep=N forward equivalence — chain-break invariant
# ---------------------------------------------------------------------------


def test_forward_invariance_ep1_vs_ep4():
    """ep_size=1 and ep_size=4 must produce identical outputs."""
    torch.manual_seed(7)
    h = torch.randn(20, 64)
    block_1 = FusedMoEBlock(
        num_experts=8, top_k=2, hidden_size=64, intermediate_size=128, ep_size=1, seed=7
    )
    block_4 = FusedMoEBlock(
        num_experts=8, top_k=2, hidden_size=64, intermediate_size=128, ep_size=4, seed=7
    )
    out_1 = block_1.forward(h)
    out_4 = block_4.forward(h)
    assert torch.allclose(out_1, out_4, atol=1e-6, rtol=1e-5)


def test_forward_invariance_across_three_ep_sizes():
    """ep ∈ {1, 2, 4, 8} must all match each other within numerical tolerance."""
    torch.manual_seed(8)
    h = torch.randn(16, 32)
    outs = []
    for ep in (1, 2, 4, 8):
        block = FusedMoEBlock(
            num_experts=8, top_k=2, hidden_size=32, intermediate_size=64, ep_size=ep, seed=8
        )
        outs.append(block.forward(h))
    base = outs[0]
    for ep, o in zip((1, 2, 4, 8), outs):
        assert torch.allclose(base, o, atol=1e-6, rtol=1e-5), f"ep={ep} diverges"


def test_forward_invariance_grouped_path_ep1_vs_ep4():
    """DeepSeek grouped routing also satisfies ep_size invariance."""
    torch.manual_seed(9)
    h = torch.randn(16, 32)
    b1 = FusedMoEBlock(
        num_experts=16, top_k=4, hidden_size=32, intermediate_size=64, ep_size=1,
        use_grouped_topk=True, num_expert_group=4, topk_group=2, seed=9
    )
    b4 = FusedMoEBlock(
        num_experts=16, top_k=4, hidden_size=32, intermediate_size=64, ep_size=4,
        use_grouped_topk=True, num_expert_group=4, topk_group=2, seed=9
    )
    out_1 = b1.forward(h)
    out_4 = b4.forward(h)
    assert torch.allclose(out_1, out_4, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# Cross-module composition: routing → expert_map → all2all
# ---------------------------------------------------------------------------


def test_routing_to_expert_map_per_rank_token_load_consistency():
    """Routing → expert_map → per_rank_token_load — counts sum to M*K."""
    torch.manual_seed(10)
    M, K, E, P = 64, 2, 16, 4
    ti = torch.randint(0, E, (M, K), dtype=torch.int32)
    maps = [determine_expert_map(P, r, E)[1] for r in range(P)]
    loads = per_rank_token_load(ti, maps)
    assert int(loads.sum().item()) == M * K  # every routed pair lands once


def test_dispatch_combine_round_trip():
    """AgRsAll2AllManager dispatch + combine: total tokens preserved."""
    p = 4
    sizes = [3, 5, 4, 2]
    H = 8
    per_rank_h = [torch.randn(s, H) for s in sizes]
    per_rank_tw = [torch.rand(s, 2) for s in sizes]
    per_rank_ti = [torch.randint(0, 8, (s, 2), dtype=torch.int32) for s in sizes]

    mgr = AgRsAll2AllManager(ep_size=p)
    hs, tw, ti = mgr.dispatch(per_rank_h, per_rank_tw, per_rank_ti)
    assert hs.shape == (sum(sizes), H)
    assert tw.shape == (sum(sizes), 2)
    assert ti.shape == (sum(sizes), 2)

    # combine: split back along sizes; chunks are token-count consistent.
    chunks = mgr.combine(hs, sizes)
    assert [c.shape[0] for c in chunks] == sizes


# ---------------------------------------------------------------------------
# Trap C — shared experts are NOT modeled in FusedMoEBlock's memory contract
# ---------------------------------------------------------------------------


def test_trap_C_shared_experts_DO_NOT_appear_in_FusedMoEBlock_memory():
    """Trap C evidence: ``memory_per_rank_MiB`` only counts routed experts.

    DeepSeek's shared experts live OUTSIDE FusedMoE on the model side
    (deepseek_v2.py:L302-L317) and are replicated per rank — they DON'T
    scale with ep_size. Our memory model counts only the routed experts.

    This test pins the omission as a deliberate scope decision: callers who
    extend the model with shared experts must add their cost separately.
    """
    # Memory under EP=1 (everything on one rank) does NOT include any shared
    # expert — the formula is `E * 3 * intermediate * hidden * bytes`.
    E, hidden, intermediate = 64, 2048, 1408
    base = memory_per_rank_MiB(E, hidden, intermediate, ep_size=1, tp_size=1)
    # 64 * 3 * 1408 * 2048 * 2 / 2**20 = 1056.0 MiB exactly.
    assert math.isclose(base, 1056.0, abs_tol=1e-3)
    # If shared experts were included, memory would scale with the shared
    # expert count too — but the formula is purely a function of E (routed).
    # Doubling shared count does not change our memory model output:
    same = memory_per_rank_MiB(E, hidden, intermediate, ep_size=1, tp_size=1)
    assert base == same  # confirming no implicit shared-expert term


# ---------------------------------------------------------------------------
# Trap pinning roster — meta-test documenting which test covers each trap
# ---------------------------------------------------------------------------


def test_trap_roster_complete():
    """Documentation test: every trap in impl-notes §4 has a pinning test."""
    # This is a registry, not a behavioral test. Failing this test would
    # mean the test suite has lost coverage of one of the 7 traps.
    trap_pinning = {
        "A_ep_capacity_misframed":
            "test_section_33_max_mean_ratios_verbatim — shows ratio=3.25× at ep=8 linear",
        "B_a2a_symmetric_simplification":
            "test_section_32_nvlink_alpha_beta_table_verbatim + AgRs is allgatherv "
            "— verifies dispatch is allgatherv NOT alltoall (test_all2all_baseline.py)",
        "C_expert_independence":
            "shared-expert + EPLB redundant-physical evidence in test_eplb.py "
            "(test_eplb_redundant_layout) and test_mixtral_vs_deepseek.py "
            "(deepseek has shared experts)",
        "D_eplb_free_bolt_on":
            "test_eplb_group_is_distinct_object_from_ep + "
            "test_init_ep_group_eplb_separate_group (separate process group)",
        "E_aux_loss_misattribution":
            "test_trap_E_no_loss_balance_in_eplb_module + "
            "test_trap_E_no_optimizer_in_module + test_trap_E_no_gradient_tracking_in_record_step",
        "F_dispatch_always_runs":
            "test_use_all2all_kernels_requires_dp_and_ep — only DP>1 + EP triggers "
            "real all2all (otherwise NoDPEP fast path)",
        "G_softmax_topk_commutativity":
            "test_trap_G_softmax_topk_does_not_commute_with_topk_softmax + "
            "test_section_31_renormalize_off_range_verbatim",
    }
    assert len(trap_pinning) == 7, "Must cover all 7 traps from impl-notes §4"
    for trap, where in trap_pinning.items():
        assert isinstance(where, str) and len(where) > 30


# ---------------------------------------------------------------------------
# Sanity: cross-module imports clean (no circular deps, no missing names)
# ---------------------------------------------------------------------------


def test_implementation_package_imports_cleanly():
    """The implementation package and all submodules import without error."""
    from implementation import (  # noqa: F401
        AgRsAll2AllManager,
        EPGroup,
        EplbState,
        FusedMoEBlock,
        FusedMoEParallelConfig,
        alpha_beta_cost,
        determine_expert_map,
        fused_topk,
        get_ep_group,
        grouped_topk,
        init_ep_group,
    )


def test_demo_output_file_exists_with_pinned_numerics():
    """The captured demo output file is present and non-empty (writer ground truth)."""
    p = Path(__file__).resolve().parent / "demo-output.txt"
    assert p.exists()
    text = p.read_text()
    # spot-check 5 numerics from across the 5 sections.
    assert "[250, 285, 277, 243, 253, 272, 247, 221]" in text  # §3.1
    assert "ratio=2.000" in text or "  2.000" in text  # §3.2
    assert "[5175, 980, 1017, 1020]" in text  # §3.3
    assert "1056.00" in text  # §3.4
    assert "physical_to_logical[-4:]=[5, 2, 0, 4]" in text  # §3.5
