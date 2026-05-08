"""Fidelity tests for eplb.py — toy EplbState runtime rebalancer.

Verifies:

- Initial layout: physical_to_logical = [0..n_log-1, redundant round-robin].
- record_step: returns False until ``rearrangement_step_interval`` elapses,
  True at the trigger, False again after.
- _rearrange: hot logical experts move into redundant physical slots.
- imbalance_ratio: max/mean = 1.0 for uniform load, > 1 for skewed.
- per_rank_load_from_logical_load: linear vs round_robin sums match
  the placement contract.
- Trap E (NEGATIVE): the EplbState code does NOT compute aux loss / gradient
  / loss-balance loss — verified by symbolic check on the source file.
- §3.5 demo numerics pinned bit-for-bit (timeline + final layout).
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import torch

from implementation import eplb as eplb_module
from implementation.eplb import (
    EplbState,
    make_skewed_routing,
    per_rank_load_from_logical_load,
)
from implementation.routing import expert_load_counts


# ---------------------------------------------------------------------------
# EplbState construction
# ---------------------------------------------------------------------------


def test_eplb_initial_layout_logical_then_redundant():
    """Initial physical_to_logical = [0..n_log-1, then redundant slots round-robin]."""
    state = EplbState(num_logical_experts=8, num_redundant_experts=4)
    p2l = state.physical_to_logical.tolist()
    assert p2l[:8] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert p2l[8:] == [0, 1, 2, 3]  # round-robin into the first 4 logical


def test_eplb_zero_redundant_no_extra_slots():
    """num_redundant_experts=0 → physical map equals logical."""
    state = EplbState(num_logical_experts=4, num_redundant_experts=0)
    assert state.physical_to_logical.tolist() == [0, 1, 2, 3]


def test_eplb_num_physical_property():
    """num_physical = logical + redundant."""
    state = EplbState(num_logical_experts=10, num_redundant_experts=3)
    assert state.num_physical_experts == 13


def test_eplb_layout_dtype_int64():
    """physical_to_logical is int64 (matches eplb_state.py contract)."""
    state = EplbState(num_logical_experts=8, num_redundant_experts=4)
    assert state.physical_to_logical.dtype == torch.int64


# ---------------------------------------------------------------------------
# record_step — interval gate
# ---------------------------------------------------------------------------


def test_record_step_false_before_interval():
    """For first (interval-1) steps record_step returns False."""
    state = EplbState(
        num_logical_experts=8, num_redundant_experts=4,
        rearrangement_step_interval=10
    )
    load = torch.ones(8, dtype=torch.int64)
    for step in range(9):
        assert state.record_step(load) is False


def test_record_step_true_at_interval():
    """At step == interval, rearrangement happens (returns True)."""
    state = EplbState(
        num_logical_experts=8, num_redundant_experts=4,
        rearrangement_step_interval=5
    )
    load = torch.ones(8, dtype=torch.int64)
    triggered = [state.record_step(load) for _ in range(5)]
    assert triggered == [False, False, False, False, True]


def test_record_step_resets_interval_counter():
    """After triggering, the next ``interval`` steps are False, then True again."""
    state = EplbState(
        num_logical_experts=8, num_redundant_experts=4,
        rearrangement_step_interval=3
    )
    load = torch.ones(8, dtype=torch.int64)
    triggers = [state.record_step(load) for _ in range(7)]
    # Steps 1,2 False; 3 True; 4,5 False; 6 True; 7 False.
    assert triggers == [False, False, True, False, False, True, False]


def test_record_step_assert_load_shape():
    """record_step asserts the load tensor has the right shape."""
    state = EplbState(num_logical_experts=8, num_redundant_experts=0)
    bad_load = torch.zeros(7, dtype=torch.int64)
    try:
        state.record_step(bad_load)
    except AssertionError:
        return
    raise AssertionError("expected AssertionError on shape mismatch")


def test_record_step_window_caps_history():
    """_load_history doesn't grow past window_size — old entries fall off."""
    state = EplbState(
        num_logical_experts=4, num_redundant_experts=0,
        rearrangement_step_interval=1000, window_size=3
    )
    for _ in range(10):
        state.record_step(torch.ones(4, dtype=torch.int64))
    assert len(state._load_history) == 3


# ---------------------------------------------------------------------------
# _rearrange — hot experts into redundant slots
# ---------------------------------------------------------------------------


def test_rearrange_hot_experts_get_redundant_slots():
    """After rearrange, the hottest logical experts occupy the redundant physical slots."""
    state = EplbState(
        num_logical_experts=8, num_redundant_experts=2,
        rearrangement_step_interval=1
    )
    # Hot experts: 5 (very hot) and 3 (hot). Cold: rest.
    load = torch.tensor([1, 1, 1, 50, 1, 100, 1, 1], dtype=torch.int64)
    state.record_step(load)
    p2l = state.physical_to_logical.tolist()
    # Logical experts 0..7 occupy positions 0..7 (each gets at least one slot).
    assert p2l[:8] == [0, 1, 2, 3, 4, 5, 6, 7]
    # Redundant slots (2 of them) duplicate the hottest experts: 5 and 3.
    assert p2l[8] == 5  # hottest
    assert p2l[9] == 3  # second-hottest


def test_rearrange_no_op_with_empty_history():
    """If _load_history is empty (no record_step yet) rearrange is no-op."""
    state = EplbState(num_logical_experts=4, num_redundant_experts=2)
    # Force-call _rearrange with no history.
    initial = state.physical_to_logical.clone()
    state._rearrange()
    assert torch.equal(state.physical_to_logical, initial)


# ---------------------------------------------------------------------------
# imbalance_ratio
# ---------------------------------------------------------------------------


def test_imbalance_uniform_load_is_one():
    """Perfectly balanced → max/mean = 1.0."""
    state = EplbState(num_logical_experts=4, num_redundant_experts=0)
    load = torch.tensor([100.0, 100.0, 100.0, 100.0])
    assert math.isclose(state.imbalance_ratio(load), 1.0, abs_tol=1e-6)


def test_imbalance_skewed_load_above_one():
    """Skewed load → max/mean > 1."""
    state = EplbState(num_logical_experts=4, num_redundant_experts=0)
    load = torch.tensor([400.0, 0.0, 0.0, 0.0])
    # max=400, mean=100 → ratio=4.
    assert math.isclose(state.imbalance_ratio(load), 4.0, abs_tol=1e-6)


def test_imbalance_zero_load_returns_one():
    """All-zero load: clamp_min(1) → ratio = max/1 = 0/1 = 0; but we report 1.0 if numel==0."""
    state = EplbState(num_logical_experts=4, num_redundant_experts=0)
    # Documented behavior: when input is zero, clamp_min(1) → mean=1, max=0 → ratio=0
    assert state.imbalance_ratio(torch.zeros(4)) == 0.0


def test_imbalance_empty_load_returns_one():
    """Empty load tensor (numel==0) returns 1.0 sentinel."""
    state = EplbState(num_logical_experts=4, num_redundant_experts=0)
    assert state.imbalance_ratio(torch.tensor([], dtype=torch.float32)) == 1.0


# ---------------------------------------------------------------------------
# per_rank_load_from_logical_load
# ---------------------------------------------------------------------------


def test_per_rank_load_linear_simple():
    """Linear placement, ep=2, E=4: ranks see consecutive blocks."""
    per_log = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    out = per_rank_load_from_logical_load(per_log, ep_size=2, placement_strategy="linear")
    assert out.tolist() == [30, 70]


def test_per_rank_load_round_robin_simple():
    """Round-robin, ep=2, E=4: rank 0 owns 0,2 (10+30); rank 1 owns 1,3 (20+40)."""
    per_log = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    out = per_rank_load_from_logical_load(per_log, ep_size=2, placement_strategy="round_robin")
    assert out.tolist() == [40, 60]


def test_per_rank_load_remainder():
    """E=7, P=3: linear → ranks own blocks of [3, 2, 2]."""
    per_log = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
    out = per_rank_load_from_logical_load(per_log, ep_size=3, placement_strategy="linear")
    # Block boundaries: [0..2]=1+2+3=6; [3..4]=4+5=9; [5..6]=6+7=13.
    assert out.tolist() == [6, 9, 13]


def test_per_rank_load_preserves_total():
    """Sum of per-rank load equals sum of per-logical load."""
    torch.manual_seed(0)
    per_log = torch.randint(0, 100, (16,), dtype=torch.int64)
    for strat in ("linear", "round_robin"):
        for ep in (1, 2, 4, 8):
            out = per_rank_load_from_logical_load(per_log, ep_size=ep, placement_strategy=strat)
            assert int(out.sum().item()) == int(per_log.sum().item())


def test_per_rank_load_invalid_strategy():
    """Unsupported strategy raises ValueError."""
    per_log = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    try:
        per_rank_load_from_logical_load(per_log, ep_size=2, placement_strategy="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# make_skewed_routing
# ---------------------------------------------------------------------------


def test_skewed_routing_shape_and_dtype():
    """make_skewed_routing returns int32 (M, K)."""
    ti = make_skewed_routing(num_tokens=32, num_experts=8, top_k=2, seed=0)
    assert ti.shape == (32, 2)
    assert ti.dtype == torch.int32


def test_skewed_routing_hot_fraction_dominates():
    """With hot_fraction=0.2, hot_load_fraction=0.6 → top 20% experts get ~60% of routing."""
    M, E, K = 4096, 32, 2
    ti = make_skewed_routing(
        num_tokens=M, num_experts=E, top_k=K,
        hot_fraction=0.2, hot_load_fraction=0.6, seed=0
    )
    counts = expert_load_counts(ti, E)
    n_hot = int(E * 0.2)
    hot_total = int(counts[:n_hot].sum().item())
    assert math.isclose(hot_total / (M * K), 0.6, abs_tol=1e-3)


def test_skewed_routing_total_matches_M_K():
    """Total routed pairs = M*K."""
    M, E, K = 1024, 16, 4
    ti = make_skewed_routing(num_tokens=M, num_experts=E, top_k=K, seed=42)
    counts = expert_load_counts(ti, E)
    assert int(counts.sum().item()) == M * K


# ---------------------------------------------------------------------------
# §3.5 demo numerics — verbatim pin
# ---------------------------------------------------------------------------


def test_section_35_initial_p2l_pin():
    """§3.5 EplbState: num_logical=32, num_redundant=4, num_physical=36."""
    state = EplbState(
        num_logical_experts=32, num_redundant_experts=4,
        ep_size=4, rearrangement_step_interval=50, window_size=50
    )
    assert state.num_logical_experts == 32
    assert state.num_redundant_experts == 4
    assert state.num_physical_experts == 36
    p2l = state.physical_to_logical.tolist()
    assert p2l[:8] == [0, 1, 2, 3, 4, 5, 6, 7]


def test_section_35_post_rearrange_p2l_pin():
    """After running 100 skewed steps, physical_to_logical[-4:] = [5, 2, 0, 4] (verbatim §3.5).

    Reproduces the exact demo loop: hot_fraction=0.2, hot_load_fraction=0.6,
    seed_for_step=100+step.
    """
    state = EplbState(
        num_logical_experts=32, num_redundant_experts=4,
        ep_size=4, rearrangement_step_interval=50, window_size=50
    )
    for step in range(100):
        ti = make_skewed_routing(
            num_tokens=1024, num_experts=32, top_k=2,
            hot_fraction=0.2, hot_load_fraction=0.6,
            seed=100 + step
        )
        per_logical = expert_load_counts(ti, 32)
        state.record_step(per_logical)
    p2l = state.physical_to_logical.tolist()
    # The chapter's headline numbers — these are the hot logical IDs after rearrange.
    assert p2l[-4:] == [5, 2, 0, 4]


def test_section_35_step0_imbalance_pin():
    """§3.5 step 0 (linear placement): rank loads ≈ [1292, 246, 257, 253], max/mean≈2.523."""
    ti = make_skewed_routing(
        num_tokens=1024, num_experts=32, top_k=2,
        hot_fraction=0.2, hot_load_fraction=0.6, seed=100
    )
    per_logical = expert_load_counts(ti, 32)
    per_rank = per_rank_load_from_logical_load(per_logical, ep_size=4, placement_strategy="linear")
    assert per_rank.tolist() == [1292, 246, 257, 253]
    state = EplbState(num_logical_experts=32, num_redundant_experts=4, ep_size=4)
    ratio = state.imbalance_ratio(per_rank)
    assert math.isclose(ratio, 2.523, abs_tol=0.01)


def test_section_35_step50_post_rebalance_pin():
    """§3.5 step 50 (after rearrange, switch to round_robin): [591, 616, 423, 418], 1.203."""
    # Step 50 used seed=150 with round_robin placement.
    ti = make_skewed_routing(
        num_tokens=1024, num_experts=32, top_k=2,
        hot_fraction=0.2, hot_load_fraction=0.6, seed=150
    )
    per_logical = expert_load_counts(ti, 32)
    per_rank = per_rank_load_from_logical_load(per_logical, ep_size=4, placement_strategy="round_robin")
    assert per_rank.tolist() == [591, 616, 423, 418]
    state = EplbState(num_logical_experts=32, num_redundant_experts=4, ep_size=4)
    ratio = state.imbalance_ratio(per_rank)
    assert math.isclose(ratio, 1.203, abs_tol=0.01)


def test_section_35_step25_imbalance_pin():
    """§3.5 step 25: [1295, 230, 261, 262]."""
    ti = make_skewed_routing(
        num_tokens=1024, num_experts=32, top_k=2,
        hot_fraction=0.2, hot_load_fraction=0.6, seed=125
    )
    per_logical = expert_load_counts(ti, 32)
    per_rank = per_rank_load_from_logical_load(per_logical, ep_size=4, placement_strategy="linear")
    assert per_rank.tolist() == [1295, 230, 261, 262]


# ---------------------------------------------------------------------------
# Trap E (NEGATIVE) — no aux-loss, no gradient anywhere in EPLB code
# ---------------------------------------------------------------------------


def test_trap_E_no_loss_balance_in_eplb_module():
    """NEGATIVE TEST for Trap E: the eplb module has zero training-loss CODE.

    We strip docstrings before grepping so explanatory prose ('vLLM is NOT
    using L_balance') doesn't false-positive. What we forbid is *executable*
    references to a backward pass, an optimizer step, or a balance-loss
    computation — none of which should appear in inference-only EPLB.
    """
    import ast

    src = inspect.getsource(eplb_module)
    # Parse and strip docstrings/comments so we only check code text.
    tree = ast.parse(src)
    code_str = ast.unparse(tree)
    # Remove module/class/function docstrings — ast.unparse keeps them as
    # bare string expressions; strip those too via walking the tree.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            ds = ast.get_docstring(node)
            if ds:
                code_str = code_str.replace(ds, "")

    forbidden_executable_terms = [
        ".backward(",
        "loss.backward",
        "torch.optim",
        "from torch.optim",
        "Optimizer(",
        "compute_aux_loss",
        "compute_balance_loss",
        "compute_load_balance_loss",
    ]
    for term in forbidden_executable_terms:
        assert term not in code_str, (
            f"TRAP E: inference-only EPLB module must not contain "
            f"training/optimization code '{term}'"
        )


def test_trap_E_no_gradient_tracking_in_record_step():
    """The load tensor passed to record_step is detached — no autograd graph survives."""
    state = EplbState(num_logical_experts=4, num_redundant_experts=0)
    load = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True).long()
    # We pass int64; record_step calls .detach().clone() to be safe.
    state.record_step(torch.tensor([1, 2, 3, 4], dtype=torch.int64))
    for h in state._load_history:
        assert not h.requires_grad
        assert not h.is_leaf or h.grad is None  # belt-and-suspenders: no grad


def test_trap_E_no_optimizer_in_module():
    """The EPLB module imports no optimizer / autograd primitives."""
    src = inspect.getsource(eplb_module)
    assert "from torch.optim" not in src
    assert "torch.autograd" not in src


def test_trap_E_vllm_source_eplb_has_no_aux_loss_computation():
    """Verify the impl-notes Trap E claim against the actual vLLM source tree.

    Greps ``instances/vllm/source/vllm/distributed/eplb/`` for executable
    aux-loss patterns. Stored config attributes
    (``router_aux_loss_coef`` carried from HuggingFace configs) are allowed
    — those are just attribute storage, never used in any forward/backward.
    What we forbid is *computed* loss in the EPLB code path.
    """
    # parents[3] = .../instances/vllm; source is at .../instances/vllm/source/...
    src_root = Path(__file__).resolve().parents[3] / "source/vllm/distributed/eplb"
    if not src_root.exists():
        # Source submodule isn't checked out — skip rather than false-positive.
        import pytest
        pytest.skip(f"vllm source not present at {src_root}")

    forbidden_patterns = [
        ".backward(",
        "loss.backward",
        "compute_aux_loss",
        "compute_balance_loss",
        "compute_load_balance_loss",
        "router_aux_loss_coef",  # config attribute, but should not appear in EPLB code
        "aux_loss =",  # assignment of an aux loss
    ]
    files_checked = 0
    for path in src_root.rglob("*.py"):
        files_checked += 1
        text = path.read_text()
        for pat in forbidden_patterns:
            assert pat not in text, (
                f"TRAP E: vllm/distributed/eplb/{path.name} contains '{pat}' "
                f"— EPLB is supposed to be inference-only statistical rebalance, "
                f"not training-time aux-loss balancing."
            )
    assert files_checked >= 3, f"expected to find ≥3 .py files in {src_root}, got {files_checked}"
