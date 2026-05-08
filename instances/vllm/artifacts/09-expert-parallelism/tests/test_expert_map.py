"""Fidelity tests for expert_map.py — determine_expert_map placement.

Verifies:

- Linear placement: rank r owns contiguous block ``[r*base + min(r,rem), ...]``.
- Round-robin placement: rank r owns ``r, r+P, r+2P, ...``.
- ``-1`` sentinel for off-rank experts (E03 in knowledge module).
- Remainder distribution: the first ``E % P`` ranks get one extra expert.
- ``ep_size==1`` short-circuits to ``(E, None)``.
- Cross-rank consistency: every expert is owned by exactly one rank.
- ``per_rank_token_load``: skew counting works for both placements.
"""

from __future__ import annotations

import torch

from implementation.expert_map import (
    VALID_STRATEGIES,
    all_rank_maps,
    determine_expert_map,
    get_compressed_expert_map,
    per_rank_token_load,
)


# ---------------------------------------------------------------------------
# ep_size==1 short-circuit
# ---------------------------------------------------------------------------


def test_ep1_returns_E_and_None_map():
    """ep_size==1 short-circuits — local==global and map is None (layer.py:L107-L109)."""
    local, m = determine_expert_map(ep_size=1, ep_rank=0, global_num_experts=16)
    assert local == 16
    assert m is None


def test_ep1_works_for_zero_rank_only():
    """Rank must be in [0, ep_size) — out-of-range ranks assert."""
    try:
        determine_expert_map(ep_size=4, ep_rank=4, global_num_experts=8)
    except AssertionError:
        return
    raise AssertionError("expected AssertionError for ep_rank == ep_size")


def test_negative_rank_asserts():
    """Negative rank also asserts."""
    try:
        determine_expert_map(ep_size=4, ep_rank=-1, global_num_experts=8)
    except AssertionError:
        return
    raise AssertionError("expected AssertionError for negative ep_rank")


# ---------------------------------------------------------------------------
# Linear placement
# ---------------------------------------------------------------------------


def test_linear_simple_8e_4ranks():
    """E=8, P=4, linear → rank 1 owns experts [2, 3] mapped to [0, 1]."""
    local, m = determine_expert_map(ep_size=4, ep_rank=1, global_num_experts=8)
    assert local == 2
    assert m.tolist() == [-1, -1, 0, 1, -1, -1, -1, -1]


def test_linear_rank0_8e_4ranks():
    """E=8, P=4, linear → rank 0 owns experts [0, 1] mapped to [0, 1]."""
    _, m = determine_expert_map(ep_size=4, ep_rank=0, global_num_experts=8)
    assert m.tolist() == [0, 1, -1, -1, -1, -1, -1, -1]


def test_linear_rank3_8e_4ranks():
    """E=8, P=4, linear → rank 3 owns experts [6, 7]."""
    _, m = determine_expert_map(ep_size=4, ep_rank=3, global_num_experts=8)
    assert m.tolist() == [-1, -1, -1, -1, -1, -1, 0, 1]


def test_linear_dtype_int32():
    """Map dtype must be int32 (matches source layer.py:L117)."""
    _, m = determine_expert_map(ep_size=4, ep_rank=0, global_num_experts=8)
    assert m.dtype == torch.int32


def test_linear_remainder_first_ranks_get_extra():
    """E=7, P=3: first 1 rank gets +1 (3 experts), the rest get 2."""
    locals_ = []
    maps = []
    for r in range(3):
        local, m = determine_expert_map(ep_size=3, ep_rank=r, global_num_experts=7)
        locals_.append(local)
        maps.append(m)
    # remainder=1 → first rank owns 3 experts, others own 2.
    assert locals_ == [3, 2, 2]
    # Every non-(-1) entry across all ranks covers all 7 experts exactly once.
    coverage = torch.zeros(7, dtype=torch.int32)
    for m in maps:
        on = (m != -1).int()
        coverage = coverage + on
    assert coverage.tolist() == [1, 1, 1, 1, 1, 1, 1]


def test_linear_remainder_block_starts():
    """Linear+remainder block-start offsets follow ``r*base + min(r, rem)`` (layer.py:L120)."""
    # E=10, P=3 → base=3, rem=1. Rank 0: start=0+0=0, owns 4. Rank 1: start=3+1=4, owns 3.
    _, m0 = determine_expert_map(ep_size=3, ep_rank=0, global_num_experts=10)
    _, m1 = determine_expert_map(ep_size=3, ep_rank=1, global_num_experts=10)
    _, m2 = determine_expert_map(ep_size=3, ep_rank=2, global_num_experts=10)
    # Rank 0 owns [0, 1, 2, 3].
    assert (m0 != -1).nonzero().squeeze(-1).tolist() == [0, 1, 2, 3]
    # Rank 1 owns [4, 5, 6].
    assert (m1 != -1).nonzero().squeeze(-1).tolist() == [4, 5, 6]
    # Rank 2 owns [7, 8, 9].
    assert (m2 != -1).nonzero().squeeze(-1).tolist() == [7, 8, 9]


# ---------------------------------------------------------------------------
# Round-robin placement
# ---------------------------------------------------------------------------


def test_round_robin_8e_2ranks_rank0():
    """E=8, P=2, round_robin → rank 0 owns 0,2,4,6 (stride P)."""
    local, m = determine_expert_map(
        ep_size=2, ep_rank=0, global_num_experts=8, expert_placement_strategy="round_robin"
    )
    assert local == 4
    assert m.tolist() == [0, -1, 1, -1, 2, -1, 3, -1]


def test_round_robin_8e_2ranks_rank1():
    """E=8, P=2, round_robin → rank 1 owns 1,3,5,7."""
    local, m = determine_expert_map(
        ep_size=2, ep_rank=1, global_num_experts=8, expert_placement_strategy="round_robin"
    )
    assert local == 4
    assert m.tolist() == [-1, 0, -1, 1, -1, 2, -1, 3]


def test_round_robin_remainder_rank0_extra():
    """E=9, P=4, round_robin: rank 0 owns 0,4,8 (3 experts); ranks 1-3 own 2 each."""
    locals_ = []
    for r in range(4):
        local, _ = determine_expert_map(
            ep_size=4, ep_rank=r, global_num_experts=9, expert_placement_strategy="round_robin"
        )
        locals_.append(local)
    assert locals_ == [3, 2, 2, 2]


def test_round_robin_coverage_complete():
    """All ranks together cover every expert exactly once."""
    P = 4
    E = 16
    coverage = torch.zeros(E, dtype=torch.int32)
    for r in range(P):
        _, m = determine_expert_map(
            ep_size=P, ep_rank=r, global_num_experts=E, expert_placement_strategy="round_robin"
        )
        coverage = coverage + (m != -1).int()
    assert torch.equal(coverage, torch.ones(E, dtype=torch.int32))


# ---------------------------------------------------------------------------
# Sentinel value
# ---------------------------------------------------------------------------


def test_minus_one_marks_off_rank_experts():
    """Off-rank entries are EXACTLY -1 (sentinel — E03 in knowledge module)."""
    _, m = determine_expert_map(ep_size=4, ep_rank=2, global_num_experts=16)
    off = m[m != -1].numel() == m[m >= 0].numel()
    assert off
    # Owned local indices are 0..(local-1) and dense.
    owned = m[m != -1].tolist()
    assert owned == sorted(owned)


def test_local_indices_are_dense_zero_based():
    """Owned local indices form ``[0, 1, ..., local_num_experts-1]`` exactly."""
    P = 5
    E = 17
    for r in range(P):
        local, m = determine_expert_map(ep_size=P, ep_rank=r, global_num_experts=E)
        owned = m[m != -1]
        assert sorted(owned.tolist()) == list(range(local))


# ---------------------------------------------------------------------------
# Cross-rank invariant
# ---------------------------------------------------------------------------


def test_every_expert_owned_by_exactly_one_rank_linear():
    """Linear: every global expert is owned by exactly one rank."""
    P, E = 8, 32
    coverage = torch.zeros(E, dtype=torch.int32)
    for r in range(P):
        _, m = determine_expert_map(ep_size=P, ep_rank=r, global_num_experts=E)
        coverage = coverage + (m != -1).int()
    assert (coverage == 1).all()


def test_every_expert_owned_by_exactly_one_rank_rr():
    """Round-robin: every global expert owned by exactly one rank."""
    P, E = 8, 32
    coverage = torch.zeros(E, dtype=torch.int32)
    for r in range(P):
        _, m = determine_expert_map(
            ep_size=P, ep_rank=r, global_num_experts=E, expert_placement_strategy="round_robin"
        )
        coverage = coverage + (m != -1).int()
    assert (coverage == 1).all()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_strategy_raises():
    """Unsupported strategies raise ValueError."""
    try:
        determine_expert_map(
            ep_size=4, ep_rank=0, global_num_experts=8, expert_placement_strategy="random"
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_valid_strategies_constant():
    """The implementation exposes the strategies it accepts as a contract."""
    assert "linear" in VALID_STRATEGIES
    assert "round_robin" in VALID_STRATEGIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_get_compressed_expert_map_format():
    """Pretty-print emits ``local->global`` pairs (mirror layer.py:L196-L214)."""
    _, m = determine_expert_map(ep_size=4, ep_rank=1, global_num_experts=8)
    s = get_compressed_expert_map(m)
    # rank 1 owns globals 2,3 → locals 0,1
    assert "0->2" in s
    assert "1->3" in s


def test_all_rank_maps_returns_P_entries():
    """``all_rank_maps`` builds maps for every rank in one call."""
    maps = all_rank_maps(ep_size=4, global_num_experts=16)
    assert len(maps) == 4
    locals_ = [m[0] for m in maps]
    assert locals_ == [4, 4, 4, 4]


def test_per_rank_token_load_linear():
    """Per-rank token-pair load matches the placement (linear, ep=4, E=32)."""
    P = 4
    E = 32
    M, K = 64, 2
    torch.manual_seed(0)
    ti = torch.randint(0, E, (M, K), dtype=torch.int32)
    maps = [determine_expert_map(P, r, E)[1] for r in range(P)]
    loads = per_rank_token_load(ti, maps)
    # Sum equals M*K (every token-slot lands on exactly one rank).
    assert int(loads.sum().item()) == M * K


def test_per_rank_token_load_ep1_returns_full_load():
    """ep_size==1 — the only rank gets all M*K pairs."""
    M, K = 32, 2
    ti = torch.randint(0, 8, (M, K), dtype=torch.int32)
    maps = [None]
    loads = per_rank_token_load(ti, maps)
    assert int(loads[0].item()) == M * K


def test_per_rank_token_load_round_robin_balances_uniform():
    """Under uniform routing + round_robin, ranks see ~ equal load."""
    torch.manual_seed(1)
    P = 4
    E = 16
    M, K = 4096, 2
    ti = torch.randint(0, E, (M, K), dtype=torch.int32)
    maps = [
        determine_expert_map(P, r, E, expert_placement_strategy="round_robin")[1]
        for r in range(P)
    ]
    loads = per_rank_token_load(ti, maps)
    mean = loads.float().mean().item()
    max_dev = (loads.float() - mean).abs().max().item()
    # uniform routing → each rank gets ~M*K/P ± noise; max deviation < 5% of mean.
    assert max_dev / mean < 0.05
