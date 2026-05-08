"""Tests for parallel_state_dcp_pcp.py — _DCP/_PCP singletons + 5D mesh init.

Source: vllm/distributed/parallel_state.py:L1234-L1782
"""

from __future__ import annotations

import pytest

from implementation import parallel_state_dcp_pcp as ps


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset _DCP / _PCP between tests (avoid cross-test leakage)."""
    ps.reset_cp_singletons()
    yield
    ps.reset_cp_singletons()


# --------------------------------------------------------------------------
# Singleton accessor uninitialised behaviour (mirrors source AssertionError)
# --------------------------------------------------------------------------


def test_get_dcp_group_raises_when_uninitialized():
    """Source raises AssertionError before init; we mirror exactly."""
    with pytest.raises(AssertionError, match="decode context model parallel"):
        ps.get_dcp_group()


def test_get_pcp_group_raises_when_uninitialized():
    """Source raises AssertionError before init; we mirror exactly."""
    with pytest.raises(AssertionError, match="prefill context parallel"):
        ps.get_pcp_group()


def test_get_context_model_parallel_group_alias_is_get_dcp_group():
    """Source backward-compat alias must point to get_dcp_group."""
    assert ps.get_context_model_parallel_group is ps.get_dcp_group


def test_get_decode_world_size_helper_after_init():
    ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=4,
        decode_context_model_parallel_size=2,
    )
    assert ps.get_decode_context_model_parallel_world_size() == 2


def test_get_decode_rank_helper_after_init():
    ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=4,
        decode_context_model_parallel_size=2,
    )
    assert ps.get_decode_context_model_parallel_rank() == 0


# --------------------------------------------------------------------------
# CPGroupCoordinator dataclass
# --------------------------------------------------------------------------


def test_cp_group_coordinator_world_size_property():
    g = ps.CPGroupCoordinator(group_name="dcp", ranks=[0, 1, 2, 3], rank_in_group=2)
    assert g.world_size == 4


def test_cp_group_coordinator_rank_in_group_zero_indexed():
    g = ps.CPGroupCoordinator(group_name="dcp", ranks=[4, 5, 6, 7], rank_in_group=0)
    assert g.rank_in_group == 0
    assert g.ranks[g.rank_in_group] == 4


# --------------------------------------------------------------------------
# tp % dcp == 0 hard constraint (parallel.py:L474-L478)
# --------------------------------------------------------------------------


def test_tp_must_be_divisible_by_dcp():
    with pytest.raises(ValueError, match="must be divisible"):
        ps.initialize_model_parallel(
            rank=0,
            world_size=4,
            tensor_model_parallel_size=4,
            decode_context_model_parallel_size=3,
        )


def test_tp_equal_to_dcp_is_ok():
    """tp == dcp is the trivial-divisible case; allowed by source."""
    ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=4,
        decode_context_model_parallel_size=4,
    )
    assert ps.get_dcp_group().world_size == 4


def test_dcp_one_is_always_ok():
    """dcp=1 means no DCP; tp % 1 == 0 trivially."""
    for tp in (1, 2, 3, 4, 7, 8, 16):
        ps.reset_cp_singletons()
        ps.initialize_model_parallel(
            rank=0,
            world_size=tp,
            tensor_model_parallel_size=tp,
            decode_context_model_parallel_size=1,
        )
        assert ps.get_dcp_group().world_size == 1


@pytest.mark.parametrize("tp,dcp,ok", [
    (8, 1, True),
    (8, 2, True),
    (8, 4, True),
    (8, 8, True),
    (8, 3, False),
    (8, 5, False),
    (8, 6, False),
    (8, 7, False),
    (4, 3, False),
    (4, 4, True),
])
def test_tp_dcp_combinations(tp, dcp, ok):
    if ok:
        ps.initialize_model_parallel(
            rank=0,
            world_size=tp,
            tensor_model_parallel_size=tp,
            decode_context_model_parallel_size=dcp,
        )
    else:
        with pytest.raises(ValueError):
            ps.initialize_model_parallel(
                rank=0,
                world_size=tp,
                tensor_model_parallel_size=tp,
                decode_context_model_parallel_size=dcp,
            )


# --------------------------------------------------------------------------
# world_size = tp x pp x pcp x dp (NOT x dcp); multiproc_executor.py:L116-L121
# --------------------------------------------------------------------------


def test_world_size_excludes_dcp():
    """world_size assertion: tp * pp * pcp * dp; dcp is folded inside TP."""
    ps.initialize_model_parallel(
        rank=0,
        world_size=8,  # 4 tp * 2 pcp = 8 (NOT * 2 dcp)
        tensor_model_parallel_size=4,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
    )


def test_world_size_mismatch_raises():
    """If world_size doesn't equal product, raise."""
    with pytest.raises(ValueError, match="world_size"):
        ps.initialize_model_parallel(
            rank=0,
            world_size=16,  # would require world to include dcp
            tensor_model_parallel_size=4,
            prefill_context_model_parallel_size=2,
            decode_context_model_parallel_size=2,
        )


def test_world_size_5d_full_product():
    """Demo §5 config: tp=4, pcp=2, pp=2, dcp=2 -> world=16 (NOT 32)."""
    groups = ps.initialize_model_parallel(
        rank=0,
        world_size=16,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
    )
    assert ps.get_dcp_group().world_size == 2
    assert ps.get_pcp_group().world_size == 2
    assert len(groups["tp"]) == 4
    assert len(groups["pp"]) == 8


# --------------------------------------------------------------------------
# 5D mesh group construction (parallel_state.py:L1569-L1633)
# --------------------------------------------------------------------------


def test_simple_tp_only_world():
    """tp=4 only → 1 TP group, 1 DCP group (size 1 each), 1 PCP group, etc."""
    groups = ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=4,
    )
    assert groups["tp"] == [[0, 1, 2, 3]]
    assert len(groups["dcp"]) == 4  # tp=4, dcp=1 → 4 sub-groups of size 1
    assert all(len(g) == 1 for g in groups["dcp"])
    assert len(groups["pcp"]) == 4
    assert len(groups["pp"]) == 4
    assert len(groups["dp"]) == 4


def test_dcp_sub_groups_chunk_tp_groups_contiguously():
    """DCP folds INSIDE TP. Each TP group of tp_size splits into tp/dcp DCP sub-groups.

    (tp=4, dcp=2) → each TP group [a,b,c,d] becomes DCP sub-groups [a,b] and [c,d].
    REFERENCE: vllm/distributed/parallel_state.py:L1594-L1614
    """
    groups = ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=4,
        decode_context_model_parallel_size=2,
    )
    assert groups["tp"] == [[0, 1, 2, 3]]
    assert groups["dcp"] == [[0, 1], [2, 3]]


def test_pcp_groups_via_transpose_3_4():
    """PCP groups are built by transposing pcp <-> tp axes BEFORE reshape.

    With tp=2, pcp=2 → all_ranks 4-tensor [[0,1],[2,3]] (pcp_axis=outer, tp=inner).
    PCP groups span the pcp axis at fixed (ext, d, p, t):
       t=0: ranks at pcp=0 → 0; pcp=1 → 2 → group [0, 2]
       t=1: ranks at pcp=0 → 1; pcp=1 → 3 → group [1, 3]
    REFERENCE: vllm/distributed/parallel_state.py:L1616-L1633
    """
    groups = ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=2,
        prefill_context_model_parallel_size=2,
    )
    assert groups["pcp"] == [[0, 2], [1, 3]]


def test_pp_groups_span_pp_axis():
    """PP groups span the pp axis at fixed (ext, d, c, t)."""
    groups = ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
    )
    # 5D mesh shape (1, 1, pp=2, pcp=1, tp=2). tp=2 so PP groups are
    # [0,2] and [1,3] (group ranks span the pp axis at fixed t).
    assert groups["pp"] == [[0, 2], [1, 3]]


def test_dp_groups_size_1_when_dp_1():
    """When dp=1, each DP group has a single rank."""
    groups = ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=4,
    )
    for g in groups["dp"]:
        assert len(g) == 1


# --------------------------------------------------------------------------
# Demo §5 ground truth: world=16 with (tp=4, pcp=2, pp=2, dcp=2)
# --------------------------------------------------------------------------


def _demo5_groups():
    """Build groups for the (tp=4, pcp=2, pp=2, dp=1, dcp=2) demo config."""
    return ps.initialize_model_parallel(
        rank=0,
        world_size=16,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
    )


def test_demo5_tp_groups_count_4():
    groups = _demo5_groups()
    assert len(groups["tp"]) == 4


def test_demo5_dcp_subgroups_count_8():
    """Demo §5: 8 DCP sub-groups (4 TP groups × 2 sub-groups each)."""
    groups = _demo5_groups()
    assert len(groups["dcp"]) == 8


def test_demo5_pcp_groups_count_8():
    """Demo §5: 8 PCP groups."""
    groups = _demo5_groups()
    assert len(groups["pcp"]) == 8


def test_demo5_pp_groups_count_8():
    """Demo §5: 8 PP groups."""
    groups = _demo5_groups()
    assert len(groups["pp"]) == 8


def test_demo5_first_tp_group():
    """Verbatim from demo §5: first TP group is [0,1,2,3]."""
    groups = _demo5_groups()
    assert groups["tp"][0] == [0, 1, 2, 3]


def test_demo5_dcp_subgroups_are_contiguous_pairs():
    """Each DCP sub-group is a contiguous chunk of size dcp inside the TP group."""
    groups = _demo5_groups()
    expected = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    assert groups["dcp"] == expected


def test_demo5_pcp_groups_first_three():
    """Verbatim from demo §5: first PCP groups are [0,4], [1,5], [2,6]."""
    groups = _demo5_groups()
    assert groups["pcp"][0] == [0, 4]
    assert groups["pcp"][1] == [1, 5]
    assert groups["pcp"][2] == [2, 6]


def test_demo5_pp_groups_first_three():
    """Verbatim from demo §5: first PP groups are [0,8], [1,9], [2,10]."""
    groups = _demo5_groups()
    assert groups["pp"][0] == [0, 8]
    assert groups["pp"][1] == [1, 9]
    assert groups["pp"][2] == [2, 10]


# --------------------------------------------------------------------------
# Singleton population per-rank
# --------------------------------------------------------------------------


def test_singletons_populated_for_rank_zero():
    ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=4,
        decode_context_model_parallel_size=2,
    )
    assert ps.get_dcp_group().rank_in_group == 0


def test_singletons_populated_for_rank_one():
    ps.initialize_model_parallel(
        rank=1,
        world_size=4,
        tensor_model_parallel_size=4,
        decode_context_model_parallel_size=2,
    )
    assert ps.get_dcp_group().rank_in_group == 1


def test_singletons_populated_for_rank_three():
    """Rank 3 in DCP sub-group [2,3] → rank_in_group is 1."""
    ps.initialize_model_parallel(
        rank=3,
        world_size=4,
        tensor_model_parallel_size=4,
        decode_context_model_parallel_size=2,
    )
    assert ps.get_dcp_group().rank_in_group == 1
    assert ps.get_dcp_group().ranks == [2, 3]


def test_pcp_singleton_rank_in_group_one():
    """In a 4-rank world with tp=2, pcp=2, rank 2 lives in PCP group [0,2] at index 1."""
    ps.initialize_model_parallel(
        rank=2,
        world_size=4,
        tensor_model_parallel_size=2,
        prefill_context_model_parallel_size=2,
    )
    assert ps.get_pcp_group().rank_in_group == 1


# --------------------------------------------------------------------------
# Each rank appears in exactly one group of each kind
# --------------------------------------------------------------------------


def test_each_rank_in_exactly_one_dcp_group():
    """Partition invariant: every rank in exactly one DCP sub-group."""
    groups = _demo5_groups()
    seen = set()
    for g in groups["dcp"]:
        for r in g:
            assert r not in seen
            seen.add(r)
    assert seen == set(range(16))


def test_each_rank_in_exactly_one_pcp_group():
    """Partition invariant: every rank in exactly one PCP group."""
    groups = _demo5_groups()
    seen = set()
    for g in groups["pcp"]:
        for r in g:
            assert r not in seen
            seen.add(r)
    assert seen == set(range(16))


def test_each_rank_in_exactly_one_tp_group():
    groups = _demo5_groups()
    seen = set()
    for g in groups["tp"]:
        for r in g:
            assert r not in seen
            seen.add(r)
    assert seen == set(range(16))


def test_each_rank_in_exactly_one_pp_group():
    groups = _demo5_groups()
    seen = set()
    for g in groups["pp"]:
        for r in g:
            assert r not in seen
            seen.add(r)
    assert seen == set(range(16))


# --------------------------------------------------------------------------
# Per-group size invariants
# --------------------------------------------------------------------------


def test_all_dcp_subgroups_have_size_dcp():
    groups = _demo5_groups()
    for g in groups["dcp"]:
        assert len(g) == 2  # dcp_size


def test_all_pcp_groups_have_size_pcp():
    groups = _demo5_groups()
    for g in groups["pcp"]:
        assert len(g) == 2


def test_all_tp_groups_have_size_tp():
    groups = _demo5_groups()
    for g in groups["tp"]:
        assert len(g) == 4


def test_all_pp_groups_have_size_pp():
    groups = _demo5_groups()
    for g in groups["pp"]:
        assert len(g) == 2
