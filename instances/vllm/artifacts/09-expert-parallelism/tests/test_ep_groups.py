"""Fidelity tests for ep_groups.py — _EP/_EPLB singletons + ParallelConfig.

Verifies:

- ``FusedMoEParallelConfig.make`` collapse rule: ``ep_size = tp*dp*pcp; tp_size=1``
  when EP is on (config.py:L1192-L1208).
- ``use_ep`` toggles correctly based on `enable_expert_parallel` and product>1.
- ``flatten_tp_across_dp_and_pcp``: rank math matches config.py:L1071.
- ``_EP`` and ``_EPLB`` are SEPARATE singletons (parallel_state.py:L1700-L1719);
  initialising EPLB does NOT alias to _EP — different `group_name`.
- ``init_ep_group`` mesh math: ``transpose(1,2).reshape(-1, dp*pcp*tp)`` produces
  contiguous EP rank groups containing the calling rank.
- world_size mismatch asserts.
- ``get_ep_group`` / ``get_eplb_group`` raise when uninitialized.
- Dense (non-MoE) models: ``_EP`` stays None.
"""

from __future__ import annotations

import torch

from implementation import ep_groups
from implementation.ep_groups import (
    EPGroup,
    FusedMoEParallelConfig,
    get_ep_group,
    get_eplb_group,
    init_ep_group,
    reset_groups,
)


# ---------------------------------------------------------------------------
# FusedMoEParallelConfig.make — EP=False path
# ---------------------------------------------------------------------------


def test_make_ep_false_keeps_tp_size():
    """No EP: tp_size stays at the flattened value, ep_size=1."""
    cfg = FusedMoEParallelConfig.make(
        tp_size_=4,
        pcp_size_=1,
        dp_size_=1,
        sp_size_=1,
        enable_expert_parallel=False,
    )
    assert cfg.use_ep is False
    assert cfg.tp_size == 4
    assert cfg.ep_size == 1
    assert cfg.ep_rank == 0


def test_make_ep_false_with_dp_pcp_flattens():
    """No EP: tp_size = dp * pcp * tp (flattened across DP+PCP)."""
    cfg = FusedMoEParallelConfig.make(
        tp_size_=2,
        pcp_size_=2,
        dp_size_=2,
        sp_size_=1,
        enable_expert_parallel=False,
    )
    assert cfg.use_ep is False
    assert cfg.tp_size == 8  # 2*2*2
    assert cfg.ep_size == 1


def test_make_world_size_one_disables_ep_even_if_requested():
    """If tp*dp*pcp == 1, EP is forced off (one rank can't have multiple EP slots)."""
    cfg = FusedMoEParallelConfig.make(
        tp_size_=1,
        pcp_size_=1,
        dp_size_=1,
        sp_size_=1,
        enable_expert_parallel=True,
    )
    assert cfg.use_ep is False
    assert cfg.ep_size == 1


# ---------------------------------------------------------------------------
# FusedMoEParallelConfig.make — EP=True path (the COLLAPSE rule)
# ---------------------------------------------------------------------------


def test_make_ep_true_collapses_tp_to_one():
    """EP=True: tp_size collapses to 1, ep_size = flattened tp*dp*pcp (E08, config.py:L1192-L1208)."""
    cfg = FusedMoEParallelConfig.make(
        tp_size_=4,
        pcp_size_=1,
        dp_size_=1,
        sp_size_=1,
        enable_expert_parallel=True,
    )
    assert cfg.use_ep is True
    assert cfg.tp_size == 1
    assert cfg.ep_size == 4


def test_make_ep_with_dp_collapses():
    """EP=True with DP: ep_size = tp*dp."""
    cfg = FusedMoEParallelConfig.make(
        tp_size_=2,
        pcp_size_=1,
        dp_size_=4,
        sp_size_=1,
        enable_expert_parallel=True,
    )
    assert cfg.ep_size == 8
    assert cfg.tp_size == 1


def test_make_ep_with_pcp_collapses():
    """EP=True with PCP: ep_size = tp*pcp."""
    cfg = FusedMoEParallelConfig.make(
        tp_size_=2,
        pcp_size_=4,
        dp_size_=1,
        sp_size_=1,
        enable_expert_parallel=True,
    )
    assert cfg.ep_size == 8
    assert cfg.tp_size == 1


def test_make_ep_full_3way_mesh():
    """EP=True with all 3 axes: ep_size = tp*dp*pcp."""
    cfg = FusedMoEParallelConfig.make(
        tp_size_=2,
        pcp_size_=2,
        dp_size_=2,
        sp_size_=1,
        enable_expert_parallel=True,
    )
    assert cfg.ep_size == 8
    assert cfg.tp_size == 1


def test_make_ep_rank_uses_flatten_formula():
    """EP rank = flatten_tp_rank = dp_rank*pcp*tp + pcp_rank*tp + 0 (config.py:L1071)."""
    cfg = FusedMoEParallelConfig.make(
        tp_size_=2,
        pcp_size_=2,
        dp_size_=2,
        sp_size_=1,
        enable_expert_parallel=True,
        dp_rank=1,
        pcp_rank=1,
    )
    # flatten_tp_rank = 1*2*2 + 1*2 + 0 = 6
    assert cfg.ep_rank == 6


def test_flatten_tp_across_dp_and_pcp_matches_formula():
    """Static helper agrees with config.py:L1071-L1079."""
    flat_size, flat_rank = FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(
        tp_size=4, dp_size=2, dp_rank=1, pcp_size=2, pcp_rank=1
    )
    assert flat_size == 4 * 2 * 2  # 16
    assert flat_rank == 1 * 2 * 4 + 1 * 4 + 0  # 12


# ---------------------------------------------------------------------------
# Other config properties
# ---------------------------------------------------------------------------


def test_use_all2all_kernels_requires_dp_and_ep():
    """``use_all2all_kernels`` is True iff dp_size>1 AND use_ep (config.py:L1019-L1020)."""
    cfg_ep_no_dp = FusedMoEParallelConfig.make(
        tp_size_=4, pcp_size_=1, dp_size_=1, sp_size_=1, enable_expert_parallel=True
    )
    assert cfg_ep_no_dp.use_all2all_kernels is False  # EP yes but dp_size=1

    cfg_ep_with_dp = FusedMoEParallelConfig.make(
        tp_size_=2, pcp_size_=1, dp_size_=4, sp_size_=1, enable_expert_parallel=True
    )
    assert cfg_ep_with_dp.use_all2all_kernels is True

    cfg_no_ep = FusedMoEParallelConfig.make(
        tp_size_=2, pcp_size_=1, dp_size_=4, sp_size_=1, enable_expert_parallel=False
    )
    assert cfg_no_ep.use_all2all_kernels is False  # no EP


def test_is_sequence_parallel_when_sp_gt_1():
    """``is_sequence_parallel`` flag is set when sp_size > 1."""
    cfg = FusedMoEParallelConfig.make(
        tp_size_=4, pcp_size_=1, dp_size_=1, sp_size_=2, enable_expert_parallel=False
    )
    assert cfg.is_sequence_parallel is True


def test_is_sequence_parallel_false_when_sp_one():
    """sp_size=1 → not sequence parallel."""
    cfg = FusedMoEParallelConfig.make(
        tp_size_=4, pcp_size_=1, dp_size_=1, sp_size_=1, enable_expert_parallel=False
    )
    assert cfg.is_sequence_parallel is False


def test_enable_eplb_propagates():
    """``enable_eplb`` flag is preserved through the config builder."""
    cfg = FusedMoEParallelConfig.make(
        tp_size_=4, pcp_size_=1, dp_size_=1, sp_size_=1,
        enable_expert_parallel=True, enable_eplb=True
    )
    assert cfg.enable_eplb is True


def test_all2all_backend_default_and_propagation():
    """Default ``all2all_backend`` is ``allgather_reducescatter`` (AgRs); custom values pass through."""
    cfg = FusedMoEParallelConfig.make(
        tp_size_=2, pcp_size_=1, dp_size_=2, sp_size_=1, enable_expert_parallel=True
    )
    assert cfg.all2all_backend == "allgather_reducescatter"

    cfg2 = FusedMoEParallelConfig.make(
        tp_size_=2, pcp_size_=1, dp_size_=2, sp_size_=1,
        enable_expert_parallel=True, all2all_backend="deepep_high_throughput"
    )
    assert cfg2.all2all_backend == "deepep_high_throughput"


# ---------------------------------------------------------------------------
# init_ep_group — mesh math + group construction
# ---------------------------------------------------------------------------


def test_init_ep_group_basic_assignment():
    """4-rank mesh (tp=2, dp=2): each rank lives in an EP group of size 4."""
    reset_groups()
    grp = init_ep_group(world_size=4, rank=0, tp_size=2, dp_size=2, pcp_size=1)
    assert grp is not None
    assert grp.world_size == 4
    assert grp.rank_in_group == 0
    assert sorted(grp.rank_list) == [0, 1, 2, 3]
    reset_groups()


def test_init_ep_group_separate_calls_pick_correct_rank():
    """Each rank in the world picks its own EP group."""
    for r in range(4):
        reset_groups()
        grp = init_ep_group(world_size=4, rank=r, tp_size=2, dp_size=2, pcp_size=1)
        assert r in grp.rank_list
        assert grp.rank_in_group == grp.rank_list.index(r)
    reset_groups()


def test_init_ep_group_world_size_mismatch_asserts():
    """world_size != tp*dp*pcp asserts."""
    reset_groups()
    try:
        init_ep_group(world_size=4, rank=0, tp_size=2, dp_size=3, pcp_size=1)
    except AssertionError:
        reset_groups()
        return
    raise AssertionError("expected AssertionError")


def test_init_ep_group_dense_model_returns_None():
    """is_moe=False → no _EP group is created (E01 — knowledge module)."""
    reset_groups()
    out = init_ep_group(world_size=4, rank=0, tp_size=2, dp_size=2, pcp_size=1, is_moe=False)
    assert out is None
    assert ep_groups._EP is None
    reset_groups()


def test_init_ep_double_init_asserts():
    """Re-initializing without reset asserts (mirror of source idempotency check)."""
    reset_groups()
    init_ep_group(world_size=4, rank=0, tp_size=2, dp_size=2, pcp_size=1)
    try:
        init_ep_group(world_size=4, rank=0, tp_size=2, dp_size=2, pcp_size=1)
    except AssertionError:
        reset_groups()
        return
    reset_groups()
    raise AssertionError("expected AssertionError on double init")


def test_init_ep_8rank_mesh_is_correct():
    """8-rank mesh (tp=2, dp=2, pcp=2): groups have size 8 (full collapse)."""
    reset_groups()
    grp = init_ep_group(world_size=8, rank=3, tp_size=2, dp_size=2, pcp_size=2)
    assert grp.world_size == 8
    # all 8 ranks should be in this rank's EP group.
    assert sorted(grp.rank_list) == list(range(8))
    reset_groups()


# ---------------------------------------------------------------------------
# _EP vs _EPLB are SEPARATE
# ---------------------------------------------------------------------------


def test_eplb_group_is_distinct_object_from_ep():
    """``_EPLB`` is a SEPARATE EPGroup instance with the SAME rank list (E02)."""
    reset_groups()
    init_ep_group(
        world_size=4, rank=0, tp_size=2, dp_size=2, pcp_size=1, enable_eplb=True
    )
    ep = get_ep_group()
    eplb = get_eplb_group()
    assert ep is not eplb  # NOT the same object
    assert ep.rank_list == eplb.rank_list  # same membership
    assert ep.group_name == "ep"
    assert eplb.group_name == "eplb"
    reset_groups()


def test_eplb_not_created_when_disabled():
    """``enable_eplb=False`` (default) → ``_EPLB`` is None; ``get_eplb_group`` asserts."""
    reset_groups()
    init_ep_group(world_size=4, rank=0, tp_size=2, dp_size=2, pcp_size=1, enable_eplb=False)
    try:
        get_eplb_group()
    except AssertionError:
        reset_groups()
        return
    reset_groups()
    raise AssertionError("expected AssertionError")


def test_get_ep_group_uninitialized_asserts():
    """``get_ep_group`` raises when not yet initialized (E01 message contract)."""
    reset_groups()
    try:
        get_ep_group()
    except AssertionError as e:
        assert "expert parallel group is not initialized" in str(e)
        return
    raise AssertionError("expected AssertionError")


def test_reset_groups_clears_both():
    """``reset_groups`` clears _EP and _EPLB so the next init can run."""
    reset_groups()
    init_ep_group(world_size=4, rank=0, tp_size=2, dp_size=2, pcp_size=1, enable_eplb=True)
    reset_groups()
    assert ep_groups._EP is None
    assert ep_groups._EPLB is None


# ---------------------------------------------------------------------------
# Mesh transpose+reshape produces correct partitions
# ---------------------------------------------------------------------------


def test_mesh_construction_partitions_world():
    """Across all ranks 0..world-1, every rank lands in exactly one EP group."""
    world = 8
    coverage = [0] * world
    for r in range(world):
        reset_groups()
        grp = init_ep_group(world_size=world, rank=r, tp_size=2, dp_size=2, pcp_size=2)
        for x in grp.rank_list:
            if x == r:
                coverage[r] += 1
    assert coverage == [1] * world
    reset_groups()


def test_eplb_group_is_repr_friendly():
    """EPGroup __repr__ includes name, world_size, rank_in_group, ranks."""
    reset_groups()
    grp = init_ep_group(world_size=4, rank=0, tp_size=2, dp_size=2, pcp_size=1)
    s = repr(grp)
    assert "EPGroup" in s
    assert "world_size=4" in s
    assert "rank_in_group=0" in s
    reset_groups()


# ---------------------------------------------------------------------------
# Trap D (E02): EPLB has its own group — separation prevents deadlock
# ---------------------------------------------------------------------------


def test_trap_D_eplb_separate_group_prevents_aliasing():
    """The point of two groups: a forward-pass dispatch uses _EP; an EPLB rebalance
    uses _EPLB — they are different Python objects and can be coordinated separately.
    Verifies E02 / Trap D from impl-notes."""
    reset_groups()
    init_ep_group(world_size=4, rank=0, tp_size=4, dp_size=1, pcp_size=1, enable_eplb=True)
    ep = ep_groups._EP
    eplb = ep_groups._EPLB
    assert ep is not None
    assert eplb is not None
    assert id(ep) != id(eplb)  # distinct heap objects
    reset_groups()
