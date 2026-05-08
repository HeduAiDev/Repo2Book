"""Tests for world_topology.py — 5D MeshConfig + per_rank_kv_fraction.

Source: vllm/v1/executor/multiproc_executor.py:L116-L121, L985-L1004
"""

from __future__ import annotations

import pytest

from implementation.world_topology import (
    MeshConfig,
    per_rank_kv_fraction,
    process_name_for_rank,
)


# --------------------------------------------------------------------------
# tp % dcp == 0 hard constraint enforced in __post_init__
# --------------------------------------------------------------------------


def test_mesh_config_default_world_size_is_one():
    m = MeshConfig()
    assert m.world_size == 1
    assert m.dcp == 1
    assert m.pcp == 1
    assert m.tp == 1


def test_mesh_config_tp_dcp_indivisible_raises():
    with pytest.raises(ValueError, match="must be divisible"):
        MeshConfig(tp=4, dcp=3)


def test_mesh_config_tp_dcp_zero_dcp_raises_division():
    """dcp=0 would be ill-defined; tp%0 raises ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        MeshConfig(tp=4, dcp=0)


@pytest.mark.parametrize("tp,dcp,ok", [
    (1, 1, True),
    (2, 1, True),
    (2, 2, True),
    (4, 1, True),
    (4, 2, True),
    (4, 4, True),
    (8, 1, True),
    (8, 2, True),
    (8, 4, True),
    (8, 8, True),
    (4, 3, False),
    (4, 5, False),
    (8, 3, False),
    (8, 5, False),
    (8, 6, False),  # 8 % 6 = 2 ≠ 0
    (8, 7, False),
])
def test_tp_dcp_divisibility_grid(tp, dcp, ok):
    if ok:
        MeshConfig(tp=tp, dcp=dcp)  # no exception
    else:
        with pytest.raises(ValueError):
            MeshConfig(tp=tp, dcp=dcp)


# --------------------------------------------------------------------------
# world_size formula = ext_dp * dp * pp * pcp * tp; DCP excluded
# REFERENCE: vllm/v1/executor/multiproc_executor.py:L116-L121
# --------------------------------------------------------------------------


def test_world_size_excludes_dcp():
    """DCP is folded inside TP; does NOT enter the world_size product."""
    m = MeshConfig(tp=8, dcp=4)
    # If DCP entered the product, world would be 32; instead it's 8.
    assert m.world_size == 8


def test_world_size_5d_full_product():
    """5D mesh: ext_dp=1, dp=2, pp=2, pcp=2, tp=2 → world=16."""
    m = MeshConfig(external_dp=1, dp=2, pp=2, pcp=2, tp=2)
    assert m.world_size == 1 * 2 * 2 * 2 * 2


def test_world_size_demo5_config():
    """Demo §5 verbatim: (tp=4, pcp=2, pp=2, dp=1, dcp=2) → world=16."""
    m = MeshConfig(external_dp=1, dp=1, pp=2, pcp=2, tp=4, dcp=2)
    assert m.world_size == 16


def test_world_size_independent_of_dcp_value():
    """For fixed tp, varying dcp does not change world_size."""
    base = MeshConfig(tp=8, dcp=1).world_size
    for dcp in (1, 2, 4, 8):
        assert MeshConfig(tp=8, dcp=dcp).world_size == base


def test_world_size_grows_with_pcp():
    """PCP DOES enter world_size (independent axis)."""
    base = MeshConfig(tp=4, pcp=1).world_size
    for pcp in (2, 4, 8):
        assert MeshConfig(tp=4, pcp=pcp).world_size == base * pcp


def test_world_size_grows_with_external_dp():
    """external_dp (verl) multiplies world_size."""
    assert MeshConfig(external_dp=4, tp=2).world_size == 8


def test_world_size_grows_with_pp():
    assert MeshConfig(pp=4, tp=2).world_size == 8


def test_world_size_grows_with_dp():
    assert MeshConfig(dp=4, tp=2).world_size == 8


# --------------------------------------------------------------------------
# total_cp_world_size = pcp * dcp (REFERENCE: backend.py:L751)
# --------------------------------------------------------------------------


def test_total_cp_world_size_formula():
    """Composed CP = pcp * dcp; this is the KV-shard count."""
    m = MeshConfig(tp=8, pcp=4, dcp=2)
    assert m.total_cp_world_size == 8


def test_total_cp_at_dcp_1_pcp_1_is_1():
    assert MeshConfig().total_cp_world_size == 1


@pytest.mark.parametrize("pcp,dcp,expected", [
    (1, 1, 1),
    (1, 2, 2),
    (2, 1, 2),
    (2, 2, 4),
    (4, 4, 16),
    (8, 4, 32),
])
def test_total_cp_grid(pcp, dcp, expected):
    m = MeshConfig(tp=max(8, dcp), pcp=pcp, dcp=dcp)
    assert m.total_cp_world_size == expected


# --------------------------------------------------------------------------
# num_dcp_subgroups = tp // dcp
# --------------------------------------------------------------------------


def test_num_dcp_subgroups_at_tp_8_dcp_2():
    """tp=8, dcp=2 → each TP group of 8 splits into 4 DCP sub-groups of 2."""
    m = MeshConfig(tp=8, dcp=2)
    assert m.num_dcp_subgroups == 4


def test_num_dcp_subgroups_at_dcp_eq_tp():
    """tp=8, dcp=8 → each TP group is one DCP group."""
    m = MeshConfig(tp=8, dcp=8)
    assert m.num_dcp_subgroups == 1


def test_num_dcp_subgroups_at_dcp_1():
    """dcp=1 → each TP rank is its own DCP sub-group."""
    m = MeshConfig(tp=8, dcp=1)
    assert m.num_dcp_subgroups == 8


# --------------------------------------------------------------------------
# per_rank_kv_fraction = 1 / (pcp * dcp)
# --------------------------------------------------------------------------


def test_per_rank_kv_fraction_no_cp():
    m = MeshConfig()
    assert per_rank_kv_fraction(m) == 1.0


def test_per_rank_kv_fraction_demo5():
    """Demo §5: total_cp = 4 → 1/4 fraction."""
    m = MeshConfig(tp=4, pcp=2, dcp=2)
    assert per_rank_kv_fraction(m) == 0.25


def test_per_rank_kv_fraction_full_4_4():
    m = MeshConfig(tp=4, pcp=4, dcp=4)
    assert per_rank_kv_fraction(m) == 1.0 / 16


# --------------------------------------------------------------------------
# process_name_for_rank — mirrors multiproc_executor.py:L985-L1004
# --------------------------------------------------------------------------


def test_process_name_default_is_just_worker():
    m = MeshConfig()
    assert process_name_for_rank(0, m) == "Worker"


def test_process_name_includes_pcp_when_pcp_gt_1():
    """REFERENCE: multiproc_executor.py:L991-L1004 — only axes >1 appear."""
    m = MeshConfig(pcp=2, tp=2)
    name = process_name_for_rank(0, m)
    assert "PCP0" in name
    assert "TP0" in name


def test_process_name_includes_dcp_when_dcp_gt_1():
    m = MeshConfig(tp=4, dcp=2)
    name = process_name_for_rank(0, m)
    assert "DCP0" in name
    assert "TP0" in name


def test_process_name_dcp_rank_for_tp_rank_3():
    """rank 3 of a (tp=4, dcp=2) mesh has dcp_rank = 3 % 2 = 1."""
    m = MeshConfig(tp=4, dcp=2)
    name = process_name_for_rank(3, m)
    assert "DCP1" in name
    assert "TP3" in name


def test_process_name_omits_dp_when_dp_1():
    m = MeshConfig(tp=2, pp=2)
    name = process_name_for_rank(0, m)
    assert "DP" not in name


def test_process_name_includes_dp_when_dp_gt_1():
    m = MeshConfig(tp=2, dp=2)
    name = process_name_for_rank(0, m)
    assert "DP0" in name


def test_process_name_includes_pp_when_pp_gt_1():
    m = MeshConfig(tp=2, pp=2)
    name = process_name_for_rank(2, m)
    assert "PP1" in name


def test_process_name_ep_optional_arg():
    m = MeshConfig(tp=2)
    name = process_name_for_rank(0, m, enable_ep=True, ep_rank=3)
    assert "EP3" in name


# --------------------------------------------------------------------------
# Frozen dataclass invariants
# --------------------------------------------------------------------------


def test_mesh_config_is_frozen():
    """MeshConfig is frozen — cannot mutate."""
    m = MeshConfig(tp=4)
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        m.tp = 8  # type: ignore[misc]
