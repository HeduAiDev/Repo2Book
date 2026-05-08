"""Tests for dcp_vs_pcp_demo.py — Trap D anchor (DCP and PCP are SEPARABLE axes).

Source: vllm/distributed/parallel_state.py:L1593-L1633 + vllm/v1/attention/backend.py:L751
"""

from __future__ import annotations

import pytest

from implementation.dcp_vs_pcp_demo import (
    CPRoles,
    explain_axis_difference,
    explain_separability,
    per_rank_kv_chunk,
    world_size_for,
)
from implementation.world_topology import MeshConfig


# --------------------------------------------------------------------------
# Trap D anchor: DCP and PCP do NOT need to match
# --------------------------------------------------------------------------


def test_trap_d_dcp_pcp_match_required_is_false():
    """Trap D: 'DCP and PCP must match' is WRONG."""
    assert CPRoles.both_match_required() is False


def test_trap_d_unequal_dcp_pcp_is_valid():
    """Production config: tp=8, dcp=2, pcp=4 — separable."""
    m = MeshConfig(tp=8, dcp=2, pcp=4)
    assert m.dcp != m.pcp
    assert m.world_size == 8 * 4  # tp * pcp; dcp NOT included


def test_trap_d_dcp_only_no_pcp():
    """tp=8, dcp=4, pcp=1 is valid."""
    m = MeshConfig(tp=8, dcp=4, pcp=1)
    assert m.world_size == 8


def test_trap_d_pcp_only_no_dcp():
    """tp=4, dcp=1, pcp=4 is valid."""
    m = MeshConfig(tp=4, dcp=1, pcp=4)
    assert m.world_size == 4 * 4
    assert m.total_cp_world_size == 4 * 1  # pcp * dcp


# --------------------------------------------------------------------------
# Only constraint: tp % dcp == 0 (REFERENCE: parallel.py:L474-L478)
# --------------------------------------------------------------------------


def test_only_constraint_is_tp_divisible_by_dcp():
    """No constraint between dcp and pcp. Only tp % dcp == 0."""
    valid_configs = [
        (8, 2, 4),  # tp=8, dcp=2, pcp=4 — Trap D anchor
        (8, 4, 2),
        (8, 8, 8),
        (4, 4, 4),
        (4, 2, 8),
        (2, 1, 16),
        (1, 1, 1),
    ]
    for tp, dcp, pcp in valid_configs:
        m = MeshConfig(tp=tp, dcp=dcp, pcp=pcp)  # no exception
        assert m.tp % m.dcp == 0


def test_invalid_configs_are_only_those_with_tp_indivisible_by_dcp():
    """Invalid: tp=8, dcp=3 (8 % 3 != 0). Valid regardless of pcp."""
    with pytest.raises(ValueError, match="must be divisible"):
        MeshConfig(tp=8, dcp=3, pcp=4)


# --------------------------------------------------------------------------
# world_size product NEVER includes DCP
# REFERENCE: vllm/v1/executor/multiproc_executor.py:L116-L121
# --------------------------------------------------------------------------


def test_world_size_for_is_independent_of_dcp():
    base = MeshConfig(tp=8, pcp=2, dcp=1).world_size
    for dcp in (1, 2, 4, 8):
        m = MeshConfig(tp=8, pcp=2, dcp=dcp)
        assert world_size_for(m) == base


def test_world_size_for_grows_with_pcp_at_fixed_dcp():
    base = MeshConfig(tp=4, dcp=2, pcp=1).world_size
    for pcp in (1, 2, 4, 8):
        m = MeshConfig(tp=4, dcp=2, pcp=pcp)
        assert world_size_for(m) == base * pcp


# --------------------------------------------------------------------------
# total_cp_world_size = pcp * dcp (REFERENCE: backend.py:L751)
# --------------------------------------------------------------------------


def test_total_cp_independent_of_axis_assignment():
    """Per-rank KV chunk depends on PRODUCT pcp*dcp, not which axis."""
    a = MeshConfig(tp=8, dcp=4, pcp=2).total_cp_world_size
    b = MeshConfig(tp=8, dcp=2, pcp=4).total_cp_world_size
    assert a == b == 8


def test_per_rank_kv_chunk_demo_5():
    """Demo §5: seq=128, mesh=(tp=4, pcp=2, dcp=2) → chunk=128/(2*2)=32."""
    m = MeshConfig(tp=4, pcp=2, dcp=2)
    assert per_rank_kv_chunk(128, m) == 32


def test_per_rank_kv_chunk_at_total_cp_1():
    """No CP → chunk = full seq."""
    m = MeshConfig()
    assert per_rank_kv_chunk(1024, m) == 1024


@pytest.mark.parametrize("dcp,pcp,expected_chunk", [
    (1, 1, 1024),
    (2, 1, 512),
    (1, 2, 512),
    (2, 2, 256),
    (4, 4, 64),
    (8, 4, 32),
])
def test_per_rank_kv_chunk_grid(dcp, pcp, expected_chunk):
    m = MeshConfig(tp=max(8, dcp), dcp=dcp, pcp=pcp)
    assert per_rank_kv_chunk(1024, m) == expected_chunk


# --------------------------------------------------------------------------
# DCP role description vs PCP role description
# --------------------------------------------------------------------------


def test_dcp_role_mentions_kv_cache_decode_folded_inside_tp():
    roles = CPRoles()
    assert "KV" in roles.dcp_role.upper() or "kv" in roles.dcp_role
    assert "decode" in roles.dcp_role.lower()
    assert "TP" in roles.dcp_role or "tp" in roles.dcp_role


def test_pcp_role_mentions_prefill_input_independent():
    roles = CPRoles()
    assert "prefill" in roles.pcp_role.lower()
    assert "world_size" in roles.pcp_role.lower() or "independent" in roles.pcp_role.lower()


def test_explain_separability_mentions_only_constraint():
    text = explain_separability()
    assert "tp % dcp == 0" in text
    assert "SEPARABLE" in text


def test_explain_axis_difference_returns_dict_with_dcp_pcp_keys():
    d = explain_axis_difference()
    assert "DCP" in d
    assert "PCP" in d


def test_explain_axis_difference_contents():
    d = explain_axis_difference()
    assert "decode" in d["DCP"].lower()
    assert "prefill" in d["PCP"].lower()


# --------------------------------------------------------------------------
# Trap D production config: (tp=8, dcp=2, pcp=4) is valid
# --------------------------------------------------------------------------


def test_production_config_tp8_dcp2_pcp4_valid():
    """Documented production config from impl-notes Trap D."""
    m = MeshConfig(tp=8, dcp=2, pcp=4)
    assert m.world_size == 32  # tp * pcp = 8 * 4
    assert m.total_cp_world_size == 8  # dcp * pcp = 2 * 4
    assert m.num_dcp_subgroups == 4  # 8 / 2


def test_production_config_per_rank_kv_chunk():
    """At (tp=8, dcp=2, pcp=4) and seq=64K, per-rank chunk = 64K/8 = 8K."""
    m = MeshConfig(tp=8, dcp=2, pcp=4)
    assert per_rank_kv_chunk(64 * 1024, m) == 8 * 1024
