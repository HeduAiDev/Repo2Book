"""Tests for dcp_alltoall.py — A2A backend, NCCL op count, payload sizes, alpha-beta.

Source: vllm/v1/attention/ops/dcp_alltoall.py + vllm/config/parallel.py:L322-L328
"""

from __future__ import annotations

import numpy as np
import pytest

from implementation.dcp_alltoall import (
    CommCost,
    a2a_op_count,
    a2a_payload_bytes,
    ag_rs_op_count,
    ag_rs_payload_bytes,
    alpha_beta_cost,
    simulate_a2a_combine,
    simulate_ag_rs_combine,
)
from implementation.lse_combine import (
    reference_attention,
    split_attention,
)


# --------------------------------------------------------------------------
# NCCL op counts: AG+RS = 3, A2A = 2 (per arxiv 2507.07120)
# REFERENCE: dcp_alltoall.py:L1-L20, parallel.py:L322-L328
# --------------------------------------------------------------------------


def test_ag_rs_op_count_is_3():
    assert ag_rs_op_count() == 3


def test_a2a_op_count_is_2():
    assert a2a_op_count() == 2


def test_a2a_reduces_ops_by_33_percent():
    """Demo §3 verbatim: 33% reduction (3 -> 2 ops)."""
    pct = (1.0 - a2a_op_count() / ag_rs_op_count()) * 100
    assert round(pct) == 33


def test_a2a_strictly_fewer_ops_than_ag_rs():
    assert a2a_op_count() < ag_rs_op_count()


# --------------------------------------------------------------------------
# Payload bytes: AG+RS replicates Q; A2A packs (1/dcp) of output + LSE
# --------------------------------------------------------------------------


def test_ag_rs_payload_bytes_independent_of_dcp():
    """AG+RS payload = num_tokens * heads * head_dim * dtype — independent of dcp."""
    p1 = ag_rs_payload_bytes(32 * 1024, 8, 128, dcp_size=2)
    p2 = ag_rs_payload_bytes(32 * 1024, 8, 128, dcp_size=4)
    p3 = ag_rs_payload_bytes(32 * 1024, 8, 128, dcp_size=8)
    assert p1 == p2 == p3


def test_ag_rs_payload_demo_value():
    """Demo §3: bytes = 32K * 8 * 128 * 2 = 67,108,864."""
    assert ag_rs_payload_bytes(32 * 1024, 8, 128, dcp_size=2) == 67_108_864


def test_a2a_payload_bytes_scales_inversely_with_dcp():
    """A2A payload = (1/dcp) of full output + LSE per rank."""
    p2 = a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=2)
    p4 = a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=4)
    p8 = a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=8)
    # roughly p4 ≈ p2 / 2 and p8 ≈ p2 / 4
    assert p4 < p2
    assert p8 < p4


def test_a2a_payload_demo_dcp_2():
    """Demo §3: A2A bytes at dcp=2 = 34,078,720."""
    assert a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=2) == 34_078_720


def test_a2a_payload_demo_dcp_4():
    """Demo §3: A2A bytes at dcp=4 = 17,039,360."""
    assert a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=4) == 17_039_360


def test_a2a_payload_demo_dcp_8():
    """Demo §3: A2A bytes at dcp=8 = 8,519,680."""
    assert a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=8) == 8_519_680


def test_a2a_payload_lse_pack_dim_2_at_bf16():
    """REFERENCE: dcp_alltoall.py:L106-L112 lse_pack_dim=2 for bf16."""
    # bf16 (dtype_bytes=2): payload = num_tokens * (heads/dcp) * (head_dim + 2) * 2
    expected = 32 * 1024 * (8 // 2) * (128 + 2) * 2
    assert a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=2, dtype_bytes=2) == expected


def test_a2a_payload_lse_pack_dim_1_at_fp32():
    """fp32 (dtype_bytes=4): lse_pack_dim=1."""
    expected = 32 * 1024 * (8 // 2) * (128 + 1) * 4
    assert a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=2, dtype_bytes=4) == expected


# --------------------------------------------------------------------------
# Alpha-beta bandwidth model
# --------------------------------------------------------------------------


def test_alpha_beta_zero_payload_equals_alpha_x_n():
    """T = N * (alpha + 0) = N * alpha when bytes=0."""
    cost = alpha_beta_cost(0, alpha_us=10.0, beta_gbps=200.0, num_collectives=3)
    assert cost == pytest.approx(30.0)


def test_alpha_beta_zero_alpha_equals_n_x_bytes_per_beta():
    """T = N * (0 + bytes / beta_bytes_per_us)."""
    bytes_payload = 200_000_000  # 200 MB
    beta_gbps = 200.0
    beta_bytes_per_us = beta_gbps * 1e3
    expected = 1 * (bytes_payload / beta_bytes_per_us)
    cost = alpha_beta_cost(bytes_payload, alpha_us=0.0, beta_gbps=beta_gbps, num_collectives=1)
    assert cost == pytest.approx(expected)


def test_alpha_beta_grows_linearly_with_collectives():
    base = alpha_beta_cost(67_108_864, 10.0, 200.0, num_collectives=1)
    triple = alpha_beta_cost(67_108_864, 10.0, 200.0, num_collectives=3)
    assert triple == pytest.approx(3 * base)


def test_alpha_beta_demo_ag_rs_at_dcp_2():
    """Demo §3: T_AG+RS at dcp=2 = 1036.6 us."""
    bytes_ag_rs = ag_rs_payload_bytes(32 * 1024, 8, 128, dcp_size=2)
    t = alpha_beta_cost(bytes_ag_rs, 10.0, 200.0, num_collectives=ag_rs_op_count())
    assert t == pytest.approx(1036.6, abs=0.5)


def test_alpha_beta_demo_a2a_at_dcp_2():
    """Demo §3: T_A2A at dcp=2 = 360.8 us."""
    bytes_a2a = a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=2)
    t = alpha_beta_cost(bytes_a2a, 10.0, 200.0, num_collectives=a2a_op_count())
    assert t == pytest.approx(360.8, abs=0.5)


def test_alpha_beta_demo_a2a_at_dcp_4():
    """Demo §3: T_A2A at dcp=4 = 190.4 us."""
    bytes_a2a = a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=4)
    t = alpha_beta_cost(bytes_a2a, 10.0, 200.0, num_collectives=a2a_op_count())
    assert t == pytest.approx(190.4, abs=0.5)


def test_alpha_beta_demo_a2a_at_dcp_8():
    """Demo §3: T_A2A at dcp=8 = 105.2 us."""
    bytes_a2a = a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=8)
    t = alpha_beta_cost(bytes_a2a, 10.0, 200.0, num_collectives=a2a_op_count())
    assert t == pytest.approx(105.2, abs=0.5)


def test_alpha_beta_demo_speedup_dcp_2_is_2_87x():
    """Demo §3: speedup at dcp=2 = 2.87x."""
    ag = ag_rs_payload_bytes(32 * 1024, 8, 128, dcp_size=2)
    a2 = a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=2)
    t_ag = alpha_beta_cost(ag, 10.0, 200.0, num_collectives=ag_rs_op_count())
    t_a2 = alpha_beta_cost(a2, 10.0, 200.0, num_collectives=a2a_op_count())
    assert (t_ag / t_a2) == pytest.approx(2.87, abs=0.02)


def test_alpha_beta_demo_speedup_dcp_4_is_5_44x():
    """Demo §3: speedup at dcp=4 = 5.44x."""
    ag = ag_rs_payload_bytes(32 * 1024, 8, 128, dcp_size=4)
    a2 = a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=4)
    t_ag = alpha_beta_cost(ag, 10.0, 200.0, num_collectives=ag_rs_op_count())
    t_a2 = alpha_beta_cost(a2, 10.0, 200.0, num_collectives=a2a_op_count())
    assert (t_ag / t_a2) == pytest.approx(5.44, abs=0.05)


def test_alpha_beta_demo_speedup_dcp_8_is_9_85x():
    """Demo §3: speedup at dcp=8 = 9.85x."""
    ag = ag_rs_payload_bytes(32 * 1024, 8, 128, dcp_size=8)
    a2 = a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=8)
    t_ag = alpha_beta_cost(ag, 10.0, 200.0, num_collectives=ag_rs_op_count())
    t_a2 = alpha_beta_cost(a2, 10.0, 200.0, num_collectives=a2a_op_count())
    assert (t_ag / t_a2) == pytest.approx(9.85, abs=0.05)


def test_alpha_beta_speedup_grows_with_dcp_size():
    """As dcp grows, A2A's payload shrinks faster than AG+RS's stays constant → speedup grows."""
    speedups = []
    for dcp in (2, 4, 8, 16):
        ag = ag_rs_payload_bytes(32 * 1024, 8, 128, dcp_size=dcp)
        a2 = a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=dcp)
        t_ag = alpha_beta_cost(ag, 10.0, 200.0, num_collectives=ag_rs_op_count())
        t_a2 = alpha_beta_cost(a2, 10.0, 200.0, num_collectives=a2a_op_count())
        speedups.append(t_ag / t_a2)
    # Monotone non-decreasing.
    assert all(speedups[i] < speedups[i + 1] for i in range(len(speedups) - 1))


# --------------------------------------------------------------------------
# Both backends produce IDENTICAL combined output (LSE algebra is the same)
# REFERENCE: dcp_alltoall.py:L1-L20 docstring
# --------------------------------------------------------------------------


def _qkv():
    rng = np.random.default_rng(seed=42)
    q = rng.standard_normal((4, 2, 8)).astype(np.float64)
    k = rng.standard_normal((16, 2, 8)).astype(np.float64)
    v = rng.standard_normal((16, 2, 8)).astype(np.float64)
    return q, k, v


def test_a2a_and_ag_rs_combine_produce_identical_output():
    """Trap F: same LSE math regardless of transport."""
    q, k, v = _qkv()
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=4)
    out_a2a = simulate_a2a_combine(parts_o, parts_lse)
    out_ag_rs = simulate_ag_rs_combine(parts_o, parts_lse)
    assert np.array_equal(out_a2a, out_ag_rs)


def test_a2a_combine_matches_single_process_attention():
    """A2A combine bit-equivalent to single-process attention."""
    q, k, v = _qkv()
    o_truth, _ = reference_attention(q, k, v)
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=4)
    out = simulate_a2a_combine(parts_o, parts_lse)
    assert np.max(np.abs(out - o_truth)) < 1e-12


def test_ag_rs_combine_matches_single_process_attention():
    """AG+RS combine bit-equivalent to single-process attention."""
    q, k, v = _qkv()
    o_truth, _ = reference_attention(q, k, v)
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=4)
    out = simulate_ag_rs_combine(parts_o, parts_lse)
    assert np.max(np.abs(out - o_truth)) < 1e-12


@pytest.mark.parametrize("num_ranks", [1, 2, 4, 8])
def test_a2a_round_trip_at_various_rank_counts(num_ranks):
    """A2A round-trip preserves attention output at any divisible num_ranks."""
    rng = np.random.default_rng(seed=num_ranks)
    L = 32
    q = rng.standard_normal((4, 2, 8)).astype(np.float64)
    k = rng.standard_normal((L, 2, 8)).astype(np.float64)
    v = rng.standard_normal((L, 2, 8)).astype(np.float64)
    o_truth, _ = reference_attention(q, k, v)
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=num_ranks)
    out = simulate_a2a_combine(parts_o, parts_lse)
    assert np.max(np.abs(out - o_truth)) < 1e-11


# --------------------------------------------------------------------------
# CommCost dataclass
# --------------------------------------------------------------------------


def test_comm_cost_total_bytes():
    cc = CommCost(num_ops=3, bytes_per_op=1024)
    assert cc.total_bytes() == 3072


def test_comm_cost_zero_ops():
    cc = CommCost(num_ops=0, bytes_per_op=1024)
    assert cc.total_bytes() == 0
