"""Fidelity tests for all2all_baseline.py — AgRsAll2AllManager + alpha-beta cost model.

Verifies:

- Trap B (asymmetric all-to-all): vLLM's AgRs is ``all_gatherv + reduce_scatterv``,
  NOT a torch.distributed.all_to_all. Tests assert dispatch is concatenation
  (not permutation) and combine is reduce-then-split.
- Per-rank `sizes` correctness: dispatch handles variable-sized chunks.
- Round-trip: ``combine(dispatch(per_rank))`` recovers the per-rank shapes.
- α-β cost model: T_AR / T_A2A == 2 across payload sizes (the §3.2 ratio).
- §3.2 demo numerics pinned bit-for-bit (NVLink + IB tables).
"""

from __future__ import annotations

import math

import torch

from implementation.all2all_baseline import (
    AgRsAll2AllManager,
    all_gatherv,
    alpha_beta_cost,
    measure_dispatch_combine,
    reduce_scatterv,
)


# ---------------------------------------------------------------------------
# all_gatherv & reduce_scatterv primitives
# ---------------------------------------------------------------------------


def test_all_gatherv_concatenates():
    """all_gatherv is plain concatenation along dim 0 (in-process simulation)."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0]])
    c = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    out = all_gatherv([a, b, c], dim=0)
    assert out.shape == (6, 2)
    assert torch.equal(out, torch.cat([a, b, c], dim=0))


def test_all_gatherv_handles_unequal_sizes():
    """Per-rank chunk sizes can differ — that's the whole point of the ``_v`` variant."""
    chunks = [torch.zeros(s, 4) for s in (3, 1, 7, 2)]
    out = all_gatherv(chunks, dim=0)
    assert out.shape[0] == 3 + 1 + 7 + 2


def test_reduce_scatterv_splits_to_sizes():
    """reduce_scatterv splits the (already-summed) tensor by sizes."""
    full = torch.arange(20).float().view(10, 2)
    chunks = reduce_scatterv(full, sizes=[3, 5, 2])
    assert [c.shape[0] for c in chunks] == [3, 5, 2]
    assert torch.equal(chunks[0], full[:3])
    assert torch.equal(chunks[1], full[3:8])
    assert torch.equal(chunks[2], full[8:])


# ---------------------------------------------------------------------------
# AgRsAll2AllManager.dispatch — concatenation behaviour
# ---------------------------------------------------------------------------


def test_dispatch_concatenates_three_streams():
    """dispatch returns concatenated (hs, tw, ti) — mirror of all2all.py:L108-L121."""
    mgr = AgRsAll2AllManager(ep_size=2)
    hs0 = torch.ones(3, 4)
    hs1 = torch.zeros(2, 4)
    tw0 = torch.ones(3, 2)
    tw1 = torch.zeros(2, 2)
    ti0 = torch.zeros(3, 2, dtype=torch.int32)
    ti1 = torch.ones(2, 2, dtype=torch.int32)
    hs, tw, ti = mgr.dispatch([hs0, hs1], [tw0, tw1], [ti0, ti1])
    assert hs.shape == (5, 4)
    assert tw.shape == (5, 2)
    assert ti.shape == (5, 2)
    # Concatenation order = rank order
    assert torch.equal(hs[:3], hs0)
    assert torch.equal(hs[3:], hs1)


def test_trap_B_dispatch_is_NOT_permutation():
    """Trap B negative test: AgRs dispatch is concatenation, not symmetric all-to-all.
    A symmetric all-to-all would PERMUTE — every rank sends 1/P of its data to each
    other rank. AgRs simply CONCATENATES — every rank ends up with everyone's data.
    Test that the result is a (sorted-by-rank) concatenation, not a transpose."""
    mgr = AgRsAll2AllManager(ep_size=4)
    # Each rank contributes a unique tag.
    chunks = [torch.full((2, 1), float(r)) for r in range(4)]
    weights = [torch.zeros(2, 1) for _ in range(4)]
    ids = [torch.zeros(2, 1, dtype=torch.int32) for _ in range(4)]
    hs, _, _ = mgr.dispatch(chunks, weights, ids)
    # rank 0's data is at positions 0..1, rank 1's at 2..3, etc.
    for r in range(4):
        assert torch.allclose(hs[2 * r : 2 * (r + 1)], torch.full((2, 1), float(r)))


# ---------------------------------------------------------------------------
# AgRsAll2AllManager.combine — split by sizes
# ---------------------------------------------------------------------------


def test_combine_splits_by_sizes():
    """combine = reduce_scatterv: split the summed tensor into per-rank pieces."""
    mgr = AgRsAll2AllManager(ep_size=3)
    full = torch.arange(60).float().view(10, 6)
    chunks = mgr.combine(full, sizes=[3, 5, 2])
    assert [c.shape for c in chunks] == [(3, 6), (5, 6), (2, 6)]


def test_round_trip_dispatch_then_combine_preserves_sizes():
    """dispatch → combine round-trips per-rank sizes (the basic shape contract)."""
    mgr = AgRsAll2AllManager(ep_size=4)
    sizes = [3, 1, 5, 2]
    chunks = [torch.randn(s, 8) for s in sizes]
    weights = [torch.zeros(s, 1) for s in sizes]
    ids = [torch.zeros(s, 1, dtype=torch.int32) for s in sizes]

    full_hs, _, _ = mgr.dispatch(chunks, weights, ids)
    out = mgr.combine(full_hs, sizes)
    for orig, recovered in zip(chunks, out):
        assert orig.shape == recovered.shape
        assert torch.equal(orig, recovered)


# ---------------------------------------------------------------------------
# Alpha-beta cost model — formulas and §3.2 demo pin
# ---------------------------------------------------------------------------


def test_alpha_beta_world_size_one_returns_zero():
    """At P=1 collectives are no-ops — return 0 cost."""
    cost = alpha_beta_cost(payload_bytes=1024, alpha_us=5.0, beta_GBps=250.0, p=1, op="all_reduce")
    assert cost == 0.0


def test_alpha_beta_all_reduce_is_2x_all_to_all():
    """T_AR = 2 × T_A2A — the chapter's headline ratio."""
    payload = 1024 * 1024  # 1 MiB
    p = 8
    t_ar = alpha_beta_cost(payload, alpha_us=5.0, beta_GBps=250.0, p=p, op="all_reduce")
    t_a2a = alpha_beta_cost(payload, alpha_us=5.0, beta_GBps=250.0, p=p, op="all_to_all")
    assert math.isclose(t_ar / t_a2a, 2.0, rel_tol=1e-6)


def test_alpha_beta_all_gather_equals_all_to_all():
    """T_A2A and T_AG share the same one-hop ring formula."""
    payload = 4096
    p = 8
    t_a2a = alpha_beta_cost(payload, alpha_us=5.0, beta_GBps=250.0, p=p, op="all_to_all")
    t_ag = alpha_beta_cost(payload, alpha_us=5.0, beta_GBps=250.0, p=p, op="all_gather")
    assert math.isclose(t_a2a, t_ag, rel_tol=1e-9)


def test_alpha_beta_unknown_op_raises():
    """Unknown op kind raises."""
    try:
        alpha_beta_cost(1024, 5.0, 250.0, 8, op="not_real")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_section_32_nvlink_pin_128tokens():
    """Pin §3.2 NVLink row at 128 tokens: T_AR=16.09μs, T_A2A=8.05μs, ratio=2.000."""
    bytes_per_token = 4096 * 2  # bf16
    nbytes = 128 * bytes_per_token
    t_ar = alpha_beta_cost(nbytes, alpha_us=5.0, beta_GBps=250.0, p=8, op="all_reduce")
    t_a2a = alpha_beta_cost(nbytes, alpha_us=5.0, beta_GBps=250.0, p=8, op="all_to_all")
    assert math.isclose(t_ar, 16.09, abs_tol=0.01)
    assert math.isclose(t_a2a, 8.05, abs_tol=0.01)
    assert math.isclose(t_ar / t_a2a, 2.000, abs_tol=1e-3)


def test_section_32_nvlink_pin_1024tokens():
    """Pin §3.2 NVLink row at 1024 tokens: T_AR=67.47μs, T_A2A=33.74μs."""
    nbytes = 1024 * 4096 * 2
    t_ar = alpha_beta_cost(nbytes, alpha_us=5.0, beta_GBps=250.0, p=8, op="all_reduce")
    t_a2a = alpha_beta_cost(nbytes, alpha_us=5.0, beta_GBps=250.0, p=8, op="all_to_all")
    assert math.isclose(t_ar, 67.47, abs_tol=0.05)
    assert math.isclose(t_a2a, 33.74, abs_tol=0.05)


def test_section_32_nvlink_pin_8192tokens():
    """Pin §3.2 NVLink row at 8192 tokens: T_AR=478.51μs, T_A2A=239.26μs."""
    nbytes = 8192 * 4096 * 2
    t_ar = alpha_beta_cost(nbytes, alpha_us=5.0, beta_GBps=250.0, p=8, op="all_reduce")
    t_a2a = alpha_beta_cost(nbytes, alpha_us=5.0, beta_GBps=250.0, p=8, op="all_to_all")
    assert math.isclose(t_ar, 478.51, abs_tol=0.5)
    assert math.isclose(t_a2a, 239.26, abs_tol=0.5)


def test_section_32_nvlink_pin_65536tokens():
    """Pin §3.2 NVLink row at 65536 tokens: T_AR=3766.85μs, T_A2A=1883.42μs."""
    nbytes = 65536 * 4096 * 2
    t_ar = alpha_beta_cost(nbytes, alpha_us=5.0, beta_GBps=250.0, p=8, op="all_reduce")
    t_a2a = alpha_beta_cost(nbytes, alpha_us=5.0, beta_GBps=250.0, p=8, op="all_to_all")
    assert math.isclose(t_ar, 3766.85, abs_tol=2.0)
    assert math.isclose(t_a2a, 1883.42, abs_tol=2.0)


def test_section_32_ib_pin_65536tokens():
    """Pin §3.2 IB headline (low end and high end): IB α=8μs, β=50 GB/s, P=8."""
    nbytes = 65536 * 4096 * 2
    t_ar = alpha_beta_cost(nbytes, alpha_us=8.0, beta_GBps=50.0, p=8, op="all_reduce")
    t_a2a = alpha_beta_cost(nbytes, alpha_us=8.0, beta_GBps=50.0, p=8, op="all_to_all")
    assert math.isclose(t_ar, 18804.48, abs_tol=10.0)
    assert math.isclose(t_a2a, 9402.24, abs_tol=10.0)


def test_section_32_ib_pin_128tokens():
    """Pin §3.2 IB headline at 128 tokens: T_AR=50.70μs, T_A2A=25.35μs."""
    nbytes = 128 * 4096 * 2
    t_ar = alpha_beta_cost(nbytes, alpha_us=8.0, beta_GBps=50.0, p=8, op="all_reduce")
    t_a2a = alpha_beta_cost(nbytes, alpha_us=8.0, beta_GBps=50.0, p=8, op="all_to_all")
    assert math.isclose(t_ar, 50.70, abs_tol=0.05)
    assert math.isclose(t_a2a, 25.35, abs_tol=0.05)


def test_alpha_beta_grows_with_payload():
    """Cost is monotone increasing in payload (β-bound regime)."""
    p = 8
    a = alpha_beta_cost(1024, 5.0, 250.0, p, "all_reduce")
    b = alpha_beta_cost(1024 * 1024, 5.0, 250.0, p, "all_reduce")
    c = alpha_beta_cost(1024 * 1024 * 1024, 5.0, 250.0, p, "all_reduce")
    assert a < b < c


def test_alpha_beta_alpha_dominates_at_small_payload():
    """At very small payload, ring (P-1)/P × α dominates; T tends to 2*(P-1)/P*α for AR."""
    p = 8
    nbytes = 1  # essentially zero payload
    t_ar = alpha_beta_cost(nbytes, alpha_us=5.0, beta_GBps=250.0, p=p, op="all_reduce")
    asymptote = 2.0 * (p - 1) / p * 5.0  # 8.75 μs
    assert math.isclose(t_ar, asymptote, abs_tol=1e-3)


# ---------------------------------------------------------------------------
# measure_dispatch_combine helper
# ---------------------------------------------------------------------------


def test_measure_dispatch_combine_returns_shape_dict():
    """measure_dispatch_combine returns a dict of shapes (no wallclock — honest demo)."""
    sizes = [4, 2, 6]
    chunks = [torch.zeros(s, 8) for s in sizes]
    weights = [torch.zeros(s, 2) for s in sizes]
    ids = [torch.zeros(s, 2, dtype=torch.int32) for s in sizes]
    info = measure_dispatch_combine(chunks, weights, ids, sizes)
    assert info["ep_size"] == 3
    assert info["hidden_after_dispatch_shape"] == (12, 8)
    assert info["per_rank_input_sizes"] == sizes
    assert len(info["per_rank_output_shapes"]) == 3
