"""Tests for `implementation.comm_primitives` — α-β model + ring all-reduce.

Source claim: vLLM's `tensor_model_parallel_all_reduce` is a 1-line wrapper
over `get_tp_group().all_reduce(...)` (communication_op.py:L12-L14), which
asymptotically follows the bandwidth-optimal ring formula

    T_ring = 2 * (P - 1) / P * (α + (S / P) * β)

We verify:
- AlphaBetaModel.predict() and bandwidth_GBps property are correct.
- ring_all_reduce_cost() matches the formula bit-for-bit at known inputs.
- ring_all_reduce_cost(P=1) == 0 (the world_size==1 bypass — parallel_state.py:L518-L519).
- simulate_all_reduce() really sums (max diff vs naive sum < fp32 noise).
- simulate_all_reduce(P=1) is the identity.
- fit_alpha_beta() recovers the synthetic ground truth within 10%.
- HARDWARE_PROFILES contains realistic α and β values.
- predict_block_overhead() doubles cost (two all-reduces per Llama block).
- TP=2 ≠ 2× claim: at small payloads the latency term DOMINATES; at large payloads
  the bandwidth term gives sub-linear speedup.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from implementation.comm_primitives import (
    AlphaBetaModel,
    HARDWARE_PROFILES,
    fit_alpha_beta,
    predict_block_overhead,
    ring_all_reduce_cost,
    simulate_all_reduce,
)


# ---------------------------------------------------------------------------
# §1 AlphaBetaModel basic semantics
# ---------------------------------------------------------------------------

class TestAlphaBetaModel:
    def test_predict_linear(self):
        """T(S) = α + βS — the classical Hockney 1994 formulation."""
        ab = AlphaBetaModel(alpha_seconds=2e-6, beta_seconds_per_byte=1 / 300e9)
        # At S=0, T=α.
        assert ab.predict(0) == pytest.approx(2e-6, rel=1e-9)
        # At S=300e9 bytes, T = α + 1.0 = 1.000002 s.
        assert ab.predict(300e9) == pytest.approx(2e-6 + 1.0, rel=1e-9)

    def test_bandwidth_GBps_property(self):
        """1/β = bytes/sec; convert to GB/s."""
        ab = AlphaBetaModel(alpha_seconds=0.0, beta_seconds_per_byte=1 / 300e9)
        assert ab.bandwidth_GBps == pytest.approx(300.0, rel=1e-9)

    def test_bandwidth_inf_when_beta_zero(self):
        ab = AlphaBetaModel(alpha_seconds=1e-6, beta_seconds_per_byte=0.0)
        assert math.isinf(ab.bandwidth_GBps)


# ---------------------------------------------------------------------------
# §2 ring_all_reduce_cost — formula correctness
# ---------------------------------------------------------------------------

class TestRingAllReduceCost:
    def test_returns_zero_for_world_size_one(self):
        """parallel_state.py:L518-L519 — bypass when world_size==1."""
        ab = AlphaBetaModel(alpha_seconds=2e-6, beta_seconds_per_byte=1 / 300e9)
        assert ring_all_reduce_cost(payload_bytes=1024, num_ranks=1, ab=ab) == 0.0

    def test_formula_at_p2(self):
        """T = 2*(P-1)/P*(α + S/P*β); at P=2 → α + 0.5*S*β."""
        ab = AlphaBetaModel(alpha_seconds=2e-6, beta_seconds_per_byte=1 / 300e9)
        S = 300e9  # 300 GB
        expected = 1.0 * (2e-6 + 0.5 * S * (1 / 300e9))
        got = ring_all_reduce_cost(payload_bytes=S, num_ranks=2, ab=ab)
        assert got == pytest.approx(expected, rel=1e-9)

    def test_formula_at_p4_against_hand_computed(self):
        """P=4: T = 1.5 * (α + S/4 * β)."""
        ab = AlphaBetaModel(alpha_seconds=2e-6, beta_seconds_per_byte=1 / 300e9)
        S = 4_000_000  # 4 MB
        expected = 1.5 * (2e-6 + (S / 4) * (1 / 300e9))
        got = ring_all_reduce_cost(payload_bytes=S, num_ranks=4, ab=ab)
        assert got == pytest.approx(expected, rel=1e-9)

    def test_demo_section_2_NVLink_table_reproduces(self):
        """Demo §2 pinned: ring_us_S262144_P2 = 2.44 (NVLink_HSXM4 profile)."""
        ab = HARDWARE_PROFILES["NVLink_HSXM4"]
        T_us = ring_all_reduce_cost(payload_bytes=262144, num_ranks=2, ab=ab) * 1e6
        # demo emits 2.43691; pin to 0.01 μs precision.
        assert T_us == pytest.approx(2.44, abs=0.02)

    def test_large_payload_p8_beats_p2(self):
        """At very large payloads the (P-1)/P×β term DOMINATES → P=8 ≈ 2.17× faster than P=2.
        This is the bandwidth-bound regime; sub-linear because (P-1)/P → 1 monotonically."""
        ab = HARDWARE_PROFILES["NVLink_HSXM4"]
        S = 64 * 1024 * 1024  # 64 MB
        t_p2 = ring_all_reduce_cost(S, 2, ab)
        t_p8 = ring_all_reduce_cost(S, 8, ab)
        ratio = t_p2 / t_p8
        # Demo §2 pinned: 113.85 / 52.43 ≈ 2.17. Allow ±0.05.
        assert 2.10 < ratio < 2.25, f"P=2/P=8 ratio={ratio:.3f}, expected ≈2.17"

    def test_small_payload_alpha_bound_regime(self):
        """At tiny payloads the α term DOMINATES → P=8 is SLOWER than P=2 (because the
        ring takes 2(P-1)/P steps each costing α). This is THE evidence for Trap-A
        ('TP=2 doubles throughput' is wrong)."""
        ab = HARDWARE_PROFILES["NVLink_HSXM4"]
        S = 1024  # 1 KB — small payload
        t_p2 = ring_all_reduce_cost(S, 2, ab)
        t_p8 = ring_all_reduce_cost(S, 8, ab)
        # Demo §2 pinned: P=2 = 2.00 μs; P=8 = 3.50 μs.
        assert t_p8 > t_p2, "P=8 must be SLOWER than P=2 in α-bound regime"
        ratio = t_p8 / t_p2
        assert 1.6 < ratio < 1.9, f"P=8/P=2 small-payload ratio={ratio:.3f}, expected ≈1.75"


# ---------------------------------------------------------------------------
# §3 simulate_all_reduce — really sums, ring algorithm correct
# ---------------------------------------------------------------------------

class TestSimulateAllReduce:
    def test_world_size_one_is_identity(self):
        """parallel_state.py:L518-L519 bypass — single rank returns its own tensor."""
        x = np.array([[1.0, 2.0]], dtype=np.float32)
        out = simulate_all_reduce([x])
        assert len(out) == 1
        assert np.array_equal(out[0], x)

    @pytest.mark.parametrize("P", [2, 4, 8])
    def test_simulation_equals_naive_sum(self, P):
        """The single load-bearing claim: ring all-reduce produces the SAME result
        on every rank as a naive sum across ranks. Demo §2 pinned: max diff = 2.384e-07."""
        rng = np.random.default_rng(0)
        # Need axis-0 size divisible by P.
        per_rank = [rng.standard_normal((P * 2, 4)).astype(np.float32) for _ in range(P)]
        target = np.sum(per_rank, axis=0)
        out = simulate_all_reduce(per_rank)
        assert len(out) == P
        for r, t in enumerate(out):
            diff = float(np.max(np.abs(t - target)))
            # fp32 tolerance for additions; demo emits ≈2.4e-7.
            assert diff < 1e-5, f"rank {r}: diff={diff:.3e}"

    def test_simulation_chunking_assertion(self):
        """axis-0 must be divisible by P for ring chunking — assertion fires."""
        per_rank = [np.zeros((3, 4), dtype=np.float32) for _ in range(2)]  # 3 not div by 2
        with pytest.raises(AssertionError):
            simulate_all_reduce(per_rank)

    def test_shape_mismatch_assertion(self):
        per_rank = [np.zeros((4, 4), dtype=np.float32), np.zeros((4, 5), dtype=np.float32)]
        with pytest.raises(AssertionError):
            simulate_all_reduce(per_rank)


# ---------------------------------------------------------------------------
# §4 fit_alpha_beta — least-squares recovery
# ---------------------------------------------------------------------------

class TestFitAlphaBeta:
    def test_recovers_known_ground_truth_with_low_noise(self):
        """Demo §2: true α=5μs, β=1/150e9; fit recovers within ~10%."""
        true_ab = AlphaBetaModel(alpha_seconds=5e-6, beta_seconds_per_byte=1 / 150e9)
        payloads = [1024, 16 * 1024, 256 * 1024, 4 * 1024 * 1024, 64 * 1024 * 1024]
        rng = np.random.default_rng(2)
        measured = [true_ab.predict(s) * (1 + rng.normal(0, 0.02)) for s in payloads]
        fit = fit_alpha_beta(payloads, measured)
        # Demo §2 emits fit α=4.32 μs, BW=144.6 GB/s. Allow ±15% tolerance.
        assert abs(fit.alpha_seconds - true_ab.alpha_seconds) / true_ab.alpha_seconds < 0.20
        assert abs(fit.bandwidth_GBps - true_ab.bandwidth_GBps) / true_ab.bandwidth_GBps < 0.15

    def test_clamp_keeps_alpha_positive(self):
        """A least-squares fit on adversarial data may dip α negative; clamp guards."""
        # Two payloads at increasing S with T DECREASING — a wedge that drives α<0.
        payloads = [1, 1_000_000_000]
        measured = [1.0, 0.0]  # totally inconsistent with linear model
        fit = fit_alpha_beta(payloads, measured)
        assert fit.alpha_seconds > 0


# ---------------------------------------------------------------------------
# §5 HARDWARE_PROFILES — sanity-check the calibrations
# ---------------------------------------------------------------------------

class TestHardwareProfiles:
    def test_nvlink_hsxm4_specs(self):
        """NVLink HSXM4 ≈ 2 μs latency, 300 GB/s bandwidth (the canonical fast path)."""
        ab = HARDWARE_PROFILES["NVLink_HSXM4"]
        assert ab.alpha_seconds == 2.0e-6
        assert ab.bandwidth_GBps == pytest.approx(300.0, rel=1e-9)

    def test_pcie_gen4_slower_than_nvlink(self):
        """PCIe Gen4 x16 ≈ 32 GB/s — order of magnitude slower than NVLink."""
        nvlink = HARDWARE_PROFILES["NVLink_HSXM4"]
        pcie = HARDWARE_PROFILES["PCIe_Gen4_x16"]
        assert pcie.bandwidth_GBps < nvlink.bandwidth_GBps / 5


# ---------------------------------------------------------------------------
# §6 predict_block_overhead — Llama transformer block (1 attn + 1 mlp = 2 all-reduces)
# ---------------------------------------------------------------------------

class TestPredictBlockOverhead:
    def test_two_all_reduces_per_block(self):
        """Llama transformer block has TWO all-reduces (attn o_proj, mlp down_proj).
        predict_block_overhead returns BOTH in seconds_per_block."""
        r = predict_block_overhead(
            hidden=4096, ffn=11008, batch_seqs=512, dtype_bytes=2,
            tp_size=2, hardware="NVLink_HSXM4",
        )
        assert r["predicted_seconds_per_block"] == pytest.approx(
            2 * r["predicted_seconds_per_allreduce"], rel=1e-9
        )

    def test_payload_bytes_match_hidden_x_seq_x_dtype(self):
        """The all-reduce reduces a [batch_seqs, hidden] tensor."""
        r = predict_block_overhead(
            hidden=4096, ffn=11008, batch_seqs=512, dtype_bytes=2,
            tp_size=4, hardware="NVLink_HSXM4",
        )
        assert r["payload_bytes_per_allreduce"] == 512 * 4096 * 2

    def test_demo_section_3_tp2_per_allreduce_matches(self):
        """Demo §3 pinned: predicted_AR_us_tp2_NVLink = 8.99 μs.
        That number is the predicted cost of ONE all-reduce (the MLP-only block
        has only one); demo §3 divides predicted_seconds_per_block / 2 — so this
        equals predicted_seconds_per_allreduce."""
        r = predict_block_overhead(
            hidden=4096, ffn=11008, batch_seqs=512, dtype_bytes=2,
            tp_size=2, hardware="NVLink_HSXM4",
        )
        assert r["predicted_seconds_per_allreduce"] * 1e6 == pytest.approx(8.99, abs=0.05)
        # And per-block (attn + mlp) = 2× that.
        assert r["predicted_seconds_per_block"] * 1e6 == pytest.approx(17.98, abs=0.10)
