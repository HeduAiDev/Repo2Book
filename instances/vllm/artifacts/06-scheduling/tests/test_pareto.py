"""Unit tests for Pareto frontier sweep.

Demo §5 numerics (verify the table reproduces and the 16x improvement claim):
- 7 EngineConfig points; sweep produces 7 PerfPoints.
- Without long_prefill_token_threshold, p95 TTFT scales inversely with budget.
- With threshold=512 at budget=2048: leftover = 1536, short(64) fits in
  ceil(64/1536)=1 step → 50ms p95 (vs 200ms without threshold).
- 800ms (max_seqs=8, B=512) → 50ms (max_seqs=128, B=8192) = 16x p95 improvement.
"""

from __future__ import annotations

from implementation.pareto import (
    EngineConfig,
    PerfPoint,
    estimate_p95_ttft,
    estimate_saturation,
    estimate_throughput,
    pareto_front,
    sweep,
)


class TestEstimateP95TTFT:
    def test_no_threshold_long_drains_first(self) -> None:
        """Without threshold, long prompt soaks budget for ceil(L/B) steps."""
        cfg = EngineConfig(max_num_seqs=8, max_num_batched_tokens=512)
        # Long=8192, B=512 → 16 steps * 50ms = 800ms.
        ttft = estimate_p95_ttft(cfg, long_prompt_tokens=8192, short_prompt_tokens=64)
        assert abs(ttft - 0.800) < 1e-6

    def test_with_threshold_short_fits_in_leftover(self) -> None:
        """With threshold=512 and B=2048: leftover=1536, short(64) needs 1 step → 50ms."""
        cfg = EngineConfig(
            max_num_seqs=32, max_num_batched_tokens=2048,
            long_prefill_token_threshold=512,
        )
        ttft = estimate_p95_ttft(cfg, long_prompt_tokens=8192, short_prompt_tokens=64)
        assert abs(ttft - 0.050) < 1e-6

    def test_demo_pareto_endpoint_50ms(self) -> None:
        """Demo §5: max_seqs=128, B=8192 → 50ms (the Pareto-frontier point)."""
        cfg = EngineConfig(max_num_seqs=128, max_num_batched_tokens=8192)
        # Long=8192, B=8192 → 1 step → 50ms.
        ttft = estimate_p95_ttft(cfg, long_prompt_tokens=8192, short_prompt_tokens=64)
        assert abs(ttft - 0.050) < 1e-6

    def test_threshold_geq_budget_treated_as_disabled(self) -> None:
        """If threshold >= budget, the check disables (long can use full budget)."""
        cfg = EngineConfig(
            max_num_seqs=8, max_num_batched_tokens=512,
            long_prefill_token_threshold=1024,  # > 512 budget
        )
        # Falls into the disabled branch.
        ttft = estimate_p95_ttft(cfg, long_prompt_tokens=8192, short_prompt_tokens=64)
        # 8192 / 512 = 16 steps * 50ms = 800ms.
        assert abs(ttft - 0.800) < 1e-6


class TestEstimateThroughput:
    def test_returns_min_of_compute_and_memory_bound(self) -> None:
        cfg = EngineConfig(max_num_seqs=128, max_num_batched_tokens=8192)
        # compute = 50000 * min(128, 256) = 6_400_000.
        # memory = 8192 * 1000 = 8_192_000.
        # min = 6_400_000.
        tp = estimate_throughput(cfg, avg_seq_len=512)
        assert tp == 6_400_000

    def test_compute_bound_at_low_concurrency(self) -> None:
        cfg = EngineConfig(max_num_seqs=8, max_num_batched_tokens=8192)
        # compute = 50000 * 8 = 400_000. Memory = 8_192_000. compute wins.
        tp = estimate_throughput(cfg, avg_seq_len=512)
        assert tp == 400_000


class TestEstimateSaturation:
    def test_clamps_at_one(self) -> None:
        """Saturation never exceeds 1.0."""
        cfg = EngineConfig(max_num_seqs=8, max_num_batched_tokens=512)
        sat = estimate_saturation(cfg, avg_seq_len=512, arrival_rate_per_sec=1000.0)
        assert sat == 1.0

    def test_low_arrival_rate(self) -> None:
        """Low demand → low saturation."""
        cfg = EngineConfig(max_num_seqs=8, max_num_batched_tokens=8192)
        sat = estimate_saturation(cfg, avg_seq_len=512, arrival_rate_per_sec=1.0)
        # demand = 1 * 512 = 512 tok/s; supply = 8192 * 20 = 163_840 tok/s.
        # 512 / 163_840 ≈ 0.003.
        assert 0.0 < sat < 0.01


class TestSweep:
    def test_returns_one_point_per_input(self) -> None:
        knobs = [
            EngineConfig(max_num_seqs=8, max_num_batched_tokens=512),
            EngineConfig(max_num_seqs=16, max_num_batched_tokens=1024),
        ]
        points = sweep(knobs, long_prompt_tokens=8192, short_prompt_tokens=64)
        assert len(points) == 2
        assert all(isinstance(p, PerfPoint) for p in points)
        # PerfPoints retain their config.
        assert points[0].config is knobs[0]
        assert points[1].config is knobs[1]


class TestParetoFront:
    def test_demo_table_reproduces_one_dominant_point(self) -> None:
        """Demo §5 ends with 'Pareto frontier has 1 non-dominated points'."""
        knobs = [
            EngineConfig(max_num_seqs=8,   max_num_batched_tokens=512),
            EngineConfig(max_num_seqs=16,  max_num_batched_tokens=1024),
            EngineConfig(max_num_seqs=32,  max_num_batched_tokens=2048),
            EngineConfig(max_num_seqs=64,  max_num_batched_tokens=4096),
            EngineConfig(max_num_seqs=128, max_num_batched_tokens=8192),
            EngineConfig(max_num_seqs=32,  max_num_batched_tokens=2048,
                         long_prefill_token_threshold=512),
            EngineConfig(max_num_seqs=64,  max_num_batched_tokens=4096,
                         long_prefill_token_threshold=1024),
        ]
        points = sweep(knobs, long_prompt_tokens=8192, short_prompt_tokens=64)
        front = pareto_front(points)
        assert len(front) == 1
        # The lone Pareto point is the largest (max_seqs=128, B=8192) row.
        winner = front[0]
        assert winner.config.max_num_seqs == 128
        assert winner.config.max_num_batched_tokens == 8192

    def test_pareto_extracts_non_dominated(self) -> None:
        """Two configs A=(tput=10, ttft=1) and B=(tput=5, ttft=2). A dominates B
        on both axes → only A is on the frontier."""
        a_cfg = EngineConfig(max_num_seqs=128, max_num_batched_tokens=8192)
        b_cfg = EngineConfig(max_num_seqs=8, max_num_batched_tokens=512)
        a_pt = PerfPoint(config=a_cfg, throughput_tokens_per_sec=10, p95_ttft_seconds=1.0, saturation=0.5)
        b_pt = PerfPoint(config=b_cfg, throughput_tokens_per_sec=5, p95_ttft_seconds=2.0, saturation=0.5)
        front = pareto_front([a_pt, b_pt])
        assert len(front) == 1
        assert front[0] is a_pt

    def test_pareto_keeps_both_when_neither_dominates(self) -> None:
        """A=(tput=10, ttft=2), B=(tput=5, ttft=1): A wins throughput, B wins TTFT.
        Neither dominates → both on the frontier."""
        cfg = EngineConfig(max_num_seqs=8, max_num_batched_tokens=512)
        a = PerfPoint(config=cfg, throughput_tokens_per_sec=10, p95_ttft_seconds=2.0, saturation=0.5)
        b = PerfPoint(config=cfg, throughput_tokens_per_sec=5, p95_ttft_seconds=1.0, saturation=0.5)
        front = pareto_front([a, b])
        assert len(front) == 2

    def test_sixteen_x_p95_ttft_improvement(self) -> None:
        """The headline claim from implementer: 800ms → 50ms = 16x.

        Compare worst-case (max_seqs=8, B=512) vs Pareto-optimal (max_seqs=128, B=8192).
        """
        worst = EngineConfig(max_num_seqs=8, max_num_batched_tokens=512)
        best = EngineConfig(max_num_seqs=128, max_num_batched_tokens=8192)
        ttft_worst = estimate_p95_ttft(worst, long_prompt_tokens=8192, short_prompt_tokens=64)
        ttft_best = estimate_p95_ttft(best, long_prompt_tokens=8192, short_prompt_tokens=64)
        ratio = ttft_worst / ttft_best
        assert abs(ratio - 16.0) < 1e-6
