"""Unit tests for PreemptionStrategy comparator (recompute / swap / abort).

Demo numerics being verified:
- 8K prompt: recompute 163.8 ms, swap 62.5 ms, abort 5000 ms — winner=swap.
- KV bytes formula: 2 * NL * NH * D * dt * L (bytes); for 8K/32/8/128/fp16 = 1 GiB.
- crossover_prompt_length: -1 (recompute always wins) when TP > BW/(4*NL*NH*D*dt).
- For demo's defaults (TP=50K tok/s, BW=32 GiB/s, model=32/8/128/fp16):
    threshold_TP = 32 GiB/s / (4*32*8*128*2 B) = 32 GiB / 256 KiB = 131072 tok/s.
    50K < 131K → swap always wins → returns 0 (NOT -1).
- E[latency] at 5% OOM/step, 100 steps: ~983ms recompute, ~476ms swap, ~25164ms abort.
"""

from __future__ import annotations

from implementation.preemption_strategy import (
    PreemptionScenario,
    PreemptionStrategy,
    crossover_prompt_length,
    expected_latency_under_oom_rate,
)


def _scenario(prompt: int = 8192) -> PreemptionScenario:
    return PreemptionScenario(
        prompt_tokens=prompt, num_layers=32, num_kv_heads=8, head_size=128,
    )


class TestKVBytes:
    def test_8k_request_is_1gib(self) -> None:
        s = _scenario(8192)
        assert s.kv_bytes == 1024**3

    def test_doubling_prompt_doubles_kv(self) -> None:
        a = _scenario(4096)
        b = _scenario(8192)
        assert b.kv_bytes == 2 * a.kv_bytes


class TestRecomputeSwap:
    def test_recompute_seconds_formula(self) -> None:
        """recompute = prompt / TP. 8192 / 50000 = 0.16384."""
        s = _scenario(8192)
        assert abs(s.recompute_seconds() - 0.16384) < 1e-6

    def test_swap_seconds_formula(self) -> None:
        """swap = 2 * KV / BW. 2 * 1 GiB / 32 GiB/s = 0.0625."""
        s = _scenario(8192)
        assert abs(s.swap_seconds() - 0.0625) < 1e-6

    def test_abort_seconds_is_user_penalty(self) -> None:
        s = _scenario(8192)
        # Default penalty is 5.0 seconds.
        assert s.abort_seconds() == 5.0

    def test_abort_penalty_is_configurable(self) -> None:
        s = PreemptionScenario(
            prompt_tokens=8192, num_layers=32, num_kv_heads=8, head_size=128,
            abort_user_penalty_seconds=2.0,
        )
        assert s.abort_seconds() == 2.0


class TestLatencyFor:
    def test_dispatches_per_strategy(self) -> None:
        s = _scenario(8192)
        assert s.latency_for(PreemptionStrategy.RECOMPUTE) == s.recompute_seconds()
        assert s.latency_for(PreemptionStrategy.SWAP) == s.swap_seconds()
        assert s.latency_for(PreemptionStrategy.ABORT) == s.abort_seconds()


class TestWinner:
    def test_8k_winner_is_swap(self) -> None:
        """Demo §3: 8K prompt → swap wins (62.5 ms) over recompute (163.8 ms)."""
        s = _scenario(8192)
        assert s.winner() is PreemptionStrategy.SWAP

    def test_winner_is_swap_for_short_prompt_too(self) -> None:
        """Demo §3 row 1: short prompt (512 tokens) — winner still swap.
        Confirms K06/P02 length-independence: at default TP/BW, swap always wins."""
        s = _scenario(512)
        assert s.winner() is PreemptionStrategy.SWAP

    def test_winner_is_swap_for_xlong_prompt(self) -> None:
        s = _scenario(32768)
        assert s.winner() is PreemptionStrategy.SWAP

    def test_demo_table_reproduces(self) -> None:
        """All four demo §3 rows: recompute / swap / abort latencies match."""
        expected = {
            512:   (10.24, 3.91, 5000.0),     # recompute, swap, abort (ms)
            2048:  (40.96, 15.625, 5000.0),
            8192:  (163.84, 62.5, 5000.0),
            32768: (655.36, 250.0, 5000.0),
        }
        for prompt, (rec_ms, swap_ms, abort_ms) in expected.items():
            s = _scenario(prompt)
            assert abs(s.recompute_seconds() * 1000 - rec_ms) < 0.01
            assert abs(s.swap_seconds() * 1000 - swap_ms) < 0.01
            assert abs(s.abort_seconds() * 1000 - abort_ms) < 0.01


class TestCrossover:
    def test_default_config_swap_always_wins(self) -> None:
        """K06 / P02: at default TP=50K and BW=32 GiB/s with 32/8/128/fp16,
        threshold_TP = 32 GiB / (4*32*8*128*2 B) = 131072 tok/s.
        50K < 131K → recompute is SLOWER → swap always wins → returns 0."""
        base = _scenario(8192)
        assert crossover_prompt_length(base) == 0

    def test_high_tp_makes_recompute_always_win(self) -> None:
        """If we crank TP above the threshold, recompute always wins → returns -1."""
        base = _scenario(8192)
        high_tp = 200_000.0  # well above threshold of ~131K
        result = crossover_prompt_length(base, prefill_throughput_tokens_per_sec=high_tp)
        assert result == -1

    def test_threshold_tp_calculation(self) -> None:
        """Compute threshold ourselves and verify."""
        base = _scenario(8192)
        bytes_per_token = 4 * 32 * 8 * 128 * 2  # 256 KiB
        threshold = base.pcie_bandwidth_bytes_per_sec / bytes_per_token
        # threshold ≈ 131072 tok/s.
        assert int(threshold) == 131072
        # Verify the function dispatches correctly around threshold.
        result_below = crossover_prompt_length(
            base, prefill_throughput_tokens_per_sec=threshold - 1,
        )
        result_above = crossover_prompt_length(
            base, prefill_throughput_tokens_per_sec=threshold + 1,
        )
        assert result_below == 0
        assert result_above == -1

    def test_length_independent(self) -> None:
        """P02: crossover decision is INDEPENDENT of prompt length.
        Build scenarios at L=512 and L=32768; both produce the same dispatch."""
        s512 = _scenario(512)
        s32k = _scenario(32768)
        # Same model shape, same TP/BW → same crossover answer.
        assert crossover_prompt_length(s512) == crossover_prompt_length(s32k)


class TestExpectedLatencyUnderOOM:
    def test_demo_row_recompute_983ms(self) -> None:
        """Demo §3 expected latencies at p=0.05, 100 steps, 8K prompt:
        E[recompute] = 163.84 + 0.05 * 100 * 163.84 = 163.84 + 819.2 = 983.04 ms."""
        s = _scenario(8192)
        e = expected_latency_under_oom_rate(s, PreemptionStrategy.RECOMPUTE, 0.05, 100)
        assert abs(e * 1000 - 983.04) < 0.5

    def test_demo_row_swap_476ms(self) -> None:
        """E[swap] = 163.84 (one prefill) + 0.05 * 100 * 62.5 = 163.84 + 312.5 = 476.34 ms."""
        s = _scenario(8192)
        e = expected_latency_under_oom_rate(s, PreemptionStrategy.SWAP, 0.05, 100)
        assert abs(e * 1000 - 476.34) < 0.5

    def test_demo_row_abort_25164ms(self) -> None:
        """E[abort] = 163.84 + 0.05 * 100 * 5000 = 163.84 + 25000 = 25163.84 ms."""
        s = _scenario(8192)
        e = expected_latency_under_oom_rate(s, PreemptionStrategy.ABORT, 0.05, 100)
        assert abs(e * 1000 - 25163.84) < 0.5

    def test_zero_oom_rate_returns_base_only(self) -> None:
        """If p_oom = 0, expected latency = base prefill."""
        s = _scenario(8192)
        e = expected_latency_under_oom_rate(s, PreemptionStrategy.SWAP, 0.0, 100)
        assert abs(e - s.recompute_seconds()) < 1e-9
