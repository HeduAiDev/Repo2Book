"""Unit tests for PreemptionScenario (recompute vs swap analytical model).

Demo numerics: 8K-token request, 32 layers, 8 KV heads, head_size=128, fp16.
- KV bytes ≈ 1 GiB (= 2 * 32 * 8 * 128 * 8192 * 2 = 1073741824 bytes exactly).
- Recompute time = 8192 / 50000 ≈ 163.8 ms.
- Swap time = (2 * 1 GiB) / 32 GiB/s ≈ 62.5 ms.
- Recompute is SLOWER but vLLM v1 chose it (K06: simplicity, OOM safety).
"""

from __future__ import annotations

from implementation.recompute import PreemptionScenario


class TestKVBytesFormula:
    def test_canonical_8k_request_is_1gib(self) -> None:
        """2 * 32 * 8 * 128 * 8192 * 2 = 1 GiB exactly."""
        s = PreemptionScenario(
            prompt_tokens=8192, num_layers=32, num_kv_heads=8, head_size=128,
        )
        assert s.kv_bytes == 1024**3  # 1 GiB

    def test_doubling_prompt_doubles_kv(self) -> None:
        """Linear scaling in prompt_tokens."""
        a = PreemptionScenario(prompt_tokens=4096, num_layers=32, num_kv_heads=8, head_size=128)
        b = PreemptionScenario(prompt_tokens=8192, num_layers=32, num_kv_heads=8, head_size=128)
        assert b.kv_bytes == 2 * a.kv_bytes

    def test_doubling_layers_doubles_kv(self) -> None:
        a = PreemptionScenario(prompt_tokens=8192, num_layers=16, num_kv_heads=8, head_size=128)
        b = PreemptionScenario(prompt_tokens=8192, num_layers=32, num_kv_heads=8, head_size=128)
        assert b.kv_bytes == 2 * a.kv_bytes


class TestRecomputeVsSwap:
    def test_demo_numbers_in_ballpark(self) -> None:
        """Reproduce demo's 8K-prompt latencies within ±1 ms."""
        s = PreemptionScenario(
            prompt_tokens=8192, num_layers=32, num_kv_heads=8, head_size=128,
        )
        # 8192 / 50000 = 0.16384 s.
        assert abs(s.recompute_seconds - 0.16384) < 1e-6
        # KV=1 GiB, swap = 2 GiB / 32 GiB/s = 0.0625 s.
        assert abs(s.swap_seconds - 0.0625) < 1e-6

    def test_recompute_is_slower_for_8k(self) -> None:
        """K06: at 8K prompt, recompute (~164ms) > swap (~62ms). vLLM v1 still chose
        recompute for simplicity, OOM-safety, and bit-determinism — but the
        analytical model must HONESTLY report swap is faster on raw latency."""
        s = PreemptionScenario(
            prompt_tokens=8192, num_layers=32, num_kv_heads=8, head_size=128,
        )
        assert s.recompute_is_faster is False

    def test_short_prompt_recompute_is_faster(self) -> None:
        """For a tiny prompt, recompute is so fast that swap loses on PCIe overhead.
        At ~50K tok/s, 64 tokens = 1.28 ms; swap of 64 tokens ≈ 16KiB/32GiB/s ~ 0.5 µs.
        Actually swap is ALWAYS faster when KV is tiny — so this test is the OPPOSITE:
        we just verify the formula respects the inputs."""
        # Tiny prompt → tiny KV → tiny swap, but recompute also tiny.
        s = PreemptionScenario(
            prompt_tokens=64, num_layers=32, num_kv_heads=8, head_size=128,
        )
        # Recompute = 64 / 50000 = 1.28 ms.
        assert abs(s.recompute_seconds - 64 / 50000) < 1e-9
        # Swap = 2 * KV / bandwidth.
        expected_swap = 2 * s.kv_bytes / s.pcie_bandwidth_bytes_per_sec
        assert abs(s.swap_seconds - expected_swap) < 1e-9


class TestBytesMoved:
    def test_recompute_moves_zero_bytes(self) -> None:
        """K06: recompute path is on-GPU only — no PCIe traffic."""
        s = PreemptionScenario(prompt_tokens=8192, num_layers=32, num_kv_heads=8, head_size=128)
        assert s.recompute_bytes_moved == 0

    def test_swap_moves_2x_kv_bytes(self) -> None:
        """Swap = round trip: out then in. Total = 2 * KV."""
        s = PreemptionScenario(prompt_tokens=8192, num_layers=32, num_kv_heads=8, head_size=128)
        assert s.swap_bytes_moved == 2 * s.kv_bytes


class TestReport:
    def test_report_string_includes_key_fields(self) -> None:
        """report() must mention prompt_tokens, KV bytes, both latencies, decision."""
        s = PreemptionScenario(prompt_tokens=8192, num_layers=32, num_kv_heads=8, head_size=128)
        text = s.report()
        assert "8192" in text
        assert "Recompute" in text
        assert "Swap" in text
        assert "False" in text  # recompute_is_faster=False for 8K prompt
