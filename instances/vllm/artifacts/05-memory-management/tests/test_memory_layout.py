"""Unit + integration tests for determine_available_memory + estimate_max_concurrency.

This is THE module readers will plug their own numbers into. The math must be
faithful to vLLM gpu_worker.py:L411-L445 down to the integer-division semantics.

Demo numerics being verified (from implementer):
- Llama-3.2-1B on H100: 35,148 num_gpu_blocks, 275 max concurrent @ 2k seq.
- block_size=16, num_kv_heads=8, head_size=128, fp16 → page_size = 64 KiB.
- weights=2.4 GiB, peak_act=1.8 GiB, non_torch=0.5 GiB, cudagraph=256 MiB.
- requested = int(80 GiB * 0.92), available_kv = requested - all - cudagraph.
"""

from __future__ import annotations

from implementation.kv_cache_spec import FullAttentionSpec
from implementation.memory_layout import (
    determine_available_memory,
    estimate_max_concurrency,
)
from implementation.mem_snapshot import MemoryProfilingResult, MemorySnapshot


GIB = 1024**3
MIB = 1024**2


def _llama_1b_inputs() -> tuple[MemorySnapshot, MemoryProfilingResult, int]:
    """Reproduce demo's _llama_3_2_1b_profile() inputs verbatim."""
    init = MemorySnapshot(
        total_memory=80 * GIB,
        free_memory=78 * GIB,
        cuda_memory=2 * GIB,
        torch_memory=0,
        non_torch_memory=2 * GIB,
        timestamp=0.0,
    )
    profile = MemoryProfilingResult(
        before_create=init,
        weights_memory=int(2.4 * GIB),
        torch_peak_increase=int(1.8 * GIB),
        non_torch_increase=int(0.5 * GIB),
        profile_time=2.13,
    )
    cudagraph = 256 * MIB
    return init, profile, cudagraph


class TestDetermineAvailableMemory:
    def test_demo_num_gpu_blocks_reproduces(self) -> None:
        """The headline number: 35,148 blocks for Llama-3.2-1B on H100."""
        init, profile, cg = _llama_1b_inputs()
        spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype_bytes=2)
        layout = determine_available_memory(
            init_snapshot=init,
            profile_result=profile,
            cudagraph_memory=cg,
            gpu_memory_utilization=0.92,
            spec=spec,
            num_layers=32,
        )
        assert layout.num_gpu_blocks == 35148

    def test_breakdown_matches_demo(self) -> None:
        """Each line of the breakdown matches the demo report (within int rounding)."""
        init, profile, cg = _llama_1b_inputs()
        spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype_bytes=2)
        layout = determine_available_memory(init, profile, cg, 0.92, spec, 32)
        # Total + requested are exact.
        assert layout.total_gpu_memory == 80 * GIB
        assert layout.requested_memory == int(80 * GIB * 0.92)
        # Categories match the inputs.
        assert layout.weights == int(2.4 * GIB)
        assert layout.peak_activation == int(1.8 * GIB)
        assert layout.non_torch == int(0.5 * GIB)
        assert layout.cudagraph_memory == cg
        # The available KV cache is the residual.
        expected_avail = (
            layout.requested_memory
            - layout.weights
            - layout.peak_activation
            - layout.non_torch
            - cg
        )
        assert layout.available_kv_cache == expected_avail
        # Page size is 64 KiB.
        assert layout.page_size_bytes == 65536
        # Wasted bytes < page_size * num_layers (the divide bound).
        assert 0 <= layout.wasted_bytes < layout.page_size_bytes * layout.num_layers

    def test_util_margin_is_eight_percent(self) -> None:
        """K03: gpu_memory_utilization=0.92 → 8% margin = 6.4 GiB on 80 GiB."""
        init, profile, cg = _llama_1b_inputs()
        spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype_bytes=2)
        layout = determine_available_memory(init, profile, cg, 0.92, spec, 32)
        # 80 - int(80*0.92) ≈ 6.4 GiB.
        assert layout.util_margin == 80 * GIB - int(80 * GIB * 0.92)
        assert abs(layout.util_margin - int(6.4 * GIB)) < GIB // 100  # within 0.01 GiB

    def test_negative_kv_clamps_to_zero(self) -> None:
        """If non_kv_cache > requested, available_kv_cache must clamp to 0
        (no negative blocks). This is the OOM guard at L122-L123."""
        init = MemorySnapshot(total_memory=10 * GIB)
        # Profile: 5 + 4 + 2 = 11 GiB > 9.2 GiB requested.
        profile = MemoryProfilingResult(
            weights_memory=5 * GIB,
            torch_peak_increase=4 * GIB,
            non_torch_increase=2 * GIB,
        )
        spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype_bytes=2)
        layout = determine_available_memory(init, profile, 0, 0.92, spec, 32)
        assert layout.available_kv_cache == 0
        assert layout.num_gpu_blocks == 0


class TestPageSizeSensitivity:
    """Sweep block_size and verify the demo's [5] table reproduces."""

    def test_block_size_sweep_matches_demo(self) -> None:
        init, profile, cg = _llama_1b_inputs()
        expected = {
            8:  70297,
            16: 35148,
            32: 17574,
            64: 8787,
        }
        for bs, blocks_expected in expected.items():
            spec = FullAttentionSpec(
                block_size=bs, num_kv_heads=8, head_size=128, dtype_bytes=2,
            )
            layout = determine_available_memory(init, profile, cg, 0.92, spec, 32)
            assert layout.num_gpu_blocks == blocks_expected, (
                f"block_size={bs}: expected {blocks_expected}, got {layout.num_gpu_blocks}"
            )


class TestEstimateMaxConcurrency:
    def test_demo_concurrency_matches(self) -> None:
        """275 concurrent at avg_seq_len=2048 (block_size=16)."""
        init, profile, cg = _llama_1b_inputs()
        spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype_bytes=2)
        layout = determine_available_memory(init, profile, cg, 0.92, spec, 32)
        conc = estimate_max_concurrency(layout, avg_seq_len=2048, block_size=16)
        # 35148 / ceil(2048/16) = 35148 / 128 = 274.59... → demo prints 275 (rounded up).
        assert int(round(conc)) in (274, 275)

    def test_zero_seq_len_returns_zero(self) -> None:
        """avg_seq_len=0 → 0 concurrent (no division by zero)."""
        init, profile, cg = _llama_1b_inputs()
        spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype_bytes=2)
        layout = determine_available_memory(init, profile, cg, 0.92, spec, 32)
        assert estimate_max_concurrency(layout, avg_seq_len=0, block_size=16) == 0.0

    def test_seq_len_smaller_than_block_size_uses_one_block(self) -> None:
        """A 1-token request still occupies 1 full block (ceil division)."""
        init, profile, cg = _llama_1b_inputs()
        spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype_bytes=2)
        layout = determine_available_memory(init, profile, cg, 0.92, spec, 32)
        conc_1tok = estimate_max_concurrency(layout, avg_seq_len=1, block_size=16)
        # 35148 / 1 = 35148.
        assert conc_1tok == 35148
