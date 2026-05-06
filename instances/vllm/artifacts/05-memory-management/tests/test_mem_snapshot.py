"""Unit tests for MemorySnapshot, MemoryProfilingResult, memory_profiling.

The math being verified:
- Subtraction across two snapshots yields per-field diffs (vLLM mem_utils.py:L128-L145).
- memory_profiling fills `non_kv_cache_memory = non_torch + peak + weights`
  (vLLM mem_utils.py:L263-L275).
- Worked example from docstring (mem_utils.py:L209-L235): cat1=1, cat2=0->2->4->3,
  cat3=0->0.5->1->1, weights=2 → result.non_kv_cache_memory = 5 GiB.
"""

from __future__ import annotations

from implementation.mem_snapshot import (
    MemoryProfilingResult,
    MemorySnapshot,
    format_gib,
    memory_profiling,
)


GIB = 1024**3


class TestMemorySnapshot:
    def test_default_construction(self) -> None:
        """Empty snapshot has all-zero counters."""
        s = MemorySnapshot()
        assert s.torch_peak == 0
        assert s.free_memory == 0
        assert s.total_memory == 0
        assert s.cuda_memory == 0
        assert s.torch_memory == 0
        assert s.non_torch_memory == 0

    def test_subtraction_per_field(self) -> None:
        """__sub__ subtracts each numeric field independently."""
        # REFERENCE: mem_utils.py:L128-L145
        a = MemorySnapshot(
            torch_peak=10, free_memory=80, total_memory=80,
            cuda_memory=20, torch_memory=15, non_torch_memory=5,
            timestamp=100.0,
        )
        b = MemorySnapshot(
            torch_peak=4, free_memory=78, total_memory=80,
            cuda_memory=8, torch_memory=6, non_torch_memory=2,
            timestamp=90.0,
        )
        d = a - b
        assert d.torch_peak == 6
        assert d.free_memory == 2
        assert d.total_memory == 0
        assert d.cuda_memory == 12
        assert d.torch_memory == 9
        assert d.non_torch_memory == 3
        assert d.timestamp == 10.0

    def test_subtraction_does_not_auto_measure(self) -> None:
        """Diff result must have auto_measure=False to not trigger measure()."""
        a = MemorySnapshot(torch_peak=2)
        b = MemorySnapshot(torch_peak=1)
        d = a - b
        assert d.auto_measure is False

    def test_measure_is_idempotent_stub(self) -> None:
        """Without a GPU, measure() only sets timestamp once."""
        s = MemorySnapshot()
        s.measure()
        first_ts = s.timestamp
        assert first_ts > 0
        # Second measure should NOT overwrite the timestamp (it's already set).
        s.measure()
        assert s.timestamp == first_ts


class TestMemoryProfiling:
    def test_worked_example_from_vllm_docstring(self) -> None:
        """Reproduce the canonical worked example from mem_utils.py:L209-L235."""
        # Before vLLM creation: cat1=1 GiB used, torch=0, non_torch=0
        before_create = MemorySnapshot(
            cuda_memory=1 * GIB, torch_memory=0, non_torch_memory=0,
        )
        # Before profile (model loaded): weights=2 GiB, NCCL=0.5 GiB
        before_profile = MemorySnapshot(
            cuda_memory=int(3.5 * GIB),
            torch_memory=2 * GIB,
            non_torch_memory=int(1.5 * GIB),  # 1 (cat1) + 0.5 (NCCL)
            torch_peak=2 * GIB,
            timestamp=10.0,
        )
        # After profile: torch peaked at 4 (acts +2), non_torch grew to 1
        after_profile = MemorySnapshot(
            cuda_memory=4 * GIB,
            torch_memory=3 * GIB,  # 2 weights + 1 leftover acts (gc'd to 3)
            non_torch_memory=2 * GIB,  # cat1=1 + NCCL=1
            torch_peak=4 * GIB,
            timestamp=12.13,
        )

        with memory_profiling(before_create, weights_memory=2 * GIB) as result:
            result.before_profile = before_profile
            result.after_profile = after_profile

        assert result.weights_memory == 2 * GIB
        # peak_increase = 4 - 2 = 2 GiB
        assert result.torch_peak_increase == 2 * GIB
        # non_torch_increase = 2 - 0 = 2 GiB (relative to before_create)
        assert result.non_torch_increase == 2 * GIB
        # non_kv_cache = 2 + 2 + 2 = 6 GiB
        assert result.non_kv_cache_memory == 6 * GIB
        # profile_time = 12.13 - 10.0
        assert abs(result.profile_time - 2.13) < 1e-6

    def test_zero_increases_when_snapshots_identical(self) -> None:
        """If before_profile == after_profile, all increases are zero."""
        snap = MemorySnapshot(torch_peak=1, non_torch_memory=1, timestamp=5.0)
        with memory_profiling(snap, weights_memory=10) as result:
            result.before_profile = snap
            result.after_profile = snap
        assert result.torch_peak_increase == 0
        assert result.non_torch_increase == 0
        # non_kv_cache_memory still includes weights even when increases are 0.
        assert result.non_kv_cache_memory == 10


class TestFormatGib:
    def test_two_decimal_places(self) -> None:
        """Matches vLLM's format_gib helper output (no unit suffix)."""
        assert format_gib(0) == "0.00"
        assert format_gib(GIB) == "1.00"
        assert format_gib(2 * GIB + 512 * 1024**2) == "2.50"
