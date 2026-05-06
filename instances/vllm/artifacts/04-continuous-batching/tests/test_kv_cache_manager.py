"""Unit tests for SimpleKVCacheManager.

The scheduler relies on three contracts (see scheduler.py:L466-L471, L961):
- allocate_slots returns a fresh list of block IDs OR None on OOM.
- block-rounding: ceil(num_new_tokens / block_size) blocks per call.
- free returns blocks to the pool AND clears request.block_ids.
"""

from __future__ import annotations

from implementation.kv_cache_manager import SimpleKVCacheManager
from implementation.request import Request


def _req(rid: str = "r1", n: int = 0) -> Request:
    return Request(rid, list(range(n)), max_tokens=1, arrival_time=0.0)


class TestAllocateSlots:
    def test_block_rounding_exact(self) -> None:
        """allocate(16, block_size=16) -> exactly 1 block."""
        kv = SimpleKVCacheManager(num_gpu_blocks=10, block_size=16)
        out = kv.allocate_slots(_req(), 16)
        assert out == [0]

    def test_block_rounding_partial(self) -> None:
        """allocate(17, block_size=16) -> 2 blocks (ceil division)."""
        kv = SimpleKVCacheManager(num_gpu_blocks=10, block_size=16)
        out = kv.allocate_slots(_req(), 17)
        assert out == [0, 1]

    def test_block_rounding_one_token(self) -> None:
        """allocate(1, block_size=16) -> 1 block (still rounds up)."""
        kv = SimpleKVCacheManager(num_gpu_blocks=10, block_size=16)
        out = kv.allocate_slots(_req(), 1)
        assert out == [0]

    def test_allocate_returns_none_on_oom(self) -> None:
        """allocate beyond num_gpu_blocks returns None — the scheduler relies
        on this to trigger the preempt-and-retry path."""
        kv = SimpleKVCacheManager(num_gpu_blocks=2, block_size=16)
        # Asking for 5 blocks of 16 = 80 tokens > 2 blocks * 16.
        out = kv.allocate_slots(_req(), 80)
        assert out is None

    def test_sequential_allocation_disjoint_ids(self) -> None:
        """Two consecutive allocations return disjoint, increasing IDs."""
        kv = SimpleKVCacheManager(num_gpu_blocks=10, block_size=16)
        a = kv.allocate_slots(_req("a"), 32)  # 2 blocks
        b = kv.allocate_slots(_req("b"), 32)  # 2 blocks
        assert a == [0, 1]
        assert b == [2, 3]
        assert kv.num_used_blocks == 4

    def test_allocate_does_not_attach_blocks(self) -> None:
        """allocate_slots returns blocks; it does NOT mutate request.block_ids.
        That's the scheduler's job at scheduler.py:L516."""
        kv = SimpleKVCacheManager(num_gpu_blocks=10, block_size=16)
        r = _req()
        out = kv.allocate_slots(r, 16)
        assert out == [0]
        assert r.block_ids == []  # Untouched.


class TestFree:
    def test_free_returns_blocks_to_pool(self) -> None:
        """free puts blocks back AND clears request.block_ids."""
        kv = SimpleKVCacheManager(num_gpu_blocks=4, block_size=16)
        r = _req()
        r.block_ids = kv.allocate_slots(r, 32)  # type: ignore[assignment]
        assert r.block_ids == [0, 1]
        assert kv.num_used_blocks == 2

        kv.free(r)
        assert r.block_ids == []
        assert kv.num_used_blocks == 0
        assert kv.num_free_blocks == 4

    def test_full_cycle_recovers_all_blocks(self) -> None:
        """allocate-then-free over many requests returns exactly to start."""
        kv = SimpleKVCacheManager(num_gpu_blocks=8, block_size=16)
        reqs = [_req(f"r{i}") for i in range(4)]
        for r in reqs:
            r.block_ids = kv.allocate_slots(r, 16)  # type: ignore[assignment]
        assert kv.num_used_blocks == 4
        for r in reqs:
            kv.free(r)
        assert kv.num_free_blocks == kv.num_gpu_blocks


class TestStepStarts:
    def test_new_step_starts_is_noop(self) -> None:
        """No state change; just here to match vLLM's KVCacheManager API."""
        kv = SimpleKVCacheManager(num_gpu_blocks=4, block_size=16)
        before = (kv.num_free_blocks, kv.num_used_blocks)
        kv.new_step_starts()
        after = (kv.num_free_blocks, kv.num_used_blocks)
        assert before == after
