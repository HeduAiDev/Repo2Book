"""Unit tests for BlockPool.

Coverage:
- null block reservation (block 0, is_null, popped from free queue at __init__)
- get_new_blocks: ref_cnt bump, eviction of cached blocks
- free_blocks: ref_cnt drop, return to free queue if ref_cnt=0, cached hash kept
- touch: re-activate freed-but-cached block (O(1) middle remove)
- LRU eviction order (head=LRU, tail=MRU)
- evict_blocks: drop cache hash without freeing
- get_usage: subtracts null block from denominator
- reset_prefix_cache: only when 1 block in use (the null block)
- enable_caching=False: cache_block / get_cached_block become no-ops
- get_new_blocks raises ValueError when not enough free blocks

Knowledge applied:
- K04: null block is block_id=0, never returned, get_usage subtracts 1
- K05: O(1) remove from middle (linked list, not deque)
- K08: free_blocks keeps cached hash; LRU eviction in get_new_blocks clears it
"""

from __future__ import annotations

import pytest

from implementation.block_pool import BlockPool


def _hash(s: str) -> bytes:
    return s.encode()


class TestInit:
    def test_null_block_reserved_at_id_zero(self) -> None:
        """K04: null block popped from free queue, has is_null=True, id=0."""
        pool = BlockPool(num_gpu_blocks=8, enable_caching=True)
        assert pool.null_block.block_id == 0
        assert pool.null_block.is_null is True
        # Free queue starts at len = num_gpu_blocks - 1 (null is excluded).
        assert pool.get_num_free_blocks() == 7

    def test_assert_positive_num_blocks(self) -> None:
        """num_gpu_blocks > 0 is enforced (vLLM block_pool.py:L156)."""
        with pytest.raises(AssertionError):
            BlockPool(num_gpu_blocks=0)


class TestAllocate:
    def test_first_allocated_block_is_id_one(self) -> None:
        """K04: block 0 is the null block; first allocation returns block 1."""
        pool = BlockPool(num_gpu_blocks=8, enable_caching=True)
        out = pool.get_new_blocks(1)
        assert len(out) == 1
        assert out[0].block_id == 1
        assert out[0].ref_cnt == 1

    def test_get_new_blocks_increments_ref_cnt(self) -> None:
        """Every block popped via get_new_blocks must have ref_cnt=1 on return."""
        pool = BlockPool(num_gpu_blocks=8, enable_caching=True)
        out = pool.get_new_blocks(3)
        for b in out:
            assert b.ref_cnt == 1

    def test_get_new_blocks_returns_in_lru_order(self) -> None:
        """Free queue is LRU; consecutive allocs get blocks in id order
        (since nothing has been freed yet)."""
        pool = BlockPool(num_gpu_blocks=8, enable_caching=True)
        out = pool.get_new_blocks(4)
        assert [b.block_id for b in out] == [1, 2, 3, 4]

    def test_oversubscribe_raises(self) -> None:
        """Asking for more blocks than free → ValueError (vLLM block_pool.py L325)."""
        pool = BlockPool(num_gpu_blocks=4, enable_caching=True)
        # Free count = 4 - 1 (null) = 3.
        with pytest.raises(ValueError):
            pool.get_new_blocks(4)


class TestFreeAndRefCnt:
    def test_free_blocks_decrements_ref_cnt(self) -> None:
        """ref_cnt drops by 1 per free; returns to free queue when ref_cnt hits 0."""
        pool = BlockPool(num_gpu_blocks=8, enable_caching=True)
        out = pool.get_new_blocks(3)
        before_free = pool.get_num_free_blocks()
        pool.free_blocks(out)
        for b in out:
            assert b.ref_cnt == 0
        assert pool.get_num_free_blocks() == before_free + 3

    def test_free_blocks_with_multiple_refs_does_not_return(self) -> None:
        """A block with ref_cnt > 1 stays out of the free queue after one free()."""
        pool = BlockPool(num_gpu_blocks=8, enable_caching=True)
        out = pool.get_new_blocks(2)
        # Manually bump ref_cnt to 2 (simulating a shared block from prefix cache hit).
        out[0].ref_cnt += 1
        before = pool.get_num_free_blocks()
        pool.free_blocks([out[0]])
        # ref_cnt is 1 — block is still in use; should NOT have been added back.
        assert out[0].ref_cnt == 1
        assert pool.get_num_free_blocks() == before  # unchanged


class TestPrefixCache:
    def test_cache_block_then_lookup(self) -> None:
        """cache_block stores the block; get_cached_block retrieves it."""
        pool = BlockPool(num_gpu_blocks=8, enable_caching=True)
        b = pool.get_new_blocks(1)[0]
        h = _hash("prefix-x")
        pool.cache_block(b, h)
        assert pool.get_cached_block(h) is b

    def test_cache_hash_persists_after_free(self) -> None:
        """K08: free_blocks does NOT clear the cache hash. The block stays
        addressable by hash until LRU eviction in get_new_blocks."""
        pool = BlockPool(num_gpu_blocks=8, enable_caching=True)
        b = pool.get_new_blocks(1)[0]
        h = _hash("prefix-y")
        pool.cache_block(b, h)
        pool.free_blocks([b])
        # After free: cache still has the entry, ref_cnt=0, block back in queue.
        assert pool.get_cached_block(h) is b
        assert b.ref_cnt == 0

    def test_touch_reactivates_cached_block_o1(self) -> None:
        """touch() on a freed-but-cached block: removes from queue, ref_cnt += 1."""
        pool = BlockPool(num_gpu_blocks=8, enable_caching=True)
        b = pool.get_new_blocks(1)[0]
        h = _hash("prefix-z")
        pool.cache_block(b, h)
        pool.free_blocks([b])
        free_before = pool.get_num_free_blocks()
        pool.touch([b])
        assert b.ref_cnt == 1
        assert pool.get_num_free_blocks() == free_before - 1

    def test_touch_skips_null_block(self) -> None:
        """touch must NOT try to free-queue-remove the null block."""
        pool = BlockPool(num_gpu_blocks=8, enable_caching=True)
        before = pool.get_num_free_blocks()
        pool.touch([pool.null_block])
        # ref_cnt bumps...
        assert pool.null_block.ref_cnt == 1
        # ...but free count unchanged (null block was never in the queue).
        assert pool.get_num_free_blocks() == before

    def test_touch_skips_already_in_use_block(self) -> None:
        """touch on an already-allocated block: just bumps ref_cnt, doesn't
        try to remove from queue (which would fail since it's not there)."""
        pool = BlockPool(num_gpu_blocks=8, enable_caching=True)
        b = pool.get_new_blocks(1)[0]
        before = pool.get_num_free_blocks()
        pool.touch([b])
        assert b.ref_cnt == 2
        assert pool.get_num_free_blocks() == before

    def test_get_new_blocks_evicts_cached_lru(self) -> None:
        """K08: When the LRU block is cached, popping it via get_new_blocks
        clears its cache hash entry (eviction)."""
        pool = BlockPool(num_gpu_blocks=4, enable_caching=True)  # 3 free
        b = pool.get_new_blocks(1)[0]  # block 1
        h = _hash("evict-me")
        pool.cache_block(b, h)
        pool.free_blocks([b])
        # b is at the tail of free queue (most recently freed), block 2,3 are LRU.
        # Allocate 3 — block 2,3, then block 1 (which is cached).
        pool.get_new_blocks(3)
        # Block 1 (which was cached) should now be evicted from the cache map.
        assert pool.get_cached_block(h) is None
        assert b.block_hash is None

    def test_caching_disabled_cache_block_is_noop(self) -> None:
        """enable_caching=False: cache_block does nothing, lookup misses."""
        pool = BlockPool(num_gpu_blocks=8, enable_caching=False)
        b = pool.get_new_blocks(1)[0]
        h = _hash("no-cache")
        pool.cache_block(b, h)
        assert pool.get_cached_block(h) is None
        assert b.block_hash is None


class TestEvictBlocks:
    def test_evict_blocks_clears_hash_only(self) -> None:
        """evict_blocks removes from cache map, does NOT free the block."""
        pool = BlockPool(num_gpu_blocks=8, enable_caching=True)
        b = pool.get_new_blocks(1)[0]
        h = _hash("transfer-away")
        pool.cache_block(b, h)
        # ref_cnt is still 1 (in use).
        pool.evict_blocks({b.block_id})
        assert pool.get_cached_block(h) is None
        assert b.block_hash is None
        # Block is still allocated (ref_cnt unchanged).
        assert b.ref_cnt == 1


class TestUsage:
    def test_get_usage_subtracts_null_block(self) -> None:
        """K04: get_usage denominator excludes the null block."""
        pool = BlockPool(num_gpu_blocks=11, enable_caching=True)  # 10 usable
        assert pool.get_usage() == 0.0
        pool.get_new_blocks(5)
        # 5 / 10 = 0.5
        assert pool.get_usage() == pytest.approx(0.5)
        pool.get_new_blocks(5)
        # 10 / 10 = 1.0
        assert pool.get_usage() == pytest.approx(1.0)

    def test_get_usage_one_block_pool_returns_zero(self) -> None:
        """num_gpu_blocks=1 means only the null block exists; usage stays at 0."""
        pool = BlockPool(num_gpu_blocks=1, enable_caching=True)
        assert pool.get_usage() == 0.0


class TestResetPrefixCache:
    def test_reset_works_when_only_null_in_use(self) -> None:
        """reset_prefix_cache clears all hashes; only allowed when no requests run."""
        pool = BlockPool(num_gpu_blocks=8, enable_caching=True)
        b = pool.get_new_blocks(1)[0]
        pool.cache_block(b, _hash("x"))
        pool.free_blocks([b])
        # Now only the null block is "in use" (ref_cnt for null was never bumped, but
        # by accounting num_used = num_gpu_blocks - num_free = 8 - 7 = 1, and that 1
        # is the null block).
        ok = pool.reset_prefix_cache()
        assert ok is True
        assert len(pool.cached_block_hash_to_block) == 0

    def test_reset_refuses_when_other_blocks_in_use(self) -> None:
        """If any non-null block is allocated, reset returns False."""
        pool = BlockPool(num_gpu_blocks=8, enable_caching=True)
        pool.get_new_blocks(2)  # blocks 1, 2 in use
        ok = pool.reset_prefix_cache()
        assert ok is False
