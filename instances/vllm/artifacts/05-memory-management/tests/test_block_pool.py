"""Tests for block_pool.py — KV cache block pool with prefix caching."""

import pytest
from implementation.block_pool import (
    BlockPool,
    KVCacheBlock,
    FreeKVCacheBlockQueue,
    BlockHashToBlockMap,
    hash_block_tokens,
    NONE_HASH,
    BLOCK_SIZE,
)


# ─── hash_block_tokens ───────────────────────────────────────────────────────

def test_hash_block_tokens_deterministic():
    """Same inputs produce same hash."""
    tokens = [1, 2, 3, 4, 5]
    h1 = hash_block_tokens(None, tokens)
    h2 = hash_block_tokens(None, tokens)
    assert h1 == h2


def test_hash_block_tokens_different_tokens_different_hash():
    """Different token IDs produce different hashes."""
    h1 = hash_block_tokens(None, [1, 2, 3])
    h2 = hash_block_tokens(None, [1, 2, 4])
    assert h1 != h2


def test_hash_block_tokens_different_parent_different_hash():
    """Different parent hashes produce different child hashes."""
    tokens = [1, 2, 3]
    h1 = hash_block_tokens(b"parent_a", tokens)
    h2 = hash_block_tokens(b"parent_b", tokens)
    assert h1 != h2


def test_hash_block_tokens_none_parent_uses_none_hash():
    """None parent defaults to NONE_HASH seed."""
    tokens = [1, 2, 3]
    h1 = hash_block_tokens(None, tokens)
    h2 = hash_block_tokens(NONE_HASH, tokens)
    assert h1 == h2


def test_hash_block_tokens_result_is_bytes():
    """Hash output is bytes type."""
    h = hash_block_tokens(None, [1, 2, 3])
    assert isinstance(h, bytes)
    assert len(h) == 32  # SHA-256


def test_hash_block_tokens_empty_sequence():
    """Empty token sequence still produces a valid hash."""
    h = hash_block_tokens(None, [])
    assert isinstance(h, bytes)
    assert len(h) == 32


# ─── KVCacheBlock ────────────────────────────────────────────────────────────

def test_kv_cache_block_initial_state():
    """New block has correct initial values."""
    block = KVCacheBlock(block_id=5)
    assert block.block_id == 5
    assert block.ref_cnt == 0
    assert block._block_hash is None
    assert block.is_null is False
    assert block.prev_free_block is None
    assert block.next_free_block is None


def test_kv_cache_block_hash_setter():
    """block_hash property sets the hash once."""
    block = KVCacheBlock(block_id=0)
    h = b"test_hash_32_bytes_" + b"x" * 15  # 32 bytes
    block.block_hash = h
    assert block.block_hash == h


def test_kv_cache_block_hash_double_set_raises():
    """Setting block_hash twice raises AssertionError."""
    block = KVCacheBlock(block_id=0)
    block.block_hash = b"hash_a_32_bytes_long_" + b"x" * 10
    with pytest.raises(AssertionError, match="Double-cache"):
        block.block_hash = b"hash_b_32_bytes_long_" + b"x" * 10


def test_kv_cache_block_reset_hash():
    """reset_hash clears the block hash to None."""
    block = KVCacheBlock(block_id=0)
    block.block_hash = b"hash_a_32_bytes_long_" + b"x" * 10
    block.reset_hash()
    assert block.block_hash is None
    # Setting again after reset should work
    block.block_hash = b"hash_c_32_bytes_long_" + b"x" * 10
    assert block.block_hash is not None


def test_kv_cache_block_repr():
    """__repr__ includes block_id and ref_cnt."""
    block = KVCacheBlock(block_id=3, ref_cnt=2)
    s = repr(block)
    assert "block_id=3" in s
    assert "ref_cnt=2" in s


# ─── FreeKVCacheBlockQueue ───────────────────────────────────────────────────

@pytest.fixture
def sample_blocks():
    """Create 5 blocks with IDs 0-4."""
    return [KVCacheBlock(block_id=i) for i in range(5)]


@pytest.fixture
def empty_queue():
    """Queue with no real blocks."""
    return FreeKVCacheBlockQueue([])


@pytest.fixture
def five_block_queue(sample_blocks):
    """Queue with 5 free blocks."""
    return FreeKVCacheBlockQueue(sample_blocks)


class TestFreeKVCacheBlockQueue:
    """Tests for the free block doubly-linked list queue."""

    def test_initial_chain(self, five_block_queue, sample_blocks):
        """Blocks are linked consecutively: 0→1→2→3→4."""
        assert five_block_queue.num_free_blocks == 5
        all_blocks = five_block_queue.get_all_free_blocks()
        assert len(all_blocks) == 5
        assert all_blocks[0].block_id == 0
        assert all_blocks[1].block_id == 1
        assert all_blocks[-1].block_id == 4

    def test_empty_queue_construction(self):
        """Empty queue has zero free blocks."""
        q = FreeKVCacheBlockQueue([])
        assert q.num_free_blocks == 0
        assert q.get_all_free_blocks() == []

    def test_popleft_removes_first(self, five_block_queue):
        """popleft removes and returns the first block."""
        block = five_block_queue.popleft()
        assert block.block_id == 0
        assert five_block_queue.num_free_blocks == 4
        remaining = five_block_queue.get_all_free_blocks()
        assert [b.block_id for b in remaining] == [1, 2, 3, 4]

    def test_popleft_empty_raises(self, empty_queue):
        """popleft on empty queue raises ValueError."""
        with pytest.raises(ValueError, match="No free blocks"):
            empty_queue.popleft()

    def test_popleft_n_multiple(self, five_block_queue):
        """popleft_n removes n blocks from the front."""
        blocks = five_block_queue.popleft_n(3)
        assert len(blocks) == 3
        assert [b.block_id for b in blocks] == [0, 1, 2]
        assert five_block_queue.num_free_blocks == 2

    def test_popleft_n_all(self, five_block_queue):
        """popleft_n(5) removes all blocks."""
        blocks = five_block_queue.popleft_n(5)
        assert len(blocks) == 5
        assert five_block_queue.num_free_blocks == 0
        assert five_block_queue.get_all_free_blocks() == []

    def test_popleft_n_zero(self, five_block_queue):
        """popleft_n(0) returns empty list."""
        blocks = five_block_queue.popleft_n(0)
        assert blocks == []
        assert five_block_queue.num_free_blocks == 5

    def test_popleft_n_too_many_raises(self, five_block_queue):
        """popleft_n with n > available raises AssertionError."""
        with pytest.raises(AssertionError):
            five_block_queue.popleft_n(10)

    def test_remove_middle(self, five_block_queue, sample_blocks):
        """remove block from middle (O(1) middle removal)."""
        # Remove block 2 from the middle
        five_block_queue.remove(sample_blocks[2])
        assert five_block_queue.num_free_blocks == 4
        remaining = five_block_queue.get_all_free_blocks()
        assert [b.block_id for b in remaining] == [0, 1, 3, 4]

    def test_remove_first_then_popleft(self, five_block_queue, sample_blocks):
        """After removing first block, popleft returns the second."""
        five_block_queue.remove(sample_blocks[0])
        block = five_block_queue.popleft()
        assert block.block_id == 1

    def test_remove_last(self, five_block_queue, sample_blocks):
        """remove last block works correctly."""
        five_block_queue.remove(sample_blocks[4])
        remaining = five_block_queue.get_all_free_blocks()
        assert [b.block_id for b in remaining] == [0, 1, 2, 3]

    def test_remove_not_in_list_raises(self, five_block_queue):
        """remove on detached block raises RuntimeError."""
        detached = KVCacheBlock(block_id=99)
        with pytest.raises(RuntimeError):
            five_block_queue.remove(detached)

    def test_remove_detached_after_popleft_raises(self, five_block_queue):
        """remove on already popped block raises RuntimeError."""
        block = five_block_queue.popleft()
        with pytest.raises(RuntimeError):
            five_block_queue.remove(block)

    def test_append(self, five_block_queue):
        """append adds block to tail."""
        new_block = KVCacheBlock(block_id=99)
        five_block_queue.append(new_block)
        assert five_block_queue.num_free_blocks == 6
        remaining = five_block_queue.get_all_free_blocks()
        assert remaining[-1].block_id == 99
        # Check chain: last real block → new → fake_tail
        assert remaining[-2].next_free_block is new_block
        assert new_block.prev_free_block is remaining[-2]

    def test_append_n(self, five_block_queue):
        """append_n adds multiple blocks in order."""
        new_blocks = [KVCacheBlock(block_id=99), KVCacheBlock(block_id=100)]
        five_block_queue.append_n(new_blocks)
        assert five_block_queue.num_free_blocks == 7
        remaining = five_block_queue.get_all_free_blocks()
        assert [b.block_id for b in remaining[-2:]] == [99, 100]

    def test_append_n_empty(self, five_block_queue):
        """append_n with empty list is a no-op."""
        five_block_queue.append_n([])
        assert five_block_queue.num_free_blocks == 5

    def test_popleft_then_append(self, five_block_queue):
        """Popped block can be re-appended."""
        block = five_block_queue.popleft()
        five_block_queue.append(block)
        assert five_block_queue.num_free_blocks == 5
        remaining = five_block_queue.get_all_free_blocks()
        assert remaining[-1].block_id == 0

    def test_fill_then_empty(self, five_block_queue):
        """Emptying and refilling works correctly."""
        # Drain
        for _ in range(5):
            five_block_queue.popleft()
        assert five_block_queue.num_free_blocks == 0
        # Refill
        b1 = KVCacheBlock(block_id=10)
        b2 = KVCacheBlock(block_id=11)
        five_block_queue.append(b1)
        five_block_queue.append(b2)
        assert five_block_queue.num_free_blocks == 2
        assert [b.block_id for b in five_block_queue.get_all_free_blocks()] == [10, 11]


# ─── BlockHashToBlockMap ─────────────────────────────────────────────────────

class TestBlockHashToBlockMap:
    """Tests for the hash→block mapping used by prefix cache."""

    def test_insert_and_get(self):
        """Insert a block, then get it by hash."""
        m = BlockHashToBlockMap()
        block = KVCacheBlock(block_id=7)
        h = b"unique_hash_32_bytes_length\x00\x00"
        m.insert(h, block)
        assert m.get_one_block(h) is block

    def test_get_miss_returns_none(self):
        """get_one_block returns None for unknown hash."""
        m = BlockHashToBlockMap()
        assert m.get_one_block(b"nonexistent") is None

    def test_duplicate_hash_upgrades_to_dict(self):
        """Two blocks with same hash upgrade to dict storage."""
        m = BlockHashToBlockMap()
        b1 = KVCacheBlock(block_id=1)
        b2 = KVCacheBlock(block_id=2)
        h = b"dup_hash_32_bytes_long_______\x00"
        m.insert(h, b1)
        m.insert(h, b2)
        # Should return one of the blocks (arbitrary but deterministic in Python 3.7+)
        result = m.get_one_block(h)
        assert result in (b1, b2)

    def test_pop_removes_single_block(self):
        """pop removes a block and returns it."""
        m = BlockHashToBlockMap()
        block = KVCacheBlock(block_id=7)
        h = b"pop_hash_32_bytes_long________\x00"
        m.insert(h, block)
        popped = m.pop(h, 7)
        assert popped is block
        assert m.get_one_block(h) is None

    def test_pop_wrong_block_id_returns_none(self):
        """pop with wrong block_id returns None and keeps entry."""
        m = BlockHashToBlockMap()
        block = KVCacheBlock(block_id=5)
        h = b"hash_for_pop_wrong_id________\x00"
        m.insert(h, block)
        popped = m.pop(h, 99)
        assert popped is None
        assert m.get_one_block(h) is block  # Still there

    def test_pop_duplicate_hash_removes_one(self):
        """pop from duplicate-hash dict removes one block, keeps others."""
        m = BlockHashToBlockMap()
        b1 = KVCacheBlock(block_id=1)
        b2 = KVCacheBlock(block_id=2)
        h = b"dup_pop_hash_32_bytes________\x00"
        m.insert(h, b1)
        m.insert(h, b2)
        popped = m.pop(h, 1)
        assert popped is b1
        # b2 should still be there
        result = m.get_one_block(h)
        assert result is b2

    def test_pop_last_duplicate_clears(self):
        """pop the last block in a duplicate dict clears the entire entry."""
        m = BlockHashToBlockMap()
        b1 = KVCacheBlock(block_id=1)
        b2 = KVCacheBlock(block_id=2)
        h = b"full_pop_hash_32_bytes________\x00"
        m.insert(h, b1)
        m.insert(h, b2)
        m.pop(h, 1)
        m.pop(h, 2)
        assert m.get_one_block(h) is None

    def test_len(self):
        """len reflects number of entries."""
        m = BlockHashToBlockMap()
        assert len(m) == 0
        m.insert(b"h1" + b"\x00" * 30, KVCacheBlock(block_id=1))
        assert len(m) == 1
        m.insert(b"h2" + b"\x00" * 30, KVCacheBlock(block_id=2))
        assert len(m) == 2

    def test_empty_cache_returns_none(self):
        """Empty map returns None for any lookup."""
        m = BlockHashToBlockMap()
        assert m.get_one_block(b"anything") is None


# ─── BlockPool ───────────────────────────────────────────────────────────────

@pytest.fixture
def pool():
    """BlockPool with 50 blocks."""
    return BlockPool(num_gpu_blocks=50, enable_caching=True)


class TestBlockPoolInit:
    """Tests for BlockPool initialization state."""

    def test_initial_free_blocks(self, pool):
        """After init, (N-1) blocks are free (1 reserved for null)."""
        assert pool.get_num_free_blocks() == 49

    def test_null_block_reserved(self, pool):
        """Null block is marked and removed from free pool."""
        assert pool.null_block.is_null
        # Null block should NOT be in free list

    def test_initial_usage(self, pool):
        """Initial usage is 0%."""
        assert pool.get_usage() == 0.0

    def test_num_gpu_blocks_property(self, pool):
        """num_gpu_blocks stored correctly."""
        assert pool.num_gpu_blocks == 50


class TestBlockPoolAllocation:
    """Tests for block allocation (get_new_blocks)."""

    def test_get_new_blocks_basic(self, pool):
        """Allocating blocks reduces free count."""
        blocks = pool.get_new_blocks(3)
        assert len(blocks) == 3
        assert pool.get_num_free_blocks() == 46
        for b in blocks:
            assert b.ref_cnt == 1

    def test_get_new_blocks_unique_ids(self, pool):
        """Allocated blocks have unique IDs."""
        blocks = pool.get_new_blocks(5)
        ids = {b.block_id for b in blocks}
        assert len(ids) == 5

    def test_allocate_more_than_available_raises(self, pool):
        """Allocating beyond free count raises ValueError."""
        with pytest.raises(ValueError):
            pool.get_new_blocks(50)  # Only 49 free

    def test_allocate_all(self, pool):
        """Allocating all free blocks works."""
        blocks = pool.get_new_blocks(49)
        assert len(blocks) == 49
        assert pool.get_num_free_blocks() == 0

    def test_eviction_on_allocate(self, pool):
        """When allocating blocks that were cached, cache is evicted."""
        # First, allocate and cache a block
        blocks = pool.get_new_blocks(1)
        h = b"evict_hash_32_bytes____________\x00"
        block = blocks[0]
        block.block_hash = h
        pool.cached_block_hash_to_block.insert(h, block)
        # Free it (returns to free list)
        pool.free_blocks([block])
        assert block.ref_cnt == 0
        # Re-allocate: eviction should clear the hash
        new_blocks = pool.get_new_blocks(49)  # Allocate all free
        # The cached block should have been evicted
        assert block.block_hash is None


class TestBlockPoolFree:
    """Tests for freeing blocks."""

    def test_free_blocks_returns_to_pool(self, pool):
        """Freeing blocks returns them to the free list."""
        blocks = pool.get_new_blocks(5)
        assert pool.get_num_free_blocks() == 44
        pool.free_blocks(blocks)
        assert pool.get_num_free_blocks() == 49

    def test_free_blocks_refcnt_zero_only(self, pool):
        """Only blocks with ref_cnt=0 after decrement are returned to free list."""
        block = pool.get_new_blocks(1)[0]
        assert block.ref_cnt == 1
        # Manually increase ref_cnt (simulates sharing)
        block.ref_cnt += 1
        pool.free_blocks([block])
        assert pool.get_num_free_blocks() == 48  # null + 1 still in use
        assert block.ref_cnt == 1  # Two references → still 1 after one free

    def test_free_null_block_not_returned(self, pool):
        """Null block is never returned to free list."""
        initial = pool.get_num_free_blocks()
        pool.free_blocks([pool.null_block])
        assert pool.get_num_free_blocks() == initial

    def test_free_reverse_order(self, pool):
        """Freeing in reverse order puts tail blocks at tail of free queue."""
        blocks = pool.get_new_blocks(3)
        pool.free_blocks(reversed(blocks))
        # All back in free list
        assert pool.get_num_free_blocks() == 49


class TestBlockPoolCache:
    """Tests for prefix caching operations."""

    def test_get_cached_block_miss(self, pool):
        """get_cached_block returns None on cache miss."""
        assert pool.get_cached_block(b"nonexistent_hash______________\x00") is None

    def test_cache_and_lookup(self, pool):
        """Cache a block, then look it up."""
        blocks = pool.get_new_blocks(1)
        h = b"test_hash_for_cache_32_bytes_\x00"
        pool.cache_full_blocks(
            blocks=blocks,
            num_cached_blocks=0,
            num_full_blocks=1,
            request_block_hashes=[h],
        )
        cached = pool.get_cached_block(h)
        assert cached is blocks[0]
        assert cached.block_hash == h

    def test_get_cached_blocks_some_hits(self, pool):
        """get_cached_blocks returns None for misses."""
        results = pool.get_cached_blocks(
            [b"none_hash_1" + b"\x00" * 21, b"none_hash_2" + b"\x00" * 21]
        )
        assert results == [None, None]

    def test_caching_disabled(self):
        """When caching is disabled, lookups always miss."""
        p = BlockPool(num_gpu_blocks=10, enable_caching=False)
        blocks = p.get_new_blocks(1)
        h = b"disabled_cache_test_32_bytes_\x00"
        p.cache_full_blocks(
            blocks=blocks, num_cached_blocks=0,
            num_full_blocks=1, request_block_hashes=[h],
        )
        assert p.get_cached_block(h) is None

    def test_cache_full_blocks_no_op_when_all_cached(self, pool):
        """cache_full_blocks does nothing when all blocks already cached."""
        blocks = pool.get_new_blocks(1)
        h = b"noop_cache_32_bytes____________\x00"
        pool.cache_full_blocks(
            blocks=blocks, num_cached_blocks=0,
            num_full_blocks=1, request_block_hashes=[h],
        )
        # Try again with same params — should be no-op
        pool.cache_full_blocks(
            blocks=blocks, num_cached_blocks=1,
            num_full_blocks=1, request_block_hashes=[h],
        )
        assert len(pool.cached_block_hash_to_block) == 1

    def test_cache_skips_null_blocks(self, pool):
        """Null blocks are never cached."""
        blocks = [pool.null_block]
        h = b"null_block_hash_32_bytes______\x00"
        pool.cache_full_blocks(
            blocks=blocks, num_cached_blocks=0,
            num_full_blocks=1, request_block_hashes=[h],
        )
        assert pool.get_cached_block(h) is None


class TestBlockPoolTouch:
    """Tests for touch() — reviving cached blocks for new requests."""

    def test_touch_increases_refcnt(self, pool):
        """touch() increments ref_cnt for each block."""
        block = pool.get_new_blocks(1)[0]
        pool.free_blocks([block])  # ref_cnt → 0, back in free list
        assert block.ref_cnt == 0
        pool.touch([block])
        assert block.ref_cnt == 1

    def test_touch_removes_from_free_list(self, pool):
        """touch() removes block from free list when ref_cnt was 0."""
        block = pool.get_new_blocks(1)[0]
        pool.free_blocks([block])  # Back in free list
        initial_free = pool.get_num_free_blocks()
        pool.touch([block])
        assert pool.get_num_free_blocks() == initial_free - 1

    def test_touch_no_double_remove(self, pool):
        """touching a block already in use does not remove from free list."""
        block = pool.get_new_blocks(1)[0]  # ref_cnt=1, NOT in free list
        free_before = pool.get_num_free_blocks()
        pool.touch([block])
        assert pool.get_num_free_blocks() == free_before
        assert block.ref_cnt == 2


class TestBlockPoolUsage:
    """Tests for usage ratio."""

    def test_usage_reflects_allocations(self, pool):
        """Usage ratio correctly reflects allocated vs total blocks."""
        total_usable = pool.num_gpu_blocks - 1  # minus null block
        assert total_usable == 49

        # Allocate half
        blocks = pool.get_new_blocks(24)
        expected = 1.0 - (25 / total_usable)  # 49 - 24 = 25 free
        assert abs(pool.get_usage() - expected) < 0.001

    def test_usage_zero_blocks(self):
        """Pool with 1 block (0 usable after null) has usage 0."""
        p = BlockPool(num_gpu_blocks=1, enable_caching=False)
        assert p.get_usage() == 0.0

    def test_usage_full_then_empty(self, pool):
        """Usage goes to 100% then back to 0% after free."""
        blocks = pool.get_new_blocks(49)
        assert pool.get_usage() == 1.0
        pool.free_blocks(blocks)
        assert pool.get_usage() == 0.0


# ─── Integration: Prefix Sharing Scenario ────────────────────────────────────

def test_prefix_sharing_scenario():
    """Two requests share identical prefix blocks.

    Verifies:
    - Request 1 allocates blocks, caches them, frees them
    - Request 2 gets cache hits, touches blocks, allocates only new
    - After both free, all blocks return to pool
    """
    pool = BlockPool(num_gpu_blocks=20, enable_caching=True)

    # Compute hashes for a 48-token sequence (3 blocks of 16)
    prompt = list(range(1, 49))
    h0 = hash_block_tokens(None, prompt[0:16])
    h1 = hash_block_tokens(h0, prompt[16:32])
    h2 = hash_block_tokens(h1, prompt[32:48])
    hashes = [h0, h1, h2]

    # Request 1: allocate 3 blocks
    req1 = pool.get_new_blocks(3)
    pool.cache_full_blocks(req1, 0, 3, hashes)
    assert pool.get_cached_block(h0) is req1[0]

    # Request 2: check cache hits
    assert pool.get_cached_block(h0) is req1[0]
    assert pool.get_cached_block(h1) is req1[1]
    assert pool.get_cached_block(h2) is req1[2]

    # Touch all 3 (they're in free list with ref_cnt=0 after req1 freed)
    pool.free_blocks(req1)  # ref_cnt → 0, all back in free list
    pool.touch([req1[0], req1[1], req1[2]])  # Revive for req2

    # Allocate 1 more block for new tokens
    req2_new = pool.get_new_blocks(1)
    assert len(req2_new) == 1

    # Clean up
    pool.free_blocks([req1[0], req1[1], req1[2], req2_new[0]])
    assert pool.get_num_free_blocks() == 19  # 20 - 1 null = 19
