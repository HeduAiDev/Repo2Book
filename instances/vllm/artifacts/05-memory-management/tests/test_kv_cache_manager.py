"""Tests for kv_cache_manager.py — scheduler-facing KV cache interface."""

import pytest
from implementation.block_pool import BlockPool, hash_block_tokens
from implementation.kv_cache_manager import (
    KVCacheManager,
    KVCacheBlocks,
    Request,
    compute_request_block_hashes,
)


# ─── KVCacheBlocks ───────────────────────────────────────────────────────────

class TestKVCacheBlocks:
    """Tests for the KVCacheBlocks allocation result data class."""

    def test_get_block_ids_single_group(self):
        """get_block_ids returns block IDs flattened across groups."""
        blocks = tuple([_make_block(1), _make_block(3)])
        kcb = KVCacheBlocks((list(blocks),))
        assert kcb.get_block_ids() == [1, 3]

    def test_new_empty(self):
        """new_empty preserves group structure with empty blocks."""
        original = KVCacheBlocks(([_make_block(1)],))
        empty = original.new_empty()
        assert empty.blocks == ((),)

    def test_empty_construction(self):
        """Empty KVCacheBlocks has zero block IDs."""
        kcb = KVCacheBlocks(((),))
        assert kcb.get_block_ids() == []


# ─── Request ─────────────────────────────────────────────────────────────────

class TestRequest:
    """Tests for the minimal Request data class."""

    def test_num_tokens_auto_set(self):
        """num_tokens is auto-set from prompt_token_ids in __post_init__."""
        req = Request(
            request_id="test-1",
            prompt_token_ids=[1, 2, 3, 4, 5],
            max_tokens=50,
        )
        assert req.num_tokens == 5

    def test_default_fields(self):
        """Default field values are correct."""
        req = Request(
            request_id="test-1",
            prompt_token_ids=[],
            max_tokens=10,
        )
        assert req.num_computed_tokens == 0
        assert req.block_hashes == []
        assert req.skip_reading_prefix_cache is False
        assert req.num_preemptions == 0


# ─── compute_request_block_hashes ────────────────────────────────────────────

class TestComputeRequestBlockHashes:
    """Tests for request block hash computation."""

    def test_full_blocks(self):
        """48 tokens → 3 blocks of 16 tokens → 3 hashes."""
        tokens = list(range(48))
        hashes = compute_request_block_hashes(tokens, block_size=16)
        assert len(hashes) == 3

    def test_deterministic(self):
        """Same input → same hashes."""
        tokens = list(range(32))
        h1 = compute_request_block_hashes(tokens)
        h2 = compute_request_block_hashes(tokens)
        assert h1 == h2

    def test_partial_last_block_not_hashed(self):
        """If token count not divisible by block_size, last partial block is skipped."""
        tokens = list(range(40))  # 2 full blocks + 8 partial
        hashes = compute_request_block_hashes(tokens, block_size=16)
        assert len(hashes) == 2

    def test_different_tokens_different_hashes(self):
        """Different token sequences produce different last hashes."""
        h1 = compute_request_block_hashes(list(range(32)))
        h2 = compute_request_block_hashes(list(range(1, 33)))
        assert h1[-1] != h2[-1]

    def test_empty_input(self):
        """Empty token list → no hashes."""
        hashes = compute_request_block_hashes([])
        assert hashes == []


# ─── KVCacheManager ──────────────────────────────────────────────────────────

@pytest.fixture
def manager():
    """KVCacheManager with 256-block pool, caching enabled."""
    pool = BlockPool(num_gpu_blocks=256, enable_caching=True)
    return KVCacheManager(
        block_pool=pool,
        block_size=16,
        max_model_len=4096,
        enable_caching=True,
    )


@pytest.fixture
def request_factory():
    """Helper to create Request objects for tests."""

    def _make(request_id, num_prompt_tokens):
        tokens = list(range(1, num_prompt_tokens + 1))
        hashes = compute_request_block_hashes(tokens, block_size=16)
        req = Request(
            request_id=request_id,
            prompt_token_ids=tokens,
            max_tokens=100,
        )
        req.block_hashes = hashes
        return req

    return _make


class TestKVCacheManagerInit:
    """Tests for manager initialization."""

    def test_initial_usage(self, manager):
        """New manager has 0% usage."""
        assert manager.usage == 0.0

    def test_free_blocks_accounts_for_null(self, manager):
        """Free blocks = total - 1 (null block)."""
        assert manager.get_num_free_blocks() == 255

    def test_caching_disabled_mode(self):
        """Manager can be constructed without caching."""
        pool = BlockPool(num_gpu_blocks=50, enable_caching=False)
        mgr = KVCacheManager(
            block_pool=pool,
            block_size=16,
            max_model_len=4096,
            enable_caching=False,
        )
        assert mgr.enable_caching is False


class TestGetComputedBlocks:
    """Tests for prefix cache lookup."""

    def test_miss_on_empty_cache(self, manager, request_factory):
        """First request on empty cache → miss."""
        req = request_factory("r1", 48)
        cached, hit_tokens = manager.get_computed_blocks(req)
        assert hit_tokens == 0
        assert cached.get_block_ids() == []

    def test_hit_after_caching(self, manager, request_factory):
        """After allocating and caching, subsequent lookups hit.

        For 48 tokens: max_cache_hit_length = 48 - 1 = 47 (last token must
        be recomputed for logits). max_num_blocks = 47 // 16 = 2 blocks.
        So hit_tokens = 2 * 16 = 32 (blocks 0 and 1 are scanned).
        """
        req_a = request_factory("a", 48)
        # Allocate and cache request A
        result = manager.allocate_slots(
            request=req_a,
            num_new_tokens=48,
        )
        assert result is not None
        req_a.num_computed_tokens = 48

        # Request B: same tokens → should hit
        req_b = request_factory("b", 48)
        cached_b, hit_tokens_b = manager.get_computed_blocks(req_b)
        # max_cache_hit_length = 47 → 47//16 = 2 blocks → 32 tokens
        assert hit_tokens_b == 32
        assert cached_b.get_block_ids()

    def test_nothing_to_compute_zero_tokens(self, manager, request_factory):
        """Request with num_tokens=0 → max_cache_hit_length = -1."""
        req = Request(
            request_id="empty",
            prompt_token_ids=[],
            max_tokens=10,
        )
        req.block_hashes = []
        cached, hit_tokens = manager.get_computed_blocks(req)
        assert hit_tokens == 0

    def test_skip_reading_prefix_cache(self, manager, request_factory):
        """When skip_reading_prefix_cache is set, always miss."""
        req = request_factory("r1", 48)
        req.skip_reading_prefix_cache = True
        # First cache the tokens normally
        result = manager.allocate_slots(request=req, num_new_tokens=48)
        assert result is not None
        req.num_computed_tokens = 48

        # Now request with skip flag
        req2 = request_factory("r2", 48)
        req2.skip_reading_prefix_cache = True
        cached, hit_tokens = manager.get_computed_blocks(req2)
        assert hit_tokens == 0


class TestAllocateSlots:
    """Tests for the 3-stage allocate_slots flow."""

    def test_basic_allocation(self, manager, request_factory):
        """Allocating 48 tokens for a new request returns 3 new block IDs.

        48 tokens / 16 block_size = 3 blocks. All are newly allocated since
        this is the first request, so get_block_ids() returns 3 IDs.
        """
        req = request_factory("r1", 48)
        result = manager.allocate_slots(request=req, num_new_tokens=48)
        assert result is not None
        alloc_ids = result.get_block_ids()
        assert len(alloc_ids) == 3

    def test_allocation_tracks_request(self, manager, request_factory):
        """After allocation, manager associates blocks with the request."""
        req = request_factory("r1", 32)  # 2 blocks
        result = manager.allocate_slots(request=req, num_new_tokens=32)
        assert result is not None
        req_blocks = manager.get_blocks("r1")
        assert len(req_blocks) == 2

    def test_oom_returns_none(self, manager, request_factory):
        """When there are not enough free blocks, allocate_slots returns None."""
        # Request way more tokens than pool can handle
        req = request_factory("r1", 100000)
        result = manager.allocate_slots(
            request=req,
            num_new_tokens=100000,
        )
        assert result is None

    def test_oom_with_small_pool(self, request_factory):
        """With a tiny pool, allocation fails predictably."""
        pool = BlockPool(num_gpu_blocks=3, enable_caching=True)
        mgr = KVCacheManager(block_pool=pool, block_size=16, max_model_len=4096)
        req = request_factory("r1", 64)  # 4 blocks
        result = mgr.allocate_slots(request=req, num_new_tokens=64)
        assert result is None  # Pool can't fit 4 blocks

    def test_prefix_cache_saves_blocks(self, manager, request_factory):
        """Request with prefix cache hit allocates fewer new blocks."""
        # Request A: full allocation
        req_a = request_factory("a", 48)  # 3 blocks
        result_a = manager.allocate_slots(request=req_a, num_new_tokens=48)
        assert result_a is not None
        req_a.num_computed_tokens = 48

        # Request B: same 48 + 16 new tokens
        req_b = request_factory("b", 64)  # 4 blocks total
        cached, hit_tokens = manager.get_computed_blocks(req_b)
        assert hit_tokens == 48  # All 3 prefix blocks cached

        # Should only need 1 new block for the 16 new tokens
        free_before = manager.get_num_free_blocks()
        result_b = manager.allocate_slots(
            request=req_b,
            num_new_tokens=req_b.num_tokens - hit_tokens,
            num_new_computed_tokens=hit_tokens,
            new_computed_blocks=cached if hit_tokens > 0 else None,
        )
        assert result_b is not None
        # Only 1 new block allocated (the 4th block)
        assert len(result_b.get_block_ids()) == 1
        # Total blocks for req_b should be 4
        assert len(manager.get_blocks("b")) == 4

    def test_allocate_slots_zero_tokens_raises(self, manager, request_factory):
        """allocate_slots with num_new_tokens=0 raises ValueError."""
        req = request_factory("r1", 16)
        with pytest.raises(ValueError, match="num_new_tokens must be > 0"):
            manager.allocate_slots(request=req, num_new_tokens=0)

    def test_allocate_with_computed_blocks(self, manager, request_factory):
        """Allocating with pre-computed blocks (cache hits) touches them."""
        req_a = request_factory("a", 32)  # 2 blocks
        result = manager.allocate_slots(request=req_a, num_new_tokens=32)
        assert result is not None
        req_a.num_computed_tokens = 32

        req_b = request_factory("b", 48)
        cached, hit_tokens = manager.get_computed_blocks(req_b)
        assert hit_tokens == 32  # 2 blocks cached
        result_b = manager.allocate_slots(
            request=req_b,
            num_new_tokens=req_b.num_tokens - hit_tokens,
            num_new_computed_tokens=hit_tokens,
            new_computed_blocks=cached if hit_tokens > 0 else None,
        )
        assert result_b is not None
        # Only 1 new block (48 tokens = 3 blocks, 2 cached → 1 new)
        assert len(result_b.get_block_ids()) == 1


class TestFindLongestCacheHit:
    """Tests for the left-to-right cache hit scanning algorithm."""

    def test_all_hit(self, manager, request_factory):
        """All blocks cached within max_cache_hit_length → full hit up to limit.

        For 48 tokens: max_cache_hit_length = 48 - 1 = 47 (last token must
        be recomputed). 47 // 16 = 2 blocks scannable → hit_tokens = 32.
        """
        req_a = request_factory("a", 48)
        manager.allocate_slots(request=req_a, num_new_tokens=48)
        req_a.num_computed_tokens = 48

        req_b = request_factory("b", 48)
        blocks, hit_tokens = manager.find_longest_cache_hit(
            req_b.block_hashes,
            req_b.num_tokens - 1,
        )
        assert hit_tokens == 32  # max_cache_hit_length // block_size = 2 blocks
        assert len(blocks) == 2

    def test_partial_hit_first_block_miss(self, manager, request_factory):
        """When first block misses, zero hits returned."""
        req = request_factory("req", 48)
        blocks, hit_tokens = manager.find_longest_cache_hit(
            req.block_hashes,
            47,  # max_cache_hit_length
        )
        assert hit_tokens == 0
        assert blocks == []

    def test_hit_stops_at_first_miss(self, manager, request_factory):
        """Left-to-right scan stops at first cache miss."""
        # Cache only the first block
        req_a = request_factory("a", 16)  # 1 block
        manager.allocate_slots(request=req_a, num_new_tokens=16)
        req_a.num_computed_tokens = 16

        # Request with 3 blocks — only first should hit
        req_b = request_factory("b", 48)
        blocks, hit_tokens = manager.find_longest_cache_hit(
            req_b.block_hashes,
            req_b.num_tokens - 1,
        )
        assert hit_tokens == 16  # Only first block cached
        assert len(blocks) == 1

    def test_max_cache_hit_length_caps_result(self, manager, request_factory):
        """max_cache_hit_length limits how many blocks are scanned."""
        req_a = request_factory("a", 48)
        manager.allocate_slots(request=req_a, num_new_tokens=48)
        req_a.num_computed_tokens = 48

        req_b = request_factory("b", 48)
        blocks, hit_tokens = manager.find_longest_cache_hit(
            req_b.block_hashes,
            max_cache_hit_length=16,  # Only 1 block worth of scanning
        )
        assert hit_tokens <= 16
        assert len(blocks) <= 1


class TestFree:
    """Tests for freeing requests."""

    def test_free_removes_request(self, manager, request_factory):
        """Free removes all blocks and clears request tracking."""
        req = request_factory("r1", 32)
        manager.allocate_slots(request=req, num_new_tokens=32)

        assert "r1" in manager.req_to_blocks
        manager.free("r1")
        assert manager.get_blocks("r1") == []
        assert "r1" not in manager.num_cached_block

    def test_free_unknown_request_noop(self, manager):
        """Freeing an unknown request ID is a no-op."""
        manager.free("nonexistent")  # Should not raise

    def test_free_restores_free_blocks(self, manager, request_factory):
        """After freeing, blocks are available again."""
        free_before = manager.get_num_free_blocks()
        req = request_factory("r1", 32)
        manager.allocate_slots(request=req, num_new_tokens=32)
        assert manager.get_num_free_blocks() < free_before
        manager.free("r1")
        assert manager.get_num_free_blocks() == free_before


class TestGetMethods:
    """Tests for query methods."""

    def test_get_blocks_empty_for_unknown_request(self, manager):
        """get_blocks returns empty list for unknown request."""
        assert manager.get_blocks("unknown") == []

    def test_get_block_ids_empty_for_unknown_request(self, manager):
        """get_block_ids returns empty list for unknown request."""
        assert manager.get_block_ids("unknown") == []

    def test_get_blocks_after_allocation(self, manager, request_factory):
        """get_blocks returns all blocks allocated to a request."""
        req = request_factory("r1", 32)
        manager.allocate_slots(request=req, num_new_tokens=32)
        blocks = manager.get_blocks("r1")
        assert len(blocks) == 2

    def test_get_block_ids_after_allocation(self, manager, request_factory):
        """get_block_ids returns numeric block IDs."""
        req = request_factory("r1", 32)
        manager.allocate_slots(request=req, num_new_tokens=32)
        ids = manager.get_block_ids("r1")
        assert len(ids) == 2
        assert all(isinstance(bid, int) for bid in ids)


class TestRemoveSkippedBlocks:
    """Tests for sliding window mechanism (full attention → no-op)."""

    def test_full_attention_no_skip(self, manager, request_factory):
        """get_num_skipped_tokens returns 0 for full attention."""
        assert manager.get_num_skipped_tokens(100) == 0

    def test_remove_skipped_blocks_noop(self, manager, request_factory):
        """remove_skipped_blocks is no-op for full attention."""
        req = request_factory("r1", 32)
        manager.allocate_slots(request=req, num_new_tokens=32)
        free_before = manager.get_num_free_blocks()
        manager.remove_skipped_blocks("r1", total_computed_tokens=32)
        assert manager.get_num_free_blocks() == free_before


class TestCacheBlocks:
    """Tests for cache_blocks marking blocks for prefix reuse."""

    def test_cache_blocks_marks_blocks(self, manager, request_factory):
        """After cache_blocks, blocks are stored in hash index."""
        req = request_factory("r1", 32)
        manager.allocate_slots(request=req, num_new_tokens=32)
        # Blocks should be cached during allocate_slots (via cache_blocks)
        assert "r1" in manager.num_cached_block
        assert manager.num_cached_block["r1"] == 2  # 2 full blocks

    def test_cache_disabled_no_cache(self, request_factory):
        """When caching is disabled, blocks are not cached."""
        pool = BlockPool(num_gpu_blocks=50, enable_caching=False)
        mgr = KVCacheManager(
            block_pool=pool, block_size=16,
            max_model_len=4096, enable_caching=False,
        )
        req = request_factory("r1", 32)
        result = mgr.allocate_slots(request=req, num_new_tokens=32)
        assert result is not None
        # No cached blocks tracked
        assert "r1" not in mgr.num_cached_block


# ─── Integration: Full Request Lifecycle ─────────────────────────────────────

def test_request_lifecycle_prefix_sharing():
    """Full lifecycle: two requests sharing prefix, one finishes, the other survives.

    This is the core demo scenario from KVCacheManager.main():
    - Request A: 48-token prompt → 3 blocks, cached
    - Request B: same 48 + 16 new tokens → hits 3 cached, allocates 1 new
    - Request A freed → shared blocks survive (ref_cnt > 0)
    """
    pool = BlockPool(num_gpu_blocks=256, enable_caching=True)
    mgr = KVCacheManager(
        block_pool=pool, block_size=16,
        max_model_len=4096, enable_caching=True,
    )

    # Request A: 48 tokens
    prompt_a = list(range(1, 49))
    block_hashes_a = compute_request_block_hashes(prompt_a)
    req_a = Request(
        request_id="req-a",
        prompt_token_ids=prompt_a,
        max_tokens=100,
    )
    req_a.block_hashes = block_hashes_a

    cached_a, hit_a = mgr.get_computed_blocks(req_a)
    assert hit_a == 0  # Empty cache

    result_a = mgr.allocate_slots(
        request=req_a, num_new_tokens=req_a.num_tokens,
    )
    assert result_a is not None
    assert len(mgr.get_blocks("req-a")) == 3

    # Request B: 64 tokens (48 same prefix + 16 new)
    prompt_b = list(range(1, 65))
    block_hashes_b = compute_request_block_hashes(prompt_b)
    req_b = Request(
        request_id="req-b",
        prompt_token_ids=prompt_b,
        max_tokens=100,
    )
    req_b.block_hashes = block_hashes_b

    cached_b, hit_b = mgr.get_computed_blocks(req_b)
    assert hit_b == 48  # 3 cached blocks

    result_b = mgr.allocate_slots(
        request=req_b,
        num_new_tokens=req_b.num_tokens - hit_b,
        num_new_computed_tokens=hit_b,
        new_computed_blocks=cached_b if hit_b > 0 else None,
    )
    assert result_b is not None
    # 4 total blocks: 3 from cache + 1 new
    assert len(mgr.get_blocks("req-b")) == 4
    assert len(result_b.get_block_ids()) == 1  # Only 1 newly allocated

    # Shared blocks have ref_cnt >= 2
    all_b = mgr.get_blocks("req-b")
    assert all_b[0].ref_cnt >= 2  # Shared by req-a and req-b

    # Free Request A
    free_before = mgr.get_num_free_blocks()
    mgr.free("req-a")
    # Shared blocks (ref_cnt > 1) should NOT return to free list yet
    # Only the unique tail blocks of A (if any) are freed
    # In this case, all 3 blocks are shared, so none returned
    assert mgr.get_num_free_blocks() >= free_before

    # Free Request B
    mgr.free("req-b")
    # Now all blocks should be free
    assert mgr.get_num_free_blocks() == 255  # 256 - 1 null


def test_oom_recovery_cycle():
    """Verify pool can recover from OOM by freeing a request."""
    pool = BlockPool(num_gpu_blocks=20, enable_caching=True)
    mgr = KVCacheManager(
        block_pool=pool, block_size=16,
        max_model_len=4096, enable_caching=True,
    )

    # Fill pool with a big request (18 blocks = 288 tokens)
    tokens = list(range(288))
    hashes = compute_request_block_hashes(tokens)
    req = Request(
        request_id="big",
        prompt_token_ids=tokens,
        max_tokens=100,
    )
    req.block_hashes = hashes

    result = mgr.allocate_slots(request=req, num_new_tokens=288)
    assert result is not None

    # Try allocating more — should fail
    tokens2 = list(range(288, 320))  # 32 tokens = 2 blocks
    hashes2 = compute_request_block_hashes(tokens2)
    req2 = Request(
        request_id="small",
        prompt_token_ids=tokens2,
        max_tokens=10,
    )
    req2.block_hashes = hashes2

    result2 = mgr.allocate_slots(request=req2, num_new_tokens=len(tokens2))
    assert result2 is None  # OOM

    # Free the big request and retry
    mgr.free("big")
    result3 = mgr.allocate_slots(request=req2, num_new_tokens=len(tokens2))
    assert result3 is not None  # Should work now


# ─── Helpers ─────────────────────────────────────────────────────────────────

from implementation.block_pool import KVCacheBlock as _KVCacheBlock


def _make_block(block_id: int) -> _KVCacheBlock:
    """Create a KVCacheBlock with the given ID."""
    return _KVCacheBlock(block_id=block_id)
