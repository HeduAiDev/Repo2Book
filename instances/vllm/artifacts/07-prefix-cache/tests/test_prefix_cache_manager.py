"""Unit tests for PrefixCacheManager (match / insert / evict / touch).

Coverage:
- find_longest_cache_hit: scan-and-stop on first miss (K03)
- cache_blocks: only FULL blocks; idempotent on re-call (Trap 2)
- evict_block: removes from index; subsequent find returns shorter prefix (chain break)
- touch: bumps ref_cnt; cached blocks survive free
- free_request: returns block_ids; cached entries persist (K10)
- prefix_aware_allocate: matched + fresh order; idempotent across requests
- max_cache_hit_length: implicit at impl L210 (find_longest_cache_hit + max_length)
"""

from __future__ import annotations

from implementation.block_hash import (
    chain_block_hashes,
    make_block_hash_with_group_id,
)
from implementation.prefix_cache_manager import (
    PrefixCacheManager,
    prefix_aware_allocate,
)


class TestFindLongestCacheHit:
    """K03 chain monotonicity: scan-and-stop on first miss."""

    def test_empty_cache_returns_empty_list(self) -> None:
        mgr = PrefixCacheManager(block_size=4)
        chain = chain_block_hashes([1, 2, 3, 4, 5, 6, 7, 8], block_size=4)
        assert mgr.find_longest_cache_hit(chain, max_length=8) == []

    def test_full_match_returns_all_blocks(self) -> None:
        """Insert two blocks via cache_blocks, then find returns both."""
        mgr = PrefixCacheManager(block_size=4)
        tokens = list(range(1, 9))  # 8 tokens = 2 full blocks
        mgr.req_to_blocks["A"] = [10, 20]
        chain = chain_block_hashes(tokens, block_size=4)
        mgr.cache_blocks("A", num_tokens=8, block_hashes=chain)
        # Now look up the same chain — both blocks should hit.
        hits = mgr.find_longest_cache_hit(chain, max_length=8)
        assert hits == [10, 20]

    def test_partial_match_stops_at_first_miss(self) -> None:
        """K03: chain breaks → return prefix only.
        Insert blocks for [1..8], look up [1,2,3,4,99,99,99,99]: hit 1 block."""
        mgr = PrefixCacheManager(block_size=4)
        # Cache request A's two blocks.
        mgr.req_to_blocks["A"] = [10, 20]
        chain_a = chain_block_hashes([1, 2, 3, 4, 5, 6, 7, 8], block_size=4)
        mgr.cache_blocks("A", num_tokens=8, block_hashes=chain_a)
        # Lookup chain B (shares first block, differs at second).
        chain_b = chain_block_hashes([1, 2, 3, 4, 99, 99, 99, 99], block_size=4)
        hits = mgr.find_longest_cache_hit(chain_b, max_length=8)
        # Only block 0 hits because block 1 of B has different hash.
        assert hits == [10]

    def test_max_length_caps_lookup_window(self) -> None:
        """max_length // block_size caps how many blocks we even ask about."""
        mgr = PrefixCacheManager(block_size=4)
        mgr.req_to_blocks["A"] = [10, 20]
        chain = chain_block_hashes([1, 2, 3, 4, 5, 6, 7, 8], block_size=4)
        mgr.cache_blocks("A", num_tokens=8, block_hashes=chain)
        # Cap to 1 block worth (4 tokens) → only first hit checked.
        hits = mgr.find_longest_cache_hit(chain, max_length=4)
        assert hits == [10]


class TestCacheBlocks:
    """impl L88-L119 — insert side of the manager."""

    def test_full_block_only(self) -> None:
        """Trap 2: partial blocks are NOT cached. cache_blocks's
        num_full_blocks = num_tokens // block_size."""
        mgr = PrefixCacheManager(block_size=4)
        mgr.req_to_blocks["A"] = [10, 20]
        # 9 tokens / 4 = 2 full blocks (last 1 token partial).
        chain = chain_block_hashes(list(range(1, 10)), block_size=4)
        # chain has 2 entries (partial dropped); ask cache_blocks to cache 9 tokens.
        n = mgr.cache_blocks("A", num_tokens=9, block_hashes=chain)
        # Only 2 full blocks cached.
        assert n == 2
        assert mgr.num_cached_block["A"] == 2

    def test_idempotent_on_re_call(self) -> None:
        """impl L103-L106: re-call with same num_tokens is a no-op."""
        mgr = PrefixCacheManager(block_size=4)
        mgr.req_to_blocks["A"] = [10, 20]
        chain = chain_block_hashes([1, 2, 3, 4, 5, 6, 7, 8], block_size=4)
        n1 = mgr.cache_blocks("A", num_tokens=8, block_hashes=chain)
        n2 = mgr.cache_blocks("A", num_tokens=8, block_hashes=chain)
        assert n1 == 2
        assert n2 == 0

    def test_growth_caches_only_new_blocks(self) -> None:
        """When more tokens get cached later, only the suffix is added."""
        mgr = PrefixCacheManager(block_size=4)
        mgr.req_to_blocks["A"] = [10, 20, 30]
        chain = chain_block_hashes(list(range(1, 13)), block_size=4)
        n1 = mgr.cache_blocks("A", num_tokens=8, block_hashes=chain)
        n2 = mgr.cache_blocks("A", num_tokens=12, block_hashes=chain)
        assert n1 == 2
        assert n2 == 1
        assert mgr.num_cached_block["A"] == 3


class TestEviction:
    """K05 lazy eviction; chain-break property."""

    def test_evict_known_block_succeeds(self) -> None:
        mgr = PrefixCacheManager(block_size=4)
        mgr.req_to_blocks["A"] = [10, 20]
        chain = chain_block_hashes([1, 2, 3, 4, 5, 6, 7, 8], block_size=4)
        mgr.cache_blocks("A", num_tokens=8, block_hashes=chain)
        ok = mgr.evict_block(10, chain[0])
        assert ok is True

    def test_evict_unknown_block_returns_false(self) -> None:
        mgr = PrefixCacheManager(block_size=4)
        chain = chain_block_hashes([1, 2, 3, 4], block_size=4)
        ok = mgr.evict_block(999, chain[0])
        assert ok is False

    def test_chain_break_after_evicting_first_block(self) -> None:
        """THE prefix-cache fundamental: evict block 0 → blocks 1+ unreachable
        via chain lookup, even if block 1's KV is still in the index.

        This is what demo §3's 'D allocates 4 blocks, hit=0' verifies."""
        mgr = PrefixCacheManager(block_size=4)
        mgr.req_to_blocks["A"] = [10, 20]
        chain = chain_block_hashes([1, 2, 3, 4, 5, 6, 7, 8], block_size=4)
        mgr.cache_blocks("A", num_tokens=8, block_hashes=chain)

        # Evict block 0 (chain[0]).
        mgr.evict_block(10, chain[0])

        # Now look up the same full chain. Block 0 misses → scan-and-stop → 0 hits.
        hits = mgr.find_longest_cache_hit(chain, max_length=8)
        assert hits == []

    def test_evicting_middle_block_keeps_prefix_hit(self) -> None:
        """Eviction only affects the index; later blocks are still in the index
        but unreachable via chain lookup (because find_longest_cache_hit STARTS
        from block 0). If we evict block 1, block 0 still hits."""
        mgr = PrefixCacheManager(block_size=4)
        mgr.req_to_blocks["A"] = [10, 20]
        chain = chain_block_hashes([1, 2, 3, 4, 5, 6, 7, 8], block_size=4)
        mgr.cache_blocks("A", num_tokens=8, block_hashes=chain)

        # Evict block 1 (chain[1]).
        mgr.evict_block(20, chain[1])

        hits = mgr.find_longest_cache_hit(chain, max_length=8)
        # Block 0 still hits; block 1 misses → stops there.
        assert hits == [10]


class TestTouchAndFree:
    def test_touch_bumps_ref_cnt(self) -> None:
        mgr = PrefixCacheManager(block_size=4)
        mgr.touch([10, 20])
        assert mgr.ref_cnt[10] == 1
        assert mgr.ref_cnt[20] == 1
        mgr.touch([10])
        assert mgr.ref_cnt[10] == 2

    def test_register_request_bumps_ref_cnt(self) -> None:
        mgr = PrefixCacheManager(block_size=4)
        mgr.register_request("A", [10, 20])
        assert mgr.ref_cnt[10] == 1
        assert mgr.ref_cnt[20] == 1
        # Another request shares block 10.
        mgr.register_request("B", [10, 30])
        assert mgr.ref_cnt[10] == 2
        assert mgr.ref_cnt[30] == 1

    def test_free_request_decrements_ref_cnt(self) -> None:
        mgr = PrefixCacheManager(block_size=4)
        mgr.register_request("A", [10, 20])
        mgr.register_request("B", [10, 30])
        freed = mgr.free_request("A")
        assert freed == [10, 20]
        # Block 10 still ref_cnt=1 (B holds it); block 20 dropped to 0.
        assert mgr.ref_cnt[10] == 1
        assert mgr.ref_cnt[20] == 0

    def test_free_request_keeps_cache_entries(self) -> None:
        """K10: cached blocks survive free_request — eviction is decoupled."""
        mgr = PrefixCacheManager(block_size=4)
        mgr.req_to_blocks["A"] = [10, 20]
        chain = chain_block_hashes([1, 2, 3, 4, 5, 6, 7, 8], block_size=4)
        mgr.cache_blocks("A", num_tokens=8, block_hashes=chain)
        cache_size_before = len(mgr.cache)

        mgr.free_request("A")
        # Cache entries STAY in the index for future hits.
        assert len(mgr.cache) == cache_size_before


class TestPrefixAwareAllocate:
    def test_first_request_no_hits(self) -> None:
        mgr = PrefixCacheManager(block_size=4)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        block_ids, n_hit = prefix_aware_allocate(mgr, "A", tokens)
        assert n_hit == 0
        # 2 full blocks (8 tokens / 4) — both fresh.
        assert len(block_ids) == 2

    def test_second_request_reuses_prefix(self) -> None:
        """Demo §3: A allocates 4 blocks, B reuses 2."""
        mgr = PrefixCacheManager(block_size=4)
        # A: 16 tokens, 4 blocks.
        a_tokens = list(range(1, 17))
        a_blocks, _ = prefix_aware_allocate(mgr, "A", a_tokens)
        # B: shares first 8 tokens (= 2 blocks), differs after.
        b_tokens = list(range(1, 9)) + [101, 102, 103, 104, 105, 106, 107, 108]
        b_blocks, n_hit = prefix_aware_allocate(mgr, "B", b_tokens)
        assert n_hit == 2
        assert len(b_blocks) == 4
        # First two of B's blocks ARE A's first two.
        assert b_blocks[:2] == a_blocks[:2]

    def test_disjoint_request_no_hits(self) -> None:
        mgr = PrefixCacheManager(block_size=4)
        prefix_aware_allocate(mgr, "A", list(range(1, 17)))
        # Completely different first token.
        c_tokens = list(range(101, 117))
        _, n_hit = prefix_aware_allocate(mgr, "C", c_tokens)
        assert n_hit == 0

    def test_partial_block_not_cached(self) -> None:
        """Trap 2: 17 tokens / block_size=4 → only first 4 blocks cached."""
        mgr = PrefixCacheManager(block_size=4)
        tokens = list(range(1, 18))  # 17 tokens
        block_ids, _ = prefix_aware_allocate(mgr, "A", tokens)
        # 4 full blocks (17 // 4 = 4); 1-token remainder dropped.
        assert len(block_ids) == 4
        assert mgr.num_cached_block["A"] == 4
