"""Unit tests for BlockHashToBlockMap (the prefix-cache hash table).

K01: hash table, NOT radix tree (block_pool.py:L34-L127).
K02: append-only / no-deduplication invariant — explains the dict-of-blocks branch.

Test surface:
- get_one_block: empty → None; int branch → returns int; dict branch → any block.
- insert: 3 branches (empty / int → dict / dict).
- pop: removes specific block_id, restores entry if mismatch.
- Multi-group lookup via get_cached_block: all-must-hit semantic (K07).
"""

from __future__ import annotations

from implementation.block_hash import make_block_hash_with_group_id
from implementation.prefix_cache_index import (
    BlockHashToBlockMap,
    get_cached_block,
)


def _key(byte: int = 0, group: int = 0) -> bytes:
    """Build a (32-byte hash, group_id) packed key."""
    h = bytes([byte]) * 32
    return make_block_hash_with_group_id(h, group)


class TestGetOneBlockAndInsert:
    def test_empty_returns_none(self) -> None:
        cache = BlockHashToBlockMap()
        assert cache.get_one_block(_key()) is None

    def test_insert_int_branch(self) -> None:
        """First insert under a key stores the int directly (impl L88-L89)."""
        cache = BlockHashToBlockMap()
        cache.insert(_key(), 7)
        assert cache.get_one_block(_key()) == 7

    def test_second_insert_promotes_to_dict_branch(self) -> None:
        """K02: append-only invariant requires storing both blocks under
        the same hash key when collision happens (impl L90-L91)."""
        cache = BlockHashToBlockMap()
        cache.insert(_key(), 7)
        cache.insert(_key(), 8)
        # get_one_block returns ANY of the stored blocks — must be 7 or 8.
        got = cache.get_one_block(_key())
        assert got in {7, 8}

    def test_third_insert_extends_dict_branch(self) -> None:
        """impl L92-L93: third insert under same key extends the dict."""
        cache = BlockHashToBlockMap()
        cache.insert(_key(), 7)
        cache.insert(_key(), 8)
        cache.insert(_key(), 9)
        # Verify all three are reachable via pop.
        ids = set()
        for bid in (7, 8, 9):
            popped = cache.pop(_key(), bid)
            if popped is not None:
                ids.add(popped)
            cache.insert(_key(), bid)  # restore for later checks
        assert ids == {7, 8, 9}


class TestPop:
    def test_pop_int_branch_match(self) -> None:
        """pop on int-branch with matching block_id removes the entry."""
        cache = BlockHashToBlockMap()
        cache.insert(_key(), 7)
        assert cache.pop(_key(), 7) == 7
        assert cache.get_one_block(_key()) is None

    def test_pop_int_branch_mismatch_restores(self) -> None:
        """impl L110-L112: pop with wrong block_id restores the entry."""
        cache = BlockHashToBlockMap()
        cache.insert(_key(), 7)
        # Try to pop a different block_id — must NOT remove the real one.
        assert cache.pop(_key(), 99) is None
        assert cache.get_one_block(_key()) == 7

    def test_pop_dict_branch(self) -> None:
        """impl L113-L117: pop from dict-branch removes the specific block."""
        cache = BlockHashToBlockMap()
        cache.insert(_key(), 7)
        cache.insert(_key(), 8)
        assert cache.pop(_key(), 7) == 7
        # The other block (8) survives.
        assert cache.get_one_block(_key()) == 8

    def test_pop_dict_branch_last_block_clears_entry(self) -> None:
        """When the dict empties out, the key should be gone."""
        cache = BlockHashToBlockMap()
        cache.insert(_key(), 7)
        cache.insert(_key(), 8)
        cache.pop(_key(), 7)
        cache.pop(_key(), 8)
        assert cache.get_one_block(_key()) is None
        assert _key() not in cache

    def test_pop_missing_key_returns_none(self) -> None:
        cache = BlockHashToBlockMap()
        assert cache.pop(_key(), 7) is None


class TestContainsAndLen:
    def test_len(self) -> None:
        cache = BlockHashToBlockMap()
        assert len(cache) == 0
        cache.insert(_key(0), 1)
        cache.insert(_key(1), 2)
        assert len(cache) == 2

    def test_contains(self) -> None:
        cache = BlockHashToBlockMap()
        cache.insert(_key(), 1)
        assert _key() in cache
        assert _key(99) not in cache


class TestGetCachedBlock:
    """K07: multi-group all-must-hit semantic."""

    def test_single_group_hit(self) -> None:
        cache = BlockHashToBlockMap()
        block_hash = bytes([5]) * 32
        cache.insert(make_block_hash_with_group_id(block_hash, 0), 42)
        result = get_cached_block(cache, block_hash, [0])
        assert result == [42]

    def test_single_group_miss(self) -> None:
        cache = BlockHashToBlockMap()
        block_hash = bytes([5]) * 32
        # Nothing inserted.
        assert get_cached_block(cache, block_hash, [0]) is None

    def test_multi_group_all_hit(self) -> None:
        """K07: when all groups have the block, returns list of block_ids per group."""
        cache = BlockHashToBlockMap()
        block_hash = bytes([5]) * 32
        cache.insert(make_block_hash_with_group_id(block_hash, 0), 42)
        cache.insert(make_block_hash_with_group_id(block_hash, 1), 43)
        result = get_cached_block(cache, block_hash, [0, 1])
        assert result == [42, 43]

    def test_multi_group_one_misses_returns_none(self) -> None:
        """K07: vLLM L184-L209 — None if ANY group misses."""
        cache = BlockHashToBlockMap()
        block_hash = bytes([5]) * 32
        cache.insert(make_block_hash_with_group_id(block_hash, 0), 42)
        # Group 1 NOT inserted.
        result = get_cached_block(cache, block_hash, [0, 1])
        assert result is None
