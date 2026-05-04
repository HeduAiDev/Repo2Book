"""Tests — Ch7 Prefix Cache."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
from implementation.prefix_cache import (
    ChainedBlockHasher, PrefixCacheIndex, PrefixCacheManager,
)


class TestChainedBlockHasher:
    def test_different_prefixes_different_hashes(self):
        hasher = ChainedBlockHasher(block_size=4)
        h1 = hasher.compute_hashes([1, 2, 3, 4, 5, 6, 7, 8])
        h2 = hasher.compute_hashes([9, 9, 9, 9, 5, 6, 7, 8])
        # First block hashes should differ because first 4 tokens differ
        assert h1[0] != h2[0]
        # Second block hashes must also differ (chained — different parent)
        assert h1[1] != h2[1]

    def test_same_prefix_same_hashes(self):
        hasher = ChainedBlockHasher(block_size=4)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        h1 = hasher.compute_hashes(tokens)
        h2 = hasher.compute_hashes(tokens)
        assert h1 == h2

    def test_chained_property(self):
        """Hash of block K depends on hash of block K-1."""
        hasher = ChainedBlockHasher(block_size=4)
        h1 = hasher.compute_hashes([1, 2, 3, 4, 5, 6, 7, 8])
        h2 = hasher.compute_hashes([1, 2, 3, 4, 9, 9, 9, 9])
        assert h1[0] == h2[0]  # Same first block
        assert h1[1] != h2[1]  # Different second block (chained divergence)

    def test_partial_block_not_hashed(self):
        hasher = ChainedBlockHasher(block_size=4)
        h = hasher.compute_hashes([1, 2, 3, 4, 5])  # 5 tokens = 1 complete block
        assert len(h) == 1


class TestPrefixCacheIndex:
    def test_insert_and_get(self):
        idx = PrefixCacheIndex()
        idx.insert(b'hash_abc', 42)
        assert idx.get(b'hash_abc') == 42
        assert idx.get(b'hash_missing') is None

    def test_collision_handling(self):
        """Two blocks with same hash → dict upgrade, get returns any."""
        idx = PrefixCacheIndex()
        idx.insert(b'hash', 10)
        idx.insert(b'hash', 20)  # collision
        result = idx.get(b'hash')
        assert result in (10, 20)

    def test_remove_single(self):
        idx = PrefixCacheIndex()
        idx.insert(b'hash', 42)
        idx.remove(b'hash', 42)
        assert idx.get(b'hash') is None

    def test_remove_from_collision_keeps_other(self):
        idx = PrefixCacheIndex()
        idx.insert(b'hash', 10)
        idx.insert(b'hash', 20)
        idx.remove(b'hash', 10)
        assert idx.get(b'hash') == 20


class TestPrefixCacheManager:
    def test_find_cache_hit(self):
        cache = PrefixCacheManager(block_size=4)
        hasher = ChainedBlockHasher(block_size=4)

        # Register and cache request A
        tokens_a = list(range(12))  # 3 blocks
        hashes_a = hasher.compute_hashes(tokens_a)
        cache.register_request("A", [0, 1, 2])
        cache.cache_blocks("A", num_computed_tokens=12, block_hashes=hashes_a)

        # Request B: same first 2 blocks (8 tokens), different tail
        tokens_b = tokens_a[:8] + [99, 99, 99, 99]
        hashes_b = hasher.compute_hashes(tokens_b)
        hits, num_tokens = cache.find_longest_cache_hit("B", hashes_b)
        assert len(hits) == 2
        assert num_tokens == 8

    def test_no_cache_hit_for_different_prefix(self):
        cache = PrefixCacheManager(block_size=4)
        hasher = ChainedBlockHasher(block_size=4)
        cache.register_request("A", [0, 1, 2])
        cache.cache_blocks("A", num_computed_tokens=12,
                          block_hashes=hasher.compute_hashes(list(range(12))))

        # Different tokens entirely
        hashes_b = hasher.compute_hashes([99] * 12)
        hits, _ = cache.find_longest_cache_hit("B", hashes_b)
        assert len(hits) == 0

    def test_cache_blocks_skips_already_cached(self):
        cache = PrefixCacheManager(block_size=4)
        hasher = ChainedBlockHasher(block_size=4)
        tokens = list(range(12))
        hashes = hasher.compute_hashes(tokens)
        cache.register_request("A", [0, 1, 2])
        cache.cache_blocks("A", num_computed_tokens=8, block_hashes=hashes)
        # Cache again with same count — should be no-op
        cache.cache_blocks("A", num_computed_tokens=8, block_hashes=hashes)
        assert cache.req_cached_count["A"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
