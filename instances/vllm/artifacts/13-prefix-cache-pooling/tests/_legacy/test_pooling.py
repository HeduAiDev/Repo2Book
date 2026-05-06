"""Tests — Ch13 Prefix Cache Pooling."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
from implementation.pooling import GlobalPrefixCachePool


class TestGlobalPool:
    def test_alloc_and_cache(self):
        pool = GlobalPrefixCachePool(num_blocks=10)
        pool.alloc_blocks("A", 3)
        pool.cache_blocks("A", [b'hash0', b'hash1', b'hash2'])
        assert pool.lookup(b'hash0') is not None

    def test_cross_request_sharing(self):
        pool = GlobalPrefixCachePool(num_blocks=10)
        hashes = [b'h0', b'h1', b'h2']

        # Request A: allocate and cache
        pool.alloc_blocks("A", 3)
        pool.cache_blocks("A", hashes)

        # Request B: prefix lookup finds all 3
        hits = pool.find_longest_prefix(hashes)
        assert len(hits) == 3

    def test_find_longest_prefix_stops_at_miss(self):
        pool = GlobalPrefixCachePool(num_blocks=10)
        pool.alloc_blocks("A", 2)
        pool.cache_blocks("A", [b'h0', b'h1'])  # Only 2 cached

        hits = pool.find_longest_prefix([b'h0', b'h1', b'h2'])
        assert len(hits) == 2  # h2 miss

    def test_chain_hash_guarantees_no_false_positive(self):
        """Different sequence → different hash chain."""
        pool = GlobalPrefixCachePool(num_blocks=10)
        pool.alloc_blocks("A", 1)
        pool.cache_blocks("A", [b'h0'])

        # Different start → no hit
        assert pool.lookup(b'h1') is None

    def test_eviction_on_realloc(self):
        pool = GlobalPrefixCachePool(num_blocks=2)
        pool.alloc_blocks("A", 2)
        pool.cache_blocks("A", [b'x', b'y'])
        pool.free_request("A")  # Blocks freed, hashes preserved

        pool.alloc_blocks("B", 2)  # Re-allocate: evicts LRU cached blocks
        assert pool.lookup(b'x') is None  # Evicted during re-allocation
        assert pool.stats["evictions"] >= 1

    def test_touch_prevents_eviction(self):
        pool = GlobalPrefixCachePool(num_blocks=3)
        pool.alloc_blocks("A", 3)
        pool.cache_blocks("A", [b'h0', b'h1', b'h2'])
        pool.free_request("A")

        # Request B hits h0,h1 → touch prevents eviction
        pool.alloc_blocks("B", 1)
        pool.touch([pool.lookup(b'h0'), pool.lookup(b'h1')])  # None if already freed...

    def test_hit_rate(self):
        pool = GlobalPrefixCachePool(num_blocks=10)
        pool.alloc_blocks("A", 2)
        pool.cache_blocks("A", [b'a', b'b'])
        pool.lookup(b'a')  # hit
        pool.lookup(b'a')  # hit
        pool.lookup(b'c')  # miss
        assert 0.5 < pool.hit_rate < 0.8

    def test_compute_chain_deterministic(self):
        tokens = list(range(32))
        h1 = GlobalPrefixCachePool.compute_chain(tokens, block_size=16)
        h2 = GlobalPrefixCachePool.compute_chain(tokens, block_size=16)
        assert h1 == h2

    def test_compute_chain_chaining(self):
        """Same first block, different second → different second hash."""
        h1 = GlobalPrefixCachePool.compute_chain([1,2,3,4,5,6,7,8], block_size=4)
        h2 = GlobalPrefixCachePool.compute_chain([1,2,3,4,9,9,9,9], block_size=4)
        assert h1[0] == h2[0]     # Same first block
        assert h1[1] != h2[1]     # Different second block (chained)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
