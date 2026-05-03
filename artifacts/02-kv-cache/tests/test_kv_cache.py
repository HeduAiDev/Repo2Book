"""Tests — Chapter 2 KV Cache (vLLM-grounded)."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
from implementation.kv_cache import (
    KVCacheBlock, FreeKVCacheBlockQueue, BlockPool, KVCacheBlocks,
    KVCacheManager, KVCacheConfig,
)


class TestKVCacheBlock:
    def test_basic(self):
        b = KVCacheBlock(block_id=5)
        assert b.block_id == 5
        assert b.ref_cnt == 0
        assert b.block_hash is None

    def test_hash_lifecycle(self):
        b = KVCacheBlock(block_id=0)
        b.set_hash(42)
        assert b.block_hash == 42
        b.reset_hash()
        assert b.block_hash is None


class TestFreeKVCacheBlockQueue:
    def test_basic_ops(self):
        blocks = [KVCacheBlock(i) for i in range(5)]
        q = FreeKVCacheBlockQueue(blocks)
        assert q.num_free_blocks == 5
        b = q.popleft()
        assert b.block_id == 0  # first in = first out (LRU)
        assert q.num_free_blocks == 4

    def test_popleft_n(self):
        q = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(10)])
        popped = q.popleft_n(3)
        assert [b.block_id for b in popped] == [0, 1, 2]
        assert q.num_free_blocks == 7

    def test_append_makes_mru(self):
        # Append → tail (most recently used), won't be popped first
        q = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(3)])
        q.append(KVCacheBlock(block_id=99))
        assert q.popleft().block_id == 0       # old block
        assert q.popleft().block_id == 1
        assert q.popleft().block_id == 2
        assert q.popleft().block_id == 99      # appended last → popped last

    def test_remove_from_middle(self):
        blocks = [KVCacheBlock(i) for i in range(5)]
        q = FreeKVCacheBlockQueue(blocks)
        q.remove(blocks[2])  # remove block_id=2
        assert q.num_free_blocks == 4
        popped = [q.popleft().block_id for _ in range(4)]
        assert 2 not in popped


class TestBlockPool:
    def test_get_new_blocks(self):
        pool = BlockPool(num_gpu_blocks=10, enable_caching=True)
        blocks = pool.get_new_blocks(3)
        assert len(blocks) == 3
        assert all(b.ref_cnt == 1 for b in blocks)
        assert pool.free_queue.num_free_blocks == 7

    def test_oom(self):
        pool = BlockPool(num_gpu_blocks=5)
        pool.get_new_blocks(5)
        with pytest.raises(RuntimeError):
            pool.get_new_blocks(1)

    def test_cache_and_lookup(self):
        pool = BlockPool(num_gpu_blocks=10)
        blocks = pool.get_new_blocks(2)
        pool.cache_full_blocks(blocks, [100, 200])
        assert pool.get_cached_block(100) is blocks[0]
        assert pool.get_cached_block(200) is blocks[1]
        assert pool.get_cached_block(999) is None

    def test_evict_on_reallocate(self):
        pool = BlockPool(num_gpu_blocks=2)
        b = pool.get_new_blocks(1)         # allocate block 0
        pool.cache_full_blocks(b, [42])    # mark as cached
        pool.free_blocks(b)                # free back (ref_cnt→0), hash preserved
        assert pool.get_cached_block(42) is not None  # still in prefix cache

        # Allocate ALL free blocks: LRU first → block 1 (never cached)
        b2 = pool.get_new_blocks(1)        # pops block 1 (not block 0 yet)
        assert pool.get_cached_block(42) is not None  # block 0's hash still intact

        # Now only block 0 is free. Allocate it → evicts hash.
        b3 = pool.get_new_blocks(1)        # pops block 0, evicts hash
        assert b3[0].block_hash is None
        assert pool.get_cached_block(42) is None     # evicted!

    def test_touch_increments_refcnt(self):
        pool = BlockPool(num_gpu_blocks=10)
        b = pool.get_new_blocks(1)
        pool.cache_full_blocks(b, [77])
        pool.touch(b)  # another request hits this cached block
        assert b[0].ref_cnt == 2
        pool.free_blocks(b)
        assert b[0].ref_cnt == 1  # still in use by the other request

    def test_usage(self):
        pool = BlockPool(num_gpu_blocks=10)
        assert pool.get_usage() == 0.0
        pool.get_new_blocks(5)
        assert pool.get_usage() == 0.5


class TestKVCacheManager:
    def test_allocate_slots(self):
        mgr = KVCacheManager(num_gpu_blocks=100, block_size=16)
        result = mgr.allocate_slots("req-1", num_new_tokens=32)
        assert result is not None
        assert len(result.get_block_ids()) == 2  # ceil(32/16)=2

    def test_oom_returns_none(self):
        mgr = KVCacheManager(num_gpu_blocks=2, block_size=16)
        result = mgr.allocate_slots("req-1", num_new_tokens=16)
        assert result is not None  # 1 block needed, 2 available → success
        result2 = mgr.allocate_slots("req-2", num_new_tokens=32)
        assert result2 is None  # 2 blocks needed, only 1 free → OOM

    def test_get_computed_blocks_prefix_cache(self):
        mgr = KVCacheManager(num_gpu_blocks=10, block_size=16, enable_caching=True)
        # First request: allocate and cache
        blocks = mgr.allocate_slots("req-A", num_new_tokens=16)
        assert blocks is not None
        block_ids = blocks.get_block_ids()
        mgr.cache_blocks("req-A", block_hashes=[42])

        # Second request: should find cached block via prefix hash lookup
        cached = mgr.get_computed_blocks("req-B", block_hashes=[42])
        assert len(cached.get_block_ids()) == 1

    def test_free_returns_blocks(self):
        mgr = KVCacheManager(num_gpu_blocks=10, block_size=16)
        assert mgr.block_pool.get_num_free_blocks() == 10
        mgr.allocate_slots("req-1", num_new_tokens=32)
        assert mgr.block_pool.get_num_free_blocks() == 8  # 2 blocks used
        mgr.free("req-1")
        assert mgr.block_pool.get_num_free_blocks() == 10  # all returned

    def test_usage_tracks_allocation(self):
        mgr = KVCacheManager(num_gpu_blocks=100, block_size=16)
        assert mgr.get_usage() == 0.0
        mgr.allocate_slots("req-1", num_new_tokens=160)  # 10 blocks
        assert mgr.get_usage() > 0.09


class TestKVCacheConfig:
    def test_num_blocks(self):
        config = KVCacheConfig(num_layers=1, num_kv_heads=2, head_dim=4,
                               block_size=16, dtype_bytes=2)
        n = config.calculate_num_blocks(total_gpu_memory=1000000,
                                        model_weight_size=100000)
        # available = 0.9 * 1M - 100K = 800K
        # block_bytes = 2*16*2*4*2*1 = 512
        # n = 800K/512 = 1562
        assert n > 0

    def test_per_token_kv_bytes(self):
        config = KVCacheConfig(num_layers=2, num_kv_heads=4, head_dim=8,
                               block_size=16, dtype_bytes=2)
        expected = 2 * 4 * 8 * 2 * 2  # 2(KV) * kv_heads * head_dim * dtype * layers
        assert config.per_token_kv_bytes() == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
