"""Tests — Ch12 KV Cache Offload."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
from implementation.kv_offload import (
    OffloadBlock, LRUPolicy, CPUOffloadingManager, pcie_bandwidth_analysis,
)


class TestOffloadBlock:
    def test_new_block_not_ready(self):
        b = OffloadBlock(block_id=0)
        assert not b.is_ready

    def test_complete_store_makes_ready(self):
        b = OffloadBlock(block_id=0)
        b.ref_cnt = 0
        assert b.is_ready


class TestLRUPolicy:
    def test_insert_and_get(self):
        p = LRUPolicy()
        p.insert(b'key1', OffloadBlock(block_id=0, ref_cnt=0, key=b'key1'))
        assert p.get(b'key1') is not None
        assert p.get(b'missing') is None

    def test_not_ready_block_not_returned(self):
        p = LRUPolicy()
        p.insert(b'key1', OffloadBlock(block_id=0, ref_cnt=-1, key=b'key1'))
        assert p.get(b'key1') is None  # Not ready

    def test_evict_lru_first(self):
        p = LRUPolicy()
        for i in range(3):
            p.insert(f'key{i}'.encode(), OffloadBlock(block_id=i, ref_cnt=0))

        evicted = p.evict(1, protected=set())
        assert len(evicted) == 1
        assert evicted[0].key == b'key0'  # First inserted = LRU

    def test_touch_moves_to_mru(self):
        p = LRUPolicy()
        for i in range(3):
            p.insert(f'key{i}'.encode(), OffloadBlock(block_id=i, ref_cnt=0))

        p.touch([b'key0'])  # Touch key0 → moves to end
        evicted = p.evict(1, protected=set())
        assert evicted[0].key == b'key1'  # key1 now LRU

    def test_protected_not_evicted(self):
        p = LRUPolicy()
        for i in range(3):
            p.insert(f'key{i}'.encode(), OffloadBlock(block_id=i, ref_cnt=0))
        evicted = p.evict(3, protected={b'key1'})
        assert len(evicted) == 2
        assert b'key1' not in [e.key for e in evicted]


class TestCPUOffloadingManager:
    def test_store_and_lookup(self):
        mgr = CPUOffloadingManager(num_blocks=10)
        mgr.prepare_store([b'hash_a'])
        assert not mgr.lookup(b'hash_a')  # Not ready yet
        mgr.complete_store([b'hash_a'])
        assert mgr.lookup(b'hash_a')

    def test_eviction_on_full(self):
        mgr = CPUOffloadingManager(num_blocks=3)
        mgr.prepare_store([b'a', b'b', b'c'])
        mgr.complete_store([b'a', b'b', b'c'])

        result = mgr.prepare_store([b'd'])
        assert len(result['evicted']) >= 1  # Should evict LRU (key 'a')

    def test_load_protection(self):
        mgr = CPUOffloadingManager(num_blocks=5)
        mgr.prepare_store([b'x'])
        mgr.complete_store([b'x'])

        mgr.prepare_load([b'x'])  # ref_cnt: 0 → 1
        mgr.complete_load([b'x'])  # ref_cnt: 1 → 0


class TestPCIEBandwidth:
    def test_reasonable_offload_time(self):
        r = pcie_bandwidth_analysis(32768, 1, 128, 64)
        assert r["per_decode_step_us"] < 100

    def test_gen5_faster_than_gen4(self):
        r4 = pcie_bandwidth_analysis(32768, 1, 128, 64, pcie_generation=4)
        r5 = pcie_bandwidth_analysis(32768, 1, 128, 64, pcie_generation=5)
        assert r5["full_seq_offload_ms"] < r4["full_seq_offload_ms"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
