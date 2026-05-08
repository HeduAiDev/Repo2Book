"""End-to-end integration tests — full offload → match → reuse cycle."""

from __future__ import annotations

import pytest

from implementation.cpu_gpu_worker import (
    CpuGpuOffloadingHandlers,
    OffloadingWorker,
    SingleDirectionOffloadingHandler,
)
from implementation.offload_manager import CPUOffloadingManager
from implementation.offload_spec import (
    CanonicalKVCacheTensor,
    CanonicalKVCaches,
    CPULoadStoreSpec,
    CPUOffloadingSpec,
    GPULoadStoreSpec,
    KV_BLOCK_BYTES,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from implementation.offloading_scheduler import (
    OffloadingConnectorScheduler,
    SchedulerOffloadConfig,
    _SimpleRequest,
)


def _bh(i: int) -> bytes:
    return bytes([i % 256]) * 28


def _key(i: int, group: int = 0) -> OffloadKey:
    return make_offload_key(_bh(i), group)


@pytest.fixture
def ctx():
    return ReqContext()


# ---------------------------------------------------------------------------
# Full lifecycle: store → complete → lookup → load → free
# ---------------------------------------------------------------------------
class TestFullLifecycle:
    def test_full_roundtrip_lru(self, ctx):
        """Demo 5 contract: 100 blocks through allocate→store→load→free."""
        n = 100
        mgr = CPUOffloadingManager(num_blocks=n, cache_policy="lru", enable_events=True)
        keys = [_key(i) for i in range(n)]

        # 1) prepare_store
        out = mgr.prepare_store(keys, ctx)
        assert out is not None
        assert len(out.keys_to_store) == n
        assert len(out.evicted_keys) == 0
        assert isinstance(out.store_spec, CPULoadStoreSpec)
        assert len(out.store_spec.block_ids) == n

        # 2) (worker uploads — simulated below)
        g2c = SingleDirectionOffloadingHandler(
            gpu_block_bytes=KV_BLOCK_BYTES, cpu_block_bytes=KV_BLOCK_BYTES,
            gpu_to_cpu=True,
        )
        g2c.transfer_async(0, (out.store_spec, out.store_spec))
        g2c.wait({0})

        # 3) complete_store
        mgr.complete_store(keys, success=True)
        assert mgr.lookup(keys[0], ctx) is True

        # 4) prepare_load
        load_spec = mgr.prepare_load(keys, ctx)
        assert len(load_spec.block_ids) == n

        # 5) (worker downloads)
        c2g = SingleDirectionOffloadingHandler(
            gpu_block_bytes=KV_BLOCK_BYTES, cpu_block_bytes=KV_BLOCK_BYTES,
            gpu_to_cpu=False,
        )
        c2g.transfer_async(1, (load_spec, load_spec))
        c2g.wait({1})

        # 6) complete_load
        mgr.complete_load(keys)
        assert mgr.num_offloaded() == n

        # 7) events
        events = list(mgr.take_events())
        n_stored = sum(1 for e in events if not e.removed)
        n_evicted = sum(1 for e in events if e.removed)
        assert n_stored == 1  # one batch
        assert n_evicted == 0

    def test_full_roundtrip_arc(self, ctx):
        """Same lifecycle on ARC policy — verifies pluggability."""
        n = 50
        mgr = CPUOffloadingManager(num_blocks=n, cache_policy="arc")
        keys = [_key(i) for i in range(n)]
        out = mgr.prepare_store(keys, ctx)
        mgr.complete_store(keys, success=True)
        # All should land in T1 (recent partition)
        for k in keys:
            assert k in mgr._policy.t1
        # Read each — promotes to T2
        spec = mgr.prepare_load(keys, ctx)
        mgr.complete_load(keys)
        # Touch promotes T1 → T2
        mgr.touch(keys)
        for k in keys:
            assert k in mgr._policy.t2


# ---------------------------------------------------------------------------
# Spec → Manager → Worker integration
# ---------------------------------------------------------------------------
class TestSpecManagerWorker:
    def test_spec_get_handlers_yields_two_directions(self):
        """CPUOffloadingSpec.get_handlers yields BOTH (G→C) and (C→G)."""
        spec = CPUOffloadingSpec(
            hash_block_size=16,
            gpu_block_size=(16,),
            kv_bytes_per_block=4096,
            cpu_bytes_to_use=4096 * 32,
        )
        caches = CanonicalKVCaches(
            tensors=[CanonicalKVCacheTensor(tensor=None, page_size_bytes=4096)],
            group_data_refs=[[]],
        )
        handlers_pairs = list(spec.get_handlers(caches))
        # Two yielded pairs: G→C and C→G
        assert len(handlers_pairs) == 2
        transfer_types = sorted([
            (src.medium(), dst.medium()) for src, dst, _ in handlers_pairs
        ])
        assert transfer_types == [("CPU", "GPU"), ("GPU", "CPU")]

    def test_spec_handlers_register_into_worker(self):
        """get_handlers output can wire into OffloadingWorker."""
        spec = CPUOffloadingSpec(
            hash_block_size=16,
            gpu_block_size=(16,),
            kv_bytes_per_block=4096,
            cpu_bytes_to_use=4096 * 16,
        )
        caches = CanonicalKVCaches(
            tensors=[CanonicalKVCacheTensor(tensor=None, page_size_bytes=4096)],
            group_data_refs=[[]],
        )
        worker = OffloadingWorker()
        for src, dst, h in spec.get_handlers(caches):
            worker.register_handler(src, dst, h)
        # Both transfer types registered
        assert ("GPU", "CPU") in worker.transfer_type_to_handler
        assert ("CPU", "GPU") in worker.transfer_type_to_handler


# ---------------------------------------------------------------------------
# Scheduler ↔ Manager integration
# ---------------------------------------------------------------------------
class TestSchedulerManagerIntegration:
    def _setup(self, num_blocks=16):
        mgr = CPUOffloadingManager(num_blocks=num_blocks, cache_policy="lru")
        cfg = SchedulerOffloadConfig.from_groups(
            gpu_block_sizes=(16,),
            block_size_factor=1,
            hash_block_size=16,
        )
        sched = OffloadingConnectorScheduler(manager=mgr, config=cfg)
        return sched, mgr

    def test_scheduler_uses_manager_lookup(self, ctx):
        """Scheduler defers all hit decisions to manager.lookup."""
        sched, mgr = self._setup()
        # populate keyspace
        keys = [make_offload_key(_bh(i), 0) for i in range(3)]
        mgr.prepare_store(keys, ctx)
        mgr.complete_store(keys, success=True)

        req = _SimpleRequest(
            request_id="r1",
            block_hashes=[_bh(i) for i in range(3)],
            num_tokens=48,
        )
        n, _ = sched.get_num_new_matched_tokens(req, num_computed_tokens=0)
        assert n == 48  # 3 blocks × 16 tokens

    def test_scheduler_touch_after_hit(self, ctx):
        """get_num_new_matched_tokens calls touch on the hit blocks (drives LRU)."""
        sched, mgr = self._setup()
        keys = [make_offload_key(_bh(i), 0) for i in range(2)]
        mgr.prepare_store(keys, ctx)
        mgr.complete_store(keys, success=True)
        # Insert another block then verify ordering moves
        extra = [make_offload_key(_bh(99), 0)]
        mgr.prepare_store(extra, ctx)
        mgr.complete_store(extra, success=True)

        # Scheduler matching keys 0, 1 should touch them — they become MRU
        req = _SimpleRequest(
            request_id="r1",
            block_hashes=[_bh(i) for i in range(2)],
            num_tokens=32,
        )
        sched.get_num_new_matched_tokens(req, num_computed_tokens=0)

        # After touch, _key(99) should be the LRU candidate
        evicted = mgr._policy.evict(1, set())
        assert evicted is not None
        assert evicted[0][0] == extra[0]


# ---------------------------------------------------------------------------
# Eviction triggers from cross-component pressure
# ---------------------------------------------------------------------------
class TestEvictionPressure:
    def test_pressure_triggers_evictions(self, ctx):
        """When capacity is exceeded, prepare_store proactively evicts and reports."""
        mgr = CPUOffloadingManager(num_blocks=4, enable_events=True)
        # Fill capacity
        a = [_key(i) for i in range(4)]
        mgr.prepare_store(a, ctx)
        mgr.complete_store(a, success=True)
        # Now store 4 more — must evict 4
        b = [_key(i + 100) for i in range(4)]
        out = mgr.prepare_store(b, ctx)
        assert out is not None
        assert len(out.evicted_keys) == 4

    def test_partial_evict_when_partial_capacity(self, ctx):
        """Evict only the deficit, not more."""
        mgr = CPUOffloadingManager(num_blocks=10)
        # Fill 8/10
        first = [_key(i) for i in range(8)]
        mgr.prepare_store(first, ctx)
        mgr.complete_store(first, success=True)
        # Add 4 more — only 2 must be evicted
        more = [_key(i + 100) for i in range(4)]
        out = mgr.prepare_store(more, ctx)
        assert out is not None
        assert len(out.evicted_keys) == 2
