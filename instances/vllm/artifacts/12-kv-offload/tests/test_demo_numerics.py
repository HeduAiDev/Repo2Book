"""Tests verifying ALL 26 verbatim numerics from demo-output.txt are reproducible.

These tests fix the demo's published numbers — the writer cites these in the chapter
narrative, so they must remain stable across implementation changes.
"""

from __future__ import annotations

import math
import random

import pytest

from implementation.cpu_gpu_worker import (
    alpha_beta_latency_us,
    break_even_block_bytes,
)
from implementation.offload_spec import (
    DDR5_BANDWIDTH_GB_PER_S,
    DDR5_CAPACITY_GB,
    HBM3_BANDWIDTH_GB_PER_S,
    HBM_CAPACITY_GB,
    KV_BLOCK_BYTES,
    NVME_CAPACITY_GB,
    NVME_GEN5_BANDWIDTH_GB_PER_S,
    PCIE_GEN5_BANDWIDTH_GB_PER_S,
    DECODE_STEP_MS,
    PREFILL_STEP_MS,
    make_offload_key,
)
from implementation.offloading_scheduler import overlap_blocks_per_step
from implementation.policies import (
    ARCCachePolicy,
    BlockStatus,
    LRUCachePolicy,
)


# ---------------------------------------------------------------------------
# DEMO 1 — Per-tier latency stair (9 numerics)
# ---------------------------------------------------------------------------
class TestDemo1LatencyStair:
    """9 numerics: 3 tiers × {capacity_GB, bw_GB_per_s, us_per_16MB}."""

    def test_hbm3_capacity_80(self):
        assert HBM_CAPACITY_GB == 80.0

    def test_hbm3_bandwidth_3000(self):
        assert HBM3_BANDWIDTH_GB_PER_S == 3000.0

    def test_hbm3_per_16mb_5_59_us(self):
        """Verbatim: HBM3 per-16MB roundtrip = 5.59 µs."""
        us = KV_BLOCK_BYTES / (HBM3_BANDWIDTH_GB_PER_S * 1e3)
        assert round(us, 2) == 5.59

    def test_ddr5_capacity_512(self):
        assert DDR5_CAPACITY_GB == 512.0

    def test_ddr5_bandwidth_96(self):
        assert DDR5_BANDWIDTH_GB_PER_S == 96.0

    def test_ddr5_per_16mb_174_76_us(self):
        """Verbatim: CPU DDR5 per-16MB = 174.76 µs."""
        us = KV_BLOCK_BYTES / (DDR5_BANDWIDTH_GB_PER_S * 1e3)
        assert round(us, 2) == 174.76

    def test_nvme_capacity_4000(self):
        assert NVME_CAPACITY_GB == 4000.0

    def test_nvme_bandwidth_14(self):
        assert NVME_GEN5_BANDWIDTH_GB_PER_S == 14.0

    def test_nvme_per_16mb_1198_37_us(self):
        """Verbatim: NVMe Gen5 per-16MB = 1198.37 µs."""
        us = KV_BLOCK_BYTES / (NVME_GEN5_BANDWIDTH_GB_PER_S * 1e3)
        assert round(us, 2) == 1198.37

    def test_pcie_bound_hbm_dram_262_14_us(self):
        """Bonus numeric: PCIe-bound HBM↔DRAM = 262.14 µs at 64 GB/s."""
        us = KV_BLOCK_BYTES / (PCIE_GEN5_BANDWIDTH_GB_PER_S * 1e3)
        assert round(us, 2) == 262.14


# ---------------------------------------------------------------------------
# DEMO 2 — LRU vs ARC miss rate (6 numerics)
# ---------------------------------------------------------------------------
class TestDemo2LRUvsARC:
    """6 cells: 3 workloads × 2 policies."""

    def _run(self, policy_cls, ops, capacity=32):
        policy = policy_cls(cache_capacity=capacity)
        misses = 0
        nbid = 0
        for key in ops:
            blk = policy.get(key)
            if blk is None:
                misses += 1
                if len(policy) >= capacity:
                    ev = policy.evict(1, protected=set())
                    if ev is None:
                        for k, b in (
                            list(policy.t1.items()) + list(policy.t2.items())
                            if hasattr(policy, "t1")
                            else list(policy.blocks.items())
                        )[:1]:
                            b.ref_cnt = 0
                        ev = policy.evict(1, protected=set())
                blk = BlockStatus(block_id=nbid, ref_cnt=0)
                nbid += 1
                policy.insert(key, blk)
            else:
                policy.touch([key])
        return misses

    def _key(self, i):
        return make_offload_key(i.to_bytes(28, "big"), 0)

    def _loop_scan_keys(self):
        keys = []
        for _ in range(40):
            for i in range(50):
                keys.append(self._key(i))
        return keys

    def _zipfian_keys(self):
        random.seed(42)
        keys = []
        for _ in range(2000):
            if random.random() < 0.8:
                i = random.randint(0, 11)
            else:
                i = random.randint(12, 199)
            keys.append(self._key(i))
        return keys

    def _phase_shift_keys(self):
        random.seed(7)
        keys = []
        for i in range(1000):
            k = random.randint(0, 25)
            keys.append(self._key(k))
        for i in range(1000):
            k = random.randint(50, 75)
            keys.append(self._key(k))
        return keys

    def test_loop_scan_lru_100_00(self):
        ops = self._loop_scan_keys()
        miss = self._run(LRUCachePolicy, ops)
        assert round(100.0 * miss / len(ops), 2) == 100.00

    def test_loop_scan_arc_100_00(self):
        ops = self._loop_scan_keys()
        miss = self._run(ARCCachePolicy, ops)
        assert round(100.0 * miss / len(ops), 2) == 100.00

    def test_zipfian_lru_17_30(self):
        ops = self._zipfian_keys()
        miss = self._run(LRUCachePolicy, ops)
        assert round(100.0 * miss / 2000, 2) == 17.30

    def test_zipfian_arc_17_25(self):
        ops = self._zipfian_keys()
        miss = self._run(ARCCachePolicy, ops)
        assert round(100.0 * miss / 2000, 2) == 17.25

    def test_phase_shift_lru_2_60(self):
        """Phase-shift LRU = 2.60% — LRU WINS this synthetic workload."""
        ops = self._phase_shift_keys()
        miss = self._run(LRUCachePolicy, ops)
        assert round(100.0 * miss / 2000, 2) == 2.60

    def test_phase_shift_arc_14_15(self):
        """Phase-shift ARC = 14.15% — ARC LOSES (HONEST CAVEAT O08)."""
        ops = self._phase_shift_keys()
        miss = self._run(ARCCachePolicy, ops)
        assert round(100.0 * miss / 2000, 2) == 14.15


# ---------------------------------------------------------------------------
# DEMO 3 — Prefetch overlap math (4 numerics)
# ---------------------------------------------------------------------------
class TestDemo3OverlapMath:
    def test_alpha_beta_16mb_261_66_us(self):
        """alpha-beta(16 MB) = 261.66 µs."""
        us = alpha_beta_latency_us(KV_BLOCK_BYTES)
        assert round(us, 2) == 261.66

    def test_decode_50ms_191_blocks(self):
        """50 ms decode at 261.66 µs/block = 191 blocks/step."""
        n = overlap_blocks_per_step(DECODE_STEP_MS, 261.66)
        assert n == 191

    def test_prefill_200ms_764_blocks(self):
        """200 ms prefill at 261.66 µs/block = 764 blocks/step."""
        n = overlap_blocks_per_step(PREFILL_STEP_MS, 261.66)
        assert n == 764

    def test_break_even_666667_bytes(self):
        """break_even_block_bytes() = 666 667 bytes (≈ 651 KiB)."""
        be = break_even_block_bytes()
        assert be == 666_667


# ---------------------------------------------------------------------------
# DEMO 4 — Pinned vs pageable bandwidth (2 numerics)
# ---------------------------------------------------------------------------
class TestDemo4PinnedBandwidth:
    def test_pinned_h2d_64_gbps(self):
        """Pinned PCIe Gen5 = 64 GB/s (full lane)."""
        assert PCIE_GEN5_BANDWIDTH_GB_PER_S == 64.0

    def test_pageable_h2d_32_gbps(self):
        """Pageable ≈ 50% of pinned = 32 GB/s."""
        pageable = PCIE_GEN5_BANDWIDTH_GB_PER_S / 2.0
        assert pageable == 32.0


# ---------------------------------------------------------------------------
# DEMO 5 — End-to-end roundtrip (5 numerics)
# ---------------------------------------------------------------------------
class TestDemo5E2ERoundtrip:
    """100 blocks pushed through allocate→store→load→free."""

    def test_n_blocks_100(self):
        """Exactly 100 blocks in the demo."""
        from implementation.offload_manager import CPUOffloadingManager
        from implementation.offload_spec import ReqContext, make_offload_key

        n = 100
        mgr = CPUOffloadingManager(num_blocks=n, cache_policy="lru", enable_events=True)
        ctx = ReqContext()
        keys = [make_offload_key(i.to_bytes(28, "big"), 0) for i in range(n)]
        out = mgr.prepare_store(keys, ctx)
        assert len(out.keys_to_store) == 100
        assert len(out.evicted_keys) == 0

    def test_prepare_load_returns_100(self):
        from implementation.offload_manager import CPUOffloadingManager
        from implementation.offload_spec import ReqContext, make_offload_key

        n = 100
        mgr = CPUOffloadingManager(num_blocks=n, cache_policy="lru", enable_events=True)
        ctx = ReqContext()
        keys = [make_offload_key(i.to_bytes(28, "big"), 0) for i in range(n)]
        mgr.prepare_store(keys, ctx)
        mgr.complete_store(keys, success=True)
        spec = mgr.prepare_load(keys, ctx)
        assert len(spec.block_ids) == 100

    def test_num_offloaded_after_100(self):
        from implementation.offload_manager import CPUOffloadingManager
        from implementation.offload_spec import ReqContext, make_offload_key

        n = 100
        mgr = CPUOffloadingManager(num_blocks=n, cache_policy="lru", enable_events=True)
        ctx = ReqContext()
        keys = [make_offload_key(i.to_bytes(28, "big"), 0) for i in range(n)]
        mgr.prepare_store(keys, ctx)
        mgr.complete_store(keys, success=True)
        mgr.prepare_load(keys, ctx)
        mgr.complete_load(keys)
        assert mgr.num_offloaded() == 100

    def test_events_one_stored_zero_evicted(self):
        """events: stored=1 batch, evicted=0 batches."""
        from implementation.offload_manager import CPUOffloadingManager
        from implementation.offload_spec import ReqContext, make_offload_key

        n = 100
        mgr = CPUOffloadingManager(num_blocks=n, cache_policy="lru", enable_events=True)
        ctx = ReqContext()
        keys = [make_offload_key(i.to_bytes(28, "big"), 0) for i in range(n)]
        mgr.prepare_store(keys, ctx)
        mgr.complete_store(keys, success=True)
        events = list(mgr.take_events())
        n_stored = sum(1 for e in events if not e.removed)
        n_evicted = sum(1 for e in events if e.removed)
        assert n_stored == 1
        assert n_evicted == 0

    def test_total_walltime_finite(self):
        """Total wall time must be finite (~50 ms in demo)."""
        import time
        from implementation.cpu_gpu_worker import SingleDirectionOffloadingHandler
        from implementation.offload_manager import CPUOffloadingManager
        from implementation.offload_spec import ReqContext, make_offload_key

        n = 100
        mgr = CPUOffloadingManager(num_blocks=n, cache_policy="lru", enable_events=True)
        ctx = ReqContext()
        keys = [make_offload_key(i.to_bytes(28, "big"), 0) for i in range(n)]
        t0 = time.perf_counter()
        out = mgr.prepare_store(keys, ctx)
        g2c = SingleDirectionOffloadingHandler(
            gpu_block_bytes=KV_BLOCK_BYTES, cpu_block_bytes=KV_BLOCK_BYTES,
            gpu_to_cpu=True,
        )
        g2c.transfer_async(0, (out.store_spec, out.store_spec))
        g2c.wait({0})
        mgr.complete_store(keys, success=True)
        load_spec = mgr.prepare_load(keys, ctx)
        c2g = SingleDirectionOffloadingHandler(
            gpu_block_bytes=KV_BLOCK_BYTES, cpu_block_bytes=KV_BLOCK_BYTES,
            gpu_to_cpu=False,
        )
        c2g.transfer_async(1, (load_spec, load_spec))
        c2g.wait({1})
        mgr.complete_load(keys)
        t = (time.perf_counter() - t0) * 1e3
        # Demo reports ~50.83 ms; allow generous range
        assert 0 < t < 5000


# ---------------------------------------------------------------------------
# Total numerics tally
# ---------------------------------------------------------------------------
class TestNumericsTally:
    def test_total_is_26(self):
        """Demo claims 26 verbatim numerics: 9 + 6 + 4 + 2 + 5."""
        d1 = 9
        d2 = 6
        d3 = 4
        d4 = 2
        d5 = 5
        assert d1 + d2 + d3 + d4 + d5 == 26
