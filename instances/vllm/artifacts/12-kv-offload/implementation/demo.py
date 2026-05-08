#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Demo driver — produces ≥26 verbatim numerics for the chapter narrative.

5 demos:
  1. Per-tier latency stair (HBM/DRAM/SSD): 9 numerics
  2. LRU vs ARC miss rate (3 workloads × 2 policies): 6 numerics
  3. Prefetch overlap math (decode + prefill steps): 4 numerics
  4. Pinned vs pageable bandwidth (analytic, not measured): 2 numerics
  5. End-to-end offload roundtrip (allocate→store→load→free): 5 numerics

Total: 9 + 6 + 4 + 2 + 5 = 26 verbatim numerics (≥20 floor, ≥26 target).

Numerics caveat (K17 OR-skip discipline):
  Demo 4 numbers are formula-driven (analytic alpha-beta model), NOT measured.
  Real hardware measurement would require torch + GPU; we mark these as
  "model-derived" in the narrative to avoid implying empirical claim.

Run:  python3 demo.py
"""

from __future__ import annotations

import os
import sys
import time

# Make local imports work when run as a script. We re-import the directory
# as a package by appending its parent and using `implementation.*` paths.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _PARENT)
sys.path.insert(0, _HERE)

# Try package-style first (works when run via `python3 -m implementation.demo`),
# fall back to script-style (works when run directly via `python3 demo.py`).
try:
    from implementation.offload_manager import CPUOffloadingManager
    from implementation.offload_spec import (
        DDR5_BANDWIDTH_GB_PER_S,
        DDR5_CAPACITY_GB,
        HBM3_BANDWIDTH_GB_PER_S,
        HBM_CAPACITY_GB,
        KV_BLOCK_BYTES,
        NVME_CAPACITY_GB,
        NVME_GEN5_BANDWIDTH_GB_PER_S,
        PCIE_GEN5_BANDWIDTH_GB_PER_S,
        PCIE_OVERHEAD_ALPHA_US,
        PCIE_OVERHEAD_BETA_US_PER_BYTE,
        DECODE_STEP_MS,
        PREFILL_STEP_MS,
        ReqContext,
        make_offload_key,
    )
    from implementation.policies import ARCCachePolicy, LRUCachePolicy, BlockStatus
    from implementation.cpu_gpu_worker import (
        SingleDirectionOffloadingHandler,
        alpha_beta_latency_us,
        break_even_block_bytes,
    )
    from implementation.offloading_scheduler import overlap_blocks_per_step
    from implementation.connector_taxonomy import (
        CONNECTOR_TAXONOMY,
        connectors_by_scope,
        count_by_status,
    )
except ImportError:
    # Script-mode: import each module flat, after stripping relative imports.
    # We do this by reading source text and exec'ing — too fragile. Instead,
    # do the simpler thing: mutate __package__ + add per-module sys.modules
    # aliases so the relative imports inside each .py resolve.
    import importlib.util
    import types

    # Create a synthetic package "implementation" so the inner relative
    # imports (`from .offload_spec import ...`) work.
    pkg = types.ModuleType("implementation")
    pkg.__path__ = [_HERE]
    sys.modules["implementation"] = pkg

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            f"implementation.{name}", os.path.join(_HERE, f"{name}.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"implementation.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    # Import in dependency order.
    _spec = _load("offload_spec")
    _pol = _load("policies")
    _mgr = _load("offload_manager")
    _reuse = _load("reuse_manager")
    _fac = _load("factory")
    _wkr = _load("cpu_gpu_worker")
    _simple = _load("simple_offload_manager")
    _tax = _load("connector_taxonomy")
    _sch = _load("offloading_scheduler")

    CPUOffloadingManager = _mgr.CPUOffloadingManager
    DDR5_BANDWIDTH_GB_PER_S = _spec.DDR5_BANDWIDTH_GB_PER_S
    DDR5_CAPACITY_GB = _spec.DDR5_CAPACITY_GB
    HBM3_BANDWIDTH_GB_PER_S = _spec.HBM3_BANDWIDTH_GB_PER_S
    HBM_CAPACITY_GB = _spec.HBM_CAPACITY_GB
    KV_BLOCK_BYTES = _spec.KV_BLOCK_BYTES
    NVME_CAPACITY_GB = _spec.NVME_CAPACITY_GB
    NVME_GEN5_BANDWIDTH_GB_PER_S = _spec.NVME_GEN5_BANDWIDTH_GB_PER_S
    PCIE_GEN5_BANDWIDTH_GB_PER_S = _spec.PCIE_GEN5_BANDWIDTH_GB_PER_S
    PCIE_OVERHEAD_ALPHA_US = _spec.PCIE_OVERHEAD_ALPHA_US
    PCIE_OVERHEAD_BETA_US_PER_BYTE = _spec.PCIE_OVERHEAD_BETA_US_PER_BYTE
    DECODE_STEP_MS = _spec.DECODE_STEP_MS
    PREFILL_STEP_MS = _spec.PREFILL_STEP_MS
    ReqContext = _spec.ReqContext
    make_offload_key = _spec.make_offload_key
    ARCCachePolicy = _pol.ARCCachePolicy
    LRUCachePolicy = _pol.LRUCachePolicy
    BlockStatus = _pol.BlockStatus
    SingleDirectionOffloadingHandler = _wkr.SingleDirectionOffloadingHandler
    alpha_beta_latency_us = _wkr.alpha_beta_latency_us
    break_even_block_bytes = _wkr.break_even_block_bytes
    overlap_blocks_per_step = _sch.overlap_blocks_per_step
    CONNECTOR_TAXONOMY = _tax.CONNECTOR_TAXONOMY
    connectors_by_scope = _tax.connectors_by_scope
    count_by_status = _tax.count_by_status


# Visual separator for demo output. Used in tests/demo-output.txt.
HR = "-" * 72


def header(title: str) -> None:
    print(HR)
    print(f"  {title}")
    print(HR)


# ---------------------------------------------------------------------------
# Demo 1 — per-tier latency stair (9 verbatim numerics)
# ---------------------------------------------------------------------------
def demo1_latency_stair() -> dict[str, float]:
    """Each row prints (capacity GB, bandwidth GB/s, per-16MB roundtrip µs).
    9 cells = 3 tiers × 3 columns.

    Trap A reframe: outline mentions NVMe SSD as third tier; vLLM's
    `vllm/v1/kv_offload/` is 2-tier (HBM ↔ CPU pinned) with NO NVMe path.
    We INCLUDE NVMe in the latency table for academic context (CXL /
    NVMe-over-fabric is the research direction) but the chapter pivots
    to 2-tier reality. NVMe roundtrip ≈ 2.3 ms is shown to make the
    "why not NVMe in vLLM today?" argument concrete.
    """
    header("Demo 1 — per-tier latency stair (HBM / DRAM / SSD)")

    # 9 verbatim numerics: 3 tiers × {capacity_GB, bw_GB_per_s, us_per_16MB}.
    block_bytes = KV_BLOCK_BYTES  # 16 MB
    rows = []
    for tier, cap_gb, bw_gb in (
        ("HBM3 (H100)", HBM_CAPACITY_GB, HBM3_BANDWIDTH_GB_PER_S),
        ("CPU DDR5", DDR5_CAPACITY_GB, DDR5_BANDWIDTH_GB_PER_S),
        ("NVMe Gen5", NVME_CAPACITY_GB, NVME_GEN5_BANDWIDTH_GB_PER_S),
    ):
        # us_per_16MB = block_bytes / (bw_GB_per_s * 1e3) [µs]
        us = block_bytes / (bw_gb * 1e3)
        rows.append((tier, cap_gb, bw_gb, us))
        print(
            f"  {tier:14s}  cap={cap_gb:7.1f} GB   "
            f"bw={bw_gb:7.1f} GB/s   per-16MB={us:8.2f} us"
        )

    # Bonus: PCIe-bound HBM↔DRAM roundtrip (this is the actual host-bound bw).
    pcie_us = block_bytes / (PCIE_GEN5_BANDWIDTH_GB_PER_S * 1e3)
    print(
        f"  PCIe-bound HBM<->DRAM: {pcie_us:.2f} us "
        f"(at {PCIE_GEN5_BANDWIDTH_GB_PER_S} GB/s)"
    )
    out = {
        "hbm_capacity_gb": rows[0][1],
        "hbm_bw_gbps": rows[0][2],
        "hbm_us_per_block": rows[0][3],
        "ddr5_capacity_gb": rows[1][1],
        "ddr5_bw_gbps": rows[1][2],
        "ddr5_us_per_block": rows[1][3],
        "nvme_capacity_gb": rows[2][1],
        "nvme_bw_gbps": rows[2][2],
        "nvme_us_per_block": rows[2][3],
    }
    return out


# ---------------------------------------------------------------------------
# Demo 2 — LRU vs ARC miss rate (6 verbatim numerics)
# ---------------------------------------------------------------------------
def demo2_lru_vs_arc() -> dict[str, float]:
    """Three synthetic workloads × two policies = 6 cells.

    Workloads:
      sequential — keys 0,1,2,...,N (one-shot scan, LRU should be poor)
      zipfian    — skewed towards key 0..K (ARC should be better; T2 catches HOT)
      mixed      — alternating hot+cold (ARC ghost-list signal active)

    Trap B reframe: outline §12.2 says LFU/attention-score eviction. vLLM
    ships LRU + ARC. Demo 2 walks LRU and ARC honestly, with a sidebar
    in the chapter on why LFU wasn't chosen.
    """
    header("Demo 2 — LRU vs ARC miss rate (3 workloads x 2 policies)")

    capacity = 32
    n_ops = 2000

    def run(policy_cls, ops):
        policy = policy_cls(cache_capacity=capacity)
        misses = 0
        # Mock manager-style allocate: when full, evict 1 then insert.
        next_block_id = 0
        for key in ops:
            blk = policy.get(key)
            if blk is None:
                misses += 1
                # need to make space if at capacity
                if len(policy) >= capacity:
                    evicted = policy.evict(1, protected=set())
                    # If nothing was evictable (e.g. all ref_cnt > 0), simulate
                    # an idle reset by clearing one entry directly. (In real
                    # vLLM, blocks always become idle after complete_load.)
                    if evicted is None:
                        # force one to idle
                        for k, b in (
                            list(policy.t1.items()) + list(policy.t2.items())
                            if hasattr(policy, "t1")
                            else list(policy.blocks.items())
                        )[:1]:
                            b.ref_cnt = 0
                        evicted = policy.evict(1, protected=set())
                blk = BlockStatus(block_id=next_block_id, ref_cnt=0)
                next_block_id += 1
                policy.insert(key, blk)
            else:
                # Hit — touch to refresh recency.
                policy.touch([key])
        return misses

    def make_keys(workload: str) -> list:
        keys = []
        if workload == "loop_scan":
            # 50 unique keys, scanned 40x, in order. Capacity=32 < 50 unique:
            # LRU evicts blocks just before they're hit again next loop —
            # 100% miss rate. ARC also can't help (no temporal locality
            # within capacity). This is the famous "Belady-defeating"
            # workload that motivates ARC's design.
            for _ in range(40):
                for i in range(50):
                    keys.append(make_offload_key(
                        i.to_bytes(28, "big"), 0))
        elif workload == "zipfian":
            # Heavy skew: 80% probability mass on top-12 keys (fits in cap=32);
            # 20% on a 200-key cold tail. ARC promotes hot keys to T2 quickly,
            # protects them from cold-tail churn. LRU treats all keys the same.
            import random

            random.seed(42)
            top_k = 12
            tail = 200
            for _ in range(n_ops):
                if random.random() < 0.8:
                    i = random.randint(0, top_k - 1)
                else:
                    i = random.randint(top_k, tail - 1)
                keys.append(make_offload_key(i.to_bytes(28, "big"), 0))
        elif workload == "phase_shift":
            # Two phases: 1000 ops on key set A, then 1000 ops on key set B.
            # Pure LRU is wasted by phase-2 churn evicting phase-1 cache.
            # ARC's ghost lists let the policy re-balance after the shift.
            import random

            random.seed(7)
            for i in range(1000):
                k = random.randint(0, 25)  # phase A
                keys.append(make_offload_key(k.to_bytes(28, "big"), 0))
            for i in range(1000):
                k = random.randint(50, 75)  # phase B
                keys.append(make_offload_key(k.to_bytes(28, "big"), 0))
        else:
            raise ValueError(workload)
        return keys

    out: dict[str, float] = {}
    for workload in ("loop_scan", "zipfian", "phase_shift"):
        ops = make_keys(workload)
        lru_miss = run(LRUCachePolicy, ops)
        arc_miss = run(ARCCachePolicy, ops)
        lru_rate = 100.0 * lru_miss / len(ops)
        arc_rate = 100.0 * arc_miss / len(ops)
        out[f"{workload}_lru_miss_pct"] = round(lru_rate, 2)
        out[f"{workload}_arc_miss_pct"] = round(arc_rate, 2)
        print(
            f"  {workload:11s}  LRU={lru_rate:6.2f}%   "
            f"ARC={arc_rate:6.2f}%   (n_ops={len(ops)})"
        )
    return out


# ---------------------------------------------------------------------------
# Demo 3 — prefetch overlap math (4 verbatim numerics)
# ---------------------------------------------------------------------------
def demo3_overlap() -> dict[str, float]:
    """N_overlap blocks per step for decode (50 ms) + prefill (200 ms) at PCIe Gen5.

    `N_overlap = step_compute / transfer_latency_per_block`.

    Trap C: a single 16 MB block is 250 µs at 64 GB/s. That's 0.5% of one
    decode step. PCIe is bandwidth-bound for LLM workloads, NOT latency-bound.
    """
    header("Demo 3 — prefetch overlap math (decode + prefill)")

    # 16 MB block at PCIe Gen5 = 16e6 / 64e9 s = 250 µs.
    block_bytes = KV_BLOCK_BYTES
    transfer_us = alpha_beta_latency_us(block_bytes)
    print(f"  alpha-beta transfer latency for 16 MB block: {transfer_us:.2f} us")

    n_decode = overlap_blocks_per_step(DECODE_STEP_MS, transfer_us)
    n_prefill = overlap_blocks_per_step(PREFILL_STEP_MS, transfer_us)
    print(f"  decode  step  ({DECODE_STEP_MS:.0f} ms): {n_decode:5d} blocks/step")
    print(f"  prefill step ({PREFILL_STEP_MS:.0f} ms): {n_prefill:5d} blocks/step")

    # Break-even block size (alpha == beta * bytes).
    be = break_even_block_bytes()
    print(f"  break-even block bytes (alpha==beta*bytes): {be} ({be/1024:.1f} KiB)")
    return {
        "transfer_us_per_16mb": round(transfer_us, 2),
        "n_overlap_decode": n_decode,
        "n_overlap_prefill": n_prefill,
        "break_even_bytes": be,
    }


# ---------------------------------------------------------------------------
# Demo 4 — pinned vs pageable bandwidth (analytic, 2 verbatim numerics)
# ---------------------------------------------------------------------------
def demo4_pin_bandwidth() -> dict[str, float]:
    """Analytic estimate of pinned vs pageable bandwidth ratio.

    Trap F: pinned memory is NOT free — it consumes locked DRAM that
    cannot be paged out. Over-pinning starves other system processes.
    The 2x figure is from `cuda_mem_ops.py` empirical measurements
    (driver pageable→pinned bounce halves throughput).

    K17 OR-skip discipline: these are NOT measured; they reflect the
    documented behavior in NVIDIA's CUDA C Programming Guide §3.2.6
    (Page-Locked Host Memory) and the simple_kv_offload `pin_tensor`
    docstring (which cites the bypass of CUDACachingHostAllocator).
    """
    header("Demo 4 — pinned vs pageable bandwidth (analytic, K17 OR-skip)")

    pinned_bw = PCIE_GEN5_BANDWIDTH_GB_PER_S  # 64 GB/s achievable with pinning
    pageable_bw = PCIE_GEN5_BANDWIDTH_GB_PER_S / 2.0  # ~32 GB/s, pageable bounce
    print(
        f"  pinned   H2D bandwidth (PCIe Gen5): {pinned_bw:.1f} GB/s  (full lane)"
    )
    print(
        f"  pageable H2D bandwidth (analytic):  {pageable_bw:.1f} GB/s  (~50% pinned)"
    )
    return {
        "pinned_bw_gbps": pinned_bw,
        "pageable_bw_gbps": pageable_bw,
    }


# ---------------------------------------------------------------------------
# Demo 5 — end-to-end offload roundtrip (5 verbatim numerics)
# ---------------------------------------------------------------------------
def demo5_e2e_roundtrip() -> dict[str, float]:
    """Instantiate CPUOffloadingManager + CpuGpuOffloadingHandlers + worker;
    push 100 blocks through allocate→store→load→free cycle.
    Output total wall time + per-step breakdown.

    Numerics: capacity, alloc count, store count, load count, total walltime.
    """
    header("Demo 5 — end-to-end offload roundtrip (100 blocks)")

    n_blocks = 100
    mgr = CPUOffloadingManager(num_blocks=n_blocks, cache_policy="lru", enable_events=True)
    ctx = ReqContext(kv_transfer_params=None)

    keys = [make_offload_key(i.to_bytes(28, "big"), 0) for i in range(n_blocks)]

    # 1) prepare_store all keys (all should fit, no evictions).
    t0 = time.perf_counter()
    out = mgr.prepare_store(keys, ctx)
    assert out is not None
    t_prep = time.perf_counter() - t0
    print(
        f"  prepare_store: keys_to_store={len(out.keys_to_store)}, "
        f"evicted={len(out.evicted_keys)}, t={t_prep*1e3:.2f} ms"
    )

    # 2) simulate worker upload via SingleDirectionOffloadingHandler.
    g2c = SingleDirectionOffloadingHandler(
        gpu_block_bytes=KV_BLOCK_BYTES,
        cpu_block_bytes=KV_BLOCK_BYTES,
        gpu_to_cpu=True,
    )
    g2c.transfer_async(0, (out.store_spec, out.store_spec))
    g2c.wait({0})

    # 3) complete_store (flip ref_cnt -1 → 0).
    mgr.complete_store(keys, success=True)

    # 4) prepare_load (bump ref_cnt; protect from eviction).
    load_spec = mgr.prepare_load(keys, ctx)

    # 5) simulate worker download.
    c2g = SingleDirectionOffloadingHandler(
        gpu_block_bytes=KV_BLOCK_BYTES,
        cpu_block_bytes=KV_BLOCK_BYTES,
        gpu_to_cpu=False,
    )
    c2g.transfer_async(1, (load_spec, load_spec))
    c2g.wait({1})

    # 6) complete_load (drop ref_cnt; eligible for eviction again).
    mgr.complete_load(keys)

    t_total = time.perf_counter() - t0

    # Drain events for prom-metrics surface check.
    events = list(mgr.take_events())
    n_stored = sum(1 for e in events if not e.removed)
    n_evicted = sum(1 for e in events if e.removed)

    print(f"  prepare_load: {len(load_spec.block_ids)} block ids returned")
    print(f"  num offloaded: {mgr.num_offloaded()}")
    print(f"  events: stored={n_stored} evicted={n_evicted}")
    print(f"  total wall time: {t_total*1e3:.2f} ms")
    return {
        "n_blocks": n_blocks,
        "store_event_batches": n_stored,
        "evict_event_batches": n_evicted,
        "num_offloaded_after": mgr.num_offloaded(),
        "total_walltime_ms": round(t_total * 1e3, 2),
    }


# ---------------------------------------------------------------------------
# Bonus: connector taxonomy summary (informational, doesn't add to ≥26)
# ---------------------------------------------------------------------------
def demo_connector_taxonomy() -> None:
    header("Connector taxonomy at vLLM 98661fe (Trap D anchor)")
    counts = count_by_status()
    print(f"  total connectors: {len(CONNECTOR_TAXONOMY)}")
    for status, n in sorted(counts.items()):
        print(f"  status={status:10s}: {n:2d} connectors")
    print(f"  in scope (ch12)        : {len(connectors_by_scope('ch12'))}")
    print(f"  punted to ch22-ch25    : {len(connectors_by_scope('ch22-ch25'))}")
    print(f"  research / debug       : "
          f"{len(connectors_by_scope('research')) + len(connectors_by_scope('debug'))}")


def main() -> None:
    print(HR)
    print("Ch12 KV Cache Offload — demo suite (target ≥26 verbatim numerics)")
    print(HR)
    d1 = demo1_latency_stair()
    d2 = demo2_lru_vs_arc()
    d3 = demo3_overlap()
    d4 = demo4_pin_bandwidth()
    d5 = demo5_e2e_roundtrip()
    demo_connector_taxonomy()

    # Tally: 9 + 6 + 4 + 2 + 5 = 26
    n_numerics = len(d1) + len(d2) + len(d3) + len(d4) + len(d5)
    print(HR)
    print(f"  Total verbatim numerics produced: {n_numerics}")
    print(f"  Per-demo: {len(d1)} + {len(d2)} + {len(d3)} + {len(d4)} + {len(d5)}")
    print(HR)


if __name__ == "__main__":
    main()
