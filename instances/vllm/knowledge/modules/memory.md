# Memory Knowledge

---

## K01: M01: format_gib helper is in vllm.utils, format string is '%.2f GiB'

**Module**: memory
**Chapter**: 05-memory-management
**Discovered by**: implementer
**TTL**: 2026-06-04
**Access count**: 0
**Tags**: logging, format

vllm/utils — the format_gib helper used in worker logs is `f'{bytes / (1024**3):.2f}'` (no unit suffix; the caller appends ' GiB'). Used at gpu_worker.py:L370, L434, L450, L452, L457, L462, L680. Ch05's MemoryLayout.report() uses the same precision so output diffs cleanly against vLLM logs.

---

## M02: int(80 GiB * 0.92) — float→int truncation, exact value 78920663040

**Module**: memory
**Chapter**: 05-memory-management
**Discovered by**: tester (Ch05 v6 test pass — pinning the int math)
**TTL**: permanent
**Access count**: 1
**Tags**: memory, integer-math, gotcha

`int(80 * 1024**3 * 0.92) == 78920663040` exactly. Don't write `0.92 * 80 * GIB` and expect equality — float precision matters when you're chaining `requested - weights - peak - non_torch - cudagraph` and any of those uses an `int(2.4 * GIB)`-style conversion.

The integer divisions in `get_num_blocks(available, layers, page)` then floor twice (once per // ), so the *wasted bytes* are `available - num_blocks * page * num_layers`, bounded above by `page * num_layers - 1` (~2 MiB for 32 layers, 64 KiB pages — matching the demo's "1.6 MiB wasted" line).

For testers reproducing demo numbers: assert the FINAL block count exactly (35,148), not the intermediate available_kv_cache (which is brittle to which `int()` happens where). Test_demo_num_gpu_blocks_reproduces in `test_memory_layout.py` is the pattern.

---

## M03: PreemptionScenario — recompute is "slower in latency, faster overall"

**Module**: memory
**Chapter**: 05-memory-management, 09-preemption
**Discovered by**: tester (Ch05 v6 test pass)
**TTL**: permanent
**Access count**: 1
**Tags**: preemption, recompute, swap, v1-design

For an 8K-token request (32 layers, 8 KV heads, 128 head_size, fp16): KV = 1 GiB exact. Recompute @ 50K tok/s = 163.84 ms. Swap round-trip @ 32 GiB/s PCIe = 62.5 ms. Recompute is 2.62× slower in *raw latency* — the test `test_recompute_is_slower_for_8k` asserts `recompute_is_faster is False`.

Why vLLM v1 picked recompute anyway (re-stating K06 with the test-author's framing): the latency comparison is wrong-axis. Real-world cost includes (a) CPU memory budget tracking (gone with recompute), (b) CPU-also-OOM failure modes (gone with recompute), (c) bit-determinism (recompute redoes the math, bit-identical to first run), (d) one code path vs two. The 100ms latency penalty buys 4 categories of complexity removal.

A writer explaining this should NOT lead with "recompute is faster" — it isn't. Lead with "recompute is simpler and OOM-safe", then show the 164ms-vs-62ms table to make the trade-off honest.
