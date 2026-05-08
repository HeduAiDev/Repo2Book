# Knowledge — kv-offload module (O-prefix)

Repo-specific facts about vLLM's KV cache offload subsystem.

**Module scope**: `vllm/v1/kv_offload/`, `vllm/v1/simple_kv_offload/`,
`vllm/distributed/kv_transfer/kv_connector/v1/` (18 connectors).

**Source pin**: vLLM commit `98661fe012c5c467252d4df8411d2f46190e9268`.

**Prefix**: `O` — chosen to avoid collision with K (kv-cache), T
(tensor-parallelism), E (expert-parallelism), P (preemption), M
(multi-token-prediction), W (wisdom-related), D (dcp-pcp), C
(continuous-batching).

---

## O01 — "no class X" series stays at N=5 within vllm

**Discovered by**: implementer (Ch12, 2026-05-08)
**Use by**: book-editor + writer when framing chapter outline reframes

The Ch07-Ch11 streak of "no class X" framing tools (5 instances) does
NOT extend to Ch12. KV offload at 98661fe HAS many concrete classes:
`OffloadingManager`, `CPUOffloadingManager`, 18 KVConnectorBase_V1
subclasses, `OffloadingHandler`, `CpuGpuOffloadingHandlers`, `BlockStatus`,
`SharedOffloadRegion`, `CanonicalKVCaches`, etc. The reader will not
be misled by these names.

**Implication**: Don't force a 6th instance. The streak should not be
artificially preserved by reframing where the source clearly has the
classes. This honestly grounds the motif as "framing tool for
outline-vs-source mismatches", not "every chapter must have one".

## O02 — Outline §12.2 LFU/attention-score reframe to LRU+ARC

**Discovered by**: implementer (Ch12)
**Use by**: writer §12.2; reviewer when checking source claims

vLLM at 98661fe ships `cpu/policies/lru.py` (46 LOC) + `cpu/policies/arc.py`
(156 LOC) only. There is **NO `lfu.py`**, **NO attention-score policy
file**. The outline mentions "LRU/LFU/attention-score-based"; the chapter
walks LRU + ARC honestly.

This is the **3rd in-chapter outline correction** within vllm:
1. Ch11 §11.4 AG+RS-vs-A2A
2. Ch11 §11.6 5D-not-3D-mesh
3. Ch12 §12.2 LFU/attention-score → LRU+ARC

**Pattern**: leave outline JSON unchanged; reframe at chapter level. The
outline is a TOPIC guide (what to teach about); source dictates HOW we
teach it. Per `feedback_outline_topic_not_contract`.

## O03 — Three-way distinction: Connector / Manager / Spec

**Discovered by**: implementer (Ch12)
**Use by**: writer when explaining the data flow

The KV offload subsystem has THREE abstractions that often get confused:
- `KVConnectorBase_V1` (`base.py:L170-L660`): orchestrates the LIFECYCLE
  (start_load_kv, wait_for_layer_load, save_kv_layer, build_connector_meta).
  One per role per process.
- `OffloadingManager` (`base.py:L110-L218`): owns the KEYSPACE (lookup,
  prepare_load, prepare_store, ref_cnt, eviction). Scheduler-side only.
- `OffloadingSpec` (`base.py:L319-L398`): factory for the above two
  (`get_manager()` + `get_handlers()`). One per offload backend.

The `OffloadingConnector` instantiates the spec, asks for the manager
(scheduler role) or handlers (worker role), then drives the lifecycle.
**Confusing the three is Trap-D-adjacent**.

## O04 — Offload ≠ swap (Trap A canonical)

**Discovered by**: implementer (Ch12)
**Use by**: writer §12.1 + glossary

SWAP (Ch05 §5.7) is **preempt-and-recompute**: blocks are evicted
entirely; sequence rolls back to last computed token; new forward pass
recomputes KV. OFFLOAD is **demote-and-rehydrate**: blocks move down
the storage tier but request stays logically alive; rehydration is
async I/O, not compute.

`vllm/v1/core/sched/scheduler.py` swap path lives in the preemption
infrastructure. `vllm/v1/kv_offload/cpu/manager.py:L91-L103` prepare_load
is the offload counterpart. Different code paths.

## O05 — Pinned memory bandwidth ≈ 2× pageable

**Discovered by**: implementer (Ch12)
**Use by**: writer §12.5 (pin memory + async copy)

Pinned (page-locked) host memory bypasses the CUDA driver's pageable→pinned
bounce buffer. Without pinning, every async copy stages through a driver
intermediate, halving effective H2D bandwidth. With pinning, full PCIe
Gen5 ×16 = 64 GB/s is achievable.

**System cost**: pinned DRAM is locked, cannot be paged out. PyTorch's
`pin_memory=True` rounds to next power of 2 (100 GB → 128 GB pinned).
`vllm/v1/simple_kv_offload/cuda_mem_ops.py:L16-L25` bypasses via
`cudaHostRegister(tensor.data_ptr(), tensor.nbytes, 0)` — no rounding.

## O06 — Per-tier bandwidth latency math

**Discovered by**: implementer (Ch12)
**Use by**: writer §12.1 + Demo 1

| Tier | Capacity | Bandwidth | Per-16MB roundtrip |
|---|---|---|---|
| HBM3 (H100) | 80 GB | 3000 GB/s | 5.59 µs |
| CPU DDR5 | 512 GB | 96 GB/s | 174.76 µs |
| NVMe Gen5 | 4 TB | 14 GB/s | 1198.37 µs |
| PCIe-bound HBM↔DRAM | (host link) | 64 GB/s | 262.14 µs |

Block_bytes = 16 MB. The PCIe-bound number (262 µs) is the actual
host-bound H↔D transfer time — it's the LIMITING factor, not DDR5's
own 174 µs at-DIMM bandwidth.

Decode step ~50 ms on H100. One PCIe block roundtrip = 0.5% of step.

## O07 — Reactive prefix lookup, NOT predictive prefetch

**Discovered by**: implementer (Ch12)
**Use by**: writer §12.3 (Trap E)

`OffloadingConnectorScheduler.get_num_new_matched_tokens` (`offloading/scheduler.py:L443-L486`)
is a **deterministic prefix scan**. It calls `_maximal_prefix_lookup`
(L244-L261) which iterates `block_hashes` linearly, asking
`manager.lookup(key)` for each. NO ML model. NO Markov chain. NO
prediction.

The `return None` defer-lookup path lets backends pipeline cache-line
warming (e.g. LMCache RPC), but the warming is internal to the backend,
not surfaced as a prediction in vLLM core.

Outline §12.3 says "predict which blocks will be used". The chapter
must reframe to "look up which blocks ARE in the cache". See impl-notes
Reframe 4.

## O08 — LRU vs ARC: ARC adapts but isn't always better

**Discovered by**: implementer (Ch12), confirmed by Demo 2 phase_shift
**Use by**: writer §12.2 + reviewer (Trap B nuance)

ARC (Megiddo & Modha 2003) maintains 4 lists: T1 (recent), T2 (frequent),
B1 (T1 ghost), B2 (T2 ghost). The B1/B2 ghost-list hits drive
`target_t1_size` adaptation: B1 hit → recency wins → grow T1; B2 hit →
frequency wins → shrink T1.

Demo 2 results (cap=32, 2000 ops):
- loop_scan (50 unique scanned 40x): both 100% miss (Belady-defeating)
- zipfian (80% on top-12): LRU 17.30%, ARC 17.25% (basically equal)
- phase_shift: LRU 2.60%, **ARC 14.15%** (ARC LOSES)

**Key finding**: ARC's accumulated ghost-list state is INVALIDATED by
sharp phase boundaries. Real production traffic has *partial* phase
shifts, where ARC wins. See Megiddo-Modha 2003 Table 4 for canonical
benchmark wins.

## O09 — SharedOffloadRegion: mmap pinned at startup

**Discovered by**: implementer (Ch12)
**Use by**: writer §12.5 (system impact); reviewer for Trap F

`vllm/v1/kv_offload/cpu/shared_offload_region.py:L27-L113` allocates ONE
mmap-backed file `/dev/shm/vllm_offload_{instance_id}.mmap` shared across
all worker processes. The file is pinned via `cudaHostRegister` (per-rank).

`MADV_POPULATE_WRITE` (Linux 5.14+, value 23) eagerly populates the
worker's slots so the first transfer doesn't pay page-fault tax.

Layout:
```
worker0_block0 | worker1_block0 | ... | worker{M-1}_block0
worker0_block1 | worker1_block1 | ... | worker{M-1}_block1
...
```
Row stride = cpu_page_size × num_workers. Each worker carves a strided
view: `as_strided(_base, (num_blocks, page_size), (row_stride, 1), worker_offset)`.

## O10 — Connector factory: lazy-loading registry pattern

**Discovered by**: implementer (Ch12)
**Use by**: writer §12.2 (taxonomy)

`vllm/v1/kv_offload/factory.py:L17-L52`. Each spec is registered with
`OffloadingSpecFactory.register_spec(name, module_path, class_name)`.
The class is imported only when `create_spec(config, kv_cache_config)`
asks for it. Pattern handles optional dependencies (LMCache, Mooncake,
Nixl) gracefully — missing deps surface at create time, not import.

The same pattern is used for `KVConnectorFactory` in `kv_connector/factory.py:L1-L228`.

## O11 — KVConnectorRole enum: SCHEDULER / WORKER

**Discovered by**: implementer (Ch12)
**Use by**: writer §12.2 (lifecycle)

`base.py:L123-L130`. The same connector class can be instantiated with
different roles. PD-disagg (Mooncake/Nixl) typically has prefill node =
WORKER + SCHEDULER, decode node = WORKER. CPU offload has both roles per
process (single-node setup).

`OffloadingConnector.__init__` (`offloading_connector.py:L51-L67`) shows
the role-conditional initialization: scheduler creates
`OffloadingConnectorScheduler`, worker creates `OffloadingConnectorWorker`.

## O12 — MultiConnector composes connectors

**Discovered by**: implementer (Ch12)
**Use by**: writer §12.2 (advanced setup)

`vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py` (629 LOC)
implements `MultiConnector(KVConnectorBase_V1, SupportsHMA)` that fan-outs
each lifecycle method to a list of child connectors. Common config:
`LMCacheConnectorV1` + `OffloadingConnector` — semantic disk cache
backed by CPU offload tier.

The composition is non-trivial because each child has its own keyspace;
MultiConnector dedupes by block-hash so a hit in any child counts once.

## O13 — simple_kv_offload/ is pedagogical; kv_offload/ is production v1

**Discovered by**: implementer (Ch12)
**Use by**: writer when picking which path to walk

vLLM ships TWO CPU offload paths at this commit:
- `vllm/v1/kv_offload/`: production v1. Pluggable policies, ARC, ghost
  lists, multi-worker shared mmap region, factory pattern, 18-connector
  composability.
- `vllm/v1/simple_kv_offload/`: minimal reference impl. Single-rank LRU
  via BlockPool (Ch02), single mode select between lazy/eager.
  Paired with `SimpleCPUOffloadConnector`.

Production teaches "what's possible" (ARC, ghost lists, mmap regions);
simple teaches "what's fundamental" (block pool, two streams, pin memory).
The chapter §3 walks the simple variant as anchor; §2 + §4 + §5 walk the
production variant for sophistication.

## O14 — Block size sensitivity to PCIe alpha-beta

**Discovered by**: implementer (Ch12)
**Use by**: writer §12.4 + reviewer (Trap C)

PCIe Gen5 alpha-beta model: `latency = alpha + beta × bytes`.
- alpha = 10 µs (transfer setup overhead)
- beta = 1.5e-5 µs/byte (=> 64 GB/s asymptotic)

**Break-even** (alpha == beta × bytes): `bytes_be = alpha/beta = 666 667`
≈ 651 KiB. vLLM's default 16 MB block is **24× past break-even** —
always in the bandwidth-bound regime. Smaller blocks would be alpha-bound
and waste setup overhead. Larger blocks (e.g. 64 MB) would not improve
throughput further but would increase tail-latency on partial misses.

Demo 3 shows: at decode step 50 ms, we can overlap 191 blocks per step.
At prefill step 200 ms, we can overlap 764 blocks per step. PCIe is
NOT a bottleneck for typical workloads.

## O15 — Eviction must be PROACTIVE (returns evicted_keys before store)

**Discovered by**: implementer (Ch12)
**Use by**: writer §12.3 + reviewer (architecture)

`vllm/v1/kv_offload/cpu/manager.py:L115-L168` prepare_store computes
evictions BEFORE allocating new blocks. The `PrepareStoreOutput` carries
`evicted_keys` so the scheduler can release worker-side state in the SAME
step.

If eviction were reactive (computed at allocate time), the scheduler
wouldn't know which downstream blocks were freed and might reuse evicted
slot ids while a transfer was still in flight. The proactive model
lets the policy.evict atomically return None if not enough idle blocks
exist, keeping the state machine consistent.

This is the architectural reason the `CachePolicy.evict` returns
`None | list[(key, block)]` rather than partial mutation: caller can
abort cleanly.

---

## O16 — Factory registration uses fully-qualified instance path; tests must re-register

**Discovered by**: tester (Ch12)
**Use by**: future testers + reviewer (test-suite portability)

`OffloadingSpecFactory.register_spec("CPUOffloadingSpec",
"instances.vllm.artifacts.12-kv-offload.implementation.offload_spec",
"CPUOffloadingSpec")` at `factory.py:L86-L90` uses the project-wide
fully-qualified path. Pytest run from the chapter dir uses
`sys.path = [chapter_root]` and only sees `implementation.*`, so
`create_spec("CPUOffloadingSpec")` raises `ModuleNotFoundError: No module
named 'instances'`.

**Fix**: tests re-register at module-load:
```python
OffloadingSpecFactory.unregister_spec("CPUOffloadingSpec")
OffloadingSpecFactory.register_spec(
    "CPUOffloadingSpec", "implementation.offload_spec", "CPUOffloadingSpec",
)
```

This is a portability fact, not a bug. The fully-qualified path makes
sense at vllm-runtime; tests run in a different sys.path regime.

## O17 — Lazy-load registry: register succeeds with bogus path; create fails late

**Discovered by**: tester (Ch12)
**Use by**: writer (factory chapter section)

`OffloadingSpecFactory.register_spec(name, "no.such.module", "Cls")`
SUCCEEDS at register time — the loader is a closure not yet invoked.
The `ModuleNotFoundError` surfaces only at `create_spec(name)`. This is
the laziness pattern from `vllm/v1/kv_offload/factory.py:L20-L52` —
optional deps (LMCache RPC, Mooncake RDMA) only crash when the user
actually requests that connector, not at vLLM startup.

Test: `test_factory.py::TestLazyLoading::test_loader_invoked_only_on_create`.

## O18 — BlockIDsLoadStoreSpec constructor copies its input list

**Discovered by**: tester (Ch12)
**Use by**: writer (data-contract sidebar)

`BlockIDsLoadStoreSpec(block_ids: list[int])` does `self.block_ids = list(block_ids)`,
making a defensive copy. Mutating the passed list afterward does NOT change
the spec. This protects the worker from "block_ids shifted under me" bugs
when the scheduler-side block-id list is later reused.

## O19 — LRUCachePolicy.touch iterates `keys` in REVERSE

**Discovered by**: tester (Ch12)
**Use by**: writer (LRU implementation walkthrough)

`vllm/v1/kv_offload/cpu/policies/lru.py:L26-L29` reverses the input
because the scheduler passes keys in chronological order; the LAST key
in `keys` is the most recently used. After iterating in reverse and
calling `move_to_end` on each, the LAST input key ends up at the MRU
position (= true MRU semantics).

**Implication**: callers MUST pass keys in chronological order — older
first, newer last. Failing this gives an inverted LRU.

## O20 — ARC evict() uses a dry-run phase to keep failure atomic

**Discovered by**: tester (Ch12)
**Use by**: writer (ARC implementation walkthrough) + reviewer

`vllm/v1/kv_offload/cpu/policies/arc.py:L97-L156` separates evict into
2 phases:
1. **Dry-run**: virtually pick N candidates from T1/T2 using a
   `virtual_t1_size` counter and `already_selected` set, WITHOUT mutating
   T1/T2/B1/B2.
2. **Apply**: only after N are confirmed eligible, perform the deletions
   and push to B1/B2.

This is what makes "return `None` on insufficient" atomic. If you mutated
during the picker phase, a partial-failure abort would leave T1/T2 in
an inconsistent state. The two-phase split is the architectural reason
the ARC code is more complex than LRU's straight scan.

## O21 — FilterReusedOffloadingManager.lookup IS the counter incrementer

**Discovered by**: tester (Ch12)
**Use by**: writer (reuse_manager walkthrough)

The `lookup` method at `reuse_manager.py:L67-L78` does double duty:
(1) increments the per-key counter, (2) delegates the actual probe to
the backing manager. Even a MISS on the backing increments the counter
on the FILTER side. So a key that misses 5 times still counts as
5-touched, which is what makes the store_threshold semantics work
(threshold is "how many times has the scheduler asked about this?",
not "how many hits?").

When `len(counts) >= max_tracker_size`, the LRU entry is evicted via
`popitem(last=False)` BEFORE inserting the new key (line L75-L77).

## O22 — complete_store(success=False) frees silently; only success=True emits event

**Discovered by**: tester (Ch12)
**Use by**: writer (event-stream sidebar) + reviewer (asymmetry callout)

`offload_manager.py:L286-L314` `complete_store`:
- `success=True`: flips `ref_cnt: -1 → 0`, appends to `stored_keys`, emits
  one OffloadingEvent with `removed=False`.
- `success=False`: removes the block from policy + frees the block_id —
  but emits NO event.

This asymmetry matters for prom-metrics: a "store failure" is invisible
to the metrics stream. If a deployment needs failure visibility, the
operator must instrument the worker side (where the actual upload
failed) — not the manager side.

---

## O23 — 4 TOPIC reframes != 6th "no class X"

**Discovered by**: writer (Ch12, 2026-05-08)
**Use by**: book-editor + future writers when an outline is partly off

When the outline mentions concrete artifacts that don't exist in source
(NVMe, LFU, attention-score, predictive prefetch in this case), the
reframe is at the TOPIC level, not the CLASS level. Treatment differs:

- **class-X reframe** (Ch07-Ch11): grep produces zero `class FooBar`,
  the chapter's central architectural metaphor. Five instances form a
  named series within the vllm instance.
- **topic-level reframe** (Ch12): grep produces zero of a TECHNIQUE
  (`nvme | ssd | disk`) or zero of a NAMED ALTERNATIVE (`lfu.py`,
  `attention_score.py`, `predict | markov`). The chapter pivots topic
  by topic, each with its own "outline corrects itself" moment.

Don't conflate the two. Forcing a 6th class-X when the source HAS the
classes (`OffloadingManager`, `CPUOffloadingManager`, 18 connectors)
would be dishonest. The cadence is "reframe where source diverges,
honestly", not "every chapter must have a flagship class absence".

## O24 — Hook-only ARC honest caveat: phase_shift demo placement matters

**Discovered by**: writer (Ch12, 2026-05-08)
**Use by**: writer + reviewer when calibrating ARC presentation

The phase_shift demo (LRU 2.60% vs ARC 14.15%) is the strongest single
piece of evidence for "ARC is not always better". Placement guidance:

- DO put it in §12.2.5 immediately after §12.2.4's "ARC vs LFU" section
  (so the reader sees "ARC > LFU, but ARC sometimes < LRU" in tight
  sequence, not as separate revelations).
- DO frame as "the cost ARC pays for adaptation didn't recover the win
  on this synthetic benchmark", NOT as a flaw in ARC.
- DO cite Megiddo-Modha 2003 Table 4 to explain why production
  workloads (partial phase shift) are different from synthetic
  full-phase-shift benchmarks.
- DON'T bury it in a Trap section without setup; the reader needs to
  have just walked the algorithm before they can absorb the caveat.

This is the K17-style honesty: real numbers, real benchmarks, real
caveats — not glossed-over wins.

## O25 — Two streams ≠ 2× speedup is a "nuance trap" (not a flat error)

**Discovered by**: writer (Ch12, 2026-05-08)
**Use by**: writer + reviewer for hardware-claim sections

Trap G ("two CUDA streams = 2× speedup") is subtler than a binary
right/wrong. The reader's intuition is partially correct:

- Two streams DO unlock parallelism (separate copy engines for two
  directions). Single stream serializes.
- Two streams DO improve throughput on bidirectional workloads. Real
  measurements typically 1.3-1.5×.
- Two streams do NOT double bandwidth — PCIe Gen5 ×16 = 64 GB/s is
  still 64 GB/s per direction; bidirectional gives 128 GB/s aggregate
  but rarely matters when one direction dominates.

When writing about hardware speedup, frame as "X unlocks Y, with
typical observed Z" — not "X gives 2× / X is useless". Readers absorb
nuance better when given both the mechanism AND the realistic number.

---

## O26 — Demo 5 wall-time is the ONLY non-deterministic verbatim numeric

**Tags**: reviewer, fidelity, demo-numerics, K17-OR-skip
**Use by**: future reviewer + writer when calling out "verbatim demo numerics"

In a 26-numeric demo set, 25 are deterministic (formula-driven: alpha-beta,
break-even = α/β, miss-rate ratios, capacity/bandwidth constants,
connector taxonomy counts) and reproduce bit-exactly across runs. The
exception is **Demo 5 end-to-end wall time** — it varies between runs
(sample observations: 50.83, 50.92, 51.01 ms) because it includes
`time.perf_counter()` boundaries around the alpha-beta sleep simulation
plus per-call Python overhead.

For verbatim-numeric review:
- Treat the wall-time line as approximate (chapter softens with "≈ 50 ms"
  in body even though the verbatim block fixes "50.92 ms" from one run).
- Test report should explicitly mark the wall-time as "~50 ms" not exact.
- Future demos that include wall-time should mark with K17 OR-skip
  caveat and use a `~` qualifier in the verbatim block.

---

## O27 — Three-anchor framing-tip discipline can be satisfied by alternative anchors

**Tags**: reviewer, framing-tips, three-anchor-rule, D28
**Use by**: future reviewer when verifying tip recap mapping

Writer's framing-tip self-mapping table at chapter end ("Tip N hook in §A,
body in §B, recap in §C") sometimes maps a recap to a Trap section that
doesn't directly contain the tip's content (e.g. Tip 2 "ARC NOT strictly
better" mapped to Trap B which is about LFU absence; the actual ARC-loses
recap lives in §12.10 chapter summary + the §12.6.6 invariants list).

Reviewer should verify three-anchor coverage **across the whole chapter**,
not just at the writer-claimed locations. If the actual three anchors
exist via different paths (e.g. body in §X, summary in §Y, invariant in
§Z) with the SAME content, the discipline is satisfied — recommend
tightening the self-mapping table in future cadence but DO NOT REVISE.

Universal pattern: three-anchor presence ≠ three-anchor self-attribution.
The literal map can be loose if the actual coverage is dense.

---

## O28 — "honest caveat" preservation is the strongest fidelity signal

**Tags**: reviewer, honest-caveat, ARC-loses, fidelity, dimensionality
**Use by**: reviewer + future writer for any chapter with counter-intuitive demo result

When tester surfaces a counter-intuitive demo result (Ch12: ARC LOSES to
LRU 5.4× on phase_shift; LRU 2.60% miss vs ARC 14.15%) and explicitly
flags it as HONEST CAVEAT, the writer's job is to preserve the result
verbatim across multiple anchors. Soft-pedaling ("ARC is generally better,
though phase shifts can hurt") loses the teaching signal.

In Ch12 the phase_shift LRU 2.60% / ARC 14.15% numbers appear at 8+
anchors: §12.0 quote-block reframe enumeration, §12.0 "这章要讲什么"
learning outcome, §12.2.5 demo body, §12.2.5 nuance prose, §12.2.7
demo-output verbatim block, §12.6.6 invariants (#7 "ARC 不总赢"),
§12.9.9 mapping table O08 row, §12.10 chapter summary.

Universal pattern: when a demo proves a textbook intuition wrong, **count
the verbatim anchors**. If <3 anchors → REVISE for missing recap. ≥5
anchors → strong fidelity. The Megiddo-Modha 2003 Table 4 academic
reference adds a critical trust signal: "this isn't a bug, this is exactly
what the original ARC paper measured" — chapter MUST cite when it has it.

---

## Last updated

2026-05-08 — initial fact set O01-O15 by implementer@book-factory at Ch12 dispatch.
2026-05-08 — appended O16-O22 by tester@book-factory after Ch12 fidelity testing
            (314 tests, 100% pass, 7-trap fidelity verified).
2026-05-08 — appended O23-O25 by writer@book-factory after Ch12 narrative
            (1577 lines, 10180 words, 285 mapping rows, 4 TOPIC reframes,
             7 traps, 5 framing tips, ARC-loses honest caveat applied).
2026-05-08 — appended O26-O28 by reviewer@book-factory after Ch12 review
            (verdict APPROVED single-cycle; hard gates 1-10 PASS;
             linters 0 blocking; N=9 v6 cadence baseline holds).
