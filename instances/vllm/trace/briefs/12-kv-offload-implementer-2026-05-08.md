# Rehydration Brief — Ch12 KV Cache Offload 与层级存储 — Implementer

- **Chapter**: `12-kv-offload`
- **Title**: KV Cache Offload 与层级存储
- **Outline level**: advanced (Part 2)
- **Status**: dispatch — first v6-grade pass for Ch12 (cadence baseline holds at N=8 after Ch11 single-cycle APPROVED with broadest comm-side surface yet — 12 vLLM modules + 5D mesh + 5th "no class X" graduated motif within vllm instance + cleanest implementer→tester handoff in book)
- **Dependencies (per outline)**: `02-kv-cache` (block-level KV layout), `05-memory-management` (physical-block pool + per-rank HBM budget)
- **Dependents downstream**: Ch13 (prefix-cache-pooling) — pool size composes with offload tier sizes; Ch22 (PD architecture) — KV-transfer protocols re-appear at PD-disagg boundary; Ch23 (PD prefix-cache) — multi-tier prefix-cache lookup composes with offload manager; Ch24 (layerwise-connectors) — per-layer connector wires through `KVConnectorBase_V1`; Ch27/Ch28 (DeepSeek deep-dives) — production offload uses LMCache / Mooncake / NixlConnector
- **Source pin**: vLLM commit `98661fe012c5c467252d4df8411d2f46190e9268` at `instances/vllm/source/` (verified by archivist 2026-05-08)
- **Brief generated**: 2026-05-08 by archivist
- **Recipient**: implementer (direct dispatch by team-lead, no book-editor relay — operational rule from Ch07-Ch11)

---

## §1 — Chapter scope (5 movements — what Ch12 actually covers)

**Core question**: A single 70B-class model at 128K context, even after DCP=4 + PCP=4 sharding (Ch11 reduces per-rank HBM from 40.0 GB to 2.5 GB), still consumes the bulk of HBM under realistic batch sizes. **Production traffic regularly OOMs**: prefix-cache hits keep storage demand ≥80% HBM; new requests need block allocations the GPU cannot satisfy. Cheaper memory exists — CPU DRAM is 8-32× larger than HBM at ~10× lower bandwidth (PCIe Gen5 64 GB/s vs HBM3 3 TB/s), and NVMe SSD adds a third tier at ~7 GB/s. **KV Offload** moves cold KV blocks down the storage hierarchy (HBM → CPU DRAM → SSD) and prefetches them back when the scheduler revisits the request. Critically, **vLLM's offload semantics are NOT the same as `swap_in`/`swap_out`** (those are Ch05's preempt-and-recompute path; offload is hot blocks staying live across multi-tier storage). The chapter covers **5 movements**:

1. **Why offload at all — HBM/DRAM/SSD bandwidth-vs-capacity hierarchy + per-tier roundtrip math.** Open `vllm/v1/kv_offload/base.py:L48-L100` (the `OffloadingManager` ABC + `LoadStoreSpec`/`PrepareStoreOutput`/`OffloadingEvent` dataclasses) — the actual abstraction layer for "where does this KV block live now?". Open `vllm/v1/kv_offload/cpu/spec.py:1-102` (`CPUOffloadingSpec`) — the CPU-DRAM impl. Walk the bandwidth-capacity tradeoff: HBM3 = ~80 GB capacity at 3 TB/s; CPU DDR5 = 512 GB capacity at 96 GB/s (host-side); PCIe Gen5 ×16 = 64 GB/s for HBM↔DRAM transfer; NVMe Gen5 = 14 GB/s sequential. **Roundtrip latency for one 16-MB KV block (16 layers × 64 heads × 128 head_dim × bf16 ≈ 16 MB conservative)**: HBM→DRAM at PCIe-bound = `16 MB / 64 GB/s = 250 µs`; DRAM→SSD = `16 MB / 7 GB/s ≈ 2.3 ms`. Compare to per-token decode latency (~50 ms for a 70B model on H100): a single block roundtrip is 0.5% of one decode step (HBM↔DRAM) or 4.6% (DRAM↔SSD). **Block size sensitivity**: vLLM's default block = 16 tokens; offload moves blocks not individual tokens (PCIe is bandwidth-bound, not latency-bound, so larger transfers amortise overhead). Derive: `transfer_overhead = α + β × bytes`; for PCIe `α≈10 µs, β≈1.5e-5 µs/byte`, the break-even block size where transfer cost equals attention compute cost is ~1 KB (always a profit at 16-MB blocks). Why is this NOT the same as "swap": swap (Ch05 §5.7) is a **preempt-and-recompute** path where blocks are EVICTED entirely; offload is a **demote-and-rehydrate** path where blocks STAY LOGICALLY LIVE for the request, just not on HBM. The chapter must surface this semantic distinction explicitly (candidate Trap A).

2. **Connector taxonomy — `KVConnectorBase_V1` family is HUGE.** Open `vllm/distributed/kv_transfer/kv_connector/v1/base.py:L170-L660` — the **`KVConnectorBase_V1` abstract class** with 30+ abstract/template methods covering the full offload lifecycle: `start_load_kv` / `wait_for_layer_load` / `save_kv_layer` / `wait_for_save` / `get_finished` / `get_block_ids_with_load_errors` / `get_num_new_matched_tokens` / `update_state_after_alloc` / `build_connector_meta`. Open `vllm/distributed/kv_transfer/kv_connector/factory.py:1-228` — the connector factory + registry. Open `vllm/distributed/kv_transfer/kv_connector/v1/` listing — **18 connector implementations** at this commit:
   - `lmcache_connector.py` (LMCacheConnectorV1) — flagship Python-side semantic-cache library with disk-backed prefix-cache
   - `mooncake/mooncake_connector.py` — Moonshot's RDMA-based KV-disaggregation backend
   - `nixl/connector.py` — NVIDIA Inference Xfer Library (RDMA + GPU-direct)
   - `hf3fs/hf3fs_connector.py` — HuggingFace's distributed file-system KV transfer
   - `offloading_connector.py` (OffloadingConnector + SupportsHMA) — the canonical CPU/SSD offload connector composing with `vllm/v1/kv_offload/`
   - `simple_cpu_offload_connector.py` — minimal reference impl (paired with `vllm/v1/simple_kv_offload/`)
   - `multi_connector.py` (MultiConnector + KVConnectorBase_V1, SupportsHMA) — composes multiple connectors (e.g. LMCache + offloading_connector)
   - `lmcache_mp_connector.py` — multi-process LMCache variant
   - `flexkv_connector.py` — research/experimental flex-KV
   - `decode_bench_connector.py`, `example_connector.py`, `example_hidden_states_connector.py` — reference / debug
   - `p2p/`, `moriio/`, `lmcache_integration/`, `offloading/` subdirs — additional adapters
   
   **Key insight**: Offload is NOT a single class — it is a **factory + registry pattern** (`factory.py:L1-L70`). Each adapter is plugged in via config (`kv_transfer_config.kv_connector` field). The 5-step rhythm should walk the `OffloadingConnector` (canonical CPU offload) + `LMCacheConnectorV1` (production-grade with disk tier) + briefly the RDMA family (NixlConnector / MooncakeConnector) for surface variety. **Out of scope for THIS chapter**: layerwise-connectors → Ch24; PD-side connectors (decode-bench, p2p) → Ch22-Ch25.

3. **CPU offload manager — `OffloadingManager` + `OffloadingSpec` + LRU/ARC eviction policies.** Open `vllm/v1/kv_offload/base.py:L110-L218` — the `OffloadingManager` ABC: `lookup(key) → bool|None`, `prepare_load(keys) → BlockIDsLoadStoreSpec`, `prepare_store(keys, evicted_keys) → PrepareStoreOutput`, `complete_load`, `complete_store`, `touch(keys)`, `take_events()`. Open `vllm/v1/kv_offload/base.py:L319-L398` — `OffloadingSpec` ABC + `get_manager()` + `get_handlers()` factory contract. Open `vllm/v1/kv_offload/cpu/manager.py:L25-L200` — `CPUOffloadingManager(OffloadingManager)` concrete impl. Open `vllm/v1/kv_offload/cpu/policies/` — three policies present: `lru.py` (Least-Recently-Used baseline), `arc.py` (Adaptive Replacement Cache — a hit-distribution-aware variant), `base.py` (policy ABC). Note: `lfu.py` does NOT exist at 98661fe — outline says "LRU/LFU/attention-score-based" but vLLM ships LRU + ARC, not LFU. **This is a candidate outline-vs-source mismatch (analogous to Ch11 §11.4 AG+RS vs A2A correction)**. Walk the math: LRU's miss rate under temporal-locality assumption (Belady's MIN as oracle baseline; LRU within 2× MIN under most workloads); ARC's adaptive split between LRU + LFU sets, sized adaptively from miss-stream signal — the "attention-score-based" policy that the outline mentions does NOT exist as an implemented policy at this commit (production research has explored it but it's not in vLLM at 98661fe). The chapter must surface this honestly (candidate Trap B: "LFU/attention-score-based policies are not implemented in vLLM at this commit"). 5-step rhythm: open `cpu/manager.py:L25` (CPUOffloadingManager init) → derive LRU vs ARC tradeoff → impl `our_offloading_manager.py` reproducing both policies in pure Python over a `dict[OffloadKey, BlockStatus]` → diff: vLLM's manager uses `BlockStatus` dataclass + per-block ref-counting + event queue for prom-metrics.

4. **Prefetch / pre-eviction overlap — schedule-aware demote-rehydrate decisions.** Open `vllm/v1/kv_offload/worker/worker.py:L1-L176` — `OffloadingHandler(ABC)` + `OffloadingWorker`. Walk the prefetch protocol: at scheduler time, a request's block-hashes are queried against `OffloadingManager.lookup()`; if a hash hits in CPU, the connector issues `start_load_kv()` ASYNC (overlapping with current decode steps); the actual GPU memory blocks are reserved via `update_state_after_alloc()`; the model forward calls `wait_for_layer_load()` which blocks ONLY when the layer is consumed (typical: the first 1-2 layers are blocking, the rest are already on GPU). **The compute-overlap math**: prefetch-N-blocks-during-current-step is profitable if `N × transfer_latency < step_compute_time`. For decode at H100 with PCIe Gen5: step_compute = 50 ms; transfer_latency_per_16MB_block = 250 µs; N_overlap = `50000/250 = 200 blocks` per decode step. For prefill: step_compute is much higher (compute-bound), so N_overlap is even larger. **Pre-eviction**: when the manager runs low on capacity, blocks must be evicted PROACTIVELY before the scheduler's next allocation; reactive eviction stalls the allocate path. Open `cpu/manager.py:L115-L195` — the `prepare_store` path that returns `(keys_to_store, store_spec, evicted_keys)`; eviction is computed BEFORE the actual store, so the scheduler sees the eviction-keys and updates worker-side state. 5-step rhythm: open `worker/worker.py:L1-L80` (handler ABC) → derive overlap math → impl `our_prefetch.py` reproducing async-prefetch on a fake transfer with `time.sleep(transfer_latency)` driven by a thread pool → diff: vLLM uses CUDA streams + pinned-memory async copies; ours simulates with sleeps.

5. **Pin memory + async copy + system impact — actual `torch.empty(pin_memory=True)` plumbing.** Open `vllm/v1/kv_offload/cpu/shared_offload_region.py:L1-L192` — the `SharedOffloadRegion` class: a single pinned-memory CPU buffer carved into block-aligned chunks, sized at startup from `kv_transfer_config.cpu_offload_size_GB`. Open `vllm/v1/kv_offload/cpu/gpu_worker.py:L1-L433` — `SingleDirectionOffloadingHandler(OffloadingHandler)` + `CpuGpuOffloadingHandlers`. Walk the H2D / D2H copy with `torch.cuda.Stream`: pinned-memory pages let CUDA-driver issue async DMA without staging through pageable memory; without pinning, the driver pageable→pinned bounce halves effective bandwidth. Open `cpu/gpu_worker.py` — search for `cudaMemcpyAsync`-equivalent (`tensor.copy_(other, non_blocking=True)`) usage with explicit stream tagging. **System impact**: HBM headroom freed = `cpu_offload_size_GB × hit_rate × HBM_to_DRAM_ratio`; under typical 60% prefix-cache hit and 100GB CPU offload pool, HBM headroom freed ≈ 60GB-equivalent of "virtual KV" not occupying HBM. But: PCIe bandwidth becomes the new bottleneck once offload-traffic exceeds ~30% of step time. The CHAPTER ENDS with the per-tier capacity-bandwidth-latency Pareto: HBM (capacity-bound, fastest), CPU DRAM (5-10× cheaper per GB, PCIe-bandwidth-bound), SSD (50-100× cheaper, SSD-bandwidth-bound, useful only for cold long-tail). 5-step rhythm: open `shared_offload_region.py:L1` → ask "what does pinning actually do?" → derive `pinned-memory-bandwidth ≈ 2 × pageable` → impl `our_pinned_buffer.py` using `torch.empty(..., pin_memory=True)` + `tensor.copy_(other, non_blocking=True)` with stream sync timing → diff: vLLM uses ref-counted region carving; ours uses simple slot allocator.

**OUT of scope** (do NOT re-cover):
- KV cache block layout / `KVCacheBlock` data structure → Ch02 territory. Reference, do NOT re-derive.
- Prefix-cache hash-chain / Trie / radix-vs-hash → Ch07 territory. Reference at the connector boundary only.
- Memory profiling / preempt-and-recompute / `swap_in`/`swap_out` → Ch05 territory. Reference for the Trap-A semantic-distinction surface.
- LMCache internal semantic-matching algorithm → only show the connector adapter surface, not the matching logic; that's a research-paper-level deep dive out of scope here (mention as forward-pointer to a hypothetical Ch29).
- PD-disagg KV transfer (Mooncake/NIXL P2P over RDMA) → Ch22-Ch25 territory. Reference Mooncake/NIXL connector classes EXIST but explicitly punt deep-dive to Ch22+.
- Layerwise-connectors / per-layer KV streaming → Ch24 territory. The `save_kv_layer` API is mentioned but the per-layer streaming protocol is not the focus.
- DCP/PCP / 5D mesh → Ch11 territory. Reference: offload is per-rank; in a DCP=4 + PCP=4 setup, each rank has its OWN local offload manager (NOT a global one) — note this composition explicitly but don't re-derive 5D mesh.

If implementer is re-deriving DCP partition math or re-implementing prefix-cache hash-chain — STOP. Those belong elsewhere.

---

## §2 — Source surface (verified at commit 98661fe)

### §2.1 — Files and exact line ranges

| File | Lines (verified) | What |
|---|---|---|
| `vllm/v1/kv_offload/base.py` | 398 lines total | Core abstractions: `OffloadKey`, `LoadStoreSpec`, `OffloadingEvent`, `OffloadingManager(ABC)`, `OffloadingSpec(ABC)`, `BlockIDsLoadStoreSpec`, `GPULoadStoreSpec`, `CanonicalKVCaches` |
| `vllm/v1/kv_offload/base.py` | L48-L100 | `ReqContext` + `LoadStoreSpec(ABC)` + `PrepareStoreOutput` + `OffloadingEvent` dataclasses — the data contract |
| `vllm/v1/kv_offload/base.py` | L110-L218 | `OffloadingManager(ABC)` — `lookup`, `prepare_load`, `touch`, `complete_load`, `prepare_store`, `complete_store`, `take_events`, `shutdown` |
| `vllm/v1/kv_offload/base.py` | L319-L398 | `OffloadingSpec(ABC)` — `get_manager()` + `get_handlers()` factory contract; consumed by `OffloadingSpecFactory` |
| `vllm/v1/kv_offload/factory.py` | 58 lines total | `OffloadingSpecFactory` — registry pattern with `register_spec(name, module_path, class_name)` + `create_spec(config, kv_cache_config)` |
| `vllm/v1/kv_offload/factory.py` | L51-L58 | `OffloadingSpecFactory.register_spec("CPUOffloadingSpec", "vllm.v1.kv_offload.cpu.spec", "CPUOffloadingSpec")` — the canonical registration |
| `vllm/v1/kv_offload/cpu/spec.py` | 102 lines total | `CPUOffloadingSpec(OffloadingSpec)` — concrete spec for CPU DRAM tier |
| `vllm/v1/kv_offload/cpu/manager.py` | 200 lines total | `CPUOffloadingManager(OffloadingManager)` — concrete LRU/ARC-backed manager for CPU |
| `vllm/v1/kv_offload/cpu/manager.py` | L25-L200 | `CPUOffloadingManager.{__init__, _get_num_free_blocks, _allocate_blocks, _free_block, lookup, prepare_load, touch, complete_load, prepare_store, complete_store, take_events}` |
| `vllm/v1/kv_offload/cpu/policies/` | 4 files | Eviction policies: `base.py` (ABC), `lru.py` (LRU), `arc.py` (Adaptive Replacement Cache). **NOTE: `lfu.py` is NOT present** — outline mentions LFU but it does not exist at 98661fe |
| `vllm/v1/kv_offload/cpu/shared_offload_region.py` | 192 lines total | `SharedOffloadRegion` — pinned CPU memory region carved into block-sized chunks |
| `vllm/v1/kv_offload/cpu/gpu_worker.py` | 433 lines total | `SingleDirectionOffloadingHandler(OffloadingHandler)` + `CpuGpuOffloadingHandlers` — H2D/D2H async copy with CUDA streams |
| `vllm/v1/kv_offload/worker/worker.py` | 176 lines total | `OffloadingHandler(ABC)` + `OffloadingWorker` — worker-side handler that executes prepare_load/store |
| `vllm/v1/kv_offload/reuse_manager.py` | 120 lines total | `FilterReusedOffloadingManager(OffloadingManager)` — wraps another manager; filters out re-used keys |
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | 662 lines total | `KVConnectorBase_V1(ABC)` + `KVConnectorRole` + `KVConnectorMetadata` + `SupportsHMA` |
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | L84-L115 | `SupportsHMA(ABC)` — Hybrid Memory Allocation contract; offload connectors implement this |
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | L123-L130 | `KVConnectorRole(enum.Enum)` — connector roles in PD/HMA |
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | L170-L660 | `KVConnectorBase_V1` — the canonical connector abstract class with 30+ template methods |
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | L298-L362 | `start_load_kv` / `wait_for_layer_load` / `save_kv_layer` — **the worker-side load/save lifecycle** |
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | L449-L506 | `get_num_new_matched_tokens` / `update_state_after_alloc` / `build_connector_meta` — **the scheduler-side prefetch protocol** |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` | 228 lines total | `KVConnectorFactory` — registry of connector implementations |
| `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` | 192 lines total | `OffloadingConnector(KVConnectorBase_V1, SupportsHMA)` — the canonical CPU-offload connector |
| `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py` | 354 lines total | `LMCacheConnectorV1(KVConnectorBase_V1)` — production-grade with disk tier (forward-pointer to a future Ch29) |
| `vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py` | 629 lines total | `MultiConnector(KVConnectorBase_V1, SupportsHMA)` — composes multiple connectors |
| `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py` | 247 lines total | Minimal reference impl paired with `vllm/v1/simple_kv_offload/` (good pedagogical baseline; quote in §12.2) |
| `vllm/distributed/kv_transfer/kv_transfer_state.py` | 78 lines total | `get_kv_transfer_group()` / `has_kv_transfer_group()` / `is_v1_kv_transfer_group()` / `ensure_kv_transfer_initialized` / `ensure_kv_transfer_shutdown` — singleton-style state container |
| `vllm/v1/simple_kv_offload/` | 4 files | `worker.py` (SimpleCPUOffloadWorker), `metadata.py` (SimpleCPUOffloadMetadata), `manager.py` (SimpleCPUOffloadScheduler) — the **minimal CPU offload reference impl** (paired with simple_cpu_offload_connector); use as the SOLE concrete walk-through |
| `vllm/v1/kv_cache_interface.py` | L81+, L745+, L760+ | `KVCacheSpec`, `KVCacheGroupSpec`, `KVCacheConfig` — the data contract that connectors consume |

### §2.2 — Outline-vs-source mismatches (CRITICAL — surface in chapter)

**Mismatch 1**: outline §12.2 says "LRU/LFU/attention-score-based选择策略" but at 98661fe vLLM ships **LRU + ARC** (`lru.py` + `arc.py`); **LFU does NOT exist** as a policy file; **attention-score-based does NOT exist** either. Reframe: §12.2 of the chapter walks LRU + ARC honestly, with sidebar (1-2 paragraphs) acknowledging that "research has explored attention-score-based eviction (Liu et al., StreamingLLM, etc.) but vLLM at this commit ships LRU + ARC; we'll walk those two." This is the **3rd 'in-chapter outline correction'** pattern (after Ch11 §11.4 AG+RS-vs-A2A and Ch11 §11.6 5D-not-3D-mesh) — leave outline JSON unchanged; reframe at chapter level.

**Mismatch 2**: outline title "KV Cache Offload 与层级存储" implies a uniform "offload abstraction" but reality is **18+ connector implementations + 1 OffloadingManager ABC + factory pattern** (`KVConnectorBase_V1` family is HUGE — see §1 movement 2). The outline is correct as a topic guide but the source is more taxonomic than monolithic. The chapter must walk the taxonomy honestly (NOT pretend offload = single class).

**Mismatch 3 (CANDIDATE — IMPLEMENTER MUST VERIFY)**: outline does NOT mention "swap_in/swap_out semantic distinction from Ch05 preempt-and-recompute" but this is a high-trap-risk surface (Trap A in §6 below). Implementer should explicitly include the Trap-A surface even though outline doesn't mention it.

### §2.3 — "no class X" check at 98661fe

**NOT a "no class X" candidate.** Unlike Ch07-Ch11 where a key outline-named entity had no class implementation in vLLM, KV offload **DOES have many concrete classes**:
- `OffloadingManager(ABC)` ✓
- `OffloadingSpec(ABC)` ✓
- `CPUOffloadingManager(OffloadingManager)` ✓
- `CPUOffloadingSpec(OffloadingSpec)` ✓
- `OffloadingConnector(KVConnectorBase_V1, SupportsHMA)` ✓
- `KVConnectorBase_V1(ABC)` ✓
- `LMCacheConnectorV1`, `MultiConnector`, `MooncakeConnector`, `NixlConnector`, `HF3FSConnector` — **18+ adapters** ✓
- `OffloadingHandler(ABC)` + `OffloadingWorker` ✓
- `SharedOffloadRegion`, `CanonicalKVCaches` ✓

So the "no class X" series **stays at 5 instances within vllm** (Ch07/Ch08/Ch09/Ch10/Ch11). Ch12 will NOT extend that streak — and that's fine; the streak should not be artificially preserved by reframing where the source clearly has the classes. This honestly grounds the motif as "framing tool for outline-vs-source mismatch", not "every chapter must have one". Document in `kv-offload.md` O01 as a non-trivial-decision insight.

---

## §3 — Outline subsection walk-through

Outline (verbatim from `book/book-outline.json`):
1. 层级存储——GPU HBM→CPU DRAM→NVMe SSD的访问延迟阶梯
2. Who-to-offload——LRU/LFU/attention-score-based选择策略 (**LFU/attention-score do NOT exist**; reframe to LRU+ARC)
3. Prefetch——预测哪些KV block会用到，提前搬回GPU
4. PCIe带宽下的offload-over-compute overlap分析
5. CPU端的KV Cache管理（pin memory + async copy）

These map naturally to §1 movements 1-5 in order. The chapter section structure should be:
- §12.1 Hierarchical storage — bandwidth-vs-capacity tradeoff with verbatim per-tier latency numbers
- §12.2 Connector taxonomy + `KVConnectorBase_V1` family + LRU vs ARC reframe (with sidebar about absent LFU/attn-score)
- §12.3 CPU offload manager — `OffloadingManager` ABC walk + `CPUOffloadingManager` impl
- §12.4 Prefetch / pre-eviction overlap math + scheduler-side protocol
- §12.5 Pinned memory + async copy + system impact + per-tier Pareto

§12.6 invariants + §12.7 language traps + §12.8 forward pointers + §12.9 mapping table — same v6 baseline structure as Ch11.

---

## §4 — Knowledge dependencies

### §4.1 — Existing modules to query

`knowledge/INDEX.md` shows:
- `kv-cache.md` (12 facts): block layout, KVCacheBlock, get/free protocol → reference for §12.1, §12.2
- `memory.md` (3 facts): memory profiling at startup → reference for §12.5
- `prefix-cache.md` (13+ facts): hash-chain, chain-break invariant → reference for §12.4 (where prefetch interacts with prefix-hits)
- `dcp-pcp.md` (29 facts): per-rank semantics, 5D mesh → reference for §12.5 system-impact
- `scheduler.md` (12 facts): scheduling protocol → reference for §12.4 prefetch-protocol
- `multi-token-prediction.md` (30 facts): SpecDecodeMetadata, weight loading → reference if forward-pointer needed

### §4.2 — NEW module: `kv-offload.md` with **O-prefix** facts

Use **O-prefix IDs** (O01, O02, ...) to avoid collision with existing prefixes (K = kv-cache, T = tensor-parallelism, E = expert-parallelism, P = preemption, M = multi-token-prediction, W = wisdom-related, D = dcp-pcp, C = continuous-batching). O is unused.

**Implementer should populate O01-O15 minimum** during impl-notes write-up:
- O01 — "no class X" series stays at N=5 within vllm (Ch12 has the classes, NOT a 6th; document as reasoning-trail)
- O02 — outline §12.2 LFU/attention-score reframe to LRU+ARC (3rd in-chapter outline correction, after Ch11 §11.4 / §11.6)
- O03 — `OffloadingConnector` vs `KVConnectorBase_V1` vs `OffloadingManager` — three-way distinction (connector orchestrates lifecycle; manager owns the keyspace; spec is factory)
- O04 — semantic distinction: offload ≠ swap (Trap A; offload preserves logical liveness; swap evicts and recomputes)
- O05 — pinned memory bandwidth ≈ 2× pageable (pin → no driver bounce → full PCIe BW)
- O06 — per-tier latency numbers (HBM 3 TB/s, PCIe Gen5 64 GB/s, NVMe Gen5 7 GB/s; per-16MB-block roundtrip 250 µs / 2.3 ms)
- O07 — prefetch overlap math: N_blocks per step = step_compute / transfer_latency
- O08 — LRU vs ARC: ARC adapts the LRU/LFU split based on miss-stream signal
- O09 — `SharedOffloadRegion` pre-allocates pinned buffer at startup (config: `cpu_offload_size_GB`)
- O10 — connector factory + registry pattern (`OffloadingSpecFactory.register_spec`)
- O11 — `KVConnectorRole` enum — connectors play different roles in PD vs HMA
- O12 — `MultiConnector` composes connectors (e.g. LMCache + offloading_connector)
- O13 — `simple_kv_offload/` is the pedagogical reference impl; `kv_offload/` is the production v1
- O14 — block_size sensitivity to PCIe overhead (always profitable at vLLM's 16-token blocks)
- O15 — eviction must be PROACTIVE (`prepare_store` returns `evicted_keys` BEFORE the store) — reactive eviction stalls scheduler

Tester will append O16+ during fidelity testing; writer will append O20+ during narrative; reviewer will append O25+ during gate-checks. Run `python3 scripts/learn.py compact kv-offload` if file exceeds 15 (P1-1 fix is operational at Ch11).

---

## §5 — Wisdom hits ranked by implementer priority

Per `wisdom/INDEX.md`, implementer's role priorities are: debugging > architecture > testing > writing.

1. **wisdom/architecture.md** — backpressure gates, lateral comm patterns. Direct hit: §12.4 prefetch protocol IS a backpressure-gated pipeline (scheduler → connector → worker → CUDA stream); the `wait_for_layer_load` blocking-only-when-consumed pattern matches the architecture wisdom.
2. **wisdom/debugging.md** — F.linear shapes (not relevant here), CUDA mismatches (relevant for H2D/D2H async copies — implementer will hit this if pinned-memory math is wrong), SVG clipping (not relevant). Direct hit: §12.5 async-copy stream-tagging is a typical CUDA-debug surface.
3. **wisdom/testing.md** — preemption-test design, OOM paths, Docker. Indirect hit: §12.3 LRU eviction can be tested with synthetic key streams; OOM-path tests are similar to Ch05 preempt path.
4. **wisdom/writing.md** — formula rules, code walkthrough, 大白话 spectrum. Standard hit: §12.1 latency-tier numbers should lead with formula (per Ch10 M17 / Ch11 D28 three-anchor), demo numbers second.

---

## §6 — Candidate language traps (5-7) — implementer should preview

1. **Trap A — "Offload is the same as swap"**: WRONG. Swap (Ch05 §5.7) is preempt-and-recompute (block evicted entirely; sequence rolls back to last token); offload is demote-and-rehydrate (block moves down hierarchy but request stays live; rehydration is async). Surface at §12.1 + Trap A in §12.7.
2. **Trap B — "LFU and attention-score-based eviction policies are implemented in vLLM"**: WRONG at 98661fe. vLLM ships LRU + ARC (`cpu/policies/lru.py` + `cpu/policies/arc.py`). LFU and attention-score-based are research papers, not in code. Surface at §12.2 with explicit grep evidence.
3. **Trap C — "CPU offload is free latency"**: WRONG. PCIe Gen5 = 64 GB/s; a 16 MB block transfer = 250 µs ≈ 0.5% of one decode step. At >30% of step time worth of offload-traffic, PCIe becomes the new bottleneck. Surface at §12.4 with break-even math.
4. **Trap D — "All connectors are interchangeable"**: WRONG. `OffloadingConnector` (CPU/SSD), `LMCacheConnectorV1` (semantic-cache + disk), `NixlConnector` (RDMA + GPU-direct), `MooncakeConnector` (PD-disagg KV transfer) — each has different transport, different protocol, different latency profile. The factory pattern lets you SELECT one, but they are NOT semantically equivalent. Surface at §12.2.
5. **Trap E — "Offload always saves memory"**: WRONG. Offload saves HBM but costs CPU DRAM (1:1 capacity tradeoff at the CPU tier). The PROFIT comes from HBM being scarcer; if the workload's KV footprint already fits in HBM, offload adds latency without saving anything. Surface at §12.1.
6. **Trap F — "Pinning memory is free"**: WRONG. Pinned memory is allocated at startup and DOESN'T release back to the OS easily (it's locked-down DRAM). Over-pinning starves other system processes. Surface at §12.5.
7. **Trap G — "v0 KV transfer = v1"**: WRONG. The v1 abstraction (`KVConnectorBase_V1`, `OffloadingManager`) is significantly richer than v0 (`kv_lookup_buffer/` is empty at this commit; v0 was deprecated). All new connectors target v1. Surface at §12.2 forward-pointer.

7 traps is the cadence (Ch09/Ch10/Ch11 each 7). Implementer's `impl-notes.md` should list all 7 with claim → 错 → 为什么 → 源码证据 → Demo/测试 substructure (per E19 / D28).

---

## §7 — Demo plan (5 demos producing ≥20 verbatim numerics)

Following Ch09/Ch10/Ch11 numerics-floor pattern (Ch11 had ≥40 verbatim numerics from the 5-demo set):

**Demo 1 — Per-tier latency stair**: write a `latency_pareto_demo.py` that enumerates HBM/CPU-DRAM/SSD with hardcoded bandwidth + capacity numbers; outputs a 3×3 table (tier × {capacity, bandwidth, per-16MB-block roundtrip}). Target verbatim numbers: 9 cells.

**Demo 2 — LRU vs ARC miss-rate**: write `policy_demo.py` running synthetic workloads (Zipfian, sequential, mixed) against pure-Python `LRUPolicy` and `ARCPolicy`; output miss rate per workload. Target verbatim numbers: 6 cells (3 workloads × 2 policies).

**Demo 3 — Prefetch overlap math**: write `overlap_demo.py` enumerating step_compute_time × N_overlap_blocks for decode-step (50ms) and prefill-step (200ms) at PCIe Gen5; output break-even N. Target verbatim numbers: 4 cells.

**Demo 4 — Pinned vs pageable bandwidth**: write `pin_demo.py` doing `torch.empty(..., pin_memory=True)` vs `torch.empty(...)` with explicit timing; output measured bandwidth ratio. Target verbatim numbers: 2 cells (pinned BW, pageable BW) — note these are MEASURED, mark with K17 OR-skip caveat in narrative.

**Demo 5 — End-to-end offload roundtrip**: write `offload_demo.py` instantiating `OurOffloadingManager` + `OurCPUOffloadingHandler`, putting 100 blocks through allocate → store → load → free cycle; output total wall time + per-step breakdown. Target verbatim numbers: 5 cells.

**Total**: 9 + 6 + 4 + 2 + 5 = 26 verbatim numerics — exceeds the ≥20 floor.

Numerics caveat discipline: if a number is MEASURED (Demo 4 pin-vs-pageable BW), narrative must mark with **K17 OR-skip** caveat ("number measured on this hardware; system-dependent"). All other demos produce DETERMINISTIC numbers (formula-driven).

---

## §8 — Floor reminders (N=8 v6 baseline)

Per `state.json v6_compliance.metrics_per_chapter` Ch11 baseline:
- **Lines**: ≥1300 (Ch11 = 1394; Ch10 = 1345; Ch09 = 1204)
- **Words**: ≥7000 (Ch11 = 8124; Ch10 = 8888; Ch09 = 7792)
- **Mapping rows**: ≥120 (Ch11 = 149; Ch10 = 206; Ch09 = 151)
- **impl_notes_source_files**: ≥10 (Ch11 = 12; Ch10 = 11; Ch09 = 10)
- **reference_comments_total**: ≥75 (Ch11 = 78 ; Ch10 = 151 inflated by proposer family; Ch09 = 66)
- **language_trap_callouts**: 7 (matches Ch09/Ch10/Ch11)
- **framing_tips_applied**: ≥5 with three-anchor (D28 rule)
- **review_cycles**: 1 (single-cycle APPROVE — N=8 in a row)
- **lint_formula_blocking**: 0 (mandatory)
- **lint_formula_warnings**: ≤15 (Ch11 ceiling — at the edge of M30 calibrated band; if Ch12 trends higher, M30 may need re-calibration)
- **lint_source_grounding**: PASS (mandatory)

---

## §9 — Cadence carry-forward from Ch11

**Ch11 graduated lessons**:
1. **Cleanest implementer→tester handoff in book** (zero patches required during testing). The pattern that produced this: brief pre-verifies outline-vs-source mismatches at source-commit BEFORE dispatch + lists exact line ranges + flags candidate language traps. Ch12 brief reproduces the same template (this section + §2 + §6).
2. **Three-anchor framing-tip discipline (D28 rule)**: every framing tip must have hook + body + recap anchors. Reviewer-3 codified this as a mechanical grep-able rule. Ch12 should produce 5 framing tips and verify each anchor with `grep -nE '<keyword>' chapter.md`.
3. **In-chapter outline correction (3rd instance)**: §12.2 LFU→LRU+ARC reframe is the 3rd in-chapter outline correction (after Ch11 §11.4 AG+RS-vs-A2A and Ch11 §11.6 5D-not-3D-mesh). Pattern is now stable for ≥3 instances within vllm; document as candidate cross-instance pattern (NOT yet wisdom-promotable per 2+ INSTANCES gate).
4. **Demo numerics caveat discipline (K17 OR-skip strict)**: any MEASURED number gets the OR-skip marker; deterministic numbers are quoted verbatim throughout.
5. **Knowledge file growth**: Ch11 dcp-pcp.md hit 29 facts (14 over compact trigger). P1-1 fix means `learn.py compact` works now. Ch12 should expect kv-offload.md to grow to 25-30 facts and trigger compaction at the end.

---

## §10 — Direct-dispatch operational notes

Per `feedback_direct_dispatch`: book-editor's idle-summary handoffs are unreliable; team-lead direct-SendMessages each agent. For Ch12:

1. **Implementer dispatch (NOW)**: team-lead SendMessage implementer with this brief; implementer produces context.json + impl modules + impl-notes + ran-linter handoff to tester.
2. **Tester dispatch**: team-lead SendMessage tester after implementer's handoff; tester produces tests + framing-tips (≥5) + ran-linter handoff to writer.
3. **Writer dispatch**: team-lead SendMessage writer after tester's handoff; writer produces chapter.md (1300+ lines, ≥75 REFERENCE comments, three-anchor framing tips, 7 traps).
4. **Reviewer dispatch**: team-lead SendMessage reviewer after writer's handoff; reviewer runs both linters + 11 hard gates + verdict.
5. **Archivist dispatch (after APPROVED)**: team-lead SendMessage archivist with verdict; archivist records delivery + updates state + writes Ch13 brief (per brief-on-approval discipline).

**Loop >3**: escalate to team-lead. Pair-deadlock (writer↔implementer modification cycle): escalate to team-lead.

**Per CLAUDE.md HARD RULE**: team-lead CANNOT directly write `narrative/chapter.md`. Edits go through writer agent or pipeline.

---

**END OF BRIEF**. Source pinned at `98661fe`. Outline-vs-source mismatches flagged at §2.2. 7 candidate traps at §6. 26 verbatim numerics target at §7. v6 floors at §8.
