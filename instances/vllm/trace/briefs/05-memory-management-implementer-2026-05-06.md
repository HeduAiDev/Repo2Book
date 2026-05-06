# Rehydration Brief — Ch05 Memory Management (Implementer)

- **Chapter**: `05-memory-management` (slug-aligned ID; legacy ID was `06-memory-management`)
- **Title**: GPU 显存管理系统
- **Outline level**: core (Part 2 lead-in)
- **Status**: `needs_rewrite` — full v6-grade rewrite per Ch04 baseline
- **Dependencies**: `04-continuous-batching` (published_v6 — use its `SimpleKVCacheManager` + `Scheduler` patterns as a reference)
- **Dependents downstream**: `06-scheduling`, `12-kv-offload`
- **Source pin**: vLLM commit `98661fe` at `instances/vllm/source/`
- **Brief generated**: 2026-05-06 by archivist
- **Recipient**: book-editor-2 → forward to implementer-2 with full context

---

## 1. Project state context

- Part 1 complete (Ch01-Ch04 published_v5/v6).
- Ch04 is the FIRST v6-grade chapter — all subsequent rewrites must meet/exceed its baseline.
- Source pinned at vLLM `98661fe`; no expected churn in `vllm/v1/worker/gpu_worker.py` or `vllm/utils/mem_utils.py` for this chapter scope.

## 2. Chapter scope (what Ch05 actually covers)

**Core question**: how does vLLM decide how much GPU memory is available for KV Cache, given that the GPU also holds model weights, peak activations, CUDA Graph buffers, and PyTorch fragmentation overhead?

**The spine of the chapter** (per legacy intro at `artifacts/05-memory-management/narrative/_legacy/chapter.md`):
1. Open `vllm/v1/worker/gpu_worker.py:L352` — `determine_available_memory()` is the entry point, runs once at engine startup.
2. Profiling vs static config: vLLM does NOT read KV cache size from config — it MEASURES by running a dummy forward pass and reading PyTorch peak-memory counters.
3. Four accounting buckets: model weights, peak activation, CUDA Graph + NCCL, KV cache. `(1 - gpu_memory_utilization)` is the safety margin.
4. The arithmetic: `available_kv_cache_bytes = requested - non_torch_increase - torch_peak_increase - weights_memory - cudagraph_memory`.
5. `MemorySnapshot.measure()` (the baseline + diff mechanic) — relies on `torch.cuda.memory_stats()` and `nvmlDeviceGetMemoryInfo`.

**Out of scope** (these belong to other chapters — do NOT bleed into Ch05):
- Block allocation / KV blocks structure → Ch02 (kv-cache) already covered.
- Prefix cache hashing / dedup → Ch07 (prefix-cache).
- Offload to CPU/disk → Ch12 (kv-offload).
- Per-layer KV transfer → Ch24 (layerwise-connectors).

## 3. Source modules to reference (commit 98661fe verified)

| File | Lines | What |
|---|---|---|
| `instances/vllm/source/vllm/v1/worker/gpu_worker.py` | L352-end of method | `GPUWorker.determine_available_memory()` — main entry point |
| `instances/vllm/source/vllm/utils/mem_utils.py` | L191 | `memory_profiling()` context manager — the accounting engine |
| `instances/vllm/source/vllm/utils/mem_utils.py` | (search) | `MemorySnapshot.measure()` — baseline + diff |
| `instances/vllm/source/vllm/v1/worker/gpu_model_runner.py` | L5845 | `profile_run()` — dummy forward pass at peak batch shape |
| `instances/vllm/source/vllm/v1/executor/uniproc_executor.py` | L182 | `Executor.determine_available_memory()` (single-proc shim) |
| `instances/vllm/source/vllm/v1/executor/abstract.py` | L146 | `Executor.determine_available_memory()` abstract base |
| `instances/vllm/source/vllm/v1/core/kv_cache_utils.py` | (search `get_kv_cache_config`) | takes `available_kv_cache_bytes` → produces `KVCacheConfig` (block count) |

The implementer MUST verify these line numbers against `git rev-parse HEAD` in `instances/vllm/source/` matching `98661fe` before writing references. If commit drifts, re-verify and update K08-style entries in `knowledge/modules/scheduler.md` (or create `knowledge/modules/memory.md`).

## 4. Knowledge module recommendation

**Create `knowledge/modules/memory.md`** — Ch05's scope (memory profiling, `mem_utils`, `gpu_worker.determine_available_memory`) is distinct from:
- `kv-cache.md` (block structures, `kv_cache_manager.py`, `block_pool.py` — Ch02/12/13)
- `scheduler.md` (already at 17 facts — over the 15-fact cap; oldest 5 should be LLM-compacted before Ch06)

Update `knowledge/INDEX.md` to add the new row.

## 5. Strengths to preserve from Ch04 (the v6 baseline)

These are precedents — replicate them in Ch05:

1. **Bubble math grounded in demo**: Ch04's `T_static = max(P_i) + max(O_i) = 416` was derived from the actual demo workload AND matched the demo printout verbatim. For Ch05, derive the four-bucket equation `available_kv_cache = requested - non_torch - torch_peak - weights - cudagraph` and demonstrate it on a concrete (small) workload — e.g. a tiny model, batch=2, with measured numbers.
2. **5-step rhythm per major section**: Source Trail → Bridge → Theory Deep Dive → Implementation → Source Diff. Ch04 had 7 sections (§4.1-§4.7), all rhythm-compliant. Ch05's natural sections: profiling pipeline overview, MemorySnapshot mechanic, four accounting buckets, KV-bytes-to-num-blocks conversion, gpu_memory_utilization margin semantics, OOM-safety properties.
3. **Source mapping table density**: Ch04 had 13 rows (≥ 5 required). Ch05 should have ≥ 10.
4. **`# REFERENCE:` comments saturated**: Ch04 had 65 across 6 impl files (avg ~11/file). Ch05 should target ≥ 60 across the impl set.
5. **Three-asserts gate framing**: Ch04 §4.2.1 framed `scheduler.py:L848-L853` as "the final correctness gate." Ch05 has analogous gates — e.g., the assertion `available_kv_cache_bytes > 0` after profiling, or `non_torch_increase >= 0` invariant. Frame these the same way.
6. **Diagrams optional when demo + tables suffice**: per reviewer-2's precedent on Ch04, no figure is mandatory if a numbered table or demo replay carries the visual load. The four-bucket pie / waterfall lends itself naturally to an SVG, but if a clean ASCII table covers it, that's also fine.
7. **F-style forward pointers for cross-chapter loose ends**: Ch04 used `F01` to defer `finished_req_ids` semantics to Ch20. Ch05 should similarly defer (a) prefix-cache impact on KV-bytes accounting → Ch07, (b) per-layer KV bytes split → Ch12.
8. **Chinese narrative + math-block split**: per K16, Chinese text NEVER goes inside `$$` blocks (no `\text{}`/`\mathrm{}` workarounds). Split prose around math fragments.

## 6. Implementation skeleton hints

The implementer's `implementation/` for Ch05 likely needs:
- `simple_memory_profiler.py` — a minimal stand-in for `mem_utils.memory_profiling()` (context manager + measure-diff).
- `simple_gpu_worker.py` — a stand-in for `GPUWorker.determine_available_memory()` calling profiler.
- `demo.py` — runs profiler on a tiny stub model, prints the four-bucket breakdown, computes available KV bytes, reports `num_blocks = available / block_bytes`.
- `impl-notes.md` — list ≥ 5 source files (gpu_worker.py, mem_utils.py, gpu_model_runner.py, executor/abstract.py, kv_cache_utils.py).

The legacy `implementation/_legacy/memory_profiler.py` (319 lines) is reference-only — do NOT copy verbatim. Inspect for accounting bucket structure; rewrite from source under v6 standards.

## 7. Process / handoff requirements (per Ch04 lessons)

Per `trace/cross-chapter/handoff-protocol-2026-05-06.md`:
1. Implementer must SendMessage tester-2 explicitly when implementation is ready (do not rely on Task hook alone).
2. Tester must SendMessage writer-2 explicitly with test-pass evidence.
3. Writer must SendMessage reviewer-2 explicitly when narrative is ready.
4. Reviewer must SendMessage archivist-2 (me) with verdict + status-JSON path on APPROVE.
5. Each stage's TaskUpdate → completed BEFORE the SendMessage, so the next agent sees a consistent task state.

## 8. Knowledge / wisdom queries the implementer should run before starting

```
# Knowledge facts relevant to Ch05
cat knowledge/modules/scheduler.md  # K10 (allocate_slots signature) is relevant for KVCache→memory linkage
cat knowledge/modules/kv-cache.md  # block_size, num_blocks math
# create knowledge/modules/memory.md  # blank — first chapter to populate it

# Wisdom for implementer role (priority: debugging > architecture > testing > writing)
cat wisdom/debugging.md  # CUDA/PyTorch memory-stats gotchas if any
cat wisdom/architecture.md  # 5-step rhythm, source-table density
cat wisdom/testing.md  # how to test profiler without a real GPU (mock torch.cuda.memory_stats)
```

## 9. Open questions for book-editor / team-lead

1. **GPU dependency**: Ch05's `profile_run()` realistically requires a GPU for accurate measurement. Two options for the simplified impl:
   - (a) **Mock approach**: stub `torch.cuda.memory_stats()` returns; demo runs on CPU. Reproducible everywhere.
   - (b) **Real approach**: tiny model on a real GPU; demo only runs in CI with GPU.
   - Recommendation: start with (a) for testability; document (b) as the "real" path in narrative §5.3.
2. **scheduler.md compaction**: the file is at 17 facts (over 15-cap). The archivist (me) should LLM-compact K05-K09 (oldest with lowest access counts) into a single summary fact BEFORE Ch06 starts work. Ch05 is unaffected (different module file).
3. **Topology**: default `linear` should suffice — no algorithm complexity at Ch04 level. Reconsider if implementer hits unknowns in `mem_utils.py`.

## 10. Deliverables checklist for implementer-2

- [ ] `implementation/simple_memory_profiler.py` with ≥ 20 `# REFERENCE:` comments
- [ ] `implementation/simple_gpu_worker.py` with ≥ 15 `# REFERENCE:` comments
- [ ] `implementation/demo.py` with reproducible four-bucket breakdown
- [ ] `implementation/impl-notes.md` listing ≥ 5 source files
- [ ] `knowledge/modules/memory.md` populated with ≥ 3 facts discovered during impl
- [ ] `knowledge/INDEX.md` updated with the new module row
- [ ] On completion: `TaskUpdate` → completed AND `SendMessage` to tester-2

---

**This brief should be forwarded by book-editor-2 to implementer-2 with full message body intact.**
