# Implementation Notes — Ch05 GPU Memory Management

Strategy A1 rewrite. vLLM source pinned at `98661fe`. All references verified
against that commit.

## Source Analysis (HARD GATE)

### 1. What vLLM files implement this feature?

| Role | Path (verified at 98661fe) | Lines actually read |
|------|----------------------------|---------------------|
| Memory snapshot + profiling | `instances/vllm/source/vllm/utils/mem_utils.py` | L60-L275 (MemorySnapshot, MemoryProfilingResult, memory_profiling) |
| KV cache spec | `instances/vllm/source/vllm/v1/kv_cache_interface.py` | L80-L205 (KVCacheSpec base, AttentionSpec, FullAttentionSpec) |
| KV cache block + free queue | `instances/vllm/source/vllm/v1/core/kv_cache_utils.py` | L113-L370 (KVCacheBlock, FreeKVCacheBlockQueue) + L689-L723 (`_check_enough_kv_cache_memory`) + L930-L947 (`get_num_blocks`) |
| Block pool + prefix cache | `instances/vllm/source/vllm/v1/core/block_pool.py` | L130-L510 (BlockPool, BlockHashToBlockMap) |
| KV cache manager | `instances/vllm/source/vllm/v1/core/kv_cache_manager.py` | L1-L541 (full file — for caller-pattern grounding) |
| Worker memory profiling | `instances/vllm/source/vllm/v1/worker/gpu_worker.py` | L340-L505 (`determine_available_memory`) |
| KV offload (CPU swap path used in v1 ONLY for prefix-cache offload) | `instances/vllm/source/vllm/v1/kv_offload/cpu/gpu_worker.py` | L1-L319 (skimmed for context — NOT a preemption path) |

### 2. Key classes and responsibilities

| Class | Owns | Delegates to |
|-------|------|--------------|
| `MemorySnapshot` | torch_peak, free/total NVML readings, torch_memory, non_torch_memory | `current_platform.mem_get_info`, `torch.accelerator.memory_stats` |
| `MemoryProfilingResult` | `non_kv_cache_memory`, `torch_peak_increase`, `non_torch_increase`, `weights_memory` | `MemorySnapshot.__sub__` for diffing snapshots |
| `KVCacheSpec` (base) | `block_size` | subclasses for `page_size_bytes` / `max_memory_usage_bytes` |
| `AttentionSpec` | `num_kv_heads`, `head_size`, `dtype` (+ optional kv_quant, page_size_padded) | `get_dtype_size` |
| `FullAttentionSpec` | `head_size_v`, `sliding_window`, `attention_chunk_size` | extends AttentionSpec |
| `KVCacheBlock` | block_id, ref_cnt, _block_hash, prev/next free pointers, is_null | — |
| `FreeKVCacheBlockQueue` | doubly-linked list of free blocks; fake head/tail | manipulates KVCacheBlock pointers directly (no GC churn) |
| `BlockPool` | `blocks` array, `free_block_queue`, `cached_block_hash_to_block`, `null_block` | FreeKVCacheBlockQueue, BlockHashToBlockMap, KVCacheMetricsCollector |
| `KVCacheManager` | `coordinator`, `block_pool`, `enable_caching`, `prefix_cache_stats` | `KVCacheCoordinator` (multi-group routing), BlockPool |
| `Worker.determine_available_memory` | `init_snapshot`, `requested_memory`, `available_kv_cache_memory_bytes`, `peak_activation_memory`, `cudagraph_memory_estimate`, `non_torch_memory` | `memory_profiling` ctx mgr, `model_runner.profile_run` |

### 3. Data flow (engine startup → KV cache allocated)

```
Engine.__init__()
  ↓
Worker.init_device():                                     # vllm/v1/worker/gpu_worker.py
  init_snapshot = MemorySnapshot(auto_measure=True)       # NVML read at boot
  ↓
Worker.load_model():
  with _scoped_allocator_max_split(20MB):                 # gpu_worker.py:L335-L343
    model_runner.load_model()                             # weights = DeviceMemoryProfiler measures
  ↓
Worker.determine_available_memory():                       # gpu_worker.py:L352
  with memory_profiling(init_snapshot, weights_memory) as result:
    model_runner.profile_run()                            # max-batch dummy fwd
    profile_torch_peak = torch.accelerator.memory_stats[allocated_bytes.all.peak]
    if cudagraph_mode != NONE:
      cudagraph_memory_estimate = profile_cudagraph_memory()
  result.torch_peak_increase = profile_torch_peak - before_profile.torch_peak
  result.non_torch_increase  = after_profile.non_torch - before_create.non_torch
  result.non_kv_cache_memory = non_torch + torch_peak + weights
  available_kv_cache = requested_memory - non_kv_cache - cudagraph_memory
  ↓
get_num_blocks(available_kv_cache, num_layers, page_size_bytes)
  → num_gpu_blocks = available // page_size // num_layers
  ↓
torch.zeros(total_block_bytes, dtype=int8, device='cuda')  # the actual GPU allocation
  → reshaped to backend layout by attn_utils._allocate_kv_cache
  ↓
KVCacheManager.__init__(kv_cache_config, ...):             # vllm/v1/core/kv_cache_manager.py:L107
  coordinator = get_kv_cache_coordinator(...)
  block_pool  = coordinator.block_pool                     # BlockPool(num_gpu_blocks, enable_caching)
    → free_block_queue.popleft() for null_block            # block_pool.py:L176-L177
  ↓
SCHEDULER USES THIS: kv_cache_manager.allocate_slots(req, num_new) per step
  → block_pool.get_new_blocks(N)                           # popleft_n + ref_cnt += 1
  → on free: block_pool.free_blocks(...)                   # ref_cnt -= 1; if 0, append_n
```

### 4. Design decisions and WHY

**D1. Three categories: torch / non-torch / external** (`mem_utils.py:L204-L235`).

vLLM splits GPU memory into:
1. memory used by *other processes* on the same device (pre-vLLM).
2. memory used by *torch in this vLLM process* (weights + activations + reserved pool).
3. memory used by *this vLLM process but not by torch* (NCCL, attention workspace, CUDA context).

WHY: only category 2 is shrinkable from inside the engine (via `empty_cache()`).
Category 3 is opaque to PyTorch but very real, hence measured separately via
NVML diffs. Without this split, the 8% safety margin would be insufficient.

**D2. gpu_memory_utilization defaults to 0.92** (`config/cache.py`, default).

The 8% margin (1.0 − 0.92) absorbs:
- PyTorch caching allocator fragmentation (≈ 2-3%)
- Activation peaks above the profile run (varies with input shape)
- CUDA context overhead that shows up between snapshots
- A real safety buffer against transient OOMs during sampling

**D3. Page size = 2 × block_size × num_kv_heads × head_size × dtype_bytes**
(`kv_cache_interface.py:L153-L170`).

WHY the formula is per-block-per-layer, not total: vLLM allocates *one flat
int8 tensor per layer* and reshapes it to the backend's expected shape.
`get_num_blocks` then divides by `page_size * num_layers` — the layer factor
is split out so heterogeneous-layer models (mamba mixed with attention) can
vary block count per group.

**D4. Doubly-linked free list, not deque** (`kv_cache_utils.py:L162-L183`).

`deque` has O(1) push/pop but O(n) middle-removal. vLLM needs the latter for
`BlockPool.touch()` — when a *currently-freed* block hits the prefix cache,
it must be removed from the middle of the free queue and ref_cnt-bumped. With
a deque the prefix-cache fast path would be O(n). The hand-rolled doubly-linked
list does it in O(1) at the cost of ~30 extra lines of pointer juggling.

**D5. Recompute, not swap, on preemption** (`scheduler.py:L952-L972`,
`scheduler.py:L964` resets `num_computed_tokens = 0`).

vLLM v0 had a swap-to-CPU preemption path. v1 removed it. Trade-off:
- Swap moves O(KV) bytes over PCIe (~32 GB/s).
- Recompute redoes prefill (typically slower than one-way swap, faster than
  a round-trip on PCIe Gen3).
- Recompute is *always available* (no CPU memory dependency), bit-deterministic,
  and one code path. Swap requires CPU memory budget tracking.

For an 8K-token prompt at fp16: KV ≈ 1 GiB, swap round-trip ≈ 62 ms,
recompute @ 50 K tok/s prefill ≈ 164 ms. Recompute is 2.6× slower in
*latency*, but the simplicity-and-safety budget wins.

**D6. The "null block" (block 0) is reserved at startup** (`block_pool.py:L173-L177`).

Block 0 is `is_null=True`, popped out of the free queue on construction, and
never freed. WHY: sliding-window attention treats blocks outside the window as
"don't load anything". The model runner needs *some* valid block_id to pad
the block table with — that's the null block. Reserving it eliminates a
nullable in the block table dtype.

### 5. Complexity preserved (NOT simplified away)

- Three-category memory split (torch peak vs non-torch vs weights).
- The exact `available_kv_cache = requested - weights - peak_activation -
  non_torch - cuda_graph` formula at `gpu_worker.py:L441-L445`.
- Page-size formula (2 × bs × num_kv_heads × head_size × dtype).
- Doubly-linked free queue with O(1) middle-removal.
- LRU eviction order (head=LRU, tail=most-recently-freed).
- Cached blocks staying in the prefix-cache map across `free` calls.
- `touch()` re-activates a freed-but-cached block (prefix cache fast path).
- Null-block reservation at boot.
- `get_usage()` subtracts the null block from the denominator.
- Recompute-only preemption (v1 design).

### What we deliberately simplified (each annotated)

- Quantization paths: nvfp4 and per-token-head kv-quant scales are referenced
  but not implemented in `AttentionSpec.real_page_size_bytes`.
- `BlockHashToBlockMap`'s "multiple blocks per hash" branch is a plain dict.
- `KVCacheManager.allocate_slots` and the multi-group `KVCacheCoordinator` are
  not modelled — the demo uses `BlockPool` directly. Ch12-13 (kv-offload,
  prefix-cache deep dive) will fill these in.
- Encoder-cache, structured output, KV connector, sliding-window per-block
  remove logic: all noted but unimplemented.
- `memory_profiling` does not actually call `gc.collect()` /
  `torch.accelerator.empty_cache()` — the demo populates snapshots manually.
- DCP/PCP context-parallel sharding factor in `max_memory_usage_bytes`.

## 1:1 Source Mapping

| Our code | vLLM source | What we changed | Why |
|----------|-------------|-----------------|-----|
| `MemorySnapshot` | `vllm/utils/mem_utils.py:L70-L157` | `auto_measure` defaults to False | demo runs without a GPU |
| `MemorySnapshot.__sub__` | `mem_utils.py:L128-L145` | identical | needed for the diff math |
| `MemoryProfilingResult` | `mem_utils.py:L160-L187` | identical fields, no `__post_init__` device wiring | callers pre-fill snapshots |
| `memory_profiling` | `mem_utils.py:L190-L275` | dropped `gc.collect` / `empty_cache` | pedagogical |
| `KVCacheSpec` (base) | `kv_cache_interface.py:L80-L127` | identical | — |
| `AttentionSpec` | `kv_cache_interface.py:L129-L170` | dropped nvfp4 + kv-quant branches | out of scope |
| `AttentionSpec.real_page_size_bytes` | `kv_cache_interface.py:L153-L170` | identical fp16/bf16 path | core formula |
| `FullAttentionSpec` | `kv_cache_interface.py:L173-L205` | dropped DCP/PCP factor | single-GPU demo |
| `get_num_blocks` | `kv_cache_utils.py:L930-L947` | dropped `may_override_num_blocks` config hook | identical math |
| `KVCacheBlock` | `kv_cache_utils.py:L113-L159` | identical | — |
| `FreeKVCacheBlockQueue` | `kv_cache_utils.py:L162-L370` | popleft_n implemented as loop | clarity > raw perf |
| `BlockPool` | `block_pool.py:L130-L510` | `dict` instead of `BlockHashToBlockMap`, dropped KV events | demo doesn't need event publisher |
| `BlockPool.get_new_blocks` | `block_pool.py:L322-L352` | dropped `metrics_collector` calls | optional path |
| `BlockPool.touch` | `block_pool.py:L391-L406` | dropped `metrics_collector.on_block_accessed` | optional path |
| `BlockPool.free_blocks` | `block_pool.py:L408-L422` | identical | — |
| `BlockPool.evict_blocks` | `block_pool.py:L424-L441` | identical | — |
| `BlockPool.get_usage` | `block_pool.py:L486-L497` | identical | — |
| `determine_available_memory` | `gpu_worker.py:L352-L505` | takes pre-built MemorySnapshot/Result | pedagogical (no real GPU) |
| `MemoryLayout` | — (synthesized from worker fields) | new dataclass | bundles breakdown for narrative |
| `estimate_max_concurrency` | `kv_cache_utils.py:L872-L890` | identical math | — |
| `PreemptionScenario` | — (analytical model) | new | recompute-vs-swap chart |

## Files in this directory

- `__init__.py` — module map.
- `mem_snapshot.py` — `MemorySnapshot`, `MemoryProfilingResult`, `memory_profiling`.
- `kv_cache_spec.py` — `KVCacheSpec`, `AttentionSpec`, `FullAttentionSpec`, `get_num_blocks`.
- `kv_cache_block.py` — `KVCacheBlock`, `FreeKVCacheBlockQueue`.
- `block_pool.py` — `BlockPool` with prefix-cache hash + LRU eviction.
- `memory_layout.py` — `determine_available_memory`, `MemoryLayout`, `estimate_max_concurrency`.
- `recompute.py` — analytical recompute-vs-swap latency comparator.
- `demo.py` — runnable end-to-end trace.
- `_legacy/` — old v5 attempt (memory_profiler.py only); kept read-only for reference.

## Running the demo

```bash
python3 -m instances.vllm.artifacts.05-memory-management.implementation.demo
```

Expected highlights:
- Llama-3.2-1B on 80 GiB H100 → ~68.65 GiB available KV → 35,148 blocks.
- ~275 concurrent requests at avg_seq_len=2048 fit in the cache.
- Block size sweep: doubling block_size halves block count, wasted ~ 1.6 MiB.
- Recompute vs swap for 8K prompt: ~164 ms vs ~62 ms (recompute 2.6× slower
  but still chosen for simplicity / OOM safety).

## Notes for the next implementer / writer

- The "wasted bytes" in the layout is *per-layer rounding*; expect ~1-2 MiB
  on H100. It's negligible at the 70 GiB scale; flag if it ever exceeds 1%.
- `BlockPool.null_block` is `block_id=0`. If you write a test that checks
  "first allocated block id", expect 1, not 0.
- `free_blocks` does NOT clear the cache hash. A re-`get_new_blocks` MAY
  re-pop the same physical block (now ref_cnt=1 again, hash cleared by
  `_maybe_evict_cached_block`). Tests need to be careful with this.
- For Ch12-13, the demo's `BlockPool` is the entry point — `KVCacheManager`
  routes per-group, but the underlying state is here.
- v1 has *no* swap preemption. If a writer reads about "swap" in older vLLM
  blog posts / papers (Kwon et al. 2023), be explicit: that was v0;
  recompute is v1's only path. The `kv_offload` subsystem is for prefix
  cache offload, not preemption.
