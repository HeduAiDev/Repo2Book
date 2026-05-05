# Implementation Notes — Ch05 GPU Memory Management

## Source Analysis

### 1. What files implement this feature?

| File | Lines | Role |
|------|-------|------|
| `vllm/v1/core/kv_cache_manager.py` | L106-L542 | `KVCacheManager` — external interface for scheduler. `allocate_slots()`, `free()`, `get_computed_blocks()`. Creates coordinator, owns `block_pool` reference, constructs `KVCacheBlocks` results. |
| `vllm/v1/core/kv_cache_manager.py` | L21-L104 | `KVCacheBlocks` — allocation result data class. Tuple-of-sequences encoding: `blocks[i][j]` = j-th block of i-th KV cache group. Methods: `get_block_ids()`, `new_empty()`. |
| `vllm/v1/core/block_pool.py` | L130-L509 | `BlockPool` — shared block pool. Manages `free_block_queue` (doubly-linked list in eviction order), `cached_block_hash_to_block` (hash→block for prefix cache), `null_block` placeholder. Key methods: `get_new_blocks()`, `free_blocks()`, `touch()`, `cache_full_blocks()`. |
| `vllm/v1/core/block_pool.py` | L34-L127 | `BlockHashToBlockMap` — hash→block cache supporting duplicate block hashes via union type (single `KVCacheBlock` or `dict[int, KVCacheBlock]`). |
| `vllm/v1/core/kv_cache_utils.py` | L113-L160 | `KVCacheBlock` — per-block metadata: `block_id`, `ref_cnt`, `block_hash`, `prev_free_block`, `next_free_block`, `is_null`. |
| `vllm/v1/core/kv_cache_utils.py` | L162-L370 | `FreeKVCacheBlockQueue` — doubly-linked list for O(1) pop/remove/append. Uses fake head/tail sentinel nodes. Ordered by eviction priority (LRU). |
| `vllm/v1/core/kv_cache_utils.py` | L37-L78 | Block hash types and helpers: `BlockHash`, `BlockHashWithGroupId`, `make_block_hash_with_group_id()`, `get_block_hash()`, `get_group_id()`. |
| `vllm/v1/core/kv_cache_utils.py` | L539-L566 | `hash_block_tokens()` — incremental hash: `hash(parent_hash, token_ids, extra_keys)`. Uses `NONE_HASH` seed for first block. |
| `vllm/v1/core/kv_cache_utils.py` | L930-L948 | `get_num_blocks()` — `available_memory // page_size // num_layers` then clamp. |
| `vllm/v1/core/kv_cache_coordinator.py` | L28-L268 | `KVCacheCoordinator` (ABC) — orchestrates multiple `SingleTypeKVCacheManager`s. Owns `block_pool`, creates managers via factory. |
| `vllm/v1/core/kv_cache_coordinator.py` | L276-L321 | `KVCacheCoordinatorNoPrefixCache` — coordinator when prefix caching is disabled. |
| `vllm/v1/core/kv_cache_coordinator.py` | L324-L389 | `UnitaryKVCacheCoordinator` — coordinator for models with exactly one KV cache type. |
| `vllm/v1/core/kv_cache_coordinator.py` | L392-L591 | `HybridKVCacheCoordinator` — coordinator for hybrid models (SWA+Full Attention). Iterative fixed-point cache hit algorithm. |
| `vllm/v1/core/single_type_kv_cache_manager.py` | L30-L444 | `SingleTypeKVCacheManager` (ABC) — per-type block management: `req_to_blocks` mapping, `num_cached_block` tracking, `allocate_new_blocks()`, `allocate_new_computed_blocks()`, `remove_skipped_blocks()`. |
| `vllm/v1/core/single_type_kv_cache_manager.py` | L446-L504 | `FullAttentionManager` — full attention cache hit: left-to-right scan, break on first miss. |
| `vllm/v1/core/single_type_kv_cache_manager.py` | L507-L641 | `SlidingWindowManager` — sliding window cache hit: right-to-left scan for contiguous hit. `remove_skipped_blocks()` frees blocks outside window. |
| `vllm/v1/core/kv_cache_metrics.py` | L46-L97 | `KVCacheMetricsCollector` — sampled block residency metrics (lifetime, idle time, reuse gaps). |

### 2. What are the key classes and their responsibilities?

- **KVCacheManager** (`kv_cache_manager.py:L106`): Top-level interface for the scheduler. Delegates to `KVCacheCoordinator`. Creates `KVCacheBlocks` wrappers. Key methods:
  - `get_computed_blocks(request)` → prefix cache hit lookup
  - `allocate_slots(request, num_new_tokens, ...)` → 3-stage allocation: free skipped blocks → handle prefix tokens → allocate new blocks
  - `free(request)` → free all blocks in reverse order
  - `get_block_ids(request_id)` → return block table for model runner

- **BlockPool** (`block_pool.py:L130`): Physical block storage. Owns three key data structures:
  - `free_block_queue: FreeKVCacheBlockQueue` — free blocks ordered by eviction priority
  - `cached_block_hash_to_block: BlockHashToBlockMap` — hash→block for prefix cache
  - `null_block: KVCacheBlock` — singleton placeholder for skipped/missing blocks

- **KVCacheBlock** (`kv_cache_utils.py:L113`): Lightweight metadata. Tracks `ref_cnt` (number of requests sharing this block — used for both active use and prefix cache), `block_hash` (set when full and cached), linked-list pointers for free queue.

- **FreeKVCacheBlockQueue** (`kv_cache_utils.py:L162`): Doubly-linked list with fake head/tail sentinels. All operations O(1) because they manipulate `prev_free_block`/`next_free_block` directly on block objects — no Python object allocation on the hot path.

- **BlockHashToBlockMap** (`block_pool.py:L34`): Union-type cache: single block → `KVCacheBlock`, duplicates → `dict[int, KVCacheBlock]`. Avoids dict overhead in the common single-block case.

- **KVCacheCoordinator** hierarchy (`kv_cache_coordinator.py`): ABC → NoPrefixCache | Unitary | Hybrid. The coordinator's `find_longest_cache_hit()` is the critical path for prefix caching. Unitary coordinator delegates to `SingleTypeKVCacheManager.find_longest_cache_hit()`.

- **SingleTypeKVCacheManager** subclasses (`single_type_kv_cache_manager.py`): `FullAttentionManager` (left→right scan, break on first miss), `SlidingWindowManager` (right→left scan for contiguous window), `ChunkedLocalAttentionManager`, `MambaManager`, `CrossAttentionManager`, `SinkFullAttentionManager`.

### 3. What is the data flow?

```
Scheduler.schedule()
  │
  ├─ KVCacheManager.get_computed_blocks(request)
  │   └─ Coordinator.find_longest_cache_hit(block_hashes, max_hit_length)
  │       └─ SingleTypeManager.find_longest_cache_hit()
  │           └─ BlockPool.get_cached_block(hash, group_ids)
  │               └─ BlockHashToBlockMap.get_one_block(key)
  │
  └─ KVCacheManager.allocate_slots(request, num_new_tokens, new_computed_blocks)
      │
      ├─ Coordinator.remove_skipped_blocks(request_id, num_computed)
      │   └─ SingleTypeManager.remove_skipped_blocks()
      │       └─ BlockPool.free_blocks(removed)  ← ref_cnt--, return to free list
      │
      ├─ Coordinator.get_num_blocks_to_allocate(...)
      │   └─ SingleTypeManager.get_num_blocks_to_allocate()
      │       └─ cdiv(num_tokens, block_size) - len(req_blocks)
      │
      ├─ Coordinator.allocate_new_computed_blocks(...)
      │   └─ BlockPool.touch(new_computed_blocks)  ← ref_cnt++, remove from free list
      │
      ├─ Coordinator.allocate_new_blocks(...)
      │   └─ SingleTypeManager.allocate_new_blocks()
      │       └─ BlockPool.get_new_blocks(num)  ← pop from free list, evict if cached
      │
      ├─ Coordinator.cache_blocks(request, num_tokens)
      │   └─ SingleTypeManager.cache_blocks()
      │       └─ BlockPool.cache_full_blocks()  ← compute hashes, insert into map
      │
      └─ Return KVCacheBlocks(new_blocks)

Scheduler.update_from_output()
  └─ if request finished:
      └─ KVCacheManager.free(request)
          └─ Coordinator.free(request_id)
              └─ SingleTypeManager.free()
                  └─ BlockPool.free_blocks(reversed(req_blocks))
                      ← ref_cnt--, append to free list (tail = eviction candidates)
```

Memory profiling flow (startup):
```
GPUWorker.determine_available_memory()
  ├─ 1. Measure total GPU memory
  ├─ 2. Load model weights → record torch memory
  ├─ 3. Dummy forward with max_num_batched_tokens → record peak activation memory
  ├─ 4. Profile CUDA Graph memory (if enabled)
  └─ 5. available_kv_cache = requested_memory - (weights + peak_act + cudagraph + non_torch)

get_kv_cache_configs()
  └─ get_num_blocks(vllm_config, num_layers, available_memory, page_size)
      └─ num_blocks = available_memory // page_size // num_layers
      └─ BlockPool(num_blocks) → KVCacheConfig → KVCacheManager
```

### 4. What design decisions did the original authors make and WHY?

**Decision 1: Doubly-linked list instead of Python `deque`** (`kv_cache_utils.py:L162-L370`)

`FreeKVCacheBlockQueue` implements its own doubly-linked list using `prev_free_block`/`next_free_block` fields on `KVCacheBlock` objects. The built-in `deque` supports O(1) pop/append at ends but O(n) remove from middle. vLLM needs O(1) remove from middle because: when a block in the free list is hit by prefix cache lookup, it must be removed from the free list (via `touch()`). A `deque.remove(block)` would be O(n). The custom linked list achieves O(1) by storing pointers on the block objects themselves.

Trade-off: more code (~200 lines vs 1 import) and manual pointer management vs guaranteed O(1) for all operations. The performance gain is significant since `touch()` is called on every prefix cache hit.

**Decision 2: Union type for `BlockHashToBlockMap`** (`block_pool.py:L34-L126`)

```python
_cache: dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]
```

The hash→block mapping uses a union type: single block when only one block has a given hash, `dict[int, KVCacheBlock]` when multiple identical-content blocks exist. This avoids the overhead of a nested dict in the common case (single block per hash). Multiple blocks with the same hash happen during prefill: if two requests process the same prefix simultaneously, both produce blocks with identical hashes before either has finished.

Trade-off: more complex insert/get logic vs ~30% less memory and faster access in the single-block case.

**Decision 3: `gpu_memory_utilization=0.92` default** (`research brief, `gpu_worker.py` → `determine_available_memory()`)

The 8% headroom accounts for: (a) non-PyTorch-managed memory that can't be profiled (cuBLAS workspace growth, NCCL buffers), (b) CUDA Graph capture memory that peaks during capture but shrinks during replay, (c) PyTorch CUDACachingAllocator fragmentation ("reserved but not usable"). Without this buffer, cudaMalloc would fail during operation. The value 0.92 comes from empirical measurement across T4/A100/H100.

Trade-off: 8% "wasted" memory vs zero OOM risk during normal operation. Users needing more KV cache can set `gpu_memory_utilization=0.98` if they know their workload.

**Decision 4: `block_size=16` default** (`cache_config.py`, `kv_cache_utils.py`)

Block size of 16 tokens balances: (a) internal fragmentation — average 7.5 tokens wasted per request (less than 3% for typical 256+ token sequences), (b) block table size — for a 2048-token sequence, 128 blocks = ~1KB block table (negligible), (c) kernel efficiency — 16 consecutive tokens benefit from vectorized memory access in attention kernels. Too small (4) means 4x block table entries and more kernel launches; too large (128) means 64-token average waste per request.

**Decision 5: Reverse-order free for prefix cache coherency** (`kv_cache_manager.py:L418-L426`, `single_type_kv_cache_manager.py:L303-L318`)

When freeing a request's blocks, they are freed in reverse order (tail first). Combined with `free_block_queue` ordering (newly freed appended to tail), this means: (a) tail blocks (decoding-only tokens) are evicted first — they're least likely to be shared, (b) prefix blocks (near the start) stay in cache longer — they're most likely to be hit by other requests. This implements an approximate LRU policy at no extra cost.

**Decision 6: Incremental block hashing with parent hash chaining** (`kv_cache_utils.py:L539-L566`)

```python
BlockHash = hash(parent_block_hash, curr_block_token_ids, extra_keys)
```

Each block's hash depends on its parent's hash (the previous block in the sequence). This means: prefix of length N hashes to exactly one chain of N blocks — no need to re-hash the entire prefix when a new block is added. The `NONE_HASH` seed for the first block is initialized from `PYTHONHASHSEED` or `os.urandom(32)` for reproducibility or randomness.

### 5. What complexity must our implementation preserve?

| Mechanism | Source | Why Preserved |
|-----------|--------|---------------|
| KVCacheBlock with ref_cnt | `kv_cache_utils.py:L113-L160` | Reference counting is the core mechanism that allows prefix cache sharing — multiple requests reference the same block |
| FreeKVCacheBlockQueue doubly-linked list | `kv_cache_utils.py:L162-L370` | O(1) middle-remove is essential for prefix cache touch() performance |
| BlockPool: free_queue + hash_map | `block_pool.py:L130-L509` | The dual data structure (free list for allocation, hash map for cache lookups) is the central insight |
| allocate_slots() 3-stage flow | `kv_cache_manager.py:L225-L416` | Free→Handle prefix→Allocate new — this order is critical for correctness |
| Reverse-order free | `single_type_kv_cache_manager.py:L303-L318` | Required to implement LRU eviction policy via free_queue ordering |
| get_num_blocks_to_allocate() admission cap | `single_type_kv_cache_manager.py:L88-L167` | Without this, SWA models would OOM |
| prefix cache: find_longest_cache_hit | `block_pool.py:L184-L209`, `single_type_kv_cache_manager.py:L446-L494` | Left-to-right scan for full attention — break on first miss. Core prefix caching algorithm |
| Memory profiling: 3-category budget | `kv_cache_utils.py:L930-L948` | Weights + peak_act + kv_cache = total GPU memory. Static calculation is wrong |
| Null block placeholder | `block_pool.py:L174-L177` | Required for sliding window models where skipped positions need placeholder entries in block table |
| BlockHashToBlockMap union type | `block_pool.py:L34-L127` | Duplicate-hash handling is needed for correctness when multiple requests share same prefix simultaneously |

## Source Mapping Table

| Our Code | Original Source | What We Changed | Why |
|----------|----------------|-----------------|-----|
| `KVCacheBlock` | `kv_cache_utils.py:L113-L160` | Removed `__repr__`, kept core fields (block_id, ref_cnt, block_hash, prev/next_free_block, is_null) | Core metadata is same; repr is debug-only |
| `FreeKVCacheBlockQueue` | `kv_cache_utils.py:L162-L370` | Kept full logic. Added `_validate()` for debugging, simplified error messages | The doubly-linked list is the core data structure; must be identical for correctness |
| `BlockHashToBlockMap` | `block_pool.py:L34-L127` | Kept union-type structure. Removed `_unexpected_blocks_type()` internal helper — inlined assertions | Core caching logic preserved; union-type handling is the essential complexity |
| `BlockPool` | `block_pool.py:L130-L509` | Removed: KV cache events, metrics collector, `evict_blocks()`, `reset_prefix_cache()`, `take_events()`. Simplified: `cache_full_blocks()` without LoRA/MM/extra_keys | Events/metrics are observability layer; block pooling logic untouched |
| `KVCacheBlocks` | `kv_cache_manager.py:L21-L104` | Removed: `__add__`, `get_unhashed_block_ids()`, overloaded `get_block_ids()`. Kept: basic `get_block_ids()`, `new_empty()` | Core data class preserves blocks tuple structure |
| `KVCacheManager` | `kv_cache_manager.py:L106-L542` | Removed: EAGLE, DCP/PCP, `log_stats`, `prefix_cache_stats`, `enable_kv_cache_events`, `get_num_common_prefix_blocks()`, `take_events()`, `evict_blocks()`, `reset_prefix_cache()`. Kept: 3-stage `allocate_slots()`, `get_computed_blocks()`, `free()`, `create_kv_cache_blocks()` | Core allocation flow is identical; removed orthogonal features |
| `memory_profiling.py` (new) | `kv_cache_utils.py:L930-L948` + `gpu_worker.py::determine_available_memory()` | Condensed profiling pipeline into single `profile_gpu_memory()` function with simulation | Multi-worker profiling is PP/TP complexity; single-GPU case captures the 3-category budget concept |
| `Coordinator` | `kv_cache_coordinator.py:L28-L389` | Inlined into `KVCacheManager` via `UnitaryKVCacheCoordinator` logic. Removed Hybrid/NoPrefixCache variants | Single attention type (full attention) means no multi-group coordination needed |
| `FullAttentionManager` | `single_type_kv_cache_manager.py:L446-L504` | Inlined into `KVCacheManager`: `find_longest_cache_hit()`, `allocate_new_computed_blocks()`, `allocate_new_blocks()`, `cache_blocks()`, `free()` | Single type means no need for ABC + per-type subclass dispatch |

### What we simplified (and why it's OK for education)

| Original Complexity | Our Simplification | Justification |
|---------------------|-------------------|---------------|
| Multi-KV-cache-type (HybridCoordinator, 3 coordinator subclasses) | Single-type unitary coordinator inlined into KVCacheManager | Core block pool allocation logic is identical; multi-type is an extension covered by the architecture discussion. Most models use single type. |
| DCP/PCP (context parallelism block_size scaling) | `block_size` is always `cache_config.block_size` | Distributed concern; single-GPU has trivial block_size scaling. |
| EAGLE speculative decoding (last-block drop, eagle_group_ids) | Not implemented | Speculative decoding is Ch21/26 content; the block pool doesn't need to know about it. |
| KV cache events (BlockStored, BlockRemoved, AllBlocksCleared) | Not implemented | Observability layer; block logic unchanged. |
| KVCacheMetricsCollector (sampled residency tracking) | Not implemented | Metrics are for production monitoring; not needed for understanding the algorithm. |
| LoRA/multimodal extra_keys in block hashing | Simple token-id-only hashing | The hashing mechanism is the same; extra_keys are additional inputs to the hash function. |
| `BlockHashWithGroupId` packing (bytes concatenation) | Simple `(hash_bytes, group_id)` tuple key | Group ID packing is a micro-optimization; the concept is identical. |
| DeepseekV4 Multi-Layer KV Cache | Not implemented | Too specialized; mentioned in research brief for awareness. |
| `num_gpu_blocks_override` / auto-fit max_model_len | Not implemented | Configuration convenience; the allocation math is what matters. |
| `check_enough_kv_cache_memory()` / binary search estimate | Not implemented | Validation helper; not core allocation logic. |
| `cache_blocks()` with `delay_cache_blocks` for P/D | Always cache immediately | P/D disaggregation (Ch22-25); single-instance always caches. |
| `allocate_slots()` external_computed_tokens (KV connectors) | Not implemented | P/D content; all tokens are local. |
| `remove_skipped_blocks()` with sliding window | `get_num_skipped_tokens()` always returns 0 (full attention) | Full attention never skips tokens; SWA is discussed in research brief. |
| `full_sequence_must_fit` admission gate | Not implemented | Chunked prefill admission; discussed conceptually. |
| `SinkFullAttentionManager` (attention sink blocks) | Not implemented | Research feature; not core to memory management. |

### Key formulas

**Single block bytes** (FullAttentionSpec):
```
page_size_bytes = 2 × block_size × num_kv_heads × head_size × dtype_size
```
Factor of 2: K cache + V cache.

**Total blocks**:
```
num_blocks = available_kv_cache_memory // page_size_bytes // num_layers
available_kv_cache_memory = total_gpu × gpu_memory_utilization - non_kv_cache_memory
```

**Per-request max blocks**:
```
max_blocks_per_request = ceil(max_model_len / block_size)
```

**Theoretical max concurrency**:
```
max_concurrency = num_blocks / max_blocks_per_request
```

**Block hash** (incremental, chained):
```
BlockHash_i = hash(BlockHash_{i-1}, token_ids[i*block_size:(i+1)*block_size])
BlockHash_0 = hash(NONE_HASH, token_ids[0:block_size])
```
