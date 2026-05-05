# Memory Management Module — Repo-Specific Facts

## File Locations
| Key File | Path | Role |
|----------|------|------|
| CacheConfig | `vllm/config/cache.py` | KV cache configuration (block_size, gpu_memory_utilization, cache_dtype, etc.) |
| BlockPool | `vllm/v1/core/block_pool.py` | Global block pool + prefix cache hash map (BlockHashToBlockMap) |
| KVCacheUtils | `vllm/v1/core/kv_cache_utils.py` | KVCacheBlock, FreeKVCacheBlockQueue, BlockHash types, get_kv_cache_configs |
| KVCacheManager | `vllm/v1/core/kv_cache_manager.py` | External interface: allocate_slots / free / get_computed_blocks |
| KVCacheCoordinator | `vllm/v1/core/kv_cache_coordinator.py` | Orchestrates multiple KV Cache Groups (find_longest_cache_hit, allocate, remove_skipped) |
| SingleTypeKVCacheManager | `vllm/v1/core/single_type_kv_cache_manager.py` | Per-type block allocation logic |
| Memory Profiling | `vllm/utils/mem_utils.py` | MemorySnapshot, memory_profiling context manager, DeviceMemoryProfiler |
| CumemAllocator | `vllm/device_allocator/cumem.py` | PyTorch pluggable allocator for sleep/wake mode |
| GPU Worker | `vllm/v1/worker/gpu_worker.py` | determine_available_memory → memory_profiling → initialize_from_config |
| BlockTable | `vllm/v1/worker/block_table.py` | GPU-side block table construction for kernel consumption |

## Key Concepts
1. **BlockPool = global shared pool** — Single pool for all KV cache blocks across attention types (Full/SWA/MLA/Mamba)
2. **Memory Profiling = runtime measurement, not static calculation** — Dummy forward with max_num_batched_tokens to measure real peak memory
3. **V0→V1 jump** — From single concatenated tensor (V0) to Coordinator + Group + BlockPool (V1)
4. **block_size=16** — Empirical sweet spot: ~3% internal fragmentation, reasonable block table size
5. **gpu_memory_utilization=0.92** — 8% safety margin for non-PyTorch memory (cuBLAS, NCCL, CUDA Graph capture peak)
6. **cumem Sleep/Wake** — Pluggable allocator for multi-tenant GPU memory sharing between vLLM instances
7. **Prefix Cache via BlockHashToBlockMap** — Hash-based (not Radix Tree like SGLang), general but less efficient for tree-structured sharing

## Recent Pitfalls
- expandable_segments conflicts with cumem Pluggable Allocator → auto-disable required (PR #40812)
- KV cache NaN from stale FP4 scale padding → torch.empty → torch.zeros (PR #d2afd40d6)
- Workspace resize leaking reserved GPU memory → fix in PR #39226

## Papers
- **vLLM (Kwon et al., arxiv 2309.06180)** — Original PagedAttention paper, virtual memory for KV cache
- **FlashAttention (Dao et al., 2022)** — IO-aware attention, O(N²)→O(N) memory
- **GANDIVA (Henderson et al., arxiv 2310.12996)** — Prefix-aware KV cache, ancestor of block hash recycling

## Writer Notes (Ch05)
- **Formula pitfalls**: `\text` in `$$` blocks must be `\mathrm`. `$$` must be on its own line. These cluster in Cell 2 disaster numbers (MiB/GiB/bytes units).
- **Reader confusion points**: "Virtual memory analogy" is the key teaching strategy — readers first understand OS page table, then map to KV Cache blocks. Table format (OS concept → vLLM equivalent) works very well.
- **Demo output alignment**: Cell 7 output comes from actual `python3 implementation/*.py` runs. Never fabricate output in the narrative — readers will try to reproduce and lose trust.
- **Profiling demo's zero budget**: memory_profiling.py shows `available_kv_cache_memory = 0` — not a bug, this is realistic. In narrative, explain WHY it's zero (max_num_batched_tokens too large) and HOW to fix (reduce batch). This is the "disaster" teaching strategy at work.
- **Reverse-order free = free LRU**: `reversed(req_blocks)` + `free_block_queue.append_n()` puts tail blocks at queue tail (last evicted), prefix blocks at head (first evicted among eviction candidates). No extra timestamps or sorting — purely through operation ordering.
