# Implementation Notes — Chapter 02 KV Cache

## Source Files Referenced

1. **vllm/v1/core/kv_cache_utils.py** (2126 lines) — KVCacheBlock, FreeKVCacheBlockQueue, block hashing, sizing
2. **vllm/v1/core/block_pool.py** (509 lines) — BlockPool, BlockHashToBlockMap
3. **vllm/v1/core/kv_cache_manager.py** (541 lines) — KVCacheManager, KVCacheBlocks
4. **vllm/v1/core/single_type_kv_cache_manager.py** (1173 lines) — Per-attention-type allocation logic
5. **vllm/v1/core/kv_cache_coordinator.py** (642 lines) — Multi-group coordination
6. **vllm/v1/kv_cache_interface.py** — KVCacheConfig, KVCacheSpec, page_size_bytes formula

## Source Mapping Table

| # | Our Class/Method | vLLM Source File:Line | Notes |
|---|---|---|---|
| 1 | `KVCacheBlock` | `kv_cache_utils.py:L114` | We use @dataclass; source uses @dataclass(slots=True) |
| 2 | `KVCacheBlock.reset_hash` | `kv_cache_utils.py:L144` | Identical logic |
| 3 | `KVCacheBlock.set_hash` | `block_pool.py:L211` (cache_full_blocks sets hash) | source uses property setter with assertion |
| 4 | `FreeKVCacheBlockQueue` | `kv_cache_utils.py:L162` | Doubly-linked list with sentinels |
| 5 | `FreeKVCacheBlockQueue.popleft` | `kv_cache_utils.py:L214` | O(1) LRU pop |
| 6 | `FreeKVCacheBlockQueue.popleft_n` | `kv_cache_utils.py:L251` | Batch pop |
| 7 | `FreeKVCacheBlockQueue.remove` | `kv_cache_utils.py:L284` | O(1) middle removal |
| 8 | `FreeKVCacheBlockQueue.append` | `kv_cache_utils.py:L304` | Append to tail (MRU) |
| 9 | `FreeKVCacheBlockQueue.append_n` | `kv_cache_utils.py:L327` | Batch append |
| 10 | `BlockPool` | `block_pool.py:L130` | Physical allocation + eviction |
| 11 | `BlockPool.get_new_blocks` | `block_pool.py:L322` | Allocate + maybe_evict |
| 12 | `BlockPool._maybe_evict_cached_block` | `block_pool.py:L354` | Evict hash on reallocation |
| 13 | `BlockPool.cache_full_blocks` | `block_pool.py:L211` | Set block_hash for prefix cache |
| 14 | `BlockPool.touch` | `block_pool.py:L391` | ref_cnt++ for shared blocks |
| 15 | `BlockPool.free_blocks` | `block_pool.py:L408` | Decrement ref_cnt, return to free queue |
| 16 | `BlockPool.evict_blocks` | `block_pool.py:L424` | Evict by block_id |
| 17 | `BlockHashToBlockMap` | `block_pool.py:L34` | Hash-to-block index (simplified to dict) |
| 18 | `KVCacheBlocks` | `kv_cache_manager.py:L22` | Simplified to flat list (source: tuple of groups) |
| 19 | `KVCacheBlocks.get_block_ids` | `kv_cache_manager.py:L65-L80` | Convert to raw block IDs |
| 20 | `KVCacheManager` | `kv_cache_manager.py:L106` | Flattened hierarchy (no coordinator) |
| 21 | `KVCacheManager.allocate_slots` | `kv_cache_manager.py:L225` | Three-stage allocation |
| 22 | `KVCacheManager.get_computed_blocks` | `kv_cache_manager.py:L183` | Prefix cache lookup |
| 23 | `KVCacheManager.cache_blocks` | `kv_cache_manager.py:L515` | Cache full blocks |
| 24 | `KVCacheManager.free` | `kv_cache_manager.py:L418` | Free in reverse order |
| 25 | `KVCacheConfig` | `kv_cache_interface.py:L760` | Simplified sizing (source: dataclass container) |
| 26 | `KVCacheConfig.calculate_num_blocks` | `kv_cache_interface.py:L145-L167` (page_size_bytes formula) | Educational simplification |

## Key Simplifications vs Production vLLM

1. **No KVCacheCoordinator / SingleTypeKVCacheManager**: Production vLLM has a three-tier architecture (KVCacheManager → Coordinator → SingleTypeManager). We flatten this into a single KVCacheManager for clarity.

2. **No multi-KV-cache-group support**: Production handles models with mixed attention types (e.g., full-attention + sliding-window). We assume a single uniform attention type.

3. **BlockHash simplified to int**: Production uses `BlockHashWithGroupId` (bytes with embedded group ID) and SHA256/CBOR hashing. We use plain `int` for educational clarity.

4. **Null block as sentinel**: Production pops a real block from the free queue and marks it null. We create a synthetic block with block_id=-1. This means our `get_usage()` denominator is `num_gpu_blocks` (not `num_gpu_blocks - 1`).

5. **No KV cache events / metrics**: Production emits events for distributed KV cache synchronization and metrics collection. We omit these for simplicity.

6. **No EAGLE / speculative decode support**: Production reserves lookahead tokens for draft model KV cache. We omit this.

7. **Single free list init**: Production initializes the free list by linking consecutive blocks directly. We use sequential `append()` calls, which produces the same ordering.
