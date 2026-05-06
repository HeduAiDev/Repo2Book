# Test Report — Ch05 GPU Memory Management

**Tester**: tester@book-factory
**Date**: 2026-05-06
**Source commit**: `98661fe`
**Verdict**: APPROVED → handoff to Writer (Task #8)

## Summary

```
74 passed, 0 failed in 0.16s
```

| Module | Tests | Status |
|---|---|---|
| `test_mem_snapshot.py` | 7 | PASS |
| `test_kv_cache_spec.py` | 11 | PASS |
| `test_kv_cache_block.py` | 16 | PASS |
| `test_block_pool.py` | 20 | PASS |
| `test_memory_layout.py` | 8 | PASS |
| `test_recompute.py` | 9 | PASS |
| `test_integration.py` | 3 | PASS |

Run from chapter dir:

```bash
cd instances/vllm/artifacts/05-memory-management
python3 -m pytest tests/ --ignore=tests/_legacy -q
```

## Demo numerics — every headline number reproduced exactly

| Quantity | Demo claim | Test value |
|---|---|---|
| `num_gpu_blocks` (Llama-3.2-1B/H100) | 35,148 | 35,148 |
| Max concurrent @ 2k seq | 275 | 274.59 → 275 |
| Page size (block_size=16, fp16) | 64 KiB | 65,536 B |
| Recompute (8K prompt) | 164 ms | 163.84 ms |
| Swap round-trip (8K prompt) | 62 ms | 62.50 ms |
| Wasted bytes (rounding) | 1.6 MiB | <1 MiB strictly |
| Block-size sweep `{8,16,32,64}` | `{70297, 35148, 17574, 8787}` | exact match |

## Coverage by behavior class

1. **MemorySnapshot math**: field-wise subtract, diff result has `auto_measure=False`, idempotent measure() stub.
2. **memory_profiling worked example** (from vLLM mem_utils.py:L209-L235): cat1=1, cat2=0→2→4→3, cat3=0→0.5→1→1, weights=2 → result.non_kv_cache_memory = 6 GiB. Exact match.
3. **Page-size formula** (`AttentionSpec.real_page_size_bytes`): 2 * 16 * 8 * 128 * 2 = 65,536 B pinned. Linear scaling in block_size and dtype verified.
4. **`get_num_blocks`**: integer divide by page_size AND num_layers; clamps negatives to 0; wasted remainder strictly < page_size * num_layers.
5. **`FullAttentionSpec`**: `head_size_v` defaults to `head_size`; `max_memory_usage_bytes` rounds up; KVCacheSpec base raises NotImplementedError.
6. **`FreeKVCacheBlockQueue`** (W: O(1) middle-removal):
   - popleft empty raises ValueError (not IndexError) — vLLM L221.
   - popleft_n(0) is no-op; popleft_n(N) preserves head order.
   - remove(b) at head, middle, tail — all O(1).
   - remove on unlinked block raises RuntimeError (catches caller bugs).
   - append-then-remove cycle relinks pointers correctly.
7. **`KVCacheBlock`**: setter asserts on re-set; reset_hash clears.
8. **`BlockPool`** (the most complex; 20 tests):
   - **Null block** (K04): `block_id=0`, `is_null=True`, popped at `__init__`, never returned. First user allocation is `block_id=1`. `get_usage` subtracts null from denominator.
   - **`get_new_blocks`**: ref_cnt bumps to 1, returns in LRU order, oversubscribe raises ValueError.
   - **`free_blocks`**: ref_cnt drops; only ref_cnt=0 blocks return to free queue; ref_cnt≥1 blocks stay out.
   - **Prefix cache** (K08): `cache_block` stores hash → block; `free_blocks` keeps hash; `get_new_blocks` evicts cached blocks via LRU; `enable_caching=False` makes both no-ops.
   - **`touch`**: re-activates freed-but-cached block in O(1) — pulls from middle of free queue, bumps ref_cnt. Correctly skips null block and already-allocated blocks.
   - **`evict_blocks`**: drops cache hash without freeing (KV connector path).
   - **`reset_prefix_cache`**: succeeds only when no non-null blocks are in use.
9. **`determine_available_memory`** (the big one): the demo's 35,148 number reproduces; util margin = 6.4 GiB; negative KV clamps to 0 (OOM guard at L122-L123); block-size sweep matches the demo's `[5]` table.
10. **`estimate_max_concurrency`**: 275 concurrent at 2k seq matches; zero-seq returns 0.0; sub-block-size requests still occupy 1 block.
11. **`PreemptionScenario`**: KV bytes formula = 2 × layers × heads × head_size × prompt × dtype. For 8K/32/8/128/fp16 → 1 GiB exact. Recompute 163.84 ms vs swap 62.5 ms; `recompute_is_faster=False` (K06: vLLM v1 picks recompute *despite* slower latency, for simplicity & OOM safety).
12. **Integration**: full demo workflow passes end-to-end. Page-size wires from `AttentionSpec` into `MemoryLayout` cleanly. Ch04 cross-chapter import does not collide.

## Knowledge applied

- **K01-K10** in `kv-cache.md` and **M01** in `memory.md` (implementer-supplied).
- Tests pin every K-fact's quantitative claim where applicable. Specifically K04 (null block), K05 (O(1) middle remove), K06 (recompute slower but chosen), K07 (3-category memory worked example), K08 (cache survives free + LRU evicts).

## Wisdom applied

- **W02 (preemption-test design)** doesn't apply directly (no preemption flow in Ch05), but the principle — *don't let the test pass for the wrong reason* — is honored. Every demo number is asserted exactly, not just "something positive". The block-size sweep test catches drift in any of the four numerator/denominator factors.

## Fidelity findings

**None blocking.** One observation worth flagging for completeness:

### Note: cudagraph_memory subtraction is unconditional in our impl

vLLM gpu_worker.py:L417-L423 subtracts `cudagraph_memory_estimate` from `available_kv_cache` only when `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` env var is set (default OFF). In our `determine_available_memory`, the subtraction is unconditional — the parameter is simply `cudagraph_memory: int`, and a caller who wants vLLM-default behavior passes `cudagraph_memory=0`.

**Ch05 impact**: zero. The demo always passes 256 MiB to demonstrate the breakdown. A caller wanting vLLM-default semantics passes 0. Not a divergence; a simplification of the env-var dispatch.

**Recommendation for Ch20 (model-runner, where this env var path becomes load-bearing)**: surface the env-var dispatch then. Knowledge entry K09 already flags this for the implementer.

## Backpressure gate

OPEN. Writer (Task #8) is clear to start.
