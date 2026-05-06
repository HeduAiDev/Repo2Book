# Kv Cache Knowledge

---

## K01: K01: Page size formula = 2 * block_size * num_kv_heads * head_size * dtype_bytes

**Module**: kv-cache
**Chapter**: 05-memory-management
**Discovered by**: implementer
**TTL**: 2026-06-04
**Access count**: 0
**Tags**: kv-cache, page-size, formula

vllm/v1/kv_cache_interface.py:L153-L170 (AttentionSpec.real_page_size_bytes). The leading 2 is K and V interleaved per-token. `head_size` (vLLM's name), not `head_dim`. dtype_bytes via `get_dtype_size(dtype)`. nvfp4 uses a different packed layout (L154-L163). page_size_padded (L147-L150) overrides the result for alignment-padded specs. Per-token-head kv-quant adds 2 * block_size * num_kv_heads * fp32_bytes (L143-L146).

---

## K02: K02: get_num_blocks divides by page_size AND num_layers

**Module**: kv-cache
**Chapter**: 05-memory-management
**Discovered by**: implementer
**TTL**: 2026-06-04
**Access count**: 0
**Tags**: kv-cache, block-count, formula

vllm/v1/core/kv_cache_utils.py:L930-L947. The formula is `num_blocks = available_memory // page_size // num_layers`. Splitting the layer factor out (rather than rolling it into page_size) lets heterogeneous-layer models (mamba mixed with attention) vary block count per cache group. Wasted bytes are typically < page_size_bytes, ~ 1-2 MiB on H100.

---

## K03: K03: gpu_memory_utilization defaults to 0.92, the 8% margin matters

**Module**: kv-cache
**Chapter**: 05-memory-management
**Discovered by**: implementer
**TTL**: 2026-06-04
**Access count**: 0
**Tags**: memory, config, default

vllm/config/cache.py default. The 8% margin (1.0 - 0.92) is NOT just a safety buffer — it specifically absorbs (a) PyTorch caching allocator fragmentation, (b) activation peaks above the profile run, (c) CUDA context overhead invisible to torch.accelerator.memory_reserved(), (d) sampler-time transient spikes. Reducing below 0.92 causes OOM under realistic workloads. Increasing above ~0.95 makes startup brittle.

---

## K04: K04: BlockPool reserves block 0 as the null_block at startup

**Module**: kv-cache
**Chapter**: 05-memory-management
**Discovered by**: implementer
**TTL**: 2026-06-04
**Access count**: 0
**Tags**: block-pool, null-block, gotcha

vllm/v1/core/block_pool.py:L173-L177. On construction, `null_block = free_block_queue.popleft()` removes block 0 from the free list and sets `is_null=True`. The null block is never returned to the queue. WHY: sliding-window attention pads the block table with the null block_id when blocks fall outside the window. A test that checks 'first allocated block_id' should expect 1, not 0. `BlockPool.get_usage()` subtracts 1 from total_gpu_blocks to account for this (L494).

---

## K05: K05: FreeKVCacheBlockQueue is doubly-linked, not deque, for O(1) middle removal

**Module**: kv-cache
**Chapter**: 05-memory-management
**Discovered by**: implementer
**TTL**: 2026-06-04
**Access count**: 0
**Tags**: block-queue, data-structure, performance

vllm/v1/core/kv_cache_utils.py:L162-L370. The fake_free_list_head/tail eliminate null-pointer branching. Unlike `collections.deque` (which is O(n) for middle removal), this implementation supports `remove(block)` in O(1). This matters for `BlockPool.touch()` (block_pool.py:L391-L406) — when a freed-but-cached block hits the prefix cache, it must be O(1)-removed from the middle of the free queue. Using deque would make the prefix-cache fast path O(n).

---

## K06: K06: v1 has no swap-to-CPU preemption; recompute is the only path

**Module**: kv-cache
**Chapter**: 05-memory-management
**Discovered by**: implementer
**TTL**: 2026-06-04
**Access count**: 0
**Tags**: preemption, v1-design, swap-vs-recompute

vllm/v1/core/sched/scheduler.py:L952-L972. _preempt_request frees blocks and resets num_computed_tokens=0 (full re-prefill). vLLM v0 had a swap path; v1 removed it. The `kv_offload/` subsystem (vllm/v1/kv_offload/) is for prefix-cache offload to CPU/disk, NOT preemption — `swap_blocks_batch` at kv_offload/cpu/gpu_worker.py:L319 is invoked only by the offload manager, never by the scheduler on preempt. For 8K-token prompts, recompute (~164 ms @ 50K tok/s prefill) is ~2.6x slower than swap round-trip (~62 ms over PCIe Gen4), but vLLM picked recompute for simplicity, OOM-safety, and bit-determinism.

---

## K07: K07: memory_profiling categorizes GPU memory into 3 categories (NOT 2)

**Module**: kv-cache
**Chapter**: 05-memory-management
**Discovered by**: implementer
**TTL**: 2026-06-04
**Access count**: 0
**Tags**: memory, profiling, category

vllm/utils/mem_utils.py:L204-L235 (docstring). The split is: (1) memory used by other processes on the same GPU; (2) memory used by torch in this vLLM instance; (3) memory used by this vLLM instance but NOT by torch (NCCL, attention workspace, CUDA context). Category 3 is opaque to PyTorch but very real, hence the explicit `non_torch_memory = cuda_memory - torch_memory` calculation at mem_utils.py:L125. Without category 3, the available_kv_cache calculation underestimates non-cache usage by ~0.5-1 GiB on a typical H100 setup.

---

## K08: K08: cache_block KEEPS block in free queue when ref_cnt drops to 0

**Module**: kv-cache
**Chapter**: 05-memory-management
**Discovered by**: implementer
**TTL**: 2026-06-04
**Access count**: 0
**Tags**: block-pool, prefix-cache, lifecycle

vllm/v1/core/block_pool.py:L408-L422 (free_blocks). When a request frees its blocks and ref_cnt hits 0, the block is `append_n`-ed back to the free queue BUT remains in `cached_block_hash_to_block` if it had a hash. A future request that prefix-cache-hits the same hash calls `touch()` to bump ref_cnt and remove it from the queue — that's where the O(1) middle-removal pays off. Eviction only happens when `get_new_blocks` pops the block off the LRU head AND the block was cached: then `_maybe_evict_cached_block` clears the hash entry.

---

## K09: K09: determine_available_memory subtracts cudagraph_memory only when VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS opt-in

**Module**: kv-cache
**Chapter**: 05-memory-management
**Discovered by**: implementer
**TTL**: 2026-06-04
**Access count**: 0
**Tags**: memory, cudagraph, env-var, gotcha

vllm/v1/worker/gpu_worker.py:L417-L423. The cudagraph estimate is computed unconditionally on CUDA (skipped on ROCm/HIP/XPU because mem_get_info behaves differently there), but it's only subtracted from available_kv_cache if the env var `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` is set. Default is OFF, meaning by default cudagraph memory comes out of the 8% safety margin, not the explicit budget. If a user sees OOM with cudagraphs enabled, suggesting they set this env var is the standard fix.

---

## K10: K10: KVCacheManager.allocate_slots first arg is Request, additional args are prefix-cache/spec-decode plumbing

**Module**: kv-cache
**Chapter**: 05-memory-management
**Discovered by**: implementer
**TTL**: 2026-06-04
**Access count**: 0
**Tags**: kv-cache-manager, api, for-ch12

vllm/v1/core/kv_cache_manager.py:L225-L416. Full signature has 9 args. The first is Request. The rest (num_new_computed_tokens, new_computed_blocks, num_lookahead_tokens, num_external_computed_tokens, delay_cache_blocks, num_encoder_tokens, full_sequence_must_fit) are all optional and zero-by-default. For Ch05's pedagogical BlockPool we expose `get_new_blocks(num_blocks)` directly. For Ch12-13 the full signature must be reproduced. The 'block layout' diagram at L262-L283 is the canonical reference for what each arg means.

---

## K11: Testing BlockPool — null-block-aware sizing

**Module**: kv-cache
**Chapter**: 05-memory-management
**Discovered by**: tester (Ch05 v6 test pass)
**TTL**: permanent
**Access count**: 1
**Tags**: testing, block-pool, null-block

When sizing a `BlockPool(num_gpu_blocks=N)` for a unit test, the *usable* count is `N - 1` (block 0 is reserved as the null block at `__init__`). Three concrete patterns:

1. To test "first allocation returns block_id=1": `BlockPool(num_gpu_blocks=8)` → `get_new_blocks(1)` returns `[block_id=1]`.
2. To test "oversubscribe raises": with `num_gpu_blocks=4`, the free count is 3 (4 minus null). `get_new_blocks(4)` raises ValueError.
3. To test `get_usage()`: denominator is `num_gpu_blocks - 1`. With `num_gpu_blocks=11`, allocating 5 gives `get_usage() == 0.5`, not `5/11 = 0.45`.

If you write `assert pool.get_num_free_blocks() == num_gpu_blocks` after construction, the test fails (off-by-one). Always assert `num_gpu_blocks - 1`.

---

## K12: cudagraph_memory subtraction is unconditional in our impl, gated by env var in vLLM

**Module**: kv-cache
**Chapter**: 05-memory-management, 20-model-runner
**Discovered by**: tester (Ch05 v6 test pass)
**TTL**: permanent
**Access count**: 1
**Tags**: memory, cudagraph, simplification, for-ch20

Our `determine_available_memory(...)` accepts `cudagraph_memory: int` and unconditionally subtracts it from `available_kv_cache`. vLLM `gpu_worker.py:L417-L423` gates this subtraction on the env var `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` (default OFF). When OFF, cudagraph memory comes out of the 8% util margin instead.

For Ch05: harmless. The demo passes 256 MiB to demonstrate the breakdown; a caller mimicking vLLM-default semantics passes 0. For Ch20 (model-runner), surface the env-var dispatch — without it, the worker over-budgets KV cache when cudagraphs are enabled and the user sees OOM with the wrong error.

For testers: don't write a Ch05 test that asserts `cudagraph_memory == 0 when env var unset` — there's no env var path in our impl. The unit test `test_breakdown_matches_demo` verifies the `requested - weights - peak - non_torch - cg` arithmetic; that's the right level of fidelity for the chapter's scope.

---

## K13: Page-size formula vs KV-bytes-per-request — same shape, different unit

**Module**: kv-cache (writing-perspective)
**Chapter**: 05-memory-management
**Discovered by**: writer (Ch05 v6 A1 rewrite)
**TTL**: permanent
**Access count**: 1
**Tags**: kv-cache, page-size, recompute, formula

`AttentionSpec.real_page_size_bytes` (`kv_cache_interface.py:L153-L170`) and `PreemptionScenario.kv_bytes` (`recompute.py:L62-L76`) have identical 5-factor structure:

```
2 × num_kv_heads × head_size × dtype_bytes × {block_size | prompt_tokens}
```

The ONLY difference is the last factor: page_size uses `block_size` (16 tokens), kv_bytes uses `prompt_tokens` (whole request). Page_size is also per-layer; kv_bytes is for the full model (multiplied by `num_layers` separately).

**For writers**: when explaining recompute KV cost, point this back at the page_size derivation in 5.3.2 — readers don't have to learn a new formula, just substitute one factor. Saves ~200 words and one numerical-trace reduction. This is the canonical "same formula, different scale" teaching pattern for any chapter that touches both block-level and request-level KV math.

**For testers**: cross-check that `kv_bytes(prompt_tokens=N) == page_size_bytes × num_layers × ceil(N / block_size)` ± alignment slack. The Ch05 demo's 8K-prompt case: `kv_bytes = 1 GiB; page_size × num_layers × 512 blocks = 64 KiB × 32 × 512 = 1 GiB exactly` — tight, no waste because 8192 is divisible by 16.

---

## K14: BlockPool demo trace — "cache_after_evictions=2" is NOT a bug

**Module**: kv-cache (writing-perspective)
**Chapter**: 05-memory-management
**Discovered by**: writer (Ch05 v6 A1 rewrite — got confused by demo output)
**TTL**: permanent
**Access count**: 1
**Tags**: block-pool, prefix-cache, demo-output, gotcha

In the Ch05 demo's section [3], after `C allocates 10` the line reads `cache_after_evictions=2`. Naive expectation: "C allocated 10 blocks, all of which were freed-but-cached, so cache should be empty / 0". Wrong.

Actual mechanics: at `C.get_new_blocks(10)` time the free queue contains a mix of NEVER-cached blocks (5..15, only 11 of them) and just-freed-but-cached blocks (the 4 from A, in some order). LRU `popleft_n(10)` pops the OLDEST blocks first — which are the 11 originally-untouched blocks 5..15. Block 1 (cached as h1) was already removed by B's `touch()`. Block 2 (cached as h2) is still sitting in the free queue but NOT popped by C because LRU order pulled the never-cached blocks first.

So h1 stays cached (block 1 is in B's hands, ref_cnt=1) and h2 stays cached (block 2 is in free queue, untouched by C). cache_size = 2.

**For writers explaining BlockPool**: the LRU + cache-survives-free interaction is **non-obvious from any single rule**. Walking the demo step-by-step (track the free queue contents as a literal list) is mandatory — handwaving "C evicts cached blocks" is wrong because LRU may pop never-cached blocks first.

**For reviewers**: when the writer draws a free-queue state-evolution diagram, verify the order matches `[never-cached blocks first, then freed-but-cached]` per LRU semantics. If a writer shows cached blocks getting popped first, that's a bug in the explanation.
