# REFERENCE: instances/vllm/source/vllm/v1/worker/gpu_worker.py:L352-L505
"""Annotated runnable trace of the full memory-management pipeline.

Usage:
    python3 -m instances.vllm.artifacts.05-memory-management.implementation.demo

What it does:
    1. Build a Llama-3.2-1B-style memory profile on a synthetic 80 GiB H100.
    2. Walk the breakdown: weights, peak activation, non-torch, CUDA graphs.
    3. Compute available KV cache and num_gpu_blocks via determine_available_memory.
    4. Allocate from a BlockPool, observe LRU + watermark behaviour.
    5. Free, re-allocate, and confirm the prefix-cache eviction path.
    6. Compare recompute vs swap latency for one request's KV cache.

All numbers are deterministic and reproducible.
"""

from __future__ import annotations

import hashlib

from .block_pool import BlockPool
from .kv_cache_spec import FullAttentionSpec
from .mem_snapshot import MemoryProfilingResult, MemorySnapshot, format_gib
from .memory_layout import determine_available_memory, estimate_max_concurrency
from .recompute import PreemptionScenario


GIB = 1024**3
MIB = 1024**2


def _llama_3_2_1b_profile() -> tuple[MemorySnapshot, MemoryProfilingResult, int]:
    """Build representative numbers for a Llama-3.2-1B forward on an H100.

    Numbers are chosen to be plausible (matching what one would see in a real
    vLLM log) without requiring a GPU to run the demo.

    REFERENCE: instances/vllm/source/vllm/v1/worker/gpu_worker.py:L386-L427
    (memory_profiling context manager populates these fields)
    """
    init = MemorySnapshot(
        total_memory=80 * GIB,
        free_memory=78 * GIB,  # 2 GiB already used by other processes
        cuda_memory=2 * GIB,
        torch_memory=0,
        non_torch_memory=2 * GIB,
        timestamp=0.0,
    )

    profile = MemoryProfilingResult(
        before_create=init,
        weights_memory=int(2.4 * GIB),       # ~2.4 GiB for 1B params @ bf16 + lm_head
        torch_peak_increase=int(1.8 * GIB),  # peak activation during max-batch profile
        non_torch_increase=int(0.5 * GIB),   # NCCL + attention workspace
        profile_time=2.13,
    )
    cudagraph_memory = 256 * MIB             # CUDA graph capture pool

    return init, profile, cudagraph_memory


def run_demo() -> None:
    print("=" * 64)
    print("Ch05 — GPU Memory Management — annotated trace")
    print("=" * 64)

    # ── Section 1 — memory layout for Llama-3.2-1B on H100 (80 GiB) ──────
    print("\n[1] Llama-3.2-1B on H100 (80 GiB) — memory layout")
    print("    Source: vllm/v1/worker/gpu_worker.py:determine_available_memory()")
    init_snap, profile, cg_mem = _llama_3_2_1b_profile()

    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype_bytes=2,
    )
    layout = determine_available_memory(
        init_snapshot=init_snap,
        profile_result=profile,
        cudagraph_memory=cg_mem,
        gpu_memory_utilization=0.92,
        spec=spec,
        num_layers=32,
    )
    print(layout.report())

    # ── Section 2 — concurrency estimate ────────────────────────────────
    print("\n[2] Max concurrent requests at avg_seq_len=2048")
    conc = estimate_max_concurrency(layout, avg_seq_len=2048, block_size=16)
    print(f"    {conc:.0f} concurrent requests fit in the KV cache")
    print(f"    Source: vllm/v1/core/kv_cache_utils.py:get_max_concurrency_for_kv_cache_config")

    # ── Section 3 — BlockPool LRU + prefix cache ────────────────────────
    print("\n[3] BlockPool LRU + prefix cache (mini scenario, 16 blocks)")
    print("    Source: vllm/v1/core/block_pool.py")
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True)
    print(f"    Initial: free={pool.get_num_free_blocks()}/{pool.num_gpu_blocks - 1}, "
          f"usage={pool.get_usage():.1%}, "
          f"null_block_id={pool.null_block.block_id}")

    # Request A: 4 blocks
    a_blocks = pool.get_new_blocks(4)
    print(f"    A allocates 4: ids={[b.block_id for b in a_blocks]}, "
          f"free={pool.get_num_free_blocks()}, usage={pool.get_usage():.1%}")

    # Cache the first two blocks of A under prefix-hashes h1, h2.
    h1 = hashlib.sha256(b"req-A:block-0").digest()
    h2 = hashlib.sha256(b"req-A:block-1").digest()
    pool.cache_block(a_blocks[0], h1)
    pool.cache_block(a_blocks[1], h2)
    print(f"    A caches blocks 0,1 under hashes h1,h2: cache_size="
          f"{len(pool.cached_block_hash_to_block)}")

    # A finishes → free its blocks. Cached ones STAY in the cache.
    pool.free_blocks(a_blocks)
    print(f"    A frees 4: free={pool.get_num_free_blocks()}, "
          f"cache_still_holds={len(pool.cached_block_hash_to_block)}")

    # Request B prefix-hits h1 → touch() re-activates that block.
    hit = pool.get_cached_block(h1)
    assert hit is not None
    pool.touch([hit])
    print(f"    B hits prefix h1 via touch(): block={hit.block_id}, "
          f"ref_cnt={hit.ref_cnt}, free={pool.get_num_free_blocks()}")

    # Request C asks for many blocks → forces eviction of cached blocks.
    c_blocks = pool.get_new_blocks(10)
    print(f"    C allocates 10: free={pool.get_num_free_blocks()}, "
          f"cache_after_evictions={len(pool.cached_block_hash_to_block)}")

    # ── Section 4 — recompute vs swap latency ───────────────────────────
    print("\n[4] Preemption: recompute vs. swap — for an 8K-token request")
    print("    Source: vllm/v1/core/sched/scheduler.py:L952-L972 (recompute path)")
    scenario = PreemptionScenario(
        prompt_tokens=8192,
        num_layers=32,
        num_kv_heads=8,
        head_size=128,
        dtype_bytes=2,
    )
    print(scenario.report())

    # ── Section 5 — page-size sensitivity ───────────────────────────────
    print("[5] page_size sensitivity (block_size sweep)")
    print("    Source: vllm/v1/kv_cache_interface.py:AttentionSpec.real_page_size_bytes")
    for bs in (8, 16, 32, 64):
        s = FullAttentionSpec(block_size=bs, num_kv_heads=8, head_size=128, dtype_bytes=2)
        layout_bs = determine_available_memory(
            init_snap, profile, cg_mem, 0.92, s, 32
        )
        print(f"    block_size={bs:3d}: page={s.page_size_bytes / 1024:5.1f} KiB, "
              f"num_blocks={layout_bs.num_gpu_blocks:6d}, "
              f"wasted={layout_bs.wasted_bytes / MIB:.2f} MiB")


if __name__ == "__main__":
    run_demo()
