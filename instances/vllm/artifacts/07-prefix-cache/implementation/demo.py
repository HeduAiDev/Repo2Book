# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_manager.py
"""Annotated runnable trace for Ch07 — prefix cache.

Usage:
    python3 -m instances.vllm.artifacts.07-prefix-cache.implementation.demo

Five sections (matching outline):

    [1] System-prompt reuse rate at varying shared-prefix fractions
    [2] Radix tree vs hash table — match-throughput microbench
    [3] match / insert / evict trace on a 3-request workload
    [4] Prefix-aware allocation savings vs naive (KV blocks reused per req)
    [5] Composition: prefix cache + PagedAttention invariant verification
"""

from __future__ import annotations

import time
from typing import Sequence

from .block_hash import chain_block_hashes, common_prefix_length
from .paged_integration import (
    allocate_with_prefix_cache,
    get_computed_blocks,
    verify_invariants,
)
from .prefix_cache_manager import PrefixCacheManager, prefix_aware_allocate
from .prefix_cache_index import BlockHashToBlockMap
from .radix_tree import RadixTree, Trie


def _section(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print("=" * 64)


def section_1_hit_rate_sweep() -> None:
    _section("[1] System-prompt reuse rate vs shared-prefix fraction")
    print("    Source: vllm/v1/core/single_type_kv_cache_manager.py:L446-L494")
    print()
    print(f"    {'shared_frac':>12}  {'requests':>10}  {'cached_tokens':>14}  {'hit_rate':>10}")

    block_size = 16
    num_requests = 100
    total_tokens_per_request = 1024  # tokens

    for shared_frac in (0.10, 0.30, 0.50, 0.70, 0.90):
        mgr = PrefixCacheManager(block_size=block_size)
        shared_len = int(total_tokens_per_request * shared_frac)
        # All requests share the first `shared_len` tokens.
        shared_prefix = list(range(1, shared_len + 1))

        total_cached_tokens = 0
        for r in range(num_requests):
            unique_tail = list(
                range(10000 + r * 1000, 10000 + r * 1000 + total_tokens_per_request - shared_len)
            )
            tokens = shared_prefix + unique_tail
            res = get_computed_blocks(mgr, tokens)
            total_cached_tokens += res.num_cache_hit_tokens
            # Allocate so future requests can hit.
            allocate_with_prefix_cache(mgr, f"r{r}", tokens)

        possible_total = num_requests * total_tokens_per_request
        hit_rate = total_cached_tokens / possible_total
        print(
            f"    {shared_frac:>12.2f}  {num_requests:>10d}  "
            f"{total_cached_tokens:>14d}  {hit_rate:>9.1%}"
        )


def section_2_radix_vs_hash() -> None:
    _section("[2] Radix tree vs hash table — match-throughput microbench")
    print("    Source: vllm uses a hash table (block_pool.py:L34-L127);")
    print("    the radix tree implementation here is pedagogical (Ch07.radix_tree)")
    print()

    # Build same dataset for both: 1000 prefixes of varying length.
    block_size = 16
    num_prefixes = 1000
    prefixes: list[list[int]] = []
    for i in range(num_prefixes):
        L = ((i * 31) % 256) + 16  # variable length 16..271
        prefixes.append(list(range(i * 1000, i * 1000 + L)))

    # Insert.
    trie = Trie()
    radix = RadixTree()
    hash_idx = BlockHashToBlockMap()
    for i, p in enumerate(prefixes):
        trie.insert(p, i)
        radix.insert(p, i)

    # For the hash table, we need block hashes per chunked block.
    # That's the actual cost we want to measure: a chained hash + dict get.
    chains = [chain_block_hashes(p, block_size) for p in prefixes]
    for i, chain in enumerate(chains):
        for j, h in enumerate(chain):
            from .block_hash import make_block_hash_with_group_id
            packed = make_block_hash_with_group_id(h, 0)
            hash_idx.insert(packed, i * 1000 + j)

    # Query: same 1000 prefixes (best case for all).
    n_lookups = 5000  # repeat for stable timing

    t0 = time.perf_counter()
    for _ in range(n_lookups):
        for p in prefixes:
            trie.find_longest_prefix(p)
    t_trie = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_lookups):
        for p in prefixes:
            radix.find_longest_prefix(p)
    t_radix = time.perf_counter() - t0

    from .block_hash import make_block_hash_with_group_id
    t0 = time.perf_counter()
    for _ in range(n_lookups):
        for chain in chains:
            for h in chain:
                hash_idx.get_one_block(make_block_hash_with_group_id(h, 0))
    t_hash = time.perf_counter() - t0

    total_lookups = n_lookups * num_prefixes
    print(f"    {'data structure':<20}  {'total_time':>12}  {'lookups/sec':>14}  {'rel speed':>10}")
    print(f"    {'Trie':<20}  {t_trie:>10.3f}s  "
          f"{total_lookups / t_trie:>13.0f}/s  {1.0:>10.2f}x")
    print(f"    {'RadixTree':<20}  {t_radix:>10.3f}s  "
          f"{total_lookups / t_radix:>13.0f}/s  {t_trie / t_radix:>10.2f}x")
    print(f"    {'Hash table (vLLM)':<20}  {t_hash:>10.3f}s  "
          f"{total_lookups / t_hash:>13.0f}/s  {t_trie / t_hash:>10.2f}x")
    print()
    print("    -> Hash table wins on per-block lookup. Trade-off: cannot")
    print("       enumerate prefixes (radix tree's only advantage).")


def section_3_match_insert_evict() -> None:
    _section("[3] match / insert / evict — three-request trace")
    mgr = PrefixCacheManager(block_size=4, group_id=0)

    # System prompt (8 tokens = 2 blocks).
    sys_prompt = [101, 102, 103, 104, 105, 106, 107, 108]

    # Request A: sys_prompt + own (8 + 8 = 16 tokens = 4 blocks)
    req_a = sys_prompt + [201, 202, 203, 204, 205, 206, 207, 208]
    res_a = allocate_with_prefix_cache(mgr, "A", req_a)
    print(f"    A allocates {len(res_a.block_ids)} blocks "
          f"(hit {res_a.num_cache_hit_blocks}, fresh {res_a.num_fresh_blocks})")

    # Request B: same sys_prompt + own (should HIT 2 blocks)
    req_b = sys_prompt + [301, 302, 303, 304, 305, 306, 307, 308]
    res_b = allocate_with_prefix_cache(mgr, "B", req_b)
    print(f"    B allocates {len(res_b.block_ids)} blocks "
          f"(hit {res_b.num_cache_hit_blocks}, fresh {res_b.num_fresh_blocks})")
    print(f"    -> B reused A's blocks {res_b.block_ids[:res_b.num_cache_hit_blocks]}")

    # Request C: different prefix, no hits.
    req_c = [501, 502, 503, 504, 505, 506, 507, 508]
    res_c = allocate_with_prefix_cache(mgr, "C", req_c)
    print(f"    C allocates {len(res_c.block_ids)} blocks "
          f"(hit {res_c.num_cache_hit_blocks}, fresh {res_c.num_fresh_blocks})")

    # Free A — its uniquely-owned blocks return to pool, but cached blocks remain.
    freed = mgr.free_request("A")
    print(f"    A frees: {freed}; cache still has {len(mgr.cache)} entries")

    # Evict the FIRST cached block of A.
    chain_a = chain_block_hashes(req_a, mgr.block_size)
    evicted = mgr.evict_block(res_a.block_ids[0], chain_a[0])
    print(f"    Evict A's block 0: success={evicted}, cache size={len(mgr.cache)}")

    # Request D: same sys_prompt — block 0 will MISS now (was evicted),
    # but block 1 is still cached.
    req_d = sys_prompt + [401, 402, 403, 404, 405, 406, 407, 408]
    res_d = allocate_with_prefix_cache(mgr, "D", req_d)
    print(f"    D allocates {len(res_d.block_ids)} blocks (after eviction):")
    print(f"      hit={res_d.num_cache_hit_blocks} — chain BROKE at evicted block 0")
    print(f"      (chain invariant: 1st miss → no later hit, even though block 1 lives)")


def section_4_prefix_aware_savings() -> None:
    _section("[4] Prefix-aware vs naive allocation — KV-block savings")
    print("    Source: vllm/v1/core/kv_cache_manager.py:L183-L223")
    print()
    block_size = 16
    num_requests = 50
    sys_prompt_tokens = 512   # 32 blocks
    user_tokens = 128         # 8 blocks
    sys_prompt = list(range(1, sys_prompt_tokens + 1))

    # Prefix-aware allocator.
    mgr = PrefixCacheManager(block_size=block_size)
    pa_blocks = 0
    for r in range(num_requests):
        tokens = sys_prompt + list(
            range(10000 + r * 1000, 10000 + r * 1000 + user_tokens)
        )
        res = allocate_with_prefix_cache(mgr, f"r{r}", tokens)
        pa_blocks += res.num_fresh_blocks

    # Naive: each request allocates its full sequence.
    naive_blocks = num_requests * (sys_prompt_tokens + user_tokens) // block_size

    saved = naive_blocks - pa_blocks
    print(f"    {'allocation':<18}  {'fresh_blocks':>14}  {'kv_bytes_at_64KiB':>20}")
    print(f"    {'naive':<18}  {naive_blocks:>14d}  "
          f"{naive_blocks * 64 / 1024:>17.2f} MiB")
    print(f"    {'prefix-aware':<18}  {pa_blocks:>14d}  "
          f"{pa_blocks * 64 / 1024:>17.2f} MiB")
    print(f"    saved:        {saved:>14d}  "
          f"{saved * 64 / 1024:>17.2f} MiB  ({saved / naive_blocks:.0%})")


def section_5_invariants() -> None:
    _section("[5] PagedAttention + prefix cache — invariant verification")
    print("    Source: vllm/v1/core/kv_cache_manager.py + block_pool.py")
    print()
    mgr = PrefixCacheManager(block_size=16)
    sys_prompt = list(range(1, 513))
    for r in range(10):
        tokens = sys_prompt + list(range(10000 + r, 10000 + r + 128))
        allocate_with_prefix_cache(mgr, f"r{r}", tokens)

    inv = verify_invariants(mgr)
    for k, v in inv.items():
        print(f"    {k:<32}  {'PASS' if v else 'FAIL'}")
    print()
    print(f"    cache_size: {len(mgr.cache)} entries")
    print(f"    requests:   {len(mgr.req_to_blocks)} alive")


def run_demo() -> None:
    print("=" * 64)
    print("Ch07 — Automatic Prefix Caching (APC) — annotated trace")
    print("=" * 64)
    section_1_hit_rate_sweep()
    section_2_radix_vs_hash()
    section_3_match_insert_evict()
    section_4_prefix_aware_savings()
    section_5_invariants()


if __name__ == "__main__":
    run_demo()
