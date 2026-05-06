# REFERENCE: instances/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py
"""PrefixCacheManager — match / insert / evict orchestration.

Three operations from the outline (§3 "APC 的 match/insert/evict"):

    match  ↔ FullAttentionManager.find_longest_cache_hit
            (single_type_kv_cache_manager.py:L446-L494)
    insert ↔ SingleTypeKVCacheManager.cache_blocks
            (single_type_kv_cache_manager.py:L277-L301) which delegates to
            BlockPool.cache_full_blocks (block_pool.py:L211-L320)
    evict  ↔ BlockPool._maybe_evict_cached_block + BlockPool.evict_blocks
            (block_pool.py:L354-L389, L424-L441)

vLLM's design has two layers:
    SingleTypeKVCacheManager   — per-attention-spec policy (window vs full)
    BlockPool                  — physical block lifecycle (alloc/free/cache)

Ch07 collapses them into one `PrefixCacheManager` because for the prefix-
cache concept itself the layering is incidental. Ch12-13 will pull them
apart again when we introduce sliding-window prefix-cache asymmetry.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .block_hash import (
    BlockHash,
    BlockHashWithGroupId,
    chain_block_hashes,
    make_block_hash_with_group_id,
)
from .prefix_cache_index import BlockHashToBlockMap, get_cached_block


class PrefixCacheManager:
    """Per-group prefix cache with match/insert/evict.

    Single attention group only — the multi-group `kv_cache_group_ids`
    list at vLLM's interface is collapsed to a single int here.
    """

    def __init__(self, block_size: int = 16, group_id: int = 0) -> None:
        self.block_size = block_size
        self.group_id = group_id
        # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L171
        self.cache = BlockHashToBlockMap()
        # REFERENCE: instances/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py:L73
        self.req_to_blocks: defaultdict[str, list[int]] = defaultdict(list)
        # REFERENCE: instances/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py:L79
        self.num_cached_block: dict[str, int] = {}
        # Block ref counts — same role as `KVCacheBlock.ref_cnt` in vLLM.
        # REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L120
        self.ref_cnt: defaultdict[int, int] = defaultdict(int)
        # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py — null block
        self._next_block_id = 1  # block 0 is reserved (null block, Ch05)

    # ── MATCH ──────────────────────────────────────────────────────────────
    # REFERENCE: instances/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py:L446-L494
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_length: int,
    ) -> list[int]:
        """Return cached block_ids for the longest leading run of hits.

        SCAN-AND-STOP, as in vLLM:L473-L483. By the chained-hash invariant
        (`block_hash.py`), a miss at position k guarantees no hit beyond k.
        """
        max_num_blocks = max_length // self.block_size
        cached: list[int] = []
        for block_hash in block_hashes[:max_num_blocks]:
            block_id = self.cache.get_one_block(
                make_block_hash_with_group_id(block_hash, self.group_id)
            )
            if block_id is None:
                # REFERENCE: instances/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py:L474-L476
                # "if a block hash is not in cached_block_hash_to_id, the
                #  following block hashes are not computed yet for sure."
                break
            cached.append(block_id)
        return cached

    # ── INSERT ─────────────────────────────────────────────────────────────
    # REFERENCE: instances/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py:L277-L301
    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L211-L320
    def cache_blocks(
        self,
        request_id: str,
        num_tokens: int,
        block_hashes: list[BlockHash],
    ) -> int:
        """Mark FULL blocks of `request_id` as cached.

        Returns the number of newly-cached blocks. Idempotent: re-calling
        with the same `num_tokens` is a no-op (mirrors `single_type_kv_cache
        _manager.py:L289-L290` early-exit).
        """
        # REFERENCE: instances/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py:L286-L290
        # REFERENCE: instances/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py:L292-L299
        # (delegates to BlockPool.cache_full_blocks with kv_cache_group_id)
        num_cached_blocks = self.num_cached_block.get(request_id, 0)
        num_full_blocks = num_tokens // self.block_size
        if num_cached_blocks >= num_full_blocks:
            return 0

        blocks = self.req_to_blocks[request_id]
        newly = 0
        # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L237-L274
        for i in range(num_cached_blocks, num_full_blocks):
            if i >= len(blocks) or i >= len(block_hashes):
                break
            packed = make_block_hash_with_group_id(block_hashes[i], self.group_id)
            self.cache.insert(packed, blocks[i])
            newly += 1
        # REFERENCE: instances/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py:L301
        self.num_cached_block[request_id] = num_full_blocks
        return newly

    # ── EVICT ──────────────────────────────────────────────────────────────
    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L354-L389
    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L322-L352
    # (eviction is invoked LAZILY inside get_new_blocks's loop, not eagerly)
    def evict_block(self, block_id: int, block_hash: BlockHash) -> bool:
        """Remove one block from the cache.

        Mirrors `BlockPool._maybe_evict_cached_block`:
        - If the block isn't in the cache → return False.
        - If it IS → reset its hash and pop it from the index.

        Note: in vLLM, eviction is invoked LAZILY — when `get_new_blocks` pops
        a freed-but-cached block off the LRU head, the call site invokes
        `_maybe_evict_cached_block` to clean up the index (`block_pool.py:L339-L345`).
        We expose it as a direct call for tests + the demo.
        """
        packed = make_block_hash_with_group_id(block_hash, self.group_id)
        return self.cache.pop(packed, block_id) is not None

    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L424-L441
    def evict_blocks(self, block_id_to_hash: Iterable[tuple[int, BlockHash]]) -> int:
        """Bulk eviction. Returns count of successful evictions."""
        return sum(1 for bid, h in block_id_to_hash if self.evict_block(bid, h))

    # ── HOUSEKEEPING ───────────────────────────────────────────────────────
    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L391-L406 (touch)
    # REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L284-L302
    # (FreeKVCacheBlockQueue.remove — the O(1) middle-removal touch() relies on)
    # REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L162-L183
    # (FreeKVCacheBlockQueue's docstring on why doubly-linked list, not deque)
    def touch(self, block_ids: Iterable[int]) -> None:
        """Bump ref_cnt; in vLLM this also removes from the LRU free queue."""
        for bid in block_ids:
            self.ref_cnt[bid] += 1

    # REFERENCE: instances/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py:L303-L318
    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L408-L422
    # (free_blocks reverses the order so most-recent blocks return to LRU first)
    def free_request(self, request_id: str) -> list[int]:
        """Drop a request's allocation. Cached blocks survive eviction."""
        blocks = self.req_to_blocks.pop(request_id, [])
        for bid in blocks:
            if self.ref_cnt[bid] > 0:
                self.ref_cnt[bid] -= 1
        self.num_cached_block.pop(request_id, None)
        return blocks

    def register_request(
        self, request_id: str, allocated_block_ids: list[int]
    ) -> None:
        """Record which physical blocks belong to `request_id`."""
        self.req_to_blocks[request_id].extend(allocated_block_ids)
        for bid in allocated_block_ids:
            self.ref_cnt[bid] += 1

    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L322-L352
    # (BlockPool.get_new_blocks — pops LRU blocks, evicts cached, ref_cnt=1)
    def fresh_block_ids(self, n: int) -> list[int]:
        """Pretend block-pool allocator. Just hands out monotonically
        increasing ids. The real BlockPool's get_new_blocks (Ch05) is the
        production path.
        """
        ids = list(range(self._next_block_id, self._next_block_id + n))
        self._next_block_id += n
        return ids


# ── PREFIX-AWARE ALLOCATION ────────────────────────────────────────────────
# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_manager.py:L183-L223
def prefix_aware_allocate(
    mgr: PrefixCacheManager,
    request_id: str,
    token_ids: list[int],
    extra_keys: tuple = (),
) -> tuple[list[int], int]:
    """Allocate blocks for a request, REUSING any prefix-cached blocks.

    Returns (block_ids, num_reused_blocks). The first `num_reused_blocks`
    entries of `block_ids` are existing cached blocks (touch'd); the rest
    are fresh.

    REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_manager.py:L183-L223
    (KVCacheManager.get_computed_blocks calls this same logic:
     find_longest_cache_hit → touch → return)
    REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_manager.py:L208-L213
    (max_cache_hit_length = num_tokens - 1, ensuring at least one token to
     recompute for the logits position)
    """
    chain = chain_block_hashes(token_ids, mgr.block_size, extra_keys)
    cached = mgr.find_longest_cache_hit(chain, max_length=len(token_ids))
    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L391-L406
    mgr.touch(cached)

    num_full_blocks = len(token_ids) // mgr.block_size
    num_fresh = max(0, num_full_blocks - len(cached))
    fresh = mgr.fresh_block_ids(num_fresh)

    blocks = cached + fresh
    mgr.register_request(request_id, blocks)
    # cache_blocks for the freshly-allocated tail so future hits can reuse.
    mgr.cache_blocks(request_id, len(token_ids), chain)
    return blocks, len(cached)
