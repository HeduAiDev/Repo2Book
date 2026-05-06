# REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py
"""BlockPool — the GPU-side block allocator with prefix-cache eviction.

Three pieces of state:

    1. `blocks: list[KVCacheBlock]` — flat array of all KVCacheBlock metadata.
       Index = block_id. Pre-allocated at construction (no GC churn).
    2. `free_block_queue: FreeKVCacheBlockQueue` — LRU doubly-linked list of
       blocks with ref_cnt==0. The "watermark" against full KV cache.
    3. `cached_block_hash_to_block: dict[hash -> block]` — prefix cache index.
       When `enable_caching=True`, freed blocks STAY in this dict so they can
       be re-touched on a future cache hit. Eviction happens on
       `get_new_blocks` if the LRU block is still cached.

A block is in *one of three states*:
    a. ref_cnt > 0                    — in use by ≥1 running request
    b. ref_cnt = 0 AND block_hash set — cached (in queue, evictable)
    c. ref_cnt = 0 AND block_hash None — uncached (in queue, fresh)

`get_num_free_blocks()` returns the queue length, which is the watermark the
scheduler reads via `kv_cache_manager.usage`.
"""

from __future__ import annotations

from typing import Iterable

from .kv_cache_block import FreeKVCacheBlockQueue, KVCacheBlock


# REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L130-L182
class BlockPool:
    """Pool of KVCacheBlock metadata + free queue + prefix-cache hash map.

    Constructor and method names match vLLM exactly so the scheduler call
    sites (`block_pool.get_new_blocks`, `.free_blocks`, `.get_num_free_blocks`,
    `.get_usage`, `.touch`, `.evict_blocks`) read identically.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool = True,
    ) -> None:
        # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L156-L177
        assert num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching

        # The flat list of block metadata, addressable by block_id.
        # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L162-L164
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(block_id=i) for i in range(num_gpu_blocks)
        ]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Prefix cache index. Hash key -> block.
        # SIMPLIFIED: vLLM uses a `BlockHashToBlockMap` that handles the (rare)
        # case of multiple blocks sharing one hash key. We use a plain dict.
        # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L34-L127
        self.cached_block_hash_to_block: dict[bytes, KVCacheBlock] = {}

        # The null block: a placeholder for sliding-window padding. Block 0
        # is reserved and pinned (popped out of the free queue at startup).
        # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L173-L177
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L322-L352
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Pop `num_blocks` LRU blocks. Caller must check capacity first.

        On the way out we evict each block from the prefix cache (if it was
        cached) and bump ref_cnt to 1.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret = self.free_block_queue.popleft_n(num_blocks)

        for block in ret:
            if self.enable_caching:
                self._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt += 1
        return ret

    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L354-L389
    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """Drop `block` from the prefix-cache hash map; reset its hash.

        Returns True if eviction happened.
        """
        bh = block.block_hash
        if bh is None:
            return False
        if self.cached_block_hash_to_block.pop(bh, None) is None:
            return False
        block.reset_hash()
        return True

    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L391-L406
    def touch(self, blocks: Iterable[KVCacheBlock]) -> None:
        """Bump ref_cnt on cached blocks; remove from free queue if it was there.

        Called when a new request prefix-cache-hits an already-allocated block.
        The O(1) `free_block_queue.remove(block)` matters here.
        """
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L408-L422
    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Decrement ref_cnt; if it hits 0, push back to the free queue.

        `ordered_blocks` is the request's blocks in REVERSE allocation order
        (kv_cache_manager.free reverses; `kv_cache_manager.py:L420-L427`),
        so the *most recently allocated* block goes back to the LRU position
        first. This makes the LRU queue order roughly arrival-time-based,
        which improves prefix-cache hit rate.

        Cached blocks (block_hash != None) are kept in `cached_block_hash_to_block`
        so a future hit can re-touch them.
        """
        blocks = list(ordered_blocks)
        for block in blocks:
            block.ref_cnt -= 1
        # Only blocks that hit ref_cnt 0 (and aren't the null block) go back.
        # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L420-L422
        self.free_block_queue.append_n(
            b for b in blocks if b.ref_cnt == 0 and not b.is_null
        )

    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L424-L441
    def evict_blocks(self, block_ids: set[int]) -> None:
        """Remove blocks from the prefix cache (without freeing them).

        Used by KV connectors to invalidate blocks that were transferred away.
        Blocks with ref_cnt > 0 stay allocated; only the cache hash entry goes.
        """
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}"
            )
            self._maybe_evict_cached_block(self.blocks[block_id])

    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L478-L484
    def get_num_free_blocks(self) -> int:
        return self.free_block_queue.num_free_blocks

    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L486-L497
    def get_usage(self) -> float:
        """Fraction of pool that is in-use. Subtracts the null block from the total."""
        total = self.num_gpu_blocks - 1  # null block is permanently used
        if total == 0:
            return 0.0
        return 1.0 - (self.get_num_free_blocks() / total)

    def cache_block(self, block: KVCacheBlock, block_hash: bytes) -> None:
        """Mark `block` as cached. Called by the coordinator when a block fills up.

        SIMPLIFIED: vLLM's `cache_full_blocks` (`block_pool.py:L211-L320`) handles
        a list of blocks at once, with KV-event emission for distributed prefix
        cache logging. We expose the single-block primitive for clarity.
        """
        if not self.enable_caching:
            return
        block.block_hash = block_hash
        self.cached_block_hash_to_block[block_hash] = block

    def get_cached_block(self, block_hash: bytes) -> KVCacheBlock | None:
        """Look up a cached block by hash. Returns None on cache miss."""
        return self.cached_block_hash_to_block.get(block_hash)

    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L443-L476
    def reset_prefix_cache(self) -> bool:
        """Clear all cache hashes. Used in RLHF flows after weight updates."""
        num_used = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used != 1:  # only the null block should be in use
            return False
        self.cached_block_hash_to_block.clear()
        for block in self.blocks:
            block.reset_hash()
        return True
