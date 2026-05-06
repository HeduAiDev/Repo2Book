"""
Global Prefix Cache Pool — Our Reimplementation.

REFERENCE sources:
    BlockPool:              vllm/v1/core/block_pool.py:L130-L495
    BlockHashToBlockMap:    vllm/v1/core/block_pool.py:L34-L127
    hash_block_tokens:      vllm/v1/core/kv_cache_utils.py:L539-L566
    get_request_block_hasher:vllm/v1/core/kv_cache_utils.py:L636-L686
    FreeKVCacheBlockQueue:  vllm/v1/core/kv_cache_utils.py:L162-L370
    KVCacheMetricsCollector:vllm/v1/core/kv_cache_metrics.py:L46-L96
    PrefixCacheStats:       vllm/v1/metrics/stats.py:L115-L142

Key insight from the source:
    BlockPool IS the global pool. It owns ALL GPU KV cache blocks for the
    entire engine session. The cached_block_hash_to_block is a SINGLE shared
    hash map — ANY request can find blocks cached by ANY PRIOR request.

    No locking needed — single-threaded scheduler processes all requests
    sequentially within each step. ref_cnt on each KVCacheBlock tracks
    how many active requests reference a shared block.
"""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set


# ═══════════════════════════════════════════════════════════════════════════
# Global Prefix Cache Pool
# REFERENCE: vllm/v1/core/block_pool.py:L130
#            vllm/v1/core/block_pool.py:L34 — BlockHashToBlockMap
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PoolBlock:
    """A block in the global pool. REFERENCE: KVCacheBlock (kv_cache_utils.py:L114)"""
    block_id: int
    ref_cnt: int = 0
    _hash: Optional[bytes] = None

    @property
    def block_hash(self) -> Optional[bytes]:
        return self._hash

    def set_hash(self, h: bytes):
        self._hash = h

    def reset_hash(self):
        self._hash = None


class GlobalPrefixCachePool:
    """
    Global pool shared across ALL requests.

    REFERENCE: vllm/v1/core/block_pool.py:L130 — BlockPool

    This is NOT a per-request cache. It's a single shared hash map
    that survives across all requests for the entire engine session.

    Key design choices from vLLM:
    1. No deduplication — same content can exist in multiple blocks
    2. LRU eviction via FreeKVCacheBlockQueue (doubly-linked list)
    3. ref_cnt sharing — multiple requests can reference the same block
    4. Lazy eviction — blocks keep their hash until physically re-allocated
    5. Hash chain validation — parent-child hash chain ensures correctness
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.blocks: List[PoolBlock] = [
            PoolBlock(block_id=i) for i in range(num_blocks)
        ]

        # Global hash index (REFERENCE: block_pool.py:L171)
        self.hash_index: Dict[bytes, int] = {}  # hash → block_id

        # LRU free list (simplified from doubly-linked list)
        self.free_ids: List[int] = list(range(num_blocks))

        # Per-request tracking (REFERENCE: single_type_kv_cache_manager.py:L73)
        self.req_blocks: Dict[str, List[int]] = {}  # request_id → [block_ids]

        # Metrics (REFERENCE: PrefixCacheStats — stats.py:L115)
        self.stats = {"queries": 0, "hits": 0, "evictions": 0}

    # ── Hash Chain ──
    # REFERENCE: kv_cache_utils.py:L539 — hash_block_tokens()

    @staticmethod
    def compute_chain(tokens: List[int], block_size: int = 16) -> List[bytes]:
        """Compute chained block hashes. Each hash depends on parent."""
        hashes = []
        prev = None
        for i in range(0, len(tokens), block_size):
            blk = tokens[i:i + block_size]
            if len(blk) < block_size:
                break  # Partial block — don't hash
            h = hashlib.sha256()
            h.update(prev if prev else b'\x00' * 32)
            h.update(bytes(blk))
            hashes.append(h.digest())
            prev = h.digest()
        return hashes

    # ── Allocation & Eviction ──
    # REFERENCE: block_pool.py:L322 — get_new_blocks()
    #            block_pool.py:L354 — _maybe_evict_cached_block()

    def alloc_blocks(self, request_id: str, n: int) -> List[int]:
        """Allocate n blocks, evicting LRU cached blocks if needed."""
        if n > len(self.free_ids):
            raise RuntimeError(f"OOM: need {n}, have {len(self.free_ids)} free")

        allocated = []
        for _ in range(n):
            blk_id = self.free_ids.pop(0)  # LRU: pop from front
            blk = self.blocks[blk_id]
            if blk.block_hash is not None:
                # Evict from hash index (REFERENCE: _maybe_evict_cached_block L354)
                del self.hash_index[blk.block_hash]
                blk.reset_hash()
                self.stats["evictions"] += 1
            blk.ref_cnt = 1
            allocated.append(blk_id)

        self.req_blocks.setdefault(request_id, []).extend(allocated)
        return allocated

    # ── Cache & Lookup ──
    # REFERENCE: block_pool.py:L211 — cache_full_blocks()
    #            block_pool.py:L184 — get_cached_block()

    def cache_blocks(self, request_id: str, hashes: List[bytes]):
        """Register blocks in the global hash index (prefix caching)."""
        blk_ids = self.req_blocks.get(request_id, [])
        for i, h in enumerate(hashes):
            if i < len(blk_ids):
                blk = self.blocks[blk_ids[i]]
                blk.set_hash(h)
                self.hash_index[h] = blk_ids[i]

    def lookup(self, block_hash: bytes) -> Optional[int]:
        """O(1) global hash table lookup."""
        self.stats["queries"] += 1
        blk_id = self.hash_index.get(block_hash)
        if blk_id is not None:
            self.stats["hits"] += 1
        return blk_id

    def find_longest_prefix(self, block_hashes: List[bytes]) -> List[int]:
        """
        Find longest contiguous prefix of cached blocks.

        REFERENCE: FullAttentionManager.find_longest_cache_hit()
                   (single_type_kv_cache_manager.py:L448-L494)

        Left-to-right scan. Stops at first miss.
        Because of chained hashing, miss at K guarantees miss at K+1.
        """
        hits = []
        for h in block_hashes:
            blk_id = self.lookup(h)
            if blk_id is None:
                break  # Chain broken — stop
            hits.append(blk_id)
        return hits

    # ── Sharing: ref_cnt ──
    # REFERENCE: block_pool.py:L391 — touch()

    def touch(self, block_ids: List[int]):
        """Another request is referencing these blocks — prevent eviction."""
        for bid in block_ids:
            self.blocks[bid].ref_cnt += 1

    def free_request(self, request_id: str):
        """Release all blocks held by a request. ref_cnt--; if 0 → free."""
        blk_ids = self.req_blocks.pop(request_id, [])
        for bid in reversed(blk_ids):  # Reverse = LRU tail-first (source: L303)
            blk = self.blocks[bid]
            blk.ref_cnt -= 1
            if blk.ref_cnt == 0:
                self.free_ids.append(bid)  # Append to tail = MRU

    # ── Metrics ──
    @property
    def hit_rate(self) -> float:
        return self.stats["hits"] / max(1, self.stats["queries"])

    @property
    def usage(self) -> float:
        return 1.0 - len(self.free_ids) / self.num_blocks


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Request Sharing Demonstration
# ═══════════════════════════════════════════════════════════════════════════

def demonstrate_sharing():
    """Two requests with same system prompt → sharing via global pool."""
    pool = GlobalPrefixCachePool(num_blocks=100)
    block_size = 4

    # System prompt (shared): 8 tokens → 2 blocks
    system = [1, 2, 3, 4, 5, 6, 7, 8]
    sys_hashes = pool.compute_chain(system, block_size)

    # Request A: system + "what is AI?" (4 tokens)
    req_a = system + [10, 20, 30, 40]
    hashes_a = pool.compute_chain(req_a, block_size)

    # A allocates and caches
    blks_a = pool.alloc_blocks("A", len(hashes_a))
    pool.cache_blocks("A", hashes_a)
    print(f"Request A: {len(blks_a)} blocks allocated, pool hash index size={len(pool.hash_index)}")

    # Request B: same system + "what is ML?"
    req_b = system + [10, 20, 30, 50]  # Different 4th token
    hashes_b = pool.compute_chain(req_b, block_size)

    # B finds prefix hits (first 2 blocks = system prompt)
    hits = pool.find_longest_prefix(hashes_b)
    shared_tokens = len(hits) * block_size
    print(f"Request B: {len(hits)} prefix blocks SHARED from A ({shared_tokens} tokens)")
    pool.touch(hits)  # Protect shared blocks

    # B allocates only NEW blocks (block 3)
    new_needed = len(hashes_b) - len(hits)
    blks_b_new = pool.alloc_blocks("B", new_needed)
    print(f"Request B: {new_needed} new blocks allocated (saved {len(hits)} allocations)")
    print(f"Pool hit rate: {pool.hit_rate:.1%}")

    # Clean up
    pool.free_request("A")
    pool.free_request("B")
    print(f"After freeing: usage={pool.usage:.1%}, evictions={pool.stats['evictions']}")

    return hits


if __name__ == "__main__":
    demonstrate_sharing()
