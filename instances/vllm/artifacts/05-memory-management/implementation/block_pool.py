# REFERENCE: vllm/v1/core/block_pool.py:L130-L509 (BlockPool)
# REFERENCE: vllm/v1/core/block_pool.py:L34-L127   (BlockHashToBlockMap)
# REFERENCE: vllm/v1/core/kv_cache_utils.py:L113-L160 (KVCacheBlock)
# REFERENCE: vllm/v1/core/kv_cache_utils.py:L162-L370 (FreeKVCacheBlockQueue)
# REFERENCE: vllm/v1/core/kv_cache_utils.py:L37-L78  (BlockHash, BlockHashWithGroupId)
# REFERENCE: vllm/v1/core/kv_cache_utils.py:L539-L566 (hash_block_tokens)
"""
BlockPool: the shared KV cache block pool with prefix caching.

This module implements the core data structures:
- KVCacheBlock: metadata for a single KV cache block
- FreeKVCacheBlockQueue: O(1) doubly-linked list of free blocks
- BlockHashToBlockMap: hash → block mapping for prefix cache lookups
- BlockPool: manages allocation, freeing, and prefix caching

The design follows the "virtual memory" analogy:
- Each block = a "page" of KV cache (default 16 tokens)
- block_table = "page table" per request (mapping token positions to block IDs)
- BlockPool = "physical memory manager" with free list + eviction
"""

import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union


# ─── Block Hash Types ───────────────────────────────────────────────────────
# SIMPLIFIED: original uses BlockHash = NewType("BlockHash", bytes) and
# BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes) with binary
# packing for efficiency. We use tuple keys for clarity.
# BlockHash = tuple[bytes, int]  <- (hash_bytes, group_id)

BLOCK_SIZE = 16    # Default: 16 tokens per block

# Seed hash for the first block in a sequence.
# In vLLM, NONE_HASH is initialized from PYTHONHASHSEED or os.urandom(32).
NONE_HASH = os.urandom(32)


def hash_block_tokens(
    parent_hash: bytes | None,
    token_ids: Sequence[int],
) -> bytes:
    """Compute incremental hash of a KV cache block.

    Each block's hash chains from the parent (previous block), forming a
    cryptographic chain: hash_i = SHA256(hash_{i-1} || token_ids_i)

    This allows:
    - Prefix of length N → one hash chain of N blocks
    - Adding new block: hash(hash_of_last, new_token_ids) — no recomputation

    REFERENCE: vllm/v1/core/kv_cache_utils.py:L539-L566
    """
    if parent_hash is None:
        parent_hash = NONE_HASH
    token_tuple = tuple(token_ids)
    data = parent_hash + str(token_tuple).encode()
    return hashlib.sha256(data).digest()


# ─── KVCacheBlock ───────────────────────────────────────────────────────────

@dataclass
class KVCacheBlock:
    """Metadata for a single KV cache block.

    Each block stores K and V tensors for block_size tokens. This is the
    metadata side (the actual tensor data lives on GPU).

    REFERENCE: vllm/v1/core/kv_cache_utils.py:L113-L160
    """
    block_id: int                    # 0 to num_gpu_blocks - 1
    ref_cnt: int = 0                 # Number of requests using this block
    _block_hash: bytes | None = None # Hash set when block is full and cached
    is_null: bool = False            # Placeholder for skipped positions

    # Doubly-linked list pointers for FreeKVCacheBlockQueue.
    # SIMPLIFIED: original uses KVCacheBlock | None type hints, same semantics.
    prev_free_block: 'KVCacheBlock | None' = None
    next_free_block: 'KVCacheBlock | None' = None

    @property
    def block_hash(self) -> bytes | None:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, value: bytes):
        assert self._block_hash is None, (
            f"Block {self.block_id} already has a hash. Double-cache detected."
        )
        self._block_hash = value

    def reset_hash(self):
        """Clear block hash when evicted from prefix cache."""
        self._block_hash = None

    def __repr__(self) -> str:
        prev_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_id = self.next_free_block.block_id if self.next_free_block else None
        return (
            f"KVCacheBlock(block_id={self.block_id}, ref_cnt={self.ref_cnt}, "
            f"hash={self._block_hash[:4].hex() if self._block_hash else None}..., "
            f"prev_free={prev_id}, next_free={next_id})"
        )


# ─── FreeKVCacheBlockQueue ─────────────────────────────────────────────────

class FreeKVCacheBlockQueue:
    """Doubly-linked list of free blocks, ordered by eviction priority.

    WHY NOT collections.deque?
    deque.remove(block) is O(n), but prefix cache touch() needs O(1) removal
    from the middle of the free list. By storing prev/next pointers on each
    KVCacheBlock directly, we get O(1) pop, append, AND middle-remove.

    Eviction order: LRU (least recently freed/used blocks at the front).
    Fake head and tail sentinel nodes reduce branching.

    REFERENCE: vllm/v1/core/kv_cache_utils.py:L162-L370
    """

    def __init__(self, blocks: list[KVCacheBlock]):
        self.num_free_blocks = len(blocks)

        # Initialize doubly-linked list: connect consecutive blocks
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # Fake sentinel nodes: head ↔ ...blocks... ↔ tail
        self.fake_head = KVCacheBlock(block_id=-1)
        self.fake_tail = KVCacheBlock(block_id=-1)

        if self.num_free_blocks > 0:
            self.fake_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_head
            self.fake_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_tail
        else:
            self.fake_head.next_free_block = self.fake_tail
            self.fake_tail.prev_free_block = self.fake_head

    def popleft(self) -> KVCacheBlock:
        """Pop the first free block (O(1))."""
        first = self.fake_head.next_free_block
        if first is self.fake_tail or first is None:
            raise ValueError("No free blocks available")

        # Re-link: head → first.next
        self.fake_head.next_free_block = first.next_free_block
        first.next_free_block.prev_free_block = self.fake_head

        # Detach block from list
        first.prev_free_block = None
        first.next_free_block = None

        self.num_free_blocks -= 1
        return first

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """Pop the first n free blocks (O(n))."""
        if n == 0:
            return []
        assert self.num_free_blocks >= n, (
            f"Requested {n} blocks but only {self.num_free_blocks} available"
        )
        self.num_free_blocks -= n

        curr = self.fake_head.next_free_block
        result = []
        for _ in range(n):
            assert curr is not None
            result.append(curr)
            last = curr
            curr = curr.next_free_block
            last.prev_free_block = None
            last.next_free_block = None

        if curr is not None:
            # Connect head to the new first real block
            self.fake_head.next_free_block = curr
            curr.prev_free_block = self.fake_head
        else:
            # Queue is now empty
            self.fake_head.next_free_block = self.fake_tail
            self.fake_tail.prev_free_block = self.fake_head
        return result

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block from the middle of the free list (O(1)).

        Called by touch() when a cached block is re-used: the block is no longer
        a free eviction candidate.
        """
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(
                f"remove() called on block {block.block_id} not in free list"
            )
        # Bridge around the block
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block

        # Detach
        block.prev_free_block = None
        block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """Append a block to the tail (O(1))."""
        last = self.fake_tail.prev_free_block
        assert last is not None

        # Insert between last and fake_tail
        last.next_free_block = block
        block.prev_free_block = last
        block.next_free_block = self.fake_tail
        self.fake_tail.prev_free_block = block

        self.num_free_blocks += 1

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """Append multiple blocks in order (O(len(blocks)))."""
        if len(blocks) == 0:
            return

        last = self.fake_tail.prev_free_block
        assert last is not None

        for block in blocks:
            block.prev_free_block = last
            last.next_free_block = block
            last = block

        last.next_free_block = self.fake_tail
        self.fake_tail.prev_free_block = last

        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Iterate the free list. Used for debugging/testing."""
        result = []
        curr = self.fake_head.next_free_block
        while curr is not None and curr is not self.fake_tail:
            result.append(curr)
            curr = curr.next_free_block
        return result


# ─── BlockHashToBlockMap ────────────────────────────────────────────────────

# Type alias: a cached block is either a single block or a dict of
# duplicate-hash blocks (happens when two requests process the same prefix
# simultaneously).
CachedBlocks = Union['KVCacheBlock', dict[int, 'KVCacheBlock']]


class BlockHashToBlockMap:
    """Hash → block mapping for prefix cache lookups.

    Union-type design:
    - Common case (single block with this hash): store KVCacheBlock directly
    - Rare case (duplicate hashes during concurrent prefill): store dict

    This avoids dict overhead for the 99% case.

    REFERENCE: vllm/v1/core/block_pool.py:L34-L127
    """

    def __init__(self):
        self._cache: dict[bytes, CachedBlocks] = {}

    def get_one_block(self, block_hash: bytes) -> KVCacheBlock | None:
        """Get any block with the given hash."""
        blocks = self._cache.get(block_hash)
        if blocks is None:
            return None
        if isinstance(blocks, KVCacheBlock):
            return blocks
        if isinstance(blocks, dict):
            return next(iter(blocks.values()))
        raise AssertionError(f"Unexpected type: {type(blocks)}")

    def insert(self, block_hash: bytes, block: KVCacheBlock) -> None:
        """Insert a block into the hash index."""
        existing = self._cache.get(block_hash)
        if existing is None:
            self._cache[block_hash] = block
        elif isinstance(existing, KVCacheBlock):
            # Upgrade: single block → dict of blocks (duplicate hash)
            self._cache[block_hash] = {
                existing.block_id: existing,
                block.block_id: block,
            }
        elif isinstance(existing, dict):
            existing[block.block_id] = block
        else:
            raise AssertionError(f"Unexpected type: {type(existing)}")

    def pop(self, block_hash: bytes, block_id: int) -> KVCacheBlock | None:
        """Remove and return the block with given hash+id from the cache."""
        blocks = self._cache.pop(block_hash, None)
        if blocks is None:
            return None
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # Hash matches but block_id doesn't — put it back (rare race)
            self._cache[block_hash] = blocks
            return None
        if isinstance(blocks, dict):
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[block_hash] = blocks  # Put remaining back
            return block
        raise AssertionError(f"Unexpected type: {type(blocks)}")

    def __len__(self) -> int:
        return len(self._cache)


# ─── BlockPool ──────────────────────────────────────────────────────────────

class BlockPool:
    """Shared pool of KV cache blocks.

    Manages:
    - free_block_queue: free blocks in eviction order (LRU)
    - cached_block_hash_to_block: hash → block for prefix cache lookups
    - null_block: singleton placeholder for skipped/missing positions

    Allocation: pop from free queue (may evict from cache)
    Freeing: push to tail of free queue (becomes LRU eviction candidate)
    Touch: remove from free queue (when prefix cache hit revives a block)

    REFERENCE: vllm/v1/core/block_pool.py:L130-L509
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool = True,
    ):
        assert num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching

        # Create all blocks
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(block_id=idx) for idx in range(num_gpu_blocks)
        ]

        # Free block queue: all blocks initially free
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Hash → block index for prefix cache
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Null block: placeholder for skipped positions
        # Uses block_id=0 from the pool. ref_cnt is NOT maintained —
        # null_block is never freed.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

    # ── Cache Lookup ────────────────────────────────────────────────────

    def get_cached_block(self, block_hash: bytes) -> KVCacheBlock | None:
        """Look up a cached block by hash. Returns None on cache miss.

        REFERENCE: vllm/v1/core/block_pool.py:L184-L209
        """
        if not self.enable_caching:
            return None
        return self.cached_block_hash_to_block.get_one_block(block_hash)

    def get_cached_blocks(
        self, block_hashes: list[bytes]
    ) -> list[KVCacheBlock | None]:
        """Look up multiple block hashes. None for each miss."""
        return [self.get_cached_block(h) for h in block_hashes]

    # ── Cache Blocks ────────────────────────────────────────────────────

    def cache_full_blocks(
        self,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        request_block_hashes: list[bytes],
    ) -> None:
        """Cache full blocks into the hash→block index for prefix caching.

        Called after new tokens are computed and blocks become full.
        Only caches blocks that are not already cached (from num_cached_blocks
        to num_full_blocks).

        REFERENCE: vllm/v1/core/block_pool.py:L211-L320
        SIMPLIFIED: no LoRA/MM extra keys, no KV cache events, no
        block_size != hash_block_size case.

        Args:
            blocks: All blocks for this request (may include null blocks).
            num_cached_blocks: Blocks already cached (prefix cache hits).
            num_full_blocks: Total blocks that are now full.
            request_block_hashes: Pre-computed hashes for each block position.
        """
        if not self.enable_caching or num_cached_blocks >= num_full_blocks:
            return

        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        new_hashes = request_block_hashes[num_cached_blocks:num_full_blocks]

        for block, block_hash in zip(new_full_blocks, new_hashes):
            if block.is_null:
                continue  # Placeholder blocks are never cached
            assert block.block_hash is None, (
                f"Block {block.block_id} already has a hash"
            )
            block.block_hash = block_hash
            self.cached_block_hash_to_block.insert(block_hash, block)

    # ── Allocation ──────────────────────────────────────────────────────

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Allocate new blocks from the free pool.

        For each allocated block:
        - If it was cached (had a hash), evict it from the cache
        - Set ref_cnt to 1

        REFERENCE: vllm/v1/core/block_pool.py:L322-L352

        Args:
            num_blocks: Number of blocks to allocate.

        Returns:
            List of allocated KVCacheBlock objects.

        Raises:
            ValueError: If insufficient free blocks.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(
                f"Cannot allocate {num_blocks} blocks; "
                f"only {self.get_num_free_blocks()} free"
            )

        result = self.free_block_queue.popleft_n(num_blocks)

        for block in result:
            if self.enable_caching:
                self._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt += 1

        return result

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """If block is cached, remove it from the hash index and reset hash.

        REFERENCE: vllm/v1/core/block_pool.py:L354-L389
        """
        block_hash = block.block_hash
        if block_hash is None:
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            return False

        block.reset_hash()
        return True

    # ── Touch ───────────────────────────────────────────────────────────

    def touch(self, blocks: list[KVCacheBlock]) -> None:
        """Increase ref_cnt of cached blocks and remove from free list.

        When a new request hits a cached block (prefix cache), the block
        must be: (a) ref_cnt increased so it's not evicted while in use,
        (b) removed from free list if it was an eviction candidate.

        REFERENCE: vllm/v1/core/block_pool.py:L391-L406

        Args:
            blocks: Blocks to touch (already identified as cache hits).
        """
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                # Block was an eviction candidate — remove from free list
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    # ── Free ────────────────────────────────────────────────────────────

    def free_blocks(self, ordered_blocks: Sequence[KVCacheBlock]) -> None:
        """Free blocks: decrement ref_cnt, return to free list if ref_cnt=0.

        Blocks with ref_cnt=0 are appended to the free queue (becoming
        eviction candidates). Blocks ordered by the caller: first element
        is evicted first (LRU order).

        Freeing in reverse-block-order means tail blocks (decoding-only
        tokens) are evicted first, preserving prefix blocks in cache.

        REFERENCE: vllm/v1/core/block_pool.py:L408-L422
        """
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1

        # Only return blocks with ref_cnt == 0 to free list
        return_to_free = [
            b for b in blocks_list
            if b.ref_cnt == 0 and not b.is_null
        ]
        if return_to_free:
            self.free_block_queue.append_n(return_to_free)

    # ── Query ───────────────────────────────────────────────────────────

    def get_num_free_blocks(self) -> int:
        """Number of free blocks available for allocation."""
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """KV cache usage ratio (0.0 to 1.0).

        REFERENCE: vllm/v1/core/block_pool.py:L486-L497
        """
        total = self.num_gpu_blocks - 1  # Subtract null block
        if total == 0:
            return 0.0
        return 1.0 - (self.get_num_free_blocks() / total)


# ─── Demonstration ──────────────────────────────────────────────────────────

def main():
    """Demonstrate BlockPool operations: allocation, caching, prefix hits."""
    print("=" * 70)
    print("BlockPool Demo: Allocation, Prefix Caching, and Freeing")
    print("=" * 70)

    # Create a pool with 10 blocks (9 usable + 1 null)
    pool = BlockPool(num_gpu_blocks=10, enable_caching=True)
    print(f"\nInitial: {pool.get_num_free_blocks()} free blocks "
          f"(1 null block reserved)")
    print(f"Usage: {pool.get_usage():.1%}")

    # ── Simulate Request 1: prompt "Hello world, this is a test" ──
    prompt_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                     17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    # 32 tokens → 2 blocks of 16 tokens each
    block_0_tokens = prompt_tokens[:16]
    block_1_tokens = prompt_tokens[16:]

    # Compute block hashes (incremental: hash(parent, tokens))
    hash_0 = hash_block_tokens(None, block_0_tokens)       # First block
    hash_1 = hash_block_tokens(hash_0, block_1_tokens)      # Second block
    block_hashes = [hash_0, hash_1]
    print(f"\nHash block 0: {hash_0[:4].hex()}... (tokens {block_0_tokens[:3]}...)")
    print(f"Hash block 1: {hash_1[:4].hex()}... (tokens {block_1_tokens[:3]}...)")

    # Allocate 2 blocks for Request 1
    req1_blocks = pool.get_new_blocks(2)
    print(f"\nRequest 1: allocated blocks {[b.block_id for b in req1_blocks]}")
    print(f"Free blocks left: {pool.get_num_free_blocks()}, Usage: {pool.get_usage():.1%}")

    # Cache full blocks (mark them as cached for prefix reuse)
    pool.cache_full_blocks(
        blocks=req1_blocks,
        num_cached_blocks=0,
        num_full_blocks=2,
        request_block_hashes=block_hashes,
    )
    print(f"Cached: {len(pool.cached_block_hash_to_block)} blocks in hash index")

    # ── Simulate Request 2: same prefix "Hello world, this is a test" + more ──
    # Check prefix cache: look up first block hash
    cached_block_0 = pool.get_cached_block(hash_0)
    print(f"\nRequest 2: prefix cache lookup for hash_0 → "
          f"{'HIT' if cached_block_0 else 'MISS'} "
          f"(block {cached_block_0.block_id if cached_block_0 else 'N/A'})")

    # Touch the cached block (increase ref_cnt, remove from free list)
    if cached_block_0:
        pool.touch([cached_block_0])
        print(f"  Touched block {cached_block_0.block_id}: "
              f"ref_cnt={cached_block_0.block_id}")
        # Since ref_cnt was 0 (in free list as eviction candidate), touch()
        # removed it. Now it's in use by both requests.

    # Allocate 1 new block for the second block (it's also cached)
    cached_block_1 = pool.get_cached_block(hash_1)
    if cached_block_1:
        pool.touch([cached_block_1])
        print(f"  Second block also HIT: block {cached_block_1.block_id}")

    # Allocate 1 more block for new tokens
    new_block = pool.get_new_blocks(1)
    print(f"  New block for extra tokens: {[b.block_id for b in new_block]}")

    print(f"Pool usage: {pool.get_usage():.1%}")

    # ── Free Request 1 (keep Request 2 alive) ──
    print(f"\nFreeing Request 1 (reverse order for LRU eviction):")
    pool.free_blocks(reversed(req1_blocks))
    print(f"  Free blocks now: {pool.get_num_free_blocks()}")

    # ── Free Request 2 ──
    all_req2 = [cached_block_0, cached_block_1] + new_block  # type: ignore
    print(f"\nFreeing Request 2:")
    pool.free_blocks(reversed(all_req2))
    print(f"  Free blocks now: {pool.get_num_free_blocks()}")
    print(f"  Usage: {pool.get_usage():.1%}")

    # ── Show eviction: allocate all blocks, cache gets evicted ──
    print(f"\n--- Eviction Stress Test ---")
    new_blocks = pool.get_new_blocks(9)  # All free blocks
    print(f"  Allocated all {len(new_blocks)} free blocks")
    print(f"  Free: {pool.get_num_free_blocks()}, Usage: {pool.get_usage():.1%}")
    # The previously cached blocks should be evicted (hash reset)
    print(f"  Block hash after eviction: {cached_block_0.block_hash}")

    print("\n" + "=" * 70)
    print("Key takeaway: BlockPool implements virtual-memory-style")
    print("block management with prefix caching. Same blocks serve")
    print("multiple requests when they share common prefixes.")
    print("=" * 70)


if __name__ == "__main__":
    main()
