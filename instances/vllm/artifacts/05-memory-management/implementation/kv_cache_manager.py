# REFERENCE: vllm/v1/core/kv_cache_manager.py:L106-L542 (KVCacheManager)
# REFERENCE: vllm/v1/core/kv_cache_manager.py:L21-L104  (KVCacheBlocks)
# REFERENCE: vllm/v1/core/kv_cache_coordinator.py:L324-L389 (UnitaryKVCacheCoordinator)
# REFERENCE: vllm/v1/core/single_type_kv_cache_manager.py:L446-L504 (FullAttentionManager)
# REFERENCE: vllm/v1/core/single_type_kv_cache_manager.py:L30-L444 (SingleTypeKVCacheManager)
"""
KVCacheManager: the scheduler-facing interface for KV cache block management.

This module unifies the three-layer V1 architecture into a single manager for
educational clarity:
  1. KVCacheManager (external API)
  2. Coordinator logic (inlined, unitary single-type)
  3. BlockPool (physical block management)

The 3-stage allocate_slots() flow:
  1. Free skipped blocks (outside attention window)
  2. Handle prefix tokens: touch cached blocks, allocate for new tokens
  3. Allocate new blocks for tokens to be computed

Key simplifications from the full vLLM implementation:
- Single KV cache type (full attention) — no HybridCoordinator needed
- No EAGLE speculative decoding support
- No DCP/PCP (context parallelism)
- No external computed tokens (KV connectors)
- No encoder-decoder (cross-attention)
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from block_pool import KVCacheBlock, BlockPool


# ─── KVCacheBlocks ──────────────────────────────────────────────────────────

@dataclass
class KVCacheBlocks:
    """Allocation result from KVCacheManager.

    Wraps allocated blocks as a tuple of sequences:
    `blocks[i][j]` = j-th block of i-th KV cache group.
    For single-type (our case), `blocks` is a tuple with one element.

    This hides BlockPool internals from the Scheduler.

    REFERENCE: vllm/v1/core/kv_cache_manager.py:L21-L104

    SIMPLIFIED: no `__add__`, `get_unhashed_block_ids()`, overloaded methods.
    """
    blocks: tuple[list['KVCacheBlock'], ...]

    def get_block_ids(self) -> list[int]:
        """Convert blocks to block_id lists for the model runner.

        Returns one list per KV cache group. For single-type, returns [list].
        """
        return [block.block_id for group in self.blocks for block in group]

    def new_empty(self) -> 'KVCacheBlocks':
        """Create a new KVCacheBlocks with the same group structure but empty."""
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


# ─── Simple Request (for demo) ─────────────────────────────────────────────

@dataclass
class Request:
    """Minimal request representation for KV cache management.

    REFERENCE: vllm/v1/request.py:L59-L308
    SIMPLIFIED: only fields needed by KVCacheManager. No multimodal, LoRA,
    structured output, streaming, spec decode, etc.
    """
    request_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    num_tokens: int = 0          # Total token count (prompt + generated)
    num_computed_tokens: int = 0 # Tokens already fed to the model
    block_hashes: list[bytes] = field(default_factory=list)
    num_preemptions: int = 0
    skip_reading_prefix_cache: bool = False

    def __post_init__(self):
        self.num_tokens = len(self.prompt_token_ids)


# ─── KVCacheManager ──────────────────────────────────────────────────────────

class KVCacheManager:
    """Unitary KV cache manager for single-attention-type models.

    This is the top-level interface the Scheduler uses. It owns:
    - A BlockPool for physical block management
    - Per-request state: block allocations, cached block counts
    - Prefix cache lookup via find_longest_cache_hit()

    REFERENCE: vllm/v1/core/kv_cache_manager.py:L106-L542
    REFERENCE: vllm/v1/core/kv_cache_coordinator.py:L324-L389 (UnitaryCoordinator)
    REFERENCE: vllm/v1/core/single_type_kv_cache_manager.py:L446-L504 (FullAttentionManager)

    SIMPLIFIED: single KV cache type (full attention), no EAGLE, no DCP/PCP,
    no external computed tokens, no metrics/events.
    """

    def __init__(
        self,
        block_pool: 'BlockPool',
        block_size: int = 16,
        max_model_len: int = 4096,
        enable_caching: bool = True,
    ):
        self.block_pool = block_pool
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching

        # Per-request block tracking: request_id → list[KVCacheBlock]
        self.req_to_blocks: defaultdict[str, list['KVCacheBlock']] = (
            defaultdict(list)
        )

        # Per-request cached block count
        self.num_cached_block: dict[str, int] = {}

        # Pre-constructed empty KVCacheBlocks (avoid allocation overhead)
        self.empty_kv_cache_blocks = KVCacheBlocks(((),))

        # Null block reference from pool
        self._null_block = block_pool.null_block

    @property
    def usage(self) -> float:
        """KV cache usage ratio (0.0 to 1.0)."""
        return self.block_pool.get_usage()

    def get_num_free_blocks(self) -> int:
        return self.block_pool.get_num_free_blocks()

    # ── Prefix Cache Lookup ─────────────────────────────────────────────

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """Find prefix cache hits for a request.

        Scans the request's block_hashes left-to-right, stopping at the first
        miss. Returns cached blocks and the number of computed tokens
        (cached_blocks * block_size, minus 1 for the last token recompute).

        Last-token recompute: even when all tokens hit the cache, the model
        must recompute the last token to produce logits. So max_cache_hit_length
        = num_tokens - 1.

        REFERENCE: vllm/v1/core/kv_cache_manager.py:L183-L223

        Args:
            request: The request to check for prefix cache hits.

        Returns:
            (KVCacheBlocks of cached blocks, num_computed_tokens)
        """
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # Last token must be recomputed to get logits
        max_cache_hit_length = request.num_tokens - 1

        computed_blocks, hit_tokens = self.find_longest_cache_hit(
            request.block_hashes, max_cache_hit_length
        )

        return self.create_kv_cache_blocks(computed_blocks), hit_tokens

    def find_longest_cache_hit(
        self,
        block_hashes: list[bytes],
        max_cache_hit_length: int,
    ) -> tuple[list['KVCacheBlock'], int]:
        """Find the longest prefix cache hit, scanning left to right.

        For full attention, this is a simple linear scan: walk through
        block_hashes in order, stopping at the first miss.

        REFERENCE: vllm/v1/core/single_type_kv_cache_manager.py:L446-L494

        Args:
            block_hashes: Pre-computed hash chain for the request.
            max_cache_hit_length: Max tokens that can be cached (num_tokens - 1).

        Returns:
            (list of cached KVCacheBlocks, num_computed_tokens)
        """
        max_num_blocks = max_cache_hit_length // self.block_size
        computed_blocks: list['KVCacheBlock'] = []

        for i in range(max_num_blocks):
            if i >= len(block_hashes):
                break
            hit_block = self.block_pool.get_cached_block(block_hashes[i])
            if hit_block is not None:
                computed_blocks.append(hit_block)
            else:
                break  # Cache miss — stop scanning

        hit_tokens = len(computed_blocks) * self.block_size
        return computed_blocks, hit_tokens

    def create_kv_cache_blocks(
        self, blocks: list['KVCacheBlock']
    ) -> KVCacheBlocks:
        """Wrap blocks in KVCacheBlocks. Returns empty wrapper if no blocks."""
        if not blocks:
            return self.empty_kv_cache_blocks
        return KVCacheBlocks((blocks,))

    # ── allocate_slots(): 3-Stage Allocation ────────────────────────────

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
    ) -> KVCacheBlocks | None:
        """Allocate KV cache slots for new tokens.

        3-STAGE ALLOCATION:
        1. Free unnecessary blocks (outside attention window — SWA only)
        2. Handle prefix tokens: touch cached blocks, allocate for new ones
        3. Allocate new blocks for tokens to be computed

        Returns None if insufficient free blocks.

        REFERENCE: vllm/v1/core/kv_cache_manager.py:L225-L416

        Args:
            request: The request needing allocation.
            num_new_tokens: Number of new tokens to allocate.
            num_new_computed_tokens: Prefix cache hit tokens (from get_computed_blocks).
            new_computed_blocks: The cached blocks (from get_computed_blocks).

        Returns:
            KVCacheBlocks or None (if allocation fails).
        """
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be > 0")

        # Unpack new computed blocks (prefix cache hits)
        new_computed_block_list: list['KVCacheBlock']
        if new_computed_blocks is not None:
            new_computed_block_list = list(new_computed_blocks.blocks[0])
        else:
            new_computed_block_list = []

        # Total computed tokens = existing + new prefix cache hits
        total_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )

        # Total tokens after this allocation.
        # num_new_tokens covers the additional tokens to compute beyond
        # total_computed_tokens. The total after allocation is the sum of
        # already-computed + newly-computed. In the scheduler, this equals
        # request.num_tokens (or num_tokens_with_spec).
        num_tokens_after = total_computed_tokens + num_new_tokens
        # Cap at the maximum for cache_blocks (only finalized tokens)
        num_tokens_to_cache = min(num_tokens_after, request.num_tokens)
        # Cap token slots at max_model_len
        num_tokens_need_slot = min(num_tokens_after, self.max_model_len)

        # ── Stage 1: Free skipped blocks ──
        # Full attention never skips tokens, but sliding window does.
        # We call this anyway for interface consistency.
        self.remove_skipped_blocks(request.request_id, total_computed_tokens)

        # ── Stage 2-3: Count needed blocks, check if we have capacity ──
        num_required_blocks = math.ceil(num_tokens_need_slot / self.block_size)
        existing_blocks = self.req_to_blocks.get(request.request_id, [])

        if request.request_id in self.num_cached_block:
            # Running request — fast path, no new prefix cache hits
            assert len(new_computed_block_list) == 0
            num_blocks_to_allocate = max(
                num_required_blocks - len(existing_blocks), 0
            )
        else:
            # New request — account for prefix cache hits and compute gap
            num_skipped_tokens = self.get_num_skipped_tokens(
                total_computed_tokens
            )
            num_skipped_blocks = num_skipped_tokens // self.block_size
            num_cached = len(new_computed_block_list) + len(existing_blocks)

            num_new_blocks = max(
                num_required_blocks - max(num_skipped_blocks, num_cached), 0
            )

            # Count evictable blocks among new_computed_blocks (they're in
            # the free queue with ref_cnt=0 — touching will remove them)
            num_skipped_new_computed = max(
                0, num_skipped_blocks - len(existing_blocks)
            )
            evictable = sum(
                1 for b in new_computed_block_list[num_skipped_new_computed:]
                if b.ref_cnt == 0 and not b.is_null
            )
            num_blocks_to_allocate = num_new_blocks + evictable

        # Check capacity
        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            return None  # OOM — signal scheduler to preempt or reject

        # ── Stage 2: Handle prefix tokens ──
        if new_computed_block_list:
            # Touch cached blocks to prevent eviction
            if self.enable_caching:
                self.block_pool.touch(new_computed_block_list)

            # Add computed blocks to the request
            req_blocks = self.req_to_blocks[request.request_id]
            req_blocks.extend(new_computed_block_list)
            self.num_cached_block[request.request_id] = len(req_blocks)

        # ── Stage 3: Allocate new blocks ──
        req_blocks = self.req_to_blocks[request.request_id]
        num_required = math.ceil(num_tokens_need_slot / self.block_size)
        num_new = num_required - len(req_blocks)

        new_blocks: list['KVCacheBlock'] = []
        if num_new > 0:
            new_blocks = self.block_pool.get_new_blocks(num_new)
            req_blocks.extend(new_blocks)

        # Cache full blocks for prefix reuse (only finalized tokens)
        if self.enable_caching:
            self.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    # ── Cache Blocks ────────────────────────────────────────────────────

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """Mark full blocks as cached for prefix reuse.

        REFERENCE: vllm/v1/core/kv_cache_manager.py:L515-L524
        REFERENCE: vllm/v1/core/single_type_kv_cache_manager.py:L277-L301
        """
        if not self.enable_caching:
            return

        num_cached = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        if num_cached >= num_full_blocks:
            return

        self.block_pool.cache_full_blocks(
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached,
            num_full_blocks=num_full_blocks,
            request_block_hashes=request.block_hashes,
        )

        self.num_cached_block[request.request_id] = num_full_blocks

    # ── Free ────────────────────────────────────────────────────────────

    def free(self, request_id: str) -> None:
        """Free all blocks for a request (reverse order for LRU eviction).

        Freeing in reverse order ensures tail blocks (decoding-only tokens)
        are evicted first — they're least likely to be shared across requests.

        REFERENCE: vllm/v1/core/kv_cache_manager.py:L418-L426
        """
        req_blocks = self.req_to_blocks.pop(request_id, [])
        if not req_blocks:
            return

        # Reverse order: tail first → evicted first
        self.block_pool.free_blocks(reversed(req_blocks))
        self.num_cached_block.pop(request_id, None)

    # ── Skipped Blocks (Sliding Window) ─────────────────────────────────

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """Free blocks outside the attention window.

        Full attention never skips tokens (get_num_skipped_tokens returns 0),
        so this is a no-op. Sliding window models override this.

        REFERENCE: vllm/v1/core/kv_cache_manager.py:L428-L439
        """
        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        if num_skipped_tokens <= 0:
            return

        blocks = self.req_to_blocks.get(request_id, [])
        if not blocks:
            return

        num_skipped_blocks = min(
            num_skipped_tokens // self.block_size, len(blocks)
        )
        removed = []
        for i in range(num_skipped_blocks - 1, -1, -1):
            if blocks[i] is self._null_block:
                break  # Already freed
            removed.append(blocks[i])
            blocks[i] = self._null_block
        self.block_pool.free_blocks(removed)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """Tokens to skip for attention. Full attention: 0. SWA: >0."""
        return 0

    # ── Query ───────────────────────────────────────────────────────────

    def get_blocks(self, request_id: str) -> list['KVCacheBlock']:
        """Get all blocks for a request."""
        return self.req_to_blocks.get(request_id, [])

    def get_block_ids(self, request_id: str) -> list[int]:
        """Get block IDs for a request (passed to model runner)."""
        return [b.block_id for b in self.get_blocks(request_id)]


# ─── Block Hash Computation ─────────────────────────────────────────────────

def compute_request_block_hashes(
    token_ids: list[int],
    block_size: int = 16,
) -> list[bytes]:
    """Compute incremental block hashes for an entire request.

    Returns a hash chain where hash_i = SHA256(hash_{i-1}, tokens_i).
    These hashes are stored on the request and used for prefix cache lookups.

    REFERENCE: vllm/v1/core/kv_cache_utils.py:L635-L686

    Args:
        token_ids: All token IDs for the request.
        block_size: Number of tokens per block.

    Returns:
        List of block hashes, one per full block.
    """
    from block_pool import hash_block_tokens, NONE_HASH

    hashes: list[bytes] = []
    prev_hash = NONE_HASH

    for start in range(0, len(token_ids), block_size):
        end = start + block_size
        if end > len(token_ids):
            break  # Partial block — don't hash
        block_tokens = token_ids[start:end]
        block_hash = hash_block_tokens(
            parent_hash=None if start == 0 else hashes[-1],
            token_ids=block_tokens,
        )
        hashes.append(block_hash)

    return hashes


# ─── Demonstration ──────────────────────────────────────────────────────────

def main():
    """Demonstrate the full KVCacheManager lifecycle.

    Simulates a simple scenario:
    1. Request A: prompt → allocate → compute → cache
    2. Request B: same prefix → prefix cache hit → allocate fewer blocks
    3. Request A finishes → free (tail blocks evicted first)
    """
    from block_pool import BlockPool

    print("=" * 70)
    print("KVCacheManager Demo: Full Request Lifecycle")
    print("=" * 70)

    # Setup: 256-block pool, 16 tokens per block
    pool = BlockPool(num_gpu_blocks=256, enable_caching=True)
    mgr = KVCacheManager(
        block_pool=pool,
        block_size=16,
        max_model_len=4096,
        enable_caching=True,
    )
    print(f"\nPool: {pool.num_gpu_blocks - 1} usable blocks "
          f"+ 1 null block")
    print(f"Block size: {mgr.block_size} tokens")
    print(f"Max model length: {mgr.max_model_len}")
    print(f"Initial usage: {mgr.usage:.1%}")

    # ── Request A: 48-token prompt ──
    prompt_a = list(range(1, 49))  # 48 tokens = 3 blocks
    block_hashes_a = compute_request_block_hashes(prompt_a, block_size=16)
    request_a = Request(
        request_id="req-a",
        prompt_token_ids=prompt_a,
        max_tokens=100,
    )
    request_a.block_hashes = block_hashes_a

    print(f"\n--- Request A: 48-token prompt (3 blocks) ---")
    print(f"  Tokens: {request_a.num_tokens}")
    print(f"  Block hashes: {len(block_hashes_a)}")

    # First, check prefix cache (should be empty for first request)
    cached_blocks, hit_tokens = mgr.get_computed_blocks(request_a)
    print(f"  Prefix cache hit: {hit_tokens} tokens "
          f"({'HIT' if hit_tokens > 0 else 'MISS'})")

    # Allocate: need 3 blocks for the prompt
    result = mgr.allocate_slots(
        request=request_a,
        num_new_tokens=request_a.num_tokens,
        num_new_computed_tokens=hit_tokens,
        new_computed_blocks=cached_blocks if hit_tokens > 0 else None,
    )
    assert result is not None, "Allocation failed!"
    block_ids_a = result.get_block_ids()
    print(f"  Allocated blocks: {block_ids_a}")
    print(f"  Pool usage: {mgr.usage:.1%}")

    # Simulate prefill: mark as computed
    request_a.num_computed_tokens = request_a.num_tokens

    # ── Request B: same 48-token prefix + 16 new tokens ──
    prompt_b = list(range(1, 65))  # 48 same + 16 new = 64 tokens = 4 blocks
    block_hashes_b = compute_request_block_hashes(prompt_b, block_size=16)
    request_b = Request(
        request_id="req-b",
        prompt_token_ids=prompt_b,
        max_tokens=100,
    )
    request_b.block_hashes = block_hashes_b

    print(f"\n--- Request B: same prefix + 16 new tokens (4 blocks) ---")
    print(f"  Tokens: {request_b.num_tokens}")
    print(f"  Block hashes: {len(block_hashes_b)}")

    # Check prefix cache — should hit!
    cached_blocks_b, hit_tokens_b = mgr.get_computed_blocks(request_b)
    print(f"  Prefix cache hit: {hit_tokens_b} tokens "
          f"({hit_tokens_b // mgr.block_size} blocks)")

    # Allocate: need 4 blocks total (64 tokens), 3 are cached, only 1 new.
    # num_new_tokens = non-cached tokens to compute = 64 - 48 = 16
    remaining_tokens = request_b.num_tokens - hit_tokens_b
    result_b = mgr.allocate_slots(
        request=request_b,
        num_new_tokens=remaining_tokens,
        num_new_computed_tokens=hit_tokens_b,
        new_computed_blocks=cached_blocks_b if hit_tokens_b > 0 else None,
    )
    assert result_b is not None, "Allocation failed!"
    block_ids_b = result_b.get_block_ids()
    new_block_count = len(block_ids_b)
    print(f"  Newly allocated: {new_block_count} block(s) {block_ids_b}")
    print(f"  Total blocks: {len(mgr.get_blocks('req-b'))}")
    print(f"  Pool usage: {mgr.usage:.1%}")

    # Verify prefix sharing: blocks 0-2 should be shared
    all_blocks_b = mgr.get_blocks("req-b")
    print(f"  Block 0 ref_cnt: {all_blocks_b[0].ref_cnt} (shared by 2 requests)")
    print(f"  Block 1 ref_cnt: {all_blocks_b[1].ref_cnt} (shared by 2 requests)")
    print(f"  Block 2 ref_cnt: {all_blocks_b[2].ref_cnt} (shared by 2 requests)")
    if len(all_blocks_b) > 3:
        print(f"  Block 3 ref_cnt: {all_blocks_b[3].ref_cnt} (unique to B)")

    # ── Request A finishes → free (reverse order) ──
    print(f"\n--- Freeing Request A (reverse order) ---")
    mgr.free("req-a")
    print(f"  Pool usage after free: {mgr.usage:.1%}")
    # Block 2 (tail, unique) freed first, blocks 0-1 still shared
    print(f"  Block 0 ref_cnt: {all_blocks_b[0].ref_cnt} (still shared by B)")
    print(f"  Block 2 ref_cnt: {all_blocks_b[2].ref_cnt} (freed, now eviction candidate)")

    # ── Show block table ──
    print(f"\n--- Final State ---")
    print(f"  Request B block_ids: {mgr.get_block_ids('req-b')}")
    print(f"  Pool usage: {mgr.usage:.1%}")
    print(f"  Free blocks: {pool.get_num_free_blocks()}")
    print(f"  Cached blocks in hash index: {len(pool.cached_block_hash_to_block)}")

    print("\n" + "=" * 70)
    print("Key takeaway: KVCacheManager orchestrates the 3-stage allocation")
    print("flow. Prefix cache sharing means Request B only allocated 1 new")
    print("block instead of 4. Reverse-order freeing preserves shared")
    print("prefix blocks in cache while evicting tail blocks first.")
    print("=" * 70)


if __name__ == "__main__":
    main()
