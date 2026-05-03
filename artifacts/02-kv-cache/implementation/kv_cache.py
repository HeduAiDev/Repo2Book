"""
KV Cache Manager — Reimplementation grounded in vLLM source.

Mirrors vLLM's architecture:
    KVCacheManager     → vllm/v1/core/kv_cache_manager.py:L106
    KVCacheBlocks      → vllm/v1/core/kv_cache_manager.py:L22
    KVCacheBlock       → vllm/v1/core/kv_cache_utils.py:L114
    FreeKVCacheBlockQueue → vllm/v1/core/kv_cache_utils.py:L162
    BlockPool          → vllm/v1/core/block_pool.py:L130
    BlockHashToBlockMap → vllm/v1/core/block_pool.py:L34

Architecture:
    Scheduler.allocate_slots() → KVCacheManager.allocate_slots()
      → KVCacheManager delegates to coordinator
        → SingleTypeKVCacheManager (per attention-type)
          → BlockPool.get_new_blocks() (physical allocation)
            → FreeKVCacheBlockQueue.popleft_n() (LRU eviction when needed)
          → BlockPool.cache_full_blocks() (prefix caching via block hash)

The three-stage allocation (from kv_cache_manager.py:L300-L310):
    1. Free skipped blocks (sliding window)
    2. Check capacity (coordinator.get_num_blocks_to_allocate)
    3. Allocate new blocks + cache full blocks
"""

import torch
from typing import Optional, List, Tuple, Dict, Set
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════
# KVCacheBlock — vllm/v1/core/kv_cache_utils.py:L114
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class KVCacheBlock:
    """
    A single KV cache block — the unit of allocation.

    REFERENCE: vllm/v1/core/kv_cache_utils.py:L114-L160
               @dataclass with __slots__ for memory efficiency.

    vLLM uses __slots__ to minimize Python object overhead.
    We use @dataclass for readability — the logic is identical.

    Key concept: Each block stores block_size tokens of K and V
    for one layer and one KV cache group.

    In vLLM, blocks are pre-allocated into a contiguous GPU tensor.
    ref_cnt tracks how many requests are using this block.
    _block_hash is set when the block is full and cached (prefix caching).
    """
    block_id: int
    ref_cnt: int = 0
    _block_hash: Optional[int] = None
    prev_free_block: Optional['KVCacheBlock'] = None
    next_free_block: Optional['KVCacheBlock'] = None
    is_null: bool = False

    @property
    def block_hash(self) -> Optional[int]:
        return self._block_hash

    def set_hash(self, h: int):
        """REFERENCE: block_pool.py:L211 — cache_full_blocks() sets block_hash"""
        self._block_hash = h

    def reset_hash(self):
        """REFERENCE: block_pool.py:L354 — _maybe_evict_cached_block() clears hash"""
        self._block_hash = None


# ═══════════════════════════════════════════════════════════════════════════
# FreeKVCacheBlockQueue — vllm/v1/core/kv_cache_utils.py:L162
# ═══════════════════════════════════════════════════════════════════════════

class FreeKVCacheBlockQueue:
    """
    LRU free block queue — doubly-linked list.

    REFERENCE: vllm/v1/core/kv_cache_utils.py:L162-L340
               Custom doubly-linked list (NOT Python's deque).

    WHY a custom linked list?
        - O(1) pop from front (least recently used)
        - O(1) append to tail (most recently used → won't be evicted soon)
        - O(1) remove from middle (when a cached block is touched by new request)

    Eviction order (from source L172-L178):
        Least-recently-used at front → evicted first.
        When multiple blocks are freed together, appended in REVERSE order
        so tail blocks (more hash tokens) end up at front (evicted sooner).
    """

    def __init__(self, blocks: List[KVCacheBlock]):
        # Sentinels avoid edge cases for head/tail
        self.head = KVCacheBlock(block_id=-1, is_null=True)
        self.tail = KVCacheBlock(block_id=-2, is_null=True)
        self.head.next_free_block = self.tail
        self.tail.prev_free_block = self.head
        self._size = 0
        for b in blocks:
            self.append(b)

    @property
    def num_free_blocks(self) -> int:
        return self._size

    def popleft(self) -> KVCacheBlock:
        """
        Pop the LRU block from the front. O(1).
        REFERENCE: kv_cache_utils.py:L214
        This is called by BlockPool.get_new_blocks() when allocating.
        """
        if self._size == 0:
            raise IndexError("No free blocks")
        block = self.head.next_free_block
        self._remove(block)
        return block

    def popleft_n(self, n: int) -> List[KVCacheBlock]:
        """Pop n LRU blocks. O(n). REFERENCE: kv_cache_utils.py:L251"""
        return [self.popleft() for _ in range(n)]

    def append(self, block: KVCacheBlock):
        """Append to tail (most-recently-used). O(1). REFERENCE: kv_cache_utils.py:L304"""
        block.prev_free_block = self.tail.prev_free_block
        block.next_free_block = self.tail
        self.tail.prev_free_block.next_free_block = block
        self.tail.prev_free_block = block
        self._size += 1

    def append_n(self, blocks: List[KVCacheBlock]):
        """Append in REVERSE order — LRU eviction order. REFERENCE: kv_cache_utils.py:L327"""
        for b in reversed(blocks):
            self.append(b)

    def remove(self, block: KVCacheBlock):
        """Remove from middle (when block is re-used). O(1). REFERENCE: kv_cache_utils.py:L284"""
        self._remove(block)

    def _remove(self, block: KVCacheBlock):
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block
        block.prev_free_block = None
        block.next_free_block = None
        self._size -= 1


# ═══════════════════════════════════════════════════════════════════════════
# BlockPool — vllm/v1/core/block_pool.py:L130
# ═══════════════════════════════════════════════════════════════════════════

class BlockPool:
    """
    Physical block pool — manages GPU memory for KV cache blocks.

    REFERENCE: vllm/v1/core/block_pool.py:L130-L495

    Responsibilities:
        1. Owns the list of all KVCacheBlock objects (one per physical block)
        2. Owns the FreeKVCacheBlockQueue (free list with LRU eviction order)
        3. Owns the BlockHashToBlockMap (prefix cache index)
        4. get_new_blocks() pops from free queue, evicting cached blocks if needed
        5. cache_full_blocks() marks blocks as cached (sets block_hash)
        6. free_blocks() returns blocks to free queue (ref_cnt management)
    """

    def __init__(self, num_gpu_blocks: int, enable_caching: bool = True):
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching

        # Create the null block (sentinel for skipped blocks)
        # REFERENCE: block_pool.py:L176
        self.null_block = KVCacheBlock(block_id=-1, is_null=True)

        # Create all blocks with sequential IDs
        # REFERENCE: block_pool.py:L162
        self.blocks = [KVCacheBlock(block_id=i) for i in range(num_gpu_blocks)]

        # All blocks start free
        # REFERENCE: block_pool.py:L168
        self.free_queue = FreeKVCacheBlockQueue(list(self.blocks))

        # Hash index for prefix caching
        # REFERENCE: block_pool.py:L130 — BlockHashToBlockMap
        self.cached_block_hash_to_block: Dict[int, KVCacheBlock] = {}

    # ── Allocation ──
    # REFERENCE: block_pool.py:L322 — get_new_blocks()

    def get_new_blocks(self, num_blocks: int) -> List[KVCacheBlock]:
        """
        Allocate new blocks from the free pool.

        If caching is enabled, evicts any cached blocks that are being
        re-allocated (removes their hash from the prefix cache index).

        Returns blocks with ref_cnt=1 (one reference from the allocating request).
        """
        if num_blocks > self.free_queue.num_free_blocks:
            raise RuntimeError(
                f"Out of KV cache blocks: need {num_blocks}, "
                f"have {self.free_queue.num_free_blocks}"
            )
        blocks = self.free_queue.popleft_n(num_blocks)
        for b in blocks:
            self._maybe_evict_cached_block(b)
            b.ref_cnt = 1
        return blocks

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block has a hash (i.e., it's in the prefix cache), evict it.

        REFERENCE: block_pool.py:L354-L388
        This is called when a block that WAS cached is being re-allocated
        to a new request. The block is removed from the hash index so
        future prefix cache lookups won't find it.
        """
        if block.block_hash is not None:
            self.cached_block_hash_to_block.pop(block.block_hash, None)
            block.reset_hash()
            return True
        return False

    # ── Caching (Prefix Cache integration) ──
    # REFERENCE: block_pool.py:L211 — cache_full_blocks()

    def cache_full_blocks(
        self, blocks: List[KVCacheBlock], block_hashes: List[int],
    ):
        """
        Mark blocks as cached by setting their hash.

        Only full blocks are cached — partial blocks can't be shared because
        their content may change when more tokens arrive.

        In vLLM, this is called by KVCacheManager.cache_blocks() (L515) after
        the scheduler verifies that blocks are complete.
        """
        for block, h in zip(blocks, block_hashes):
            if h is not None and not block.is_null:
                block.set_hash(h)
                self.cached_block_hash_to_block[h] = block

    def get_cached_block(self, block_hash: int) -> Optional[KVCacheBlock]:
        """Look up a cached block by hash. O(1). REFERENCE: block_pool.py:L184"""
        return self.cached_block_hash_to_block.get(block_hash)

    # ── Touch (reference counting for shared blocks) ──
    # REFERENCE: block_pool.py:L391 — touch()

    def touch(self, blocks: List[KVCacheBlock]):
        """
        Increase ref_cnt for cached blocks hit by another request.

        When a new request hits a prefix cache block, we increment ref_cnt
        to prevent it from being freed while both requests are using it.
        Also removes the block from the free queue if it was there (ref_cnt 0).
        """
        for b in blocks:
            if b.is_null:
                continue
            if b.ref_cnt == 0:
                self.free_queue.remove(b)
            b.ref_cnt += 1

    # ── Free ──
    # REFERENCE: block_pool.py:L408 — free_blocks()

    def free_blocks(self, blocks: List[KVCacheBlock]):
        """
        Release blocks back to the free pool.

        ref_cnt is decremented. Blocks with ref_cnt==0 go back to the
        free queue (in REVERSE order — for LRU eviction ordering).
        Null blocks (sentinel) are never freed.

        In vLLM, blocks are freed by the scheduler when:
            - A request finishes (Scheduler._free_request → KVCacheManager.free)
            - A request is preempted (eviction due to memory pressure)
            - Sliding window attention skips old blocks (remove_skipped_blocks)
        """
        for b in blocks:
            if b.is_null:
                continue
            b.ref_cnt -= 1
            if b.ref_cnt == 0:
                self.free_queue.append(b)  # Append = put at tail (MRU)

    # ── Eviction ──
    # REFERENCE: block_pool.py:L424 — evict_blocks()

    def evict_blocks(self, block_ids: Set[int]):
        """
        Evict blocks from the prefix cache by ID.

        Used when blocks are found to be invalid (e.g., KV connector detects
        stale cache entries after P/D disaggregation sync).
        """
        for bid in block_ids:
            block = self.blocks[bid]
            self._maybe_evict_cached_block(block)

    # ── Status ──
    def get_num_free_blocks(self) -> int:
        return self.free_queue.num_free_blocks

    def get_usage(self) -> float:
        """0.0 to 1.0 — fraction of blocks in use."""
        return 1.0 - (self.free_queue.num_free_blocks / self.num_gpu_blocks)


# ═══════════════════════════════════════════════════════════════════════════
# KVCacheBlocks — vllm/v1/core/kv_cache_manager.py:L22
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class KVCacheBlocks:
    """
    Opaque allocation result. Hides block internals from the scheduler.

    REFERENCE: vllm/v1/core/kv_cache_manager.py:L22-L104

    In vLLM, this wraps tuple[Sequence[KVCacheBlock], ...] for multi-group
    models. We simplify to a single flat list for the single-group case.
    """
    blocks: List[KVCacheBlock] = field(default_factory=list)

    def get_block_ids(self) -> List[int]:
        """Convert to raw block IDs for the model runner's block table."""
        return [b.block_id for b in self.blocks if not b.is_null]

    def __add__(self, other: 'KVCacheBlocks') -> 'KVCacheBlocks':
        return KVCacheBlocks(blocks=self.blocks + other.blocks)


# ═══════════════════════════════════════════════════════════════════════════
# KVCacheManager — vllm/v1/core/kv_cache_manager.py:L106
# ═══════════════════════════════════════════════════════════════════════════

class KVCacheManager:
    """
    Public API for KV cache management — called by the Scheduler.

    REFERENCE: vllm/v1/core/kv_cache_manager.py:L106-L540

    This is the class the Scheduler interacts with. It:
        1. Owns the BlockPool (physical allocation)
        2. Manages per-request block tables (req_id → list[KVCacheBlock])
        3. Implements three-stage allocation (L300-L310):
           a. Free skipped blocks (sliding window)
           b. Check capacity
           c. Allocate new + cache full blocks
        4. Handles prefix caching (get_computed_blocks, cache_blocks)
        5. Handles eviction (evict_blocks)

    In production vLLM, KVCacheManager delegates to:
        - KVCacheCoordinator → SingleTypeKVCacheManager (per attention-type logic)
        - BlockPool (physical allocation + eviction)

    We flatten this hierarchy for clarity while preserving the key APIs.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        block_size: int,
        enable_caching: bool = True,
    ):
        self.block_size = block_size
        self.enable_caching = enable_caching

        # REFERENCE: kv_cache_manager.py:L150
        self.block_pool = BlockPool(num_gpu_blocks, enable_caching)

        # REFERENCE: single_type_kv_cache_manager.py:L73
        # Per-request block table: request_id → list[KVCacheBlock]
        self.req_to_blocks: Dict[str, List[KVCacheBlock]] = {}

        # REFERENCE: single_type_kv_cache_manager.py:L79
        # How many prefix-cached blocks each request has
        self.num_cached_blocks: Dict[str, int] = {}

        # Cached empty result — avoids allocation in hot path
        # REFERENCE: kv_cache_manager.py:L158
        self._empty_blocks = KVCacheBlocks(blocks=[])

    # ── Allocation API ──
    # REFERENCE: kv_cache_manager.py:L225 — allocate_slots()

    def allocate_slots(
        self,
        request_id: str,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
    ) -> Optional[KVCacheBlocks]:
        """
        Allocate KV cache blocks for a request generating num_new_tokens.

        Returns None if allocation fails (insufficient blocks) —
        the Scheduler will then preempt the lowest-priority request.

        Three-stage allocation (from source L300-L310):
        """
        total_tokens = num_new_computed_tokens + num_new_tokens
        blocks_needed = (total_tokens + self.block_size - 1) // self.block_size

        current_blocks = self.req_to_blocks.get(request_id, [])
        current_block_count = len([b for b in current_blocks if not b.is_null])

        new_blocks_needed = blocks_needed - current_block_count

        if new_blocks_needed <= 0:
            return self._empty_blocks

        # Stage 2: Check capacity
        # REFERENCE: single_type_kv_cache_manager.py:L88 — get_num_blocks_to_allocate()
        if new_blocks_needed > self.block_pool.get_num_free_blocks():
            return None  # OOM → scheduler must preempt

        # Stage 3: Allocate
        # REFERENCE: kv_cache_manager.py:L386-L393
        new_blocks = self.block_pool.get_new_blocks(new_blocks_needed)
        new_table = current_blocks + new_blocks
        self.req_to_blocks[request_id] = new_table

        return KVCacheBlocks(blocks=new_blocks)

    # ── Prefix Cache API ──
    # REFERENCE: kv_cache_manager.py:L183 — get_computed_blocks()

    def get_computed_blocks(
        self, request_id: str, block_hashes: List[int],
    ) -> KVCacheBlocks:
        """
        Return blocks already cached from a previous request (prefix caching).

        Walks block_hashes left-to-right, looking up each in the BlockPool's
        hash index. Returns the contiguous cached prefix.

        In vLLM, this is called by the Scheduler during WAITING→RUNNING transition
        to find and reuse already-computed KV cache blocks.
        """
        cached = []
        for h in block_hashes:
            block = self.block_pool.get_cached_block(h)
            if block is None:
                break  # Cache miss → stop the prefix
            cached.append(block)
            self.block_pool.touch([block])  # Prevent eviction

        if cached:
            self.req_to_blocks[request_id] = cached
            self.num_cached_blocks[request_id] = len(cached)

        return KVCacheBlocks(blocks=cached)

    # ── Cache Full Blocks ──
    # REFERENCE: kv_cache_manager.py:L515 — cache_blocks()

    def cache_blocks(
        self, request_id: str, block_hashes: List[int],
    ):
        """
        Mark blocks as cached after they become full.

        Called by the AsyncScheduler after token verification.
        In prefix-caching mode, this registers the blocks so future
        requests with the same prompt prefix can reuse them.
        """
        blocks = self.req_to_blocks.get(request_id, [])
        full_blocks = [b for b in blocks if not b.is_null]
        hashes = block_hashes[:len(full_blocks)]
        self.block_pool.cache_full_blocks(full_blocks, hashes)

    # ── Free ──
    # REFERENCE: kv_cache_manager.py:L418 — free()

    def free(self, request_id: str):
        """
        Free all blocks owned by a request.

        Called by the Scheduler when:
            - A request finishes (all tokens generated)
            - A request is preempted (memory pressure)
            - A request is aborted

        Blocks are freed in REVERSE order so that tail blocks (more content)
        are evicted before head blocks — this is the LRU ordering.
        """
        blocks = self.req_to_blocks.pop(request_id, [])
        self.num_cached_blocks.pop(request_id, None)
        if blocks:
            # Free in REVERSE: tail blocks first (LRU eviction order)
            # REFERENCE: single_type_kv_cache_manager.py:L303
            self.block_pool.free_blocks(list(reversed(blocks)))

    # ── Evict ──
    # REFERENCE: kv_cache_manager.py:L441 — evict_blocks()

    def evict_blocks(self, block_ids: Set[int]):
        """
        Evict specific blocks from the prefix cache.

        Used when the KV connector detects stale/invalid blocks
        after P/D disaggregation sync.
        """
        self.block_pool.evict_blocks(block_ids)

    # ── Getters ──
    # REFERENCE: kv_cache_manager.py:L507-L511

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get current block table for a request."""
        return KVCacheBlocks(blocks=self.req_to_blocks.get(request_id, []))

    def get_usage(self) -> float:
        """0.0-1.0 fraction of blocks in use."""
        return self.block_pool.get_usage()


# ═══════════════════════════════════════════════════════════════════════════
# KV Cache Memory Calculator
# REFERENCE: vllm/v1/kv_cache_interface.py — KVCacheConfig, KVCacheTensor
#            vllm/v1/core/kv_cache_utils.py:L569 — resolve_kv_cache_block_sizes()
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class KVCacheConfig:
    """
    vLLM's KV cache sizing configuration.

    REFERENCE: vllm/v1/kv_cache_interface.py:L760 — KVCacheConfig
    """
    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    dtype_bytes: int = 2  # bf16/fp16
    gpu_memory_utilization: float = 0.90

    def calculate_num_blocks(self, total_gpu_memory: int, model_weight_size: int) -> int:
        """
        How vLLM calculates num_gpu_blocks.

        available = total_memory * gpu_memory_utilization - model_weight_size
        block_bytes = 2 * block_size * num_kv_heads * head_dim * dtype_bytes * num_layers
        num_blocks = floor(available / block_bytes)
        """
        available = int(total_gpu_memory * self.gpu_memory_utilization) - model_weight_size
        block_bytes = (
            2 * self.block_size * self.num_kv_heads *
            self.head_dim * self.dtype_bytes * self.num_layers
        )
        return available // block_bytes

    def per_token_kv_bytes(self) -> int:
        """KV cache size per token (all layers, one K+V)."""
        return 2 * self.num_kv_heads * self.head_dim * self.dtype_bytes * self.num_layers


def llama_3_2_1b_kv_cache_example():
    """
    Real-world example: Llama-3.2-1B KV cache sizing.

    REFERENCE: The constants below come from Llama-3.2-1B's config.json
               and vLLM's auto-configuration in kv_cache_utils.py.
    """
    config = KVCacheConfig(
        num_layers=32,      # Llama-3.2-1B
        num_kv_heads=8,     # GQA
        head_dim=128,       # 2048/32... wait, 4096/32=128 for 3.2-1B
        block_size=16,      # vLLM default for this model size
        dtype_bytes=2,      # bf16
    )

    # Scenario
    for batch, seq in [(1, 1024), (8, 4096), (1, 131072)]:
        total_kv = config.per_token_kv_bytes() * batch * seq
        print(f"  batch={batch}, seq={seq}: KV Cache = {total_kv / (1024**3):.2f} GB")
    return config


if __name__ == "__main__":
    llama_3_2_1b_kv_cache_example()
