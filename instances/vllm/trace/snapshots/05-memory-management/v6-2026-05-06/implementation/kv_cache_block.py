# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L113-L370
"""KVCacheBlock + FreeKVCacheBlockQueue — the unit of GPU memory accounting.

A KVCacheBlock is metadata for one fixed-size slab of GPU KV-cache memory —
*not* the GPU bytes themselves (those live in a flat torch.zeros tensor
allocated by the worker). This metadata struct tracks:

    - block_id            : the slot index in the GPU tensor
    - ref_cnt             : how many running requests are using this block
    - block_hash          : the prefix-cache hash key when the block is full
    - prev/next free      : doubly-linked list pointers for the free queue
    - is_null             : the placeholder block 0 for sliding-window padding

The free queue is a doubly-linked list, ordered LRU. `popleft()` returns the
least-recently-used block; `append_n()` pushes recently-freed blocks back.
Why not a `deque`? Because vLLM also needs *O(1) remove from the middle* (when
a previously-freed block is re-touched by a prefix-cache hit). `deque` is O(n)
for that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L113-L159
@dataclass(slots=True)
class KVCacheBlock:
    """Metadata for one fixed-size KV cache block.

    Field names match vLLM exactly. `is_null` is the placeholder bit for the
    "block 0" sliding-window padding.
    """

    # REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L117-L131
    block_id: int
    ref_cnt: int = 0
    _block_hash: Optional[bytes] = None  # vLLM uses BlockHashWithGroupId

    prev_free_block: Optional["KVCacheBlock"] = None
    next_free_block: Optional["KVCacheBlock"] = None

    is_null: bool = False

    @property
    def block_hash(self) -> Optional[bytes]:
        # REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L133-L135
        return self._block_hash

    @block_hash.setter
    def block_hash(self, value: bytes) -> None:
        # REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L137-L142
        assert self._block_hash is None, "Block already has a hash; this is a bug."
        self._block_hash = value

    def reset_hash(self) -> None:
        # REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L144-L146
        self._block_hash = None


# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L162-L370
class FreeKVCacheBlockQueue:
    """Doubly-linked list of free blocks, with O(1) middle-removal.

    Using a linked list (not deque) is deliberate: when a request hits the
    prefix cache for a block currently in the free queue, vLLM needs to
    `remove(block)` from the middle in O(1) — see `BlockPool.touch()`.
    A deque would be O(n) for that, undermining the entire prefix-cache
    fast path.

    The fake head/tail eliminate branching on null pointers. Real blocks
    sit between `fake_free_list_head` and `fake_free_list_tail`.

    LRU eviction order:
        head = least-recently-used (next to be allocated/evicted)
        tail = most-recently-freed
    """

    # REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L184-L212
    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)

        # Initial double-link of consecutive blocks.
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # Fake head / tail eliminate null-pointer branching.
        # REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L194-L212
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

    # REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L214-L249
    def popleft(self) -> KVCacheBlock:
        """Pop and return the LRU (head) block."""
        first = self.fake_free_list_head.next_free_block
        if first is self.fake_free_list_tail or first is None:
            assert self.num_free_blocks == 0
            raise ValueError("No free blocks available")

        nxt = first.next_free_block
        assert nxt is not None
        self.fake_free_list_head.next_free_block = nxt
        nxt.prev_free_block = self.fake_free_list_head
        first.prev_free_block = first.next_free_block = None
        self.num_free_blocks -= 1
        return first

    # REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L251-L282
    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """Pop n LRU blocks. Bulk variant of popleft for `BlockPool.get_new_blocks`."""
        if n == 0:
            return []
        assert self.num_free_blocks >= n
        return [self.popleft() for _ in range(n)]

    # REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L284-L302
    def remove(self, block: KVCacheBlock) -> None:
        """O(1) removal from the middle. Called when a freed block is re-touched."""
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"remove() called on an invalid block: {block}")
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    # REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L304+
    def append(self, block: KVCacheBlock) -> None:
        """Append `block` to the tail (most-recently-freed end)."""
        prev = self.fake_free_list_tail.prev_free_block
        assert prev is not None
        prev.next_free_block = block
        block.prev_free_block = prev
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block
        self.num_free_blocks += 1

    def append_n(self, blocks: Iterable[KVCacheBlock]) -> None:
        """Append many blocks. Same as `BlockPool.free_blocks` calls into."""
        for b in blocks:
            self.append(b)

    def __len__(self) -> int:
        return self.num_free_blocks
