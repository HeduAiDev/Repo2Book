# SOURCE: vllm/v1/core/kv_cache_utils.py
# 分页账本三件套的最底两件：KVCacheBlock 七字段元数据 + FreeKVCacheBlockQueue
# 侵入式自由队列（m2/m3 主角）。哈希两字段（_block_hash/_block_hash_num_tokens）
# 本章只当「块上留着缓存账位」——链式哈希/前缀命中工具族归 ch15。
# SUBTRACTED: 哈希侧全链（dossier.delete 第 3 条）：hash_block_tokens/
#   generate_block_hash_extra_keys/need_extra_keys/get_request_block_hasher/
#   request_block_hasher/resolve_block_hashes/BlockHashList(WithBlockSize)/
#   find_longest_cache_hit 族/init_none_hash/CBOR 哈希函数（→ ch15 精简版）。
from dataclasses import dataclass
from typing import NewType

# SOURCE: vllm/v1/core/kv_cache_utils.py:L41-L44 BlockHash（前缀缓存块哈希，
# ch15 的键类型——本章仅作字段类型账位保留）
BlockHash = NewType("BlockHash", bytes)

# SOURCE: vllm/v1/core/kv_cache_utils.py:L46-L49 BlockHashWithGroupId
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)


# SOURCE: vllm/v1/core/kv_cache_utils.py:L117 KVCacheBlock
@dataclass(slots=True)
class KVCacheBlock:
    """KV-cache block metadata."""

    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L121-L122
    block_id: int
    # Reference count.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L123-L124
    ref_cnt: int = 0
    # The hash key (block hash + group id) of the block, only available
    # when the block is full and cached.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L125-L127
    _block_hash: BlockHashWithGroupId | None = None
    # Number of prefix tokens covered by _block_hash. For full blocks this is
    # the full block boundary; partial entries can end inside a cache block.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L128-L130
    _block_hash_num_tokens: int | None = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L132-L135
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    # Whether the block is a null block that should never be cached.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L137-L138
    is_null: bool = False

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L140 block_hash property
    @property
    def block_hash(self) -> BlockHashWithGroupId | None:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L142
        return self._block_hash

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L144 block_hash_num_tokens property
    @property
    def block_hash_num_tokens(self) -> int | None:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L146
        return self._block_hash_num_tokens

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L148 set_block_hash
    def set_block_hash(
        self,
        block_hash: BlockHashWithGroupId,
        num_tokens: int | None = None,
    ) -> None:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L153-L157
        assert self.block_hash is None and self._block_hash_num_tokens is None, (
            "The block already has a hash. This should not happen."
        )
        self._block_hash = block_hash
        self._block_hash_num_tokens = num_tokens

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L159 reset_hash
    def reset_hash(self):
        """Reset the block hash when the block is evicted."""
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L161-L162
        self._block_hash = None
        self._block_hash_num_tokens = None

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L164 __repr__
    def __repr__(self) -> str:
        # Use block_id instead of KVCacheBlock object to avoid calling __repr__
        # on KVCacheBlock object recursively.
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L167-L176
        prev_block_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_block_id = self.next_free_block.block_id if self.next_free_block else None
        return (
            f"KVCacheBlock(block_id={self.block_id}, "
            f"ref_cnt={self.ref_cnt}, "
            f"_block_hash={self._block_hash!r}, "
            f"_block_hash_num_tokens={self._block_hash_num_tokens}, "
            f"prev_free_block={prev_block_id}, "
            f"next_free_block={next_block_id})"
        )


# SUBTRACTED: KVCacheBlockCopy（L179-L181 NamedTuple）——CoW 拷贝对的跨进程
#   载体（dossier.delete 第 9 条，→ ch15）。


# SOURCE: vllm/v1/core/kv_cache_utils.py:L184 FreeKVCacheBlockQueue
class FreeKVCacheBlockQueue:
    """This class organizes a list of KVCacheBlock objects to a doubly linked
    list of free blocks. We implement this class instead of using Python
    builtin deque to support removing a block in the middle of the queue
    in O(1) time. To close the performance gap to the builtin deque which is
    implemented in C++, this class does not allocate any Python objects when
    manipulating the linked list. Instead, this class manipulates the
    prev_free_block and next_free_block attributes of the given blocks.

    The queue is ordered by block ID in the beginning. When a block is allocated
    and then freed, it will be appended back with the eviction order:
    1. The least recent used block is at the front (LRU).
    2. If two blocks have the same last accessed time (allocated by
       the same sequence), the one with more hash tokens (the tail of a block
       chain) is at the front.
    Note that we maintain this order by reversing the block order when free
    blocks of a request. This operation is outside of this class.

    Args:
        blocks: A list of KVCacheBlock objects.
    """

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L206 __init__
    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L207-L214（相邻块互串）
        self.num_free_blocks = len(blocks)

        # Initialize doubly links of consecutive blocks
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # Create a fake head and a tail block for the doubly linked list to
        # reduce branching in the code
        #
        # The implementation guaranteed that the fake head and tail
        # are NEVER got popped, so we could safely assume each real blocks
        # in the queue has prev and next blocks.
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L222-L230（哨兵挂两端）
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            # Connect fake_head and fake_tail to the first and last block
            # respectively.
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            # For empty list, simply connect the fake head and tail.
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L236 popleft
    def popleft(self) -> KVCacheBlock:
        """Pop the first free block and reduce num_free_blocks by 1.

        Returns:
            The first free block.
        """
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L242-L250（空队护栏）
        if (
            self.fake_free_list_head.next_free_block is self.fake_free_list_tail
            or self.fake_free_list_head.next_free_block is None
        ):
            assert self.num_free_blocks == 0, (
                f"num_free_blocks ({self.num_free_blocks}) is out of sync "
                "with the free list."
            )
            raise ValueError("No free blocks available")

        # SOURCE: vllm/v1/core/kv_cache_utils.py:L252-L260
        first_block: KVCacheBlock = self.fake_free_list_head.next_free_block

        if first_block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError(
                "Invalid block found in popleft() "
                "which doesn't have a valid next_free_block"
            )

        # Connect fake_head and the next block of first_block (i.e. second block
        # or fake tail).
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L262-L268
        self.fake_free_list_head.next_free_block = first_block.next_free_block
        first_block.next_free_block.prev_free_block = self.fake_free_list_head

        # Remove the block from the linked list.
        first_block.prev_free_block = first_block.next_free_block = None

        # SOURCE: vllm/v1/core/kv_cache_utils.py:L270-L271
        self.num_free_blocks -= 1
        return first_block

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L273 popleft_n
    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """Pop the first n free blocks and reduce num_free_blocks by n.

        Args:
            n: The number of blocks to pop.

        Returns:
            A list of n free blocks.
        """
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L282-L285
        if n == 0:
            return []
        assert self.num_free_blocks >= n
        self.num_free_blocks -= n

        # SOURCE: vllm/v1/core/kv_cache_utils.py:L287-L297（就地摘除、指针清零）
        curr_block = self.fake_free_list_head.next_free_block
        # Pop n blocks from the head of the list
        ret = []
        for _ in range(n):
            assert curr_block is not None
            ret.append(curr_block)
            last_block = curr_block
            curr_block = curr_block.next_free_block
            # Reset prev_free_block and next_free_block of all popped blocks
            last_block.prev_free_block = None
            last_block.next_free_block = None

        # SOURCE: vllm/v1/core/kv_cache_utils.py:L299-L304
        if curr_block is not None:
            # The queue is not empty, connect the fake head to
            # the new first block.
            self.fake_free_list_head.next_free_block = curr_block
            curr_block.prev_free_block = self.fake_free_list_head
        return ret

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L306 remove
    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block in the free list and reduce num_free_blocks by 1.

        Args:
            block: The block to remove.
        """
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L312-L324（O(1) 中间摘——
        # touch 救回命中块的关键原语，ch15 前置）
        if block.prev_free_block is None or block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError(f"remove() called on an invalid block: {block}")

        # Link the previous block to the next block.
        block.prev_free_block.next_free_block = block.next_free_block
        # Link the next block to the previous block.
        block.next_free_block.prev_free_block = block.prev_free_block

        # Remove the block from the linked list.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L326 append
    def append(self, block: KVCacheBlock) -> None:
        """Put a block back into the free list and increase
        num_free_blocks by 1.

        Args:
            block: The block to append.
        """
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L333-L347
        if self.fake_free_list_tail.prev_free_block is None:
            raise RuntimeError(
                "prev_free_block of fake_free_list_tail should always exist"
            )
        last_block: KVCacheBlock = self.fake_free_list_tail.prev_free_block

        # Connect the new block after the last block.
        last_block.next_free_block = block
        block.prev_free_block = last_block

        # Connect the fake tail after the new block.
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block

        self.num_free_blocks += 1

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L349 prepend_n
    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
        """Put a list of blocks at the front of the free list."""
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L351-L368
        if len(blocks) == 0:
            return

        first_block = self.fake_free_list_head.next_free_block
        assert first_block is not None, (
            "next_free_block of fake_free_list_head should always exist"
        )

        prev_block = self.fake_free_list_head
        for block in blocks:
            block.prev_free_block = prev_block
            prev_block.next_free_block = block
            prev_block = block

        prev_block.next_free_block = first_block
        first_block.prev_free_block = prev_block

        self.num_free_blocks += len(blocks)

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L370 append_n
    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """Put a list of blocks back into the free list

        Args:
            blocks: The blocks to append.
        """
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L376-L393
        if len(blocks) == 0:
            return

        last_block = self.fake_free_list_tail.prev_free_block
        assert last_block is not None, (
            "prev_free_block of fake_free_list_tail should always exist"
        )
        # Add inter-connections between consecutive blocks
        for block in blocks:
            block.prev_free_block = last_block
            last_block.next_free_block = block
            last_block = block

        # Connect the last block of <blocks> to the fake tail
        last_block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = last_block

        self.num_free_blocks += len(blocks)

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L395 get_all_free_blocks
    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Get all free blocks in the free list. Mainly used for testing.

        Returns:
            A list of free blocks.
        """
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L401-L413
        ret = []
        if self.fake_free_list_head.next_free_block is None:
            raise RuntimeError(
                "next_free_block of fake_free_list_head should always exist"
            )
        # Start from the first block
        curr_block: KVCacheBlock = self.fake_free_list_head.next_free_block
        # As long as next_free_block is available, we haven't reached to
        # the fake tail yet.
        while curr_block.next_free_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret
