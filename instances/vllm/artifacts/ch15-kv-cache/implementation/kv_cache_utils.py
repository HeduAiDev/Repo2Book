# SPDX-License-Identifier: Apache-2.0
# 只做减法的精简版 —— 忠实子集，与 vLLM 同名同结构同控制流。
# 验收判据：把真实 vLLM 删掉所有 # SUBTRACTED 分支，应当 ≈ 得到本文件。
#
# 对应 vllm/v1/core/kv_cache_utils.py：KVCacheBlock 元数据、FreeKVCacheBlockQueue
# 双向链表 LRU、extra keys 生成、链式块哈希、请求侧逐满块哈希。
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NewType

from .request import Request

# SOURCE: vllm/v1/core/kv_cache_utils.py:L39 (BlockHash / BlockHashWithGroupId)
BlockHash = NewType("BlockHash", bytes)
# `BlockHashWithGroupId` combines a `BlockHash` with its KV cache group ID.
# It is represented as raw bytes for compactness and efficiency.
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)


# SOURCE: vllm/v1/core/kv_cache_utils.py:L53 (make_block_hash_with_group_id)
def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int
) -> BlockHashWithGroupId:
    """Pack a `BlockHash` and group id into a `BlockHashWithGroupId`.

    The group id is encoded using 4 bytes in big-endian order and appended to
    the block hash bytes. This avoids creating tuples while still allowing us
    to recover both components when needed.
    """
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))


# SOURCE: vllm/v1/core/kv_cache_utils.py:L91 (NONE_HASH / init_none_hash)
# The hash seed for the first block of any prefix block sequence. We use a
# random value to avoid hash collisions, aligning with Python's hash().
# SUBTRACTED: init_none_hash(hash_fn) 的 PYTHONHASHSEED 分支与 CBOR 警告
# （L95-L110）—— 可复现哈希的运维选项；精简版直接用随机种子初始化。
NONE_HASH: BlockHash = BlockHash(os.urandom(32))


@dataclass(slots=True)
# SOURCE: vllm/v1/core/kv_cache_utils.py:L113 (class KVCacheBlock)
class KVCacheBlock:
    """KV-cache block metadata."""

    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # The hash key (block hash + group id) of the block, only available
    # when the block is full and cached.
    _block_hash: BlockHashWithGroupId | None = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    # Whether the block is a null block that should never be cached.
    is_null: bool = False

    @property
    def block_hash(self) -> BlockHashWithGroupId | None:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L133 (block_hash getter)
        return self._block_hash

    @block_hash.setter
    def block_hash(self, block_hash: BlockHashWithGroupId):
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L137 (block_hash setter)
        assert self.block_hash is None, (
            "The block already has a hash. This should not happen."
        )
        self._block_hash = block_hash

    def reset_hash(self):
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L144 (reset_hash)
        """Reset the block hash when the block is evicted."""
        self._block_hash = None

    # SUBTRACTED: __repr__（L148-L159）仅调试打印块指针，不参与控制流。


# SOURCE: vllm/v1/core/kv_cache_utils.py:L162 (class FreeKVCacheBlockQueue)
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
    2. If two blocks have the same last accessed time (allocated by the
       same sequence), the one with more hash tokens (the tail of a block
       chain) is at the front.
    Note that we maintain this order by reversing the block order when free
    blocks of a request. This operation is outside of this class.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L184 (FreeKVCacheBlockQueue.__init__)
        self.num_free_blocks = len(blocks)

        # Initialize doubly links of consecutive blocks
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # Create a fake head and a tail block for the doubly linked list to
        # reduce branching in the code.
        #
        # The implementation guaranteed that the fake head and tail
        # are NEVER got popped, so we could safely assume each real blocks
        # in the queue has prev and next blocks.
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            # Connect fake_head and fake_tail to the first and last block.
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            # For empty list, simply connect the fake head and tail.
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

    def popleft(self) -> KVCacheBlock:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L214 (popleft)
        """Pop the first free block and reduce num_free_blocks by 1."""
        if (
            self.fake_free_list_head.next_free_block is self.fake_free_list_tail
            or self.fake_free_list_head.next_free_block is None
        ):
            assert self.num_free_blocks == 0, (
                f"num_free_blocks ({self.num_free_blocks}) is out of sync "
                "with the free list."
            )
            raise ValueError("No free blocks available")

        first_block: KVCacheBlock = self.fake_free_list_head.next_free_block

        if first_block.next_free_block is None:
            # This should not happen if the block is from the free list.
            raise RuntimeError(
                "Invalid block found in popleft() "
                "which doesn't have a valid next_free_block"
            )

        # Connect fake_head and the next block of first_block (i.e. second block
        # or fake tail).
        self.fake_free_list_head.next_free_block = first_block.next_free_block
        first_block.next_free_block.prev_free_block = self.fake_free_list_head

        # Remove the block from the linked list.
        first_block.prev_free_block = first_block.next_free_block = None

        self.num_free_blocks -= 1
        return first_block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L251 (popleft_n)
        """Pop the first n free blocks and reduce num_free_blocks by n."""
        if n == 0:
            return []
        assert self.num_free_blocks >= n
        self.num_free_blocks -= n

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

        if curr_block is not None:
            # The queue is not empty, connect the fake head to
            # the new first block.
            self.fake_free_list_head.next_free_block = curr_block
            curr_block.prev_free_block = self.fake_free_list_head
        return ret

    def remove(self, block: KVCacheBlock) -> None:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L284 (remove)
        """Remove a block in the free list and reduce num_free_blocks by 1."""
        if block.prev_free_block is None or block.next_free_block is None:
            # This should not happen if the block is from the free list.
            raise RuntimeError(f"remove() called on an invalid block: {block}")

        # Link the previous block to the next block.
        block.prev_free_block.next_free_block = block.next_free_block
        # Link the next block to the previous block.
        block.next_free_block.prev_free_block = block.prev_free_block

        # Remove the block from the linked list.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L304 (append)
        """Put a block back into the free list and increase
        num_free_blocks by 1."""
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

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L327 (append_n)
        """Put a list of blocks back into the free list."""
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

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L352 (get_all_free_blocks)
        """Get all free blocks in the free list. Mainly used for testing."""
        ret = []
        if self.fake_free_list_head.next_free_block is None:
            raise RuntimeError(
                "next_free_block of fake_free_list_head should always exist"
            )
        # Start from the first block
        curr_block: KVCacheBlock = self.fake_free_list_head.next_free_block
        # As long as next_free_block is available, we haven't reached the tail.
        while curr_block.next_free_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret


# SOURCE: vllm/v1/core/kv_cache_utils.py:L373 (need_extra_keys)
def need_extra_keys(request: Request) -> bool:
    """Check whether the blocks allocated to this request need extra hash keys."""
    # Multimodal requests need to include the MM hash.
    # LoRA requests need to include the LoRA name.
    # Request with provided cache salt need to include the salt.
    return (
        bool(request.mm_features)
        or (request.lora_request is not None)
        or (request.cache_salt is not None)
    )


# SOURCE: vllm/v1/core/kv_cache_utils.py:L393 (_gen_mm_extra_hash_keys)
def _gen_mm_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[list[Any], int]:
    """Generate extra keys related to MultiModal request for block hash
    computation. The extra keys are (mm_hash, start_offset) that indicate a mm
    input contained in the block and its starting offset in the block tokens."""
    extra_keys: list[Any] = []

    mm_features = request.mm_features
    if not mm_features:
        return extra_keys, start_mm_idx

    # Note that we assume mm_features are sorted by mm_position.offset.
    last_pos = mm_features[-1].mm_position
    if last_pos.offset + last_pos.length <= start_token_idx:
        return extra_keys, start_mm_idx

    # Support start_mm_idx == -1 to indicate the last mm input.
    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_features)
        start_mm_idx = len(mm_features) + start_mm_idx

    curr_mm_idx = start_mm_idx
    while mm_features and curr_mm_idx < len(mm_features):
        mm_feature = mm_features[curr_mm_idx]
        assert mm_feature.identifier is not None
        offset = mm_feature.mm_position.offset
        length = mm_feature.mm_position.length
        if end_token_idx > offset:
            if start_token_idx >= offset + length:
                # This block has passed the current mm input.
                curr_mm_idx += 1
                continue

            # The block contains the current mm input. Include its offset
            # relative to the start of the block so prefix-cache keys stay
            # distinct when the same MM item appears at different positions.
            extra_keys.append((mm_feature.identifier, offset - start_token_idx))

            if end_token_idx >= offset + length:
                # This block contains the end of the current mm input; move on.
                curr_mm_idx += 1
            else:
                # Otherwise this block is done with mm inputs.
                break
        else:
            # This block has not reached the current mm input.
            break
    return extra_keys, curr_mm_idx


# SOURCE: vllm/v1/core/kv_cache_utils.py:L460 (_gen_lora_extra_hash_keys)
def _gen_lora_extra_hash_keys(request: Request) -> list[str]:
    """Return LoRA name of the request if it is a LoRA request, else []."""
    if not request.lora_request:
        return []
    return [request.lora_request.lora_name]


# SUBTRACTED: _gen_prompt_embeds_extra_hash_keys（L475-L498）—— 仅 prompt-embeds
# 输入触发；常规 token 请求恒返回 []。保留 mm/lora/cache_salt 三类已足以演示语义隔离。


# SOURCE: vllm/v1/core/kv_cache_utils.py:L501 (generate_block_hash_extra_keys)
def generate_block_hash_extra_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[tuple[Any, ...] | None, int]:
    """Generate extra keys for the block hash. The extra keys can come from
    the multi-modal inputs and request specific metadata (e.g., LoRA names)."""
    mm_extra_keys: list[Any]
    mm_extra_keys, new_start_mm_idx = _gen_mm_extra_hash_keys(
        request, start_token_idx, end_token_idx, start_mm_idx
    )
    lora_extra_keys: list[str] = _gen_lora_extra_hash_keys(request)
    cache_salt_keys: list[str] = (
        [request.cache_salt] if (start_token_idx == 0 and request.cache_salt) else []
    )
    # SUBTRACTED: prompt_embeds_keys = _gen_prompt_embeds_extra_hash_keys(...)
    # （L525-L527）—— 见上方函数删除说明，恒为 []。

    extra_keys: list[Any] = lora_extra_keys + mm_extra_keys + cache_salt_keys

    if not extra_keys:
        return None, new_start_mm_idx

    return tuple(extra_keys), new_start_mm_idx


# SOURCE: vllm/v1/core/kv_cache_utils.py:L539 (hash_block_tokens)
def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching."""
    if not parent_block_hash:
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )


# SOURCE: vllm/v1/core/kv_cache_utils.py:L635 (get_request_block_hasher)
def get_request_block_hasher(
    block_size: int,
    caching_hash_fn: Callable[[Any], bytes],
) -> Callable[[Request], list[BlockHash]]:
    """Returns a function which computes the list of un-computed block hashes
    of a request."""

    def request_block_hasher(request: Request) -> list[BlockHash]:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L643 (request_block_hasher 闭包)
        start_token_idx = len(request.block_hashes) * block_size
        num_tokens = request.num_tokens

        if start_token_idx + block_size > num_tokens:
            # Early stop when there no new full blocks created.
            return []

        curr_mm_idx = 0
        if start_token_idx > 0:
            # Set curr_mm_idx = -1 to indicate the last mm input. Since we reach
            # this branch only when the block is completed with generated tokens,
            # we only need to consider the last mm input.
            curr_mm_idx = -1

        prev_block_hash_value = (
            request.block_hashes[-1] if request.block_hashes else None
        )
        new_block_hashes: list[BlockHash] = []
        while True:
            end_token_idx = start_token_idx + block_size
            if end_token_idx > num_tokens:
                # We only hash full blocks
                break

            # MM and LoRA requests need extra keys for block-hash computation.
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start_token_idx, end_token_idx, curr_mm_idx
            )

            # Compute the hash of the current block
            block_tokens = request.all_token_ids[start_token_idx:end_token_idx]
            block_hash = hash_block_tokens(
                caching_hash_fn, prev_block_hash_value, block_tokens, extra_keys
            )

            new_block_hashes.append(block_hash)
            start_token_idx += block_size
            prev_block_hash_value = block_hash

        return new_block_hashes

    return request_block_hasher
