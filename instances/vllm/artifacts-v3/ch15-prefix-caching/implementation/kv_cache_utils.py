# SOURCE: vllm/v1/core/kv_cache_utils.py
# **哈希链族**（m1/m3/m12，本章第一主角）：BlockHash/BlockHashWithGroupId 的
# 键打包 → NONE_HASH 随机种子（PYTHONHASHSEED 可共享）→ hash_block_tokens
# （hash_i = H(parent, 本块 tokens, extra_keys)——Merkle 链）→ extra keys 谓词
# 与四源组装（mm/lora/cache_salt 仅首块；prompt_embeds 删）→ 请求侧增量
# hasher（只算新满 hash_block_size 块）→ BlockHashListWithBlockSize 惰性重串
# （链尾即前缀指纹）+ resolve_block_hashes（细粒度保留/粗粒度视图）。
# 池的元数据载体（KVCacheBlock 七字段含哈希两字段、FreeKVCacheBlockQueue
# 全原语）ch13 已建全量切面——本章哈希面直接复用同一套原语。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 1 条观测旁路：maybe_convert_block_hash（kv events 的 int 哈希兼容位）；
#   第 8 条 prompt_embeds：_gen_prompt_embeds_extra_hash_keys；
#   第 9 条调试辅助：KVCacheBlock.__repr__、FreeKVCacheBlockQueue.
#     get_all_free_blocks/iter_blocks_after；
#   定账/组化/张量布局族（_check_enough/estimate/get_kv_cache_configs 等
#     L751-L2242——ch14 全量切面已建；resolve_kv_cache_block_sizes 的
#     hash_block_size=GCD/prefix_match_unit 选择也归 ch14，本章只消费）；
#   第 3/4 条 eagle/DCP 乘子（本文件不涉）。
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, NewType, TypeAlias, overload

from .request import Request


# BlockHash represents the hash of a single KV-cache block used for
# prefix caching.  Treating it as a distinct type from `bytes` helps
# catch accidental misuse when passing around raw byte strings.
# SOURCE: vllm/v1/core/kv_cache_utils.py:L44 BlockHash
BlockHash = NewType("BlockHash", bytes)

# `BlockHashWithGroupId` combines a `BlockHash` with its KV cache group ID.
# It is represented as raw bytes for compactness and efficiency. The helper
# functions below pack/unpack the `BlockHash` and group id into/from the key.
# SOURCE: vllm/v1/core/kv_cache_utils.py:L49 BlockHashWithGroupId
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

# ExternalBlockHash is used for reproducible prefix-cache block hashing.
# It's a union of `bytes` and `int` to keep backward compatibility
# after we default block hashing to use sha256 bytes.
# SOURCE: vllm/v1/core/kv_cache_utils.py:L54 ExternalBlockHash
ExternalBlockHash: TypeAlias = bytes | int

# SUBTRACTED: maybe_convert_block_hash（L79-L82——kv events 的 int 块哈希
#   兼容位，第 1 条观测旁路）。


# SOURCE: vllm/v1/core/kv_cache_utils.py:L57 make_block_hash_with_group_id
def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int
) -> BlockHashWithGroupId:
    """Pack a `BlockHash` and group id into a `BlockHashWithGroupId`.

    The group id is encoded using 4 bytes in big-endian order and appended to
    the block hash bytes.  This representation avoids creating tuples while
    still allowing us to recover both components when needed.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L66
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))


# SOURCE: vllm/v1/core/kv_cache_utils.py:L69 get_block_hash
def get_block_hash(key: BlockHashWithGroupId) -> BlockHash:
    """Extract the `BlockHash` from a `BlockHashWithGroupId`."""
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L71
    return BlockHash(key[:-4])


# SOURCE: vllm/v1/core/kv_cache_utils.py:L74 get_group_id
def get_group_id(key: BlockHashWithGroupId) -> int:
    """Extract the group id from a `BlockHashWithGroupId`."""
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L76
    return int.from_bytes(key[-4:], "big", signed=False)


# The hash seed for the first block of any prefix block sequence.
#
# We use a random value to avoid hash collisions or PYTHONHASHSEED environment
# variable if set such that processes can share the seed if needed. This aligns
# with the behavior of Python's hash() function, which also uses a random seed
# if PYTHONHASHSEED is not set.
#
# The function `init_none_hash` initializes this variable globally.
# SOURCE: vllm/v1/core/kv_cache_utils.py:L95 NONE_HASH
NONE_HASH: BlockHash
# SUBTRACTED: _CBOR_HASH_FUNCTIONS 集合（L96——cbor 变体随 hashing.py 删）。


# SOURCE: vllm/v1/core/kv_cache_utils.py:L99 init_none_hash
def init_none_hash(hash_fn: Callable[[Any], bytes]):
    # SUBTRACTED: CBOR 无 PYTHONHASHSEED 的告警段（L102-L109——条件依赖已删
    #   的 cbor 变体集合，sha256 路恒不触发）
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L111-L114（未设 → 32 随机字节；
    #   设了 → hash_fn(seed) 派生——跨进程可共享同一前缀哈希空间）
    global NONE_HASH

    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is None:
        NONE_HASH = BlockHash(os.urandom(32))
    else:
        NONE_HASH = BlockHash(hash_fn(hash_seed))


# SOURCE: vllm/v1/core/kv_cache_utils.py:L117 KVCacheBlock
@dataclass(slots=True)
class KVCacheBlock:
    """KV-cache block metadata."""

    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L121-L130（七字段：block_id/
    #   ref_cnt/哈希两字段/prev/next/is_null）
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # The hash key (block hash + group id) of the block, only available
    # when the block is full and cached.
    _block_hash: BlockHashWithGroupId | None = None
    # Number of prefix tokens covered by _block_hash. For full blocks this is
    # the full block boundary; partial entries can end inside a cache block.
    _block_hash_num_tokens: int | None = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    # Whether the block is a null block that should never be cached.
    is_null: bool = False

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L140 block_hash property
    @property
    def block_hash(self) -> BlockHashWithGroupId | None:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L142
        return self._block_hash

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L144 block_hash_num_tokens
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
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L153-L157（带 num_tokens 的
        #   哈希落块原语——部分条目语义核心）
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

    # SUBTRACTED: __repr__（L164-L176——dossier.delete 第 9 条调试辅助）。


# SOURCE: vllm/v1/core/kv_cache_utils.py:L179 KVCacheBlockCopy
class KVCacheBlockCopy(NamedTuple):
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L180-L181（CoW 拷贝对的跨进程
    #   负载形态：只带块号对——真拷贝在 worker）
    src_block_id: int
    dst_block_id: int


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
    2. If two blocks have the same last accessed time (allocated by the
       same sequence), the one with more hash tokens (the tail of a block
       chain) is at the front.
    Note that we maintain this order by reversing the block order when free
    blocks of a request. This operation is outside of this class.

    Args:
        blocks: A list of KVCacheBlock objects.
    """

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L206 __init__
    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L207-L214
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
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L222-L234
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
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L242-L250
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
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L282-L297
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
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L312-L315
        if block.prev_free_block is None or block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError(f"remove() called on an invalid block: {block}")

        # Link the previous block to the next block.
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L317-L323
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
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L333-L345
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
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L351-L368（劈分不变量二的
        #   取头原语——无哈希块从这头进、先被取走驱逐）
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
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L376-L393（挂队尾原语——
        #   LRU 端，带哈希块从这头进、最可复用）
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

    # SUBTRACTED: get_all_free_blocks（L395-L413）/iter_blocks_after
    #   （L415-L427）——dossier.delete 第 9 条调试/测试遍历辅助。


# SOURCE: vllm/v1/core/kv_cache_utils.py:L430 need_extra_keys
def need_extra_keys(request: Request) -> bool:
    """Check whether the blocks allocated to this request need extra hash keys.

    Args:
        request (Request): The request.

    Returns:
        bool: Whether blocks allocated to this request need extra hash keys.
    """

    # Multimodal requests need to include the MM hash.
    # LoRA requests need to include the LoRA name.
    # Request with provided cache salt need to include the salt.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L443-L447
    return (
        bool(request.mm_features)
        or (request.lora_request is not None)
        or (request.cache_salt is not None)
    )


# SOURCE: vllm/v1/core/kv_cache_utils.py:L450 _gen_mm_extra_hash_keys
def _gen_mm_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[list[Any], int]:
    """Generate extra keys related to MultiModal request for block hash
    computation. For multi-modal inputs, the extra keys are
    (mm_hash, start_offset) that indicate a mm input contained in the block
    and its starting offset in the block tokens.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L467-L514
    extra_keys: list[Any] = []

    mm_features = request.mm_features
    if not mm_features:
        return extra_keys, start_mm_idx

    # Note that we assume mm_features are sorted by mm_position.offset.
    # We do not need to check all mm inputs if the start token index is out of
    # range. This usually happens in the late prefill phase and decoding phase.
    last_pos = mm_features[-1].mm_position
    if last_pos.offset + last_pos.length <= start_token_idx:
        return extra_keys, start_mm_idx

    # Support start_mm_idx == -1 to indicate the last mm input.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L480-L483
    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_features)
        start_mm_idx = len(mm_features) + start_mm_idx

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L485-L514
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
            # distinct when the same MM item appears at different positions
            # within otherwise-identical placeholder blocks.
            extra_keys.append((mm_feature.identifier, offset - start_token_idx))

            if end_token_idx >= offset + length:
                # If this block contains the end of the current mm input,
                # move to the next mm input as this block may also contain
                # the next mm input.
                curr_mm_idx += 1
            else:
                # Otherwise this block is done with mm inputs.
                break
        else:
            # This block has not reached the current mm input.
            break
    return extra_keys, curr_mm_idx


# SOURCE: vllm/v1/core/kv_cache_utils.py:L517 _gen_lora_extra_hash_keys
def _gen_lora_extra_hash_keys(request: Request) -> list[str]:
    """Generate extra keys related to LoRA for block hash computation.

    Args:
        request: The request object.

    Returns:
        Return LoRA name of the request if it is a LoRA request. Return empty
        list otherwise.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L527-L529
    if not request.lora_request:
        return []
    return [request.lora_request.lora_name]


# SUBTRACTED: _gen_prompt_embeds_extra_hash_keys（L532-L555——dossier.delete
#   第 8 条 prompt_embeds 专用；常规 token 请求不触发，mm/lora/cache_salt
#   三源已够讲语义隔离）。


# SOURCE: vllm/v1/core/kv_cache_utils.py:L558 generate_block_hash_extra_keys
def generate_block_hash_extra_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[tuple[Any, ...] | None, int]:
    """Generate extra keys for the block hash. The extra keys can come from
    the multi-modal inputs, request specific metadata (e.g., LoRA names), and
    hashed data from prompt embeddings.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L574-L593（四源并三源：
    #   prompt_embeds_keys 删——第 8 条）
    mm_extra_keys: list[Any]
    mm_extra_keys, new_start_mm_idx = _gen_mm_extra_hash_keys(
        request, start_token_idx, end_token_idx, start_mm_idx
    )
    lora_extra_keys: list[str] = _gen_lora_extra_hash_keys(request)
    # cache_salt 只拌进首块（start_token_idx == 0）——跨租户隔离靠它
    cache_salt_keys: list[str] = (
        [request.cache_salt] if (start_token_idx == 0 and request.cache_salt) else []
    )

    extra_keys: list[Any] = lora_extra_keys + mm_extra_keys + cache_salt_keys

    if not extra_keys:
        return None, new_start_mm_idx

    return tuple(extra_keys), new_start_mm_idx


# SOURCE: vllm/v1/core/kv_cache_utils.py:L596 hash_block_tokens
def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.
    Args:
        hash_function: The hash function used to compute block hash.
        parent_block_hash: The hash of the parent block. None
            if this is the first block.
        curr_block_token_ids: A list of token ids in the current
            block. The current block is assumed to be full.
        extra_keys: Extra keys for the block.
    Returns:
        The hash value of the block and the token ids in the block.
        The entire tuple is used as the hash key of the block.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L617-L623（链式本体：首块
    #   parent=NONE_HASH 种子；本块哈希 = H((parent, tokens, extra_keys))）
    if not parent_block_hash:
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )


# SUBTRACTED: resolve_kv_cache_block_sizes（L626-L688——scheduler/hash 两粒度
#   的选择（GCD 或 prefix_match_unit）归 ch14 全量切面；本章消费
#   hash_block_size 参数本身）。


# SOURCE: vllm/v1/core/kv_cache_utils.py:L691 get_request_block_hasher
def get_request_block_hasher(
    hash_block_size: int,
    caching_hash_fn: Callable[[Any], bytes],
) -> Callable[[Request], list[BlockHash]]:
    """
    Returns a function which computes the list of un-computed block hashes
    of a request.

    Hashes are computed at ``hash_block_size`` granularity and chained over the
    full prefix, so each hash uniquely fingerprints the prefix ending at its
    boundary. Coarser group block sizes and partial-cache boundaries reuse
    these hashes directly (see ``BlockHashListWithBlockSize``).
    """

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L705 request_block_hasher
    def request_block_hasher(request: Request) -> list[BlockHash]:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L706-L711（早停：没有新满
        #   块就返回空——哈希随 token 到达、增量算）
        start_token_idx = len(request.block_hashes) * hash_block_size
        num_tokens = request.num_tokens

        if start_token_idx + hash_block_size > num_tokens:
            # Early stop when there no new full blocks created.
            return []

        # SOURCE: vllm/v1/core/kv_cache_utils.py:L713-L719（start>0 时
        #   curr_mm_idx=-1 指最后一个 mm 输入）
        curr_mm_idx = 0
        if start_token_idx > 0:
            # Set curr_mm_idx = -1 to indicate the last mm input.
            # Note that since we reach to this branch only when the block is
            # completed with generated tokens, we only need to consider the
            # last mm input.
            curr_mm_idx = -1

        # SOURCE: vllm/v1/core/kv_cache_utils.py:L721-L723（prev = 上一块
        #   哈希——链式推进的衔接点）
        prev_block_hash_value = (
            request.block_hashes[-1] if request.block_hashes else None
        )
        new_block_hashes: list[BlockHash] = []
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L724-L746（只哈希满块、
        #   逐块问 extra_keys、链式推进）
        while True:
            end_token_idx = start_token_idx + hash_block_size
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
            start_token_idx += hash_block_size
            prev_block_hash_value = block_hash

        return new_block_hashes

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L748
    return request_block_hasher


# SUBTRACTED: 定账/组化/张量布局族（L751-L2242——_check_enough_kv_cache_
#   memory/estimate_max_model_len/get_kv_cache_configs 等全段，ch14 全量
#   切面已建；本章哈希面不触）。


# SOURCE: vllm/v1/core/kv_cache_utils.py:L2245 BlockHashListWithBlockSize
class BlockHashListWithBlockSize:
    """
    Convert block-hash granularity from `hash_block_size` to `target_block_size`.
    Used when KV cache groups have different block sizes: `hash_block_size`
    is the size used to compute the original `block_hashes`; `target_block_size`
    is the group's actual block size.

    Currently, only scaling up by an integer factor is supported (i.e.,
    `target_block_size` is a multiple of `hash_block_size`). Conversion is
    performed lazily on access for efficiency. Each `hash_block_size` hash is
    already chained over its entire prefix, so the hash at the last
    `hash_block_size` boundary of a `target_block_size` block uniquely
    fingerprints that block's prefix; we use it directly.

    Example (`hash_block_size` = 16, `target_block_size` = 32):
    the second 16-size hash already covers tokens 0-31, so it is the 32-size
    hash:

    Block hashes with block_size 16:
    | Token Range | 0-15 | 16-31 | 32-47 | 48-63 |
    |-------------|------|-------|-------|-------|
    | Hash        | A    | B     | C     | D     |

    Block hashes with block_size 32:
    | Token Range | 0-31 | 32-63 |
    |-------------|------|-------|
    | Hash        | B    | D     |

    Args:
        block_hashes: Block hashes to convert, computed at `hash_block_size`.
        hash_block_size: Block size at which `block_hashes` were computed.
        target_block_size: Desired block size; must be a multiple of `hash_block_size`.
    """

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2279 __init__
    def __init__(
        self,
        block_hashes: list[BlockHash],
        hash_block_size: int,
        target_block_size: int,
    ):
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L2285-L2287
        self.block_hashes = block_hashes
        assert target_block_size % hash_block_size == 0
        self.scale_factor = target_block_size // hash_block_size

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2289 __len__
    def __len__(self) -> int:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L2290
        return len(self.block_hashes) // self.scale_factor

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2292-L2296 overload 桩
    @overload
    def __getitem__(self, idx: int) -> BlockHash: ...  # SOURCE: ...:L2293

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2295-L2296
    @overload
    def __getitem__(self, idx: slice) -> list[BlockHash]: ...  # SOURCE: ...:L2296

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2298 __getitem__
    def __getitem__(self, idx):
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L2299-L2306
        if isinstance(idx, int):
            return self._get_value_at(idx)

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self._get_value_at(i) for i in range(start, stop, step)]

        raise TypeError(f"Invalid index type: {type(idx)!r}")

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2308 __iter__
    def __iter__(self) -> Iterator[BlockHash]:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L2309-L2310
        for i in range(len(self)):
            yield self._get_value_at(i)

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2312 _get_value_at
    def _get_value_at(self, idx: int) -> BlockHash:
        # The last hash_block_size hash within the target block already chains
        # over the whole prefix, so it is the target block's hash.
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L2315（链尾即前缀指纹——
        #   粗块哈希重串零成本的秘密）
        return self.block_hashes[(idx + 1) * self.scale_factor - 1]


# SOURCE: vllm/v1/core/kv_cache_utils.py:L2318 BlockHashList
BlockHashList = list[BlockHash] | BlockHashListWithBlockSize


# SOURCE: vllm/v1/core/kv_cache_utils.py:L2321 resolve_block_hashes
def resolve_block_hashes(
    block_hashes: BlockHashList,
    hash_block_size: int,
    block_size: int,
    *,
    supports_fine_grained_hash_lookup: bool = False,
    alignment_tokens: int | None = None,
) -> BlockHashList:
    """Resolve the block-hash view at ``block_size``.

    When ``block_size`` equals ``hash_block_size``, reuse the precomputed block
    hashes directly; otherwise view them at ``block_size`` granularity.
    Fine-grained lookup keeps the original hashes for partial cache hits.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2335-L2336（等粒度直接复用）
    if block_size == hash_block_size:
        return block_hashes
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2337-L2340（已是块尺寸视图）
    if isinstance(block_hashes, BlockHashListWithBlockSize):
        # Already a block-size view
        assert block_hashes.scale_factor == block_size // hash_block_size
        return block_hashes
    # Fine-grained partial hits keep the raw hashes. The caller passes
    # alignment_tokens = hash_block_size to enable them, else >= block_size.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2341-L2349（细粒度查找保留
    #   原始哈希列表供块内探测）
    if (
        supports_fine_grained_hash_lookup
        and alignment_tokens is not None
        and alignment_tokens < block_size
        and block_size % alignment_tokens == 0
    ):
        return block_hashes
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2350-L2351（否则包成粗视图）
    assert block_size % hash_block_size == 0
    return BlockHashListWithBlockSize(block_hashes, hash_block_size, block_size)
