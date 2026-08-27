# SOURCE: vllm/v1/core/kv_cache_utils.py
# 账本核心（m1-m9/m15）：KVCacheBlock/FreeKVCacheBlockQueue（池的元数据
# 载体，ch13 全文已建——本章组化/回收复用同一套原语）+ 定账总控
# get_kv_cache_configs（合并 spec → 分组 → override 折算 → auto-fit →
# 逐 worker 护栏 → 出 config → PP 取最小）+ 护栏四道（_check_enough /
# estimate_max_model_len 二分 / _auto_fit / override 折算）+ 组化
# （get_kv_cache_groups / unify_kv_cache_spec_page_size / 等量化组）+
# 张量布局（get_kv_cache_config_from_groups）+ 对齐粒度
# （resolve_kv_cache_block_sizes）+ 一份账喂两侧（generate_scheduler_
# kv_cache_config / get_kv_cache_capacity）。
# LOGGER SEAM：vllm.logger.init_logger → stdlib logging（同构的
#   warning/info 账目；观测面归 dossier.delete 第 9/11 条）。
# SUBTRACTED（dossier.delete 批准项的落点）：
#   第 4 条 DSV4/SlidingWindowMLA 特路：group_and_unify_kv_cache_specs/
#     _get_kv_cache_groups_uniform_groups/_approximate_gcd/_annotate_eagle_
#     groups_deepseek_v4 + packed 布局三件（_get_packed_kv_cache_layout/
#     _use_packed_kv_cache_config/_get_kv_cache_config_packed，L1283-L1358、
#     L1592-L1754）+ _max_memory_usage_bytes_from_groups 的全 UniformType
#     DSV4 分支（L1913-L1937）；
#   第 5 条 HiddenState 分组抽离段（get_kv_cache_groups L1821-L1830、
#     L1843-L1850）与 R-SWA/MLA/Cross/SinkFull 的 promote 分支；
#   第 6 条 eagle：_project_kv_cache_groups_to_worker 的 is_eagle_group 透传；
#   第 8 条 DCP/PCP 乘子（单卡恒 1 烘干——resolve/multi 处 ×dcp 保留乘位、
#     值恒 1）；
#   第 9 条 metrics/events 贯穿调用；
#   哈希链族（BlockHash/hash_block_tokens/get_request_block_hasher/
#     BlockHashListWithBlockSize L460-L623、L691-L748——→ ch15；
#     KVCacheBlock 的哈希两字段与 set/reset_hash 保留作账位）。
import copy
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Callable, Iterable

from .kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from .math_utils import cdiv
from .mem_utils import format_gib

if TYPE_CHECKING:
    from .config import VllmConfig

logger = logging.getLogger(__name__)  # LOGGER SEAM：init_logger 同构账位


# --------------------------------------------------------------------------- #
# 池的元数据载体（ch13 全文已建；本章组化/回收复用同一套原语）
# --------------------------------------------------------------------------- #


# SOURCE: vllm/v1/core/kv_cache_utils.py:L117 KVCacheBlock
@dataclass(slots=True)
class KVCacheBlock:
    """KV-cache block metadata."""

    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L121-L130（七字段：block_id/
    #   ref_cnt/哈希两字段/prev/next/is_null——哈希两字段是 ch15 账位）
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # The hash key (block hash + group id) of the block, only available
    # when the block is full and cached.
    _block_hash: object | None = None
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
    def block_hash(self):
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
        block_hash,
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

    # SUBTRACTED: __repr__（L164-L176——观测面）。


# SUBTRACTED: KVCacheBlockCopy（L179-L181——CoW 拷贝管线 → ch15）；
#   BlockHash/BlockHashWithGroupId/hash 族（L60-L115——ch15 哈希链）。


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

        # SOURCE: vllm/v1/core/kv_cache_utils.py:L347
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
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L402-L413
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


# SUBTRACTED: 哈希链族 L460-L623（hash_block_tokens/BlockHash 生成 → ch15）。
#   注意「链尾即前缀指纹」的粗块哈希重串（BlockHashListWithBlockSize）也
#   在其列——m8 的 hash_block_size 只留粒度判定，哈希构造 → ch15。


# --------------------------------------------------------------------------- #
# m8 两个对齐粒度
# --------------------------------------------------------------------------- #


# SOURCE: vllm/v1/core/kv_cache_utils.py:L626 resolve_kv_cache_block_sizes
def resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: "VllmConfig",
) -> tuple[int, int]:
    """Resolve (scheduler_block_size, hash_block_size).

    - ``scheduler_block_size`` is the token-alignment invariant used by the
      scheduler (e.g. for ``num_computed_tokens`` rounding). Single group:
      ``cache_config.block_size * dcp``. Multiple groups: LCM of every
      group's effective block size. Attention groups are scaled by DCP;
      Mamba groups keep their full per-rank state and are not scaled.
    - ``hash_block_size`` is the granularity at which ``Request.block_hashes``
      is computed. Single group: equals scheduler block size. Multiple groups:
      ``cache_config.prefix_match_unit`` override if set, else the GCD of
      group block sizes; every group's block size must be divisible by it.
      Returns the scheduler block size (i.e. disables finer hashing) if block
      hashing is inactive or a mamba group's block size diverges from the
      cache block size (mamba_cache_mode != "align").
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L645-L647（dcp 乘位保留、值恒 1）
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L649-L651
    if len(groups) <= 1:
        bs = cache_config.block_size * dcp
        return bs, bs

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L653-L659（mamba 组不乘 dcp）
    group_block_sizes = [
        g.kv_cache_spec.block_size * dcp
        if isinstance(g.kv_cache_spec, AttentionSpec)
        else g.kv_cache_spec.block_size
        for g in groups
    ]
    scheduler_block_size = math.lcm(*group_block_sizes)

    # Block hashes are only consumed by prefix caching and KV connectors
    # (P/D, offloading); when neither is active, keep hash_block_size equal
    # to the scheduler block size.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L661-L666
    connector_enabled = vllm_config.kv_transfer_config is not None
    if not (cache_config.enable_prefix_caching or connector_enabled):
        return scheduler_block_size, scheduler_block_size

    # Mamba groups with block_size != cache_config.block_size
    # (mamba_cache_mode != "align") break divisibility; back off to the
    # scheduler block size.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L668-L676
    if any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.block_size != cache_config.block_size
        for g in groups
    ):
        return scheduler_block_size, scheduler_block_size

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L678-L687
    requested = cache_config.prefix_match_unit
    hash_block_size = (
        requested if requested is not None else math.gcd(*group_block_sizes)
    )
    if any(bs % hash_block_size != 0 for bs in group_block_sizes):
        raise ValueError(
            f"Invalid prefix_match_unit={hash_block_size}; all KV cache group "
            f"block sizes must be divisible by prefix_match_unit. "
            f"Got group block sizes={group_block_sizes}."
        )
    return scheduler_block_size, hash_block_size


# SUBTRACTED: get_request_block_hasher（L691-L748——哈希构造 → ch15）。


# --------------------------------------------------------------------------- #
# m4 护栏四道
# --------------------------------------------------------------------------- #


# SOURCE: vllm/v1/core/kv_cache_utils.py:L751 _check_enough_kv_cache_memory
def _check_enough_kv_cache_memory(
    available_memory: int,
    get_needed_memory: Callable[[], int],
    max_model_len: int,
    estimate_max_model_len: Callable[[int], int],
):
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L757-L765（available ≤ 0 → raise）
    if available_memory <= 0:
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when initializing the engine "
            "(this flag also controls CPU memory reservation on the CPU "
            "backend, despite its name). "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more details."
        )

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L767
    needed_memory = get_needed_memory()

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L769-L788（不够 → 二分估可行
    #   长度写进报错，引导调 util 或降 max_model_len）
    if needed_memory > available_memory:
        estimated_max_len = estimate_max_model_len(available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}. "
            )

        raise ValueError(
            f"To serve at least one request with the model's max seq len "
            f"({max_model_len}), ({format_gib(needed_memory)} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({format_gib(available_memory)} GiB). {estimated_msg}"
            f"Try increasing `gpu_memory_utilization` (which also controls "
            f"CPU memory on the CPU backend) or decreasing `max_model_len` "
            f"when initializing the engine. "
            f"See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            f"for more details."
        )


# SOURCE: vllm/v1/core/kv_cache_utils.py:L791 max_memory_usage_bytes
def max_memory_usage_bytes(
    vllm_config: "VllmConfig", kv_cache_specs: Iterable[KVCacheSpec]
) -> int:
    """
    Get the maximum memory usage in bytes for the given KV cache specs.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L797
    return sum(spec.max_memory_usage_bytes(vllm_config) for spec in kv_cache_specs)


# SOURCE: vllm/v1/core/kv_cache_utils.py:L800 estimate_max_model_len
def estimate_max_model_len(
    vllm_config: "VllmConfig",
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
) -> int:
    """
    Estimates the maximum model length that can fit in the available memory
    using binary search.

    This function temporarily modifies max_model_len during estimation but
    restores the original value before returning, ensuring no side effects.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The estimated maximum model length that fits in the available memory.
    """
    # Save the original max_model_len to restore after estimation
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L820-L821
    original_max_model_len = vllm_config.model_config.max_model_len

    # Define a function to check if a given model length fits in memory
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L823-L829 fits_in_memory
    def fits_in_memory(model_len: int) -> bool:
        # Temporarily modify the max_model_len for this calculation
        vllm_config.model_config.max_model_len = model_len
        # Calculate memory needed for the given model length
        memory_needed = max_memory_usage_bytes(vllm_config, kv_cache_spec.values())
        return memory_needed <= available_memory

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L831-L848（单调不减 → 标准
    #   upper-bound 二分 O(log L)）
    try:
        # Binary search for the maximum model length
        left, right = 1, original_max_model_len

        # If even the smallest model length doesn't fit, return 0
        if not fits_in_memory(left):
            return 0

        # Binary search for the maximum model length that fits
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits_in_memory(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        # Always restore the original max_model_len to avoid side effects
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L850-L851
        vllm_config.model_config.max_model_len = original_max_model_len


# SOURCE: vllm/v1/core/kv_cache_utils.py:L854 check_enough_kv_cache_memory
def check_enough_kv_cache_memory(
    vllm_config: "VllmConfig",
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
):
    """
    Checks whether `available_memory` is enough for the KV cache to hold at
    least one request with the model's max_model_len.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Raises:
        ValueError: If there is not enough memory available for the KV cache.
    """

    # No need to check for available memory if the kv_cache_spec is empty
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L872-L879
    if kv_cache_spec:
        _check_enough_kv_cache_memory(
            available_memory,
            lambda: max_memory_usage_bytes(vllm_config, kv_cache_spec.values()),
            vllm_config.model_config.max_model_len,
            lambda am: estimate_max_model_len(vllm_config, kv_cache_spec, am),
        )


# --------------------------------------------------------------------------- #
# m5 组化
# --------------------------------------------------------------------------- #


# SOURCE: vllm/v1/core/kv_cache_utils.py:L882 create_kv_cache_group_specs
def create_kv_cache_group_specs(
    kv_cache_spec: dict[str, KVCacheSpec], grouped_layer_names: list[list[str]]
) -> list[KVCacheGroupSpec]:
    """
    Create KVCacheGroupSpec object for each kv cache group layer.
    The layers in the same group should share the same
    KVCacheSpec.

    Args:
        kv_cache_spec:
            A mapping from each layer name to its corresponding KVCacheSpec.
        grouped_layer_names:
            A list of kv cache groups, where each element is a list of layer
            names that belong to the same group and should share the same
            KVCacheSpec.
    Returns:
        A list of KVCacheGroupSpec objects, one for each group.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L900-L909（merge 同组 spec——
    #   Full 版 merge 还会合窗口尺寸并断言同型）
    kv_cache_groups = []
    for layer_names_one_group in grouped_layer_names:
        layer_specs = [
            kv_cache_spec[layer_name] for layer_name in layer_names_one_group
        ]
        merged_layer_spec = layer_specs[0].merge(layer_specs)
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names_one_group, merged_layer_spec)
        )
    return kv_cache_groups


# SOURCE: vllm/v1/core/kv_cache_utils.py:L912 is_kv_cache_spec_uniform
def is_kv_cache_spec_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same KV cache spec.
    Note that we regard FullAttentionSpec with and without sliding window as
    the same type.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        True if all layers have the same type, False otherwise.
    """

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L925-L934
    if not kv_cache_spec:
        # Encoder-only models do not have KV cache, kv_cache_type can be
        # regarded as uniform.
        return True
    try:
        kv_cache_spec_values = list(kv_cache_spec.values())
        _ = kv_cache_spec_values[0].merge(kv_cache_spec_values)
    except AssertionError:
        return False
    return True


# --------------------------------------------------------------------------- #
# m15 容量核算
# --------------------------------------------------------------------------- #


# SOURCE: vllm/v1/core/kv_cache_utils.py:L937 get_max_concurrency_for_kv_cache_config
def get_max_concurrency_for_kv_cache_config(
    vllm_config: "VllmConfig", kv_cache_config: KVCacheConfig
) -> float:
    """
    Get the maximum concurrency for the given KV cache configuration.

    A request at max_model_len consumes whole blocks from each group's block
    table — cdiv(per-request bytes, page bytes) of the group's spec — and all
    groups draw those block ids from one shared pool, so the per-request
    total is the sum over groups. The memory/page ratio is identical whether
    a group carries an aggregated UniformTypeKVCacheSpecs (worker config) or
    a representative per-layer spec (scheduler config), so both capacity
    call sites agree.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L951-L959（混合布局按组求和）
    num_blocks_per_request = sum(
        cdiv(
            group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
            group.kv_cache_spec.page_size_bytes,
        )
        for group in kv_cache_config.kv_cache_groups
    )
    max_concurrency = kv_cache_config.num_blocks / num_blocks_per_request
    return max_concurrency


# SOURCE: vllm/v1/core/kv_cache_utils.py:L962 may_override_num_blocks
def may_override_num_blocks(vllm_config: "VllmConfig", num_blocks: int) -> int:
    """
    Override the number of kv cache blocks if `num_gpu_blocks_override` is set.
    The override is logged once, at the call site in `get_kv_cache_configs`.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L967-L969
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_blocks = vllm_config.cache_config.num_gpu_blocks_override
    return num_blocks


# SOURCE: vllm/v1/core/kv_cache_utils.py:L972 _pool_bytes_per_block
def _pool_bytes_per_block(
    vllm_config: "VllmConfig", kv_cache_groups: list[KVCacheGroupSpec]
) -> int:
    """
    Bytes consumed by one block in the worker's shared KV cache pool, mirroring
    the divisor used by `get_kv_cache_config_from_groups` to convert
    `available_memory` into `num_blocks`. Used to compute the effective KV cache
    capacity once `num_gpu_blocks_override` is applied.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L981-L984（单组异宽：聚合页）
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        return kv_cache_groups[0].kv_cache_spec.page_size_bytes
    # SUBTRACTED: packed 分支（L985-L987——第 4 条）
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L988-L990（通用：page × group_size）
    group_size = max(len(g.layer_names) for g in kv_cache_groups)
    page_size = get_uniform_page_size([g.kv_cache_spec for g in kv_cache_groups])
    return page_size * group_size


# --------------------------------------------------------------------------- #
# m1 定块数总算术
# --------------------------------------------------------------------------- #


# SOURCE: vllm/v1/core/kv_cache_utils.py:L993 get_num_blocks
def get_num_blocks(
    vllm_config: "VllmConfig",
    num_layers: int,
    available_memory: int,
    page_size: int,
) -> int:
    """
    Get the number of kv cache blocks.

    Args:
        vllm_config: The global VllmConfig
        num_layers: The number of layers
        available_memory: Memory available for KV cache in bytes.
        page_size: The page size of the KV cache.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1008-L1010（available // page
    #   // num_layers（多组时 = group_size）；max 防负；override 凌驾）
    num_blocks = int(available_memory // page_size // num_layers)
    num_blocks = max(num_blocks, 0)
    return may_override_num_blocks(vllm_config, num_blocks)


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1013 get_uniform_page_size
def get_uniform_page_size(kv_cache_specs: Iterable[KVCacheSpec]) -> int:
    """
    Get the page size of the KV cache.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1017-L1019（多组共享的前提：
    #   全组页大小相等，不等即断言炸）
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    assert len(page_sizes) == 1
    return page_sizes.pop()


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1022 _get_kv_cache_groups_uniform_spec
def _get_kv_cache_groups_uniform_spec(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with the same KV cache
    spec for all layers.

    Args:
        kv_cache_specs: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroupSpecs
    """

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1036
    return create_kv_cache_group_specs(kv_cache_specs, [list(kv_cache_specs.keys())])


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1039 _get_kv_cache_groups_uniform_type
def _get_kv_cache_groups_uniform_type(
    spec: UniformTypeKVCacheSpecs,
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with one type of KV cache
    but different hidden sizes. All layers are merged into one group.

    Args:
        spec: The UniformTypeKVCacheSpecs of the model

    Returns:
        The generated KVCacheGroupSpecs
    """

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1053
    return [KVCacheGroupSpec(list(spec.kv_cache_specs.keys()), spec)]


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1056 is_kv_cache_page_size_uniform
def is_kv_cache_page_size_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same page size.
    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        True if all layers have the same page size, False otherwise.
    """

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1066-L1067
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    return len(page_sizes) == 1


# --------------------------------------------------------------------------- #
# m6 页大小统一
# --------------------------------------------------------------------------- #


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1070 unify_kv_cache_spec_page_size
def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """
    Unify the page size of the given KVCacheSpec. If the page size of all layers
    are the same, return the original KVCacheSpec. If not same, unify the page
    size by increasing the block size of layers with smaller page size. Two
    cases cannot be unified by block size alone and pad their physical page to
    the maximum instead: Mamba layers, whose page size comes from state shapes
    and is independent of block size; and attention layers whose page does not
    evenly divide the maximum and whose backend opts in via
    ``AttentionSpec.indexes_kv_by_block_stride`` (the padded page is read through
    a strided view, which not every backend handles). Raise NotImplementedError
    if failed to unify the page size.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        The updated KVCacheSpec with the same page_size_bytes.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1091-L1094
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # All layers have the same page size, no need to unify.
        return kv_cache_spec

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1096-L1110（以最大页为基准：
    #   Mamba 页由状态形状决定不随块缩放、改为物理 pad）
    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        elif isinstance(layer_spec, MambaSpec):
            new_spec: KVCacheSpec = replace(layer_spec, page_size_padded=max_page_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
        else:
            # SOURCE: vllm/v1/core/kv_cache_utils.py:L1112-L1131（普通注意力
            #   层：能整除则调大 block_size / 否则 stride 索引才可 pad /
            #   都不行 raise）
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size == 0:
                ratio = max_page_size // layer_page_size
                new_block_size = layer_spec.block_size * ratio
                new_spec = replace(layer_spec, block_size=new_block_size)
            elif (
                isinstance(layer_spec, AttentionSpec)
                and layer_spec.indexes_kv_by_block_stride
            ):
                new_spec = replace(layer_spec, page_size_padded=max_page_size)
            else:
                raise NotImplementedError(
                    f"Layer {layer_name}: page size is not divisible by the "
                    "maximum page size and cannot be padded. Padding is only "
                    "supported for attention layers whose backend indexes KV "
                    "pages by the block stride (indexes_kv_by_block_stride is "
                    "True)."
                )
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1135 is_kv_cache_type_attention_free
def is_kv_cache_type_attention_free(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    # kv_cache_spec is an empty dict for attention free models
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1137
    return not kv_cache_spec


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1140 _get_kv_cache_groups_uniform_page_size
def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache groups for hybrid models with multiple
    attention types but still with a uniform page size (physical memory per
    block per layer) for all layers.

    Detailed explanation about kv cache management of hybrid models:
    The layers in the models are repeated with some patterns, e.g., a model
    with 10 full attention layers and 20 sliding window attention layers can be
    regarded as repeating the pattern (1 * full, 2 * sw) 10 times.
    The KVCacheManager allocates different block tables for each of the 3 layers
    in the pattern, and repeats each of them 10 times to generate the
    block_table for the 30 layers in the model.
    Therefore, we can group the layers in the model into 3 kv_cache_groups, each
    of which contains 10 layers in the model.
    The KVCacheManager allocates the block_table for each group based on its
    kv_cache spec, and the model runner applies the block table to each layer
    in the group.
    For example:
    1. A model only uses full attention. The pattern is
    (num_hidden_layers * full), so there is only one group and the block table
    is shared by all layers. It is already handled by
    `_get_kv_cache_config_uniform_type`.
    2. A model with 10 full attention layers and 20 sliding window
    attention layers. There are 3 layers in the pattern (1 * full, 2 * sw), so
    there are 3 kv_cache_groups, each of which represents 10 layers.

    To simplify the implementation, we make the following assumptions:
    1. Physical memory per block: Must be the same across all KV cache groups.
    Breaking this assumption is non-trivial due to memory fragmentation concerns
    when allocating blocks of different sizes.
    2. Tokens per block (block_size): Currently, we directly use
    `CacheConfig.block_size` for all layers. It can be extended to vary by KV
    cache group, but within each KV cache group, all layers must share the same
    block size.
    3. Physical memory per token per layer: This property is decided by model
    config. Currently we only support models that have the same physical memory
    per token per layer for all layers. Can be relaxed with a simple extension,
    but still need to keep physical memory per block the same for all groups.
    4. Number of layers per group: Currently assumed the same for all layers.
    Can be relaxed with a simple extension, but still need to keep physical
    memory per block the same for all groups.
    5. Attention type within groups: All layers in a group must share the same
    attention type. One exception is that, when
    `--disable-hybrid-kv-cache-manager` is true, the single group for full
    attention layers may also include attention layers using sliding window or
    LLaMA 4 local attention. See `unify_hybrid_kv_cache_specs` for more details.
    6. Support for multiple attention types: The design for most components is
    general to an arbitrary number of attention types. But
    `find_longest_cache_hit` only supports one attention type or two
    types of full-attention plus exactly one another type. The general
    implementation of this function is feasible but we don't know how to
    implement it cleanly yet.

    As we assume tokens per block, physical memory per token per layer, and
    number of layers per group are the same now, we can ensure that physical
    memory per block is the same for all groups.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model
    Returns:
        The generated KVCacheGroupSpecs
    """
    # Group all layers by kv_cache_spec.
    # E.g., 2 full attention layers and 3 sliding window attention layers,
    # -> (full.0, full.1), (sw.0, sw.1, sw.2).
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1205-L1210（按 spec 值分桶）
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec].append(layer_name)

    # Attempt to further merge same-type layers based on whether their KV
    # cache specs can be merged, to minimize the group count. This benefits
    # situations where specs share a block layout and differ only in a
    # property it can reconcile (e.g. full attention layers differing only in
    # sliding window / attention chunk size).
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1217-L1231（可 merge 的同型层
    #   合桶——merge raise 即不兼容）
    layer_buckets: list[list[str]] = []
    spec_buckets: list[list[KVCacheSpec]] = []
    for layer_spec, layer_names in same_type_layers.items():
        for names, specs in zip(layer_buckets, spec_buckets):
            try:
                # A raise means that the specs are incompatible.
                type(specs[0]).merge([*specs, layer_spec])
            except (AssertionError, ValueError):
                continue
            names.extend(layer_names)
            specs.append(layer_spec)
            break
        else:
            layer_buckets.append(list(layer_names))
            spec_buckets.append([layer_spec])

    # Split each group into smaller groups, to make the number of layers in each
    # group identical. Add padding to the last group of each type if necessary.
    # E.g., (full.0, full.1), (sw.0, sw.1, sw.2)
    # split to 3 groups with 2 layers each:
    # (full.0, full.1), (sw.0, sw.2), (sw.1, padding).
    # FIXME(Chen): At the moment of writing this code (2025-06-02), all
    # open-source hybrid model follows a n:1 pattern between different attention
    # types (e.g., Gemma3 5:1 between sw and full, LLaMA4 3:1 between local and
    # full), so we can use the "1" in the n:1 pattern as the group size, which
    # is the minimum number of layers among all attention types. Need a better
    # strategy if we want to support more complex patterns (e.g., 20 full + 30
    # sw, where the group size should be 10).
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1245-L1256（1.5 启发式：12 SW +
    #   13 full 补成 13/13 而非 12/24）
    min_num_layers = min([len(layers) for layers in layer_buckets])
    group_size = min_num_layers
    max_num_layers = max([len(layers) for layers in layer_buckets])
    if max_num_layers < min_num_layers * 1.5:
        # If the number of layers is not much larger than the minimum number of
        # layers, use the maximum number of layers as the group size to avoid
        # too many padding layers. A typical example is gpt-oss-20b + eagle,
        # with 12 sw + 13 full. We pad it to (13 sw, 13 full) instead of
        # (12 sw, 24 full). 1.5 is a heuristic to avoid too many padding
        # layers while accommodating speculative decoding drafters that add
        # extra layers to one attention type.
        group_size = max_num_layers
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1257-L1279（padding warning +
    #   PP layers[i::num_groups] 交错分派——某 stage 不出空组）
    grouped_layers = []
    for layers in layer_buckets:
        num_padding_layers = group_size - len(layers) % group_size
        if num_padding_layers != group_size:
            logger.warning(
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",  # noqa
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)
        # In PP case, say if we have
        # - stage 0: full.0, sw.0, sw.1
        # - stage 1: full.1, sw.2, sw.3
        # We should have 3 groups: (full.0, full.1), (sw.0, sw.2), (sw.1, sw.3)
        # It can't be (full.0, full.1), (sw.0, sw.1), (sw.2, sw.3) because
        # the 3 groups in stage 0 will be (full.0), (sw.0, sw.1), (empty group)
        # and it will be padded to (full.0, padding), (sw.0, sw.1),
        # (padding, padding) to ensure the number of layers in each group is
        # the same and will cause memory waste.
        # To avoid this, we assign layers[i::num_groups] to the i-th group
        # instead of layers[i * group_size: (i + 1) * group_size]
        for i in range(num_groups):
            grouped_layers.append(layers[i::num_groups])
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1280
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)


# SUBTRACTED: packed 布局三件（L1283-L1358——dossier.delete 第 4 条
#   DSV4/cross-layers 特路的重叠别名布局；正文以 why 注一句话点名）。


# --------------------------------------------------------------------------- #
# m7 张量布局
# --------------------------------------------------------------------------- #


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1361 get_kv_cache_config_from_groups
def get_kv_cache_config_from_groups(
    vllm_config: "VllmConfig",
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """
    Generate the KV cache configuration from the KV cache groups and spec
    of each layer.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_groups: The KV cache groups
        available_memory: Memory available for KV cache in bytes
    Returns:
        The generated KVCacheConfig
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1377-L1384（attention-free：
    #   num_blocks=1——BlockPool 永远需要 null_block）
    if len(kv_cache_groups) == 0:
        # Attention free models do not have KV cache.
        # Return num_blocks=1 as BlockPool always needs a null_block.
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        )

    # Determine how model runners should initialize the KV cache tensors.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1387-L1404（单组异宽：每层
    #   一张张量、按各自页大小分账）
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # Special case: all layers have the same type of KV cache but with
        # different hidden sizes. Allocate different amount of memory for each
        # layer based on its hidden size.
        num_blocks = (
            available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
        )
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        kv_cache_tensors = [
            KVCacheTensor(
                size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                shared_by=[layer_name],
            )
            for layer_name in kv_cache_groups[0].layer_names
        ]
    # SUBTRACTED: packed 分支（L1405-L1410——第 4 条 DeepSeek V4 默认 /
    #   --enable-cross-layers 的重叠布局；正文以 why 注点名）
    else:
        # General case:
        # We will have group_size memory pools, each is shared by one layer from
        # each group. As layers of different groups have different block table,
        # they will use different parts of the shared Tensor.
        # The memory layout for 3 groups (full.0, full.1), (sw.0, sw.2),
        # (sw.1, padding) will be: (group_size = 2)
        # full.0, sw.0, sw.1: share a Tensor with size=available_memory//2
        # full.1, sw.2: share another Tensor with size=available_memory//2
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L1411-L1437（group_size 池、
        #   每池由每组各出一层共享——一个 block_id 同一时刻只归一个组用）
        group_size = max(len(group.layer_names) for group in kv_cache_groups)

        page_size = get_uniform_page_size(
            [group.kv_cache_spec for group in kv_cache_groups]
        )
        assert group_size > 0, "group_size must be greater than 0"
        num_blocks = get_num_blocks(
            vllm_config, group_size, available_memory, page_size
        )
        kv_cache_tensors = []
        for i in range(group_size):
            shared_by = []
            for j in range(len(kv_cache_groups)):
                if i < len(kv_cache_groups[j].layer_names):
                    shared_by.append(kv_cache_groups[j].layer_names[i])
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            )

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1439-L1443
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


# --------------------------------------------------------------------------- #
# m5 disable 回退
# --------------------------------------------------------------------------- #


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1446 _promote_local_kv_cache_specs
def _promote_local_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """Use full-attention allocation for local-attention cache specs.

    The returned specs affect KV cache management only. Attention modules keep
    their original sliding-window or chunked-local compute behavior.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1454-L1459
    promoted_specs = kv_cache_spec.copy()

    if is_kv_cache_spec_uniform(
        promoted_specs
    ) or UniformTypeKVCacheSpecs.is_uniform_type(promoted_specs):
        return promoted_specs

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1461-L1477
    has_full_attention = any(
        isinstance(spec, FullAttentionSpec) for spec in promoted_specs.values()
    )
    has_sliding_window = any(
        isinstance(spec, SlidingWindowSpec) for spec in promoted_specs.values()
    )
    has_chunked_local_attention = any(
        isinstance(spec, ChunkedLocalAttentionSpec) for spec in promoted_specs.values()
    )
    full_block_sizes = {
        spec.block_size
        for spec in promoted_specs.values()
        if isinstance(spec, FullAttentionSpec)
    }
    full_attention_block_size = (
        next(iter(full_block_sizes)) if len(full_block_sizes) == 1 else None
    )

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1479-L1485 promoted_page_size_padded
    def promoted_page_size_padded(spec: AttentionSpec, block_size: int) -> int | None:
        if spec.page_size_padded is None:
            return None
        unpadded_page_size = (
            spec.unpadded_page_size_bytes * block_size // spec.block_size
        )
        return max(spec.page_size_padded, unpadded_page_size)

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1487-L1523（SWA/chunked →
    #   FullAttentionSpec（记录窗口/块尺寸作档案）；SlidingWindowMLA 分支
    #   L1489-L1501 随第 4 条删）
    if has_full_attention and (has_sliding_window or has_chunked_local_attention):
        for layer_name, spec in kv_cache_spec.items():
            if isinstance(spec, SlidingWindowSpec):
                block_size = full_attention_block_size or spec.block_size
                promoted_specs[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    head_size_v=spec.head_size_v,
                    dtype=spec.dtype,
                    kv_quant_mode=spec.kv_quant_mode,
                    sliding_window=spec.sliding_window,
                    page_size_padded=promoted_page_size_padded(spec, block_size),
                )
            elif isinstance(spec, ChunkedLocalAttentionSpec):
                block_size = full_attention_block_size or spec.block_size
                promoted_specs[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    attention_chunk_size=spec.attention_chunk_size,
                    page_size_padded=promoted_page_size_padded(spec, block_size),
                )

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1525-L1529
    if not (
        is_kv_cache_spec_uniform(promoted_specs)
        or UniformTypeKVCacheSpecs.is_uniform_type(promoted_specs)
    ):
        raise ValueError("Failed to promote local KV cache specs to one unified type.")

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1531
    return promoted_specs


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1534 _try_get_full_allocation_fallback_groups
def _try_get_full_allocation_fallback_groups(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Try a supported full-allocation fallback for local-attention layers."""
    # SUBTRACTED: HiddenState/SlidingWindowMLA/ChunkedLocal 的早退判定
    #   （L1538-L1544——第 4/5 条；MLA+SWA 的回退族随 SlidingWindowMLA 删，
    #   chunked-local 本就无回退）
    # SUBTRACTED: 本函数的 MLA 回退主体（L1546-L1565——第 4 条：真实只服务
    #   MLA+SWA 页不齐的混合；本章族里 unify 失败直接 raise（NotImplemented
    #   Error 由调用点透传）。保留函数面作 get_kv_cache_groups 控制流的
    #   账位——非 MLA 混合恒 None。
    return None


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1568 unify_hybrid_kv_cache_specs
def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]):
    """
    This function tries to convert the KV cache specs to one type if the model
    is a hybrid model with multiple type of KV cache. It will convert all
    SlidingWindowSpec to FullAttentionSpec if both types are present.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model
    """

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1578-L1581
    if is_kv_cache_spec_uniform(
        kv_cache_spec
    ) or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec):
        return

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1583-L1589（warning 原话：不省
    #   显存——窗外 KV 白占，但滑窗计算本身仍省）
    logger.warning(
        "Hybrid KV cache manager is disabled for this hybrid model, "
        "This means we do not enable any optimizations for saving KV cache "
        "memory (e.g., dropping the KV cache outside the sliding window). "
        "The compute of layers like sliding window is still saved."
    )
    kv_cache_spec.update(_promote_local_kv_cache_specs(kv_cache_spec))


# SUBTRACTED: group_and_unify_kv_cache_specs / _get_kv_cache_groups_uniform_
#   groups / _approximate_gcd / _annotate_eagle_groups_deepseek_v4
#   （L1592-L1754、L1757-L1778——dossier.delete 第 4/6 条：DSV4 特路 +
#   eagle 草稿组标注）。


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1781 get_kv_cache_groups
def get_kv_cache_groups(
    vllm_config: "VllmConfig", kv_cache_spec: dict[str, KVCacheSpec]
) -> list[KVCacheGroupSpec]:
    """
    Split the layers in the model into groups with the same KV cache spec.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroups
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1794-L1795（disable 回退先跑——
    #   原位改写 kv_cache_spec）
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1797-L1800
    if is_kv_cache_type_attention_free(kv_cache_spec):
        # This returns an empty list to allow for the KVCacheManager to handle
        # attention free models.
        return []

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1802-L1811（uniform 单组 /
    #   同型异宽单组）
    if is_kv_cache_spec_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        # All layers need the same number of token slots (e.g., all layers are
        # full attention, or all layers are sliding window attention with the
        # same window size). Put all layers into one group.
        return _get_kv_cache_groups_uniform_type(uniform_spec)
    # SUBTRACTED: DSV4 分组分支（L1812-L1819——第 4 条）
    # SUBTRACTED: HiddenState 层抽离段（L1821-L1830、L1843-L1850——第 5 条）

    # Prefer preserving each layer's cache semantics. If physical pages cannot
    # be unified, try a supported allocation-only fallback before failing.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1832-L1841（先统一页大小再等
    #   量化分组；unify 失败时回退链在非 MLA 混合下透传 raise）
    try:
        filtered_spec = unify_kv_cache_spec_page_size(kv_cache_spec)
    except NotImplementedError:
        fallback_groups = _try_get_full_allocation_fallback_groups(kv_cache_spec)
        if fallback_groups is None:
            raise
        return fallback_groups
    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1852
    return groups


# --------------------------------------------------------------------------- #
# m9 一份账喂两侧
# --------------------------------------------------------------------------- #


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1855 generate_scheduler_kv_cache_config
def generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],
) -> KVCacheConfig:
    """
    Generate the KV cache configuration for the scheduler.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1861-L1863（各 worker num_blocks
    #   必须一致——单源即防漂的结构保证）
    assert all(
        [cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs]
    )
    # All workers have the same kv_cache_config except layer names, so use
    # an arbitrary one to initialize the scheduler.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1866-L1874（拍平：UniformType
    #   任取代表层）
    cfg = copy.deepcopy(kv_cache_configs[0])
    for group in cfg.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # so use an arbitrary one to initialize the scheduler.
            group.kv_cache_spec = next(
                iter(group.kv_cache_spec.kv_cache_specs.values())
            )
    return cfg


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1877 get_kv_cache_capacity
def get_kv_cache_capacity(
    vllm_config: "VllmConfig", kv_cache_config: KVCacheConfig
) -> tuple[int, float]:
    """
    Get the group-aware KV cache token capacity and max concurrency.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1883-L1887（容量 = 并发 ×
    #   max_model_len——混合布局按组求和也正确）
    max_model_len = vllm_config.model_config.max_model_len
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    return int(max_concurrency * max_model_len), max_concurrency


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1890 _max_memory_usage_bytes_from_groups
def _max_memory_usage_bytes_from_groups(
    vllm_config: "VllmConfig",
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """
    Calculate maximum memory usage in bytes from KV cache groups.

    This correctly accounts for padding in hybrid models. For example, if a
    model has 8 full attention layers and 9 sliding window layers, they will
    be padded to 9 full + 9 sliding window for uniform group sizes.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1901-L1902
    if not kv_cache_groups:
        return 0

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1904-L1912（单组异宽：逐层求和）
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # UniformTypeKVCacheSpecs special case (single group, per-layer specs)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        return sum(
            spec.max_memory_usage_bytes(vllm_config)
            for spec in per_layer_specs.values()
        )
    # SUBTRACTED: 全 UniformType（DSV4）分支（L1913-L1937——第 4 条）

    # General case: group_size pools, each shared by one layer per group
    # Memory = group_size * page_size * blocks_for_max_len
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1939-L1950（按组求和 × group_size
    #   × page——padding 层的账也算进去）
    group_size = max(len(group.layer_names) for group in kv_cache_groups)
    page_size = get_uniform_page_size(
        [group.kv_cache_spec for group in kv_cache_groups]
    )
    blocks_needed = sum(
        cdiv(group.kv_cache_spec.max_memory_usage_bytes(vllm_config), page_size)
        for group in kv_cache_groups
    )

    return group_size * page_size * blocks_needed


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1953 _estimate_max_model_len_from_groups
def _estimate_max_model_len_from_groups(
    vllm_config: "VllmConfig",
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> int:
    """
    Binary search for the maximum model length that fits in available memory.
    Returns 0 if even 1 token doesn't fit.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1962-L1985（per-worker 投影组的
    #   二分版；try/finally 恢复原值）
    original_max = vllm_config.model_config.max_model_len

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L1964 fits
    def fits(model_len: int) -> bool:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L1965-L1969
        vllm_config.model_config.max_model_len = model_len
        return (
            _max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)
            <= available_memory
        )

    try:
        left, right = 1, original_max
        if not fits(left):
            return 0
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        vllm_config.model_config.max_model_len = original_max


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1988 _auto_fit_max_model_len
def _auto_fit_max_model_len(
    vllm_config: "VllmConfig",
    projected_groups_per_worker: list[list[KVCacheGroupSpec]],
    available_memory: list[int],
) -> None:
    """
    When max_model_len is set to -1, this function estimates the largest
    context length that can be supported with the available GPU memory.
    It uses binary search to find the maximum length that fits across all
    workers.

    Args:
        vllm_config: The global VllmConfig (will be modified in-place)
        projected_groups_per_worker: KV cache groups projected to each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2005
    original_max = vllm_config.model_config.max_model_len

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2007-L2014（attention-free 早退）
    if all(not groups for groups in projected_groups_per_worker):
        # All workers have empty specs (attention-free model)
        logger.info(
            "Auto-fit max_model_len: attention-free model, "
            "using derived max_model_len=%d",
            original_max,
        )
        return

    # Find the max_model_len that fits across all workers.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2016-L2025（各 worker 分别二分
    #   取最小——limiting worker）
    auto_fit_max = original_max
    limiting_worker_mem = available_memory[0]
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        worker_max = _estimate_max_model_len_from_groups(vllm_config, groups, avail_mem)
        if worker_max < auto_fit_max:
            auto_fit_max = worker_max
            limiting_worker_mem = avail_mem

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2027-L2031
    if auto_fit_max <= 0:
        raise ValueError(
            "Cannot auto-fit max_model_len: not enough GPU memory available "
            "to serve even a single token. Try increasing `gpu_memory_utilization`."
        )

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2033-L2049（全装得下保持原值 /
    #   否则原位改小 + 日志）
    if auto_fit_max >= original_max:
        # The model's full context length fits in memory
        logger.info(
            "Auto-fit max_model_len: full model context length %d fits in "
            "available GPU memory",
            original_max,
        )
    else:
        # Need to reduce max_model_len to fit in memory
        vllm_config.model_config.max_model_len = auto_fit_max
        logger.info(
            "Auto-fit max_model_len: reduced from %d to %d to fit in "
            "available GPU memory (%s GiB available for KV cache)",
            original_max,
            auto_fit_max,
            format_gib(limiting_worker_mem),
        )


# SOURCE: vllm/v1/core/kv_cache_utils.py:L2052 _project_kv_cache_groups_to_worker
def _project_kv_cache_groups_to_worker(
    global_kv_cache_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Projects global KV cache groups onto a single worker's assigned layers.

    In pipeline parallelism, each worker only owns a subset of layers. This
    function filters the global groups to include only layers present on the
    given worker, adjusting UniformTypeKVCacheSpecs accordingly.

    Args:
        global_kv_cache_groups: The global KV cache groups for the whole model.
        worker_spec: The KV cache spec of each layer on this worker.

    Returns:
        The projected KV cache groups containing only this worker's layers.
    """
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2070-L2091（过滤本 worker 层；
    #   SUBTRACTED：UniformType 的逐层重建内景 L2076-L2083（dossier.delete
    #   第 12 条——保留函数与过滤逻辑；is_eagle_group 透传 L2088 随第 6 条删））
    projected_groups: list[KVCacheGroupSpec] = []
    for group in global_kv_cache_groups:
        worker_layer_names = [
            layer_name for layer_name in group.layer_names if layer_name in worker_spec
        ]
        group_spec = group.kv_cache_spec
        projected_groups.append(
            KVCacheGroupSpec(
                worker_layer_names,
                group_spec,
            )
        )
    return projected_groups


# --------------------------------------------------------------------------- #
# m1/m4 定账总控
# --------------------------------------------------------------------------- #


# SOURCE: vllm/v1/core/kv_cache_utils.py:L2094 get_kv_cache_configs
def get_kv_cache_configs(
    vllm_config: "VllmConfig",
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """
    Generates the KV cache configurations for a model.
    Since we use a shared centralized controller for all workers, we need the
    `kv_cache_config` to be consistent across all workers to make sure
    the KV cache allocation can be applied to all workers. However, different
    workers may have different memory available, and different type of layers
    (when pipeline parallel is enabled). To handle the difference between
    workers, the current implementation is:
    1. Merge the KV cache specs of all workers to get the KVCacheSpecs for
       the whole model.
    2. Generate the KV cache groups based on the layer ratio of the whole model.
       This also handles spec unification for hybrid models.
    3. Handle auto-fit max_model_len and memory checks using per-worker
       projected groups to account for PP sharding.
    4. Generate the KV cache configs for each worker based on the KV cache
       grouping strategy. (This is reasonable because the layer ratio of
       different PP stages are similar.)
    5. Change the num_blocks of each worker to the smallest among all workers
       and shrink tensor sizes proportionally to avoid allocating unused memory.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_specs: List of dict[layer_name, KVCacheSpec] for each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.

    Returns:
        The generated KVCacheConfigs for each worker.
    """

    # Merge the KV cache specs of all workers. Different PP stages may have
    # different layer names, and different TP ranks of the same PP stage should
    # have the same KV cache spec.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2129-L2141
    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_name] = layer_spec
            else:
                assert merged_kv_cache_specs[layer_name] == layer_spec, (
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet."
                )

    # Check if the KV cache specs are registered correctly.
    # This is to prevent that some layers are initialized with unregistered specs.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2143-L2145
    from .kv_cache_spec_registry import KVCacheSpecRegistry

    KVCacheSpecRegistry.check_kv_cache_spec_registry(merged_kv_cache_specs)
    # Get global KV cache groups. This also handles spec unification for
    # hybrid models when disable_hybrid_kv_cache_manager is enabled.
    # After this call, merged_kv_cache_specs may be modified in-place.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2146-L2149
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    # If original_max_model_len was -1, automatically
    # determine the maximum model length that fits in available GPU memory.
    # We use per-worker projected groups to account for PP sharding.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2151-L2157
    projected_groups_per_worker = [
        _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec)
        for worker_spec in kv_cache_specs
    ]

    # If `num_gpu_blocks_override` is set, the cache size that will actually
    # be allocated is decoupled from the profiled `available_memory`:
    # `may_override_num_blocks` in `get_kv_cache_config_from_groups` clamps
    # `num_blocks` to the override. Reflect that in `available_memory` here so
    # auto-fit, the admission check, and the per-worker config builder all
    # plan against the same effective capacity.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2159-L2179（override 折算——
    #   available 换算成 override×每块字节，账本不漂）
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        adjusted_memory: list[int] = []
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
            if not groups:
                adjusted_memory.append(avail_mem)
                continue
            bytes_per_block = _pool_bytes_per_block(vllm_config, groups)
            logger.info(
                "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
                avail_mem // bytes_per_block,
                override,
            )
            adjusted_memory.append(override * bytes_per_block)
        available_memory = adjusted_memory

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2181-L2184
    if vllm_config.model_config.original_max_model_len == -1:
        _auto_fit_max_model_len(
            vllm_config, projected_groups_per_worker, available_memory
        )

    # Check if the available memory is enough per worker.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2186-L2195（逐 worker 护栏）
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        _check_enough_kv_cache_memory(
            avail_mem,
            partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(_estimate_max_model_len_from_groups, vllm_config, groups),
        )

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2197-L2208（逐 worker 出 config）
    kv_cache_configs: list[KVCacheConfig] = []
    for projected_groups, kv_cache_spec_one_worker, available_memory_one_worker in zip(
        projected_groups_per_worker, kv_cache_specs, available_memory
    ):
        assert sum(len(group.layer_names) for group in projected_groups) == len(
            kv_cache_spec_one_worker
        ), "Some layers are not assigned to any group."
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(
                vllm_config, projected_groups, available_memory_one_worker
            )
        )

    # Change the num_blocks of each rank to the smallest among all ranks.
    # We also need to shrink the tensor size proportionally to avoid
    # allocating unused memory.
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2210-L2223（PP 取最小 + 按
    #   比例缩张量）
    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    for kv_cache_config in kv_cache_configs:
        num_blocks_old = kv_cache_config.num_blocks
        kv_cache_config.num_blocks = min_num_blocks

        # Shrink tensor size proportionally
        for tensor in kv_cache_config.kv_cache_tensors:
            assert tensor.size % num_blocks_old == 0
            tensor.size = tensor.size // num_blocks_old * min_num_blocks

        # SOURCE: vllm/v1/core/kv_cache_utils.py:L2225-L2240（容量/并发日志：
        #   混合布局按组求和也正确）
        if len(kv_cache_config.kv_cache_groups) > 0:
            max_model_len = vllm_config.model_config.max_model_len
            # GPU KV cache size in tokens = max_concurrency * max_model_len:
            # the total tokens of context the pool can hold at peak
            # utilization. Sourcing this from the concurrency calculation
            # handles hybrid layouts correctly.
            num_tokens, max_concurrency = get_kv_cache_capacity(
                vllm_config, kv_cache_config
            )

            logger.info("GPU KV cache size: %s tokens", f"{num_tokens:,}")
            logger.info(
                "Maximum concurrency for %s tokens per request: %.2fx",
                f"{max_model_len:,}",
                max_concurrency,
            )

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L2242
    return kv_cache_configs
