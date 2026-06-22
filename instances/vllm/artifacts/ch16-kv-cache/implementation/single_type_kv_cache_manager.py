# SPDX-License-Identifier: Apache-2.0
# 只做减法的精简版 —— 忠实子集，与 vLLM 同名同结构同控制流。
# 验收判据：把真实 vLLM 删掉所有 # SUBTRACTED 分支，应当 ≈ 得到本文件。
#
# 对应 vllm/v1/core/single_type_kv_cache_manager.py：单一注意力类型的块账本。
# 本章在 ch15（单组全注意力主路径）之上补全三件深水区：
#   1. get_num_blocks_to_allocate 的 skipped 折抵 + 可驱逐块计数 + admission cap；
#   2. allocate_new_computed_blocks 的 null 填充 skipped 段 + external 命中块分配；
#   3. remove_skipped_blocks 逆序回收窗外块；
# 并补齐 SlidingWindowManager / ChunkedLocalAttentionManager 两个差异化子类
# （get_num_skipped_tokens 各异）+ spec_manager_map + get_manager_for_kv_cache_spec。
# Mamba / CrossAttention / Sink 三类只保留类壳与映射条目（专用体 SUBTRACTED）。
import itertools
from collections import defaultdict
from collections.abc import Sequence

from .block_pool import BlockPool
from .kv_cache_utils import BlockHash, KVCacheBlock
from .request import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    Request,
    SlidingWindowSpec,
    cdiv,
)


# SUBTRACTED: TQFullAttentionSpec / MLAAttentionSpec / SlidingWindowMLASpec /
# SinkFullAttentionSpec / CrossAttentionSpec / MambaSpec 的独立 spec 类
# （vllm/v1/kv_cache_interface.py）—— 它们各自映射到本文件已有的 manager 类
# （见 spec_manager_map 注释），本章三类注意力路径不依赖其专用字段。


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L30 (SingleTypeKVCacheManager)
class SingleTypeKVCacheManager:
    """
    A manager that handles the kv cache management logic of one specific type
    of attention layer.
    """
    # SUBTRACTED: ABC / @abstractmethod 抽象基类语义（L30, L320, L336）—— 精简版直接
    # 提供 get_num_skipped_tokens 默认实现与具体子类，无需抽象约束。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L36 (__init__)
    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        max_admission_blocks_per_request: int | None = None,
    ) -> None:
        """
        Args:
            max_admission_blocks_per_request: Recycling-aware per-request
                block cap used by `get_num_blocks_to_allocate`. Only set for
                spec types that recycle blocks across chunks (SWA,
                chunked-local); `None` (the default) means no cap, which is
                correct for full-attention-style specs that hold every block
                until the request finishes.
        """
        # SUBTRACTED: dcp_world_size / pcp_world_size 乘子（L42-L43, L60-L63）—— 上下文
        # 并行，单卡为 1，block_size 乘子无效果。
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.enable_caching = enable_caching
        self._max_admission_blocks_per_request = max_admission_blocks_per_request
        self.new_block_ids: list[int] = []

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

        # {req_id: The number of cached blocks for this given request}.
        # Only used to track the RUNNING requests.
        self.num_cached_block: dict[str, int] = {}

        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L84
    @classmethod
    def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L85
        return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L88 (get_num_blocks_to_allocate)
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        """Get the number of blocks needed to be allocated for the request."""
        num_required_blocks = cdiv(num_tokens, self.block_size)
        if apply_admission_cap and self._max_admission_blocks_per_request is not None:
            # Recycling-aware specs (SWA, chunked-local) cap the per-request
            # reservation here so admission matches the startup pool sizer
            # (single source of truth: the spec method). `remove_skipped_blocks`
            # runs from `allocate_slots` before each chunk's
            # `get_num_blocks_to_allocate`, so per-request peak real-held blocks
            # <= this cap, which keeps `sum(reservations) <= pool` <=>
            # `sum(peak_real_held) <= pool`.
            num_required_blocks = min(
                num_required_blocks, self._max_admission_blocks_per_request
            )
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            assert len(new_computed_blocks) == 0
            # NOTE: With speculative decoding, request's blocks may be allocated
            # for draft tokens which are later rejected. In this case,
            # num_required_blocks may be smaller than num_req_blocks.
            return max(num_required_blocks - num_req_blocks, 0)

        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
        # Number of whole blocks that are skipped by the attention window.
        # If nothing is skipped (full attention), this is 0.
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # We need blocks for the non-skipped suffix. If there are still
        # local-computed blocks inside the window, they contribute to the
        # required capacity; otherwise, skipped blocks dominate.
        num_new_blocks = max(
            num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),
            0,
        )

        # Among the `new_computed_blocks`, the first `num_skipped_blocks` worth
        # of blocks are skipped; `num_req_blocks` of those may already be in
        # `req_to_blocks`, so only skip the remainder from `new_computed_blocks`.
        num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)

        # If a computed block is an eviction candidate (in the free queue and
        # ref_cnt == 0), it will be removed from the free queue when touched by
        # the allocated request, so we must count it in the free-capacity check.
        num_evictable_blocks = self._get_num_evictable_blocks(
            new_computed_blocks[num_skipped_new_computed_blocks:]
        )
        return num_new_blocks + num_evictable_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L169
    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the new computed blocks to the request. This involves three steps:
        1. Touch the computed blocks to make sure they won't be evicted.
        1.5. (Optional) For sliding window, skip blocks are padded with null blocks.
        2. Add the remaining computed blocks.
        3. (Optional) For KV connectors, allocate new blocks for external computed
            tokens (if any).
        """
        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            assert len(new_computed_blocks) == 0
            return

        # A new request.
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            # It is possible that all new computed blocks are skipped when
            # num_skipped_blocks > len(new_computed_blocks).
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
            # Some external computed tokens may be skipped too.
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), (
                "Computed blocks should be empty when prefix caching is disabled"
            )

        # Skip blocks are padded with null blocks.
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the remaining computed blocks.
        req_blocks.extend(new_computed_blocks)
        # All cached hits (including skipped nulls) are already cached; mark
        # them so cache_blocks() will not try to re-cache blocks that already
        # have a block_hash set.
        self.num_cached_block[request_id] = len(req_blocks)

        if num_external_computed_tokens > 0:
            # Allocate new blocks for external computed tokens.
            allocated_blocks = self.block_pool.get_new_blocks(
                cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
            )
            req_blocks.extend(allocated_blocks)
            # SUBTRACTED: TQFullAttentionSpec 一并纳入条件（L239）—— 张量量化全注意力
            # 与 FullAttentionSpec 共用 FullAttentionManager；本章只造 FullAttentionSpec。
            if type(self.kv_cache_spec) is FullAttentionSpec:
                self.new_block_ids.extend(b.block_id for b in allocated_blocks)

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L242
    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        """Allocate new blocks for the request to give it at least
        `num_tokens` token slots."""
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            # SUBTRACTED: TQFullAttentionSpec 一并纳入条件（L267）—— 见上同名说明。
            if type(self.kv_cache_spec) is FullAttentionSpec:
                self.new_block_ids.extend(b.block_id for b in new_blocks)
            return new_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L271 (take_new_block_ids)
    def take_new_block_ids(self) -> list[int]:
        """Drain and return block IDs allocated since the last call."""
        ids = self.new_block_ids
        self.new_block_ids = []
        return ids

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L277
    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """Cache the full blocks for the request."""
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        if num_cached_blocks >= num_full_blocks:
            return

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
        )

        self.num_cached_block[request.request_id] = num_full_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L303
    def free(self, request_id: str) -> None:
        """Free the blocks for the request."""
        # Default to [] in case a request is freed (aborted) before alloc.
        req_blocks = self.req_to_blocks.pop(request_id, [])

        # Free blocks in reverse order so that the tail blocks are freed first.
        ordered_blocks = reversed(req_blocks)

        self.block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request_id, None)

    # SUBTRACTED: get_num_common_prefix_blocks 抽象方法（L320-L334）—— 级联注意力公共
    # 前缀统计，非分配/命中核心控制流。各子类的具体实现也一并 SUBTRACTED。
    # SUBTRACTED: find_longest_cache_hit 抽象声明（L336-L383）—— 由各子类 classmethod
    # 提供实体（本章保留 Full / SlidingWindow / ChunkedLocal 三个实体）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L385 (remove_skipped_blocks)
    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """
        Remove and free the blocks that are no longer needed for attention
        computation. The removed blocks should be replaced by null_block.

        This depends on `get_num_skipped_tokens`, implemented differently for
        each attention type.
        """
        # Remove the blocks that will be skipped during attention computation.
        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        if num_skipped_tokens <= 0:
            # This indicates that ALL tokens are inside attention window.
            # Thus we do not need to free any blocks outside attention window.
            # A typical case is full attention that we never free any token
            # before the request is finished.
            return
        blocks = self.req_to_blocks[request_id]
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # `num_skipped_tokens` may include tokens that haven't been allocated yet
        # (e.g., when the attention window moves into the external computed tokens
        # range), so we must cap to the number of blocks that currently exist for
        # this request.
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))
        removed_blocks: list[KVCacheBlock] = []
        # Because the block starts from index 0, the num_skipped_block-th block
        # corresponds to index num_skipped_blocks - 1.
        for i in range(num_skipped_blocks - 1, -1, -1):
            if blocks[i] == self._null_block:
                # If the block is already a null block, the blocks before it
                # should also have been set to null blocks by the previous calls
                # to this function.
                break
            removed_blocks.append(blocks[i])
            blocks[i] = self._null_block
        self.block_pool.free_blocks(removed_blocks)

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L428 (get_num_skipped_tokens)
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """Get the number of tokens that will be skipped for attention
        computation. The default behavior is to not skip any tokens."""
        return 0

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L441 (new_step_starts)
    def new_step_starts(self) -> None:
        # do nothing by default
        return None


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L446 (FullAttentionManager)
class FullAttentionManager(SingleTypeKVCacheManager):
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L447 (find_longest_cache_hit)
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
    ) -> tuple[list[KVCacheBlock], ...]:
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L447 (find_longest_cache_hit)
        # SUBTRACTED: dcp_world_size / pcp_world_size 参数与 block_size 乘子
        # （L457-L458, L470-L471）—— 上下文并行，单卡为 1。
        assert isinstance(
            kv_cache_spec, FullAttentionSpec | ChunkedLocalAttentionSpec
        ), (
            "FullAttentionManager can only be used for full attention "
            "and chunked local attention groups"
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        max_num_blocks = max_length // block_size
        for block_hash in itertools.islice(block_hashes, max_num_blocks):
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_block, the following block hashes are
            # not computed yet for sure.
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        # SUBTRACTED: use_eagle 丢尾块（L484-L487）—— 投机解码草稿头专用，非投机路径
        # use_eagle=False 不触发。
        # Re-align to a multiple of `alignment_tokens` (the LCM of all group
        # block sizes in hybrid models). No-op when block_size == alignment.
        while (
            block_size != alignment_tokens  # Faster for common case.
            and len(computed_blocks[0]) * block_size % alignment_tokens != 0
        ):
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    # SUBTRACTED: get_num_common_prefix_blocks（L496-L504）—— 级联前缀统计，非命中核心。


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L507 (SlidingWindowManager)
class SlidingWindowManager(SingleTypeKVCacheManager):
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L508 (__init__)
    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L512 (find_longest_cache_hit)
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
    ) -> tuple[list[KVCacheBlock], ...]:
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L512 (find_longest_cache_hit)
        # SUBTRACTED: dcp_world_size / pcp_world_size 参数与 assert==1（L522-L529）。
        assert isinstance(kv_cache_spec, SlidingWindowSpec), (
            "SlidingWindowManager can only be used for sliding window groups"
        )

        # The number of contiguous blocks needed for prefix cache hit.
        # -1 since the input token itself is also included in the window
        sliding_window_contiguous_blocks = cdiv(
            kv_cache_spec.sliding_window - 1, kv_cache_spec.block_size
        )
        # SUBTRACTED: use_eagle 时 +1 多匹配一块再丢尾（L536-L541）—— 投机解码草稿头专用。

        # TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
        # optimize the time complexity from O(max_num_blocks) to
        # O(max_num_blocks / sliding_window_contiguous_blocks +
        # sliding_window_contiguous_blocks),
        # which is good for low cache hit rate scenarios.
        max_num_blocks = max_length // kv_cache_spec.block_size
        computed_blocks = tuple(
            [block_pool.null_block] * max_num_blocks
            for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        num_contiguous_blocks = 0
        match_found = False
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # Skip prefix matching check if the block is not aligned with
                # `alignment_tokens`.
                if num_contiguous_blocks == 0 and block_size != alignment_tokens:
                    # SUBTRACTED: use_eagle 分支取 i 而非 i+1（L564）。
                    post_pop_blocks = i + 1
                    if (post_pop_blocks * block_size) % alignment_tokens != 0:
                        continue
                # Add the cached block to the computed blocks.
                for computed, cached in zip(computed_blocks, cached_block):
                    computed[i] = cached
                num_contiguous_blocks += 1
                if num_contiguous_blocks >= sliding_window_contiguous_blocks:
                    # Trim the trailing blocks.
                    # E.g., [NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                    # when sliding_window_contiguous_blocks=2.
                    for computed in computed_blocks:
                        del computed[i + num_contiguous_blocks :]
                    match_found = True
                    break
            else:
                num_contiguous_blocks = 0
        if not match_found:
            # The first `num_contiguous_blocks` is a cache hit even if
            # `num_contiguous_blocks < sliding_window_contiguous_blocks`.
            for computed in computed_blocks:
                del computed[num_contiguous_blocks:]
            while (
                block_size != alignment_tokens  # Faster for common case.
                and len(computed_blocks[0]) * block_size % alignment_tokens != 0
            ):
                for computed in computed_blocks:
                    computed.pop()
        # SUBTRACTED: use_eagle 丢尾块 + 重对齐（L592-L603）—— 投机解码草稿头专用。
        return computed_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L606 (get_num_skipped_tokens)
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        For sliding window, the skipped tokens are those prior to the current
        sliding window.

        Example: sliding_window=4, num_computed_tokens=7. The current window
        contains tokens 4~7; tokens 0~3 are outside the window and skipped, so
        get_num_skipped_tokens(7) == 4.
        """
        return max(0, num_computed_tokens - self.sliding_window + 1)

    # SUBTRACTED: get_num_common_prefix_blocks（L634-L641）—— 级联前缀统计，滑窗恒返 0。


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L644 (ChunkedLocalAttentionManager)
class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L645 (__init__)
    def __init__(self, kv_cache_spec: ChunkedLocalAttentionSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.attention_chunk_size = kv_cache_spec.attention_chunk_size

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L649 (find_longest_cache_hit)
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
    ) -> tuple[list[KVCacheBlock], ...]:
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L649 (find_longest_cache_hit)
        """
        For chunked local attention, blocks fully outside the local window are
        marked as computed with null blocks; blocks inside the window are
        resolved by cache lookup.

        Example: attention chunk size 8, block size 4, max length 15. For next
        token at index 15, tokens 0~7 are outside the window (already computed,
        marked null); we look up block 3 (tokens 8~11) -> [null, null, block 3]
        on hit, else [null, null].
        """
        # SUBTRACTED: dcp_world_size / pcp_world_size 参数与 assert==1（L659-L705 散点）。
        assert isinstance(kv_cache_spec, ChunkedLocalAttentionSpec), (
            "ChunkedLocalAttentionManager can only be used for "
            "chunked local attention groups"
        )
        assert use_eagle is False, (
            "Hybrid KV cache is not supported for "
            "eagle + chunked local attention."
        )
        assert kv_cache_spec.block_size == alignment_tokens, (
            "KV cache groups with different block sizes are not compatible with "
            "chunked local attention now"
        )
        max_num_blocks = max_length // kv_cache_spec.block_size
        if max_length > 0:
            local_attention_start_idx = (
                max_length
                // kv_cache_spec.attention_chunk_size
                * kv_cache_spec.attention_chunk_size
            )
        else:
            local_attention_start_idx = 0
        # we marked blocks out of window as computed with null blocks, and
        # blocks inside window based on cache lookup result.
        local_attention_start_block_idx = (
            local_attention_start_idx // kv_cache_spec.block_size
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [block_pool.null_block] * local_attention_start_block_idx
            for _ in range(len(kv_cache_group_ids))
        )
        for i in range(local_attention_start_block_idx, max_num_blocks):
            block_hash = block_hashes[i]
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        return computed_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L741 (get_num_skipped_tokens)
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        For chunked local attention, the skipped tokens are those on the left
        side of the current chunk (rounded down to a chunk boundary).

        Example: chunk size 8, num_computed_tokens=13. Tokens 0~7 are skipped
        (the current chunk starts at 8), so get_num_skipped_tokens(13) == 8.
        """
        num_skipped_tokens = (
            num_computed_tokens // self.attention_chunk_size
        ) * self.attention_chunk_size
        return num_skipped_tokens

    # SUBTRACTED: get_num_common_prefix_blocks（L787-L791）—— 级联前缀统计，恒返 0。


# SUBTRACTED: MambaManager（L794-...）的 align 模式实现体（last_state_block_idx /
# _allocated_block_reqs / cached_blocks_this_step / num_speculative_blocks 分支及其
# get_num_blocks_to_allocate / allocate_new_blocks 重写）—— Mamba 状态空间模型块语义
# 自成体系，留作后续章节。保留类壳与 spec_manager_map 条目以维持映射完整。
# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L794 (MambaManager)
class MambaManager(SingleTypeKVCacheManager):
    pass


# SUBTRACTED: CrossAttentionManager（L...-L1115）的 encoder-decoder（Whisper）专用体
# （cache_blocks/find_longest_cache_hit 抛错、按 num_encoder_tokens 静态分配）——
# decoder-only 模型 num_encoder_tokens 恒 0，分支不触发。保留类壳与映射条目。
# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py (CrossAttentionManager)
class CrossAttentionManager(SingleTypeKVCacheManager):
    pass


# SUBTRACTED: SinkFullAttentionManager（L1118-L1139）的 __init__ 内 sink 块摘取
# （free_block_queue.popleft_n）—— 仅 attention-sink 模型用。保留类壳与映射条目；
# 它继承 FullAttentionManager 的命中查找。
# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1118 (SinkFullAttentionManager)
class SinkFullAttentionManager(FullAttentionManager):
    pass


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1142 (spec_manager_map)
# spec 类型 → manager 类的注册表。get_manager_for_kv_cache_spec 据此分发，是
# "多注意力类型如何落到不同 manager" 的真相源。
spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    FullAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
    # SUBTRACTED: TQFullAttentionSpec / MLAAttentionSpec → FullAttentionManager、
    # SlidingWindowMLASpec → SlidingWindowManager、MambaSpec → MambaManager、
    # CrossAttentionSpec → CrossAttentionManager、SinkFullAttentionSpec →
    # SinkFullAttentionManager（L1144-L1151）—— 这些 spec 类本身已 SUBTRACTED；映射
    # 结构与上述三条完全同构，删除不影响本章三类注意力的分发演示。
}


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1155 (get_manager_for_kv_cache_spec)
def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    max_num_batched_tokens: int,
    max_model_len: int,
    **kwargs,
) -> SingleTypeKVCacheManager:
    manager_class = spec_manager_map[type(kv_cache_spec)]
    # SlidingWindow / ChunkedLocalAttention managers recycle blocks across
    # chunks; the runtime admission cap must match the recycling-aware bound
    # the startup pool sizer uses (single source of truth: the spec method).
    if isinstance(kv_cache_spec, (SlidingWindowSpec, ChunkedLocalAttentionSpec)):
        kwargs["max_admission_blocks_per_request"] = (
            kv_cache_spec.max_admission_blocks_per_request(
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
            )
        )
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager
