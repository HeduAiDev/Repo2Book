# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py
# 每注意力类型一份块账本（m11/m13）：SingleTypeKVCacheManager 基类——
# req_to_blocks 逻辑块表、get_num_blocks_to_allocate 需块预测（预测器与
# 分配器同构；apply_admission_cap 只由 full-ISL 门传 True）、remove_
# skipped_blocks 窗外块回收 + _remove_blocks_in_range 逆序 null 换位；
# FullAttentionManager（全历史从不回收）/ SlidingWindowManager（窗外回收）
# / ChunkedLocalAttentionManager（块对齐窗外回收）；get_manager_for_
# kv_cache_spec 注册表查建 + 回收型准入上限注入（运行期门与启动期定账
# 同源的装配点）；register_all_kvcache_specs 内置族注册。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 4 条 DSV4/TQ/MLA 零清类型表扩位；
#   第 5 条 RSWAManager/SinkFullAttentionManager/CrossAttentionManager/
#     MambaManager（mamba 状态管理 → 邻章——MambaSpec 只进定账算术）；
#   第 6 条 eagle：use_eagle 字段与 _contiguous_blocks_for_hit 的 eagle 位；
#   第 8 条 DCP/PCP 乘子（单卡恒 1 烘干：block_size 不放大）；
#   第 9 条 metrics 贯穿调用；
#   哈希侧 find_longest_cache_hit 基类抽象与 Full/SWA/Chunked 覆写
#     （L545-L593、L681-L830、L896-L1055、L1100-L1198——不动点命中与
#     链式哈希 → ch15）；reachable_block_mask/retention（稀疏驻留 → ch15）；
#   CoW/partial-hit 管线（_apply_cow/take_pending_cow_copies——→ ch15）；
#   allocate_external_computed_blocks 与 partial-tail offload（→ ch16）；
#   get_num_common_prefix_blocks 抽象与覆写（级联注意力旁路——ch13 同款
#     边界）。
from abc import ABC
from collections import defaultdict
from collections.abc import Sequence
from typing import ClassVar

from .block_pool import BlockPool
from .kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)
from .kv_cache_spec_registry import KVCacheSpecRegistry
from .kv_cache_utils import KVCacheBlock
from .math_utils import cdiv
from .request import Request


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L36 SingleTypeKVCacheManager
class SingleTypeKVCacheManager(ABC):
    """
    An abstract base class for a manager that handle the kv cache management
    logic of one specific type of attention layer.
    """

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L42
    supports_fine_grained_hash_lookup: ClassVar[bool] = False

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L44 __init__
    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        scheduler_block_size: int,
        needs_kv_cache_zeroing: bool = False,
        max_admission_blocks_per_request: int | None = None,
    ) -> None:
        """
        Initializes the SingleTypeKVCacheManager.
        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            block_pool: The block pool.
            kv_cache_group_id: The id of the kv cache group of this manager.
            scheduler_block_size: The scheduling granularity (LCM of all group
                block sizes); a multiple of this manager's ``block_size``.
            needs_kv_cache_zeroing: Whether worker-side KV cache zeroing needs
                newly allocated block IDs from this manager.
            max_admission_blocks_per_request: Recycling-aware per-request
                block cap used by `get_num_blocks_to_allocate`. Only set for
                spec types that recycle blocks across chunks (SWA,
                chunked-local); `None` (the default) means no cap, which is
                correct for full-attention-style specs that hold every
                block until the request finishes.
        """
        # SUBTRACTED: dcp_world_size/pcp_world_size 参数与乘子放大
        #   （L51-L52、L76-L79——dossier.delete 第 8 条，单卡恒 1 烘干）。
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L73-L75
        self.scheduler_block_size = scheduler_block_size
        # The block size for this manager; used for actual block allocation.
        self.block_size = kv_cache_spec.block_size
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L80-L83
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.enable_caching = enable_caching
        self._max_admission_blocks_per_request = max_admission_blocks_per_request
        # Record newly allocated block ids only when worker-side zeroing will
        # consume them and this manager holds a spec type that gets zeroed.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L86-L92（零清
        #   类型表缩到本章的 FullAttentionSpec——TQ/MLA/HiddenState 族随
        #   第 4/5 条删）
        self._record_new_block_ids = needs_kv_cache_zeroing and type(kv_cache_spec) in (
            FullAttentionSpec,
        )
        self.new_block_ids: list[int] = []

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L94-L97（逻辑
        #   块表本体）
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track
        # the data for the preempted ones.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L99-L103
        self.num_cached_block: dict[str, int] = {}

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L105-L106
        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

        # SUBTRACTED: use_eagle（L108-L112——第 6 条）、_partial_hit_reqs/
        #   _pending_cow_copies（L114-L117——CoW → ch15）、_pending_partial_
        #   tail_offloads（L118-L126——ch16）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L128 _get_num_evictable_blocks
    @classmethod
    def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L130
        return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L132 _has_partial_local_hit
    def _has_partial_local_hit(
        self,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
    ) -> bool:
        # The local prefix-cache hit ends inside one of this manager's
        # blocks: the shared tail block needs CoW.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L139-L142
        return (
            len(new_computed_blocks) > 0
            and num_local_computed_tokens % self.block_size != 0
        )

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L144 get_num_blocks_to_allocate
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.
            total_computed_tokens: Include both local and external computed
                tokens.
            num_local_computed_tokens: The number of local prefix-cache computed
                tokens.
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            apply_admission_cap: If True, clamp by `num_required_blocks` by
                `_max_admission_blocks_per_request`for recycling-aware specs
                (SWA, chunked-local).

        Returns:
            The number of blocks to allocate.
        """

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L178 cdiv 主算术
        num_required_blocks = cdiv(num_tokens, self.block_size)
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L179-L191 准入
        #   夹取段（#39734 铁律注释原文）：remove_skipped_blocks 在每 chunk
        #   的预测前先跑 ⇒ 峰值实持 ≤ cap ⇒ sum(预约) ≤ pool ⇔
        #   sum(峰值实持) ≤ pool——漂移即死锁或 mid-prefill OOM
        if apply_admission_cap and self._max_admission_blocks_per_request is not None:
            # Recycling-aware specs (SWA, chunked-local) cap the per-request
            # reservation here so admission matches the startup pool sizer
            # (`SlidingWindowSpec.max_admission_blocks_per_request` / its
            # chunked-local counterpart). `remove_skipped_blocks` runs from
            # `allocate_slots` before each chunk's `get_num_blocks_to_allocate`,
            # so per-request peak real-held blocks <= this cap, which keeps
            # `sum(reservations) <= pool` <=> `sum(peak_real_held) <= pool`.
            # Drift between the two would re-introduce the deadlock from
            # issue #39734 or, worse, mid-prefill OOM.
            num_required_blocks = min(
                num_required_blocks, self._max_admission_blocks_per_request
            )
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L192
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L194-L200 running
        #   fast-path 差值
        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            assert len(new_computed_blocks) == 0
            # NOTE: With speculative decoding, request's blocks may be allocated
            # for draft tokens which are later rejected. In this case,
            # num_required_blocks may be smaller than num_req_blocks.
            return max(num_required_blocks - num_req_blocks, 0)

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L202-L218（窗外
        #   跳块对差值的影响——SWA 组的回收把已持块从需求里减掉）
        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
        # Number of whole blocks that are skipped by the attention window.
        # If nothing is skipped, this is 0.
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
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L220-L225 可驱逐
        #   命中块计数
        num_evictable_blocks = self._get_num_evictable_blocks(
            new_computed_blocks[num_skipped_new_computed_blocks:]
        )
        # SUBTRACTED: partial-hit CoW 预留 +1（L226-L229——ch15）。
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L230
        return num_new_blocks + num_evictable_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L232 add_local_computed_blocks
    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
    ) -> None:
        """
        Add the locally cached (prefix-hit) blocks to the request:
        1. Touch the computed blocks (paired with adding them to `req_blocks`)
           so their ref_cnt exactly tracks the referencing requests.
        1.5. (Optional) For sliding window, skipped blocks are padded with nulls.
        2. Add the remaining computed blocks.

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
        """
        # SUBTRACTED: num_external_computed_tokens 参数与外部块半边
        #   （L237、L257-L259——第 13 条 → ch16）。
        # The coordinator only calls this for first-time allocations (running
        # requests are short-circuited there), so the request has no blocks yet.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L253-L265（跳过
        #   的命中块整块剔除——窗外命中不挂账）
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_computed_tokens = num_local_computed_tokens
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            # It is possible that all new computed blocks are skipped when
            # num_skipped_blocks > len(new_computed_blocks).
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]

        # Touch the computed blocks to make sure they won't be evicted.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L267-L273
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), (
                "Computed blocks should be empty when prefix caching is disabled"
            )

        # Skip blocks are padded with null blocks.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L275-L282
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the remaining computed blocks.
        req_blocks.extend(new_computed_blocks)
        # All cached hits (including skipped nulls) are already cached; mark
        # them so cache_blocks() will not try to re-cache blocks that already
        # have a block_hash set.
        self.num_cached_block[request_id] = len(req_blocks)
        # SUBTRACTED: partial-hit CoW 尾账（L283-L289——ch15）。

    # SUBTRACTED: allocate_external_computed_blocks（L291-L328——外部
    #   (connector) 计算块分配 → ch16）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L330 allocate_new_blocks
    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
        Returns:
            The new allocated blocks.
        """
        # SUBTRACTED: partial-hit CoW 重定向段（L347-L357——ch15）。
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L359-L369 差值
        #   分配主干（req_blocks.extend 挂账）
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            if self._record_new_block_ids:
                self.new_block_ids.extend(b.block_id for b in new_blocks)
            return new_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L371 records_new_block_ids
    @property
    def records_new_block_ids(self) -> bool:
        """Whether this manager's new blocks are zeroed by the worker."""
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L374
        return self._record_new_block_ids

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L376 take_new_block_ids
    def take_new_block_ids(self) -> list[int]:
        """Drain and return block IDs allocated since the last call."""
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L378-L380
        ids = self.new_block_ids
        self.new_block_ids = []
        return ids

    # SUBTRACTED: take_pending_cow_copies / take_pending_partial_tail_
    #   offloads（L382-L403——CoW → ch15 / offload → ch16）；_apply_cow
    #   （L405-L425——ch15）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L427 cache_blocks
    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
    ) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_tokens: The total number of tokens
                that need to be cached
                (including the tokens that are already cached).
        """
        # SUBTRACTED: retention_interval 参数与稀疏驻留掩码（L431、
        #   L451-L466——VLLM_PREFIX_CACHE_RETENTION_INTERVAL → ch15；
        #   reachable_block_mask 基类与 SWA/Mamba 覆写一并删）。
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L445-L449 幂等闸
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        if num_cached_blocks >= num_full_blocks:
            return

        # SUBTRACTED: cache_full_blocks 调用（L467-L475——ch15 哈希登记；
        #   本章 enable_caching=False 支恒不达此处——allocate_slots 在
        #   False 支早退）。
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L477
        self.num_cached_block[request.request_id] = num_full_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L500 pop_blocks_for_free
    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        """
        Pop the request's bookkeeping and return its blocks without yet
        returning them to the block pool. The caller is responsible for
        eventually passing the returned blocks to `block_pool.free_blocks`,
        freeing them in reverse order (so that tail blocks are evicted first).

        Args:
            request_id: The request ID.

        Returns:
            The request's blocks in allocation order.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L513-L517
        req_blocks = self.req_to_blocks.pop(request_id, [])
        self.num_cached_block.pop(request_id, None)
        return req_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L519 free
    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        # Free blocks in reverse order so that the tail blocks are freed first.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L526-L527
        self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))

    # SUBTRACTED: get_num_common_prefix_blocks 抽象（L529-L543——级联注意力
    #   旁路，ch13 同款边界）；find_longest_cache_hit 抽象（L545-L593
    #   ——链式哈希命中 → ch15）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L595 _remove_blocks_in_range
    def _remove_blocks_in_range(
        self,
        request_id: str,
        first_block: int,
        last_block: int,
    ) -> None:
        """Free blocks in ``[first_block, last_block)`` and replace with null_block.

        Iterates backward so newly-evictable tail blocks are reached even after
        earlier blocks in the range were nulled in a prior call.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L606-L609
        if request_id not in self.req_to_blocks:
            return
        if first_block >= last_block:
            return
        blocks = self.req_to_blocks[request_id]
        last_block = min(last_block, len(blocks))

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L613-L620 逆序
        #   null 换位（遇 null 早停——null 后面早已回收过）
        freed: list[KVCacheBlock] = []
        for i in range(last_block - 1, first_block - 1, -1):
            if blocks[i] == self._null_block:
                break
            freed.append(blocks[i])
            blocks[i] = self._null_block
        if freed:
            self.block_pool.free_blocks(freed)

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L622 remove_skipped_blocks
    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """
        Remove and free the blocks that are no longer needed for attention computation.
        The removed blocks should be replaced by null_block.

        This function depends on `get_num_skipped_tokens`, which need to be implemented
        differently for each attention type.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length for attention types (e.g.
                R-SWA) that evict a middle gap rather than a head prefix. Ignored
                by the default implementation.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L643（R-SWA 用
        #   参数账位——del 保留调用面契约）
        del num_prompt_tokens
        # Remove the blocks that will be skipped during attention computation.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L644-L651
        num_skipped_tokens = self.get_num_skipped_tokens(processed_computed_tokens)
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
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L653-L659
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))
        self._remove_blocks_in_range(request_id, 0, num_skipped_blocks)

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L661 get_num_skipped_tokens
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        # The default behavior is to not skip any tokens.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L672（full
        #   attention 基类恒 0——从不回收）
        return 0

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L674 new_step_starts
    def new_step_starts(self) -> None:
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L675
        return None


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L678 FullAttentionManager
class FullAttentionManager(SingleTypeKVCacheManager):
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L679
    supports_fine_grained_hash_lookup: ClassVar[bool] = True

    # SUBTRACTED: find_longest_cache_hit 覆写（L681-L777——链式哈希命中
    #   → ch15）；cache_blocks 覆写（L779-L819——retention 边界事件，
    #   ch15）；get_num_common_prefix_blocks 覆写（L821-L829——级联旁路）。
    # 全历史类型：基类行为即全部——从不 skip、整序列持块到请求结束。


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L878 SlidingWindowManager
class SlidingWindowManager(SingleTypeKVCacheManager):
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L879 __init__
    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L880-L881
        super().__init__(kv_cache_spec, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window

    # SUBTRACTED: _contiguous_blocks_for_hit 的 eagle 位（L883-L894——第 6 条）、
    #   find_longest_cache_hit 覆写（L896-L995——ch15）、reachable_block_mask
    #   覆写（L996-L1055——稀疏驻留 → ch15）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1057 get_num_skipped_tokens
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        For sliding window, this corresponds to the tokens that are prior to
        the current sliding window.

        Example:
        sliding_window=4, num_computed_tokens=7

        Tokens:   [ 0  1  2  3  4  5  6  7 ]
                  | ---- computed -----|
                                         ^ next token to be computed
                               |-----------| sliding window for next token
                  |--skipped---|

        The current window contains tokens 4~7. Tokens 0~3 will be skipped for
        attention computation since they are outside the sliding window.
        Thus, get_num_skipped_tokens(7) == 4.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1083（窗口
        #   左沿 = computed − window + 1）
        return max(0, num_computed_tokens - self.sliding_window + 1)

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1085 get_num_common_prefix_blocks
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        NOTE(Chen): The prefix blocks are null blocks for sliding window layers.
        So it's not correct to count ref_cnt like FullAttentionManager. Return
        0 here for correctness. Need to support cascade attention + sliding
        window in the future.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1092
        return 0


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1095 ChunkedLocalAttentionManager
class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1096 __init__
    def __init__(self, kv_cache_spec: ChunkedLocalAttentionSpec, **kwargs) -> None:
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1097-L1098
        super().__init__(kv_cache_spec, **kwargs)
        self.attention_chunk_size = kv_cache_spec.attention_chunk_size

    # SUBTRACTED: find_longest_cache_hit 覆写（L1100-L1198——ch15）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1200 get_num_skipped_tokens
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        For chunked local attention, this corresponds to the tokens that are on
        the left side of the current chunk.

        Example 1:
        chunk size = 8, num_computed_tokens = 13
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | ----- computed ---------------|
                                                  ^^ next token to be computed
                                   |----------------| <-- attention window for
                                                          next token
                 |--- skipped -----|
        Output: get_num_skipped_tokens(13) == 8

        Example 2:
        chunk size = 8, num_computed_tokens = 8
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | --- computed ---|
                                     ^ next token to be computed
                                   |--| <-- attention window for next token
                 | --- skipped ----|
        Output: get_num_skipped_tokens(8) == 8

        Example 3:
        chunk size = 8, num_computed_tokens = 7
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 |---computed---|
                                 ^ next token to be computed
                 |-----------------| <-- attention window for next token
                 no token should be skipped.
        Output: get_num_skipped_tokens(7) == 0

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1241-L1244
        # （整 chunk 对齐跳过：当前 chunk 左边界以前的全部跳过）
        num_skipped_tokens = (
            num_computed_tokens // self.attention_chunk_size
        ) * self.attention_chunk_size
        return num_skipped_tokens

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1246 get_num_common_prefix_blocks
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        cascade attention is not supported by chunked local attention.
        """
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1250
        return 0


# SUBTRACTED: MambaManager（L1253-L1833——mamba 状态管理（align 模式状态槽
#   /CoW/partial-tail）→ 邻章：本章 MambaSpec 只进定账算术（组化 pad /
#   resolve 回退判定），不建运行期 manager；RSWAManager（L832-L877）、
#   SinkFullAttentionManager（L1800-L1833）随 dossier.delete 第 5 条删。


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1836 get_manager_for_kv_cache_spec
def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    max_in_flight_tokens: int,
    max_model_len: int,
    **kwargs,
) -> SingleTypeKVCacheManager:
    """
    Get the appropriate manager for a given KVCacheSpec.

    Uses the KVCacheSpecRegistry to look up the manager class, supporting
    both built-in and custom specs registered via @register_kv_cache_spec
    and KVCacheSpecRegistry.register.

    Args:
        kv_cache_spec: The KVCacheSpec instance
        max_in_flight_tokens: The max tokens scheduled but not yet settled
            (one batch per concurrent step); see `VllmConfig.max_in_flight_tokens`
        max_model_len: The maximum context length the model could serve
    Returns:
        An instance of the appropriate SingleTypeKVCacheManager subclass
    """
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1857-L1860
    manager_class = KVCacheSpecRegistry.get_manager_class(kv_cache_spec)
    assert manager_class is not None, (
        f"No manager registered for KVCacheSpec {type(kv_cache_spec)}"
    )
    # SlidingWindow / ChunkedLocalAttention managers recycle blocks;
    # the runtime admission cap must match the recycling-aware bound the
    # startup pool sizer uses (single source of truth: the spec method).
    # R-SWA also recycles gap blocks but peak physical KV still fits the
    # full-attention bound (prefix + window <= max_model_len), so it inherits
    # FullAttentionSpec sizing without a separate admission cap.
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1861-L1877（准入
    #   上限注入——运行期门与启动期池大小器同源的装配点）
    if isinstance(
        kv_cache_spec,
        (SlidingWindowSpec, ChunkedLocalAttentionSpec),
    ):
        kwargs["max_admission_blocks_per_request"] = (
            kv_cache_spec.max_admission_blocks_per_request(
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
            )
        )
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1877-L1878
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1881 register_all_kvcache_specs
def register_all_kvcache_specs(vllm_config=None):
    """Built-in spec registration"""
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1883-L1887
    KVCacheSpecRegistry.register(
        FullAttentionSpec,
        FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1889-L1893
    KVCacheSpecRegistry.register(
        SlidingWindowSpec,
        SlidingWindowManager,
        uniform_type_base_spec=SlidingWindowSpec,
    )

    # SUBTRACTED: SlidingWindowMLASpec（L1894-L1898——第 4 条）、
    #   MambaSpec（L1900-L1902——mamba manager → 邻章，注册随 manager 删）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1903-L1907
    KVCacheSpecRegistry.register(
        ChunkedLocalAttentionSpec,
        ChunkedLocalAttentionManager,
        uniform_type_base_spec=ChunkedLocalAttentionSpec,
    )

    # SUBTRACTED: CrossAttentionSpec（L1908-L1912——第 5 条）；TQ/MLA/RSWA/
    #   HiddenState/SinkFull（L1915-L1938——第 4/5 条）；
    #   current_platform.register_custom_kv_cache_specs（L1940-L1942——平台域）。
