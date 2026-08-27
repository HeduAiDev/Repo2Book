# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py
# SingleTypeKVCacheManager / FullAttentionManager——每注意力类型一份块账本
# （m5/m6/m11/m12 的 manager 半边）：req_to_blocks 逻辑块表、num_cached_block
# 入口账位、get_num_blocks_to_allocate 需块预测（预测器与分配器同构——
# None→抢占 的账本依据）、allocate_new_blocks 拿块挂账、take_new_block_ids
# 供清零、free 逆序还块。
# SUBTRACTED（dossier.delete 批准项的落点）：
#   第 4 条 混合/多组家族：SlidingWindowManager/ChunkedLocalAttentionManager/
#     MambaManager/RSWAManager/CrossAttentionManager/SinkFullAttentionManager
#     （L832-L1833）与 reachable_block_mask/reachable 驻留、remove_skipped_
#     blocks 子类实现与 get_num_skipped_tokens 子类覆写、admission cap
#     （_max_admission_blocks_per_request/apply_admission_cap）、register_all_
#     kvcache_specs 注册表（L1881-L1942）；
#   第 3 条 哈希侧：find_longest_cache_hit 基类抽象 + FullAttention 覆写
#     （L545-L593 / L681-L777）、BlockHashList(WithBlockSize) 消费；
#   第 5 条 eagle：use_eagle 装配与引用；
#   第 6 条 DCP/PCP：dcp_world_size/pcp_world_size 乘子（单卡恒 1 烘干）；
#   第 7 条 connector：allocate_external_computed_blocks（L291-L328）与
#     _pending_partial_tail_offloads/take_pending_partial_tail_offloads；
#   第 9 条 CoW/partial-hit：_partial_hit_reqs/_pending_cow_copies/_apply_cow/
#     take_pending_cow_copies/_has_partial_local_hit；
#   第 11 条 get_num_common_prefix_blocks（级联注意力旁路）。
from abc import ABC
from collections import defaultdict
from collections.abc import Sequence
from typing import ClassVar

from .block_pool import BlockPool
from .kv_cache_interface import FullAttentionSpec, KVCacheSpec
from .kv_cache_utils import KVCacheBlock
from .math_utils import cdiv
from .request import Request


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L36 SingleTypeKVCacheManager
class SingleTypeKVCacheManager(ABC):
    """
    An abstract base class for a manager that handle the kv cache management
    logic of one specific attention type.
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
        """
        # SUBTRACTED: max_admission_blocks_per_request 参数（L54 与 L66-L71
        #   docstring——SWA/chunked-local 回收型准入门，第 4 条 → ch14）与
        #   dcp_world_size/pcp_world_size 乘子（L51-L52、L76-L79——第 6 条，
        #   单卡恒 1 烘干：block_size 不放大）。
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L73-L75
        self.scheduler_block_size = scheduler_block_size
        # The block size for this manager; used for actual block allocation.
        self.block_size = kv_cache_spec.block_size
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L80-L83
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.enable_caching = enable_caching
        # Record newly allocated block ids only when worker-side zeroing will
        # consume them and this manager holds a spec type that gets zeroed.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L86-L92（零清
        # 类型表缩到本章的 FullAttentionSpec——TQ/MLA/HiddenState spec 族
        # → ch14）
        self._record_new_block_ids = needs_kv_cache_zeroing and type(kv_cache_spec) in (
            FullAttentionSpec,
        )
        self.new_block_ids: list[int] = []

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L94-L97（逻辑
        # 块表本体——全章的轴）
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for preempted ones.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L99-L103
        self.num_cached_block: dict[str, int] = {}

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L105-L106
        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

        # SUBTRACTED: use_eagle（L108-L112——第 5 条，→ ch33）、_partial_hit_
        #   reqs/_pending_cow_copies（L114-L117——第 9 条）、_pending_partial_
        #   tail_offloads（L118-L126——第 7 条）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L128 _get_num_evictable_blocks
    @classmethod
    def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L130
        return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)

    # SUBTRACTED: _has_partial_local_hit（L132-L142——细粒度部分命中判定，
    #   第 9 条 → ch15）。

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L144 get_num_blocks_to_allocate
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
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

        Returns:
            The number of blocks to allocate.
        """
        # SUBTRACTED: apply_admission_cap 参数与 cap 分支（L152/L170-L172/
        #   L179-L191——回收型 spec 的每请求上限，第 4 条 → ch14）。

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L178 cdiv 主算术
        num_required_blocks = cdiv(num_tokens, self.block_size)
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

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L202-L218（滑窗
        # 外跳块对单组全注意力恒 0——get_num_skipped_tokens 基类返回 0）
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
        # SUBTRACTED: partial-hit CoW 预留 +1（L226-L229——第 9 条 → ch15）。
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L230
        return num_new_blocks + num_evictable_blocks

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L232 add_local_computed_blocks
    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
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
            num_external_computed_tokens: The number of external computed tokens.
        """
        # SUBTRACTED: num_external_computed_tokens 合账半边（L257-L261 的
        #   external 部分——第 7 条；本章恒 0，skipped 头部裁剪对全注意力为 0）。
        # The coordinator only calls this for first-time allocations (running
        # requests are short-circuited there), so the request has no blocks yet.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L253-L256
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        # SUBTRACTED: skipped 块 null 填充（L260-L265——滑窗族，第 4 条）。

        # Touch the computed blocks to make sure they won't be evicted.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L267-L273
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), (
                "Computed blocks should be empty when prefix caching is disabled"
            )

        # SUBTRACTED: skipped 块 null 填充（L275-L276——第 4 条）。
        # Add the remaining computed blocks.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L277-L278
        req_blocks.extend(new_computed_blocks)
        # All cached hits (including skipped nulls) are already cached; mark
        # them so cache_blocks() will not try to re-cache blocks that already
        # have a block_hash set.
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L279-L282
        self.num_cached_block[request_id] = len(req_blocks)
        # SUBTRACTED: partial-hit 尾块登记（L283-L289——第 9 条 → ch15）。

    # SUBTRACTED: allocate_external_computed_blocks（L291-L328——connector
    #   外部已算 token 的补块，第 7 条 → ch16；无 connector 时 num_external_
    #   computed_tokens 恒 0 不触发）。

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
        # SUBTRACTED: partial-hit CoW 重定向前缀（L347-L357——共享尾块原地换
        #   私有 cow_block，第 9 条 → ch15）。
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L359-L369 差值
        #   分配主干（req_blocks.extend 挂账 + new_block_ids 记录）
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

    # SUBTRACTED: take_pending_cow_copies（L382-L388）与 take_pending_partial_
    #   tail_offloads（L390-L403）、_apply_cow（L405-L425）——第 9/7 条。

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
            num_tokens: The total number of tokens that need to be cached
                (including tokens that are already cached).
        """
        # SUBTRACTED: retention_interval 参数（L431 与 L440-L443 docstring——
        #   SWA/Mamba 稀疏驻留粒度，第 4 条）。
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L445-L449 幂等
        #   闸（num_cached_block 避免重复缓存）
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        if num_cached_blocks >= num_full_blocks:
            return

        # SUBTRACTED: reachable_boundaries/block_mask 稀疏驻留掩码（L451-L466
        #   ——reachable_block_mask 族，第 4 条）与 block_pool.cache_full_blocks
        #   满块哈希登记（L467-L475——第 3 条 → ch15）。调用点与账位保留：本章
        #   讲「写回满块」这一步的位置，哈希登记内景归 ch15。

        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L477 入口账位
        #   推进（已缓存满块数）
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
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L526-L527 逆序
        #   还块（"tail blocks are freed first"——LRU 不变量的半边，ch15）
        self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))

    # SUBTRACTED: get_num_common_prefix_blocks 抽象（L529-L543）与
    #   find_longest_cache_hit 抽象（L545-L593）——第 11/3 条（级联注意力
    #   旁路 / 哈希命中 → ch15）。

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
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L606-L620
        if request_id not in self.req_to_blocks:
            return
        if first_block >= last_block:
            return
        blocks = self.req_to_blocks[request_id]
        last_block = min(last_block, len(blocks))

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
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L643-L659（单组
        #   全注意力下 num_skipped_tokens==0 → 早退 no-op；滑窗回收 → ch14）
        del num_prompt_tokens
        # Remove the blocks that will be skipped during attention computation.
        num_skipped_tokens = self.get_num_skipped_tokens(processed_computed_tokens)
        if num_skipped_tokens <= 0:
            # This indicates that ALL tokens are inside attention window.
            # Thus, we do not need to free any blocks outside attention window.
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
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L672
        return 0

    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L674 new_step_starts
    def new_step_starts(self) -> None:
        # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L675
        return None


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L678 FullAttentionManager
class FullAttentionManager(SingleTypeKVCacheManager):
    # SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L679
    supports_fine_grained_hash_lookup: ClassVar[bool] = True

    # SUBTRACTED: find_longest_cache_hit（L681-L777——链式哈希逐块命中 + 细粒度
    #   内部边界探测，第 3 条 → ch15）；cache_blocks 覆写与 _cache_partial_
    #   tail_block（L779-L819——hash_block_size≠block_size 的部分尾块登记，
    #   → ch15；基类 cache_blocks 账位语义已足）；get_num_common_prefix_blocks
    #   （L821-L829——第 11 条，级联注意力）。
    # 单组全注意力主路径下本类与基类行为一致（blocks are allocated for all
    # tokens——L236-L243 类 docstring 的语义）。


# SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1836 get_manager_for_kv_cache_spec
def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    max_model_len: int,
    **kwargs,
) -> SingleTypeKVCacheManager:
    """
    Get the appropriate manager for a given KVCacheSpec.

    Args:
        kv_cache_spec: The KVCacheSpec instance
        max_model_len: The maximum context length the model could serve
    Returns:
        An instance of the appropriate SingleTypeKVCacheManager subclass
    """
    # SUBTRACTED: KVCacheSpecRegistry 查表（L1857-L1877——自定义 spec 注册族，
    #   第 4 条）与 SWA/chunked-local 准入上限实参（L1861-L1876 → ch14）；
    #   max_in_flight_tokens 参数（准入上限专用）。按 register_all_kvcache_
    #   specs（L1883-L1888）的 FullAttentionSpec→FullAttentionManager 对应
    #   直连——本章锁定单组全注意力主路径。
    assert isinstance(kv_cache_spec, FullAttentionSpec), (
        "ch13 精简版只装配 FullAttentionSpec（多组/滑窗/Mamba → ch14）"
    )
    manager = FullAttentionManager(kv_cache_spec, **kwargs)
    return manager
