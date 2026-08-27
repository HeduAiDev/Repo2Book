# SOURCE: vllm/v1/core/kv_cache_manager.py
# KVCacheManager——对 Scheduler 的门面（m6 绝对主角）：allocate_slots 三段式
# （容量检查→挂命中块→分新块→写回满块）、free 终局还块、take_new_block_ids
# 清零通道；KVCacheBlocks 包装专门隐藏内部结构（docstring 原话）。
# 本章精简版跑 enable_prefix_caching=False：allocate_slots 在 "not enable_
# caching" 早退、不进 cache_blocks；前缀命中在精简版恒 0（get_computed_blocks
# 哈希侧 → ch15）。
# SUBTRACTED（dossier.delete 批准项的落点）：
#   第 1 条 KV cache events：enable_kv_cache_events/kv_cache_event_metadata/
#     take_events；
#   第 2 条 metrics/log_stats/prefix_cache_stats；
#   第 3 条 哈希侧：get_computed_blocks(+_for_connector)/make_prefix_cache_
#     stats/record_prefix_cache_stats；
#   第 5 条 use_eagle；
#   第 6 条 DCP/PCP 乘子；
#   第 7 条 connector：num_external_computed_tokens/allocate_external_
#     computed_blocks 调用/delay_cache_blocks/reserved_blocks/_partial_tail_
#     pins/take_partial_tail_offloads/evict_blocks；
#   第 8 条 full_sequence_must_fit 准入闸与 watermark 分支（L463-L488 → ch14
#     ——保留普通容量检查 L510-L527 即闭合；watermark_blocks 字段账位保留）；
#   第 9 条 CoW：take_kv_cache_block_copies/kv_cache_block_copies；
#   第 11 条 reset_prefix_cache/get_num_common_prefix_blocks/get_block_ids_
#     for_computed_tokens/get_unhashed_block_ids(_all_groups)。
import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from .kv_cache_coordinator import get_kv_cache_coordinator
from .kv_cache_interface import KVCacheConfig
from .kv_cache_utils import KVCacheBlock
from .math_utils import cdiv
from .request import Request


# SOURCE: vllm/v1/core/kv_cache_manager.py:L32 KVCacheBlocks
@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L40-L53
    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    `blocks[i][j]` refers to the i-th kv_cache_group
    and the j-th block of tokens.We don't use block of
    tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of blocks, which is true for now but
    will be broken if we want to give different block_size to different
    kv_cache_groups in the future.

    Each single type KVCacheBlocks could be represented as:
    - list[KVCacheBlock] for more than one KVCacheBlock
    - an empty tuple for requests without KVCacheBlock
      (a precomputed KVCacheBlocks is in KVCacheManager to avoid GC overhead)
    """

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L55 __add__
    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L57-L62
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    # SUBTRACTED: @overload 两个签名桩（L64-L74——纯 typing 细化
    #   （Literal[False]/Literal[True] 两型）；实现签名下方逐字保留）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L76 get_block_ids 实现
    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where:
                - the outer tuple corresponds to KV cache groups
                - each inner list contains the block_ids of the blocks in that
                  group
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L89-L91（allow_none=True 且
        #   全组空 → None 不占带宽——在跑请求增量打包的依据）
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    # SUBTRACTED: get_unhashed_block_ids / get_unhashed_block_ids_all_groups
    #   （L93-L108——第 11 条，级联前缀旁路）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L110 new_empty
    def new_empty(self) -> "KVCacheBlocks":
        """
        Creates a new KVCacheBlocks instance with no blocks.
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L114
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


# SOURCE: vllm/v1/core/kv_cache_manager.py:L117 KVCacheManager
class KVCacheManager:
    # SOURCE: vllm/v1/core/kv_cache_manager.py:L118 __init__
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        scheduler_block_size: int,
        hash_block_size: int,
        enable_caching: bool = True,
        watermark: float = 0.0,
    ) -> None:
        # SUBTRACTED: max_in_flight_tokens/use_eagle/log_stats/enable_kv_cache_
        #   events/dcp_world_size/pcp_world_size/metrics_collector 参数与
        #   prefix_cache_stats 字段（L124-L132、L138-L149——第 1/2/4/5/6 条；
        #   max_in_flight_tokens 是 SWA/chunked-local 准入上限的输入 → ch14）。
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L134
        self.max_model_len = max_model_len

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L141
        self.enable_caching = enable_caching

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L151-L163 协调器装配
        #   （False → NoPrefixCache 源码原生路径）
        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.enable_caching,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L164-L166
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

        # Watermark: minimum number of KV cache blocks to keep free when
        # admitting waiting/preempted requests, to avoid frequent preemptions.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L168-L171（字段账位保留；
        #   其消费分支在 allocate_slots 的 watermark 段——第 8 条 → ch14）
        assert watermark >= 0.0, "watermark must be non-negative"
        self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
        # SUBTRACTED: kv_cache_event_metadata（L172-L178——第 11 条）。

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L180-L187 预构空对象避免 GC
        #   （WC2 物证）
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

        # SUBTRACTED: _partial_tail_pins（L189-L191——第 7 条）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L193 usage
    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L200
        return self.block_pool.get_usage()

    # SUBTRACTED: make_prefix_cache_stats / prefix_cache_lookup_enabled /
    #   record_prefix_cache_stats / get_computed_blocks / get_computed_blocks_
    #   for_connector（L202-L342——第 2/3 条：前缀查表与统计旁路 → ch15；
    #   本章前缀命中恒 0，调度侧拿 empty_kv_cache_blocks）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L344 allocate_slots —— 三段式
    #   总控（本章绝对主角）
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_encoder_tokens: int = 0,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of new tokens to be allocated and computed.
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens, grouped as a tuple by kv cache groups.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            num_encoder_tokens: The number of encoder tokens to allocate for
                cross-attention in encoder-decoder models(e.g., Whisper).
                For decoder-only models, this should be 0.

        Returns:
            A list of new allocated blocks.
        """
        # SUBTRACTED: num_external_computed_tokens / delay_cache_blocks /
        #   full_sequence_must_fit / reserved_blocks / has_scheduled_reqs 五参
        #   与 docstring 的 Blocks layout 五段长图（L350-L356、L390-L438——第
        #   7/8 条：connector 在途 → ch16、水位与整序列准入门 → ch14、五段图
        #   里的 <ext_comp>/<lookahead> 段对单请求无 spec 无 connector 恒 0）。
        # When loading KV data asynchronously, we may have zero new tokens to
        # compute while still allocating slots for externally computed tokens.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L440-L446 零 token 护栏
        if num_new_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L448-L451（缺省用预构空对象）
        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L453-L461 本地已算 token 合账
        #   （external 半边删——第 7 条，无 connector 恒 0）
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(num_local_computed_tokens, self.max_model_len)

        # SUBTRACTED: watermark 分支（L463-L470——waiting/preempted 才计水位
        #   块位 → ch14）与 full_sequence_must_fit 整序列准入门（L472-L488
        #   ——第 8 条 → ch14）。

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L490-L493
        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens, self.max_model_len
        )

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        # Free on the processed-token basis: in-flight steps' attention windows
        # still read blocks below the optimistic boundary, and rejected spec
        # tokens can roll it back.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L495-L508（单组全注意力下
        #   no-op——滑窗外块回收 → ch14）
        self.coordinator.remove_skipped_blocks(
            request.request_id,
            max(0, total_computed_tokens - request.num_in_flight_tokens),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L510-L519 需块预测
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens,
            num_local_computed_tokens=num_local_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        # Keep `reserved_blocks` free for other in-flight sequences, and an
        # additional watermark of headroom for waiting/preempted admissions.
        # SUBTRACTED: reserved_blocks 扣减与 watermark headroom（L521-L524
        #   ——第 7/8 条；本章容量检查只看需块 vs 空闲）。
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L525-L527 容量检查（不够
        #   None——ch10「拿不到块 break」与 ch11「抢占唯一触发信号」的内因）
        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L529-L540 挂命中块（touch
        #   → ch15；本章 False 支不触发——empty 判同短路）
        if new_computed_block_list is not self.empty_kv_cache_blocks.blocks:
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
            )

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L542-L547 分新块（本章主角）
        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        # SUBTRACTED: delay_cache_blocks（L552——P/D 延迟缓存 → ch16）。
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L551-L552（caching 关 →
        #   早退不进 cache_blocks——本章 False 支的控制流闭合点）
        if not self.enable_caching:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L554-L563 写回满块（哈希
        #   登记内部 → ch15）
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L567 free —— 终局还块（门面侧）
    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        # SUBTRACTED: _partial_tail_pins 放钉（L575-L577——第 7 条）。
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L578
        self.coordinator.free(request.request_id)

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L580 remove_skipped_blocks
    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length for R-SWA gap eviction.
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L595-L597
        self.coordinator.remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens
        )

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L599 pop_blocks_for_free
    def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:
        """Pop the request's bookkeeping and return its blocks without
        returning them to the block pool. The caller must eventually free
        them in reverse order (so that tail blocks are evicted first).

        Args:
            request: The request to pop the blocks for.

        Returns:
            The request's blocks in allocation order.
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L610（deferred free 的半边
        #   ——async 调度先摘账等步序栅栏，ch12）
        # SUBTRACTED: _partial_tail_pins 随行（L611-L616——第 7 条）。
        return self.coordinator.pop_blocks_for_free(request.request_id)

    # SUBTRACTED: evict_blocks（L619-L625——第 7 条）、reset_prefix_cache
    #   （L627-L641——第 11 条）、get_num_common_prefix_blocks（L643-L675
    #   ——第 11 条）、take_events（L677-L701——第 1 条）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L703 get_blocks
    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request."""
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L705
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L707 get_block_ids
    def get_block_ids(self, request_id: str) -> "tuple[list[int], ...]":
        """Get the block ids of a request."""
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L709
        return self.get_blocks(request_id).get_block_ids()

    # SUBTRACTED: get_block_ids_for_computed_tokens（L711-L729——第 11 条）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L731 estimate_cached_tokens
    def estimate_cached_tokens(self, request: Request) -> int:
        """Estimate the number of tokens cached by the request."""
        # SUBTRACTED: CrossAttention/EncoderOnly 组跳过（L738-L743——第 4 条；
        #   本章全组都是注意力组）。
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L734-L758（按 block_hash_
        #   num_tokens 取最深账位——本章哈希恒空 → 0）
        cached_tokens: int | None = None
        for group, blocks in zip(
            self.kv_cache_config.kv_cache_groups,
            self.get_blocks(request.request_id).blocks,
        ):
            group_cached_tokens = 0
            for block in blocks:
                group_cached_tokens = max(
                    group_cached_tokens,
                    block.block_hash_num_tokens or 0,
                )

            cached_tokens = (
                group_cached_tokens
                if cached_tokens is None
                else min(cached_tokens, group_cached_tokens)
            )

        return cached_tokens or 0

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L760 cache_blocks ——『写回满块』
    #   调用点（三段式第四段；内部哈希登记 → ch15）
    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled.

        Args:
            request: The request to cache the blocks for.
            num_computed_tokens: The number of computed tokens, including tokens
                that are already cached and tokens to be cached.
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L768-L770
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L771 create_kv_cache_blocks
    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        # Only create new KVCacheBlocks for non-empty blocks
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L774-L775（空结果复用预构
        #   对象）
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks

    # SUBTRACTED: truncate_computed_blocks（L777-L794——connector 局部命中
    #   截视图，第 7 条 → ch16）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L796 take_new_block_ids —— 清零
    #   通道（排干新块 id 给 worker）
    def take_new_block_ids(self) -> list[int]:
        """Drain and return new attention block IDs for zeroing."""
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L798-L801
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            ids.extend(mgr.take_new_block_ids())
        return ids

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L803 get_zeroing_block_ids_in_range
    def get_zeroing_block_ids_in_range(
        self, request_id: str, start_token: int, end_token: int
    ) -> list[int]:
        """The request's block ids covering [start_token, end_token), from
        the groups whose new blocks are zeroed by the worker."""
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L808-L815
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            if mgr.records_new_block_ids:
                start_idx = start_token // mgr.block_size
                end_idx = cdiv(end_token, mgr.block_size)
                blocks = mgr.req_to_blocks[request_id]
                ids.extend(blk.block_id for blk in blocks[start_idx:end_idx])
        return ids

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L817 record_blocks_for_zeroing
    def record_blocks_for_zeroing(self, request_id: str, start_token: int) -> None:
        """Re-record the request's blocks from start_token onwards for
        zeroing, e.g. blocks a failed async KV load left unwritten.

        start_token must be block-aligned: zeroing a partially-valid block
        would wipe its valid prefix.
        """
        # SUBTRACTED: async KV load 失败补登记的场景接线 → ch16；方法本体
        #   逐字保留（清零通道的正交补口）。
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L824-L829
        for mgr in self.coordinator.single_type_managers:
            if mgr.records_new_block_ids:
                assert start_token % mgr.block_size == 0
                start_idx = start_token // mgr.block_size
                blocks = mgr.req_to_blocks[request_id]
                mgr.new_block_ids.extend(blk.block_id for blk in blocks[start_idx:])

    # SUBTRACTED: take_kv_cache_block_copies（L831-L846——第 9 条 CoW）与
    #   take_partial_tail_offloads（L848-L874——第 7 条）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L876 new_step_starts
    def new_step_starts(self) -> None:
        """Notify the coordinator that a new step is starting."""
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L878（事件段删——第 11 条）
        self.coordinator.new_step_starts()
