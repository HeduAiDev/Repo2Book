# SOURCE: vllm/v1/core/kv_cache_manager.py
# **对调度器的门面**（m4/m11/m14/m16/m18/m20 的入口）：get_computed_blocks
# （skip 谓词 → max_cache_hit_length=num_tokens−1 → coordinator 三元组 →
# shared_prefix_boundary 折算）、allocate_slots 三段式（ch13 已立骨架，本章
# 打开命中段：挂命中块+CoW+写回）、free（先放 pins 再逆序——pins 随 ch16 删）、
# take_kv_cache_block_copies（CoW 拷贝对 drain → KVCacheBlockCopy + retained
# 两端）、reset_prefix_cache（RLHF 失效面）。KVCacheBlocks 包装（对 Scheduler
# 隐藏内部结构 + 预构空对象防 GC）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 1 条 kv events：kv_cache_event_metadata/emit_cached_block_events 发布段
#     （L172-L178、L266-L284）与 take_events（L677-L701）；
#   第 2 条 metrics_collector 参数与透传；
#   第 3/4 条 use_eagle/dcp/pcp 参数；
#   第 5 条 connector：get_computed_blocks_for_connector（L297-L342）、
#     evict_blocks（L619-L625）、take_partial_tail_offloads（L848-L874）、
#     _partial_tail_pins（L189-L191、L575-L577、L614-L616）、
#     get_zeroing_block_ids_in_range/record_blocks_for_zeroing（L803-L829——
#     async KV load 覆写区）、get_block_ids_for_computed_tokens 的裁剪面
#     （L711-L729——ch16 消费）、estimate_cached_tokens（L731-L758）；
#   第 10 条 log_stats：prefix_cache_stats 字段与 record/make 两口、
#     reset 里的 reset 旗标（m19 概念在 stats.py 讲，调用点删）；
#   watermark/full-ISL 两道门（L168-L171、L463-L488——ch14 全章主角）、
#     reserved_blocks（L521-L523——ch16 async 载入预留）、num_lookahead_tokens
#     （ch33）；truncate_computed_blocks（L777-L794——远端命中仲裁 → ch16）。
import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from .kv_cache_coordinator import get_kv_cache_coordinator
from .kv_cache_interface import KVCacheConfig
from .kv_cache_utils import KVCacheBlock, KVCacheBlockCopy
from .request import Request, RequestStatus


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

    # SUBTRACTED: @overload 两个签名桩（L64-L74——纯 typing 细化）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L76 get_block_ids
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
        #   全组空 → None 不占带宽）
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    # SUBTRACTED: get_unhashed_block_ids 族（L93-L108——级联注意力旁路）。

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
        max_in_flight_tokens: int | None = None,
        enable_caching: bool = True,
    ) -> None:
        # SUBTRACTED: use_eagle/log_stats/enable_kv_cache_events/dcp/pcp/
        #   metrics_collector/watermark 八参数（第 1/2/3/4/10 条 + ch14 的
        #   watermark 门）。
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L134-L139
        self.max_model_len = max_model_len
        # When unset, fall back to `max_model_len` so the recycling-aware cap
        # collapses to the prior (uncapped) admission behavior. The scheduler
        # always supplies the real value at runtime.
        if max_in_flight_tokens is None:
            max_in_flight_tokens = max_model_len

        self.enable_caching = enable_caching

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L151-L166 建 coordinator
        #   （三态分派：NoPrefixCache / Unitary / Hybrid）+ 池引用
        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_in_flight_tokens=max_in_flight_tokens,
            enable_caching=self.enable_caching,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L180-L187 预构空对象
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

        # SUBTRACTED: _partial_tail_pins（L189-L191——第 5 条 connector 的
        #   off-table cow 块钉住 → ch16）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L193 usage
    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L200
        return self.block_pool.get_usage()

    # SUBTRACTED: make_prefix_cache_stats/record_prefix_cache_stats
    #   （L202-L227——第 10 条 log_stats 记录；观测概念在 stats.py 的
    #   PrefixCacheStats 里，调用点删）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L214 prefix_cache_lookup_enabled
    def prefix_cache_lookup_enabled(self, request: Request) -> bool:
        """Whether a local prefix cache lookup may be run for this request."""
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L216（关缓存或请求标跳读
        #   ——prompt logprobs/pooling）
        return self.enable_caching and not request.skip_reading_prefix_cache

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L229 get_computed_blocks
    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
                - ``shared_prefix_boundary``: the block-aligned token position of
                  a shared prefix that a sparse-retention group (Mamba / sliding
                  window) has not cached yet (Marconi-style APC), or 0 if none.
                  Pinned so ``VLLM_PREFIX_CACHE_RETENTION_INTERVAL`` does not drop
                  the junction and defeat cross-request reuse.
        """
        # We skip finding the prefix cache hit when prefix caching is
        # disabled or the request is marked as skipping kv cache read
        # (which happens when the request requires prompt logprobs
        # or calls a pooling model with all pooling).
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L246-L251 skip 谓词
        if not self.prefix_cache_lookup_enabled(request):
            return self.empty_kv_cache_blocks, 0, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L253-L259 max_cache_hit_length
        #   （全命中退一 token 拿 logits；块对齐可能回退整块）
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens, num_uncached = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        # SUBTRACTED: kv_cache_report_mode='full' 的 BlockStored 发布段
        #   （L266-L284——第 1 条 kv events 观测旁路）。

        # The junction to pin is where the lagging sparse-retention group stops
        # (``num_new_computed_tokens``) plus the uncached shared prefix -- i.e.
        # the longest single-group hit. Sub-block gaps are left to the mask,
        # which floors to the alignment boundary (a no-op there).
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L286-L292（junction 折算：
        #   命中长 + 未缓共享前缀 = 最长单组命中——Marconi 钉住的写回值）
        shared_prefix_boundary = (
            num_new_computed_tokens + num_uncached if num_uncached else 0
        )

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L294-L295
        blocks = self.create_kv_cache_blocks(computed_blocks)
        return blocks, num_new_computed_tokens, shared_prefix_boundary

    # SUBTRACTED: get_computed_blocks_for_connector（L297-L342——第 5 条
    #   connector 的混合发散回退半边 → ch16）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L344 allocate_slots
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        has_scheduled_reqs: bool = True,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of new tokens to be allocated and computed.
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens, grouped as a tuple by kv cache groups.
            num_external_computed_tokens: The number of tokens that their
                KV caches are not cached by vLLM but cached by the connector.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.
            num_encoder_tokens: The number of encoder tokens to allocate for
                cross-attention in encoder-decoder models(e.g., Whisper).
                For decoder-only models, this should be 0.
            has_scheduled_reqs: Whether any requests are already scheduled to run
                this step, controls whether watermark is applied.

        Returns:
            A list of new allocated blocks.
        """
        # SUBTRACTED: num_lookahead_tokens（ch33）、full_sequence_must_fit/
        #   reserved_blocks 两道门（ch14/ch16）与 watermark 应用段（L463-L470、
        #   L521-L524——本章无水印）；Blocks layout 图与三段式说明 docstring
        #   的 lookahead/external 列随参数面减。
        # When loading KV data asynchronously, we may have zero new tokens to
        # compute while still allocating slots for externally computed tokens.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L440-L446 零 token 护栏
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L448-L451
        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L453-L461 本地命中合账
        #   （external 项恒 0 保留算位）
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L490-L493 主模型 token 面
        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model, self.max_model_len
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
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L495-L508 先回收窗外块
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
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_local_computed_tokens=num_local_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L525-L527 容量检查（不够
        #   None——ch11 抢占唯一触发信号的内因）
        required_blocks = num_blocks_to_allocate
        if required_blocks > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L529-L547 挂命中块 + 分
        #   新块（CoW 换尾发生在 manager 内部）
        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L549-L552（delay_cache_
        #   blocks 参数占位保留——ch16 的 P/D 早退；关缓存也走这条）
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L554-L563 写回（封顶在
        #   request.num_tokens——只缓存已定案的 token）
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L567 free
    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        # SUBTRACTED: pins 先放段（L575-L577——_partial_tail_pins 随第 5 条删）
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L578（注释原话『tail blocks
        #   are evicted first』；coordinator → 逐 manager 逆序）
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
        # SUBTRACTED: pins 拼接段（L614-L616——第 5 条）。
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L610（先摘账后还块——
        #   延迟释放面用）
        return self.coordinator.pop_blocks_for_free(request.request_id)

    # SUBTRACTED: evict_blocks（L619-L625——第 5 条 connector 按号驱逐）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L627 reset_prefix_cache
    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L636-L641（log_stats 的
        #   reset 旗标行随第 10 条删——权重变了缓存必须整体作废）
        if not self.block_pool.reset_prefix_cache():
            return False
        return True

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L643 get_num_common_prefix_blocks
    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Calculate the number of common prefix blocks for each kv cache group.

        The function selects a running request and iterates through its blocks.
        A block is considered a common prefix block if ALL requests with
        allocated KV cache share it (i.e., ref_cnt equals the number of entries
        in req_to_blocks).

        NOTE(woosuk): The number of requests with allocated KV cache is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because having allocated KV cache only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must have allocated KV cache, the inverse
        is not necessarily true. There may be requests with allocated KV cache
        that are not scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled requests that do not share the
        common prefix. Currently, this case cannot be easily detected, so the
        function returns 0 in such cases.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache
            group.
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L675
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    # SUBTRACTED: take_events（L677-L701——第 1 条 kv events）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L703 get_blocks
    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request."""
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L705
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L707 get_block_ids
    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L709
        return self.get_blocks(request_id).get_block_ids()

    # SUBTRACTED: get_block_ids_for_computed_tokens/estimate_cached_tokens/
    #   truncate_computed_blocks（L711-L794——ch16 的仲裁/裁剪与观测面）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L760 cache_blocks
    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled.

        Args:
            request: The request to cache the blocks.
            num_computed_tokens: The number of computed tokens, including tokens
                that are already cached and tokens to be cached.
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L768-L769
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L771 create_kv_cache_blocks
    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        # Only create new KVCacheBlocks for non-empty blocks
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L775
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L796 take_new_block_ids
    def take_new_block_ids(self) -> list[int]:
        """Drain and return new attention block IDs for zeroing."""
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L798-L801
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            ids.extend(mgr.take_new_block_ids())
        return ids

    # SUBTRACTED: get_zeroing_block_ids_in_range/record_blocks_for_zeroing
    #   （L803-L829——async KV load 覆写区跳过/重记，第 5 条 → ch16）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L831 take_kv_cache_block_copies
    def take_kv_cache_block_copies(
        self,
    ) -> tuple[list[KVCacheBlockCopy], list[KVCacheBlock]]:
        """Drain pending copies and return their retained endpoints."""
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L834-L846（drain 各 manager
        #   的 (source, cow) 对 → 转块号对跨进程负载 + retained 两端——
        #   拷完前不许回收）
        pending_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
        for mgr in self.coordinator.single_type_managers:
            pending_copies.extend(mgr.take_pending_cow_copies())
        copies = [
            KVCacheBlockCopy(
                src_block_id=source_block.block_id,
                dst_block_id=cow_block.block_id,
            )
            for source_block, cow_block in pending_copies
        ]
        retained_blocks = [block for pair in pending_copies for block in pair]
        return copies, retained_blocks

    # SUBTRACTED: take_partial_tail_offloads（L848-L874——第 5 条 connector 的
    #   producer 手递手 + 钉住 → ch16）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L876 new_step_starts
    def new_step_starts(self) -> None:
        """Notify the coordinator that a new step is starting."""
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L878
        self.coordinator.new_step_starts()
