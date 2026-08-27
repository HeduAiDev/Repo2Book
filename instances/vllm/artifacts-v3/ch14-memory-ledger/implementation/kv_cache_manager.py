# SOURCE: vllm/v1/core/kv_cache_manager.py
# KVCacheManager——对调度器的门面 + 本章的两道门（m10/m12）：
# watermark_blocks = watermark × num_blocks（L168-L171，精修版只对
# WAITING/PREEMPTED 且 has_scheduled_reqs 生效）；allocate_slots 的
# full-ISL 门（L472-L488，整序列预留 vs 只查第一 chunk 的超收——#39734
# 第一半根治）→ remove_skipped_blocks 先回收窗外块 → required ≤ free −
# reserved − watermark（L510-L527，None = ch11 抢占信号）。KVCacheBlocks
# 包装（对 Scheduler 隐藏内部结构 + 预构空对象防 GC）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 13 条 lookahead/encoder/external(connector)/delay 参数分支
#     （L350-L352、L367-L377、L419、L458-L461 的 external 合账、
#     L490-L493 的 lookahead 封顶、L531-L540 的 external 挂块——
#     参数面以账位保留或整删，控制流主干不变）；
#   第 9 条 events/metrics/log_stats 观测面（L127-L131、L142-L149、
#     L172-L178、L266-L284）；
#   第 6 条 eagle：use_eagle 装配（L126、L143、L155）；
#   第 8 条 DCP/PCP 乘子透传（L129-L130、L158-L159）；
#   哈希侧 get_computed_blocks（L229-L295——前缀命中 → ch15；调用面在
#     scheduler，本章站点不触）与 get_computed_blocks_for_connector
#     （L297-L342——ch16）；
#   _partial_tail_pins（L189-L191、L575-L577——ch16）。
import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from .kv_cache_coordinator import get_kv_cache_coordinator
from .kv_cache_interface import KVCacheConfig
from .kv_cache_utils import KVCacheBlock
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

    # SUBTRACTED: get_unhashed_block_ids 族（L93-L108——级联旁路）。

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
        watermark: float = 0.0,
    ) -> None:
        # SUBTRACTED: use_eagle/log_stats/enable_kv_cache_events/dcp_world_
        #   size/pcp_world_size/metrics_collector 参数与 prefix_cache_stats
        #   字段（L126-L132、L143-L149——第 6/8/9 条）。
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L134
        self.max_model_len = max_model_len
        # When unset, fall back to `max_model_len` so the recycling-aware cap
        # collapses to the prior (uncapped) admission behavior. The scheduler
        # always supplies the real value at runtime.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L138-L139
        if max_in_flight_tokens is None:
            max_in_flight_tokens = max_model_len

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L141
        self.enable_caching = enable_caching

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L151-L163 协调器装配
        #   （False → NoPrefixCache 源码原生路径——支持任意组数）
        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_in_flight_tokens=max_in_flight_tokens,
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
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L168-L171（水位块数 =
        #   watermark × num_blocks；默认 0.0 即关）
        assert watermark >= 0.0, "watermark must be non-negative"
        self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
        # SUBTRACTED: kv_cache_event_metadata（L172-L178——第 9 条）。

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L180-L187 预构空对象避免 GC
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

        # SUBTRACTED: _partial_tail_pins（L189-L191——ch16）。

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
    #   record_prefix_cache_stats（L202-L227——第 9 条观测 + ch15）；
    #   get_computed_blocks（L229-L295——前缀命中 → ch15）；
    #   get_computed_blocks_for_connector（L297-L342——ch16）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L344 allocate_slots
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
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
            full_sequence_must_fit: Only allocate blocks if the KV cache has enough
                free blocks to hold the full sequence, accounting for prefix cache hits
                and sliding window. Used as an admission gate to prevent over-admitting
                requests when chunked prefill would otherwise only check the first chunk
            reserved_blocks: Number of free blocks that must be left available for
                other in-flight sequences to complete. The actual allocation is only
                made if it fits within (free blocks - reserved_blocks). Used to gate
                async KV-connector loads so their initial allocation cannot consume
                blocks an already in-flight (prefilling) sequence is relying on.
            has_scheduled_reqs: Whether any requests are already scheduled to run
                this step, controls whether watermark is applied.

        Blocks layout:
        ```
        ----------------------------------------------------------------------
        | < comp > | < new_comp > | < new > |
        ----------------------------------------------------------------------
                                    |   < to be computed >                 |
        ----------------------------------------------------------------------
                                    |            < to be allocated>        |
        ----------------------------------------------------------------------
        ```

        Abbrivations:

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        new       = num_new_tokens
        ```

        The allocation has three stages:
        - Free unnecessary blocks in `comp` and check
           if we have sufficient free blocks (return None if not).
        - Handle prefix tokens (`comp + new_comp`):
            - Free unnecessary blocks (e.g., outside sliding window)
        - Allocate new blocks for tokens to be computed (`new`)

        Returns:
            A list of new allocated blocks.
        """
        # When loading KV data asynchronously, we may have zero new tokens to
        # compute while still allocating slots for externally computed tokens.
        # SUBTRACTED: external 半边（第 13 条 → ch16——本章 num_new_tokens=0
        #   即 raise）
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L442-L446 零 token 护栏
        if num_new_tokens == 0:
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
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L453-L457（external 合账
        #   半边随第 13 条删）
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L458-L461
        total_computed_tokens = min(
            num_local_computed_tokens,
            self.max_model_len,
        )

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L463-L470 水位条件计入
        #   （精修版：只对 WAITING/PREEMPTED 且本步已有调度——非 v0 全局
        #   静态垫片）
        watermark_blocks = 0
        # The watermark is applied to waiting/preempted requests only, and only
        # when there's at least one request already scheduled.
        if has_scheduled_reqs and request.status in (
            RequestStatus.WAITING,
            RequestStatus.PREEMPTED,
        ):
            watermark_blocks = self.watermark_blocks

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L472-L488 full-ISL 准入门
        #   （整序列算块（含 cap）对比 free——chunked prefill 只查第一 chunk
        #   的超收漏洞由这道门堵上）
        if full_sequence_must_fit:
            # First check and fail if the full request sequence won't fit.
            full_num_tokens = min(request.num_tokens, self.max_model_len)

            num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=full_num_tokens,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=0,  # SUBTRACTED: encoder 实参（第 13 条，恒 0）
                total_computed_tokens=total_computed_tokens,
                num_local_computed_tokens=num_local_computed_tokens,
                num_tokens_main_model=full_num_tokens,
                apply_admission_cap=True,
            )
            required_blocks = num_blocks_to_allocate + watermark_blocks
            if required_blocks > self.block_pool.get_num_free_blocks():
                return None

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L490-L493（lookahead 封顶
        #   随第 13 条删——num_tokens_need_slot = min(main, max_len)）
        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(num_tokens_main_model, self.max_model_len)

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
        #   （processed-token 基准：在途步还在读的块不收）
        self.coordinator.remove_skipped_blocks(
            request.request_id,
            max(0, total_computed_tokens - request.num_in_flight_tokens),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L510-L519
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=0,  # SUBTRACTED: encoder 实参（第 13 条，恒 0）
            total_computed_tokens=num_local_computed_tokens,
            num_local_computed_tokens=num_local_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        # Keep `reserved_blocks` free for other in-flight sequences, and an
        # additional watermark of headroom for waiting/preempted admissions.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L521-L527 稳态门（required
        #   ≤ free − reserved，否则 None = ch11 抢占信号的出生地）
        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
        required_blocks = num_blocks_to_allocate + watermark_blocks
        if required_blocks > available_blocks:
            # Cannot allocate new blocks
            return None

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L529-L540（命中块挂账；
        #   external 分配半边随第 13 条删）
        if new_computed_block_list is not self.empty_kv_cache_blocks.blocks:
            # Append the new computed blocks to the request blocks until now to
            # avoid the case that the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
            )

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L542-L547
        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
        )

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L549-L552（delay_cache_
        #   blocks 参数随第 13 条删；本章 caching 关早退 / 开则写回账位）
        if not self.enable_caching:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # + num_new_tokens, but must exclude "non-committable" tokens (e.g.,
        # draft tokens that could be rejected). Therefore, we cap the number
        # at `request.num_tokens`, ensuring only "finalized" tokens are cached.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L554-L563
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L565
        return self.create_kv_cache_blocks(new_blocks)

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L567 free
    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks for.
        """
        # SUBTRACTED: _partial_tail_pins 释放（L575-L577——ch16）。
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

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L771 create_kv_cache_blocks
    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        # Only create new KVCacheBlocks for non-empty blocks
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L775
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks

    # SUBTRACTED: truncate_computed_blocks / take_new_block_ids /
    #   get_zeroing_block_ids_in_range / record_blocks_for_zeroing /
    #   get_num_free_blocks 转发等（L777-L876——零清与哈希通道归 ch13/ch15
    #   的切面）。
