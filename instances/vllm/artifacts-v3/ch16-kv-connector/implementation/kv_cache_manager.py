# SOURCE: vllm/v1/core/kv_cache_manager.py
# KVCacheManager 的 **connector 面**（本章池侧第一主角）：
# get_computed_blocks_for_connector（L297-L342——混合感知本地命中：非混合
# 直通 get_computed_blocks；混合逐组查、lagging 组深过 full → 回退全组
# 一致边界、否则以 full 组为本地前缀并报 hit_diverged）+ allocate_slots 的
# ext_comp 段（ext 计入总账、挂块条件、delay_cache_blocks『已分配未缓存』
# 早退）+ truncate_computed_blocks（子块尾砍刀）+ record_blocks_for_
# zeroing（失败重算区补登记清零）+ take_partial_tail_offloads（producer
# 部分尾交接与钉住）+ pop_blocks_for_free（账实分离取块形态）。
# 块布局注释图（L390-L446）是本章最值得对着图读的注释——五段布局。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 3 条观测面：prefix_cache_stats/make_prefix_cache_stats/
#     record_prefix_cache_stats/take_events/kv_cache_event_metadata 与
#     get_computed_blocks 的 kv_cache_report_mode='full' 事件段（L266-L284）；
#   第 10 条 CoW 打包段：take_kv_cache_block_copies（L831-L846——归 ch15；
#     scheduler 的 CoW 拷贝过线随删）；
#   estimate_cached_tokens/get_num_common_prefix_blocks/reset_prefix_cache
#     （L627-L641、L643-L675——ch15 水位/命中率切面）；
#   哈希链细节归 ch15（本章消费 find_longest_cache_hit 家族）；账本/准入门
#     归 ch14（watermark/full-ISL 门逻辑逐字保留——本章护轨的第三项）。
import itertools
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    get_kv_cache_coordinator,
)
from .kv_cache_interface import (
    AttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
)
from .kv_cache_utils import KVCacheBlock
from .math_utils import cdiv
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
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L89-L91
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    # SUBTRACTED: get_unhashed_block_ids 族（L93-L108——级联旁路）。


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
        # SUBTRACTED: use_eagle/log_stats/enable_kv_cache_events/dcp/pcp/
        #   metrics_collector 参数与 prefix_cache_stats 字段（L126-L149
        #   ——观测/eagle/上下文并行面）。
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
        #   （三态分派：False→NoPrefixCache / 单组→Unitary / 多组→Hybrid）
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
        #   watermark × num_blocks；护轨的第三项 free−reserved−watermark）
        assert watermark >= 0.0, "watermark must be non-negative"
        self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L180-L187
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

        # Partial-tail pins: blocks handed off to the connector live off the
        # request block table; pin them until the request's blocks are freed.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L189-L191 _partial_tail_pins
        #   ——m15 钉住账本
        self._partial_tail_pins: defaultdict[str, list[KVCacheBlock]] = defaultdict(
            list
        )

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L193 usage
    @property
    def usage(self) -> float:
        """Get the KV cache usage."""
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L200
        return self.block_pool.get_usage()

    # SUBTRACTED: make_prefix_cache_stats / prefix_cache_lookup_enabled /
    #   record_prefix_cache_stats（L202-L227——第 3 条观测 + ch15 命中率
    #   口径；skip_reading 判定折入 get_computed_blocks 的早退条件）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L229 get_computed_blocks
    #   ——本地命中（非混合路径；max_cache_hit_length=num_tokens−1 的
    #   全命中退一 token 契约在源头）
    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L246-L251（prefix_cache_
        #   lookup_enabled 谓词面折入：关缓存或 skip_reading → 恒空）
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L253-L259
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens, num_uncached = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        # SUBTRACTED: kv_cache_report_mode='full' 事件段（L266-L284——第 3 条）。

        # The junction to pin is where the lagging sparse-retention group stops
        # (``num_new_computed_tokens``) plus the uncached shared prefix -- i.e.
        # the longest single-group hit.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L286-L292
        shared_prefix_boundary = (
            num_new_computed_tokens + num_uncached if num_uncached else 0
        )

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L294-L295
        blocks = self.create_kv_cache_blocks(computed_blocks)
        return blocks, num_new_computed_tokens, shared_prefix_boundary

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L297
    #   get_computed_blocks_for_connector——混合感知本地命中（m3 的
    #   查找半边；connector 在场时本地走这条而不是 get_computed_blocks）
    def get_computed_blocks_for_connector(
        self, request: Request
    ) -> tuple[KVCacheBlocks, int, int, bool]:
        """Local prefix-cache lookup for a request scheduled with a KV connector.

        Hybrid (Mamba + full-attention) models can have per-group prefix hits
        diverge under block pressure: the full-attention tail may be evicted
        while a deeper Mamba state survives, or vice versa. Report the
        full-attention hit as the local prefix - the connector transfers the
        remaining suffix and the Mamba state is transferred unconditionally by
        nixl's ``_apply_prefix_caching`` - and flag when that hit ran deeper
        than a lagging group. Such a hit only has a valid Mamba state at its
        boundary if the connector supplies it, so the caller must fall back to
        ``get_computed_blocks`` to reconcile when no external tokens are found.

        Non-hybrid models and already-convergent hits use ``get_computed_blocks``.

        Returns:
            The ``get_computed_blocks`` triple (blocks, number of local computed
            tokens, shared-prefix boundary) plus ``hit_diverged``.
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L318-L324（非混合/已收敛
        #   → 直通 get_computed_blocks、hit_diverged=False）
        coordinator = self.coordinator
        if not (
            self.kv_cache_config.has_mamba_layers
            and isinstance(coordinator, HybridKVCacheCoordinator)
            and coordinator.full_attention_group_id is not None
        ):
            return *self.get_computed_blocks(request), False

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L326-L327
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0, 0, False

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L329-L337（逐组查；任何
        #   一组深过 full 组 → full 块被逐出过 → 回退全组一致边界）
        fa_group_id = coordinator.full_attention_group_id
        computed, per_group_hits = coordinator.find_longest_cache_hit_per_group(
            request.block_hashes, request.num_tokens - 1
        )
        if any(hit > per_group_hits[fa_group_id] for hit in per_group_hits):
            # A lagging group hit deeper than full attention means its
            # full-attention blocks were evicted; use the reconciled boundary
            # that every group agrees on.
            return *self.get_computed_blocks(request), False

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L339-L342（以 full 组命中
        #   为本地前缀；min(per_group_hits) < full 命中即 hit_diverged——
        #   无外部 token 撑腰时调度器须回退）
        num_local = per_group_hits[fa_group_id]
        blocks = self.create_kv_cache_blocks(computed)
        # Per-group lookups do not detect an uncached shared prefix (boundary 0).
        return blocks, num_local, 0, min(per_group_hits) < num_local

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L344 allocate_slots（connector
    #   面签名：ext/delay/reserved 三参数是本章新增面）
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
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
            num_lookahead_tokens: The number of speculative tokens to allocate.
            num_external_computed_tokens: The number of tokens that their
                KV caches are not cached by vLLM but cached by the connector.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.
            num_encoder_tokens: The number of encoder tokens to allocate for
                cross-attention in encoder-decoder models(e.g., Whisper).
                For decoder-only models, this should be 0.
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
        | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
        ----------------------------------------------------------------------
                                                  |   < to be computed >     |
        ----------------------------------------------------------------------
                                  |            < to be allocated >           |
        ----------------------------------------------------------------------
                                  | < to be cached (roughly, |
                                  | details below)>          |
        ----------------------------------------------------------------------
        | Prefix-cached tokens from either vLLM   |
        | or connector. Can be safely removed if  |
        | they are outside sliding window.        |
        ----------------------------------------------------------------------
        |   < cached by vLLM >    | not cached by |
                                  | vLLM, but     |
        | ref_cnt  | ref_cnt not  | cached by     |
        | increased| increased yet| connector     |
        ----------------------------------------------------------------------
        ```

        Abbrivations:

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        ext_comp  = num_external_computed_tokens, cached by the connector
        new       = num_new_tokens, including unverified draft tokens
        lookahead = num_lookahead_tokens
        ```

        NOTE: for new tokens which include both verified and unverified draft
        tokens, we only cache the verified tokens (by capping the number at
        `request.num_tokens`).

        The allocation has three stages:
        - Free unnecessary blocks in `comp` and check
           if we have sufficient free blocks (return None if not).
        - Handle prefix tokens (`comp + new_comp + ext_comp`):
            - Free unnecessary blocks (e.g., outside sliding window)
            - Allocate new blocks for `ext_comp` tokens inside
              sliding window
        - Allocate new blocks for tokens to be computed (`new + lookahead`)

        Returns:
            A list of new allocated blocks.
        """
        # When loading KV data asynchronously, we may have zero new tokens to
        # compute while still allocating slots for externally computed tokens.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L440-L446（零 token 护栏：
        #   num_new_tokens=0 且 ext=0 → raise——异步步『只占块不算 token』
        #   合法、纯空调度不合法）
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
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L453-L457
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L458-L461（ext 计入总账——
        #   ext_comp 段的通货进 max_model_len 封顶）
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L463-L470 水位条件（只对
        #   WAITING/PREEMPTED 且本步已有调度）
        watermark_blocks = 0
        # The watermark is applied to waiting/preempted requests only, and only
        # when there's at least one request already scheduled.
        if has_scheduled_reqs and request.status in (
            RequestStatus.WAITING,
            RequestStatus.PREEMPTED,
        ):
            watermark_blocks = self.watermark_blocks

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L472-L488 full-ISL 准入门
        #   （ch14 已立——本章透传）
        if full_sequence_must_fit:
            # First check and fail if the full request sequence won't fit.
            full_num_tokens = min(request.num_tokens, self.max_model_len)

            num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=full_num_tokens,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=total_computed_tokens,
                num_local_computed_tokens=num_local_computed_tokens,
                num_tokens_main_model=full_num_tokens,
                apply_admission_cap=True,
            )
            required_blocks = num_blocks_to_allocate + watermark_blocks
            if required_blocks > self.block_pool.get_num_free_blocks():
                return None

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L490-L493（lookahead 封顶）
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
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L495-L508
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
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_local_computed_tokens=num_local_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        # Keep `reserved_blocks` free for other in-flight sequences, and an
        # additional watermark of headroom for waiting/preempted admissions.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L521-L527 护轨门（required
        #   ≤ free − reserved，None = 调度失败的信号——m5 的落点）
        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
        required_blocks = num_blocks_to_allocate + watermark_blocks
        if required_blocks > available_blocks:
            # Cannot allocate new blocks
            return None

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L529-L540（命中挂块 +
        #   ext 分配：条件加了 `or num_external_computed_tokens > 0`——
        #   没有本地命中但 ext>0 也要走挂块路径（ext 段的分配发生在这））
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

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L542-L547
        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L549-L552——『已分配未
        #   缓存』窗口：delay → 直接返回不入哈希表（等 _update_waiting_
        #   for_remote_kv 补缓存）
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
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
            request: The request to free.
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L575-L578（钉住块随请求
        #   释放路径解钉——producer 的交接块到这才真正归还）
        pins = self._partial_tail_pins.pop(request.request_id, None)
        if pins:
            self.block_pool.free_blocks(pins)
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
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L595-L597
        self.coordinator.remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens
        )

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L599 pop_blocks_for_free——
    #   账实分离取块形态（m12 栅栏的入队端）
    def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:
        """Pop the request's bookkeeping and return its blocks without
        returning them to the block pool. The caller must eventually free
        them in reverse order (so that the tail blocks are evicted first).

        Args:
            request: The request to pop the blocks for.

        Returns:
            The request's blocks in allocation order.
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L610-L617（pins 前置随行
        #   ——被抢占可能在交接仍排队时释放钉）
        blocks = self.coordinator.pop_blocks_for_free(request.request_id)
        # Pins ride the same (possibly deferred) free as the request blocks.
        # Preemption may release a pin under a still-queued offload — the same
        # exposure normal saves of table blocks already have.
        pins = self._partial_tail_pins.pop(request.request_id, None)
        if pins:
            blocks = pins + blocks
        return blocks

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L619 evict_blocks——失败块
    #   逐出（fail 策略下 sync 命中过哈希表的坏块）
    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L624-L625
        self.block_pool.evict_blocks(block_ids)

    # SUBTRACTED: reset_prefix_cache / get_num_common_prefix_blocks /
    #   take_events（L627-L701——ch15 切面与观测面）。

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

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L711
    #   get_block_ids_for_computed_tokens——交接块表按 num_computed 裁剪
    #   （_connector_finished 消费）
    def get_block_ids_for_computed_tokens(
        self,
        request_id: str,
        num_computed_tokens: int,
    ) -> tuple[list[int], ...]:
        """Get block ids covering the request's computed tokens."""
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L717-L729
        block_ids = self.get_block_ids(request_id)
        clipped_block_ids: list[list[int]] = []
        for group, ids in zip(self.kv_cache_config.kv_cache_groups, block_ids):
            spec = group.kv_cache_spec
            if not isinstance(spec, AttentionSpec) or isinstance(
                spec, (CrossAttentionSpec, EncoderOnlyAttentionSpec)
            ):
                clipped_block_ids.append(ids)
                continue

            num_valid_blocks = cdiv(num_computed_tokens, spec.block_size)
            clipped_block_ids.append(ids[:num_valid_blocks])
        return tuple(clipped_block_ids)

    # SUBTRACTED: estimate_cached_tokens（L731-L758——prefill_stats 观测面）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L760 cache_blocks——补缓存
    #   原语（_update_waiting_for_remote_kv 在传输完成时调它把延迟的
    #   缓存补上；enable_caching=False 时 no-op）
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

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L777 truncate_computed_blocks
    #   ——子块尾砍刀（m3 仲裁的执行端：纯切片、ref 不动、块对齐断言）
    def truncate_computed_blocks(
        self, blocks: KVCacheBlocks, num_computed_tokens: int
    ) -> KVCacheBlocks:
        """Return a lookup-result view truncated at an aligned token endpoint.

        Pure slicing: refcounts are untouched and ``blocks`` is not mutated.
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L784-L794
        truncated: list[list[KVCacheBlock]] = []
        for group_blocks, manager in zip(
            blocks.blocks,
            self.coordinator.single_type_managers,
            strict=True,
        ):
            assert num_computed_tokens % manager.block_size == 0
            num_blocks = num_computed_tokens // manager.block_size
            assert num_blocks <= len(group_blocks)
            truncated.append(list(group_blocks[:num_blocks]))
        return self.create_kv_cache_blocks(tuple(truncated))

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L796 take_new_block_ids——
    #   清零账 drain（站 6 的 _get_new_block_ids_to_zero 消费）
    def take_new_block_ids(self) -> list[int]:
        """Drain and return new attention block IDs for zeroing."""
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L798-L801
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            ids.extend(mgr.take_new_block_ids())
        return ids

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L803
    #   get_zeroing_block_ids_in_range——异步加载将覆写块的取账（站 6：
    #   _skip_zero_block_ids 的登记来源）
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
    #   ——失败重算区补登记清零（块对齐：半有效块清零会抹掉有效前缀）
    def record_blocks_for_zeroing(self, request_id: str, start_token: int) -> None:
        """Re-record the request's blocks from start_token onwards for
        zeroing, e.g. blocks a failed async KV load left unwritten.

        start_token must be block-aligned: zeroing a partially-valid block
        would wipe its valid prefix.
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L824-L829
        for mgr in self.coordinator.single_type_managers:
            if mgr.records_new_block_ids:
                assert start_token % mgr.block_size == 0
                start_idx = start_token // mgr.block_size
                blocks = mgr.req_to_blocks[request_id]
                mgr.new_block_ids.extend(blk.block_id for blk in blocks[start_idx:])

    # SUBTRACTED: take_kv_cache_block_copies（L831-L846——第 10 条 CoW
    #   打包段归 ch15）。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L848 take_partial_tail_offloads
    #   ——producer 部分尾交接的 drain + 钉住（m15）
    def take_partial_tail_offloads(self) -> dict[str, list[tuple[int, int, int]]]:
        """Drain producer partial-tail offload hand-offs per request.

        Returns ``{request_id: [(group_id, block_id, boundary_tokens), ...]}``
        for the durable boundary blocks of producers' last-prompt-boundary
        partial tails. Only mamba "align" groups contribute; empty otherwise.
        A KV connector reads the referenced blocks and offloads them so a later
        request can hit the sub-block prefix.

        Each handed-off block lives off the request block table, so it is
        pinned here and unpinned when the request's blocks are freed — for a
        producer with saved tokens, after the connector reports sends done.
        """
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L861-L874（drain + touch
        #   钉住 + 钉账本）
        offloads: dict[str, list[tuple[int, int, int]]] = {}
        for mgr in self.coordinator.single_type_managers:
            for (
                req_id,
                group_id,
                block,
                boundary_tokens,
            ) in mgr.take_pending_partial_tail_offloads():
                self.block_pool.touch((block,))
                self._partial_tail_pins.setdefault(req_id, []).append(block)
                offloads.setdefault(req_id, []).append(
                    (group_id, block.block_id, boundary_tokens)
                )
        return offloads

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L876 new_step_starts
    def new_step_starts(self) -> None:
        """Notify the coordinator that a new step is starting."""
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L878
        self.coordinator.new_step_starts()
