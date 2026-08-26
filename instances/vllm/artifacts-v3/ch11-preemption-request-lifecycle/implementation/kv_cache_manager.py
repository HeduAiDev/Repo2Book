# SOURCE: vllm/v1/core/kv_cache_manager.py + vllm/v1/core/kv_cache_utils.py
# KVCacheManager 的**接口契约面**（分页块池内部归 ch13/14、链式哈希归 ch15——
# dossier.key_classes「当黑盒契约面用」）。本章保留三个可观测语义：
#   allocate_slots —— 满则 None（RUNNING 侧触发抢占 / WAITING 侧 break 的唯一
#                    信号）+ full_sequence_must_fit 整序列准入门 + **watermark
#                    水位**（仅 WAITING/PREEMPTED 且 has_scheduled_reqs 时计入
#                    headroom——m8 机制本体，kv_cache_manager.py:L463-L470 与
#                    L521-L527 两处原文保留）；
#   get_computed_blocks —— 前缀重命中（m7/F2）：沿 request.block_hashes 连续
#                    匹配 free 后仍留表的哈希；max_cache_hit_length = num_tokens-1
#                    （全命中也必须重算最后一个 token 才有 logits）；
#   free —— 块归还空闲池但**块哈希不清**（F2 证据锚：真实 block_pool.py:
#                    L719-L742 free_blocks 只动 ref_cnt/自由队列，从不 touch
#                    哈希）——被抢者恢复时重命中自己的前缀。
# 另含 get_request_block_hasher（vllm/v1/core/kv_cache_utils.py:L691-L748 的
# 请求侧增量哈希器——真实装配：EngineCore 建请求时注入，core.py:L220-L227）。
from __future__ import annotations

from collections.abc import Callable

from .request import RequestStatus


# SOURCE: vllm/v1/core/kv_cache_manager.py:L33 KVCacheBlocks
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    # SUBTRACTED: blocks 分组元组（L40-L53——多 KV 组的混合注意力布局）与
    #   __add__（L55-L62，块句柄拼接）——单组全注意力下退化为单列表。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L33 KVCacheBlocks
    def __init__(self, block_ids: list[int]) -> None:
        self._block_ids = block_ids

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L64 get_block_ids
    def get_block_ids(self, allow_none: bool = False) -> tuple[list[int], ...]:
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L78-L86
        # SUBTRACTED: allow_none 的 None 组语义（多 KV 组）——单组恒返回块 id 列表。
        return (list(self._block_ids),)


# SOURCE: vllm/v1/core/kv_cache_utils.py:L691 get_request_block_hasher
def get_request_block_hasher(block_size: int) -> Callable[[object], list[int]]:
    """
    Returns a function which computes the list of un-computed block hashes
    of a request.

    Hashes are chained over the full prefix, so each hash uniquely
    fingerprints the prefix ending at its boundary.
    """
    # SUBTRACTED: hash_block_size≠block_size 的粗粒度组对齐参数（L692-L694 与
    #   L699-L703 注释——单组全注意力下 hash_block_size==block_size）；mm/LoRA
    #   的 extra_keys 融合（L713-L734）；真实 caching_hash_fn（defaultdict 哈希
    #   器，L738-L740）换等值 tuple 哈希——链式父哈希语义保留。

    # SOURCE: vllm/v1/core/kv_cache_utils.py:L705 request_block_hasher
    def request_block_hasher(request) -> list[int]:
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L706-L711
        start_token_idx = len(request.block_hashes) * block_size
        num_tokens = request.num_tokens

        if start_token_idx + block_size > num_tokens:
            # Early stop when there no new full blocks created.
            return []

        # SOURCE: vllm/v1/core/kv_cache_utils.py:L721-L723
        prev_block_hash_value = (
            request.block_hashes[-1] if request.block_hashes else None
        )
        new_block_hashes: list[int] = []
        # SOURCE: vllm/v1/core/kv_cache_utils.py:L725-L746
        while True:
            end_token_idx = start_token_idx + block_size
            if end_token_idx > num_tokens:
                # We only hash full blocks
                break

            # Compute the hash of the current block
            # SOURCE: vllm/v1/core/kv_cache_utils.py:L737-L740 hash_block_tokens
            block_tokens = request.all_token_ids[start_token_idx:end_token_idx]
            block_hash = hash((prev_block_hash_value, tuple(block_tokens)))

            new_block_hashes.append(block_hash)
            start_token_idx += block_size
            prev_block_hash_value = block_hash

        return new_block_hashes

    return request_block_hasher


# SOURCE: vllm/v1/core/kv_cache_manager.py:L125 KVCacheManager（契约面）
class KVCacheManager:
    # SOURCE: vllm/v1/core/kv_cache_manager.py:L128 __init__
    def __init__(
        self,
        num_gpu_blocks: int = 1 << 30,
        block_size: int = 16,
        max_model_len: int = 1 << 20,
        watermark: float = 0.0,
    ) -> None:
        # SUBTRACTED: 真实构造从 kv_cache_config 建 coordinator/块池/分组哈希
        #   表/事件发布器（L128-L166——分页池内部归 ch13/14）；这里以『空闲块
        #   计数 + 每请求持有块账 + 满块哈希表』的最小可运行账本承载同一批
        #   调度侧可观测语义。
        self.num_gpu_blocks = num_gpu_blocks
        self.num_free_blocks = num_gpu_blocks
        self.block_size = block_size
        self.max_model_len = max_model_len
        # req_id -> 该请求当前持有的块 id（含前缀命中挂账的 -1 共享块）
        self._blocks: dict[str, list[int]] = {}
        self._next_block_id = 0
        # 满块哈希表（真实在 BlockPool.cached_block，含 LRU 驱逐候选序——
        # ch13/15 的主角）。本章只保留『free 不清哈希』这一条事实的最小账：
        # 哈希登记于 allocate（真实：cache_blocks 提交），free 只还块不清哈希。
        self.cached_block_hashes: set[int] = set()
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L158 empty_kv_cache_blocks
        self.empty_kv_cache_blocks = KVCacheBlocks([])

        # Watermark: minimum number of KV cache blocks to keep free when
        # admitting waiting/preempted requests, to avoid frequent preemptions.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L168-L171 watermark_blocks
        assert watermark >= 0.0, "watermark must be non-negative"
        self.watermark_blocks = int(watermark * num_gpu_blocks)

        # SUBTRACTED: enable_caching/enable_kv_cache_events/use_eagle/log_stats/
        #   metrics_collector/prefix_cache_stats/coordinator/empty_kv_cache_
        #   blocks 的分组构造/_partial_tail_pins（L141-L191）——观测与分页池细节。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L876 new_step_starts
    def new_step_starts(self) -> None:
        """Notify the coordinator that a new step is starting."""
        # SUBTRACTED: coordinator.new_step_starts() 的每拍重置（L878——块池内部
        #   簿记，ch13）。账本翻页动作本身保留为调度侧可见的调用点。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L229 get_computed_blocks
    def get_computed_blocks(self, request) -> tuple[KVCacheBlocks, int, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.
        """
        # SUBTRACTED: prefix_cache_lookup_enabled 门（skip_reading_prefix_cache，
        #   L246-L251——缓存读取开关与 KV events，ch15）；真实 find_longest_
        #   cache_hit 按链式哈希逐块匹配 + CoW 处理（L259-L295）——换沿
        #   request.block_hashes 的连续命中计数，保留同一可观测语义：
        #   **free 不清哈希 → 被抢者恢复时重命中自己的前缀**（F2）。
        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L253-L259
        max_cache_hit_length = request.num_tokens - 1
        max_hit_blocks = max_cache_hit_length // self.block_size
        num_hit_blocks = 0
        for block_hash in request.block_hashes[:max_hit_blocks]:
            if block_hash in self.cached_block_hashes:
                num_hit_blocks += 1
            else:
                break
        num_new_computed_tokens = num_hit_blocks * self.block_size
        # SUBTRACTED: shared_prefix_boundary 的推导（L286-L292——Marconi 稀疏
        #   驻留归 ch15）——纯 full-attention 单组下恒 0；返回三元组结构保留。
        return self.empty_kv_cache_blocks, num_new_computed_tokens, 0

    # SUBTRACTED: get_computed_blocks_for_connector（L297-L342，hybrid-aware
    #   查找）与 make_prefix_cache_stats/prefix_cache_lookup_enabled/
    #   record_prefix_cache_stats/usage（L193-L227——可观测性，第 10 条）——
    #   dossier.delete 批准。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L344 allocate_slots
    def allocate_slots(
        self,
        request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        has_scheduled_reqs: bool = True,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append.

        Returns:
            A list of new allocated blocks.
        """
        # SUBTRACTED: num_external_computed_tokens / delay_cache_blocks /
        #   num_encoder_tokens / reserved_blocks 四参（connector/encoder，
        #   dossier.delete 第 1/2 条）与 blocks-layout 长图（L390-L438——分页
        #   池内部）；coordinator 的分组块计算换单组『目标 token 覆盖块数 −
        #   已持有块数』等价算术（ch13）。
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L442-L446 零 token 护栏
        if num_new_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L453-L457
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L463-L470 watermark 应用（原文）
        watermark_blocks = 0
        # The watermark is applied to waiting/preempted requests only, and only
        # when there's at least one request already scheduled.
        if has_scheduled_reqs and request.status in (
            RequestStatus.WAITING,
            RequestStatus.PREEMPTED,
        ):
            watermark_blocks = self.watermark_blocks

        if full_sequence_must_fit:
            # First check and fail if the full request sequence won't fit.
            # SOURCE: vllm/v1/core/kv_cache_manager.py:L472-L488 整序列准入门
            full_num_tokens = min(request.num_tokens, self.max_model_len)

            num_blocks_to_allocate = self._get_num_blocks_to_allocate(
                request.request_id,
                full_num_tokens,
                num_new_computed_tokens,
            )
            required_blocks = num_blocks_to_allocate + watermark_blocks
            if required_blocks > self.num_free_blocks:
                return None

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L490-L493
        num_tokens_main_model = num_local_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens, self.max_model_len
        )

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L510-L527（need vs free，原文）
        num_blocks_to_allocate = self._get_num_blocks_to_allocate(
            request.request_id,
            num_tokens_need_slot,
            num_new_computed_tokens,
        )
        # Keep `reserved_blocks` free for other in-flight sequences, and an
        # additional watermark of headroom for waiting/preempted admissions.
        # SUBTRACTED: reserved_blocks 扣减（connector 在途预约，第 1 条）——
        #   available 即 free。
        required_blocks = num_blocks_to_allocate + watermark_blocks
        if required_blocks > self.num_free_blocks:
            # Cannot allocate new blocks
            return None

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L529-L547（挂命中块 + 拨新块）
        cur = self._blocks.setdefault(request.request_id, [])
        if num_new_computed_tokens > 0:
            # 前缀命中块直接挂到请求账上（共享缓存块，不占空闲池——真实语义
            # 是 ref_cnt+1 复用）。
            # SUBTRACTED: allocate_new_computed_blocks 的 CoW/引用计数（ch13）。
            cur.extend([-1] * (num_new_computed_tokens // self.block_size))
        new_blocks: list[int] = []
        if num_blocks_to_allocate > 0:
            new_blocks = list(
                range(
                    self._next_block_id,
                    self._next_block_id + num_blocks_to_allocate,
                )
            )
            self._next_block_id += num_blocks_to_allocate
            self.num_free_blocks -= num_blocks_to_allocate
            cur.extend(new_blocks)

        # SOURCE: vllm/v1/core/kv_cache_manager.py:L549-L563 cache 提交
        # SUBTRACTED: coordinator.cache_blocks 的 CoW/部分块哈希与 num_tokens_
        #   to_cache 钳制（ch13/15）——只保留『满块哈希登记进表』这一动作：
        #   此后 free 不清（F2 事实锚，block_pool.py:L719-L742 只动 ref_cnt/
        #   自由队列）。
        self.cached_block_hashes.update(request.block_hashes)

        return KVCacheBlocks(new_blocks)

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L476/L510 get_num_blocks_to_allocate
    def _get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        num_new_computed_tokens: int = 0,
    ) -> int:
        # SUBTRACTED: 真实 coordinator.get_num_blocks_to_allocate 的多组/滑窗/
        #   编码器/准入上限块计算（kv_cache_coordinator.py，ch13/14）——单组全
        #   注意力下等价于：目标 token 覆盖块数 − 已持有块数（命中块视同已持有）。
        need = -(-num_tokens // self.block_size)  # ceil
        held = len(self._blocks.get(request_id, [])) + (
            num_new_computed_tokens // self.block_size
        )
        return max(need - held, 0)

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L567 free —— F2 的证据锚
    def free(self, request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.
        """
        # SUBTRACTED: _partial_tail_pins 与 coordinator.free 的逆序引用计数
        #   归还（L575-L578——LRU 序与侵入式链表归 ch13/15）。两条可观测事实
        #   保留：①块归还空闲池（-1 命中占位是共享缓存块，不入空闲计数）；
        #   ②**块哈希不清**——cached_block_hashes 原样保留（真实 free_blocks
        #   只动 ref_cnt/自由队列，block_pool.py:L719-L742），被抢者恢复时
        #   get_computed_blocks 仍能命中自己的前缀，下一个同前缀请求也能。
        ids = self._blocks.pop(request.request_id, [])
        self.num_free_blocks += len([b for b in ids if b != -1])

    # SUBTRACTED: remove_skipped_blocks/pop_blocks_for_free/evict_blocks/
    #   reset_prefix_cache/cache_blocks/truncate_computed_blocks/
    #   take_new_block_ids/get_zeroing_block_ids_in_range/take_kv_cache_block_
    #   copies/take_partial_tail_offloads/get_num_common_prefix_blocks
    #   （L580-L874——滑窗/LRU 驱逐/CoW/mamba 清零/级联注意力，各归 ch13/14/15
    #   与模型章）——dossier.delete 批准的邻章范围。

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L703 get_blocks
    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L703-L713
        return KVCacheBlocks(list(self._blocks.get(request_id, [])))
