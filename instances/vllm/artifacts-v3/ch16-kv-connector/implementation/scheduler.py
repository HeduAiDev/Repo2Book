# SOURCE: vllm/v1/core/sched/scheduler.py
# Scheduler 的 **connector 面**（本章调度侧第一主角——一个请求的异步 KV
# 一生在调度器上的全部落点）：
#   构造四旗标（L125-L158：connector 装配/recompute_kv_load_failures/
#     defer_block_free/requires_kv_delivery + encoder-decoder 拒绝）
#     + bind_gpu_block_pool（L289-L294）+ 双收发账本（L205-L207）+
#     步序栅栏账（L327-L331）+ _skip_zero_block_ids（L310-L313）；
#   schedule 的 waiting 循环双查+仲裁（L744-L832：get_computed_blocks_
#     for_connector → get_num_new_matched_tokens（None→skipped）→ 子块尾
#     仲裁/混合回退）、护轨分配（L934-L985：reserved_blocks + allocate_
#     slots(ext, delay)）、update_state_after_alloc + 外部命中率记账
#     （L996-L1014）、WAITING_FOR_REMOTE_KVS 先行记账与跳清零（L1023-L1053）、
#     producer partial-tail drain（L1165-L1179）、元数据过线（L1233-L1258）；
#   update_from_output 的栅栏推进（L1684-L1688）+ invalid blocks 消化
#     （L1697-L1705）+ KVConnectorOutput 消化（L1974-L1976）；
#   回收与失败族：_update_from_kv_xfer_finished（L2714-L2741）、
#     _update_waiting_for_remote_kv（L2635-L2676）、_try_promote_blocked_
#     waiting_request（L2678-L2693）、_update_requests_with_invalid_blocks
#     （L2743-L2844 第一个坏块截断+共享去重）、_handle_invalid_blocks
#     （L2846-L2915 双策）；
#   终局与边界：_free_request（L2300-L2327）、finish_requests（L2237-L2298
#     的 WAITING_FOR_REMOTE_KVS 延迟释放分支）、_connector_finished（L2577-
#     L2612）、_free_request_blocks/_drain_deferred_frees（L2341-L2380）、
#     _preempt_request 的 drop_stale_output（L1274-L1315）、has_finished_
#     requests/has_requests（L2394-L2421）、_request_remaining_blocks/
#     _inflight_prefill_reserved_blocks（L2614-L2633）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 1 条 ECConnector 全分支（self.ec_connector/ec_xfer_params 及调用点）；
#   第 3 条观测面：log_stats 事件/record_prefix_cache_stats（L1016-L1020）、
#     kv_connector_stats 聚合（L1978-L1993）、kv events 发布（L1995-L2010）、
#     EngineCoreOutputs 装配（L2012-L2046）；
#   第 4 条 P/D 握手聚合（engine/core.py 段——不在本文件）；
#   第 9 条与本章无关大块：LoRA max_loras 约束（L724-L737）、spec decode
#     （pad/dynamic_sd/spec_token_ids/num_sampled_tokens_per_step）、
#     encoder-decoder 调度与 encoder_cache_manager、streaming/pause 状态机、
#     routed_experts、DP prefill balancing（defer_prefills）、长 prefill
#     阈值与 chunked prefill 的 enable_chunked_prefill 面、mm 预取、
#     V2 model runner 面与 grammar/streaming 提升分支（_try_promote 的
#     另两状态）、EngineCoreOutputs/stats 全量装配；
#   第 10 条 CoW 打包段：take_kv_cache_block_copies/_free_cow_retained_
#     blocks/kv_cache_block_copies（L1181-L1190、L2356-L2365——归 ch15；
#     栅栏队 deferred_frees 本体保留）；
#   第 6 条 mamba 对齐切分 _mamba_block_aligned_split 调用（L934-L943
#     ——ch15 m16；need_mamba_block_aligned_split 账位保留）；
#   抢占的 PRIORITY 支（L590-L613——ch10/11）；语法编译失败账
#     （grammar_compile_error_reqs——ch05）。
import logging
from collections import deque
from typing import Any

from .base import KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole, SupportsHMA
from .factory import KVConnectorFactory
from .kv_cache_manager import KVCacheBlocks, KVCacheManager
from .kv_cache_utils import KVCacheBlock
from .output import CachedRequestData, NewRequestData, SchedulerOutput
from .outputs import KVConnectorOutput, ModelRunnerOutput
from .request import Request, RequestStatus
from .request_queue import create_request_queue
from .stats import PrefixCacheStats

logger = logging.getLogger(__name__)  # LOGGER SEAM


# SOURCE: vllm/v1/core/sched/scheduler.py:L~267 Scheduler（connector 面切面
#   ——真实类为完整调度器，ch10/11 全文）
class Scheduler:
    # SOURCE: vllm/v1/core/sched/scheduler.py:L~60 __init__（切面装配：
    #   站 1 的四旗标 + 双队列 + 账本）
    def __init__(
        self,
        vllm_config,
        kv_cache_config,
        block_size: int,
        hash_block_size: int | None = None,
        max_model_len: int | None = None,
        log_stats: bool = False,
    ) -> None:
        # SUBTRACTED: 队列策略/输出处理器/preemptor/observability/grammar/
        #   encoder cache 装配面（ch02/05/10/11）。
        self.vllm_config = vllm_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.kv_cache_config = kv_cache_config
        self.max_model_len = (
            max_model_len if max_model_len is not None else vllm_config.model_config.max_model_len
        )
        self.log_stats = log_stats
        self.is_encoder_decoder = vllm_config.is_encoder_decoder

        # Scheduling constraints.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L109-L114（token 预算账位）
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_batched_tokens
        )

        # SOURCE: vllm/v1/core/sched/scheduler.py:L125-L158 站 1：调度器侧
        #   connector 装配 + 四旗标（逐字）
        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        self.defer_block_free = False
        # Whether a preempted request's in-flight output must be dropped; see
        # KVConnectorBase_V1.requires_kv_delivery.
        self.requires_kv_delivery = False
        kv_transfer_config = self.vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = kv_transfer_config.kv_load_failure_policy
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

            # With overlapping batches (async scheduling or PP), a step may
            # still be writing a freed request's KV blocks. A consumer KV
            # Connector can reallocate and fill those blocks via a load that
            # isn't ordered against that write, so defer freeing them.
            multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
            if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
                self.defer_block_free = True

            self.requires_kv_delivery = self.connector.requires_kv_delivery

        # SUBTRACTED: kv_event_publisher/ec_connector 装配（L160-L168——
        #   第 1/3 条）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L173 block_size
        self.block_size = block_size

        # req_id -> Request
        # SOURCE: vllm/v1/core/sched/scheduler.py:L177-L178
        self.requests: dict[str, Request] = {}
        # SOURCE: vllm/v1/core/sched/scheduler.py:L187-L190 双队列（skipped
        #   ——退避队列，None 语义的落点）
        self.waiting = create_request_queue()
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue()
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L192-L196
        self.finished_req_ids: set[str] = set()

        # IDs of requests preempted since the last call to schedule().
        # SOURCE: vllm/v1/core/sched/scheduler.py:L198-L199
        self.reset_preempted_req_ids: set[str] = set()

        # KV Connector: requests in process of async KV loading or recving
        # SOURCE: vllm/v1/core/sched/scheduler.py:L205-L207 双收发账本
        #   （提升判据/失败重试账）
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Create the KV cache manager.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L272-L290（watermark 实参
        #   落地；use_eagle/dcp/events 参数面随各章删）
        if hash_block_size is None:
            hash_block_size = block_size
        self.hash_block_size = hash_block_size
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            scheduler_block_size=self.block_size,
            hash_block_size=hash_block_size,
            watermark=self.scheduler_config.watermark,
        )
        # Bind GPU block pool to the KV connector. This must happen after
        # kv_cache_manager is constructed so block_pool is available.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L291-L294
        if self.connector is not None:
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)

        # SOURCE: vllm/v1/core/sched/scheduler.py:L305-L307（full-ISL 门开关）
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl
        )

        # SOURCE: vllm/v1/core/sched/scheduler.py:L309-L310
        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
        # Blocks that async KV loads will overwrite this step, skipped from
        # zeroing since the zeroing could race the out-of-band write.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L311-L313 _skip_zero_block_ids
        self._skip_zero_block_ids: set[int] = set()
        # SOURCE: vllm/v1/core/sched/scheduler.py:L314-L316（mamba 对齐切分
        #   开关——本章不驱动，账位保留）
        self.need_mamba_block_aligned_split = False

        # SOURCE: vllm/v1/core/sched/scheduler.py:L~318 在途 prefill 集合
        #   （护轨 reserved_blocks 的求和对象）
        self._inflight_prefills: set[Request] = set()

        # SOURCE: vllm/v1/core/sched/scheduler.py:L327-L331 步序栅栏账
        #   （sched_step_seq 单调推进 / processed_step_seq 步处理水位 /
        #   deferred_frees 栅栏队——defer_block_free 的三件套）
        self.sched_step_seq = 0
        self.processed_step_seq = 0
        # Blocks whose freeing must wait until processed_step_seq >= fence_seq.
        self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()

    # ==============================
    # 入队与新请求登记
    # ==============================

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2213 add_request（else 支——
    #   新请求：入队 + requests 登记 + connector 登记钩子）
    def add_request(self, request: Request) -> None:
        # SUBTRACTED: streaming 会话面（L2214-L2226——ch02）。
        self._enqueue_waiting_request(request)
        self.requests[request.request_id] = request
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2232-L2233（on_new_
        #   request 调用点保留）
        if self.connector is not None:
            self.connector.on_new_request(request)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2058 _enqueue_waiting_request
    def _enqueue_waiting_request(self, request: Request) -> None:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2059-L2062（blocked 状态
        #   直接进 skipped_waiting——退避队列的入队端）
        if self._is_blocked_waiting_status(request.status):
            self.skipped_waiting.add_request(request)
        else:
            self.waiting.add_request(request)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2064 _select_waiting_queue_
    #   for_scheduling（FCFS：skipped 优先）
    def _select_waiting_queue_for_scheduling(self):
        # SUBTRACTED: PRIORITY 比较（L2068-L2072——ch10/11）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2065-L2066
        return self.skipped_waiting or self.waiting or None

    # ==============================
    # schedule：waiting 循环的 connector 路径 + 元数据过线
    # ==============================

    # SOURCE: vllm/v1/core/sched/scheduler.py:L~443 schedule（connector 面
    #   切面：RUNNING 循环预算面 + WAITING 循环 connector 全路径 + 收尾）
    def schedule(self) -> SchedulerOutput:
        # SUBTRACTED: pause/DP throttle/spec 账（L460-L481——各邻章）。
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens

        # For logging.
        # SUBTRACTED: scheduled_timestamp = time.monotonic()（事件面）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L475
        self.kv_cache_manager.new_step_starts()

        # SUBTRACTED: defer_prefills（L477-L481——DP prefill balancing）。

        # First, schedule the RUNNING requests.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L483-L514（RUNNING 循环
        #   骨架；spec/placeholders/cadence/encoder 段删——num_new_tokens
        #   = num_tokens − num_computed 的本质）
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = request.num_tokens - request.num_computed_tokens
            num_new_tokens = min(num_new_tokens, token_budget)
            # SUBTRACTED: max_model_len 截尾的 spec 采样修正
            #   （L525-L532——num_sampled_tokens_per_step=1 面归 ch33）。

            if num_new_tokens == 0:
                # The request cannot be scheduled.
                # 1. No new tokens to schedule.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L576-L629（allocate-
            #   while-preempt 环：失败 → 抢占队尾（FCFS 支）→ drop_stale_
            #   output=self.requires_kv_delivery（m13 落点，L614-L621 逐字））
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
            )

            if new_blocks is None:
                # The request cannot be scheduled.
                # Preempt the lowest-priority request.
                # SUBTRACTED: PRIORITY 支（L590-L613——ch10/11）。
                preempted_req = self.running.pop()

                self._preempt_request(
                    preempted_req,
                    0.0,
                    drop_stale_output=self.requires_kv_delivery,
                )
                preempted_reqs.append(preempted_req)
                if preempted_req == request:
                    # No more request to preempt. Cannot schedule this request.
                    break
                continue

            # Schedule the request.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L631-L638
            scheduled_running_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

        # Then, schedule the WAITING requests.
        # SUBTRACTED: constraint_checking/encoder budget 账（L520-L543——
        #   ch02 面的骨架账位）与 preempted_reqs/pause 条件（L684——
        #   本章 RUNNING 循环的抢占面已含）。
        if token_budget > 0:
            # SOURCE: vllm/v1/core/sched/scheduler.py:L685 step_skipped_waiting
            step_skipped_waiting = create_request_queue()

            # SOURCE: vllm/v1/core/sched/scheduler.py:L687（双队列都排干才停）
            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                # SOURCE: vllm/v1/core/sched/scheduler.py:L690-L692（运行数
                #   上限守卫——streaming 账位随 ch02 删）
                num_running = len(self.running)
                if num_running >= self.max_num_running_reqs:
                    break

                # SOURCE: vllm/v1/core/sched/scheduler.py:L694-L695（FCFS：
                #   skipped 优先——退避队头先重试）
                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                # SOURCE: vllm/v1/core/sched/scheduler.py:L697-L698
                request = request_queue.peek_request()
                request_id = request.request_id

                # try to promote blocked statuses while traversing skipped queue.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L700-L711（提升失败
                #   → 留 skipped 不堵队头）
                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # SOURCE: vllm/v1/core/sched/scheduler.py:L713-L722（可交付的
                #   在途 stale 产出未排空前不恢复——ch12 的联动账）
                if (
                    request.num_stale_output_tokens > 0
                    and not request.drop_stale_output
                ):
                    # Deliverable stale output still in flight: resuming now
                    # could resample a position that output later delivers.
                    # It drains within the pipeline depth.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # SUBTRACTED: LoRA max_loras 约束（L724-L737——第 9 条）。

                # SOURCE: vllm/v1/core/sched/scheduler.py:L739-L742
                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
                did_prefix_cache_lookup = False

                # Get already-cached tokens.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L744-L832 站 3/4：
                #   双查 + 仲裁（逐字）
                if request.num_computed_tokens == 0:
                    did_prefix_cache_lookup = True
                    hit_diverged = False
                    # Get locally-cached tokens.
                    if self.connector is not None:
                        # A KV connector transfers the missing suffix, which needs a
                        # hybrid-aware lookup that can diverge across groups.
                        (
                            new_computed_blocks,
                            num_new_local_computed_tokens,
                            request.shared_prefix_boundary,
                            hit_diverged,
                        ) = self.kv_cache_manager.get_computed_blocks_for_connector(
                            request
                        )
                    else:
                        (
                            new_computed_blocks,
                            num_new_local_computed_tokens,
                            # Marconi shared-prefix junction to pin; 0 if none.
                            request.shared_prefix_boundary,
                        ) = self.kv_cache_manager.get_computed_blocks(request)

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        # Present a block-aligned local hit to the connector so
                        # a strictly longer remote hit can supersede a local
                        # sub-block tail without racing its copy-on-write.
                        partial_tail = num_new_local_computed_tokens % self.block_size
                        block_aligned_local = (
                            num_new_local_computed_tokens - partial_tail
                        )
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, block_aligned_local
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                        if partial_tail and ext_tokens > partial_tail:
                            # Remote strictly exceeds the full local hit: drop the
                            # sub-block tail so no CoW is needed, and let the load
                            # cover it. Trim the partial block out of the local
                            # computed blocks so it is not adopted from the cache.
                            new_computed_blocks = (
                                self.kv_cache_manager.truncate_computed_blocks(
                                    new_computed_blocks, block_aligned_local
                                )
                            )
                            num_new_local_computed_tokens = block_aligned_local
                            num_external_computed_tokens = ext_tokens
                        elif partial_tail:
                            # Remote does not exceed the full local hit: keep the
                            # local sub-block tail and load nothing external.
                            num_external_computed_tokens = 0
                            # Nothing to load remotely -> not an async-load step;
                            # clearing avoids the `load_kv_async` assert below.
                            load_kv_async = False
                        else:
                            num_external_computed_tokens = ext_tokens

                        if hit_diverged and num_external_computed_tokens == 0:
                            # No external tokens back the deeper local hit, so its
                            # resume boundary would have no valid Mamba state.
                            # Reconcile to the boundary every group agrees on.
                            (
                                new_computed_blocks,
                                num_new_local_computed_tokens,
                                request.shared_prefix_boundary,
                            ) = self.kv_cache_manager.get_computed_blocks(request)

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens

                    # SUBTRACTED: mm 预取门（L834-L844——EC 面）与
                    #   prefill_stats 记账（L846-L853——观测面）。
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    # SOURCE: vllm/v1/core/sched/scheduler.py:L854-L859
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                # SOURCE: vllm/v1/core/sched/scheduler.py:L861-L864（encoder
                #   预算账位——encoder-decoder 已拒，账位归零）
                encoder_inputs_to_schedule = None

                # SOURCE: vllm/v1/core/sched/scheduler.py:L866-L869（async
                #   期不算新 token——assert ext>0）
                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    # SOURCE: vllm/v1/core/sched/scheduler.py:L874-L879
                    num_new_tokens = request.num_tokens - num_computed_tokens

                    # SUBTRACTED: DP defer/spec pad/长 prefill 阈值/chunked
                    #   prefill 停点（L870-L932——ch10/33 邻章；预算取 min
                    #   保留在 RUNNING 循环同款语义）
                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                # SUBTRACTED: mamba 对齐切分调用（L934-L943——ch15 m16；
                #   本章配置 need_mamba_block_aligned_split=False）。

                # During async KV load, no forward pass is run yet.
                # Allocate speculative lookahead slots later to avoid
                # mismatching local and remote block counts.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L945-L951（本章
                #   num_lookahead_tokens=0——spec 面归 ch33，账位烘干为 0）
                limit_lookahead_tokens = False
                effective_lookahead_tokens = 0

                # Determine if we need to allocate cross-attention blocks.
                # SUBTRACTED: encoder-decoder cross-attention 分支（L953-L963
                #   ——dossier elide 批准：本章 encoder-decoder 构造期已拒）。
                num_encoder_tokens = 0

                # SOURCE: vllm/v1/core/sched/scheduler.py:L965-L971 站 5：
                #   护轨分配（async load 只许 fits in (free − 其余在途预约)，
                #   防死锁与可预期抢占）
                reserved_blocks = 0
                if load_kv_async:
                    # An async load holds its blocks for the whole transfer with
                    # no forward progress and isn't preemptible here. Admit it
                    # only if it fits in (free - other in-flight reservations), to
                    # avoid deadlock and predictable preemptions.
                    reserved_blocks = self._inflight_prefill_reserved_blocks()

                # SOURCE: vllm/v1/core/sched/scheduler.py:L973-L985
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=0,  # SUBTRACTED: encoder 实参（恒 0）
                    full_sequence_must_fit=self.scheduler_reserve_full_isl,
                    reserved_blocks=reserved_blocks,
                    has_scheduled_reqs=bool(self.running),
                )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    # SOURCE: vllm/v1/core/sched/scheduler.py:L987-L994（encoder
                    #   free 段删——waiting 面无 preempt，break）
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L996-L1014 站 5 尾：
                #   update_state_after_alloc + 外部命中率官方口径记账
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                # SUBTRACTED: record_prefix_cache_stats（L1016-L1020——ch15
                #   本地命中率口径）。

                # SOURCE: vllm/v1/core/sched/scheduler.py:L1022 站 6：pop 队头
                request = request_queue.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    # SOURCE: vllm/v1/core/sched/scheduler.py:L1026-L1053 逐字
                    #   （先行记账 + skipped 队 + 跳清零登记）
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number
                    # of computed tokens.
                    # _update_waiting_for_remote_kv will then cache
                    # only the successfully loaded tokens.
                    request.num_computed_tokens = num_computed_tokens
                    self._inflight_prefills.add(request)
                    if self.needs_kv_cache_zeroing:
                        # Skip zeroing of the blocks the async load will
                        # overwrite; the zeroing could race the write.
                        self._skip_zero_block_ids.update(
                            self.kv_cache_manager.get_zeroing_block_ids_in_range(
                                request.request_id,
                                num_new_local_computed_tokens,
                                num_computed_tokens,
                            )
                        )
                    continue

                # SOURCE: vllm/v1/core/sched/scheduler.py:L1055-L1082（入
                #   running + 状态推进 + 在途 prefill 记账）
                self.running.append(request)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Only track requests that will still be prefilling after this chunk.
                if num_computed_tokens + num_new_tokens < request.num_tokens:
                    self._inflight_prefills.add(request)

            # re-queue requests skipped in this pass ahead of older skipped items.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1099-L1101
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

            # SUBTRACTED: DP prefill balancing 记账（L1103-L1106）。

        # Check if the scheduling constraints are satisfied.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1108-L1113
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs

        # SUBTRACTED: get_num_common_prefix_blocks（L1121-L1129——级联注意
        #   力面 → 后章）。

        # Construct the scheduler output.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1143-L1149（new reqs 数据）
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids()
            )
            for req in scheduled_new_reqs
        ]

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1151-L1158（cached reqs
        #   数据）
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            req_to_new_blocks,
        )

        # SUBTRACTED: prev_step_scheduled_req_ids（L1160-L1163——MRV1）。

        # Producer partial-tail hand-off for external KV connectors. Drained
        # before the CoW retentions are released below, so the pin lands while
        # the cow block still holds a retention ref. Without a producer-side
        # connector nothing consumes the hand-off, so skip the drain (and its
        # pin); the manager drops stale entries when the request's blocks are
        # popped for free.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1165-L1179 站 12③：m15
        pending_partial_tail_offloads = None
        if (
            self.connector is not None
            and self.vllm_config.kv_transfer_config is not None
            and self.vllm_config.kv_transfer_config.is_kv_producer
        ):
            pending_partial_tail_offloads = (
                self.kv_cache_manager.take_partial_tail_offloads() or None
            )

        # SUBTRACTED: CoW 拷贝打包段（L1181-L1190——第 10 条归 ch15）。
        # SUBTRACTED: dynamic spec K（L1192-L1197——ch33）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1208-L1229（SchedulerOutput
        #   装配——EC/spec 字段随删）
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            preempted_req_ids=self.reset_preempted_req_ids,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between the
            # previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            new_block_ids_to_zero=self._get_new_block_ids_to_zero(),
            partial_tail_offloads=pending_partial_tail_offloads,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1231-L1237 站 7：元数据
        #   过线（不透明计划挂上 SchedulerOutput）
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # SUBTRACTED: ECConnector meta（L1239-L1244——第 1 条）。

        # Advance the fence only for non-empty steps (those that actually
        # write KV and have their output processed later in update_from_output).
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1246-L1249 栅栏推进
        if self.defer_block_free and total_num_scheduled_tokens > 0:
            self.sched_step_seq += 1

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1251-L1253
        self._update_after_schedule(scheduler_output)
        return scheduler_output

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1255 _build_kv_connector_meta
    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1258
        return connector.build_connector_meta(scheduler_output)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1260 _get_new_block_ids_to_zero
    def _get_new_block_ids_to_zero(self) -> list[int] | None:
        # Drain new attention block ids every step so the manager-side list
        # does not grow unbounded; only kv-cache zeroing consumes them.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1262-L1264
        new_block_ids_to_zero = self.kv_cache_manager.take_new_block_ids()
        if not self.needs_kv_cache_zeroing:
            return None

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1267-L1272（站 6 的跳过
        #   过滤：清零账滤掉 _skip_zero_block_ids、每步即焚）
        if self._skip_zero_block_ids:
            skip = self._skip_zero_block_ids
            new_block_ids_to_zero = [b for b in new_block_ids_to_zero if b not in skip]
            skip.clear()

        return new_block_ids_to_zero or None

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1274 _preempt_request
    def _preempt_request(
        self, request: Request, timestamp: float, drop_stale_output: bool = False
    ) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.

        drop_stale_output: drop (rather than deliver) any in-flight output; used
        by reset_prefix_cache, whose same-step resume would otherwise deliver
        tokens out of order, and for connectors with a pending KV hand-off,
        which the preemption's block free would leave without valid KV.
        """
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1287-L1296
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self._free_request_blocks(request)
        self._inflight_prefills.discard(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        # Async scheduling: mark all in-flight output as stale. Its tokens are
        # still delivered on return (dropping them would perturb spec-decode
        # acceptance) but must not mutate the reset counters; each step drains
        # its share in update_from_output. num_in_flight_tokens already
        # includes any undrained stale share, so assign rather than accumulate.
        # An undrained drop-mode share stays dropped: its positions have
        # already been resampled.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1297-L1308（drop_stale_
        #   output 记账——m13 的在途产出账）
        request.drop_stale_output = drop_stale_output or (
            request.drop_stale_output and request.num_stale_output_tokens > 0
        )
        request.num_stale_output_tokens = request.num_in_flight_tokens
        request.num_output_placeholders = 0
        request.num_preemptions += 1
        # SUBTRACTED: PREEMPTED 事件（L1310-L1311——观测面）。

        # Put the request back to the waiting queue.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1313-L1315
        self.waiting.prepend_request(request)
        self.reset_preempted_req_ids.add(request.request_id)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1317 _update_after_schedule
    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g., spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1327-L1343
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.num_in_flight_tokens += num_scheduled_token
            if self.defer_block_free:
                # Record the in-flight step, to fence deferred block freeing.
                request.last_sched_seq = self.sched_step_seq
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            # Drop from the in-flight-prefill set once it's no longer prefilling.
            if not request.is_prefill_chunk:
                self._inflight_prefills.discard(request)

        # SUBTRACTED: routed_experts 快照（L1345-L1359——观测面）。

        # Clear the finished and preempted request IDs.
        # NOTE: We shouldn't just clear() here because it will also affect
        # the scheduler output.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1361-L1365
        self.finished_req_ids = set()
        self.reset_preempted_req_ids = set()

    # SOURCE: vllm/v1/core/sched/scheduler.py:_make_cached_request_data（切面：
    #   running/resumed 的差分载荷——ExampleConnector 的 resumed 支已删，
    #   常规差分面保留）
    def _make_cached_request_data(
        self,
        scheduled_running_reqs: list[Request],
        scheduled_resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        # SUBTRACTED: spec token 差分与 all_token_ids 回查（MRV1/PP 面）。
        # SOURCE: vllm/v1/core/sched/scheduler.py（req_ids/resumed/new_block_
        #   ids/num_computed/num_output 的差分装配骨架）
        req_ids = [req.request_id for req in scheduled_running_reqs]
        resumed_req_ids = {req.request_id for req in scheduled_resumed_reqs}
        new_block_ids: list[tuple[list[int], ...] | None] = [
            req_to_new_blocks[req.request_id].get_block_ids(allow_none=True)
            if req.request_id in req_to_new_blocks
            else None
            for req in scheduled_running_reqs + scheduled_resumed_reqs
        ]
        num_computed_tokens = [
            req.num_computed_tokens for req in scheduled_running_reqs + scheduled_resumed_reqs
        ]
        num_output_tokens = [
            req.num_output_tokens for req in scheduled_running_reqs + scheduled_resumed_reqs
        ]
        return CachedRequestData(
            req_ids=req_ids + [req.request_id for req in scheduled_resumed_reqs],
            resumed_req_ids=resumed_req_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    # ==============================
    # update_from_output：栅栏推进 + 失败消化 + 回传消化
    # ==============================

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1670 update_from_output
    #   （connector 面切面）
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> None:
        # SUBTRACTED: 返回的 EngineCoreOutputs 装配（dict[int, ...]——
        #   第 3 条观测/输出面归 ch02；本章消费侧直接消化状态）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1678
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        kv_connector_output = model_runner_output.kv_connector_output

        # Every GPU write enqueued by this and earlier steps has completed, so it is
        # safe to return deferred-free blocks to the pool.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1684-L1688 栅栏水位推进
        #   + drain
        if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
            self.processed_step_seq += 1
            self._drain_deferred_frees()

        # SUBTRACTED: perf_stats（L1690-L1692）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1697-L1705 invalid blocks
        #   消化（失败块 → 受影响请求识别与 num_computed 调整）
        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # SUBTRACTED: routed_experts 快照落账（L1707-L1726）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1731-L1732 主循环骨架
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            # SUBTRACTED: assert num_tokens_scheduled > 0（L1734——chunked
            #   prefill 面的守卫，本章预算面恒正）
            request = self.requests.get(req_id)
            output_is_stale = False
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1737-L1743（在途账
            #   排水：stale 份额按步锁步扣）
            if request is not None:
                request.num_in_flight_tokens -= num_tokens_scheduled
                # Drain any stale share (see _preempt_request) in lockstep.
                if request.num_stale_output_tokens > 0:
                    output_is_stale = True
                    request.num_stale_output_tokens -= num_tokens_scheduled
                    assert request.num_stale_output_tokens >= 0
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1744-L1746（失败/
            #   重排请求跳过）
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1747-L1755（已 finish
            #   的跳过——延迟释放者仍在 self.requests）
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                continue

            # Drop-mode stale output (same-step resume) is discarded entirely.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1757-L1759
            if output_is_stale and request.drop_stale_output:
                continue

            # SUBTRACTED: req_index/spec/pooler/grammar/logprobs 段
            #   （L1761-L1805——ch07/08/33 邻章）。

            # SOURCE: vllm/v1/core/sched/scheduler.py:L1797 停止判定骨架
            stopped = False
            new_token_ids = (
                model_runner_output.sampled_token_ids[
                    model_runner_output.req_id_to_index[req_id]
                ]
                if model_runner_output.sampled_token_ids
                else []
            )

            # Check for stop and update request status.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1807-L1811
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids, is_stale=output_is_stale
                )

            # SUBTRACTED: L1812-L1901 的输出装配/routed/logprobs 面。

            # SOURCE: vllm/v1/core/sched/scheduler.py:L1886-L1899（stopped →
            #   _free_request + 出队账）
            if stopped:
                self._free_request(request)

                if request.status == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

        # Remove the stopped requests from the running and waiting queues.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1946-L1952
        if stopped_running_reqs:
            self.running = [r for r in self.running if r not in stopped_running_reqs]
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.skipped_waiting.remove_requests(stopped_preempted_reqs)

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1954-L1972 error 收尾
        #   （grammar 段删；kv 失败 fail 策略 → FINISHED_ERROR）
        error_req_ids = set()
        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            error_req_ids.update(failed_kv_load_req_ids)

        if error_req_ids:
            self.finish_requests(error_req_ids, RequestStatus.FINISHED_ERROR)

        # KV Connector: update state for finished KV Transfers.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1974-L1976 站 9 入口
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # SUBTRACTED: stats 聚合/kv events 发布/EngineCoreOutputs 装配
        #   （L1978-L2048——第 3 条观测面）。

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2094 _update_request_with_output
    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        # is_stale is only used by the AsyncScheduler override.
        # Append generated tokens and check for stop. Note that if a
        # request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2101-L2111
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    # ==============================
    # 终局释放与步序栅栏
    # ==============================

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2237 finish_requests
    def finish_requests(
        self, request_ids, finished_status: RequestStatus
    ) -> list[Request]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        Returns:
            List of requests that were aborted. Will not include any that were
            already finished.
        """
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2251-L2257
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2259-L2283
        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2278-L2283
        if running_requests_to_remove:
            self.running = [r for r in self.running if r not in running_requests_to_remove]
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2285-L2296（WAITING_FOR_
        #   REMOTE_KVS 的延迟释放分支——传输未完成者押后放块）
        for request in valid_requests:
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)

        return valid_requests

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2300 _free_request——延迟
    #   释放决策点（m11）
    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        assert request.is_finished()

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2305-L2306（先问 connector
        #   接不接管）
        self._inflight_prefills.discard(request)
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        # SUBTRACTED: EC Connector 镜像钩子（L2308-L2315——第 1 条）。

        # SUBTRACTED: encoder_cache_manager.free（L2317——EC 面）。
        request_id = request.request_id
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2318-L2321
        self.finished_req_ids.add(request_id)

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2323-L2325（接管 → 块不
        #   释放、请求留在 self.requests——get_finished 报完成才 _free_
        #   blocks）
        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params, None

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2329 _free_blocks
    def _free_blocks(self, request: Request):
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2330-L2332
        assert request.is_finished()
        self._free_request_blocks(request)
        del self.requests[request.request_id]

    # SUBTRACTED: pause_state（L2334-L239——streaming 面归 ch02）。

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2341 _free_request_blocks——
    #   栅栏/直接 free 分流（m12 本体）
    def _free_request_blocks(self, request: Request):
        """Free the request's KV blocks, deferring the return to the block
        pool when an in-flight GPU step may still write them.
        """
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2345-L2354（账实分离：
        #   账先摘进 deferred_frees，物理归还等栅栏）
        if not self.defer_block_free or (
            # Last scheduled step already processed: no in-flight write remains
            # (always the case for a normal finish), so free now.
            request.last_sched_seq <= self.processed_step_seq
        ):
            self.kv_cache_manager.free(request)
            return
        blocks = self.kv_cache_manager.pop_blocks_for_free(request)
        if blocks:
            self.deferred_frees.append((self.sched_step_seq, blocks))

    # SUBTRACTED: _free_cow_retained_blocks（L2356-L2365——第 10 条 CoW
    #   保留块的栅栏归 ch15）。

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2367 _drain_deferred_frees
    def _drain_deferred_frees(self):
        """Return deferred blocks whose fence step has completed.

        Fences are appended in near-monotonic order (a CoW retention fence
        can lead request-free fences by one step), so stop at the first
        pending one; any satisfied entry behind it is merely freed later.
        """
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2374-L2380（过栅栏逆序
        #   归还——尾块先驱逐的 LRU 不变量在延迟路径同样成立）
        while self.deferred_frees:
            fence, _ = self.deferred_frees[0]
            if fence > self.processed_step_seq:
                break
            _, blocks = self.deferred_frees.popleft()
            # Free in reverse order so that the tail blocks are evicted first.
            self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2382 get_num_unfinished_requests
    def get_num_unfinished_requests(self) -> int:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2387-L2392（pause 段删
        #   ——streaming 面；waiting+skipped+running 三队列和）
        num_waiting = len(self.waiting) + len(self.skipped_waiting)
        return num_waiting + len(self.running)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2394 has_finished_requests——
    #   接管中的请求还占着 self.requests 的账（m11）
    def has_finished_requests(self) -> bool:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2395-L2396
        if self.finished_req_ids:
            return True
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2397-L2398
        if self.connector is None:
            return False
        # Finished requests waiting on delayed connector cleanup remain in
        # self.requests after they have been removed from scheduling queues.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2399-L2404
        num_in_queues = (
            len(self.waiting) + len(self.skipped_waiting) + len(self.running)
        )
        return len(self.requests) > num_in_queues

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2406 has_requests——push 型
    #   保活（m17）
    def has_requests(self) -> bool:
        # Override the interface default to also keep the engine alive while a
        # connector still has pending push work (e.g. push-mode WRITE transfers
        # in flight after all "live" requests have finished). Without this hook
        # the engine would quiesce before the connector can drain completions.
        # TODO: replace with a more general mechanism for connectors to keep
        # the scheduler alive.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2413-L2416（connector 的
        #   has_pending_push_work 也算活）
        return (
            self.get_num_unfinished_requests() > 0
            or self.has_finished_requests()
            or (self.connector is not None and self.connector.has_pending_push_work())
        )

    # SUBTRACTED: reset_prefix_cache/reset_connector_cache/reset_encoder_
    #   cache（L2423-L2493——ch15 m20/EC 面）。

    # ==============================
    # 终局交接与护轨预测
    # ==============================

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2577 _connector_finished——
    #   窗外回收 + 整块表交接（m11 的调度器侧半边）
    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2586-L2587
        if self.connector is None:
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector, on the processed-token basis (see `allocate_slots`).
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2589-L2597（窗外回收：
        #   processed-token 基准——在途步还在读的块不交）
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            processed_computed_tokens=max(
                0, request.num_computed_tokens - request.num_in_flight_tokens
            ),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2599-L2602（整块表交接：
        #   按 num_computed_tokens 裁剪）
        block_ids = self.kv_cache_manager.get_block_ids_for_computed_tokens(
            request_id=request.request_id,
            num_computed_tokens=request.num_computed_tokens,
        )

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2604-L2612（非 HMA 单组
        #   路径（TODO 明言待废）/ SupportsHMA 逐组交接）
        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2614 _request_remaining_blocks
    #   ——单请求预约预测（m5 的预测器：get_num_blocks_to_allocate 同源）
    def _request_remaining_blocks(self, request: Request) -> int:
        """Blocks `request` still needs to allocate to hold its full sequence."""
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2616-L2626（apply_
        #   admission_cap=True 与 full-ISL 门同源）
        full_num_tokens = min(request.num_tokens, self.max_model_len)
        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=request.num_computed_tokens,
            num_local_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
        )

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2628 _inflight_prefill_
    #   reserved_blocks——在途预约求和（防死锁护轨的 Σ）
    def _inflight_prefill_reserved_blocks(self) -> int:
        """Num blocks in-flight prefills still need to finish (their reservation)."""

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2631-L2633
        return sum(
            self._request_remaining_blocks(req) for req in self._inflight_prefills
        )

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2635 _update_waiting_for_
    #   remote_kv——补缓存 + 全命中退一 token + 失败分支（m9/m10 交汇点）
    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2643
        assert self.connector is not None

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2645-L2665 失败分支逐字
        #   （有效前缀补缓存 + 重算区补登记清零；无有效 token → free 全部）
        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
                if self.needs_kv_cache_zeroing:
                    # The failed load left the blocks beyond the valid
                    # prefix unwritten and their zeroing was skipped; zero
                    # them before they are recomputed locally.
                    self.kv_cache_manager.record_blocks_for_zeroing(
                        request.request_id, request.num_computed_tokens
                    )
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                # (Freed blocks are re-recorded for zeroing when
                # reallocated, so the skipped blocks need no handling.)
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L2666-L2669（补缓存：
            #   延迟入哈希表到此刻）
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            # SOURCE: vllm/v1/core/sched/scheduler.py:L2671-L2674（全命中
            #   退一 token——要 logits）
            if request.num_computed_tokens == request.num_tokens:
                request.num_computed_tokens = request.num_tokens - 1

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2676
        self.finished_recving_kv_req_ids.remove(request.request_id)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2678 _try_promote_blocked_
    #   waiting_request——提升回 WAITING/PREEMPTED（状态机闭环）
    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2682-L2693（WAITING_FOR_
        #   REMOTE_KVS 支逐字；grammar/streaming 两支删）
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # finished_recving_kv_req_ids is populated during
            # update_from_output(), based on worker-side connector signals
            # in KVConnectorOutput.finished_recving
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False
            self._update_waiting_for_remote_kv(request)
            if request.num_preemptions:
                request.status = RequestStatus.PREEMPTED
            else:
                request.status = RequestStatus.WAITING
            return True

        # SUBTRACTED: grammar/streaming 提升支（L2695-L2707——ch05/ch02）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2709-L2712（不可达守卫）
        raise AssertionError(
            "Unexpected blocked waiting status in promotion: "
            f"{request.status.name} for request {request.request_id}"
        )

    # SOURCE: vllm/v1/core/sched/scheduler.py:L~2050 _is_blocked_waiting_status
    def _is_blocked_waiting_status(self, status: RequestStatus) -> bool:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2052-L2056（blocked 三状态
        #   判定；grammar/streaming 两态随各章删——本章只立
        #   WAITING_FOR_REMOTE_KVS 一态）
        return status in (RequestStatus.WAITING_FOR_REMOTE_KVS,)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2714 _update_from_kv_xfer_
    #   finished——回收分流（m9/m11 交汇）
    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2725-L2726（调度器侧
        #   connector 先消化回传）
        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2728-L2737（finished_
        #   recving 分流：仍等待者入集、已 finish 者直接放块）
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2738-L2741（finished_
        #   sending 放块——producer 交接的终点）
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    # ==============================
    # 失败回滚
    # ==============================

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2743 _update_requests_with_
    #   invalid_blocks——第一个坏块截断 + 共享去重
    def _update_requests_with_invalid_blocks(
        self,
        requests,
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            num_scheduled_tokens: req_id -> number of scheduled tokens.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2774-L2844 截断循环逐字
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens

                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2846 _handle_invalid_blocks——
    #   双策入口（async/sync 分扫 + fail/recompute）
    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2855
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2857-L2870
        async_load_reqs = (
            req
            for req in self.skipped_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=False,
            )
        )

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2872-L2873
        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2875-L2883
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
            )
        )

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2882-L2883
        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2885-L2886（无人受影响
        #   → 空）
        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2888-L2892
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2894-L2903（fail 默认：
        #   整请求 FINISHED_ERROR）
        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2905-L2910
        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2912-L2915
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids


# check_stop（模块级停判：到达 max_model_len 即停——本章消费面）
# SOURCE: vllm/v1/core/sched/scheduler.py check_stop
def check_stop(request: Request, max_model_len: int | None = None) -> bool:
    # SUBTRACTED: stop_token/stop_str/length_cap 面（ch02 采样面）——
    #   max_model_len 截断即停（本章测试的最小停判）
    if max_model_len is not None and request.num_tokens >= max_model_len:
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True
    return False
