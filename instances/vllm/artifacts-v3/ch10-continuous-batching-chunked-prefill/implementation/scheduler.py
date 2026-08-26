# SOURCE: vllm/v1/core/sched/scheduler.py
# 连续批处理调度器（只做减法精简版）。与真实 vllm/v1/core/sched/scheduler.py
# 同名同结构同控制流：schedule() 两阶段（先 RUNNING 后 WAITING）按单一
# token_budget 分账；_update_after_schedule 乐观推进 num_computed_tokens；
# _make_cached_request_data 产增量下发数据。删除项全部 dossier.subtraction_plan
# .delete 批准（KVConnector/encoder/structured/mamba/LoRA/PRIORITY/DP/PP+V2/
# spec 统计/streaming/可观测性/async 残留/skipped 阻塞态细节/pause 深挖）。
from __future__ import annotations

import itertools
import time

from .interface import PauseState, SchedulerInterface
from .kv_cache_manager import KVCacheBlocks, KVCacheManager
from .output import CachedRequestData, NewRequestData, SchedulerOutput
from .request import Request, RequestStatus
from .request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from .scheduler_config import SchedulerConfig


# SUBTRACTED: record_function_or_nullcontext（torch profiler 探针，dossier.delete
#   第 11 条批准）——原 scheduler.py 用它包 allocate_slots 等热点；这里以空上下文
#   顶替，控制流与缩进不变。
class _nullctx:
    # SOURCE: vllm/v1/core/sched/scheduler.py（record_function_or_nullcontext 包裹）
    def __enter__(self):
        # SOURCE: vllm/v1/utils.py record_function_or_nullcontext
        return None

    def __exit__(self, *args):
        # SOURCE: vllm/v1/utils.py record_function_or_nullcontext
        return False


# SOURCE: vllm/v1/utils.py record_function_or_nullcontext
def record_function_or_nullcontext(_name: str) -> _nullctx:
    # SOURCE: vllm/v1/utils.py record_function_or_nullcontext（探针 → 空上下文）
    return _nullctx()


# SOURCE: vllm/v1/core/sched/scheduler.py:L69 Scheduler
class Scheduler(SchedulerInterface):
    # SOURCE: vllm/v1/core/sched/scheduler.py:L70 __init__
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        max_model_len: int,
        num_gpu_blocks: int = 1 << 30,
        block_size: int = 16,
        num_spec_tokens: int = 0,
        log_stats: bool = False,
    ) -> None:
        # SUBTRACTED: 真实 __init__ 从 VllmConfig/KVCacheConfig/StructuredOutput
        #   Manager 装配几十个字段（L70-L360：kv events/connector/ec_connector/
        #   encoder 预算/LoRA/routed experts/perf metrics/mamba/DCP-PCP/defer
        #   fences/use_pp/use_v2_model_runner/prefill_capacity_bound 等）——均为
        #   dossier.delete 批准的子系统或邻章精简版范围；这里以裸标量承载同一批
        #   调度约束字段。
        self.scheduler_config = scheduler_config
        self.log_stats = log_stats
        # SOURCE: vllm/v1/core/sched/scheduler.py:L120-L123
        # Diffusion models may not sample any tokens for a denoising step.
        self.num_sampled_tokens_per_step = 1
        # SUBTRACTED: diffusion 模型 is_diffusion 特判置 0（L121-L123）——
        #   diffusion 属另章；本章恒 1。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L246-L247 spec 配置
        self.num_spec_tokens = num_spec_tokens
        self.num_lookahead_tokens = 0
        # SUBTRACTED: eagle/dflash/dspark 的 lookahead 变体与 dynamic_sd_lookup
        #   （L248-L270，spec 统计，dossier.delete 第 9 条批准——ch33 话头）。

        # Scheduling constraints.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L109
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        # SOURCE: vllm/v1/core/sched/scheduler.py:L110-L114 缺省回落 batched
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens is not None
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = max_model_len
        # SUBTRACTED: enable_kv_cache_events（L116-L119，kv events 可观测性）。

        # Create the KV cache manager.
        # SUBTRACTED: 真实按 kv_cache_config/hash_block_size/watermark 构造
        #   KVCacheManager 并 bind connector 块池（L272-L294）——分页池内部归
        #   ch13/14；这里换接口契约面（签名与『满则 None』语义一致）。
        self.kv_cache_manager = KVCacheManager(
            num_gpu_blocks=num_gpu_blocks,
            block_size=block_size,
            max_model_len=max_model_len,
        )

        # req_id -> Request
        # SOURCE: vllm/v1/core/sched/scheduler.py:L177-L178
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        # SOURCE: vllm/v1/core/sched/scheduler.py:L180-L185
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L187
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L189
        self.skipped_waiting = create_request_queue(self.policy)
        # SOURCE: vllm/v1/core/sched/scheduler.py:L190
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

        # Track requests scheduled in prior step (MRV1-only).
        # SOURCE: vllm/v1/core/sched/scheduler.py:L105-L106
        self.prev_step_scheduled_req_ids: set[str] = set()

        # Scheduler iteration counter. Drives the V2+PP+async decode-throttle
        # cadence (`next_decode_eligible_step`).
        # SOURCE: vllm/v1/core/sched/scheduler.py:L298-L300
        self.current_step = 0

        # SOURCE: vllm/v1/core/sched/scheduler.py:L305-L307
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl
        )

        # SOURCE: vllm/v1/core/sched/scheduler.py:L356
        self._pause_state: PauseState = PauseState.UNPAUSED

        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads). Their remaining-block reservation gates async loads.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L358-L360
        self._inflight_prefills: set[Request] = set()

        # SUBTRACTED: connector/finished_recving_kv/grammar 错误集合/encoder
        #   cache/speculative_config 细节/use_pp/use_v2_model_runner/prefill_
        #   capacity_bound/mamba/needs_kv_cache_zeroing/sched_step_seq fence/
        #   defer_block_free/requires_kv_delivery/kv_event_publisher 等
        #   （L102-L104、L125-L168、L196-L354 其余）——dossier.delete 批准。

    # ------------------------------------------------------------------ #
    # 入队
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2213 add_request
    def add_request(self, request: Request) -> None:
        # SUBTRACTED: 重复 request_id 的流式会话续跑分支（L2214-L2226：
        #   streaming_queue/_update_request_as_session，dossier.delete 第 10 条
        #   批准）、resumable 的 streaming_queue 建队（L2228-L2229）、connector
        #   .on_new_request（L2232-L2233，第 1 条）与 log_stats 的 record_event
        #   (QUEUED)（L2234-L2235，第 11 条）——一次性请求走 else 主路径：
        #   入队 + 登记即完整正确。
        self._enqueue_waiting_request(request)
        self.requests[request.request_id] = request

    # ------------------------------------------------------------------ #
    # schedule —— 连续批处理一拍（两阶段按单一 token 预算分账）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L439 schedule
    def schedule(self) -> SchedulerOutput:
        # SUBTRACTED: throttle_prefills 参数（DP prefill balancing，dossier.delete
        #   第 7 条批准——单机部署恒 False）。
        self.current_step += 1
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # SUBTRACTED: encoder_compute_budget / scheduled_encoder_inputs 初始化
        #   （L464-L466，encoder，dossier.delete 第 2 条批准）。
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        # Whether the running batch contains any prefill requests.
        prefill_scheduled = False

        # For logging.
        scheduled_timestamp = time.monotonic()

        # SOURCE: vllm/v1/core/sched/scheduler.py:L475
        self.kv_cache_manager.new_step_starts()

        # SUBTRACTED: defer_prefills 的 DP prefill balancing 计算（L477-L481，
        #   dossier.delete 第 7 条批准——throttle_prefills 恒 False）。

        # First, schedule the RUNNING requests.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L483-L485
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            # SUBTRACTED: async 调度的提前剪枝（L488-L502：num_output_
            #   placeholders>0 且必达 max_tokens 则跳过——同步版占位恒 0，
            #   dossier.delete 第 12 条批准，占位机制归 ch12）、
            #   next_decode_eligible_step 的 V2+PP+async 步距 continue
            #   （L504-L508，第 8 条批准）、defer_prefills 的续 chunk 延后
            #   continue（L510-L514，第 7 条批准）。

            # SOURCE: vllm/v1/core/sched/scheduler.py:L516-L532 追赶公式 + 双钳制
            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )

            # SUBTRACTED: _try_schedule_encoder_inputs 的 encoder 预算截断
            #   （L534-L550，第 2 条批准）与 _mamba_block_aligned_split 的
            #   mamba 对齐切块（L552-L555，第 4 条批准）。

            # SOURCE: vllm/v1/core/sched/scheduler.py:L557-L573
            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L576-L629 抢占重试环
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    # SUBTRACTED: PRIORITY 策略的 max(running, key=(priority,
                    #   arrival_time)) 选择与已调度回滚（L590-L613，dossier.delete
                    #   第 6 条批准——默认 policy=fcfs，走抢队尾即可讲清
                    #   『RUNNING 可抢占』；PRIORITY 是同构变体，ch11 可再提）。
                    preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L627-L629
                break

            # Schedule the request.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L631-L638
            scheduled_running_reqs.append(request)
            prefill_scheduled |= request.is_prefill_chunk
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # SUBTRACTED: spec decode 的 scheduled_spec_decode_tokens 登记
            #   （L640-L656，dossier.delete 第 9 条批准——簿记细节归 ch33；
            #   追赶公式里的 spec 项与占位项已保留）与 encoder 缓存 allocate
            #   （L658-L671，第 2 条批准）。

        # SUBTRACTED: scheduled_loras 的 LoRA 记账与 max_loras 约束
        #   （L673-L681，dossier.delete 第 5 条批准）。

        # Next, schedule the WAITING requests.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L683-L684 守卫
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            # SUBTRACTED: step_skipped_waiting 的收集与回插（L685 创建与
            #   L1099-L1101 回插，dossier.delete 第 13 条批准）——阻塞态全部
            #   来自已删子系统（connector/grammar/streaming），删后 skipped_
            #   waiting 恒空但双队列结构本身保留（ch11 展开）。

            # SOURCE: vllm/v1/core/sched/scheduler.py:L687-L692
            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                # SUBTRACTED: 流式会话的 num_waiting_for_streaming_input 计数
                #   （L688-L690，dossier.delete 第 10 条批准）。
                num_running = len(self.running)
                if num_running >= self.max_num_running_reqs:
                    break

                # SOURCE: vllm/v1/core/sched/scheduler.py:L694-L695
                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                # SOURCE: vllm/v1/core/sched/scheduler.py:L697-L698
                request = request_queue.peek_request()
                request_id = request.request_id

                # SUBTRACTED: 阻塞态提升 _try_promote_blocked_waiting_request
                #   与 WAITING_FOR_REMOTE_KVS 的 debug 日志（L700-L711，
                #   dossier.delete 第 13 条批准）、stale 输出在途跳过
                #   （L713-L722：num_stale_output_tokens 只在 async 抢占时>0，
                #   随第 12 条批准的 async 残留删除——stale 输出语义归 ch11）、
                #   LoRA max_loras 约束（L724-L737，第 5 条批准）。

                # SUBTRACTED: connector 外部命中簿记初始化（L739-L742：
                #   num_external_computed_tokens/load_kv_async/connector_
                #   prefix_cache_queries/did_prefix_cache_lookup，第 1 条批准）。

                # Get already-cached tokens.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L744-L766 前缀命中折算
                if request.num_computed_tokens == 0:
                    # SUBTRACTED: connector 分支 get_computed_blocks_for_
                    #   connector（L749-L759，hybrid-aware 查找，第 1 条批准
                    #   ——P/D 场景 ch36 展开，默认 connector=None 走 else）。
                    (
                        new_computed_blocks,
                        num_new_local_computed_tokens,
                        # Marconi shared-prefix junction to pin; 0 if none.
                        request.shared_prefix_boundary,
                    ) = self.kv_cache_manager.get_computed_blocks(request)

                    # SUBTRACTED: KVConnector 外部命中合并（L768-L826，第 1 条
                    #   批准）与 prefill_stats.set（L846-L853，第 11 条批准）。
                    # Total computed tokens (local + external).
                    # SOURCE: vllm/v1/core/sched/scheduler.py:L828-L832
                    num_computed_tokens = num_new_local_computed_tokens
                    assert num_computed_tokens <= request.num_tokens
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    # SOURCE: vllm/v1/core/sched/scheduler.py:L854-L859
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                # SUBTRACTED: encoder 变量与 pad_spec_decode 初始化
                #   （L861-L864，第 2/9 条批准）、load_kv_async 的零 token 分支
                #   与 defer_prefills 的 prefill 延后 break（L866-L873，
                #   第 1/7 条批准）——if/elif 链坍缩为原 else 主路径。

                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed
                # requests, which have output tokens.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L874-L879
                num_new_tokens = request.num_tokens - num_computed_tokens

                # SUBTRACTED: spec 均匀 pad 分支（L881-L897：'Prefer to not
                #   schedule than schedule un-padded'——保 CUDA graph 的
                #   num_spec_tokens pad，默认 0 不触发，dossier.delete 第 9 条
                #   批准——WC2 代价清单的源码自供证据，正文引用原文）。

                # SOURCE: vllm/v1/core/sched/scheduler.py:L899-L901 threshold 钳制
                threshold = self.scheduler_config.long_prefill_token_threshold
                if 0 < threshold < num_new_tokens:
                    num_new_tokens = threshold

                # chunked prefill has to be enabled explicitly to allow
                # pooling requests to be chunked
                # SOURCE: vllm/v1/core/sched/scheduler.py:L903-L911 chunked 开关
                if (
                    not self.scheduler_config.enable_chunked_prefill
                    and num_new_tokens > token_budget
                ):
                    # If chunked_prefill is disabled,
                    # we can stop the scheduling here.
                    break

                # SOURCE: vllm/v1/core/sched/scheduler.py:L913-L914
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # SUBTRACTED: WAITING 侧 encoder inputs 调度（L916-L932，第 2
                #   条批准）与 mamba 对齐切块（L934-L943，第 4 条批准）、
                #   load_kv_async 的 lookahead 延后（L945-L951，第 1 条批准）、
                #   编码器交叉注意力块数（L953-L963，第 2 条批准）、async KV
                #   load 的 reserved_blocks 在途预约（L965-L971，第 1 条批准）。

                # SOURCE: vllm/v1/core/sched/scheduler.py:L973-L985 准入
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                    # SUBTRACTED: num_external_computed_tokens /
                    #   delay_cache_blocks / num_encoder_tokens /
                    #   reserved_blocks 四参（connector/encoder，第 1/2 条批准）。
                    full_sequence_must_fit=self.scheduler_reserve_full_isl,
                    has_scheduled_reqs=bool(self.running),
                )

                # SOURCE: vllm/v1/core/sched/scheduler.py:L987-L994 None → break
                if new_blocks is None:
                    # The request cannot be scheduled.

                    # SUBTRACTED: encoder_cache_manager.free 的 un-touch
                    #   （L990-L993，encoder，第 2 条批准）。
                    # WAITING 阶段绝不触发抢占——与 RUNNING 侧抢占环的反差。
                    break

                # SUBTRACTED: connector update_state_after_alloc 与
                #   connector_prefix_cache_stats 记录（L996-L1014，第 1 条批准）、
                #   record_prefix_cache_stats（L1016-L1020，第 11 条批准）。

                # SOURCE: vllm/v1/core/sched/scheduler.py:L1022 出队
                request = request_queue.pop_request()
                # SUBTRACTED: load_kv_async 的 WAITING_FOR_REMOTE_KVS 整段
                #   （L1023-L1053，第 1 条批准——默认 connector=None 时直接
                #   走 self.running.append）。

                # SOURCE: vllm/v1/core/sched/scheduler.py:L1055 入 running
                self.running.append(request)
                # SUBTRACTED: log_stats 的 record_event(SCHEDULED)
                #   （L1056-L1059，第 11 条批准）。
                # SOURCE: vllm/v1/core/sched/scheduler.py:L1060-L1065 状态分流
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                # SUBTRACTED: LoRA 登记（L1067-L1068，第 5 条批准）。
                # SOURCE: vllm/v1/core/sched/scheduler.py:L1069-L1075 落位记账
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # SUBTRACTED: pad_spec_decode 的 -1 占位登记（L1076-L1079，
                #   第 9 条批准）。
                # Only track requests that will still be prefilling after this chunk.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L1080-L1082
                if num_computed_tokens + num_new_tokens < request.num_tokens:
                    self._inflight_prefills.add(request)
                # SUBTRACTED: encoder 缓存 allocate（L1083-L1097，第 2 条批准）、
                #   prefill_capacity_bound 记录（L1103-L1106，第 7 条批准）。

        # Check if the scheduling constraints are satisfied.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1108-L1119 守恒断言
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # SUBTRACTED: num_common_prefix_blocks 的分组展开与 get_num_common_
        #   prefix_blocks 实算（L1121-L1129——cascade attention 的消费侧在模型
        #   章；单 KV 组全注意力下恒为 [0]，调度决策不读它）。
        num_common_prefix_blocks = [0]

        # Construct the scheduler output.
        # SUBTRACTED: use_v2_model_runner 的 resumed 合并分支（L1132-L1142，
        #   dossier.delete 第 8 条批准——V2 runner 是实验路径，ch18 话头）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1144-L1149 new 全量打包
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids()
            )
            for req in scheduled_new_reqs
        ]

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1151-L1158 cached 增量打包
        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step (MRV1-only).
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1160-L1163
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        # SUBTRACTED: connector partial-tail offload / CoW 块拷贝收割与
        #   _free_cow_retained_blocks（L1165-L1190，第 1 条批准）、动态 spec
        #   的 num_spec_tokens_to_schedule（L1192-L1197，第 9 条批准）、
        #   scheduled_encoder_input_stats（L1199-L1206，第 2/11 条批准）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1208-L1229 SchedulerOutput 落袋
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids=self.reset_preempted_req_ids,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            # SUBTRACTED: free_encoder_mm_hashes / new_block_ids_to_zero /
            #   kv_cache_block_copies / partial_tail_offloads /
            #   num_spec_tokens_to_schedule / ec_manager_metadata 六参
            #   （encoder/mamba/connector/spec 统计字段，dossier.delete 批准
            #   ——默认 None/空，不影响调度决策）。
        )

        # SUBTRACTED: connector/ec_connector 的 metadata 注入（L1231-L1244，
        #   第 1/2 条批准）、defer_block_free 的 sched_step_seq 推进
        #   （L1246-L1249，第 12 条批准）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1251-L1253
        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    # ------------------------------------------------------------------ #
    # 抢占（细节深挖归 ch11；本章只到『allocate_slots None → 抢队尾』因果层）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L1274 _preempt_request
    def _preempt_request(
        self, request: Request, timestamp: float, drop_stale_output: bool = False
    ) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        # SUBTRACTED: drop_stale_output 的 docstring 说明（connector 同步恢复/
        #   reset_prefix_cache 场景）——参数保留原签名默认 False；调用侧的
        #   requires_kv_delivery 判据随 connector 删除。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1287-L1289
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self._free_request_blocks(request)
        # SUBTRACTED: encoder_cache_manager.free（L1291，encoder，第 2 条批准）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1292-L1296
        self._inflight_prefills.discard(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        # SUBTRACTED: async 在途输出的 stale 标记（L1297-L1308：drop_stale_
        #   output/num_stale_output_tokens/num_output_placeholders=0——同步版
        #   三者恒 False/0/0，置零是无操作语义；dossier.delete 第 12 条批准，
        #   占位机制归 ch12、抢占 stale 输出归 ch11）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1309
        request.num_preemptions += 1
        # SUBTRACTED: log_stats 的 record_event(PREEMPTED)（L1310-L1311，
        #   第 11 条批准——timestamp 参数保留原签名）。

        # Put the request back to the waiting queue.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1313-L1315
        self.waiting.prepend_request(request)
        self.reset_preempted_req_ids.add(request.request_id)

    # ------------------------------------------------------------------ #
    # 调度后乐观推进
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L1317 _update_after_schedule
    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1330-L1331
            request.num_computed_tokens += num_scheduled_token
            request.num_in_flight_tokens += num_scheduled_token
            # SUBTRACTED: defer_block_free 的 last_sched_seq 记录（L1332-L1334，
            #   async 残留，dossier.delete 第 12 条批准）。
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1335-L1337
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            # SUBTRACTED: has_structured_output_requests 的 |= 累计（L1338-L1340，
            #   structured，dossier.delete 第 3 条批准）。
            # Drop from the in-flight-prefill set once it's no longer prefilling.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1341-L1343
            if not request.is_prefill_chunk:
                self._inflight_prefills.discard(request)

        # SUBTRACTED: routed experts 的块 id 快照（L1345-L1359，可观测性，
        #   dossier.delete 第 11 条批准）。

        # Clear the finished and preempted request IDs.
        # NOTE: We shouldn't just clear() here because it will also affect
        # the scheduler output.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1361-L1365 换新不 clear
        self.finished_req_ids = set()
        self.reset_preempted_req_ids = set()

    # ------------------------------------------------------------------ #
    # 增量打包
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L1410 _make_cached_request_data
    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1426-L1427
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # SUBTRACTED: use_pp 的采样 token 回传分支（L1430-L1445，PP，
            #   dossier.delete 第 8 条批准——非 PP 时 new_token_ids 恒空，
            #   worker 自己缓存采样 token）。
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1446-L1447
            if idx >= num_running_reqs:
                resumed_req_ids.add(req_id)
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1448-L1450 prev_step 判定
            # （use_v2_model_runner 条件随第 8 条批准拆除，判定本体保留）
            if req_id not in self.prev_step_scheduled_req_ids:
                all_token_ids[req_id] = req.all_token_ids.copy()
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1451-L1457
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1459-L1467
        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    # ------------------------------------------------------------------ #
    # waiting/skipped 双队列路由（结构保留；阻塞态细节归 ch11）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2050-L2056 _is_blocked_waiting_status
    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2052-L2056（保留原判——
        # 对已删子系统状态无副作用：本章请求永远不进这三种状态）
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2058 _enqueue_waiting_request
    def _enqueue_waiting_request(self, request: Request) -> None:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2059-L2062
        if self._is_blocked_waiting_status(request.status):
            self.skipped_waiting.add_request(request)
        else:
            self.waiting.add_request(request)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2064 _select_waiting_queue_for_scheduling
    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2065-L2066
        if self.policy == SchedulingPolicy.FCFS:
            return self.skipped_waiting or self.waiting or None

        # SUBTRACTED: PRIORITY 模式的双队头比较分支（L2068-L2072，
        #   dossier.delete 第 6 条批准——本精简版 policy 恒 FCFS，走不到）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2074
        return self.waiting or self.skipped_waiting or None

    # ------------------------------------------------------------------ #
    # 块归还
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2341 _free_request_blocks
    def _free_request_blocks(self, request: Request):
        """Free the request's KV blocks, deferring the return to the block
        pool when an in-flight GPU step may still write them.
        """
        # SUBTRACTED: defer_block_free 的延迟释放栅栏（L2345-L2354：
        #   last_sched_seq/processed_step_seq/deferred_frees——async 调度残留，
        #   dossier.delete 第 12 条批准）；同步版直接归还。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2350
        self.kv_cache_manager.free(request)

    # SUBTRACTED: update_from_output/update_draft_token_ids*/finish_requests/
    #   _free_request/get_num_unfinished_requests/has_finished_requests/
    #   _handle_stopped_request/_update_request_with_output/make_stats/
    #   set_pause_state/reset_prefix_cache/get_request_counts/shutdown 等
    #   （L1367-L2048、L2076-L2211、L2237-L2340、L2382-L2627）——⑤ 拍状态推进
    #   与停止判定归 ch9/ch11 的精简版；可观测性与已删子系统的钩子随批删。
