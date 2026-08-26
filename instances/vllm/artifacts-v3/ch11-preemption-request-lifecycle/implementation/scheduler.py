# SOURCE: vllm/v1/core/sched/scheduler.py
# 抢占与请求的一生（只做减法精简版）。与真实 vllm/v1/core/sched/scheduler.py
# 同名同结构同控制流，两条主线全保留：
#   段一 抢占与恢复——RUNNING 抢占重试环（L575-L629）/ _preempt_request 六件事
#   （L1274-L1315，stale 标记全保留）/ 守卫关闸（L683-L684）/ 双队列遍历 + 阻塞
#   跳过 + stale 推迟（L687-L722）/ 前缀重命中（L744-L766）/ 水位准入
#   （L973-L985）/ 回流落位 + resumed 分流（L1055-L1075）；
#   段二 一生的收尾——update_from_output 热循环（L1670-L2048）/ 逐 token
#   _update_request_with_output（L2094-L2111）/ check_stop 五连判（utils.py）/
#   finish_reason 时序 + 停止分流（L1895-L1907）/ 批量摘除（L1946-L1952）/
#   _free_request/_free_blocks 终点（L2300-L2338）/ finish_requests 外部死法
#   （L2237-L2298）。
# 删除项全部 dossier.subtraction_plan.delete 批准（PRIORITY 分支/encoder/
# connector/structured/streaming/spec/async+PP 基建/mamba/reset_prefix_cache/
# 观测统计/PAUSED——共 11 条）。LoRA 分支未获删除批准，原样保留（lora_config
# 默认 None 时全为旁路）。
from __future__ import annotations

import itertools
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .engine import EngineCoreOutput, EngineCoreOutputs
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
from .utils import check_stop, remove_all


# SUBTRACTED: 真实 record_function_or_nullcontext 包 torch profiler 探针
#   （vllm/v1/utils.py）——dossier.delete 第 10 条（观测/统计）批准；这里以空
#   上下文顶替，控制流与缩进不变。
class _nullctx:
    # SOURCE: vllm/v1/utils.py record_function_or_nullcontext（探针 → 空上下文）
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
        log_stats: bool = False,
    ) -> None:
        # SUBTRACTED: 真实 __init__ 从 VllmConfig/KVCacheConfig/StructuredOutput
        #   Manager 装配几十个字段（L70-L360：kv events/connector/ec_connector/
        #   encoder 预算/LoRA config/routed experts/perf metrics/mamba/DCP-PCP/
        #   defer fences/use_pp/use_v2_model_runner/prefill_capacity_bound/
        #   grammar 错误集合/finished_recving_kv 等）——均为 dossier.delete
        #   批准的子系统；这里以裸标量承载同一批调度约束字段。
        self.scheduler_config = scheduler_config
        self.log_stats = log_stats
        # SOURCE: vllm/v1/core/sched/scheduler.py:L120-L123
        # Diffusion models may not sample any tokens for a denoising step.
        self.num_sampled_tokens_per_step = 1
        # SUBTRACTED: diffusion 模型 is_diffusion 特判置 0（L121-L123）——
        #   diffusion 属另章；本章恒 1。

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
        # SUBTRACTED: 真实按 kv_cache_config/hash_block_size 构造并 bind
        #   connector 块池（L272-L294）——契约面口径（kv_cache_manager.py 头注）；
        #   watermark 原样传入（L289）——本章 introduces 的旋钮。
        self.kv_cache_manager = KVCacheManager(
            num_gpu_blocks=num_gpu_blocks,
            block_size=block_size,
            max_model_len=max_model_len,
            watermark=self.scheduler_config.watermark,
        )

        # SUBTRACTED: lora_config 从 VllmConfig 取（L84）——LoRA 分支保留原判
        #   （L673-L681/L724-L737/L1067-L1068，未获删除批准），默认 None 旁路。
        self.lora_config = None

        # req_id -> Request
        # SOURCE: vllm/v1/core/sched/scheduler.py:L177-L185
        self.requests: dict[str, Request] = {}
        # Scheduling policy
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
        # SUBTRACTED: next_decode_eligible_step 的步距判定（L504-L508——V2+PP+
        #   async，dossier.delete 第 7 条批准）；计数器本身保留。

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

        # SUBTRACTED: connector/ec_connector/requires_kv_delivery/
        #   finished_recving/failed_recving_kv_req_ids/encoder 预算与缓存/
        #   num_spec_tokens/num_lookahead_tokens/dynamic_sd_lookup/
        #   has_mamba_layers/sched_step_seq/deferred_frees/perf_metrics/
        #   routed_experts/grammar_compile_error_reqs/num_waiting_for_
        #   streaming_input/finished_req_ids_dict 等（L101-L176、L196-L354 其余）
        #   ——dossier.delete 第 1/2/4/5/6/7/10 条批准的子系统字段。

    # ------------------------------------------------------------------ #
    # 入队
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2213 add_request
    def add_request(self, request: Request) -> None:
        # SUBTRACTED: 重复 request_id 的流式会话续跑分支（L2214-L2226——
        #   streaming_queue/_update_request_as_session/finish_requests 收尾，
        #   dossier.delete 第 5 条批准）、resumable 的 streaming_queue 建队
        #   （L2228-L2229）、connector.on_new_request（L2232-L2233——第 1 条）、
        #   log_stats 的 record_event(QUEUED)（L2234-L2235——第 10 条）——
        #   一次性请求走 else 主路径：入队 + 登记即完整正确。
        self._enqueue_waiting_request(request)
        self.requests[request.request_id] = request

    # ------------------------------------------------------------------ #
    # schedule —— 一拍：两阶段分账（RUNNING 先行 + WAITING 守卫）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L439 schedule
    def schedule(self) -> SchedulerOutput:
        # SUBTRACTED: throttle_prefills 参数（DP prefill balancing，dossier.
        #   delete 第 11 条批准）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L440
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

        # SOURCE: vllm/v1/core/sched/scheduler.py:L452-L455
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        # SOURCE: vllm/v1/core/sched/scheduler.py:L457-L459
        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # SUBTRACTED: PAUSED_ALL 的 token_budget=0 短路（L460-L462——暂停机制，
        #   dossier.delete 第 11 条批准；守卫里的 UNPAUSED 比较保留）。

        # SUBTRACTED: encoder_compute_budget / scheduled_encoder_inputs 初始化
        #   （L464-L466——encoder，第 2 条批准）。
        # Spec decode-related.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L467-L468（簿记字典保留，
        #   登记分支随第 6 条删）
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        # Whether the running batch contains any prefill requests.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L469-L470
        prefill_scheduled = False

        # For logging.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L472-L473
        scheduled_timestamp = time.monotonic()

        # SOURCE: vllm/v1/core/sched/scheduler.py:L475
        self.kv_cache_manager.new_step_starts()

        # SUBTRACTED: defer_prefills 的 DP prefill balancing 计算（L477-L481
        #   ——第 7/11 条批准）。

        # First, schedule the RUNNING requests.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L483-L485
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            # SUBTRACTED: async 调度的提前剪枝（L488-L502：num_output_
            #   placeholders>0 且必达 max_tokens 则跳过——同步版占位恒 0，
            #   dossier.delete 第 6 条批准，占位机制归 ch12）、
            #   next_decode_eligible_step 的 V2+PP+async 步距 continue
            #   （L504-L508——第 7 条）、defer_prefills 的续 chunk 延后
            #   continue（L510-L514——第 11 条）。

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
            #   （L534-L550——第 2 条）与 _mamba_block_aligned_split 的 mamba
            #   对齐切块（L552-L555——第 8 条）。

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
            # SOURCE: vllm/v1/core/sched/scheduler.py:L575-L629 抢占重试环（核心一）
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    # SUBTRACTED: num_lookahead_tokens 实参（第 6 条——
                    #   spec lookahead 已删，走默认 0）。
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    # SUBTRACTED: PRIORITY 策略分支（L590-L613——max(running,
                    #   key=(priority, arrival_time)) 选择 + 被抢者本拍已领的
                    #   token/块/预算/encoder 回滚，dossier.delete 第 1 条批准；
                    #   默认 policy='fcfs' 走抢队尾——FCFS 队尾必是本拍未调度者，
                    #   无需回滚）。
                    # SOURCE: vllm/v1/core/sched/scheduler.py:L614-L615 FCFS 抢队尾
                    preempted_req = self.running.pop()

                    # SUBTRACTED: drop_stale_output=self.requires_kv_delivery
                    #   实参（L617-L620——connector 来源，第 1 条批准；参数本身
                    #   保留默认 False）。
                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        # SOURCE: vllm/v1/core/sched/scheduler.py:L623-L625
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
            #   （L640-L656——第 6 条批准，簿记细节归 ch33）与 encoder 缓存
            #   allocate（L658-L671——第 2 条）。

        # Record the LoRAs in scheduled_running_reqs
        # SOURCE: vllm/v1/core/sched/scheduler.py:L673-L681（未获删除批准，原样保留）
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Next, schedule the WAITING requests.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L683-L684 守卫（核心：本拍
        # 抢占过 = 内存紧张信号 → 整拍不收新）
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            # SOURCE: vllm/v1/core/sched/scheduler.py:L685 步内跳过收集队列
            step_skipped_waiting = create_request_queue(self.policy)

            # SOURCE: vllm/v1/core/sched/scheduler.py:L687-L692
            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
                # in `running` but still hold a model-runner request slot.
                # SUBTRACTED: num_waiting_for_streaming_input 计数（L690——
                #   流式，第 5 条批准）。
                num_running = len(self.running)
                if num_running >= self.max_num_running_reqs:
                    break

                # SOURCE: vllm/v1/core/sched/scheduler.py:L694-L695
                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                # SOURCE: vllm/v1/core/sched/scheduler.py:L697-L698
                request = request_queue.peek_request()
                request_id = request.request_id

                # try to promote blocked statuses while traversing skipped queue.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L700-L711 阻塞态跳过
                # （m6 防队头阻塞：promote 失败 → pop 进 step_skipped_waiting）
                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):
                    # SUBTRACTED: WAITING_FOR_REMOTE_KVS 的 debug 日志（L704-
                    #   L708——connector，第 1 条）。
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # SOURCE: vllm/v1/core/sched/scheduler.py:L713-L722 stale 在途
                # 推迟一拍（m4：现在恢复会重采输出稍后要送的位置）
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

                # Check that adding the request still respects the max_loras
                # constraint.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L724-L737（LoRA，原样保留）
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # SUBTRACTED: connector 外部命中簿记初始化（L739-L742——
                #   num_external_computed_tokens/load_kv_async/did_prefix_
                #   cache_lookup，第 1/10 条批准）。

                # Get already-cached tokens.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L744-L766 前缀重命中
                # （m7/F2：free 不清哈希 → 被抢者沿自己的 block_hashes 重命中）
                if request.num_computed_tokens == 0:
                    # SUBTRACTED: connector 分支 get_computed_blocks_for_
                    #   connector（L749-L759——hybrid-aware 查找，第 1 条批准，
                    #   P/D 归 ch36）与 hit_diverged 簿记（L747——供已删的
                    #   connector 调和分支）。
                    (
                        new_computed_blocks,
                        num_new_local_computed_tokens,
                        # Marconi shared-prefix junction to pin; 0 if none.
                        request.shared_prefix_boundary,
                    ) = self.kv_cache_manager.get_computed_blocks(request)

                    # SUBTRACTED: KVConnector 外部命中合并（L768-L826——第 1
                    #   条）、ec_connector 的 mm prefetch 跳过（L834-L844——第 2
                    #   条）、prefill_stats.set（L846-L853——第 10 条）。
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

                # SUBTRACTED: encoder 变量与 pad_spec_decode 初始化（L861-L864
                #   ——第 2/6 条）、load_kv_async 的零 token 分支与 defer_
                #   prefills 的 prefill 延后 break（L866-L873——第 1/11 条）——
                #   if/elif 链坍缩为原 else 主路径。
                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed
                # requests, which have output tokens.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L874-L879
                num_new_tokens = request.num_tokens - num_computed_tokens

                # SUBTRACTED: spec 均匀 pad 分支（L881-L897——'Prefer to not
                #   schedule than schedule un-padded'，第 6 条批准）。

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

                # SUBTRACTED: WAITING 侧 encoder inputs 调度（L916-L932——第 2
                #   条）、mamba 对齐切块（L934-L943——第 8 条）、load_kv_async
                #   的 lookahead 延后（L945-L951——第 1 条）、编码器交叉注意力
                #   块数（L953-L963——第 2 条）、async KV load 的 reserved_
                #   blocks 在途预约（L965-L971——第 1 条）。

                # SOURCE: vllm/v1/core/sched/scheduler.py:L973-L985 恢复准入
                # （m8：full_sequence_must_fit 整序列门 + has_scheduled_reqs
                # 水位开关——WAITING 侧 None 只 break，绝不触发抢占）
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    # SUBTRACTED: num_lookahead_tokens / num_external_
                    #   computed_tokens / delay_cache_blocks / num_encoder_
                    #   tokens / reserved_blocks 五参（第 1/2/6 条批准）。
                    full_sequence_must_fit=self.scheduler_reserve_full_isl,
                    has_scheduled_reqs=bool(self.running),
                )

                # SOURCE: vllm/v1/core/sched/scheduler.py:L987-L994 None → break
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # SUBTRACTED: encoder_cache_manager.free 的 un-touch
                    #   （L990-L993——encoder，第 2 条批准）。
                    # WAITING 阶段绝不触发抢占——与 RUNNING 侧抢占环的反差。
                    break

                # SUBTRACTED: connector update_state_after_alloc 与
                #   connector_prefix_cache_stats 记录（L996-L1014——第 1 条）、
                #   record_prefix_cache_stats（L1016-L1020——第 10 条）。

                # SOURCE: vllm/v1/core/sched/scheduler.py:L1022 出队
                request = request_queue.pop_request()
                # SUBTRACTED: load_kv_async 的 WAITING_FOR_REMOTE_KVS 整段
                #   （L1023-L1053——第 1 条批准；默认 connector=None 时真实
                #   代码也直接走到 self.running.append）。

                # SOURCE: vllm/v1/core/sched/scheduler.py:L1055 入 running
                self.running.append(request)
                # SUBTRACTED: log_stats 的 record_event(SCHEDULED)（L1056-L1059
                #   ——第 10 条批准，m18 正文讲注释与字段即可）。
                # SOURCE: vllm/v1/core/sched/scheduler.py:L1060-L1065 状态分流
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                # SOURCE: vllm/v1/core/sched/scheduler.py:L1067-L1068（LoRA 原样保留）
                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                # SOURCE: vllm/v1/core/sched/scheduler.py:L1069-L1075 落位记账
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # SUBTRACTED: pad_spec_decode 的 -1 占位登记（L1076-L1079——
                #   第 6 条批准）。
                # Only track requests that will still be prefilling after this chunk.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L1080-L1082
                if num_computed_tokens + num_new_tokens < request.num_tokens:
                    self._inflight_prefills.add(request)
                # SUBTRACTED: encoder 缓存 allocate（L1083-L1097——第 2 条）。

            # re-queue requests skipped in this pass ahead of older skipped items.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1099-L1101 步末重排（m6）
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

            # SUBTRACTED: prefill_capacity_bound 记录（L1103-L1106——DP，第 11 条）。

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

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        # SUBTRACTED: get_num_common_prefix_blocks 的分组展开与实算（L1121-
        #   L1129——cascade attention 的消费侧在模型章；单 KV 组全注意力下
        #   恒为 [0]，调度决策不读它）。
        num_common_prefix_blocks = [0]

        # Construct the scheduler output.
        # SUBTRACTED: use_v2_model_runner 的 resumed 合并分支（L1132-L1142——
        #   V2 runner 是实验路径，dossier.delete 第 7 条批准，ch18 话头）。
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
        #   _free_cow_retained_blocks（L1165-L1190——第 1 条）、动态 spec 的
        #   num_spec_tokens_to_schedule（L1192-L1197——第 6 条）、
        #   scheduled_encoder_input_stats（L1199-L1206——第 2/10 条）。

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

        # SUBTRACTED: connector/ec_connector 的 metadata 注入（L1231-L1244——
        #   第 1 条）、defer_block_free 的 sched_step_seq 推进（L1246-L1249——
        #   第 7 条）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1251-L1253
        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    # ------------------------------------------------------------------ #
    # 抢占：六件事全记录（本章绝对核心二）
    # ------------------------------------------------------------------ #
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
        # SUBTRACTED: encoder_cache_manager.free（L1291——encoder，第 2 条批准）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1292-L1296
        self._inflight_prefills.discard(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        # Async scheduling: mark all in-flight output as stale. Its tokens are
        # still delivered on return (dropping them would perturb spec-decode
        # acceptance) but must not mutate the reset counters; each step drains
        # its share in update_from_output. num_in_flight_tokens already
        # includes any undrained stale share, so assign rather than accumulate.
        # An undrained drop-mode share stays dropped: its positions have
        # already been resampled.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1297-L1308 stale 标记（m4 载体）
        request.drop_stale_output = drop_stale_output or (
            request.drop_stale_output and request.num_stale_output_tokens > 0
        )
        request.num_stale_output_tokens = request.num_in_flight_tokens
        request.num_output_placeholders = 0
        request.num_preemptions += 1
        # SUBTRACTED: log_stats 的 record_event(PREEMPTED)（L1310-L1311——第 10
        #   条批准；timestamp 参数保留原签名）。

        # Put the request back to the waiting queue.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1313-L1315 回队头
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
        # 3. If some tokens (e.g., spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1327-L1331
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.num_in_flight_tokens += num_scheduled_token
            # SUBTRACTED: defer_block_free 的 last_sched_seq 记录（L1332-L1334
            #   ——async 残留，dossier.delete 第 7 条批准）。
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1335-L1337
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            # SUBTRACTED: has_structured_output_requests 的 |= 累计（L1338-L1340
            #   ——structured，第 4 条批准）。
            # Drop from the in-flight-prefill set once it's no longer prefilling.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1341-L1343
            if not request.is_prefill_chunk:
                self._inflight_prefills.discard(request)

        # SUBTRACTED: routed experts 的块 id 快照（L1345-L1359——可观测性，
        #   dossier.delete 第 10 条批准）。

        # Clear the finished and preempted request IDs.
        # NOTE: We shouldn't just clear() here because it will also affect
        # the scheduler output.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1361-L1365 换新不 clear
        self.finished_req_ids = set()
        self.reset_preempted_req_ids = set()

    # SUBTRACTED: _update_request_as_session（L1367-L1408——流式输入会话续接，
    #   dossier.delete 第 5 条批准）。

    # ------------------------------------------------------------------ #
    # 增量打包（resumed_req_ids：worker 侧块表『整体替换而非追加』的通告集）
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
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1418-L1424
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
            # SUBTRACTED: use_pp 的采样 token 回传分支（L1430-L1445——PP，
            #   dossier.delete 第 7 条批准——非 PP 时 new_token_ids 恒空，
            #   worker 自己缓存采样 token）。
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1446-L1447
            if idx >= num_running_reqs:
                resumed_req_ids.add(req_id)
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1448-L1450 prev_step 判定
            # （use_v2_model_runner 条件随第 7 条拆除，判定本体保留）
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

    # SUBTRACTED: _try_schedule_encoder_inputs / _free_encoder_inputs /
    #   _make_scheduled_encoder_input_stats（L1469-L1661 其余——encoder，
    #   dossier.delete 第 2 条批准）与 _get_grammar_bitmask（L1640-L1668——
    #   structured，第 4 条）。

    # ------------------------------------------------------------------ #
    # ⑤ 拍热循环：请求生命周期的收尾处（本章绝对核心三）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L1670 update_from_output
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output,
    ) -> dict[int, EngineCoreOutputs]:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1675/L1678
        sampled_token_ids = model_runner_output.sampled_token_ids
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        # SUBTRACTED: logprobs / prompt_logprobs_dict / pooler_outputs /
        #   num_nans_in_logits / kv_connector_output / cudagraph_stats 的取用
        #   （L1676-L1682——第 10/1 条）、defer_block_free 的栅栏推进
        #   （L1684-L1688——第 7 条）、perf_stats（L1690-L1692——第 10 条）、
        #   failed_kv_load 的 invalid blocks 处理（L1697-L1705——第 1 条）、
        #   routed_experts 的 routing_offsets 构建（L1707-L1726——第 10/7 条）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1694
        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1728-L1732 热循环开场
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            output_is_stale = False
            if request is not None:
                # SOURCE: vllm/v1/core/sched/scheduler.py:L1737-L1738 扣在途
                request.num_in_flight_tokens -= num_tokens_scheduled
                # Drain any stale share (see _preempt_request) in lockstep.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L1739-L1743 stale 排干
                if request.num_stale_output_tokens > 0:
                    output_is_stale = True
                    request.num_stale_output_tokens -= num_tokens_scheduled
                    assert request.num_stale_output_tokens >= 0
            # SUBTRACTED: failed_kv_load_req_ids 跳过（L1744-L1746——connector
            #   加载失败回滚，第 1 条批准）。
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1747-L1755 abort 期完成
            # continue（幂等——ch9 abort 双投递成立前提）
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # SUBTRACTED: delay_free_blocks 的 NOTE(Kuntai) 注释半段
                #   （L1751-L1754——connector 场景）。
                continue

            # Drop-mode stale output (same-step resume) is discarded entirely.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1757-L1759
            if output_is_stale and request.drop_stale_output:
                continue

            # SOURCE: vllm/v1/core/sched/scheduler.py:L1761-L1764 定位采样行
            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            # SUBTRACTED: spec 拒绝回扣分支（L1766-L1791——num_rejected 回扣
            #   num_computed_tokens / num_output_placeholders：追赶公式的安全网，
            #   dossier.delete 第 6 条批准，深挖归 ch12/spec 章）与 encoder
            #   释放（L1793-L1795——第 2 条）。

            # SOURCE: vllm/v1/core/sched/scheduler.py:L1797-L1805
            stopped = False
            new_token_ids = generated_token_ids
            status_before_stop = request.status

            # Check for stop and update request status.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1807-L1811
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids, is_stale=output_is_stale
                )
            # SUBTRACTED: pooling 的 FINISHED_STOPPED（L1812-L1815——第 10 条）、
            #   structured 的 should_advance/trim_reasoning/grammar.accept
            #   （L1817-L1843——第 4 条）、routed_experts 切片（L1845-L1883——
            #   第 10 条）。

            # SOURCE: vllm/v1/core/sched/scheduler.py:L1885-L1887（pooler 判项
            # 与 prefill_stats 终结随第 10 条删）
            should_emit_output = bool(new_token_ids or stopped)

            # SOURCE: vllm/v1/core/sched/scheduler.py:L1895-L1907 停止分流
            finish_reason = None
            if stopped:
                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    # SUBTRACTED: kv/ec transfer 参数解包（connector，第 1 条）。
                    self._free_request(request)

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # SUBTRACTED: logprobs 切片 / num_nans 记账 / prompt_logprobs 取用
            #   （L1909-L1921——第 10 条批准）。

            if should_emit_output:
                # Add EngineCoreOutput for this Request.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L1922-L1941（四字段版）
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        stop_reason=request.stop_reason,
                    )
                )
                # SUBTRACTED: else 分支的 prompt_logprobs 断言（L1942-L1944——
                #   随第 10 条删除的观测面）。

        # Remove the stopped requests from the running and waiting queues.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1946-L1952 批量摘除
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.skipped_waiting.remove_requests(stopped_preempted_reqs)

        # SUBTRACTED: grammar 编译失败 / failed_kv_load 的 error 攒批段
        #   （L1954-L1972——第 1/4 条）、connector stats 与 KV events 发布
        #   （L1974-L2010——第 1/10 条）、finished_req_ids_dict 按 client 分桶
        #   （L2019-L2031——V2 细节）、make_stats（L2033-L2046——第 10 条）。

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2012-L2017
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        return engine_core_outputs

    # ------------------------------------------------------------------ #
    # waiting/skipped 双队列路由（m6 防队头阻塞）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2050 _is_blocked_waiting_status
    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2052-L2056（三阻塞态清单
        # 原样保留——状态机完整性；其运行时来源随子系统删除）
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
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2065-L2066（FCFS: skipped 优先）
        if self.policy == SchedulingPolicy.FCFS:
            return self.skipped_waiting or self.waiting or None

        # SUBTRACTED: PRIORITY 模式的双队头比较分支（L2068-L2072——
        #   dossier.delete 第 1 条批准——本精简版 policy 恒 FCFS，走不到）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2074
        return self.waiting or self.skipped_waiting or None

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2678 _try_promote_blocked_waiting_request
    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        # SUBTRACTED: 三个阻塞态的提升分支——WAITING_FOR_REMOTE_KVS（L2682-
        #   L2693：finished_recving_kv 信号驱动 + _update_waiting_for_remote_kv，
        #   dossier.delete 第 1 条批准，归 ch16）、WAITING_FOR_STRUCTURED_
        #   OUTPUT_GRAMMAR（L2695-L2703：grammar 就绪检查/编译失败攒批，第 4
        #   条批准，归 ch30）、WAITING_FOR_STREAMING_REQ（L2705-L2707——真实
        #   此分支本就无条件 return False：提升走 add_request 路径）。三个
        #   状态机成员保留在枚举与 _is_blocked_waiting_status 原判里（状态机
        #   完整性），但其运行时来源全随子系统删除——精简版中提升永不成功：
        #   return False = 『仍在等』，遍历方跳过（不卡队头）。
        return False

    # ------------------------------------------------------------------ #
    # 停止分流与逐 token 收账
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2076 _handle_stopped_request
    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        # SUBTRACTED: resumable 流式会话分流（L2078-L2092——streaming_queue 有
        #   下一段就 _update_request_as_session 续跑、否则转 WAITING_FOR_
        #   STREAMING_REQ 等后续输入，dossier.delete 第 5 条批准：精简版假设
        #   一次性输入，停止即真完成；RequestStatus 枚举成员保留（顺序不可动）。
        #   非流式请求恒 return True——真终点。
        return True

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2094 _update_request_with_output
    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        # is_stale is only used by the AsyncScheduler override.
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2101-L2111（逐 token 循环原文）
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

    # SUBTRACTED: update_draft_token_ids / update_draft_token_ids_in_output
    #   （L2146-L2203——spec decode 簿记与 structured 校验，dossier.delete
    #   第 4/6 条批准）、get_kv_cache_usage（L2209-L2211——可观测性，第 10 条）。

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2205 get_request_counts
    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2206-L2207
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)

    # ------------------------------------------------------------------ #
    # 外部死法：断连 abort / 语法编译失败也走这里（m17）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2237 finish_requests
    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[Request]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

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

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2263-L2276
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                # SUBTRACTED: WAITING_FOR_STREAMING_REQ 的计数递减（L2274-
                #   L2275——流式，第 5 条批准）。
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2278-L2283 三队列摘除
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2285-L2296
        for request in valid_requests:
            # SUBTRACTED: WAITING_FOR_REMOTE_KVS 的 delay_free_blocks 判定与
            #   finished/failed_recving_kv 摘除（L2287-L2293——connector，
            #   第 1 条批准）。
            request.status = finished_status
            self._free_request(request)

        return valid_requests

    # ------------------------------------------------------------------ #
    # 生命终点：登记 finished_req_ids + KV 释放 + 账本除名（m16）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2300 _free_request
    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2303
        assert request.is_finished()

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2305
        self._inflight_prefills.discard(request)
        # SUBTRACTED: _connector_finished 的 kv_xfer 参数与延迟释放判定
        #   （L2306——connector，第 1 条）、ec_connector.request_finished 钩子
        #   （L2308-L2315——EC connector，第 1 条）、encoder_cache_manager.free
        #   （L2317——第 2 条）、finished_req_ids_dict 按 client 分桶（L2320-
        #   L2321——V2 细节，第 10 条 elide）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2318-L2319 登记（下拍随
        # SchedulerOutput 通知 worker 清缓存）
        request_id = request.request_id
        self.finished_req_ids.add(request_id)

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2323-L2325
        # SUBTRACTED: delay_free_blocks |= connector_delay_free_blocks（第 1 条）
        #   ——参数保留原签名，恒 False 立即释放。
        if not delay_free_blocks:
            self._free_blocks(request)

        return None, None

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2329 _free_blocks
    def _free_blocks(self, request: Request):
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2330-L2332
        assert request.is_finished()
        self._free_request_blocks(request)
        del self.requests[request.request_id]

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2334 pause_state property
    @property
    def pause_state(self) -> PauseState:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2335-L2336
        return self._pause_state

    # SUBTRACTED: set_pause_state（L2338-L2339——暂停机制，dossier.delete
    #   第 11 条批准；守卫 `not preempted_reqs and UNPAUSED` 的 UNPAUSED
    #   比较保留，恒成立）。

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2341 _free_request_blocks
    def _free_request_blocks(self, request: Request):
        """Free the request's KV blocks, deferring the return to the block
        pool when an in-flight GPU step may still write them.
        """
        # SUBTRACTED: defer_block_free 的延迟释放栅栏（L2345-L2354——
        #   last_sched_seq / processed_step_seq / deferred_frees / pop_blocks_
        #   for_free，dossier.delete 第 7 条批准）；同步版直接归还。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2350
        self.kv_cache_manager.free(request)

    # SUBTRACTED: _free_cow_retained_blocks / _drain_deferred_frees
    #   （L2356-L2380——defer_block_free 栅栏与 connector CoW，第 1/7 条批准）。

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2382 get_num_unfinished_requests
    def get_num_unfinished_requests(self) -> int:
        # SUBTRACTED: PAUSED_ALL / PAUSED_NEW 分支（L2383-L2386——暂停机制，
        #   dossier.delete 第 11 条批准）与 num_waiting_for_streaming_input
        #   递减（L2390——流式，第 5 条）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2387-L2392
        num_waiting = len(self.waiting) + len(self.skipped_waiting)
        return num_waiting + len(self.running)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2394 has_finished_requests
    def has_finished_requests(self) -> bool:
        # SUBTRACTED: connector 的延迟清理滞留判定（L2397-L2404——第 1 条
        #   批准）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2395-L2396
        if self.finished_req_ids:
            return True
        return False

    # SUBTRACTED: has_requests 的 connector pending-push 覆写（L2406-L2421——
    #   第 1 条；基类默认 has_unfinished or has_finished 即足够）、
    #   reset_prefix_cache / reset_connector_cache（L2423-L2550——缓存管理
    #   运维入口，第 9 条批准；drop_stale_output 协议本身保留，只删调用方）、
    #   make_stats / shutdown / _build_kv_connector_meta / _get_new_block_ids_
    #   to_zero / _connector_finished / _inflight_prefill_reserved_blocks /
    #   _update_waiting_for_remote_kv / _update_from_kv_xfer_finished /
    #   _update_requests_with_invalid_blocks / _handle_invalid_blocks
    #   （L2495-L2915 其余——可观测性与 connector 子系统，第 1/8/10 条批准）。
