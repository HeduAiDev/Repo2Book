# SOURCE: vllm/v1/core/sched/scheduler.py
# 同步版 Scheduler —— AsyncScheduler 的基类。本章切面（相对 ch10/ch11 的取舍）：
#   * RUNNING 循环全保 async 分支——early-stop 剪枝（L488-L502，m9）与追赶公式
#     的占位项（L516-L520，m8）逐字；
#   * _update_after_schedule（L1317-L1365）保乐观推进三件套：computed +=
#     scheduled、in_flight += scheduled、is_prefill_chunk 用『num_tokens +
#     placeholders』重算（L1335-L1337——占位让全量 prefill 也能判 False），
#     has_structured_output_requests |=（L1338-L1340——async 标志对之一）；
#   * _preempt_request 的 async 账单（L1296-L1308，m16）逐字；
#   * update_from_output 热循环（L1670-L2048 切面）：扣在途、stale 锁步 drain
#     （L1736-L1743）、spec 拒绝回扣（L1769-L1784，m15——ch11 删给本章）、
#     _update_request_with_output(request, new_token_ids, is_stale) 调用位
#     （L1807-L1811——AsyncScheduler 覆写的动态分发点）；
#   * get_grammar_bitmask（L1646-L1668，m14 deferred 补采的依赖）。
# 删除项全部 dossier.subtraction_plan.delete 批准（DP/PP/V2/spec 深层/观测）或
#   邻章边界（PRIORITY 抢占选择/前缀缓存深水/准入水位——ch10/ch11/ch15 已立）。
from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable

from .engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from .interface import SchedulerInterface
from .kv_cache_manager import KVCacheBlocks, KVCacheManager
from .output import CachedRequestData, NewRequestData, SchedulerOutput
from .request import Request, RequestStatus
from .request_queue import create_request_queue
from .utils import check_stop, remove_all


# SOURCE: vllm/v1/core/sched/scheduler.py:L69 Scheduler
class Scheduler(SchedulerInterface):
    # SOURCE: vllm/v1/core/sched/scheduler.py:L70 __init__（装配切面：真实
    # 签名 (vllm_config, kv_cache_config, structured_output_manager,
    # include_finished_set, log_stats, block_size, hash_block_size) 保留）
    def __init__(
        self,
        vllm_config,
        kv_cache_config=None,
        structured_output_manager=None,
        include_finished_set: bool = False,
        log_stats: bool = False,
        block_size: int = 16,
        hash_block_size: int | None = None,
        num_gpu_blocks: int = 1 << 30,
    ) -> None:
        # SUBTRACTED: 真实 __init__ 的几百行装配（kv events/connector/encoder
        #   预算/LoRA/routed experts/perf metrics/mamba/DCP-PCP/defer fences/
        #   use_pp/use_v2_model_runner/prefill_capacity_bound——L70-L360）——
        #   各归邻章；本章以裸标量承载调度约束字段。
        self.scheduler_config = vllm_config.scheduler_config
        self.max_model_len = vllm_config.max_model_len
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager
        # SUBTRACTED: V2 runner 面（use_v2_model_runner——dossier.delete 第 4 条
        #   批准：AsyncScheduler 的 V2 分支已删；恒 False）。
        self.use_v2_model_runner = False
        # SOURCE: vllm/v1/core/sched/scheduler.py:L120-L123
        # Diffusion models may not sample any tokens for a denoising step.
        self.num_sampled_tokens_per_step = 1
        # SOURCE: vllm/v1/core/sched/scheduler.py:L109-L114 约束字段
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens is not None
            else self.scheduler_config.max_num_batched_tokens
        )
        # SUBTRACTED: spec 面（num_spec_tokens/num_lookahead_tokens——第 6 条；
        #   无 spec 配置恒 0）。
        self.num_spec_tokens = 0

        # Create the KV cache manager.
        # SUBTRACTED: kv_cache_config 驱动装配（L272-L294——ch13）。
        self.kv_cache_manager = KVCacheManager(
            num_gpu_blocks=num_gpu_blocks,
            block_size=block_size,
            max_model_len=self.max_model_len,
        )

        # SUBTRACTED: lora_config（L84——LoRA 面归 worker 侧第 7 条删除；调度侧
        #   分支不再保留）。req_id -> Request
        # SOURCE: vllm/v1/core/sched/scheduler.py:L177-L185
        self.requests: dict[str, Request] = {}
        # Priority queues for requests.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L187
        self.waiting = create_request_queue(self.scheduler_config.policy)
        # requests skipped in waiting flow due async deps or constraints.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L189
        self.skipped_waiting = create_request_queue(self.scheduler_config.policy)
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

        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads).
        # SOURCE: vllm/v1/core/sched/scheduler.py:L358-L360
        self._inflight_prefills: set[Request] = set()

        # ENGINE SEAM observation：测试对账位（update_from_output 前后状态）。
        self.last_output: SchedulerOutput | None = None

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2213 add_request
    def add_request(self, request: Request) -> None:
        # SUBTRACTED: 重复 request_id 的流式会话续跑（L2214-L2226——streaming）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:_enqueue_waiting_request 调用位
        self.waiting.add_request(request)
        self.requests[request.request_id] = request

    # ------------------------------------------------------------------ #
    # schedule —— 一拍：两阶段分账（RUNNING 先行 + WAITING 准入）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L439 schedule（throttle_prefills
    # 实参已随 DP 面删除——dossier.delete 第 5 条）
    def schedule(self) -> SchedulerOutput:
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

        # SOURCE: vllm/v1/core/sched/scheduler.py:L452-L459
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens

        # Spec decode-related.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L467-L468（簿记字典保留，登记
        # 分支随第 6 条删）
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        # Whether the running batch contains any prefill requests.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L469-L470
        prefill_scheduled = False

        # For logging.
        # SUBTRACTED: scheduled_timestamp = time.monotonic()（L472-L473——观测）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L475
        self.kv_cache_manager.new_step_starts()

        # SUBTRACTED: defer_prefills 的 DP prefill balancing 计算（L477-L481
        #   ——第 5 条批准）。

        # First, schedule the RUNNING requests.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L483-L485
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            # SOURCE: vllm/v1/core/sched/scheduler.py:L488-L502 early-stop 剪枝
            # （m9：async 专属——确信上拍已达 max_tokens 就不排多余一步）
            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            # SUBTRACTED: next_decode_eligible_step 步距判定（L504-L508——V2+PP+
            #   async，dossier.delete 第 4 条批准）、defer_prefills 续 chunk 延后
            #   （L510-L514——DP，第 5 条）。

            # SOURCE: vllm/v1/core/sched/scheduler.py:L516-L520 追赶公式（m8：
            # 同步版占位恒 0，async 下 num_output_placeholders 项灌上占位数）
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
            # SOURCE: vllm/v1/core/sched/scheduler.py:L525-L532 max_model_len 钳制
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )

            # SUBTRACTED: encoder 预算截断（L534-L550）与 mamba 对齐切块
            #   （L552-L555）——encoder/hybrid 面。

            # SOURCE: vllm/v1/core/sched/scheduler.py:L557-L573 0-token continue
            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # SUBTRACTED: encoder 预算/缓存耗尽两因与 mamba 块对齐一因的
                #   注释半段（L285-L288）。
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L575-L629 抢占重试环（内景
            # 归 ch11：FCFS 抢队尾；PRIORITY 选择分支已删——ch10/ch11 拥有）
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                )

                if new_blocks is not None:
                    # The request can be scheduled.
                    break

                # The request cannot be scheduled.
                # Preempt the lowest-priority request.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L614-L615 FCFS 抢队尾
                preempted_req = self.running.pop()
                # SUBTRACTED: drop_stale_output=requires_kv_delivery 实参
                #   （L617-L620——connector 面；参数默认 False 保留）。
                self._preempt_request(preempted_req, time.monotonic())
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
            #   （L640-L656——dossier.delete 第 6 条批准：spec 深层归 ch33；
            #   占位公式里的 spec 项由 AsyncScheduler 以 cur_num_spec_tokens=0
            #   消费，不影响主线）与 encoder 缓存 allocate（L658-L671）。

        # SUBTRACTED: LoRA 登记（L673-L681——worker 侧正交面，第 7 条）。

        # Next, schedule the WAITING requests.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L683-L684 守卫（本拍抢占过 =
        # 内存紧张信号 → 整拍不收新；pause 态随 ch11 删）
        if not preempted_reqs:
            # SOURCE: vllm/v1/core/sched/scheduler.py:L685 步内跳过收集队列
            step_skipped_waiting = create_request_queue(self.scheduler_config.policy)

            # SOURCE: vllm/v1/core/sched/scheduler.py:L687-L692
            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                # SUBTRACTED: 流式会话占位计数（L690——streaming）。
                num_running = len(self.running)
                if num_running >= self.max_num_running_reqs:
                    break

                # SOURCE: vllm/v1/core/sched/scheduler.py:L694-L698 队列选择
                # （FCFS: skipped 优先）
                request_queue = self.skipped_waiting or self.waiting or None
                assert request_queue is not None

                # SOURCE: vllm/v1/core/sched/scheduler.py:L697-L698
                request = request_queue.peek_request()
                request_id = request.request_id

                # SOURCE: vllm/v1/core/sched/scheduler.py:L713-L722 stale 在途
                # 推迟一拍（现在恢复会重采输出稍后要送的位置——stale 账的准入半边）
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

                # SUBTRACTED: 阻塞态提升判定（L700-L711——三阻塞态的运行时来源
                #   随子系统删除，本章不保留）、LoRA 约束跳过（L724-L737）。

                # Get already-cached tokens.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L744-L766 前缀重命中
                # （精简版恒 0 命中——命中深水归 ch15）
                if request.num_computed_tokens == 0:
                    (
                        new_computed_blocks,
                        num_new_local_computed_tokens,
                        request.shared_prefix_boundary,
                    ) = self.kv_cache_manager.get_computed_blocks(request)

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

                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed
                # requests, which have output tokens.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L874-L879
                num_new_tokens = request.num_tokens - num_computed_tokens

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

                # SUBTRACTED: WAITING 侧 encoder/mamba/async KV load 预约
                #   （L916-L971——各子系统）。

                # SOURCE: vllm/v1/core/sched/scheduler.py:L973-L985 恢复准入
                # （水位面归 ch11；精简版直调 allocate_slots）
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                )

                # SOURCE: vllm/v1/core/sched/scheduler.py:L987-L994 None → break
                # （WAITING 阶段绝不触发抢占——与 RUNNING 侧抢占环的反差）
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # SOURCE: vllm/v1/core/sched/scheduler.py:L1022 出队
                request = request_queue.pop_request()

                # SOURCE: vllm/v1/core/sched/scheduler.py:L1055 入 running
                self.running.append(request)
                # SOURCE: vllm/v1/core/sched/scheduler.py:L1060-L1075 状态分流
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                # SOURCE: vllm/v1/core/sched/scheduler.py:L1069-L1075 落位记账
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Only track requests that will still be prefilling after this chunk.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L1080-L1082
                if num_computed_tokens + num_new_tokens < request.num_tokens:
                    self._inflight_prefills.add(request)

            # re-queue requests skipped in this pass ahead of older skipped items.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1099-L1101 步末重排
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

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
        # SUBTRACTED: get_num_common_prefix_blocks 实算（L1121-L1129——cascade
        #   attention 消费侧在模型章；单 KV 组全注意力下恒 [0]）。
        num_common_prefix_blocks = [0]

        # Construct the scheduler output.
        # SUBTRACTED: use_v2_model_runner 的 resumed 合并分支（L1132-L1142——
        #   第 4 条批准）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1144-L1149 new 全量打包
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids()
            )
            for req in scheduled_new_reqs
        ]

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1151-L1158 cached 增量打包
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

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1208-L1229 SchedulerOutput 落袋
        # （async 标志对由 _update_after_schedule 就地置位；数据类默认 False）
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
        )

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1251-L1253
        self._update_after_schedule(scheduler_output)
        self.last_output = scheduler_output  # ENGINE SEAM observation
        return scheduler_output

    # ------------------------------------------------------------------ #
    # 抢占：async 账单半边（内景归 ch11）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L1274 _preempt_request
    def _preempt_request(
        self, request: Request, timestamp: float, drop_stale_output: bool = False
    ) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1287-L1296
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self._free_request_blocks(request)
        # SUBTRACTED: encoder_cache_manager.free（L1291——encoder）。
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
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1297-L1308 stale 标记（m16）
        request.drop_stale_output = drop_stale_output or (
            request.drop_stale_output and request.num_stale_output_tokens > 0
        )
        request.num_stale_output_tokens = request.num_in_flight_tokens
        request.num_output_placeholders = 0
        request.num_preemptions += 1
        # SUBTRACTED: log_stats 的 record_event(PREEMPTED)（L1310-L1311）。

        # Put the request back to the waiting queue.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1313-L1315 回队头
        self.waiting.prepend_request(request)
        self.reset_preempted_req_ids.add(request.request_id)

    # ------------------------------------------------------------------ #
    # 调度后乐观推进（AsyncScheduler 覆写的基类半边）
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
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1327-L1343
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.num_in_flight_tokens += num_scheduled_token
            # SUBTRACTED: defer_block_free 的 last_sched_seq 记录（L1332-L1334）。
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1335-L1337 is_prefill_chunk
            # 重算——占位项在这里进判据（async 下全量 prefill 也能判 False）
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1338-L1340 async 标志一
            # 的置位处
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )
            # Drop from the in-flight-prefill set once it's no longer prefilling.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1341-L1343
            if not request.is_prefill_chunk:
                self._inflight_prefills.discard(request)

        # SUBTRACTED: routed experts 块 id 快照（L1345-L1359——观测）。

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
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1418-L1427
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1426-L1427
        for idx, req in enumerate(running_reqs + resumed_reqs):
            req_id = req.request_id
            req_ids.append(req_id)
            # SUBTRACTED: use_pp 的采样 token 回传分支（L1430-L1445——PP 面，
            #   dossier.delete 第 5 条批准；非 PP 时 new_token_ids 恒空——
            #   worker 自己缓存采样 token）。
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1446-L1447
            if idx >= num_running_reqs:
                resumed_req_ids.add(req_id)
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1448-L1450 prev_step 判定
            if req_id not in self.prev_step_scheduled_req_ids:
                all_token_ids[req_id] = req.all_token_ids.copy()
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1451-L1457 num_output_
            # tokens 含占位（async 专属灌值）
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
    # 结构化输出 bitmask（m14 deferred 补采的依赖；掩码怎么算归 ch30）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L1646 get_grammar_bitmask
    def get_grammar_bitmask(
        self, scheduler_output: "SchedulerOutput"
    ) -> "object | None":
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1650-L1661 快返：本批没有
        # 结构化请求
        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1663-L1668
        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        # SUBTRACTED: log_stats 的 prefill_stats 终结（L1667 附近——观测）。
        from .output import GrammarOutput

        return GrammarOutput(structured_output_request_ids, bitmask)

    # ------------------------------------------------------------------ #
    # ⑤ 拍热循环：真记账（本章 async 切面）
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
        # SUBTRACTED: logprobs/pooler/connector/routed/perf 的取用（L1676-L1692
        #   ——观测与子系统）、defer_block_free 栅栏（L1684-L1688）、failed_kv_
        #   load（L1697-L1705）、routing_offsets（L1707-L1726）。

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
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1736-L1743 扣在途 + stale
            # 锁步 drain（m16 收账半边）
            output_is_stale = False
            if request is not None:
                request.num_in_flight_tokens -= num_tokens_scheduled
                # Drain any stale share (see _preempt_request) in lockstep.
                if request.num_stale_output_tokens > 0:
                    output_is_stale = True
                    request.num_stale_output_tokens -= num_tokens_scheduled
                    assert request.num_stale_output_tokens >= 0
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1747-L1755 abort 期完成
            # continue（幂等——ch9 abort 双投递成立前提）
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
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

            # SOURCE: vllm/v1/core/sched/scheduler.py:L1766-L1791 spec 拒绝回扣
            # （m15：追赶公式的安全网——AsyncScheduler 的占位覆盖 spec token）
            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and (
                generated_token_ids or self.num_sampled_tokens_per_step == 0
            ):
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                num_rejected = num_draft_tokens - num_accepted
                # Rejections roll back num_computed_tokens (and, under async
                # scheduling, num_output_placeholders, which covers the spec
                # tokens). A stale rejection count predates the preemption
                # rollback and must not apply.
                if not output_is_stale:
                    if request.num_computed_tokens > 0:
                        request.num_computed_tokens -= num_rejected
                    if request.num_output_placeholders > 0:
                        request.num_output_placeholders -= num_rejected
                # SUBTRACTED: make_spec_decoding_stats（L1785-L1791——观测）。
            # SUBTRACTED: encoder 释放（L1793-L1795）。

            # SOURCE: vllm/v1/core/sched/scheduler.py:L1797-L1805
            stopped = False
            new_token_ids = generated_token_ids
            status_before_stop = request.status

            # Check for stop and update request status.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1807-L1811 逐 token 收账
            # （AsyncScheduler 覆写版经动态分发在此接管：占位扣减+块转正）
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids, is_stale=output_is_stale
                )
            # SUBTRACTED: pooling/structured/routed 切片（L1812-L1843）。

            # SOURCE: vllm/v1/core/sched/scheduler.py:L1885-L1887
            should_emit_output = bool(new_token_ids or stopped)

            # SOURCE: vllm/v1/core/sched/scheduler.py:L1895-L1907 停止分流
            finish_reason = None
            if stopped:
                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    self._free_request(request)

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

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

        # Remove the stopped requests from the running and waiting queues.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1946-L1952 批量摘除
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.skipped_waiting.remove_requests(stopped_preempted_reqs)

        # SUBTRACTED: grammar 编译失败/connector stats/KV events（L1954-L2010）。

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2012-L2017
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        return engine_core_outputs

    # ------------------------------------------------------------------ #
    # 停止分流与逐 token 收账
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2076 _handle_stopped_request
    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        # SUBTRACTED: resumable 流式会话分流（L2078-L2092——非流式恒真终点）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2092
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

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2587 update_draft_token_ids（精简：
    # spec 簿记内景归 ch33；方法面保留供 post_step 同步路径调用）
    def update_draft_token_ids(self, draft_token_ids) -> None:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2588-L2592 签名面
        # SUBTRACTED: 逐请求草稿回填与 structured 校验循环（L2593-L2620——
        #   ch33；无 spec 配置时调用方拿到 None 不会走到这里）。
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids, draft_token_ids.spec_token_ids
        ):
            if req_id in self.requests:
                self.requests[req_id].spec_token_ids = list(spec_token_ids)

    # SUBTRACTED: update_draft_token_ids_in_output（L2168-L2203——deferred+spec
    #   叠加的草稿校验，dossier.delete 第 8 条批准压缩为存证注释；方法名在
    #   engine.py 的 deferred 分支注释中保留）。

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2205 get_request_counts
    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2206-L2207
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2237 finish_requests
    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[Request]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
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
            request.status = finished_status
            self._free_request(request)

        return valid_requests

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2300 _free_request
    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> None:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2303
        assert request.is_finished()

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2305
        self._inflight_prefills.discard(request)
        # SUBTRACTED: connector/encoder 钩子（L2306-L2317）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2318-L2319 登记（下拍随
        # SchedulerOutput 通知 worker 清缓存）
        request_id = request.request_id
        self.finished_req_ids.add(request_id)

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2323-L2325
        if not delay_free_blocks:
            self._free_blocks(request)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2329 _free_blocks
    def _free_blocks(self, request: Request):
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2330-L2332
        assert request.is_finished()
        self._free_request_blocks(request)
        del self.requests[request.request_id]

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2341 _free_request_blocks
    def _free_request_blocks(self, request: Request):
        # SUBTRACTED: defer_block_free 延迟释放栅栏（L2345-L2354）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2350
        self.kv_cache_manager.free(request)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2382 get_num_unfinished_requests
    def get_num_unfinished_requests(self) -> int:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2387-L2392
        num_waiting = len(self.waiting) + len(self.skipped_waiting)
        return num_waiting + len(self.running)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2394 has_finished_requests
    def has_finished_requests(self) -> bool:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2395-L2396
        if self.finished_req_ids:
            return True
        return False

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2406 has_requests（基类版：
    # 未完成 or 有待收账）
    def has_requests(self) -> bool:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2406-L2421（connector pending
        # 覆写已删——基类语义）
        return self.get_num_unfinished_requests() > 0 or self.has_finished_requests()

    # SUBTRACTED: set_pause_state/reset_prefix_cache/make_stats/shutdown/
    #   connector 族方法（L2423-L2915——各归邻章）。
