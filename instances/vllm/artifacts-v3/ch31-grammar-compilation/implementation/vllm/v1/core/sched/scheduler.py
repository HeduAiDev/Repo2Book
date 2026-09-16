# SOURCE: vllm/v1/core/sched/scheduler.py
# 只做减法的忠实精简版（pin v0.27.1 / 6e448d0ea）——真实 Scheduler 是数千行
# 的调度核心（schedule/update_from_output 全貌已归 ch10-ch12）。本章只保留
# 『语法门』这一实例的调度侧：阻塞态判定与侧队分流（站 5）、WAITING 相位窥
# 队头的每拍晋级尝试（站 6）、编译失败同拍收账（m13）、采样后推进（站 7）。
# 非语法域的 KV 装配/抢占/P-D/流式/encoder 一律按章边界 SUBTRACTED（见行内
# 注释），保留真实方法名与控制流骨架。
# SUBTRACTED: SPDX 版权头；SchedulerInterface 基类面。
from collections import defaultdict
from contextlib import nullcontext as record_function_or_nullcontext  # HOST SEAM

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    SchedulingPolicy,
)
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreOutput, EngineCoreEventType
from vllm.v1.request import Request, RequestStatus
# StructuredOutputManager 的编排面在 vllm/v1/structured_output/__init__.py：
# grammar_init（IO 线程把编译提交 executor，工作函数 _create_grammar）、
# _use_async_grammar_compilation（external_launcher 同步回退开关）、
# _get_reasoner / _find_reasoning_end_index（思考门装配与边界定位）——
# 调度器消费推进侧的 should_advance / trim_reasoning_for_advance（站 7）。
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_types import StructuredOutputGrammar

logger = init_logger(__name__)


# SOURCE: vllm/v1/core/sched/scheduler.py:L69 Scheduler
class Scheduler:
    # SOURCE: vllm/v1/core/sched/scheduler.py:L70-L437 __init__（装配切面）
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry=None,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        # SUBTRACTED: cache_config/lora_config/kv_cache_config 的真实消费面
        #   （KV 装配域 ch13/ch16——参数保留原位）。
        self.structured_output_manager = structured_output_manager
        self.log_stats = log_stats

        # SOURCE: vllm/v1/core/sched/scheduler.py:L108-L109
        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        # SOURCE: vllm/v1/core/sched/scheduler.py:L110-L114
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens is not None
            else self.scheduler_config.max_num_batched_tokens
        )
        # SUBTRACTED: L115-L116 max_model_len/pcp_world_size（长度与 PP 域）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L177-L189
        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        # SUBTRACTED: create_request_queue 工厂（L187——精简版 FCFS-only，
        #   PriorityRequestQueue 归 ch11）
        self.waiting = FCFSRequestQueue()
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = FCFSRequestQueue()
        self.running: list[Request] = []
        # SUBTRACTED: L191-L212 finished_req_ids/reset_preempted_req_ids/
        #   num_waiting_for_streaming_input/finished_recving_kv_req_ids/
        #   failed_recving_kv_req_ids 与 mm 预算（流式/P-D/抢占/mm 域）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L210-L212
        # Grammar compilation failures to finish as per-request errors in
        # update_from_output.
        self.grammar_compile_error_reqs: set[str] = set()

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2213-L2235 add_request
    def add_request(self, request: Request) -> None:
        # SUBTRACTED: L2214-L2226 existing 分支的流式会话续跑
        #   （StreamingUpdate.from_request/streaming_queue——ch38 域）；
        #   `existing = ...; if existing is not None:` 分流头随之减去，
        #   else 体提升为直线。
        # SUBTRACTED: L2227-L2228 `if request.resumable:
        #   request.streaming_queue = deque()`（流式域）
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2229-L2230
        self._enqueue_waiting_request(request)
        self.requests[request.request_id] = request
        # SUBTRACTED: L2231-L2232 connector.on_new_request（KV 连接器域）
        if self.log_stats:
            # SOURCE: vllm/v1/core/sched/scheduler.py:L2233-L2235
            request.record_event(EngineCoreEventType.QUEUED)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L439-L1253 schedule（语法门切面）
    def schedule(self) -> SchedulerOutput:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L459
        token_budget = self.max_num_scheduled_tokens
        # SUBTRACTED: L460-L462 _pause_state 守卫（暂停域）。

        # SUBTRACTED: L464-L681 RUNNING 相位（预算追赶/抢占重试环/spec 登记/
        #   encoder/LoRA——ch10/ch11/ch33 域）。preempted_reqs 恒空。
        preempted_reqs: list[Request] = []

        # SUBTRACTED: L467-L468 req_to_new_blocks/簿记字典（KV 记账域）。
        num_scheduled_tokens: dict[str, int] = {}
        # SUBTRACTED: L470-... scheduled_encoder_inputs/scheduled_spec_decode_
        #   tokens/prefill_scheduled 等 RUNNING 相位簿记。

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        # SUBTRACTED: scheduled_running_reqs/scheduled_loras（RUNNING/LoRA 面）。

        # Next, schedule the WAITING requests.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L683-L684 守卫（本拍抢占过 =
        # 内存紧张信号 → 整拍不收新）
        if not preempted_reqs:
            # SOURCE: vllm/v1/core/sched/scheduler.py:L685 步内跳过收集队列
            step_skipped_waiting = FCFSRequestQueue()

            # SOURCE: vllm/v1/core/sched/scheduler.py:L687-L692
            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                # SUBTRACTED: L690 流式会话占位计数（streaming）。
                num_running = len(self.running)
                if num_running >= self.max_num_running_reqs:
                    break

                # SOURCE: vllm/v1/core/sched/scheduler.py:L694-L698 队列选择
                #   （_select_waiting_queue_for_scheduling；FCFS: skipped 优先）
                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                # SOURCE: vllm/v1/core/sched/scheduler.py:L697-L698
                request = request_queue.peek_request()
                request_id = request.request_id

                # SOURCE: vllm/v1/core/sched/scheduler.py:L700-L711 —— 逐字
                #   （**每拍窥队·100µs 探测→未就绪 prepend 回侧队**——站 6）
                # try to promote blocked statuses while traversing skipped queue.
                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):
                    # SUBTRACTED: L704-L709 REMOTE_KVS 的 logger.debug
                    #   （delete[7]；三阻塞态共用此机制，ch16 的那一个实例
                    #   归 KV 传输域）
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # SUBTRACTED: L713-L722 stale 在途推迟一拍（ch12 域）、
                #   L724-L737 LoRA 约束跳过。

                # Get already-cached tokens.
                # SUBTRACTED: L744-L766 前缀重命中（kv_cache_manager.
                #   get_computed_blocks——ch15 域；精简版恒 0 命中）。
                num_computed_tokens = request.num_computed_tokens

                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed
                # requests, which have output tokens.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L874-L879
                num_new_tokens = request.num_tokens - num_computed_tokens

                # SUBTRACTED: L881-L897 spec 均匀填充（pad_spec_decode——
                #   ch33 域）；L899-L911 long_prefill 阈值钳制与 chunked_
                #   prefill 断流（ch10 域——精简版 token_budget 内一次吃满）。
                # SOURCE: vllm/v1/core/sched/scheduler.py:L913-L914
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # SUBTRACTED: L916-L1017 encoder/mamba/异步 KV load 预约与
                #   allocate_slots/None-break（KV 装配域 ch13——精简版无 KV
                #   面，准入恒成功）。

                # SOURCE: vllm/v1/core/sched/scheduler.py:L1022 出队
                request = request_queue.pop_request()

                # SOURCE: vllm/v1/core/sched/scheduler.py:L1055 入 running
                self.running.append(request)
                # SUBTRACTED: L1056-L1058 log_stats 的 SCHEDULED 事件时间戳
                #   参数（观测面）。
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED)
                # SOURCE: vllm/v1/core/sched/scheduler.py:L1060-L1066 状态分流
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                # SUBTRACTED: L1067-L1071 LoRA 登记。
                # SOURCE: vllm/v1/core/sched/scheduler.py:L1072-L1074 落位记账
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # SUBTRACTED: L1076-L1078 spec 占位登记（ch33）；
                #   L1079-L1082 _inflight_prefills 登记（切块 prefill 域——
                #   is_prefill_chunk 的推进在 _update_after_schedule 内）；
                #   L1083-L1091 encoder 登记。

            # SOURCE: vllm/v1/core/sched/scheduler.py:L1102-L1103
            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

            # SUBTRACTED: L1105-L1109 DP prefill balancing 记账。

        # Check if the scheduling constraints are satisfied.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1112-L1113
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1115
        assert token_budget >= 0
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1116-L1121
        assert len(self.running) <= self.max_num_running_reqs

        # SUBTRACTED: L1124-L1205 num_common_prefix_blocks/NewRequestData/
        #   CachedRequestData/连接器 meta/dynamic spec K（worker 增量下发
        #   与 KV/连接器域——ch10/ch16/ch18）。

        # Construct the scheduler output.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1207-L1228 —— 字段面精简
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=None,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
        )

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1251-L1252
        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1317-L1345 _update_after_schedule
    #   —— 逐 token 记账 + is_prefill_chunk + has_structured_output_requests 旗标
    # SOURCE: vllm/v1/core/sched/scheduler.py:L1317-L1345
    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.num_in_flight_tokens += num_scheduled_token
            # SUBTRACTED: defer_block_free 的 last_sched_seq 栅栏（延迟释放域）。
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )
            # SUBTRACTED: L1344-L1345 _inflight_prefills.discard（切块登记域）。

        # SUBTRACTED: L1347-L1360 routed experts 快照（ch23 域）。

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1669-L2051 update_from_output
    #   （语法门切面：站 7 推进 + m13 收账）
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output,
    ) -> dict[int, list[EngineCoreOutput]]:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1671-L1672
        sampled_token_ids = model_runner_output.sampled_token_ids
        # SUBTRACTED: L1673-L1682 logprobs/pooler/connector/routed/perf 的
        #   取用与 defer_block_free 排空（观测与子系统域——HOST SEAM 的
        #   ModelRunnerOutput 只带 req_id_to_index/sample 两个采样面字段）。
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        # SUBTRACTED: L1684-L1691 perf_stats；L1693 outputs 簿记同款保留。
        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        # SUBTRACTED: L1697-L1726 failed_kv_load/routing_offsets（KV/路由域）。

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1727-L1732
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1735-L1743 扣在途
            if request is not None:
                request.num_in_flight_tokens -= num_tokens_scheduled
            # SUBTRACTED: L1744-L1747 stale 锁步 drain（ch12 域）；
            #   L1748-L1751 failed_kv_load 跳过（KV 域）。
            if request is None or request.is_finished():
                # SOURCE: vllm/v1/core/sched/scheduler.py:L1752-L1762 abort 期
                #   完成幂等 continue（注释原话节选）
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.
                # in pipeline parallelism or async scheduling).
                continue

            # SUBTRACTED: L1764-L1795 stale drop/spec 拒绝回扣/encoder 释放
            #   （ch12/ch33/mm 域）。

            # SOURCE: vllm/v1/core/sched/scheduler.py:L1766-L1769 定位采样行
            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            # SOURCE: vllm/v1/core/sched/scheduler.py:L1797-L1799
            stopped = False
            new_token_ids = generated_token_ids
            # SUBTRACTED: L1800-L1806 new_logprobs/pooler/kv 参数位与
            #   status_before_stop/num_output_tokens_before 簿记（观测域）。

            # Check for stop and update request status.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1808-L1810
            #   _update_request_with_output 的调用位（stop 检查本体 =
            #   check_stop/长度帽/eos——ch9/ch11 域；精简版直接挂账：
            #   token 进 output/all 双列表，视为未停）
            request.append_output_token_ids(new_token_ids)
            # SUBTRACTED: L1811-L1813 pooling 分支（池化域）。

            # SOURCE: vllm/v1/core/sched/scheduler.py:L1817-L1843 —— 逐字
            #   （**采样后推进 FSM：本步真正采出的 token 才喂语法状态机**；
            #   should_advance→trim_reasoning_for_advance→accept_tokens；
            #   拒收=引擎 bug→FINISHED_ERROR+resumable=False）
            if new_token_ids and self.structured_output_manager.should_advance(
                request, new_token_ids=new_token_ids
            ):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                grammar = struct_output_request.grammar
                assert isinstance(grammar, StructuredOutputGrammar)
                # new_token_ids can be a mixed block of reasoning content, then
                # the reasoning end marker, then the start of the grammar content.
                # Trim the reasoning content so the grammar only sees grammar content.
                advance_token_ids = (
                    self.structured_output_manager.trim_reasoning_for_advance(
                        request, new_token_ids
                    )
                )
                if advance_token_ids and not grammar.accept_tokens(
                    req_id, advance_token_ids
                ):
                    # SUBTRACTED: logger.error("Unexpected: grammar rejected
                    #   tokens %s for request %s. Terminating request.",
                    #   ...)（L1833-L1838——delete[7]；真实源码自认措辞归正文）
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True

            # SUBTRACTED: L1845-L1938 routed experts/finished 输出装配/
            #   preempted 收集/preemption 处理（ch11/ch23 域）；stopped 由
            #   语法拒收位直接置位后在此收账。
            if stopped:
                stopped_running_reqs.add(request)

        # Remove the stopped requests from the running and waiting queues.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1949-L1953
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.skipped_waiting.remove_requests(stopped_preempted_reqs)

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1954-L1957 —— 逐字（**编译
        #   失败同拍收账：只杀单请求**；failed_kv_load 同路合并减去——KV 域）
        error_req_ids = set(self.grammar_compile_error_reqs)
        self.grammar_compile_error_reqs.clear()

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1959-L1971 —— 逐字（空 token
        #   错误回执带回前端）
        if error_req_ids:
            error_reqs = self.finish_requests(
                error_req_ids, RequestStatus.FINISHED_ERROR
            )
            for request in error_reqs:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )

        # SUBTRACTED: L1973-L2040 KV connector 收尾/stats/engine_core_outputs
        #   合成（ch16/观测域——返回值退化为 dict[int, list[EngineCoreOutput]]，
        #   与真实返回型 dict[int, EngineCoreOutputs] 的 outputs 字段同内容面）。
        return outputs

    # SUBTRACTED: get_grammar_bitmask（L1646-L1668）——本章交棒点：ch9 站 4
    #   已嵌同段真源码；掩码批装配/行序/GrammarOutput 全归 ch31（delete[2]
    #   的既定边界：调度侧此函数不进精简版）。

    # SUBTRACTED: spec 路径的 should_advance 调用块（L2155-L2235——Rejection
    #   Sampler/validate_tokens 的调用点，ch32/ch33 域）。

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2050-L2055 —— 逐字（**阻塞态
    #   判定：三阻塞态共用**——ch11 已立机制全貌，本章讲语法门这一个实例）
    @staticmethod
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2050-L2055
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2057-L2062 —— 逐字（阻塞态进
    #   skipped_waiting 侧队而非正常 waiting——不挡人）
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2057-L2062
    def _enqueue_waiting_request(self, request: Request) -> None:
        if self._is_blocked_waiting_status(request.status):
            self.skipped_waiting.add_request(request)
        else:
            self.waiting.add_request(request)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2064-L2074 —— 逐字（FCFS:
    #   skipped 优先窥）
    # SOURCE: vllm/v1/core/sched/scheduler.py:L2064-L2074
    def _select_waiting_queue_for_scheduling(self):
        if self.policy == SchedulingPolicy.FCFS:
            return self.skipped_waiting or self.waiting or None

        # PRIORITY mode: compare queue heads when both queues are non-empty.
        # SUBTRACTED: L2071-L2073 PRIORITY 双头比较（精简版 FCFS-only）
        return self.waiting or self.skipped_waiting or None

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2678-L2712 _try_promote_blocked_
    #   waiting_request（**门控另一半**：grammar 就绪→WAITING；
    #   Exception→grammar_compile_error_reqs）
    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        # SUBTRACTED: L2683-L2694 WAITING_FOR_REMOTE_KVS 分支（P/D KV 传输域
        #   ch16——依赖 finished_recving_kv_req_ids/_update_waiting_for_remote_
        #   kv/kv_cache_manager，均不在本章切面）。
        if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            structured_output_req = request.structured_output_request
            if not structured_output_req or structured_output_req.grammar is None:
                return False
            if isinstance(structured_output_req.grammar, Exception):
                self.grammar_compile_error_reqs.add(request.request_id)
                return False
            request.status = RequestStatus.WAITING
            return True

        # SUBTRACTED: L2705-L2708 WAITING_FOR_STREAMING_REQ 分支（流式会话域
        #   ch38——依赖 streaming_queue）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2710-L2712 —— 逐字
        raise AssertionError(
            "Unexpected blocked waiting status in promotion: "
            f"{request.status.name} for request {request.request_id}"
        )

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2237-L2299 finish_requests
    #   （两遍式：先出队、再改状态+释放）
    def finish_requests(
        self, request_ids, finished_status: RequestStatus
    ) -> list[Request]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            List of requests that were aborted. Will not include any that were
            already finished.
        """
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
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2263-L2276 —— 逐字
        #   （streaming 计数行减去——ch38 域）
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
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2279-L2284
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2287-L2297
        for request in valid_requests:
            # SUBTRACTED: WAITING_FOR_REMOTE_KVS 的 delay_free 判定（KV 域）
            request.status = finished_status
            self._free_request(request)

        return valid_requests

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2300-L2340 _free_request（收账
    #   面：del requests + KV 释放钩子减去）
    def _free_request(self, request: Request) -> None:
        assert request.is_finished()

        # SUBTRACTED: _inflight_prefills.discard/_connector_finished/
        #   delay_free_blocks KV 释放面（ch13/ch16 域）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L2332
        del self.requests[request.request_id]
