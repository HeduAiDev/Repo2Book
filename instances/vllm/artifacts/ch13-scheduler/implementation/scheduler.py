# SOURCE: vllm/v1/core/sched/scheduler.py
# Token 为中心的连续批处理调度器（只做减法精简版）。与真实
# vllm/v1/core/sched/scheduler.py 同名同结构同控制流：schedule() 两阶段
# （先 RUNNING 后 WAITING）按 token_budget 预算分配；update_from_output() 吃
# 模型输出、追加 token、判 stop、free。删除项全部 # SUBTRACTED: 标注，仅删
# dossier.subtraction_plan.delete 批准的子系统（KVConnector/encoder/mm/
# structured/mamba/LoRA/streaming/PRIORITY/PP/spec 统计/可观测性）。
from __future__ import annotations

import itertools
import time
from collections import defaultdict

from .interface import PauseState
from .kv_cache_manager import KVCacheBlocks, KVCacheManager
from .output import CachedRequestData, NewRequestData, SchedulerOutput
from .request import RequestStatus, check_stop
from .request_queue import SchedulingPolicy, create_request_queue


# SUBTRACTED: record_function_or_nullcontext（torch profiler 包裹，可观测性，
#   dossier.delete 批准）—— 原 scheduler.py 全程用它包 allocate_slots 等；这里用
#   一个空上下文管理器顶替，控制流不变。
class _nullctx:
    # SOURCE: vllm/v1/core/sched/scheduler.py (record_function_or_nullcontext 包裹)
    def __enter__(self):
        # SOURCE: vllm/v1/core/sched/scheduler.py
        return None

    def __exit__(self, *a):
        # SOURCE: vllm/v1/core/sched/scheduler.py
        return False


def record_function_or_nullcontext(_name: str) -> _nullctx:
    # SOURCE: vllm/v1/core/sched/scheduler.py (profiler 包裹，精简为空上下文)
    return _nullctx()


# SOURCE: vllm/v1/core/sched/scheduler.py:L52 class Scheduler
class Scheduler:
    # SOURCE: vllm/v1/core/sched/scheduler.py:L63 __init__
    def __init__(
        self,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 2048,
        max_model_len: int = 4096,
        num_gpu_blocks: int = 1 << 30,
        block_size: int = 16,
        long_prefill_token_threshold: int = 0,
        enable_chunked_prefill: bool = True,
        num_spec_tokens: int = 0,
        log_stats: bool = False,
    ) -> None:
        # SUBTRACTED: 真实 __init__ 从 VllmConfig 拆出几十个字段并初始化
        #   KVConnector/EncoderCacheManager/StructuredOutputManager/
        #   kv_event_publisher/perf_metrics/lora 等（原 scheduler.py:L68-L301）——
        #   均为 dossier.delete 批准的子系统；这里直接用裸标量构造，只保留连续批处理
        #   骨架所需字段。
        self.scheduler_config = _SchedulerConfig(
            long_prefill_token_threshold=long_prefill_token_threshold,
            enable_chunked_prefill=enable_chunked_prefill,
        )
        self.max_model_len = max_model_len
        self.num_spec_tokens = num_spec_tokens
        self.num_lookahead_tokens = num_spec_tokens
        self.log_stats = log_stats

        # SOURCE: vllm/v1/core/sched/scheduler.py:L101
        self.max_num_running_reqs = max_num_seqs
        # SOURCE: vllm/v1/core/sched/scheduler.py:L102 缺省回退到 max_num_batched_tokens
        self.max_num_scheduled_tokens = max_num_batched_tokens

        self.policy = SchedulingPolicy.FCFS

        self.kv_cache_manager = KVCacheManager(
            num_gpu_blocks=num_gpu_blocks, block_size=block_size
        )

        # req_id -> Request
        self.requests: dict[str, object] = {}
        # SOURCE: vllm/v1/core/sched/scheduler.py:L162
        self.waiting = create_request_queue(self.policy)
        # skipped_waiting：被阻塞态(本章无)暂时跳过的队列，保留以维持 schedule() 逻辑
        self.skipped_waiting = create_request_queue(self.policy)
        # SOURCE: vllm/v1/core/sched/scheduler.py:L165
        self.running: list = []

        # 跨拍状态
        self.finished_req_ids: set[str] = set()
        # SOURCE: vllm/v1/core/sched/scheduler.py: prev_step_scheduled_req_ids
        self.prev_step_scheduled_req_ids: set[str] = set()
        # SOURCE: vllm/v1/core/sched/scheduler.py:L258
        self._pause_state: PauseState = PauseState.UNPAUSED
        self.num_waiting_for_streaming_input = 0

    # ------------------------------------------------------------------ #
    # 入队
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L1665 add_request（精简：无 streaming session）
    def add_request(self, request) -> None:
        # SUBTRACTED: streaming-input session 复用分支（_update_request_as_session），
        #   dossier.delete 批准；普通请求直接入 requests + waiting。
        self.requests[request.request_id] = request
        self.waiting.add_request(request)

    # ------------------------------------------------------------------ #
    # schedule —— 连续批处理一拍的入口
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L310 schedule
    def schedule(self) -> SchedulerOutput:
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

        scheduled_new_reqs: list = []
        scheduled_resumed_reqs: list = []
        scheduled_running_reqs: list = []
        preempted_reqs: list = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        # SUBTRACTED: encoder_compute_budget / scheduled_encoder_inputs 初始化
        #   （encoder/mm，dossier.delete 批准）。

        # First, schedule the RUNNING requests.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L345
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            # SOURCE: vllm/v1/core/sched/scheduler.py:L350 async 提前剪枝
            if (
                request.num_output_placeholders > 0
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: avoid scheduling an extra step when we are
                # sure the previous step has reached request.max_tokens.
                req_index += 1
                continue

            # SOURCE: vllm/v1/core/sched/scheduler.py:L366 ‘追赶’公式
            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )

            # SUBTRACTED: _try_schedule_encoder_inputs（encoder 预算截断）、
            #   _mamba_block_aligned_split（mamba 对齐）—— dossier.delete 批准。

            # SOURCE: vllm/v1/core/sched/scheduler.py:L404
            if num_new_tokens == 0:
                # The request cannot be scheduled (e.g. PP>1 prompt fully
                # scheduled but not finished, or async already reached max).
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
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
                    # SUBTRACTED: PRIORITY 分支（max(...) 选最低优先级 + req_index
                    #   回退，dossier.delete 批准）。FCFS 默认抢队尾即可讲清机制：
                    preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # SOURCE: vllm/v1/core/sched/scheduler.py:L482 spec decode 占位记账
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
                request.spec_token_ids = []
            # SUBTRACTED: encoder 缓存 allocate（encoder/mm，dossier.delete 批准）。

        # SUBTRACTED: scheduled_loras 记账（LoRA，dossier.delete 批准）。

        # Next, schedule the WAITING requests.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L526 —— `if not preempted_reqs` 守卫
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            step_skipped_waiting = create_request_queue(self.policy)

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                request = request_queue.peek_request()
                request_id = request.request_id

                # SUBTRACTED: 被阻塞 waiting 状态(WAITING_FOR_REMOTE_KVS 等)的提升、
                #   max_loras 约束、KVConnector 外部命中分支（dossier.delete 批准）。

                # Get already-cached tokens.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L571
                if request.num_computed_tokens == 0:
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )
                    # SUBTRACTED: KVConnector 外部命中 num_external_computed_tokens
                    num_computed_tokens = num_new_local_computed_tokens
                    assert num_computed_tokens <= request.num_tokens
                else:
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                # SOURCE: vllm/v1/core/sched/scheduler.py:L630
                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed
                # requests, which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens
                threshold = self.scheduler_config.long_prefill_token_threshold
                if 0 < threshold < num_new_tokens:
                    num_new_tokens = threshold

                # chunked prefill has to be enabled explicitly to allow
                # pooling requests to be chunked
                if (
                    not self.scheduler_config.enable_chunked_prefill
                    and num_new_tokens > token_budget
                ):
                    # If chunked_prefill is disabled, stop scheduling here.
                    break

                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # SUBTRACTED: encoder inputs 调度、mamba 对齐、effective_lookahead
                #   的 P/D edge case（dossier.delete 批准）。

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # SUBTRACTED: KVConnector update_state_after_alloc / load_kv_async
                #   置 WAITING_FOR_REMOTE_KVS 分支（dossier.delete 批准）。

                request = request_queue.pop_request()
                # SOURCE: vllm/v1/core/sched/scheduler.py:L765
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

            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

        # Check if the scheduling constraints are satisfied.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L806
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

        # SUBTRACTED: num_common_prefix_blocks 计算（cascade attention，
        #   dossier.delete 批准）。

        # Construct the scheduler output.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L829
        # SUBTRACTED: use_v2_model_runner 分支（合并 resumed 进 new + _all_token_ids）。
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids()
            )
            for req in scheduled_new_reqs
        ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            # finished_req_ids contains the request IDs that are finished in
            # between the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
        )

        # SUBTRACTED: KVConnector / ECConnector metadata 注入（dossier.delete 批准）。

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    # ------------------------------------------------------------------ #
    # 抢占
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L910 _preempt_request
    def _preempt_request(self, request, timestamp: float) -> None:
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self.kv_cache_manager.free(request)
        # SUBTRACTED: encoder_cache_manager.free（encoder，dossier.delete 批准）。
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        # SUBTRACTED: record_event（可观测性，dossier.delete 批准）。
        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)

    # ------------------------------------------------------------------ #
    # 调度后乐观推进
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L932 _update_after_schedule
    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled. This lets us schedule the prefill request
        # again immediately in the next scheduling step. If some tokens (e.g.
        # spec tokens) are rejected later, the number of computed tokens will
        # be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            # SUBTRACTED: has_structured_output_requests 累计（约束解码，delete 批准）。

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    # ------------------------------------------------------------------ #
    # 增量打包
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L1001 _make_cached_request_data
    def _make_cached_request_data(
        self,
        running_reqs: list,
        resumed_reqs: list,
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # SUBTRACTED: PP(pipeline parallel) 下回传采样 token 的 new_token_ids 分支
            #   （use_pp，dossier.delete 批准）—— 非 PP 时 new_token_ids 恒空，worker
            #   自己缓存采样 token。
            scheduled_in_prev_step = req_id in self.prev_step_scheduled_req_ids
            if idx >= num_running_reqs:
                assert not scheduled_in_prev_step
                resumed_req_ids.add(req_id)
            if not scheduled_in_prev_step:
                all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1529 _select_waiting_queue_for_scheduling
    def _select_waiting_queue_for_scheduling(self):
        # SUBTRACTED: PRIORITY 比较分支（dossier.delete 批准）。FCFS：先清 skipped。
        return self.skipped_waiting or self.waiting or None

    # ------------------------------------------------------------------ #
    # 反馈环：吃模型输出
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/core/sched/scheduler.py:L1248 update_from_output
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output,
    ) -> dict[int, list]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        outputs: dict[int, list] = defaultdict(list)

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck.
        stopped_running_reqs: set = set()
        stopped_preempted_reqs: set = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in async scheduling).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            # SOURCE: vllm/v1/core/sched/scheduler.py:L1312 spec 拒绝回退
            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # If some tokens are rejected, num_computed_tokens is decreased
                # by the number of rejected tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                # SUBTRACTED: make_spec_decoding_stats（spec 统计，delete 批准）。

            # SUBTRACTED: _free_encoder_inputs（encoder，delete 批准）。

            stopped = False
            new_token_ids = generated_token_ids
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            # SUBTRACTED: pooling 停止分支 + structured grammar accept_tokens
            #   （pooling/约束解码，delete 批准）。

            finish_reason = None
            if stopped:
                finish_reason = request.get_finished_reason()
                self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # SUBTRACTED: logprobs / prompt_logprobs / num_nans_in_logits 抽取
            #   （ch10 主题，delete 批准）。
            if new_token_ids or stopped:
                # Add output for this Request.
                outputs[request.client_index].append(
                    _EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                    )
                )

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = [r for r in self.running if r not in stopped_running_reqs]
        if stopped_preempted_reqs:
            self.waiting.remove_requests(stopped_preempted_reqs)

        # SUBTRACTED: KV xfer finished / KV cache events publish / stats（delete 批准）。

        return dict(outputs)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1559 _update_request_with_output
    def _update_request_with_output(
        self, request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if a request is
        # still being prefilled, we expect the model runner to return empty
        # token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)
            # Check for stop and update request state.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1750 _free_request
    def _free_request(self, request) -> None:
        assert request.is_finished()
        # SUBTRACTED: _connector_finished / delay_free_blocks（KVConnector，delete 批准）。
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        self.kv_cache_manager.free(request)
        del self.requests[request_id]

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1777 set_pause_state
    def set_pause_state(self, pause_state: PauseState) -> None:
        self._pause_state = pause_state


# SUBTRACTED: 真实 SchedulerConfig 是 VllmConfig 的庞大子配置；这里只保留 schedule()
#   实际读取的两个字段。
class _SchedulerConfig:
    # SOURCE: vllm/config/scheduler.py:SchedulerConfig（仅留 schedule() 读取的字段）
    def __init__(
        self, long_prefill_token_threshold: int, enable_chunked_prefill: bool
    ) -> None:
        # SOURCE: vllm/config/scheduler.py:SchedulerConfig
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.enable_chunked_prefill = enable_chunked_prefill
        self.async_scheduling = False


# SUBTRACTED: 真实 EngineCoreOutput 含 logprobs/events/kv_transfer_params 等大量字段
#   （ch10 主题）；这里只留 update_from_output 产出的最小三元组。
class _EngineCoreOutput:
    # SOURCE: vllm/v1/engine/__init__.py:EngineCoreOutput（仅留最小三元组，ch10 主题）
    def __init__(self, request_id: str, new_token_ids: list[int], finish_reason) -> None:
        # SOURCE: vllm/v1/engine/__init__.py:EngineCoreOutput
        self.request_id = request_id
        self.new_token_ids = new_token_ids
        self.finish_reason = finish_reason
