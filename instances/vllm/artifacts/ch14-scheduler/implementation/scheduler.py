# SPDX-License-Identifier: Apache-2.0
# 只做减法的精简版 Scheduler —— 与 vLLM 同名同结构同控制流。
# 聚焦 ch14：RUNNING 阶段抢占循环、waiting/skipped 双队列、update_from_output 生命周期回流。
# 验收判据：把真实 vLLM 删掉所有 # SUBTRACTED 分支 ≈ 本文件。
from request import Request, RequestStatus
from request_queue import FCFSRequestQueue, SchedulingPolicy, create_request_queue
from kv_cache_manager import KVCacheManager
from utils import check_stop, remove_all


# SOURCE: vllm/v1/core/sched/output.py (class SchedulerOutput) —— 仅本章相关字段
class SchedulerOutput:
    def __init__(self):
        # 首次调度的 WAITING 请求
        self.scheduled_new_reqs: list[str] = []
        # 抢占后回流又被拉起的 PREEMPTED 请求（resumed）
        self.scheduled_resumed_reqs: list[str] = []
        # 本拍继续推进的 RUNNING 请求
        self.scheduled_running_reqs: list[str] = []
        # SOURCE: vllm/v1/core/sched/output.py (preempted_req_ids)
        self.preempted_req_ids: set[str] = set()
        # SOURCE: vllm/v1/core/sched/output.py (finished_req_ids)
        self.finished_req_ids: set[str] = set()
        self.num_scheduled_tokens: dict[str, int] = {}
        # SUBTRACTED: SchedulerOutput 的增量/全量请求体组装、grid_mapping、
        # scheduled_encoder_inputs、num_common_prefix_blocks 等 —— 由 ch13 负责。
        self.scheduled_spec_decode_tokens: dict[str, list[int]] = {}


# SOURCE: vllm/v1/core/sched/scheduler.py:L? (class Scheduler) —— 只保留本章字段与方法
class Scheduler:
    def __init__(self, block_capacity: int = 8, max_num_running_reqs: int = 8,
                 max_model_len: int = 1024, policy: SchedulingPolicy = SchedulingPolicy.FCFS):
        self.policy = policy
        self.max_num_running_reqs = max_num_running_reqs
        self.max_model_len = max_model_len
        self.kv_cache_manager = KVCacheManager(block_capacity)

        # 三队列：running + waiting + skipped_waiting（双队列防队头阻塞）
        self.running: list[Request] = []
        self.waiting = create_request_queue(policy)
        # SOURCE: vllm/v1/core/sched/scheduler.py (self.skipped_waiting)
        self.skipped_waiting = create_request_queue(policy)

        self.requests: dict[str, Request] = {}
        # SOURCE: vllm/v1/core/sched/scheduler.py (self.finished_req_ids)
        self.finished_req_ids: set[str] = set()
        self.finished_req_ids_dict = None

        # SUBTRACTED: num_lookahead_tokens / log_stats / encoder_cache_manager /
        # connector / structured_output_manager / _pause_state(恒 UNPAUSED) /
        # lora_config / vllm_config 等 —— spec/统计/编码器/远程KV/约束解码/暂停/LoRA
        # 均按 subtraction_plan.delete 删除，与抢占·回流主线正交。
        self.num_lookahead_tokens = 0

    # SOURCE: vllm/v1/core/sched/scheduler.py (add_request 简化入口)
    def add_request(self, request: Request) -> None:
        self.requests[request.request_id] = request
        self._enqueue_waiting_request(request)

    # ---- 双队列归类与选择（防队头阻塞） ----

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1516 (_is_blocked_waiting_status)
    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1516
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1523 (_enqueue_waiting_request)
    def _enqueue_waiting_request(self, request: Request) -> None:
        if self._is_blocked_waiting_status(request.status):
            self.skipped_waiting.add_request(request)
        else:
            self.waiting.add_request(request)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1529 (_select_waiting_queue_for_scheduling)
    def _select_waiting_queue_for_scheduling(self) -> "FCFSRequestQueue | None":
        # SUBTRACTED: PRIORITY 分支（比较两队队头）—— PRIORITY 整块按计划删除。
        # 原 vllm/v1/core/sched/scheduler.py:L1533-L1537。
        # FCFS：skipped 优先，再 waiting。
        return self.skipped_waiting or self.waiting or None

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1998 (_try_promote_blocked_waiting_request)
    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        # SUBTRACTED: 远程 KV / grammar / streaming 阻塞态在条件满足时提升回
        # WAITING/PREEMPTED 的具体判定 —— 对应特性（connector/约束解码/流式输入）
        # 均按 delete 删除。精简版保留调用点；恒返回 False（阻塞态本拍跳过）。
        return False

    # ---- 抢占 ----

    # SOURCE: vllm/v1/core/sched/scheduler.py:L910 (_preempt_request)
    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self.kv_cache_manager.free(request)
        # SUBTRACTED: self.encoder_cache_manager.free(request) —— 编码器缓存释放，
        # 多模态正交。原 vllm/v1/core/sched/scheduler.py:L920。
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        # SUBTRACTED: log_stats 下 record_event(PREEMPTED) —— 纯观测。原 L968-L969。

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L422-L472 (schedule 内的抢占 while True 块)
    # NOTE: 真实 vLLM 中此 while True 抢占循环*内联*在 schedule() 里。精简版把它抽到
    # _allocate_with_preemption 以便讲解/单测，控制流与原内联块一一对应（无新增逻辑）。
    def _allocate_with_preemption(self, request: Request, num_new_tokens: int):
        # SOURCE: vllm/v1/core/sched/scheduler.py:L422-L472
        preempted_reqs: list[Request] = []
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
            # SUBTRACTED: if self.policy == SchedulingPolicy.PRIORITY 整块
            # （max(running, key=(priority, arrival_time)) 抢最低优先级 + encoder 预算
            # 归还）—— PRIORITY 是等价的另一条路径，按 delete 删除。精简版只演示
            # FCFS LIFO 抢占。原 vllm/v1/core/sched/scheduler.py:L437-L460。
            preempted_req = self.running.pop()  # FCFS：抢 RUNNING 末尾（LIFO）

            self._preempt_request(preempted_req, 0.0)
            preempted_reqs.append(preempted_req)
            if preempted_req == request:
                # No more request to preempt. Cannot schedule this request.
                break
        return new_blocks, preempted_reqs

    # 测试用薄封装：把当前 request 从 running 取出后驱动抢占循环（对应 schedule 内逻辑）
    def _run_preemption_loop_for(self, request: Request, num_new_tokens: int):
        # SOURCE: vllm/v1/core/sched/scheduler.py:L422-L472 (抢占 while True 块的可单测入口)
        _, preempted = self._allocate_with_preemption(request, num_new_tokens)
        return preempted

    # SOURCE: vllm/v1/core/sched/scheduler.py:L? (schedule) —— 聚焦本章两条分支
    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []
        # 简化的 token 预算（主线由 ch13 负责）：每请求记 1 token。
        token_budget = 1 << 30

        # ---- RUNNING 阶段：遍历 running，分配失败则抢占 ----
        # SUBTRACTED: ch13 的 num_new_tokens 计算 / token 预算扣减 / encoder 输入调度 /
        # spec token 组装 —— 那些是 ch13 主线。本章只保留「allocate_slots 失败 → 抢占」。
        req_index = 0
        running_snapshot = list(self.running)
        for request in running_snapshot:
            if request not in self.running:
                # 已在抢占循环里被抢走
                continue
            num_new_tokens = 1
            # SOURCE: vllm/v1/core/sched/scheduler.py:L422-L472 抢占 while True
            new_blocks, just_preempted = self._allocate_with_preemption(
                request, num_new_tokens
            )
            preempted_reqs.extend(just_preempted)

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            out.num_scheduled_tokens[request.request_id] = num_new_tokens
            req_index += 1

        # ---- WAITING 阶段：本拍抢占过就完全跳过 ----
        # SOURCE: vllm/v1/core/sched/scheduler.py:L525-L526
        # SUBTRACTED: and self._pause_state == PauseState.UNPAUSED —— PAUSED 控制删除
        #（精简版恒 UNPAUSED）；但 `not preempted_reqs` 守卫是本章核心，保留。原 L568。
        if not preempted_reqs:
            # SOURCE: vllm/v1/core/sched/scheduler.py:L527
            step_skipped_waiting = create_request_queue(self.policy)

            # SOURCE: vllm/v1/core/sched/scheduler.py:L529
            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                # SOURCE: vllm/v1/core/sched/scheduler.py:L533
                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                request = request_queue.peek_request()

                # try to promote blocked statuses while traversing skipped queue.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L539-L550
                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):
                    # SUBTRACTED: WAITING_FOR_REMOTE_KVS 的 logger.debug —— 纯日志。原 L585-L589。
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # SUBTRACTED: max_loras 约束跳过分支（L596-L607）/ 前缀缓存
                # get_computed_blocks / connector 远程 KV load（L609-L805）—— LoRA/前缀
                # 缓存/远程 KV 均与本章正交，按 delete 删除。

                num_new_tokens = 1
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens
                )
                if new_blocks is None:
                    # 内存不足，本拍放不下更多 waiting；停止。
                    break

                request_queue.pop_request()

                # SOURCE: vllm/v1/core/sched/scheduler.py:L765-L775
                self.running.append(request)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                out.num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING

            # SOURCE: vllm/v1/core/sched/scheduler.py:L802-L804
            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

        out.scheduled_new_reqs = [r.request_id for r in scheduled_new_reqs]
        out.scheduled_resumed_reqs = [r.request_id for r in scheduled_resumed_reqs]
        out.scheduled_running_reqs = [r.request_id for r in scheduled_running_reqs]
        out.preempted_req_ids = {r.request_id for r in preempted_reqs}
        return out

    # ---- 生命周期回流：update_from_output ----

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1559 (_update_request_with_output)
    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> "tuple[list[int], bool]":
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
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

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1541 (_handle_stopped_request)
    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:
            return True

        # SUBTRACTED: streaming_queue 续接 / WAITING_FOR_STREAMING_REQ 重入队 /
        # _update_request_as_session —— streaming-input 多轮会话，按 delete 删除。
        # 精简版假设请求一次性输入，停止即真完成（resumable 恒 False，上面直接 return）。
        # 原 vllm/v1/core/sched/scheduler.py:L1546-L1556。
        self._enqueue_waiting_request(request)
        return False

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1750 (_free_request)
    def _free_request(self, request: Request, delay_free_blocks: bool = False):
        assert request.is_finished()

        # SUBTRACTED: _connector_finished（远程 KV 收尾 + delay_free_blocks）/
        # encoder_cache_manager.free / finished_req_ids_dict[client_index] —— connector/
        # encoder/多客户端分发均正交，按 delete 删除。原 L1818-L1825。
        request_id = request.request_id
        self.finished_req_ids.add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)
        return None

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1768 (_free_blocks)
    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1331-L1481 (update_from_output 本章相关段)
    def update_from_output(self, num_scheduled_tokens: "dict[str, list[int]]",
                           scheduled_spec_decode_tokens: "dict | None" = None):
        """num_scheduled_tokens: req_id -> 本拍模型采样出的 token id 列表（精简签名）。

        真实 update_from_output 入参为 (scheduler_output, model_runner_output)；
        本章只关心「每请求采样 token」这一驱动生命周期回流的核心输入。
        """
        scheduled_spec_decode_tokens = scheduled_spec_decode_tokens or {}
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()

        for req_id, generated_token_ids in num_scheduled_tokens.items():
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1296-L1305
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request is already finished (e.g., aborted while executing).
                continue

            # SUBTRACTED: failed_kv_load_req_ids 跳过 / req_index 取值 / pooler_output /
            # encoder input free —— connector/pooling/encoder 正交。原 L1335-L1352,L1380-L1382。

            # SOURCE: vllm/v1/core/sched/scheduler.py:L1312-L1329 (spec 回退)
            scheduled_spec_token_ids = scheduled_spec_decode_tokens.get(req_id)
            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # 被拒绝的草稿 token 须从 num_computed_tokens 回扣。
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # async 调度下 num_output_placeholders 同样回扣。
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                # SUBTRACTED: make_spec_decoding_stats 统计聚合。原 L1372-L1378。

            stopped = False
            new_token_ids = generated_token_ids
            status_before_stop = request.status

            # SOURCE: vllm/v1/core/sched/scheduler.py:L1349-L1353
            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            # SUBTRACTED: pooling 分支（pooler_output → FINISHED_STOPPED）/ structured
            # output grammar.accept_tokens 校验（L1396-L1416）—— pooling/约束解码正交。

            finish_reason = None
            if stopped:
                # SOURCE: vllm/v1/core/sched/scheduler.py:L1385-L1395
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

            # SUBTRACTED: logprobs 切片 + EngineCoreOutput 组装（L1435-L1474）——
            # 输出装配是 Part II 主题，本章只关心状态迁移。
            _ = finish_reason  # 真实在此组装进 EngineCoreOutput.finish_reason

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1438-L1443
        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)
