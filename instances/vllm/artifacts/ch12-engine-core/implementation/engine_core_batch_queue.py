# ch12 精简版：EngineCore.step_with_batch_queue —— 用 (SchedulerOutput, future) 对
# 填满流水线并行 stage、消除气泡。
#
# 本文件是 vLLM v1 真实源码（pin f3fef123）的「只做减法」精简版：与 vLLM
# 同名、同结构、同控制流，只删不增。每个 def/class 标 `# SOURCE: vllm/...:Lxxx`，
# 每处删除标 `# SUBTRACTED:`。把真实 vLLM 删掉所有 SUBTRACTED 分支应当 ≈ 本文件。
#
# 为了让读者能「跑起来、打断点、追数值」而不必拉起 CUDA/PP/IPC，本文件用最小的
# 测试替身（FakeExecutor / FakeScheduler / 同步 Future）驱动真正的 step_with_batch_queue
# 控制流。这些替身替换的是 dossier subtraction_plan.delete[4] 批准删除的
# 进程/IPC/CUDA 编排（ch07/ch11 内容，与本章 deque 机制正交），不是对主线方法的改写。

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, cast

# ----------------------------------------------------------------------------
# 测试替身：替换被 subtraction_plan.delete[4] 批准移除的 IPC/CUDA/PP 编排。
# 不是 vLLM 的精简，而是让真正的 step_with_batch_queue 能在 host 上跑的最小脚手架。
# ----------------------------------------------------------------------------


@dataclass
class ModelRunnerOutput:  # SOURCE: vllm/v1/outputs.py:L166 (占位替身)
    """精简的模型输出占位。真实定义在 vllm/v1/outputs.py，本章只关心它作为
    future.result() 的返回值在队列里流动，不关心其字段。"""

    batch_id: int = -1


@dataclass
class EngineCoreOutputs:  # SOURCE: vllm/v1/engine/__init__.py:L212 (占位替身)
    """精简的引擎输出占位（真实定义 vllm/v1/engine/__init__.py）。"""

    batch_id: int = -1


@dataclass
class SchedulerOutput:
    # SOURCE: vllm/v1/core/sched/output.py:L221-L227
    # 两个布尔标志（真实源码中仅 async scheduling 下置位）：
    #   has_structured_output_requests —— 本批是否含结构化输出请求；
    #   pending_structured_output_tokens —— 本批是否还缺算 grammar bitmask 所需的
    #     输出 token，是触发 deferred sampling 的开关。
    # Whether any of the scheduled requests use structured output.
    # Set only in async scheduling case.
    has_structured_output_requests: bool = False
    # Whether the scheduled requests have all the output tokens they
    # need to perform grammar bitmask computation.
    pending_structured_output_tokens: bool = False

    # 本批被调度的 token 总数；>0 表示真的执行了模型。真实字段同名。
    total_num_scheduled_tokens: int = 0
    batch_id: int = -1


# SOURCE: vllm/v1/executor/abstract.py:L210-L258 (Executor 执行接口) —— 测试替身
# 替换 dossier subtraction_plan.delete[4] 移除的 IPC/CUDA worker RPC 派发；
# 用立即完成的同步 Future 让真正的 step_with_batch_queue 能在 host 上跑。
class FakeExecutor:
    """替身 executor，提供 step_with_batch_queue 依赖的三个非阻塞接口
    （execute_model / sample_tokens / take_draft_token_ids）与 max_concurrent_batches。
    真实 executor 把这些 RPC 派发到 worker 进程；这里用立即完成的同步 Future 替代，
    便于 host 上确定性地观察队列时间轴。"""

    # SOURCE: vllm/v1/executor/abstract.py:L210-L258 (测试替身构造)
    def __init__(self, max_concurrent_batches: int = 2):
        self._max_concurrent_batches = max_concurrent_batches
        self.draft_token_ids: Any = None

    @property
    def max_concurrent_batches(self) -> int:
        # SOURCE: vllm/v1/executor/abstract.py:L256-L258 (max_concurrent_batches)
        # 真实三处定义见 implementation/executor_max_concurrent_batches.py。
        return self._max_concurrent_batches

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L306 (execute_model，non_block)
    def execute_model(self, scheduler_output, non_block: bool = False) -> Future:
        # 非阻塞提交执行，返回 exec_future。替身里立即完成。
        f: Future = Future()
        f.set_result(ModelRunnerOutput(batch_id=scheduler_output.batch_id))
        return f

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L318 (sample_tokens，non_block)
    def sample_tokens(self, grammar_output, non_block: bool = False) -> Future:
        # 从前向里拆出的非阻塞采样调用；立即采样与 deferred 采样都靠它。
        f: Future = Future()
        f.set_result(ModelRunnerOutput())
        return f

    # SOURCE: vllm/v1/executor/abstract.py:L252-L254 (take_draft_token_ids)
    def take_draft_token_ids(self):
        # deferred + spec decode 时取上一步草稿 token，先于 bitmask 计算。
        return self.draft_token_ids


# SOURCE: vllm/v1/core/sched/scheduler.py (Scheduler) —— 测试替身
# 替换 dossier delete[4] 移除的真实调度器内部；只实现本章控制流读写的方法。
class FakeScheduler:
    """替身 scheduler，提供 step_with_batch_queue 依赖的接口。真实 Scheduler 在
    vllm/v1/core/sched/scheduler.py；这里只实现本章控制流读写的方法，行为可由测试编排。"""

    # SOURCE: vllm/v1/core/sched/scheduler.py (Scheduler.__init__，测试替身)
    def __init__(self):
        self._pending: deque[SchedulerOutput] = deque()
        self.get_grammar_bitmask_calls: list[SchedulerOutput] = []
        self.update_from_output_calls: list[tuple] = []
        self.update_draft_calls: list[tuple] = []

    # SOURCE: vllm/v1/core/sched/scheduler.py (测试编排辅助，非 vLLM 接口)
    def queue(self, scheduler_output: SchedulerOutput) -> None:
        self._pending.append(scheduler_output)

    # SOURCE: vllm/v1/core/sched/interface.py:L185 (has_requests)
    def has_requests(self) -> bool:
        return bool(self._pending)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L310 (schedule)
    def schedule(self) -> SchedulerOutput:
        return self._pending.popleft()

    def get_grammar_bitmask(self, scheduler_output: SchedulerOutput):
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1224-L1246
        # 真实实现：为结构化输出请求计算 grammar bitmask，依赖
        # scheduler_output.scheduled_spec_decode_tokens —— 这正是 spec decode +
        # structured output 必须把采样推迟到拿到上一步 draft token 之后的根因。
        # 替身记录调用次序，正文用真实源码内嵌解读其内部。
        self.get_grammar_bitmask_calls.append(scheduler_output)
        return None

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1248 (update_from_output)
    def update_from_output(self, scheduler_output, model_output) -> dict:
        # pop 取结果后提交给 scheduler，产出 engine_core_outputs。
        self.update_from_output_calls.append((scheduler_output, model_output))
        return {0: EngineCoreOutputs(batch_id=scheduler_output.batch_id)}

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1623 (update_draft_token_ids_in_output)
    def update_draft_token_ids_in_output(self, draft_token_ids, scheduler_output) -> None:
        self.update_draft_calls.append((draft_token_ids, scheduler_output))


@dataclass
class _MinimalEngineCore:
    """承载 step_with_batch_queue 及其在 __init__ 尾段落地的字段的最小 EngineCore。

    真实类是 vllm/v1/engine/core.py:EngineCore（含 IPC/handshake/DP 等）。本精简版
    # SUBTRACTED: _perform_handshakes / DP coordinator / tensor_ipc / EngineCoreProc
    #             的 ZMQ 线程、is_ec_consumer/request_block_hasher 等与 batch queue
    #             主线无关的初始化全部不纳入（dossier subtraction_plan.delete[4]，
    #             那是 ch07/ch11 的内容；原 vllm/v1/engine/core.py EngineCoreProc 及以下）。
    只保留 __init__ 尾段中 batch queue 相关字段的落地（core.py:L184-L216）。
    """

    model_executor: FakeExecutor
    scheduler: FakeScheduler
    async_scheduling: bool = False
    use_spec_decode: bool = False
    aborts: list[str] = field(default_factory=list)

    # SOURCE: vllm/v1/engine/core.py:L184-L216  (EngineCore.__init__ 尾段，batch queue 部分)
    def __post_init__(self) -> None:
        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
        ) = None
        if self.batch_queue_size > 1:
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        # SUBTRACTED: is_ec_consumer / is_pooling_model / request_block_hasher 初始化
        #             (core.py:L196-L211) —— 与 batch queue 主线无关
        #             (dossier embed_excerpts elide + delete[1]/[2])。
        #             常规部署 is_ec_consumer 恒 True，故下面 model_executed 直接取
        #             scheduler_output.total_num_scheduled_tokens > 0。

        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
        # async_scheduling 是 SchedulerConfig 标志，间接经 max_concurrent_batches
        # 决定 batch_queue 是否启用，从而决定 step_fn 绑哪个 step。
        # 在真实代码里这里从 vllm_config 读；精简版由构造参数注入。

    # SOURCE: vllm/v1/engine/core.py:L406-L436  (EngineCore.step，非队列变体；本章对照用)
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output."""
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        # SUBTRACTED: log_error_detail / log_iteration_details 上下文管理器
        #             (core.py:L416-L419) —— 纯日志/诊断，不改控制流与数值
        #             (dossier subtraction_plan.delete[3])。
        model_output = future.result()
        if model_output is None:
            model_output = self.model_executor.sample_tokens(grammar_output)
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    # SOURCE: vllm/v1/engine/core.py:L447-L563  (EngineCore.step_with_batch_queue —— 本章主角)
    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """

        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule()
            # SUBTRACTED: with self.log_error_detail(scheduler_output): 包裹
            #             (core.py:L472) —— 纯日志上下文 (delete[3])。
            exec_future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )
            # SUBTRACTED: is_ec_consumer 分支 (core.py:L476-L477) —— 仅分布式
            #             encoder-cache transfer 非消费端把 model_executed 强制 False。
            #             常规部署 is_ec_consumer 恒 True (delete[1])，故直接：
            model_executed = scheduler_output.total_num_scheduled_tokens > 0

            # SUBTRACTED: is_pooling_model 快路 (core.py:L479 中的 is_pooling_model 条件)
            #             —— pooling/embedding 模型无采样，把 exec_future 直接 cast 成
            #             最终 future。本章聚焦生成路径；保留 `not model_executed` 那支即可
            #             覆盖『没调度到 token 无需采样』(delete[2])。
            if not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                batch_queue.appendleft((future, scheduler_output, exec_future))
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, True

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        # SUBTRACTED: with self.log_error_detail / log_iteration_details: 包裹
        #             (core.py:L517-L520) —— 纯日志/诊断上下文 (delete[3])。
        model_output = future.result()
        if model_output is None:
            # None from sample_tokens() implies that the original execute_model()
            # call failed - raise that exception.
            exec_model_fut.result()
            raise RuntimeError("unexpected error")

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # If we are doing speculative decoding with structured output,
            # we need to get the draft token ids from the prior step before
            # we can compute the grammar bitmask for the deferred request.
            if self.use_spec_decode:
                draft_token_ids = self.model_executor.take_draft_token_ids()
                assert draft_token_ids is not None
                # Update the draft token ids in the scheduler output to
                # filter out the invalid spec tokens, which will be padded
                # with -1 and skipped by the grammar bitmask computation.
                self.scheduler.update_draft_token_ids_in_output(
                    draft_token_ids, deferred_scheduler_output
                )
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

        return engine_core_outputs, model_executed

    # SOURCE: vllm/v1/engine/core.py:L565-L572  (EngineCore._process_aborts_queue，精简)
    def _process_aborts_queue(self):
        # 真实实现从 self.aborts_queue 取出执行期到达的 abort 请求并下发给 scheduler。
        # 本章只需它作为 pop 取结果后、update_from_output 前的一个调用点存在。
        self.aborts.clear()

    # SOURCE: vllm/v1/engine/core.py:L1156-L1162  (EngineCore.has_work)
    def has_work(self) -> bool:
        """Returns true if the engine should be stepped."""
        # SUBTRACTED: self.engines_running （DP 协调相关，core.py:L1155）—— 不在本章
        #             进程/IPC 编排范围 (delete[4])。保留队列保活语义这一支：
        return self.scheduler.has_requests() or bool(self.batch_queue)
