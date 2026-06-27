# 只做减法的忠实精简版 —— 镜像 vllm/v1/engine/core.py（pin f3fef123）
# 与 vLLM 同名、同结构、同控制流；只删不增。
#
# 本精简版保留两条主线：
#   1. EngineCore.step() / step_with_batch_queue() —— 一次推理迭代的编排。
#   2. EngineCoreProc.run_busy_loop() + 生命周期（pause/resume/sleep/wake_up）。
#
# model_executor / scheduler 是 EngineCore 真实持有的外部协作者（不在本章范围内），
# 这里以最小协议桩注入——不是杜撰玩具逻辑，而是把真实方法调用原样保留、由测试提供
# spy。删去所有 # SUBTRACTED 分支后，本文件 ≈ 真实 vllm/v1/engine/core.py 的单引擎子集。
#
# SUBTRACTED: SPDX 版权头、init_logger、以及大量与 step/忙循环/生命周期无关的导入
#             （tracing/instrument、numa_utils、gc_utils、tensor_ipc、kv_cache_utils、
#             structured_output、msgspec/zmq 等）—— 见各处具体删除注释。
import queue
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future
from enum import IntEnum
from functools import partial
from typing import Any

# SUBTRACTED: 从 vllm.* 包导入改为从本精简版同目录模块导入（仅 import 路径调整，
#             符号名/语义不变）。
from interfaces import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequestType,
    FinishReason,
    PauseMode,
    PauseState,
    RequestStatus,
    UtilityOutput,
    UtilityResult,
)


class EngineCore:
    # SOURCE: vllm/v1/engine/core.py:L91-L92
    """Inner loop of vLLM's Engine."""

    # SOURCE: vllm/v1/engine/core.py:L94-L229
    def __init__(
        self,
        vllm_config: Any,
        model_executor: Any,
        scheduler: Any,
        log_stats: bool = False,
    ):
        # SUBTRACTED: load_general_plugins() / 初始化日志（vllm/v1/engine/core.py:L102-L113）
        #             —— 插件加载与启动日志不改变 step 数据流。
        self.vllm_config = vllm_config
        self.log_stats = log_stats

        # SUBTRACTED: self.model_executor = executor_class(vllm_config) +
        #             register_failure_callback +  _initialize_kv_caches +
        #             StructuredOutputManager 装配（vllm/v1/engine/core.py:L117-L161）——
        #             executor/scheduler/structured_output_manager 的构造属其它章节，
        #             本章直接注入真实协作者对象。
        self.model_executor = model_executor
        self.scheduler = scheduler

        # SUBTRACTED: VLLM_ELASTIC_EP_SCALE_UP_LAUNCH / _eep_scale_up_before_kv_init
        #             （vllm/v1/engine/core.py:L124-L125）—— 弹性专家并行特性。
        # SUBTRACTED: KV connector 握手元数据收集（vllm/v1/engine/core.py:L163-L182）——
        #             分离式 KV 传输是独立子系统，不影响单机 step。
        # SUBTRACTED: mm_receiver_cache / request_block_hasher（L158-L211）——
        #             多模态接收缓存与 prefix-cache hash 装配不参与 step 控制流。
        self.use_spec_decode = getattr(vllm_config, "use_spec_decode", False)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        # SOURCE: vllm/v1/engine/core.py:L184-L194
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future, Any, Future]] | None
        ) = None
        if self.batch_queue_size > 1:
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        # SUBTRACTED: is_ec_consumer / is_pooling_model（vllm/v1/engine/core.py:L196-L200）
        #             —— encoder-cache 跨实例传输与 pooling 模型是边缘特性；
        #             step_with_batch_queue 中保留对它们的引用并给出单引擎默认值。
        self.is_ec_consumer = True
        self.is_pooling_model = False

        # SOURCE: vllm/v1/engine/core.py:L213-L215
        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
        # SOURCE: vllm/v1/engine/core.py:L216
        self.async_scheduling = getattr(vllm_config, "async_scheduling", False)

        # SOURCE: vllm/v1/engine/core.py:L218
        self.aborts_queue: queue.Queue = queue.Queue()

        # SOURCE: vllm/v1/engine/core.py:L220
        self._idle_state_callbacks: list = []

        # SUBTRACTED: freeze_gc_heap / maybe_attach_gc_debug_callback /
        #             enable_envs_cache（vllm/v1/engine/core.py:L222-L229）—— GC/env 调优旁路。

    # SOURCE: vllm/v1/engine/core.py:L312-L313
    def get_supported_tasks(self) -> tuple:
        return self.model_executor.supported_tasks

    # SOURCE: vllm/v1/engine/core.py:L315-L346
    def add_request(self, request: Any, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        # SUBTRACTED: pooling_params 任务校验 + kv_transfer_params 告警
        #             （vllm/v1/engine/core.py:L327-L344）—— pooling/KV-transfer 校验属边缘特性，
        #             不改变"校验后转交 scheduler.add_request"这一主线。
        self.scheduler.add_request(request)

    # SOURCE: vllm/v1/engine/core.py:L352-L358
    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    # SUBTRACTED: log_error_detail / log_iteration_details 两个上下文管理器
    #             （vllm/v1/engine/core.py:L360-L404）—— 故障转储 + 可选迭代日志，
    #             不改变数据流。step() 中用一个透传的 nullcontext 占位（见下）。
    from contextlib import nullcontext as _nullcontext  # noqa: E402

    def log_error_detail(self, scheduler_output: Any):
        # SOURCE: vllm/v1/engine/core.py:L360-L374 (SUBTRACTED body: dump_engine_exception)
        return self._nullcontext()

    def log_iteration_details(self, scheduler_output: Any):
        # SOURCE: vllm/v1/engine/core.py:L376-L404 (SUBTRACTED body: compute_iteration_details)
        return self._nullcontext()

    # SOURCE: vllm/v1/engine/core.py:L406-L435
    def step(self) -> tuple[dict[int, "EngineCoreOutputs"], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    # SOURCE: vllm/v1/engine/core.py:L437-L445
    def post_step(self, model_executed: bool) -> None:
        # When using async scheduling we can't get draft token ids in advance,
        # so we update draft token ids in the worker process and don't
        # need to update draft token ids here.
        if not self.async_scheduling and self.use_spec_decode and model_executed:
            # Take the draft token ids.
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)

    # SOURCE: vllm/v1/engine/core.py:L447-L563
    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, "EngineCoreOutputs"] | None, bool]:
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
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
            if self.is_ec_consumer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if self.is_pooling_model or not model_executed:
                # No sampling required (no requests scheduled).
                future = exec_future
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
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
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

    # SOURCE: vllm/v1/engine/core.py:L565-L573
    def _process_aborts_queue(self):
        if not self.aborts_queue.empty():
            request_ids = []
            while not self.aborts_queue.empty():
                ids = self.aborts_queue.get_nowait()
                # Should be a list here, but also handle string just in case.
                request_ids.extend((ids,) if isinstance(ids, str) else ids)
            # More efficient to abort all as a single batch.
            self.abort_requests(request_ids)

    # SOURCE: vllm/v1/engine/core.py:L575-L586
    def shutdown(self):
        # SUBTRACTED: structured_output_manager.clear_backend() / gc.unfreeze()
        #             （vllm/v1/engine/core.py:L576, L582）—— 与上面被减的装配对应。
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

    # SUBTRACTED: profile / reset_mm_cache / reset_encoder_cache / add_lora /
    #             remove_lora / list_loras / pin_lora / save_sharded_state /
    #             collective_rpc / execute_dummy_batch
    #             （vllm/v1/engine/core.py:L588-L604, L609-L627, L731-L763）——
    #             这些是经 UTILITY 路径转发给 executor 的管理 API，与 step/忙循环/sleep
    #             主线无关。保留 reset_prefix_cache 因 sleep 依赖 _reset_caches。

    # SOURCE: vllm/v1/engine/core.py:L606-L611
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    # SOURCE: vllm/v1/engine/core.py:L633-L636
    def _reset_caches(self, reset_running_requests=True) -> None:
        self.reset_prefix_cache(reset_running_requests=reset_running_requests)
        # SUBTRACTED: reset_mm_cache() / reset_encoder_cache()
        #             （vllm/v1/engine/core.py:L635-L636）—— 多模态/编码器缓存清理，
        #             与 sleep 主线（卸权重弃 KV）无关。

    # SOURCE: vllm/v1/engine/core.py:L638-L667
    def pause_scheduler(
        self, mode: "PauseMode" = "abort", clear_cache: bool = True
    ) -> Future | None:
        """Pause generation; behavior depends on mode.

        - ``abort``: PAUSED_NEW + abort all + (optional) clear caches.
        - ``wait``: PAUSED_NEW, keep stepping until drained (not allowed in-proc).
        - ``keep``: PAUSED_ALL; queue completes when output queue empty.
        """
        if mode not in ("keep", "abort", "wait"):
            raise ValueError(f"Invalid pause mode: {mode}")
        if mode == "wait":
            raise ValueError("'wait' mode can't be used in inproc-engine mode")

        if mode == "abort":
            self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        self.scheduler.set_pause_state(pause_state)
        if clear_cache:
            self._reset_caches()

        return None

    # SOURCE: vllm/v1/engine/core.py:L669-L671
    def resume_scheduler(self) -> None:
        """Resume the scheduler and flush any requests queued while paused."""
        self.scheduler.set_pause_state(PauseState.UNPAUSED)

    # SOURCE: vllm/v1/engine/core.py:L673-L675
    def is_scheduler_paused(self) -> bool:
        """Return whether the scheduler is in any pause state."""
        return self.scheduler.pause_state != PauseState.UNPAUSED

    # SOURCE: vllm/v1/engine/core.py:L677-L713
    def sleep(self, level: int = 1, mode: "PauseMode" = "abort") -> None | Future:
        """Put the engine to sleep at the specified level.

        - Level 0: Pause scheduling only. No GPU memory changes.
        - Level 1: Offload model weights to CPU, discard KV cache.
        - Level 2: Discard all GPU memory.
        """

        # Pause scheduler before sleeping.
        clear_prefix_cache = level >= 1
        pause_future = self.pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)
        if level < 1:
            return pause_future

        # Level 1+: Delegate to executor for GPU memory management
        model_executor = self.model_executor
        if pause_future is None:
            model_executor.sleep(level)
            return None

        future: Future = Future()

        def pause_complete(f: Future):  # SOURCE: vllm/v1/engine/core.py:L704-L709
            try:
                f.result()  # propagate any exception
                future.set_result(model_executor.sleep(level))
            except Exception as e:
                future.set_exception(e)

        pause_future.add_done_callback(pause_complete)
        return future

    # SOURCE: vllm/v1/engine/core.py:L715-L729
    def wake_up(self, tags: list[str] | None = None):
        """Wake up the engine from sleep."""
        if tags is not None and "scheduling" in tags:
            # Remove "scheduling" from tags if there are other tags to process.
            tags = [t for t in tags if t != "scheduling"]

        if tags is None or tags:
            self.model_executor.wake_up(tags)

        # Resume scheduling (applies to all levels)
        self.resume_scheduler()

    # SOURCE: vllm/v1/engine/core.py:L731-L733
    def is_sleeping(self) -> bool:
        """Check if engine is sleeping at any level."""
        return self.is_scheduler_paused() or self.model_executor.is_sleeping

    # SOURCE: vllm/v1/engine/core.py:L769-L791
    def preprocess_add_request(self, request: Any) -> tuple[Any, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # SUBTRACTED: mm_receiver_cache.get_and_update_features /
        #             Request.from_engine_core_request / structured_output_manager.grammar_init
        #             （vllm/v1/engine/core.py:L778-L790）—— 多模态特征接收、
        #             EngineCoreRequest→Request 转换、grammar 编译初始化属其它章节；
        #             这里保留方法骨架与"返回 (req, current_wave)"契约。
        req = self.model_executor.from_engine_core_request(request)
        return req, getattr(request, "current_wave", 0)

    # SUBTRACTED: _eep_scale_up_before_kv_init / _eep_send_engine_core_notification
    #             （vllm/v1/engine/core.py:L793-L801）—— 弹性专家并行（EEP）通知，删除安全。


# SOURCE: vllm/v1/engine/core.py:L804-L807
class EngineShutdownState(IntEnum):
    RUNNING = 0
    REQUESTED = 1
    SHUTTING_DOWN = 2


class EngineCoreProc(EngineCore):
    # SOURCE: vllm/v1/engine/core.py:L810-L814
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"

    # SOURCE: vllm/v1/engine/core.py:L816-L923
    def __init__(
        self,
        vllm_config: Any,
        model_executor: Any,
        scheduler: Any,
        log_stats: bool = False,
        *,
        engine_index: int = 0,
    ):
        # SOURCE: vllm/v1/engine/core.py:L829-L838
        self.input_queue: queue.Queue = queue.Queue()
        self.output_queue: queue.Queue = queue.Queue()
        # executor_fail_callback 投 EXECUTOR_FAILED 哨兵入 input_queue。
        self._executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )
        self.engine_index = engine_index
        self.engines_running = False
        self.shutdown_state = EngineShutdownState.RUNNING

        # SUBTRACTED: TensorIpcReceiver（tensor_queue）装配
        #             （vllm/v1/engine/core.py:L840-L844）—— 多模态张量零拷贝跨进程共享。
        # SUBTRACTED: _perform_handshakes / DP coordinator 地址 / publish_dp_lb_stats /
        #             VLLM_ELASTIC_EP_SCALE_UP_LAUNCH 通知 / _init_data_parallel
        #             （vllm/v1/engine/core.py:L846-L886）—— 启动握手与 DP/EEP 编排；
        #             精简版直接进入 super().__init__ 并固定 process_input_queue_block=True。
        self.process_input_queue_block = True

        super().__init__(vllm_config, model_executor, scheduler, log_stats)

        # Background Threads and Queues for IO. These enable us to
        # overlap ZMQ socket IO with GPU since they release the GIL,
        # and to overlap some serialization/deserialization with the
        # model forward pass.
        # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
        # SOURCE: vllm/v1/engine/core.py:L888-L923
        # SUBTRACTED: 真正 spawn process_input_sockets / process_output_sockets 守护线程
        #             的代码（建 ZMQ socket、传 addresses/coordinator 地址、ready_event 等
        #             待 DP coordinator）—— 精简版把这两个 IO 线程保留为方法（见下），但不
        #             在 __init__ 里启动，便于单进程测试直接驱动忙循环。线程启动逻辑见
        #             run_engine_core 注释。

    # SOURCE: vllm/v1/engine/core.py:L1067-L1151
    @staticmethod
    def run_engine_core(engine_core: "EngineCoreProc"):
        """Launch EngineCore busy loop in background process."""
        # SUBTRACTED: maybe_register_config_serialize_by_value / set_process_title /
        #             maybe_init_worker_tracer / decorate_logs / numa 绑定 / DP rank 装配 /
        #             DPEngineCoreProc 分支（vllm/v1/engine/core.py:L1071-L1114）——
        #             进程命名、tracing、NUMA、DP actor 选择均为部署编排旁路。
        import signal

        try:
            # SOURCE: vllm/v1/engine/core.py:L1118-L1131
            def wakeup_engine():  # SOURCE: vllm/v1/engine/core.py:L1118-L1122
                # Wakes up idle engine via input_queue when shutdown is requested
                # Not safe in a signal handler - we may interrupt the main thread
                # while it is holding the non-reentrant input_queue.mutex
                engine_core.input_queue.put_nowait(
                    (EngineCoreRequestType.WAKEUP, None)
                )

            def signal_handler(signum, frame):  # SOURCE: vllm/v1/engine/core.py:L1126-L1128
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                wakeup_engine()

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()

        except SystemExit:
            raise
        finally:
            # SOURCE: vllm/v1/engine/core.py:L1145-L1151
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if engine_core is not None:
                engine_core.shutdown()

    # SUBTRACTED: _init_data_parallel (no-op in base, DP override)
    #             （vllm/v1/engine/core.py:L1153-L1154）

    # SOURCE: vllm/v1/engine/core.py:L1156-L1162
    def has_work(self) -> bool:
        """Returns true if the engine should be stepped."""
        return (
            self.engines_running
            or self.scheduler.has_requests()
            or bool(self.batch_queue)
        )

    # SOURCE: vllm/v1/engine/core.py:L1164-L1166
    def is_running(self) -> bool:
        """Returns true if shutdown has not been requested."""
        return self.shutdown_state == EngineShutdownState.RUNNING

    # SOURCE: vllm/v1/engine/core.py:L1168-L1176
    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

        raise SystemExit

    # SOURCE: vllm/v1/engine/core.py:L1178-L1207
    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        while not self.has_work() and self.is_running():
            # Notify callbacks waiting for engine to become idle.
            self._notify_idle_state_callbacks()
            if self.input_queue.empty():
                # Drain aborts queue; all aborts are also processed via input_queue.
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                # SUBTRACTED: logger.isEnabledFor(DEBUG) "waiting for work" 日志分支
                #             （vllm/v1/engine/core.py:L1189-L1191）—— 仅影响日志。
            block = self.process_input_queue_block
            try:
                req = self.input_queue.get(block=block)
                self._handle_client_request(*req)
            except queue.Empty:
                break
            if not block:
                break

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    # SOURCE: vllm/v1/engine/core.py:L1209-L1227
    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # Post-step hook.
        self.post_step(model_executed)

        # If no model execution happened but there are waiting requests
        # (e.g., WAITING_FOR_REMOTE_KVS), yield the GIL briefly to allow
        # background threads (like NIXL handshake) to make progress.
        # Without this, the tight polling loop can starve background threads.
        if not model_executed and self.scheduler.has_unfinished_requests():
            time.sleep(0.001)

        return model_executed

    # SOURCE: vllm/v1/engine/core.py:L1229-L1232
    def _notify_idle_state_callbacks(self) -> None:
        while self._idle_state_callbacks:
            callback = self._idle_state_callbacks.pop()
            callback(self)

    # SOURCE: vllm/v1/engine/core.py:L1234-L1268
    def _handle_shutdown(self) -> bool:
        # Check if shutdown was requested and handle it
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return True

        if self.shutdown_state == EngineShutdownState.REQUESTED:
            shutdown_timeout = self.vllm_config.shutdown_timeout

            if shutdown_timeout == 0:
                aborted_reqs = self.scheduler.finish_requests(
                    None, RequestStatus.FINISHED_ABORTED
                )
                self._send_abort_outputs(aborted_reqs)
            # else: drain in-flight requests until timeout (logging only here).

            self.shutdown_state = EngineShutdownState.SHUTTING_DOWN

        # Exit when no work remaining
        if not self.has_work():
            return False

        return True

    # SOURCE: vllm/v1/engine/core.py:L1270-L1303
    def _handle_client_request(
        self, request_type: "EngineCoreRequestType", request: Any
    ) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.WAKEUP:
            return
        elif request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            if self._reject_add_in_shutdown(req):
                return
            self.add_request(req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            if self._reject_utility_in_shutdown(client_idx, call_id, method_name):
                return
            output = UtilityOutput(call_id)
            # Lazily look-up utility method so that failure will be handled/returned.
            get_result = lambda: (
                (method := getattr(self, method_name))
                and method(*args)
            )
            # SUBTRACTED: _convert_msgspec_args(method, args) —— 把 msgspec 参数按目标
            #             方法签名转换（vllm/v1/engine/core.py:L1292, L1341-L1356）；
            #             精简版直接透传 args。
            enqueue_output = lambda out: self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=out))
            )
            self._invoke_utility_method(method_name, get_result, output, enqueue_output)
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            # SUBTRACTED: logger.error 未知类型（vllm/v1/engine/core.py:L1301-L1303）
            pass

    # SOURCE: vllm/v1/engine/core.py:L1305-L1311
    def _reject_add_in_shutdown(self, request: Any) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False
        self._send_abort_outputs_to_client(
            [request.request_id], getattr(request, "client_index", 0)
        )
        return True

    # SOURCE: vllm/v1/engine/core.py:L1313-L1324
    def _reject_utility_in_shutdown(
        self, client_idx: int, call_id: int, method_name: str
    ) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False
        output = UtilityOutput(call_id, failure_message="Server shutting down")
        self.output_queue.put_nowait(
            (client_idx, EngineCoreOutputs(utility_output=output))
        )
        return True

    # SOURCE: vllm/v1/engine/core.py:L1326-L1343
    @staticmethod
    def _invoke_utility_method(
        name: str, get_result, output: "UtilityOutput", enqueue_output
    ):
        # SOURCE: vllm/v1/engine/core.py:L1326-L1343
        try:
            result = get_result()
            if isinstance(result, Future):
                # Defer utility output handling until future completion.
                callback = lambda future: EngineCoreProc._invoke_utility_method(
                    name, future.result, output, enqueue_output
                )
                result.add_done_callback(callback)
                return
            output.result = UtilityResult(result)
        except Exception as e:
            output.failure_message = f"Call to {name} method failed: {str(e)}"
        enqueue_output(output)

    # SOURCE: vllm/v1/engine/core.py:L1376-L1468
    def process_input_sockets(self):
        """Input socket IO thread."""
        # SUBTRACTED: 整个方法体（建 DEALER/XSUB socket、发 EngineCoreReadyResponse、
        #             poller.poll()、msgpack decode、preprocess_add_request、把
        #             (type, payload) 塞 input_queue；ABORT 双投 input_queue + aborts_queue）
        #             （vllm/v1/engine/core.py:L1383-L1468）—— ZMQ/handshake/DP coordinator
        #             细节属部署编排。精简版的"请求如何进 input_queue"由 simulate_recv 演示：
        raise NotImplementedError(
            "ZMQ input socket thread subtracted; see simulate_recv()."
        )

    def simulate_recv(self, request_type, request):
        # SOURCE: vllm/v1/engine/core.py:L1450-L1468 (核心控制流，不含 ZMQ/解码)
        # 复现 process_input_sockets 末段：ADD 先 preprocess，ABORT 双投，再统一入 input_queue。
        if request_type == EngineCoreRequestType.ADD:
            request = self.preprocess_add_request(request)
        elif request_type == EngineCoreRequestType.ABORT:
            # Aborts are added to *both* queues, allows us to eagerly
            # process aborts while also ensuring ordering in the input
            # queue to avoid leaking requests. This is ok because
            # aborting in the scheduler is idempotent.
            self.aborts_queue.put_nowait(request)
        # Push to input queue for core busy loop.
        self.input_queue.put_nowait((request_type, request))

    # SOURCE: vllm/v1/engine/core.py:L1470-L1535
    def process_output_sockets(self):
        """Output socket IO thread."""
        # SUBTRACTED: 整个方法体（建 PUSH socket、msgpack encode_into 零拷贝、
        #             send_multipart(track=True) 缓冲复用、ENGINE_CORE_DEAD 发送）
        #             （vllm/v1/engine/core.py:L1475-L1535）—— ZMQ 编码/发送细节属部署编排。
        #             精简版的"输出如何出 output_queue"由 simulate_send 演示：
        raise NotImplementedError(
            "ZMQ output socket thread subtracted; see simulate_send()."
        )

    def simulate_send(self):
        # SOURCE: vllm/v1/engine/core.py:L1504-L1511 (核心控制流，不含 ZMQ/编码)
        # 复现 process_output_sockets 主循环取一项的语义：阻塞取 output_queue 一项并返回。
        output = self.output_queue.get()
        if output == EngineCoreProc.ENGINE_CORE_DEAD:
            return output
        client_index, outputs = output
        outputs.engine_index = self.engine_index
        return client_index, outputs

    # SOURCE: vllm/v1/engine/core.py:L1546-L1586
    def pause_scheduler(
        self, mode: "PauseMode" = "abort", clear_cache: bool = True
    ) -> Future | None:
        """Pause generation; behavior depends on mode (multi-proc async version).

        Unlike the in-proc base version (which returns None synchronously),
        this returns a Future that completes once in-flight work has drained,
        via an idle-state callback fired from _process_input_queue.
        """
        if mode not in ("keep", "abort", "wait"):
            raise ValueError(f"Invalid pause mode: {mode}")

        def engine_idle_callback(engine: "EngineCoreProc", future: Future) -> None:
            # SOURCE: vllm/v1/engine/core.py:L1565-L1568
            if clear_cache:
                engine._reset_caches()
            future.set_result(None)

        if mode == "abort":
            aborted_reqs = self.scheduler.finish_requests(
                None, RequestStatus.FINISHED_ABORTED
            )
            self._send_abort_outputs(aborted_reqs)

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        self.scheduler.set_pause_state(pause_state)

        if self._pause_complete():
            if clear_cache:
                self._reset_caches()
            return None

        future: Future = Future()
        self._idle_state_callbacks.append(
            partial(engine_idle_callback, future=future)
        )
        return future

    # SOURCE: vllm/v1/engine/core.py:L1588-L1593
    def _pause_complete(self) -> bool:
        """Returns True if the pause has fully completed and the caller can
        return ``None`` synchronously; False if the pause is still pending
        and the caller should register an idle-state callback to finish it.
        """
        return not self.has_work()

    # SOURCE: vllm/v1/engine/core.py:L1595-L1603
    def _send_finish_outputs_to_client(
        self, req_ids: list[str], client_index: int, finish_reason: "FinishReason"
    ) -> None:
        outputs = [
            EngineCoreOutput(req_id, [], finish_reason=finish_reason)
            for req_id in req_ids
        ]
        eco = EngineCoreOutputs(finished_requests=req_ids, outputs=outputs)
        self.output_queue.put_nowait((client_index, eco))

    # SOURCE: vllm/v1/engine/core.py:L1605-L1608
    def _send_abort_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ABORT)

    # SOURCE: vllm/v1/engine/core.py:L1615-L1623
    def _send_abort_outputs(self, aborted_reqs: list[tuple[str, int]]) -> None:
        # TODO(nick) this will be moved inside the scheduler
        if aborted_reqs:
            # Map client_index to list of request_ids that belong to that client.
            by_client: dict[int, set[str]] = defaultdict(set)
            for req_id, client_index in aborted_reqs:
                by_client[client_index].add(req_id)
            for client_index, req_ids in by_client.items():
                self._send_abort_outputs_to_client(list(req_ids), client_index)


# SUBTRACTED: class DPEngineCoreProc / DPMoEEngineCoreActor / EngineCoreActor /
#             EngineCoreActorMixin（vllm/v1/engine/core.py:L1626+）—— 数据并行与 Ray
#             actor 是多引擎部署形态；本章讲单 EngineCore，DP 仅在叙事中高层提及。
