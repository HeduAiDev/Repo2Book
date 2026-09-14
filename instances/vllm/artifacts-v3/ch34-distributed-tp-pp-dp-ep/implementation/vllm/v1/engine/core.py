# SOURCE: vllm/v1/engine/core.py
# ch34 切面（m13/m17/m19/m20）：run_engine_core 的 DP 出生分叉 + DPEngineCoreProc
# 全景（_init_data_parallel 的 stateless dp_group / 两阶段暂停 / busy loop 的
# dummy 锁步与 32 步共识 / -1 哨兵发布）。
# 父类 EngineCoreProc 按 ch09 域收窄为协作件装配面（行内逐处标注）；删除（dossier
# 删除项 6）：DPEngineCoreProc 的 elastic EP 面（eep_scaling_state 推进 L2112-L2122、
# reinitialize_distributed L2190 起）、_should_throttle_prefills（L2092-L2099）、
# iteration_details 统计分支（L2138-L2142）、非 MoE 的 _maybe_publish_request_counts
# （L1391-L1402，二选一保留 DPE 版）。

from __future__ import annotations

import contextlib
import queue
import signal
from enum import IntEnum
from typing import Any

from vllm.config import ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine import (
    EngineCoreOutputs,
    EngineCoreRequestType,
    SchedulerStats,
)

# SUBTRACTED: fault_tolerance/elastic_ep/tracing/tensor-IPC/handshake 族 import
#   （L20-L100）——各归其域。

logger = init_logger(__name__)


# SOURCE: vllm/v1/engine/utils.py:L253-L277 SignalCallback — 逐字（信号上下文
#   安全回调；run_engine_core 的关停唤醒通道）
class SignalCallback:
    """Safely trigger a callback from signal handler context via a dedicated thread."""
    # SOURCE: vllm/v1/engine/utils.py:L253-L277 SignalCallback（锚点双置）

    def __init__(self, callback):
        # SOURCE: vllm/v1/engine/utils.py:L256-L265 __init__（锚点双置）
        import threading

        self._callback = callback
        self._event = threading.Event()
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="signal-callback",
        )
        self._thread.start()

    def _run(self):
        # SOURCE: vllm/v1/engine/utils.py:L267-L270 _run（锚点双置）
        self._event.wait()
        if not self._stopped:
            self._callback()

    def trigger(self):
        # SOURCE: vllm/v1/engine/utils.py:L272-L273 trigger（锚点双置）
        self._event.set()

    def stop(self):
        # SOURCE: vllm/v1/engine/utils.py:L275-L277 stop（锚点双置）
        self._stopped = True
        self._event.set()


# SOURCE: vllm/v1/engine/core.py:L1002-L1005 EngineShutdownState — 逐字
class EngineShutdownState(IntEnum):
    RUNNING = 0
    REQUESTED = 1
    SHUTTING_DOWN = 2


# SOURCE: vllm/v1/engine/core.py:L1008-L1915 EngineCoreProc 类头 —— 逐字 minus
#   基类 EngineCore（L103-L1000，ch09 域的调度/统计/生命周期主体）：本章把
#   DPE 下游触碰的 EngineCore 方法按原行号并入本载体（add_request L439 /
#   capture_iteration_details L510 / shutdown L751 / resume_scheduler L860 /
#   execute_dummy_batch L928），IO 双线程/握手族按 ch05 域裁除。
class EngineCoreProc:
    # SOURCE: vllm/v1/engine/core.py:L1008-L1010 EngineCoreProc 类头（锚点双置）
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"

    # SOURCE: vllm/v1/engine/core.py:L1015-L1127 __init__ —— 协作件装配子集
    #   （ch05 握手/IO 线程与 ch09 EngineCore 装配按域收窄；本档保留下游 DPE 代码
    #   触碰的全部属性，装配顺序对齐真实源码）
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type | None,
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue=None,
        *,
        engine_index: int = 0,
    ):
        # SOURCE: vllm/v1/engine/core.py:L1015-L1127 __init__（锚点双置）
        self.input_queue: queue.Queue = queue.Queue()
        self.output_queue: queue.Queue = queue.Queue()

        self.engine_index = engine_index
        self.engines_running = False
        self.shutdown_state = EngineShutdownState.RUNNING
        self.log_stats = log_stats
        self.vllm_config = vllm_config
        # SUBTRACTED: tensor IPC receiver / _perform_handshakes 两层握手 /
        #   input/output 双 IO 线程（L1038-L1104）——ch05 域；has_coordinator 与
        #   publish_dp_lb_stats 的判定语义保留：
        self.has_coordinator = client_handshake_address is not None
        internal_dp_balancing = (
            self.has_coordinator
            and not vllm_config.parallel_config.data_parallel_external_lb
        )
        # Only publish request queue stats to coordinator for "internal"
        # and "hybrid" LB modes.
        self.publish_dp_lb_stats = internal_dp_balancing
        self.last_counts = (0, 0)

        self.process_input_queue_block = True
        self._init_data_parallel(vllm_config)
        # SUBTRACTED: EngineCore.__init__ 的 scheduler/executor 装配（ch09/ch10
        #   域）——协作件由宿主/测试注入（scheduler / model_executor）。

    # SOURCE: vllm/v1/engine/core.py:L1362-L1363 _init_data_parallel — 逐字
    #   （非 DP 基类是 no-op；DPE 版覆写为 stateless dp_group 构造）
    def _init_data_parallel(self, vllm_config: VllmConfig):
        # SOURCE: vllm/v1/engine/core.py:L1362-L1363 _init_data_parallel（锚点双置）
        pass

    # SOURCE: vllm/v1/engine/core.py:L1365-L1371 has_work — 逐字
    def has_work(self) -> bool:
        """Returns true if the engine should be stepped."""
        return (
            self.engines_running
            or self.scheduler.has_requests()
            or bool(self.batch_queue)
        )

    # SOURCE: vllm/v1/engine/core.py:L1373-L1375 is_running — 逐字
    def is_running(self) -> bool:
        """Returns true if shutdown has not been requested."""
        return self.shutdown_state == EngineShutdownState.RUNNING

    # SOURCE: vllm/v1/engine/core.py:L1378-L1389 run_busy_loop（非 DP 版）—— 逐字
    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # Publish request counts before and after GPU step to ensure freshness.
            self._maybe_publish_request_counts()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()
            self._maybe_publish_request_counts()

        raise SystemExit

    # SUBTRACTED: 非 MoE 引擎的 _maybe_publish_request_counts（L1391-L1402）——
    #   删除项 6：二选一保留 DPEngineCoreProc 版（盖 step/wave 章）。
    # SUBTRACTED: _process_input_queue（L1404-L1433）——ch09 域的输入队列轮询体
    #   （aborts 队列排空/idle 回调/block 语义）；本章测试以注入承载，两个
    #   run_busy_loop 的调用位 L1382/L2113 逐字保留。

    # SOURCE: vllm/v1/engine/core.py:L1435-L1452 _process_engine_step —— 语义子集
    #   （ch09 的 step 五拍按域收窄；返回『本拍是否真执行了模型』的契约保留）
    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""
        # SOURCE: vllm/v1/engine/core.py:L1435-L1452 _process_engine_step（锚点双置）

        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # SUBTRACTED: post_step 钩子与 1ms GIL 让渡（L1443-L1450）——ch09 域。

        return model_executed

    # SOURCE: vllm/v1/engine/core.py:L1459+ _handle_shutdown —— 头部语义子集
    #   （RUNNING 即活；REQUESTED 的 abort/drain 双模式归 ch09/ch07 域）
    def _handle_shutdown(self) -> bool:
        # Check if shutdown was requested and handle it
        # SOURCE: vllm/v1/engine/core.py:L1459-L1505 _handle_shutdown（锚点双置）
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return True
        return False

    # SOURCE: vllm/v1/engine/core.py:L510-L555 capture_iteration_details —— 头部
    #   语义子集（观测细节 L527-L558 归 ch08 域；log_stats 关闭恒 yield None 的
    #   快速路径保留）
    @contextlib.contextmanager
    def capture_iteration_details(self, scheduler_output):
        # SOURCE: vllm/v1/engine/core.py:L509 装饰器 + L510-L555 本体（锚点双置）
        if not self.log_stats:
            yield None
            return
        # SUBTRACTED: 观测记录体（L513-L514 的 observability 判定与 L527 起
        #   的记录段）——ch08 域。
        yield None

    # SOURCE: vllm/v1/engine/core.py:L928-L929 execute_dummy_batch —— 逐字（锚点双置）
    def execute_dummy_batch(self):
        self.model_executor.execute_dummy_batch()

    # SOURCE: vllm/v1/engine/core.py:L439-L483 add_request —— 校验族（request_id
    #   类型/pooling 任务/KV·EC connector 检查，L444-L478）按 ch04/ch16 域裁除，
    #   调度器入队位 L479 保留；DPE 版在其上叠 stale-wave 上报。
    def add_request(self, request, request_wave: int = 0):
        # SOURCE: vllm/v1/engine/core.py:L439-L483 add_request（锚点双置）
        self.scheduler.add_request(request)

    # SOURCE: vllm/v1/engine/core.py:L860-L862 resume_scheduler —— ch11 域收窄（调度恢复）
    def resume_scheduler(self):
        return None

    # SOURCE: vllm/v1/engine/core.py:L751-L767 shutdown —— ch09 域收窄（executor
    #   关停面 L753-L767）；标记 RUNNING→SHUTTING_DOWN 的状态面 L752 保留。
    def shutdown(self):
        # SOURCE: vllm/v1/engine/core.py:L751-L767 shutdown（锚点双置）
        self.shutdown_state = EngineShutdownState.SHUTTING_DOWN

    # SOURCE: vllm/v1/engine/core.py:L1507-L1540 _handle_client_request —— ch09
    #   域收窄（ADD/ABORT/UTILITY 的分派体 L1518-L1540）；DPE 版在其前拦
    #   START_DP_WAVE。
    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        # SUBTRACTED: ADD/ABORT/UTILITY 分派体——ch09 域。
        # SOURCE: vllm/v1/engine/core.py:L1507-L1540 _handle_client_request（锚点双置）
        pass

    # SUBTRACTED: process_input_sockets / process_output_sockets 双 IO 线程
    #   （L1569-L1810）——ch05 域（本章正文按锚引用其 client_index 分桶段，
    #   由 narrative 内嵌真源码承担；伴读不携带）。SUBTRACTED 同理：
    #   _send_finish_outputs_to_client 族与 pause/resume 的 future 机制（ch09）。

    # SOURCE: vllm/v1/engine/core.py:L1272-L1360 run_engine_core —— DP 出生分叉
    #   段逐字（L1295-L1318）+ 关停/信号骨架；握手与 tracer 面按域收窄
    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""
        # SOURCE: vllm/v1/engine/core.py:L1272-L1360 run_engine_core（锚点双置）

        # SUBTRACTED: maybe_register_config_serialize_by_value / tracer /
        #   numa 绑定与进程标题装饰（L1274-L1293）——ch05/环境域；标题语义保留：
        from vllm.utils.system_utils import set_process_title

        engine_core: EngineCoreProc | None = None
        signal_callback: SignalCallback | None = None
        try:
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config: ParallelConfig = vllm_config.parallel_config
            data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
            if data_parallel:
                parallel_config.data_parallel_rank_local = local_dp_rank
                process_title = f"EngineCore_DP{dp_rank}"
            else:
                process_title = "EngineCore"
            set_process_title(process_title)

            # SUBTRACTED: kv_transfer_config 的 engine_id 后缀（L1295-L1304）——
            #   ch16 域。

            parallel_config.data_parallel_index = dp_rank
            if data_parallel and vllm_config.model_config.is_moe:
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                # Non-MoE DP ranks are completely independent, so treat like DP=1.
                # Note that parallel_config.data_parallel_index will still reflect
                # the original DP rank.
                parallel_config.data_parallel_size = 1
                parallel_config.data_parallel_size_local = 1
                parallel_config.data_parallel_rank = 0
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

            assert engine_core is not None

            def wakeup_engine():
                # Wakes up idle engine via input_queue when shutdown is requested
                # Not safe in a signal handler - we may interrupt the main thread
                # while it is holding the non-reentrant input_queue.mutex
                # SOURCE: vllm/v1/engine/core.py:L1322-L1328 wakeup_engine（锚点双置）
                engine_core.input_queue.put_nowait(
                    (EngineCoreRequestType.WAKEUP, None)
                )

            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum, frame):
                # SOURCE: vllm/v1/engine/core.py:L1330-L1339 signal_handler（锚点双置）
                signal_name = signal.Signals(signum).name
                logger.info(
                    "[shutdown] EngineCore: trigger received signal=%s",
                    signal_name,
                )
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.info_once("[shutdown] EngineCore: exiting busy loop")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if engine_core is not None:
                engine_core.shutdown()

    # SOURCE: vllm/v1/engine/core.py:L1605-L1617 _send_engine_dead —— 逐字
    #   （executor_fail_callback 的哨兵入队列；IO 线程侧消费归 ch05）
    def _send_engine_dead(self):
        # SOURCE: vllm/v1/engine/core.py:L1605-L1617 _send_engine_dead（锚点双置）
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)


# SOURCE: vllm/v1/engine/core.py:L1918-L2188 DPEngineCoreProc — 逐字 minus
#   删除项 6（elastic EP 面 / _should_throttle_prefills / iteration_details
#   统计分支 / reinitialize_distributed）
class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    # SOURCE: vllm/v1/engine/core.py:L1922-L1967 __init__ — 逐字 minus
    #   eep_scaling_state（L1952-L1954，删除项 6）
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[object] | None,
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue=None,
    ):
        # SOURCE: vllm/v1/engine/core.py:L1922-L1967 __init__（DPEngineCoreProc）（锚点双置）
        assert vllm_config.model_config.is_moe, (
            "DPEngineCoreProc should only be used for MoE models"
        )

        scheduler_config = vllm_config.scheduler_config
        self.prefill_schedule_interval = scheduler_config.prefill_schedule_interval

        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.step_counter = 0
        self.current_wave = 0

        # Two-phase pause protocol state. When pending_pause is True, the
        # engine keeps stepping (dummy batches) while waiting for all DP
        # ranks to also set pending_pause. Once all ranks agree via
        # all-reduce, ignore_start_dp_wave is set so that stale
        # START_DP_WAVE messages cannot re-wake the engines.
        self.pending_pause = False
        self.ignore_start_dp_wave = False

        # SUBTRACTED: eep_scaling_state 的构造（L1952-L1954）——删除项 6。

        # Initialize the engine.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(
            vllm_config,
            local_client,
            handshake_address,
            executor_class,
            log_stats,
            client_handshake_address,
            engine_index=dp_rank,
            tensor_queue=tensor_queue,
        )

    # SOURCE: vllm/v1/engine/core.py:L1969-L1983 _init_data_parallel — 逐字
    #   （引擎级 stateless gloo dp_group 的构造点）
    def _init_data_parallel(self, vllm_config: VllmConfig):
        # Configure GPUs and stateless process group for data parallel.
        # SOURCE: vllm/v1/engine/core.py:L1969-L1983 _init_data_parallel（DPE）（锚点双置）
        parallel_config = vllm_config.parallel_config
        dp_rank = parallel_config.data_parallel_rank
        dp_size = parallel_config.data_parallel_size
        local_dp_rank = parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        self.dp_rank = dp_rank
        self.dp_size = dp_size
        dp_group, dp_store = parallel_config.stateless_init_dp_group(return_store=True)
        self.dp_group, self.dp_store = dp_group, dp_store

    # SOURCE: vllm/v1/engine/core.py:L1985-L1988 shutdown — 逐字
    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            from vllm.distributed.utils import (
                stateless_destroy_torch_distributed_process_group,
            )

            stateless_destroy_torch_distributed_process_group(dp_group)

    # SOURCE: vllm/v1/engine/core.py:L1990-L2006 _pause_complete — 逐字（两阶段
    #   暂停的 docstring 原文）
    def _pause_complete(self) -> bool:
        """Two-phase DP-aware pause.

        Phase 1: Set local pause state and ``pending_pause`` flag. If the
        engines are idle, kick-start them by setting ``engines_running`` to
        True so ranks enter the stepping loop and reach the all-reduce
        consensus checkpoint in ``_has_global_unfinished_reqs``.

        Phase 2 (in ``_has_global_unfinished_reqs``): Once the all-reduce
        confirms that **all** ranks have ``pending_pause`` set, collectively
        stop stepping and set ``ignore_start_dp_wave`` so that stale
        ``START_DP_WAVE`` messages cannot re-wake any engine.
        """
        # SOURCE: vllm/v1/engine/core.py:L1990-L2006 _pause_complete（锚点双置）
        self.pending_pause = True
        self.engines_running = True

        return False

    # SOURCE: vllm/v1/engine/core.py:L2008-L2022 add_request — 逐字（stale wave
    #   上报即 m20 的引擎侧半边）
    def add_request(self, request, request_wave: int = 0):
        # SOURCE: vllm/v1/engine/core.py:L2008-L2022 add_request（DPE）（锚点双置）
        super().add_request(request, request_wave)
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif (
                not self.engines_running
                and self.scheduler.pause_state == PauseState.UNPAUSED
            ):
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.engines_running = True
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )

    # SOURCE: vllm/v1/engine/core.py:L2024-L2047 resume_scheduler — 逐字
    def resume_scheduler(self):
        if self.pending_pause or (self.engines_running and self.ignore_start_dp_wave):
            raise RuntimeError(
                "resume_scheduler called while pause is still in "
                "flight. Wait for the pause future to resolve before "
                "resuming."
            )
        if self.engines_running:
            logger.debug("Resume called while engines are not paused, ignoring.")
            return

        super().resume_scheduler()
        self.ignore_start_dp_wave = False

        # Barrier: wait for all DP ranks to have resumed (and cleared
        # ignore_start_dp_wave) before any rank starts stepping. Uses
        # the existing all-reduce which is safe because engines are
        # stopped.
        has_global_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, self.scheduler.has_unfinished_requests()
        )

        if has_global_unfinished:
            self.engines_running = True

    # SOURCE: vllm/v1/engine/core.py:L2049-L2053 barrier — 逐字
    def barrier(self):
        """Blocking barrier on the DP process group (test-only utility)."""
        import torch.distributed as dist

        dist.barrier(group=self.dp_group)

    # SOURCE: vllm/v1/engine/core.py:L2055-L2073 _handle_client_request — 逐字
    #   （START_DP_WAVE 的 gate：ignore / exclude / 旧 wave 三判）
    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        # SOURCE: vllm/v1/engine/core.py:L2055-L2073 _handle_client_request（DPE）（锚点双置）
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            if self.ignore_start_dp_wave:
                return
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                new_wave >= self.current_wave
            ):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug(
                        "EngineCore starting idle loop for wave %d.",
                        new_wave,
                    )
                    self.engines_running = True
        else:
            super()._handle_client_request(request_type, request)

    # SOURCE: vllm/v1/engine/core.py:L2075-L2090 _maybe_publish_request_counts
    #   —— 逐字（counts 变化才发 + step/wave 盖章 + -1 哨兵）
    def _maybe_publish_request_counts(self):
        # SOURCE: vllm/v1/engine/core.py:L2075-L2090 _maybe_publish_request_counts（锚点双置）
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed), stamped with the
        # lockstep-synchronized step counter and wave number.
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts,
                kv_cache_usage=self.scheduler.get_kv_cache_usage(),
                step_counter=self.step_counter,
                current_wave=self.current_wave,
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))

    # SUBTRACTED: _should_throttle_prefills（L2092-L2099）——删除项 6。

    # SOURCE: vllm/v1/engine/core.py:L2102-L2169 run_busy_loop — 逐字 minus
    #   elastic EP 推进段（L2112-L2122）与 iteration_details 统计分支
    #   （L2138-L2142），均删除项 6
    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""
        # SOURCE: vllm/v1/engine/core.py:L2102-L2169 run_busy_loop（DPE）（锚点双置）

        # Loop until process is sent a SIGINT or SIGTERM
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # Publish request counts before and after GPU step to ensure freshness.
            self._maybe_publish_request_counts()

            # SUBTRACTED: eep_scaling_state 推进段（L2112-L2122）——删除项 6。

            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # Execute a dummy pass when no ready requests ran, unless the
                # engine is sleeping.
                elif not self.model_executor.is_sleeping:
                    with self.capture_iteration_details(None) as iteration_details:
                        self.execute_dummy_batch()
                    # SUBTRACTED: iteration_details 统计分支（L2138-L2142）。

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0

        raise SystemExit

    # SOURCE: vllm/v1/engine/core.py:L2171-L2188 _has_global_unfinished_reqs
    #   —— 逐字（32 步门 + 共识后置 ignore 标志）
    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # Optimization - only perform finish-sync all-reduce every 32 steps.
        # SOURCE: vllm/v1/engine/core.py:L2171-L2188 _has_global_unfinished_reqs（锚点双置）
        self.step_counter += 1
        if self.step_counter % 32 != 0:
            return True

        has_unfinished, pause_consensus = ParallelConfig.sync_dp_state(
            self.dp_group,
            has_unfinished=local_unfinished,
            pending_pause=self.pending_pause,
        )

        if pause_consensus:
            self.ignore_start_dp_wave = True
            self.pending_pause = False
            logger.debug("DP pause consensus reached, ignoring START_DP_WAVE.")

        return has_unfinished

    # SUBTRACTED: reinitialize_distributed（L2190-L2300）与 DPMoEEngineCoreActor /
    #   EngineCoreActor（L2305-L2488）——elastic EP 与 ray actor 面（删除项 6 同族）。
