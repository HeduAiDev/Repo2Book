# Subtract-only companion for v3 ch17 — vllm/v1/executor/multiproc_executor.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions, each marked `# SUBTRACTED:`.
#
# Deletions here (dossier subtraction_plan.delete):
#   #2  多节点路径：DP-leader 日志（L140-L150）、peer_worker_response_mqs 跨节点
#       装配（L215-L220）、_init_message_queues 的 else 分支（L587-L605）、
#       get_response_mqs（L249-L258）；
#   #3  kv_output_aggregator 聚合分支（collective_rpc 内 L375-L382）；
#   #4  平台/可观测性装饰：OMPProcessManager 上下文（L174-L181）、inherited_fds
#       fork 适配（L167-L172/L193-L195、L688-L690）、numa_utils（L712-L716）、
#       set_worker_net_device（L853-L854）、VLLM_ELASTIC_EP_SCALE_UP_LAUNCH
#       （L645-L648）、maybe_init_worker_tracer（L871-L877）、
#       setup_proc_title_and_log_prefix 全方法（L635-L637/L641-L644 调用位 +
#       L1024-L1058 定义）、set_multiprocessing_worker_envs 的 OMP 调优体
#       （L1068-L1089）、@instrument 装饰。

from __future__ import annotations

import multiprocessing
import os
import pickle
import queue
import signal
import threading
import time
import traceback
import weakref
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Lock as LockType
from threading import Thread
from typing import Any, cast

import cloudpickle

from .._host_seams import (
    AsyncModelRunnerOutput,
    GrammarOutput,
    SchedulerOutput,
    current_platform,
    destroy_distributed_environment,
    destroy_model_parallel,
    enable_envs_cache,
    envs,
    get_distributed_init_method,
    get_loopback_ip,
    get_open_port,
    init_logger,
)
from .._shm_broadcast_seam import Handle, MessageQueue
from ..worker.worker_base import WorkerWrapperBase
from .abstract import Executor, FailureCallback

logger = init_logger(__name__)

# HOST SEAM (win32): PipeConnection 在 Windows 上不是 Connection 的子类
# （两者同出 _ConnectionBase；unix 上 PipeConnection 才别名 Connection）——
# wait_for_ready 的 isinstance 断言放宽到两者，语义不变。
import sys as _sys  # HOST SEAM

if _sys.platform == "win32":  # HOST SEAM
    from multiprocessing.connection import Connection as _MpConnection
    from multiprocessing.connection import PipeConnection as _MpPipeConnection

    _ConnTypes = (_MpConnection, _MpPipeConnection)
else:  # HOST SEAM
    _ConnTypes = (Connection,)


# SOURCE: vllm/v1/executor/multiproc_executor.py:L70-L100 FutureWrapper
class FutureWrapper(Future):
    # SOURCE: vllm/v1/executor/multiproc_executor.py:L71-L81 __init__
    def __init__(
        self,
        futures_queue: deque["FutureWrapper"],
        get_response: Callable[[], Any],
        aggregate: Callable = lambda x: x,
    ):
        self.futures_queue = futures_queue
        self.get_response = get_response
        self.aggregate = aggregate
        super().__init__()
        self.futures_queue.appendleft(self)

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L83-L91 result — FIFO 排空
    def result(self, timeout=None):
        if timeout is not None:
            raise RuntimeError("timeout not implemented")

        # Drain any futures ahead of us in the queue.
        while not self.done():
            future = self.futures_queue.pop()
            future._wait_for_response()
        return super().result()

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L93-L100 _wait_for_response
    def _wait_for_response(self):
        try:
            response = self.aggregate(self.get_response())
            with suppress(InvalidStateError):
                self.set_result(response)
        except Exception as e:
            with suppress(InvalidStateError):
                self.set_exception(e)


# SOURCE: vllm/v1/executor/multiproc_executor.py:L103-L104 MultiprocExecutor
class MultiprocExecutor(Executor):
    # SOURCE: vllm/v1/executor/multiproc_executor.py:L104 supports_pp
    supports_pp: bool = True

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L106-L108 __init__
    def __init__(self, vllm_config, monitor_workers: bool = True):
        self.monitor_workers = monitor_workers
        super().__init__(vllm_config)

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L110-L247 _init_executor
    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.failure_callback: FailureCallback | None = None

        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )

        set_multiprocessing_worker_envs()

        # use the loopback address get_loopback_ip() for communication.
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port()
        )
        self.rpc_broadcast_mq: MessageQueue | None = None
        scheduler_output_handle: Handle | None = None
        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        if self.parallel_config.node_rank_within_dp == 0:
            # For leader node within each dp rank,
            # each dp will have its own leader multiproc executor.
            # SUBTRACTED: DP group leader 启动日志（multiproc_executor.py:
            #   L140-L150——DP 组头的多节点拓扑打印，删除项 2）。
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            # SUBTRACTED: mq_connect_ip = get_ip()（L139）——远端读者轴随删除
            #   项 2 裁除，seam 广播队列固定 loopback。
            self.rpc_broadcast_mq = MessageQueue(
                self.world_size,
                self.local_world_size,
                max_chunk_bytes=max_chunk_bytes,
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        # Create workers
        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            global_start_rank = (
                self.local_world_size * self.parallel_config.node_rank_within_dp
            )
            # SUBTRACTED: fork 模式的 inherited_fds 跟踪（L167-L172——spawn
            #   宿主上 inherited_fds 恒 None；make_worker_process 的参数与
            #   worker_main 的关闭循环保留，走空表行为，删除项 4）。
            inherited_fds: list[int] | None = None

            # SUBTRACTED: OMPProcessManager / cpu_omp_manager 上下文
            #   （L174-L181——CPU 后端的 OpenMP 线程亲和，删除项 4）。
            for local_rank in range(self.local_world_size):
                global_rank = global_start_rank + local_rank
                is_driver_worker = self._is_driver_worker(global_rank)
                unready_worker_handle = WorkerProc.make_worker_process(
                    vllm_config=self.vllm_config,
                    local_rank=local_rank,
                    rank=global_rank,
                    distributed_init_method=distributed_init_method,
                    input_shm_handle=scheduler_output_handle,
                    shared_worker_lock=shared_worker_lock,
                    is_driver_worker=is_driver_worker,
                    inherited_fds=inherited_fds,
                )
                unready_workers.append(unready_worker_handle)
                # SUBTRACTED: fork 模式下登记 death/ready pipe fd 到
                #   inherited_fds（L193-L195，删除项 4）。

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.

            # Wait for all local workers to be ready.
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # Start background thread to monitor worker health if not in headless mode.
            if self.monitor_workers:
                self.start_worker_monitor()

            self.response_mqs = []
            # Only leader node have remote response mqs
            if self.parallel_config.node_rank_within_dp == 0:
                for rank in range(self.world_size):
                    if rank < self.local_world_size:
                        local_message_queue = self.workers[rank].worker_response_mq
                        assert local_message_queue is not None
                        self.response_mqs.append(local_message_queue)
                    # SUBTRACTED: 跨节点 peer_worker_response_mqs 装配
                    #   （multiproc_executor.py:L215-L220——远端 rank 的 MQ 从
                    #   workers[0].peer_worker_response_mqs 取，删除项 2）。

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.

            # Wait for all input mqs to be ready.
            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            # Wait for all remote response mqs to be ready.
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready()

            self.futures_queue = deque[FutureWrapper]()

            self._post_init_executor()

            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                        uw.death_writer = None
                self._ensure_worker_termination([uw.proc for uw in unready_workers])

        self.output_rank = self._get_output_rank()

    # SUBTRACTED: get_response_mqs（multiproc_executor.py:L249-L258——按
    #   unique_reply_rank 选应答 MQ 的辅助查询面；collective_rpc 内联同款
    #   选择逻辑，删除项 2 随多节点装配一并裁除）。

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L260-L271 _get_parallel_sizes
    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        self.world_size = self.parallel_config.world_size
        assert self.world_size % self.parallel_config.nnodes_within_dp == 0, (
            f"global world_size ({self.parallel_config.world_size}) must be "
            f"divisible by nnodes_within_dp "
            f"({self.parallel_config.nnodes_within_dp}). "
        )
        self.local_world_size = self.parallel_config.local_world_size
        tp_size = self.parallel_config.tensor_parallel_size
        pp_size = self.parallel_config.pipeline_parallel_size
        pcp_size = self.parallel_config.prefill_context_parallel_size
        return tp_size, pp_size, pcp_size

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L273-L274 _post_init_executor
    def _post_init_executor(self) -> None:
        pass

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L276-L277 _is_driver_worker
    def _is_driver_worker(self, rank: int) -> bool:
        return rank % self.parallel_config.tensor_parallel_size == 0

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L279-L313 start_worker_monitor
    def start_worker_monitor(self, inline=False) -> None:
        workers = self.workers
        self_ref = weakref.ref(self)

        # Monitors worker process liveness. If any die unexpectedly,
        # logs an error, shuts down the executor and invokes the failure
        # callback to inform the engine.
        # SOURCE: vllm/v1/executor/multiproc_executor.py:L286-L305 monitor_workers
        def monitor_workers():
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or getattr(_self, "shutting_down", False):
                logger.debug("MultiprocWorkerMonitor: shutdown already initiated")
                return
            _self.is_failed = True
            proc = next(h.proc for h in workers if h.proc.sentinel == died[0])
            logger.error(
                "Worker proc %s died unexpectedly (exit code: %s), "
                "shutting down executor.",
                proc.name,
                proc.exitcode,
            )
            _self.shutdown()
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()

        if not inline:
            Thread(
                target=monitor_workers, daemon=True, name="MultiprocWorkerMonitor"
            ).start()
            return

        monitor_workers()

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L315-L319 register_failure_callback
    def register_failure_callback(self, callback: FailureCallback):
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L321-L331 execute_model 覆写
    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> Any:
        return self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            kv_output_aggregator=self.kv_output_aggregator,
        )

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L333-L343 sample_tokens 覆写
    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> Any:
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            kv_output_aggregator=self.kv_output_aggregator,
        )

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L345-L346 execute_dummy_batch
    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch", unique_reply_rank=self.output_rank)

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L348-L352 take_draft_token_ids
    def take_draft_token_ids(self) -> Any:
        # OPTIMIZATION: Get output only from a single worker (output_rank)
        return self.collective_rpc(
            "take_draft_token_ids", unique_reply_rank=self.output_rank
        )

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L354-L416 collective_rpc
    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator=None,
    ) -> Any:
        """Returns single result if unique_reply_rank and/or kv_output_aggregator
        is provided, otherwise list."""
        assert self.rpc_broadcast_mq is not None, (
            "collective_rpc should not be called on follower node"
        )
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # SUBTRACTED: kv_output_aggregator 聚合分支（multiproc_executor.py:
        #   L375-L379——partial(kv_output_aggregator.aggregate, ...) 把应答
        #   收敛为 KV 输出聚合；PD 解耦域（ch16/ch36），删除项 3。保留
        #   unique_reply_rank 单 rank 应答 + 默认恒等 aggregate。
        output_rank = unique_reply_rank
        aggregate = lambda x: x  # noqa: E731

        if isinstance(method, str):
            send_method = method
        else:
            send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))

        response_mqs: Sequence[MessageQueue] = self.response_mqs
        if output_rank is not None:
            response_mqs = (response_mqs[output_rank],)

        # SOURCE: vllm/v1/executor/multiproc_executor.py:L394-L410 get_response
        def get_response():
            responses = []
            for mq in response_mqs:
                dequeue_timeout = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                try:
                    status, result = mq.dequeue(timeout=dequeue_timeout)
                except TimeoutError as e:
                    raise TimeoutError(f"RPC call to {method} timed out.") from e
                if status != WorkerProc.ResponseStatus.SUCCESS:
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause"
                    )
                responses.append(result)
            return responses[0] if output_rank is not None else responses

        future = FutureWrapper(
            self.futures_queue, get_response=get_response, aggregate=aggregate
        )

        return future if non_block else future.result()

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L418-L468 _ensure_worker_termination
    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""

        # SOURCE: vllm/v1/executor/multiproc_executor.py:L424-L434 wait_for_termination
        def wait_for_termination(procs, timeout):
            if not time:
                # If we are in late stage shutdown, the interpreter may replace
                # `time` with `None`.
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        # SOURCE: vllm/v1/executor/multiproc_executor.py:L436-L437
        active_procs = lambda: [proc for proc in worker_procs if proc.is_alive()]  # noqa: E731
        initial_count = len(active_procs())

        # Give processes time to clean themselves up properly first
        logger.info(
            "[shutdown] Executor: waiting for worker exit count=%d",
            initial_count,
        )
        if wait_for_termination(
            active_procs(), timeout=envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
        ):
            logger.info_once("[shutdown] Executor: all workers exited gracefully")
            return

        # Send SIGTERM if still running
        remaining = active_procs()
        logger.warning(
            "[shutdown] Executor: workers still running after grace period; "
            "sending SIGTERM count=%d",
            len(remaining),
        )
        for p in remaining:
            p.terminate()
        if not wait_for_termination(active_procs(), 4):
            # Send SIGKILL if still running
            remaining = active_procs()
            logger.warning(
                "[shutdown] Executor: workers still running after SIGTERM; "
                "sending SIGKILL count=%d",
                len(remaining),
            )
            for p in remaining:
                p.kill()

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L470-L503 shutdown
    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if not getattr(self, "shutting_down", False):
            worker_count = len(getattr(self, "workers", None) or [])
            logger.debug(
                "[shutdown] Executor: start worker_count=%d",
                worker_count,
            )
            self.shutting_down = True

            # Make sure all the worker processes are terminated first.
            if workers := getattr(self, "workers", None):
                for w in workers:
                    # Close death_writer to signal child processes to exit
                    if w.death_writer is not None:
                        w.death_writer.close()
                        w.death_writer = None
                self._ensure_worker_termination([w.proc for w in workers])

                for w in workers:
                    # Shutdown response queues
                    if w.worker_response_mq is not None:
                        w.worker_response_mq.shutdown()
                        w.worker_response_mq = None

        if rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None):
            rpc_broadcast_mq.shutdown()
            self.rpc_broadcast_mq = None
        if response_mqs := getattr(self, "response_mqs", None):
            for mq in response_mqs:
                mq.shutdown()
            self.response_mqs = []

        logger.debug_once("[shutdown] Executor: complete")

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L505-L507 check_health
    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)
        return

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L509-L523 _get_output_rank
    def _get_output_rank(self) -> int:
        # Only returns ModelRunnerOutput from TP rank=0 and PP rank=-1
        # (the first TP worker of the last PP stage).
        # Example:
        # Assuming TP=8, PP=4, then the world_size=32
        # 0-7, PP rank 0
        # 8-15, PP rank 1
        # 16-23, PP rank 2
        # 24-31, PP rank 3
        # so world_size - tp_size = 32 - 8 = 24 should be PP rank = -1 (i.e. 3)
        return (
            self.world_size
            - self.parallel_config.tensor_parallel_size
            * self.parallel_config.prefill_context_parallel_size
        )
    @classmethod
    # SOURCE: vllm/v1/executor/multiproc_executor.py:L525-L527 supports_async_scheduling
    def supports_async_scheduling(cls) -> bool:
        return True

@dataclass
# SOURCE: vllm/v1/executor/multiproc_executor.py:L530-L537 UnreadyWorkerProcHandle
class UnreadyWorkerProcHandle:
    """WorkerProcess handle before READY."""

    proc: BaseProcess
    rank: int
    ready_pipe: Connection
    death_writer: Connection | None = None


# SOURCE: vllm/v1/executor/multiproc_executor.py:L540-L565 WorkerProcHandle
@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    # The worker process writes to this MQ in single-node mode
    worker_response_mq: MessageQueue | None
    # This is only non empty on driver node,
    # the peer worker process i writes to MQ
    # `peer_worker_response_mqs[i]`
    peer_worker_response_mqs: list[MessageQueue | None]
    death_writer: Connection | None = None
    @classmethod
    # SOURCE: vllm/v1/executor/multiproc_executor.py:L552-L565 from_unready_handle
    def from_unready_handle(
        cls,
        unready_handle: UnreadyWorkerProcHandle,
        worker_response_mq: MessageQueue | None,
        peer_worker_response_mqs: list[MessageQueue | None],
    ) -> "WorkerProcHandle":
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
            peer_worker_response_mqs=peer_worker_response_mqs,
            death_writer=unready_handle.death_writer,
        )


# SOURCE: vllm/v1/executor/multiproc_executor.py:L568-L573 WorkerProc
class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L571 READY_STR
    READY_STR = "READY"
    # SOURCE: vllm/v1/executor/multiproc_executor.py:L572-L573 MQ 属性声明
    rpc_broadcast_mq: MessageQueue | None
    worker_response_mq: MessageQueue | None

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L575-L605 _init_message_queues
    def _init_message_queues(
        self, input_shm_handle: Handle, vllm_config
    ) -> None:
        if vllm_config.parallel_config.nnodes_within_dp == 1:
            # Initialize MessageQueue for receiving SchedulerOutput
            self.rpc_broadcast_mq = MessageQueue.create_from_handle(
                input_shm_handle, self.worker.rank
            )

            # Initializes a message queue for sending the model output
            self.worker_response_mq = MessageQueue(1, 1)
            self.peer_response_handles = []
        # SUBTRACTED: 多节点分支（multiproc_executor.py:L587-L605——
        #   get_inner_dp_world_group 的远端广播 MQ / 单读者广播 MQ 装配，
        #   删除项 2）。

    # SUBTRACTED: @instrument(span_name="Worker init") 观测装饰（删除项 4）。
    # SOURCE: vllm/v1/executor/multiproc_executor.py:L608-L670 WorkerProc.__init__
    def __init__(
        self,
        vllm_config,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
        is_driver_worker: bool,
    ):
        self.rank = rank
        wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)
        # TODO: move `init_worker` to executor level as a collective rpc call
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[local_rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
            "shared_worker_lock": shared_worker_lock,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        # SUBTRACTED: setup_proc_title_and_log_prefix 两处调用（L635-L637 /
        #   L641-L644——进程标题与日志前缀装饰，全方法随删除项 4 裁除）。

        # Load model
        self.worker.init_device()
        # SUBTRACTED: setup_proc_title_and_log_prefix（并行组初始化后再装饰，
        #   L641-L644，删除项 4）。
        # SUBTRACTED: VLLM_ELASTIC_EP_SCALE_UP_LAUNCH 分支（L645-L648——
        #   弹性 EP 特性开关，删除项 4）。
        self.worker.load_model()

        scheduler_config = vllm_config.scheduler_config
        self.use_async_scheduling = scheduler_config.async_scheduling
        if self.use_async_scheduling:
            self.async_output_queue: queue.Queue = queue.Queue()
            self.async_output_copy_thread = Thread(
                target=self.async_output_busy_loop,
                daemon=True,
                name="WorkerAsyncOutputCopy",
            )
            self.async_output_copy_thread.start()

        # Set block size based on the attention backends
        current_platform.update_block_size_for_backend(vllm_config)

        # Initialize message queues after init_device() since multi-node setups
        # (nnodes_within_dp > 1) require distributed groups to be initialized
        self._init_message_queues(input_shm_handle, vllm_config)

        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()
    @staticmethod
    # SOURCE: vllm/v1/executor/multiproc_executor.py:L672-L723 make_worker_process
    def make_worker_process(
        vllm_config,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # Receive SchedulerOutput
        shared_worker_lock: LockType,
        is_driver_worker: bool,
        inherited_fds: list[int] | None = None,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # Ready pipe to communicate readiness from child to parent
        ready_reader, ready_writer = context.Pipe(duplex=False)
        # Death pipe to let child detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)
        # SUBTRACTED: fork 模式下把本 worker 的 pipe fd 并入 inherited_fds
        #   （multiproc_executor.py:L688-L690——spawn 宿主恒 None，删除项 4）。
        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": ready_writer,
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
            "is_driver_worker": is_driver_worker,
            # Have the worker close parent end of this worker's pipes too
            "inherited_fds": inherited_fds if inherited_fds is not None else [],
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(
            target=WorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=True,
        )

        # SUBTRACTED: numa_utils.configure_subprocess NUMA 绑定上下文
        #   （multiproc_executor.py:L712-L716——平台亲和，删除项 4）。
        proc.start()

        # Close child ends of pipes here in the parent
        ready_writer.close()
        death_reader.close()
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L725-L744 wait_for_response_handle_ready
    @staticmethod
    def wait_for_response_handle_ready(
        handles: dict[str, Any], proc_handle: UnreadyWorkerProcHandle
    ) -> WorkerProcHandle:
        response_handle = handles["handle"]
        worker_response_mq: MessageQueue | None = None
        if len(response_handle.local_reader_ranks) > 0:
            worker_response_mq = MessageQueue.create_from_handle(response_handle, 0)
        peer_response_handles = handles["peer_response_handles"]
        peer_worker_response_mqs = [
            # SOURCE: vllm/v1/executor/multiproc_executor.py:L734-L739 远端读者 MQ
            MessageQueue.create_from_handle(handle, -1)
            if handle.remote_subscribe_addr is not None
            else None
            for handle in peer_response_handles
        ]
        return WorkerProcHandle.from_unready_handle(
            proc_handle,
            worker_response_mq,
            peer_worker_response_mqs=peer_worker_response_mqs,
        )
    @staticmethod
    # SOURCE: vllm/v1/executor/multiproc_executor.py:L746-L782 wait_for_ready
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle],
    ) -> list[WorkerProcHandle]:
        e = Exception(
            "WorkerProc initialization failed due to an exception in a "
            "background process. See stack trace for root cause."
        )

        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        ready_proc_handles: list[WorkerProcHandle | None] = [None] * len(
            unready_proc_handles
        )
        while pipes:
            ready = multiprocessing.connection.wait(pipes.keys())
            for pipe in ready:
                assert isinstance(pipe, _ConnTypes)  # HOST SEAM (win32 放宽)
                try:
                    # Wait until the WorkerProc is ready.
                    unready_proc_handle = pipes.pop(pipe)
                    response: dict[str, Any] = pipe.recv()
                    if response["status"] != "READY":
                        raise e

                    idx = unready_proc_handle.rank % len(ready_proc_handles)
                    ready_proc_handles[idx] = WorkerProc.wait_for_response_handle_ready(
                        response, unready_proc_handle
                    )
                except EOFError:
                    e.__suppress_context__ = True
                    raise e from None

                finally:
                    # Close connection.
                    pipe.close()

        return cast(list[WorkerProcHandle], ready_proc_handles)

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L784-L793 WorkerProc.shutdown
    def shutdown(self):
        if self.rpc_broadcast_mq is not None:
            self.rpc_broadcast_mq.shutdown()
        if self.worker_response_mq is not None:
            self.worker_response_mq.shutdown()
        self.worker.shutdown()
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L795-L818 monitor_death_pipe
    def monitor_death_pipe(self, death_pipe, shutdown_requested: threading.Event):
        if death_pipe is None:
            return

        # SOURCE: vllm/v1/executor/multiproc_executor.py:L799-L810 death_pipe_monitor
        def death_pipe_monitor(queues_to_shutdown: list[MessageQueue]):
            try:
                # This will block until parent process exits (pipe closes)
                death_pipe.recv()
            except EOFError:
                logger.info_once("Parent process exited, terminating worker queues")
                shutdown_requested.set()
                for mq in queues_to_shutdown:
                    if mq is not None:
                        mq.shutdown()
            except Exception as e:
                logger.warning("Death monitoring error: %s", e)

        # Pass queue references directly to avoid gc issues if passing self
        Thread(
            target=death_pipe_monitor,
            args=([self.rpc_broadcast_mq, self.worker_response_mq],),
            daemon=True,
            name="DeathPipeMonitor",
        ).start()

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L820-L944 worker_main
    @staticmethod
    def worker_main(*args, **kwargs):
        """Worker initialization and execution loops.
        This runs a background process"""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = threading.Event()

        # SOURCE: vllm/v1/executor/multiproc_executor.py:L830-L837 signal_handler
        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested.is_set():
                shutdown_requested.set()
                logger.debug(
                    "WorkerProc handling signal %d, raising SystemExit", signum
                )
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # SUBTRACTED: assigned_physical_gpu_ids 前置发布段（multiproc_executor.py:
        #   L843-L851——逻辑卡号到物理卡号的映射，删除项 4 的平台适配轴）。
        # SUBTRACTED: set_worker_net_device（L853-L854——NIC 亲和环境注入，
        #   删除项 4）。

        worker = None
        ready_writer = kwargs.pop("ready_pipe")
        death_pipe = kwargs.pop("death_pipe", None)

        # Close inherited pipes from parent (incl. other worker pipes)
        # Explicitly passing in existing pipes and closing them makes the pipe
        # behave when using fork. Otherwise, a hidden reference to the pipes
        # exist in the child process and prevents EOF closure.
        for fd in kwargs.pop("inherited_fds", []):
            try:
                os.close(fd)
            except Exception as e:
                logger.warning("Error closing inherited connection: %s: %s", type(e), e)

        try:
            # SUBTRACTED: maybe_init_worker_tracer 初始化（L871-L877——worker
            #   进程 tracer，删除项 4）。

            worker = WorkerProc(*args, **kwargs)
            assert worker.worker_response_mq is not None
            # SUBTRACTED: numa_utils.log_current_affinity_state（L881-L882——
            #   NUMA 亲和日志，删除项 4）。

            worker.monitor_death_pipe(death_pipe, shutdown_requested)

            # Send READY once we know everything is loaded
            ready_writer.send(
                {
                    "status": WorkerProc.READY_STR,
                    "handle": worker.worker_response_mq.export_handle(),
                    "peer_response_handles": worker.peer_response_handles,
                }
            )

            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            if worker.rpc_broadcast_mq is not None:
                worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            worker.worker_busy_loop()

        except Exception:
            # NOTE: if an Exception arises in busy_loop, we send
            # a FAILURE message over the MQ RPC to notify the Executor,
            # which triggers system shutdown.
            # TODO(rob): handle case where the MQ itself breaks.

            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            elif shutdown_requested.is_set():
                logger.debug_once(
                    "[shutdown] WorkerProc: exiting after shutdown request"
                )
            else:
                logger.exception("WorkerProc failed.")

            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            shutdown_requested.set()

        except SystemExit as e:
            # SystemExit is raised on SIGTERM or SIGKILL, which usually indicates that
            # the graceful shutdown process did not succeed
            if shutdown_requested.is_set():
                logger.debug_once(
                    "[shutdown] WorkerProc: terminated by shutdown signal"
                )
            else:
                logger.warning("WorkerProc was terminated")
            # SystemExit must never be ignored
            raise e

        finally:
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L946-L948 ResponseStatus
    class ResponseStatus(Enum):
        # SOURCE: vllm/v1/executor/multiproc_executor.py:L947
        SUCCESS = auto()
        # SOURCE: vllm/v1/executor/multiproc_executor.py:L948
        FAILURE = auto()

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L950-L967 enqueue_output
    def enqueue_output(self, output: Any):
        """Prepares output from the worker and enqueues it to the
        worker_response_mq. If the output is an Exception, it is
        converted to a FAILURE response.
        """
        if isinstance(output, AsyncModelRunnerOutput):
            try:
                output = output.get_output()
            except Exception as e:
                logger.exception("Error getting async model runner output")
                output = e

        if isinstance(output, Exception):
            result = (WorkerProc.ResponseStatus.FAILURE, str(output))
        else:
            result = (WorkerProc.ResponseStatus.SUCCESS, output)
        if (response_mq := self.worker_response_mq) is not None:
            response_mq.enqueue(result)

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L969-L977 handle_output
    def handle_output(self, output: Any):
        """Handles output from the worker. If async scheduling is enabled,
        it is passed to the async_output_busy_loop thread. Otherwise, it is
        enqueued directly to the worker_response_mq.
        """
        if self.use_async_scheduling:
            self.async_output_queue.put(output)
        else:
            self.enqueue_output(output)

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L979-L995 async_output_busy_loop
    def async_output_busy_loop(self):
        """Entrypoint for the thread which handles outputs asynchronously."""

        # set device to the worker device for the thread.
        # a thread will not inherit the context of the main thread.
        # when calling any cuda runtime functions, it will implicitly
        # create a new cuda context on device 0, consuming extra memory.
        # here we set the device to the worker device for the thread,
        # enforcing the context to be the same as the main thread.
        from .._host_seams import current_platform

        if hasattr(self.worker, "device"):
            current_platform.set_device(self.worker.device)

        while True:
            output = self.async_output_queue.get()
            self.enqueue_output(output)

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L997-L1022 worker_busy_loop
    def worker_busy_loop(self):
        """Main busy loop for Multiprocessing Workers"""
        assert self.rpc_broadcast_mq is not None
        while True:
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(
                indefinite=True
            )
            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)

                output = func(*args, **kwargs)

                if output_rank is None or self.rank == output_rank:
                    self.handle_output(output)
            except Exception as e:
                # Notes have been introduced in python 3.11
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                # exception might not be serializable, so we convert it to
                # string, only for logging purpose.
                if output_rank is None or self.rank == output_rank:
                    self.handle_output(e)

    # SUBTRACTED: setup_proc_title_and_log_prefix 全方法（multiproc_executor.py:
    #   L1024-L1058——按 DP/PP/PCP/TP/DCP/EP 组装饰进程标题与日志前缀，
    #   删除项 4）。


# SOURCE: vllm/utils/system_utils.py set_multiprocessing_worker_envs（OMP 调优
#   体已按删除项 4 裁除，保留函数面与 spawn 纠偏调用）
# SOURCE: (见 impl-notes.md §Source Map——executor/multiproc_executor.py)
def set_multiprocessing_worker_envs():
    """Set up environment variables that should be used when there are workers
    in a multiprocessing environment. This should be called by the parent
    process before worker processes are created"""

    _maybe_force_spawn()

    # SUBTRACTED: OMP_NUM_THREADS 线程并行度调优体（multiproc_executor.py:
    #   L1068-L1089——每 GPU 多 worker 的 CPU 争用调优，删除项 4）。


# SOURCE: vllm/utils/system_utils.py _maybe_force_spawn — HOST SEAM（win32 宿主
# 本就只有 spawn；unix 上真实逻辑按 CUDA/方法名强制 spawn）
# SOURCE: (见 impl-notes.md §Source Map——executor/multiproc_executor.py)
def _maybe_force_spawn() -> None:  # HOST SEAM
    return None


# SOURCE: vllm/utils/system_utils.py:L168-L181 get_mp_context — HOST SEAM:
# 真实默认 fork/unix；zmq 句柄与 CUDA context 过不了 fork，win32 宿主只有
# spawn——观测契约一致（WorkerProc.worker_main 在独立进程里跑）。
# SOURCE: (见 impl-notes.md §Source Map——executor/multiproc_executor.py)
def get_mp_context():
    import sys as _sys

    import multiprocessing as mp

    method = "spawn" if _sys.platform == "win32" else envs.VLLM_WORKER_MULTIPROC_METHOD
    return mp.get_context(method)
