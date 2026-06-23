# 精简版（subtract-only companion）—— vllm/v1/executor/multiproc_executor.py
#
# 多进程编排层：把 GroupCoordinator 在每个 worker 真正拉起来、并通过一条共享内存
# 广播队列做控制面 RPC。本章聚焦『一次 collective_rpc 的广播与回收』骨架：
# leader 建 rpc_broadcast_mq → spawn 每 rank 一个 WorkerProc → collective_rpc
# enqueue 广播 → 各 worker 的 worker_busy_loop dequeue 反射调用 → 仅 output_rank
# 回写 response_mq → FutureWrapper 有序排空取回。
#
# 环境桥接（NOT vLLM 新抽象）：
#   - MessageQueue：真实 vLLM 用 device_communicators.shm_broadcast.MessageQueue
#     （跨进程共享内存广播）。host 上为了无依赖可运行，用进程内等价队列替换，
#     保留 enqueue/dequeue/export/wait_until_ready 接口与广播语义。
#   - WorkerProc 用线程而非 multiprocessing.Process 承载（同一台机、单进程内即可
#     演示 RPC 广播/回收的控制流），spawn/fork/inherited_fds/monitor 全部 SUBTRACTED。
from __future__ import annotations

import pickle
import threading
import time
from collections import deque
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from enum import Enum, auto
from functools import partial
from typing import Any, Callable, Sequence

from _mq_bridge import MessageQueue, get_mp_context


# SOURCE: vllm/v1/executor/multiproc_executor.py:L69
class FutureWrapper(Future):
    # SOURCE: vllm/v1/executor/multiproc_executor.py:L70
    def __init__(
        self,
        futures_queue: "deque[FutureWrapper]",
        get_response: Callable[[], Any],
        aggregate: Callable = lambda x: x,
    ):
        self.futures_queue = futures_queue
        self.get_response = get_response
        self.aggregate = aggregate
        super().__init__()
        self.futures_queue.appendleft(self)

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L82
    def result(self, timeout=None):
        if timeout is not None:
            raise RuntimeError("timeout not implemented")

        # Drain any futures ahead of us in the queue.
        while not self.done():
            future = self.futures_queue.pop()
            future._wait_for_response()
        return super().result()

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L92
    def _wait_for_response(self):
        try:
            response = self.aggregate(self.get_response())
            with suppress(InvalidStateError):
                self.set_result(response)
        except Exception as e:
            with suppress(InvalidStateError):
                self.set_exception(e)


# SOURCE: vllm/v1/executor/multiproc_executor.py:L102
class MultiprocExecutor:
    # SUBTRACTED: 原版继承 Executor（vllm.v1.executor.abstract.Executor）并经其
    # __init__ 调 _init_executor；本章无 vLLM 配置体系，直接在 __init__ 里完成
    # 等价编排（leader 建 mq、spawn worker、收 response_mqs、算 output_rank）。
    supports_pp: bool = True

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L109（_init_executor 骨架）
    def __init__(
        self,
        world_size: int,
        worker_factory: Callable[[int], Any],
        tensor_parallel_size: int = 1,
        prefill_context_parallel_size: int = 1,
    ):
        # SUBTRACTED: vllm_config / monitor_workers / _finalizer / failure_callback /
        # set_multiprocessing_worker_envs / distributed_init_method 等启动环境
        # （vllm/v1/executor/multiproc_executor.py:L105-L129）。worker_factory(rank)
        # 是测试注入的『worker 对象工厂』，替代真实 init_worker→init_device→load_model。
        self.world_size = world_size
        self.local_world_size = world_size
        self.tensor_parallel_size = tensor_parallel_size
        self.prefill_context_parallel_size = prefill_context_parallel_size
        self.is_failed = False

        self.rpc_broadcast_mq: MessageQueue | None = None
        scheduler_output_handle = None
        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        # SUBTRACTED: node_rank_within_dp 多节点 leader/follower 判定
        # （multiproc_executor.py:L134-L149）——本章按单节点 leader 主线，必建 mq。
        self.rpc_broadcast_mq = MessageQueue(self.world_size, self.local_world_size)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Create workers
        context = get_mp_context()
        # SUBTRACTED: shared_worker_lock / inherited_fds(fork) / cpu_omp_manager /
        # try-finally 清理（multiproc_executor.py:L159-L244）——单节点主线不需要。
        unready_workers: list = []
        for local_rank in range(self.local_world_size):
            global_rank = local_rank
            unready_worker_handle = WorkerProc.make_worker_process(
                worker=worker_factory(global_rank),
                local_rank=local_rank,
                rank=global_rank,
                input_shm_handle=scheduler_output_handle,
            )
            unready_workers.append(unready_worker_handle)

        # Workers must be created before wait_for_ready to avoid
        # deadlock, since worker.init_device() does a device sync.
        self.workers = WorkerProc.wait_for_ready(unready_workers)

        # SUBTRACTED: start_worker_monitor（健康监控线程，multiproc_executor.py:L203）。

        self.response_mqs: list[MessageQueue] = []
        # Only leader node have remote response mqs
        for rank in range(self.world_size):
            local_message_queue = self.workers[rank].worker_response_mq
            assert local_message_queue is not None
            self.response_mqs.append(local_message_queue)

        # Ensure message queues are ready. Will deadlock if re-ordered
        # Must be kept consistent with the WorkerProc.
        if self.rpc_broadcast_mq is not None:
            self.rpc_broadcast_mq.wait_until_ready()
        for response_mq in self.response_mqs:
            response_mq.wait_until_ready()

        self.futures_queue: "deque[FutureWrapper]" = deque()

        self.output_rank = self._get_output_rank()

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L339
    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Returns single result if unique_reply_rank is provided,
        otherwise list."""
        assert self.rpc_broadcast_mq is not None, (
            "collective_rpc should not be called on follower node"
        )
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # SUBTRACTED: kv_output_aggregator 分支（DP 下重组 KV 输出的聚合，
        # multiproc_executor.py:L360-L364）——主线 aggregate 为恒等。
        output_rank = unique_reply_rank
        aggregate: Callable[[Any], Any] = lambda x: x

        if isinstance(method, str):
            send_method = method
        else:
            # SUBTRACTED: 原版用 cloudpickle.dumps(method)；标准库 pickle 可序列化
            # 顶层可调用对象，足以演示『method 可为 str 或 bytes』两条分支。
            send_method = pickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))

        response_mqs: Sequence[MessageQueue] = self.response_mqs
        if output_rank is not None:
            response_mqs = (response_mqs[output_rank],)

        # SOURCE: vllm/v1/executor/multiproc_executor.py:L379
        def get_response():
            responses = []
            for mq in response_mqs:
                dequeue_timeout = (
                    None if deadline is None else (deadline - time.monotonic())
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
            self.futures_queue,
            get_response=get_response,
            aggregate=aggregate,
        )

        return future if non_block else future.result()

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L480
    def _get_output_rank(self) -> int:
        # Only returns ModelRunnerOutput from TP rank=0 and PP rank=-1
        # (the first TP worker of the last PP stage).
        # Example:
        # Assuming TP=8, PP=4, then the world_size=32
        # 0-7, PP rank 0 ... 24-31, PP rank 3
        # so world_size - tp_size = 32 - 8 = 24 should be PP rank = -1 (i.e. 3)
        return (
            self.world_size
            - self.tensor_parallel_size * self.prefill_context_parallel_size
        )

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L432（shutdown，按线程模型裁剪）
    def shutdown(self):
        # SUBTRACTED: 真实 shutdown 经 death pipe + terminate/kill 回收 worker 进程
        # （multiproc_executor.py:L405-L478）；本章 worker 是守护线程，发哨兵令其退出。
        if self.rpc_broadcast_mq is not None:
            for w in self.workers:
                w.stop()


# SOURCE: vllm/v1/executor/multiproc_executor.py:L539
class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L897
    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L546
    def _init_message_queues(self, input_shm_handle) -> None:
        # SUBTRACTED: nnodes_within_dp>1 的 remote MQ 分支
        # （multiproc_executor.py:L558-L576）——单节点主线：从 leader 的 handle 连上
        # 同一条广播队列，并为本 worker 建一个 response 队列。
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.rank
        )
        self.worker_response_mq = MessageQueue(1, 1)

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L579
    def __init__(
        self,
        worker: Any,
        local_rank: int,
        rank: int,
        input_shm_handle,
    ):
        self.rank = rank
        # SUBTRACTED: WorkerWrapperBase / init_worker / setup_proc_title /
        # init_device / load_model / async scheduling 线程
        # （multiproc_executor.py:L590-L633）——这些是把真实 Worker 拉起、在
        # init_device 内部调 init_distributed_environment + initialize_model_parallel
        # 建群组的步骤；本章用注入的 worker 对象替代，聚焦 RPC 控制流。
        self.worker = worker
        self.use_async_scheduling = False

        # Initialize message queues after init_device().
        self._init_message_queues(input_shm_handle)

    @staticmethod
    def make_worker_process(
        worker: Any,
        local_rank: int,
        rank: int,
        input_shm_handle,
    ) -> "WorkerProc":
        # SOURCE: vllm/v1/executor/multiproc_executor.py:L643
        # SUBTRACTED: get_mp_context().Process spawn + ready/death Pipe 握手
        # （multiproc_executor.py:L654-L716）——本章用线程承载 worker_busy_loop。
        proc = WorkerProc(worker, local_rank, rank, input_shm_handle)
        proc._thread = threading.Thread(target=proc.worker_main, daemon=True)
        proc._thread.start()
        return proc

    @staticmethod
    def wait_for_ready(unready: list) -> list["WorkerProc"]:
        # SOURCE: vllm/v1/executor/multiproc_executor.py:L718
        # SUBTRACTED: 通过 ready_pipe 收 READY 信号 / 超时处理
        # （multiproc_executor.py:L718-L789）——线程模型下 _init_message_queues 已就绪，
        # 这里只把句柄原样返回，保留『wait_for_ready 收齐所有 worker』的语义位点。
        return unready

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L791
    def worker_main(self):
        # SUBTRACTED: 真实 worker_main 含信号处理、shutdown_requested、READY 上报、
        # try/except 退出（multiproc_executor.py:L791-L895）；保留进入 busy loop 的主线。
        self.worker_response_mq.wait_until_ready()
        try:
            self.worker_busy_loop()
        except _WorkerStop:
            return

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L901
    def enqueue_output(self, output: Any):
        """Prepares output from the worker and enqueues it to the
        worker_response_mq. If the output is an Exception, it is
        converted to a FAILURE response.
        """
        # SUBTRACTED: AsyncModelRunnerOutput.get_output() 拆包
        # （multiproc_executor.py:L906-L907）——async scheduling 专用。
        if isinstance(output, Exception):
            result = (WorkerProc.ResponseStatus.FAILURE, str(output))
        else:
            result = (WorkerProc.ResponseStatus.SUCCESS, output)
        if (response_mq := self.worker_response_mq) is not None:
            response_mq.enqueue(result)

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L916
    def handle_output(self, output: Any):
        """Handles output from the worker. If async scheduling is enabled,
        it is passed to the async_output_busy_loop thread. Otherwise, it is
        enqueued directly to the worker_response_mq.
        """
        # SUBTRACTED: async scheduling 分支（async_output_queue.put，
        # multiproc_executor.py:L921-L922）——本章按同步 enqueue 主线。
        self.enqueue_output(output)

    # SOURCE: vllm/v1/executor/multiproc_executor.py:L944
    def worker_busy_loop(self):
        """Main busy loop for Multiprocessing Workers"""
        assert self.rpc_broadcast_mq is not None
        while True:
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(
                indefinite=True
            )
            if method == _STOP_SENTINEL:
                raise _WorkerStop
            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    # SUBTRACTED: 原版 cloudpickle.loads；标准库 pickle 对称还原。
                    func = partial(pickle.loads(method), self.worker)

                output = func(*args, **kwargs)
            except _WorkerStop:
                raise
            except Exception as e:
                # Notes have been introduced in python 3.11
                if hasattr(e, "add_note"):
                    e.add_note("")
                # exception might not be serializable, so we convert it to
                # string, only for logging purpose.
                if output_rank is None or self.rank == output_rank:
                    self.handle_output(e)
                continue

            if output_rank is None or self.rank == output_rank:
                self.handle_output(output)

    def stop(self):
        # SOURCE: vllm/v1/executor/multiproc_executor.py:L432（companion-only 收尾钩子，
        # 对应真实 shutdown 经 death pipe 令进程退出；线程模型改用哨兵令 busy loop 退出）
        self.rpc_broadcast_mq.enqueue((_STOP_SENTINEL, (), {}, None))


class _WorkerStop(Exception):
    # SOURCE: vllm/v1/executor/multiproc_executor.py:L791（companion-only 线程退出哨兵，
    # 对应真实 worker_main 收到 shutdown 信号后跳出 busy loop 的控制流）
    pass


_STOP_SENTINEL = "__stop__"
