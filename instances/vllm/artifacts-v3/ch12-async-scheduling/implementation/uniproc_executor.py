# SOURCE: vllm/v1/executor/uniproc_executor.py
# UniProcExecutor —— 单进程直调后端。本章切面：AsyncOutputFuture（L26-L42
# 惰性收割，m13）+ collective_rpc 的 non_block 路径（L79-L106——AsyncModelRunner
# Output 包 future）+ execute_model/sample_tokens/take_draft_token_ids 转发面
# （L108-L134）+ supports_async_scheduling（L145-L147——仲裁链的一票）。
# 『异步』在 uniproc 不是另一个线程，是推迟的等待。
from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, Callable

from .logger import init_logger
from .outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from .output import SchedulerOutput

logger = init_logger(__name__)


# SOURCE: vllm/v1/serial_utils.py run_method（uniproc 的直调面）—— ENGINE
# SEAM：真实经序列化 RPC 分发（multiproc/ray 才需要）；单进程即反射直调。
def run_method(worker, method: str | Callable, args: tuple, kwargs: dict) -> Any:
    # SOURCE: vllm/v1/serial_utils.py run_method — ENGINE SEAM
    if callable(method):
        return method(*args, **kwargs)
    return getattr(worker, method)(*args, **kwargs)


# SOURCE: vllm/v1/executor/abstract.py:L44 Executor —— 转发骨架（分布式面归
# ch17；本章保仲裁链消费的 supports_async_scheduling 默认位）
class Executor(ABC):
    # SOURCE: vllm/v1/executor/abstract.py:L364-L368 supports_async_scheduling
    # （默认 False——仲裁链的一票否决权；uniproc/multiproc 覆写 True，ray 未
    # 覆写即 False → 默认禁 async）
    @classmethod
    def supports_async_scheduling(cls) -> bool:
        # SOURCE: vllm/v1/executor/abstract.py:L365-L368
        return False

    # SOURCE: vllm/v1/executor/abstract.py Executor.execute_model（抽象面）
    @abstractmethod
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> Any:
        # SOURCE: vllm/v1/executor/abstract.py Executor.execute_model
        raise NotImplementedError

    # SUBTRACTED: 工厂/生命周期/健康检查面（ch17）。

    # SOURCE: vllm/v1/executor/abstract.py:L44-L47 Executor.__init__（骨架：
    # vllm_config 落字段 + _init_executor 钩子）
    def __init__(self, vllm_config, *args, **kwargs):
        # SOURCE: vllm/v1/executor/abstract.py Executor.__init__
        self.vllm_config = vllm_config
        self._init_executor(*args, **kwargs)


# SOURCE: vllm/v1/executor/uniproc_executor.py:L26 AsyncOutputFuture（逐字——m13）
class AsyncOutputFuture(Future):
    # SOURCE: vllm/v1/executor/uniproc_executor.py:L27-L30 __init__ (逐字)
    def __init__(self, async_output: AsyncModelRunnerOutput, single_value: bool):
        # SOURCE: vllm/v1/executor/uniproc_executor.py:L28-L30
        self.async_output = async_output
        self.single_value = single_value
        super().__init__()

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L32-L42 result (逐字)
    def result(self, timeout=None):
        # SOURCE: vllm/v1/executor/uniproc_executor.py:L33-L34
        if timeout is not None:
            raise RuntimeError("timeout not implemented")

        # SOURCE: vllm/v1/executor/uniproc_executor.py:L36-L42 result() 才收割
        if not super().done():
            try:
                output = self.async_output.get_output()
                self.set_result(output if self.single_value else [output])
            except Exception as e:
                self.set_exception(e)
        return super().result()


# SOURCE: vllm/v1/executor/uniproc_executor.py:L45 UniProcExecutor
class UniProcExecutor(Executor):
    # SOURCE: vllm/v1/executor/uniproc_executor.py:L46-L69 _init_executor —
    # ENGINE SEAM（ch17 边界）：真实走 WorkerWrapperBase/init_device/load_model；
    # 本章 driver_worker = GPUWorker 壳直构。
    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        # SOURCE: vllm/v1/executor/uniproc_executor.py:L48 — ENGINE SEAM
        from .gpu_worker import GPUWorker

        self.driver_worker = GPUWorker(self.vllm_config)

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L79 collective_rpc（转发面
    # 逐字——non_block 路径把 AsyncModelRunnerOutput 包成 AsyncOutputFuture）
    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        # SOURCE: vllm/v1/executor/uniproc_executor.py:L87-L88
        if kwargs is None:
            kwargs = {}

        # SOURCE: vllm/v1/executor/uniproc_executor.py:L91-L95 阻塞路径
        if not non_block:
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                result = result.get_output()
            return result if single_value else [result]

        # SOURCE: vllm/v1/executor/uniproc_executor.py:L97-L106 非阻塞路径
        # （异步输出包装位）
        try:
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                return AsyncOutputFuture(result, single_value)
            future = Future[Any]()
            future.set_result(result if single_value else [result])
        except Exception as e:
            future = Future[Any]()
            future.set_exception(e)
        return future

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L108-L121 execute_model
    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # SOURCE: vllm/v1/executor/uniproc_executor.py:L111-L116
        output = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            single_value=True,
        )
        # In non-blocking mode, surface any exception as early as possible.
        # SOURCE: vllm/v1/executor/uniproc_executor.py:L117-L120
        if non_block and output.done():
            # Raise the exception in-line if the task failed.
            output.result()
        return output

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L123-L131 sample_tokens
    def sample_tokens(  # type: ignore[override]
        self, grammar_output, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # SOURCE: vllm/v1/executor/uniproc_executor.py:L126-L131
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            non_block=non_block,
            single_value=True,
        )

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L133-L134 take_draft_token_ids
    def take_draft_token_ids(self):
        # SOURCE: vllm/v1/executor/uniproc_executor.py:L133-L134
        return self.collective_rpc("take_draft_token_ids", single_value=True)

    # SUBTRACTED: check_health/shutdown/external launcher（L136-L147+——ch17）。

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L145-L147 supports_async_
    # scheduling（覆写 True——仲裁链的一票）
    @classmethod
    def supports_async_scheduling(cls) -> bool:
        # SOURCE: vllm/v1/executor/uniproc_executor.py:L146-L147
        return True


# SUBTRACTED: ExecutorWithExternalLauncher（L150+——torchrun 面）。
