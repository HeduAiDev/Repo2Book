# Subtract-only companion for v3 ch17 — vllm/v1/executor/uniproc_executor.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions, each marked `# SUBTRACTED:`.
#
# Deletions here (dossier subtraction_plan.delete):
#   #1  ExecutorWithExternalLauncher 全类（uniproc_executor.py:L150-L196）；
#   #4  set_worker_net_device 调用（L59-L60）、VLLM_ELASTIC_EP_SCALE_UP_LAUNCH
#       分支（L65-L68）、@instrument 装饰。

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from multiprocessing import Lock
from typing import Any

from .._host_seams import (
    GrammarOutput,
    SchedulerOutput,
    current_platform,
    get_distributed_init_method,
    get_ip,
    get_open_port,
    init_logger,
)
from ..serial_utils import run_method
from .._host_seams import AsyncModelRunnerOutput
from ..worker.worker_base import WorkerWrapperBase
from .abstract import Executor

logger = init_logger(__name__)


# SOURCE: vllm/v1/executor/uniproc_executor.py:L26-L42 AsyncOutputFuture
class AsyncOutputFuture(Future):
    # SOURCE: vllm/v1/executor/uniproc_executor.py:L27-L30 __init__
    def __init__(self, async_output: AsyncModelRunnerOutput, single_value: bool):
        self.async_output = async_output
        self.single_value = single_value
        super().__init__()

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L32-L42 result
    def result(self, timeout=None):
        if timeout is not None:
            raise RuntimeError("timeout not implemented")

        if not super().done():
            try:
                output = self.async_output.get_output()
                self.set_result(output if self.single_value else [output])
            except Exception as e:
                self.set_exception(e)
        return super().result()


# SOURCE: vllm/v1/executor/uniproc_executor.py:L45 UniProcExecutor
class UniProcExecutor(Executor):
    # SOURCE: vllm/v1/executor/uniproc_executor.py:L46-L69 _init_executor
    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        self.driver_worker = WorkerWrapperBase(rpc_rank=0)
        distributed_init_method, rank, local_rank = self._distributed_args()
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=True,
            shared_worker_lock=Lock(),
        )

        # SUBTRACTED: set_worker_net_device(local_rank, self.vllm_config)
        #   （uniproc_executor.py:L59-L60——NIC 亲和平台适配，删除项 4）。

        self.driver_worker.init_worker(all_kwargs=[kwargs])
        self.driver_worker.init_device()

        # SUBTRACTED: VLLM_ELASTIC_EP_SCALE_UP_LAUNCH 分支（L65-L68——弹性 EP
        #   特性开关走 elastic_ep_execute，删除项 4）。
        self.driver_worker.load_model()
        current_platform.update_block_size_for_backend(self.vllm_config)

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L71-L77 _distributed_args
    def _distributed_args(self) -> tuple[str, int, int]:
        """Return (distributed_init_method, rank, local_rank)."""
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
        # set local rank as the device index if specified
        device_info = self.vllm_config.device_config.device.__str__().split(":")
        local_rank = int(device_info[1]) if len(device_info) > 1 else 0
        return distributed_init_method, 0, local_rank

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L79-L106 collective_rpc 直调
    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        if not non_block:
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                result = result.get_output()
            return result if single_value else [result]

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

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L108-L121 execute_model 覆写
    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> Any:
        output = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            single_value=True,
        )
        # In non-blocking mode, surface any exception as early as possible.
        if non_block and output.done():
            # Raise the exception in-line if the task failed.
            output.result()
        return output

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L123-L131 sample_tokens 覆写
    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> Any:
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            non_block=non_block,
            single_value=True,
        )

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L133-L134 take_draft_token_ids
    def take_draft_token_ids(self) -> Any:
        return self.collective_rpc("take_draft_token_ids", single_value=True)

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L136-L139 check_health
    def check_health(self) -> None:
        # UniProcExecutor will always be healthy as long as
        # it's running.
        return

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L141-L143 shutdown
    def shutdown(self) -> None:
        if worker := self.driver_worker:
            worker.shutdown()
    @classmethod
    # SOURCE: vllm/v1/executor/uniproc_executor.py:L145-L147 supports_async_scheduling
    def supports_async_scheduling(cls) -> bool:
        return True


# SUBTRACTED: ExecutorWithExternalLauncher 全类（uniproc_executor.py:L150-L196，
#   torchrun 兼容的离线 TP 特例——删除项 1；其 determine_available_memory 的
#   跨 rank all-reduce MIN 亦随之删除）。
