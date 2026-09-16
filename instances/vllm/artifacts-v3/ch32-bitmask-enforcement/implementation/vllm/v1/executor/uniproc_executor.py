# SOURCE: vllm/v1/executor/uniproc_executor.py
# v3 ch31 脊柱⑫：UniProcExecutor 的三方法转发面——AsyncOutputFuture
# （L26-L42，只在 result() 才等 D2H 事件——重叠编排成立的前提之一）、
# collective_rpc（L85-L106）、execute_model/sample_tokens/take_draft_token_ids
# （L108-L137，EngineCore 两段式调用的真实受话端）。
# SUBTRACTED：executor 工厂/init_worker/load_model 装配（L45-L83——ch17
#   启动域）、check_health/collective_rpc 的 cloudpickle 字节分支（经
#   serial_utils 的 SEAM 裁剪）。
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.serial_utils import run_method


# SOURCE: vllm/v1/executor/uniproc_executor.py:L26-L42 AsyncOutputFuture —— 逐字
class AsyncOutputFuture(Future):
    # SOURCE: vllm/v1/executor/uniproc_executor.py:L27-L30 AsyncOutputFuture.__init__ —— 逐字
    def __init__(self, async_output: AsyncModelRunnerOutput, single_value: bool):
        self.async_output = async_output
        self.single_value = single_value
        super().__init__()

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L32-L42 result —— 逐字（只在 result() 才等 D2H）
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


# SOURCE: vllm/v1/executor/uniproc_executor.py:L45 UniProcExecutor —— 本章切面
class UniProcExecutor:
    # SUBTRACTED: vllm/v1/executor/uniproc_executor.py:L46-L83 _init_executor/
    #   _distributed_args（worker 装配与分布式握手——ch17）。ENGINE SEAM：
    #   driver_worker 由外部注入（测试替身或真实 worker 直挂）。

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L85-L106 collective_rpc —— 逐字
    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
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

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L108-L121 execute_model —— 逐字
    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
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

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L123-L131 sample_tokens —— 逐字
    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            non_block=non_block,
            single_value=True,
        )

    # SOURCE: vllm/v1/executor/uniproc_executor.py:L133-L134 take_draft_token_ids —— 逐字
    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.collective_rpc("take_draft_token_ids", single_value=True)

    # SUBTRACTED: vllm/v1/executor/uniproc_executor.py:L136-L139 check_health ——
    #   健康检查面（ch17 运维域）。
