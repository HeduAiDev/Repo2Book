"""AsyncLLM.add_request 的 n>1 扇出分支——本章主控制流（兑现 ch04 延迟的 n>1）。

n==1/pooling 走单请求 fast path；n>1 时建 ParentRequest 并循环扇出 n 个独立 child
EngineCoreRequest，各自经「本进程 OutputProcessor + 独立进程 EngineCore」下发。
"""

from __future__ import annotations

from copy import copy

from .input_processor import InputProcessor
from .output_processor import OutputProcessor
from .parallel_sampling import ParentRequest
from .types import (
    EngineCoreRequest,
    RequestOutputCollector,
    SamplingParams,
)


# SOURCE: vllm/v1/engine/core_client.py — EngineCore 客户端（跨进程 IPC 入口，本章只关心 add）
class EngineCore:
    """独立进程的引擎核心。本章关键事实：它把 n 个 child 当作 n 个普通独立请求接收，

    引擎侧无任何 n>1 批量语义。这里收集已下发的 child 以便测试观察「确实下发了 n 个」。
    """

    def __init__(self) -> None:
        # SOURCE: vllm/v1/engine/core_client.py — EngineCore 客户端构造（精简）
        self.received: list[EngineCoreRequest] = []

    # SOURCE: vllm/v1/engine/core_client.py — add_request_async
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        # SUBTRACTED: 真实实现经 msgpack 序列化 + ZMQ 发往 EngineCore 进程并入调度队列
        #            （vllm/v1/engine/core_client.py）——这里只记录「作为独立请求下发」这一事实。
        self.received.append(request)


# SOURCE: vllm/v1/engine/async_llm.py — class AsyncLLM（本章只保留 add_request 扇出主线）
class AsyncLLM:
    # SOURCE: vllm/v1/engine/async_llm.py — __init__（精简：只装配本章三方）
    def __init__(self) -> None:
        self.input_processor = InputProcessor()
        self.output_processor = OutputProcessor()
        self.engine_core = EngineCore()

    # SOURCE: vllm/v1/engine/async_llm.py — _run_output_handler（本章占位）
    def _run_output_handler(self) -> None:
        # SUBTRACTED: 首次 add_request 时惰性启动 output_handler 协程
        #            （vllm/v1/engine/async_llm.py:L355-L358）——与扇出控制流无关。
        pass

    # SOURCE: vllm/v1/engine/async_llm.py:L368 def add_request（n>1 扇出分支）
    async def add_request(
        self,
        request: EngineCoreRequest,
        is_pooling: bool = False,
    ) -> RequestOutputCollector:
        # SUBTRACTED: process_inputs（tokenize/clone params）、AsyncGenerator 流式输入分支、
        #            EngineCoreRequest 直传 deprecated 分支、reasoning_ended/reasoning_parser_kwargs
        #            注入、kv_sharing_fast_prefill + prompt_logprobs 前置校验
        #            （vllm/v1/engine/async_llm.py 上文，至 L367）——均与 n>1 扇出主线无关。
        self.input_processor.assign_request_id(request)

        # We start the output_handler on the first call to add_request() so
        # we can call __init__ before the event loop.
        self._run_output_handler()

        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        # Create a new output collector for the request.
        queue = RequestOutputCollector(params.output_kind, request.request_id)

        if is_pooling or params.n == 1:
            await self._add_request(request, None, 0, queue)
            return queue

        parent_params = params
        assert isinstance(parent_params, SamplingParams)

        # Fan out child requests (for n>1).
        parent_request = ParentRequest(request)
        for idx in range(parent_params.n):
            request_id, child_params = parent_request.get_child_info(idx)
            child_request = request if idx == parent_params.n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params
            await self._add_request(child_request, parent_request, idx, queue)
        return queue

    # SOURCE: vllm/v1/engine/async_llm.py:L400 def _add_request
    async def _add_request(
        self,
        request: EngineCoreRequest,
        parent_req: ParentRequest | None,
        index: int,
        queue: RequestOutputCollector,
    ) -> None:
        # Add the request to OutputProcessor (this process).
        self.output_processor.add_request(request, parent_req, index, queue)
        # Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(request)
        # SUBTRACTED: log_requests 日志（vllm/v1/engine/async_llm.py:L413-L414）。
