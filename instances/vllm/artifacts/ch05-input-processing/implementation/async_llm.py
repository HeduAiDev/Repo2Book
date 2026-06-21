"""AsyncLLM.add_request 的 n 路由片段 —— 只做减法的精简版。

本章只关心 add_request 里『输入处理 → assign_request_id → 按 n 路由』这一段：
n==1（或 pooling）直接单请求接入；n>1 用 ParentRequest fan-out 出 n 个子请求。
完整 add_request（事件循环、output_handler、resumable 等）属第三/四阶段，留后续章节。
"""

from __future__ import annotations

from copy import copy

from messages import PoolingParams, SamplingParams
from parallel_sampling import ParentRequest


# SOURCE: vllm/v1/engine/async_llm.py:L280 (AsyncLLM.add_request)
class AsyncLLM:
    def __init__(self, input_processor, supported_tasks=("generate",)):
        # SOURCE: vllm/v1/engine/async_llm.py (AsyncLLM.__init__) — 替身
        self.input_processor = input_processor
        self._supported_tasks = supported_tasks
        # SUBTRACTED: engine_core/output_processor/事件循环等 — 留 Stage 2/3。

    def _add_request(self, request, parent_request, index):
        # SOURCE: vllm/v1/engine/async_llm.py:L400 (AsyncLLM._add_request)
        # SUBTRACTED: output_processor.add_request + engine_core.add_request_async
        #   过进程边界 — 属请求接入下游（Stage 2），原 async_llm.py:L400+。
        #   精简版记录接入结果供测试断言。
        self.added.append((request.request_id, parent_request, index))

    # SOURCE: vllm/v1/engine/async_llm.py:L280 (add_request) —— 仅保留输入处理 + n 路由片段
    def add_request(
        self,
        request_id: str,
        prompt,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request=None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ):
        self.added: list = []
        is_pooling = isinstance(params, PoolingParams)

        request = self.input_processor.process_inputs(
            request_id,
            prompt,
            params,
            supported_tasks=self._supported_tasks,
            arrival_time=arrival_time,
            lora_request=lora_request,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )

        self.input_processor.assign_request_id(request)

        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        # SOURCE: vllm/v1/engine/async_llm.py:L381
        if is_pooling or params.n == 1:
            self._add_request(request, None, 0)
            return request

        parent_params = params
        assert isinstance(parent_params, SamplingParams)

        # Fan out child requests (for n>1).
        parent_request = ParentRequest(request)
        for idx in range(parent_params.n):
            request_id, child_params = parent_request.get_child_info(idx)
            child_request = request if idx == parent_params.n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params
            self._add_request(child_request, parent_request, idx)
        return parent_request
