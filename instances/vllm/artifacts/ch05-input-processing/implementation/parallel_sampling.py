"""n>1 并行采样：ParentRequest —— 只做减法的精简版。

process_inputs 只产出**单个**父 EngineCoreRequest；当 SamplingParams.n>1 时，
请求接入层（AsyncLLM/LLMEngine 的 add_request）用 ParentRequest fan-out 出 n 个子请求，
并负责把 n 路子输出聚合回去。与真实 vllm/v1/engine/parallel_sampling.py 同名同结构同控制流。
"""

from __future__ import annotations

from copy import copy
from typing import cast

from messages import (
    CompletionOutput,
    EngineCoreRequest,
    RequestOutputKind,
    SamplingParams,
)


class ParentRequest:
    """Info, state & processing for parallel sampling request.

    Store parent request ID and sampling params.
    Facilitate generating child request sampling params.
    """

    request_id: str
    external_req_id: str
    sampling_params: SamplingParams
    child_requests: set[str]
    output_aggregator: list[CompletionOutput]
    max_num_generation_tokens: int
    cached_child_sampling_params: SamplingParams | None

    # SOURCE: vllm/v1/engine/parallel_sampling.py:L36 (__init__)
    def __init__(self, request: EngineCoreRequest) -> None:
        assert request.external_req_id is not None
        sampling_params = request.params
        self.request_id = request.request_id
        self.external_req_id = request.external_req_id
        self.sampling_params = sampling_params

        self.child_requests = set()
        self.output_aggregator = (
            [cast(CompletionOutput, None)] * sampling_params.n
            if (sampling_params.output_kind == RequestOutputKind.FINAL_ONLY)
            else []
        )
        self.max_num_generation_tokens = 0
        self.cached_child_sampling_params = None

    # SOURCE: vllm/v1/engine/parallel_sampling.py:L52 (_get_child_sampling_params)
    def _get_child_sampling_params(self, index: int) -> SamplingParams:
        """Efficiently obtain child `sampling_params`.

        If `sampling_params.seed` is not `None` then each child request requires
        a unique clone of parent `sampling_params` with a unique seed.
        """
        seed = self.sampling_params.seed
        if self.cached_child_sampling_params:
            # Reuse child sampling_params data structure
            return self.cached_child_sampling_params
        # Build child sampling_params
        child_sampling_params = copy(self.sampling_params)
        child_sampling_params.n = 1
        if seed is None:
            # Cache child sampling_params for later reuse
            self.cached_child_sampling_params = child_sampling_params
        else:
            # Each child gets a clone with a unique seed
            child_sampling_params.seed = seed + index
        return child_sampling_params

    # SOURCE: vllm/v1/engine/parallel_sampling.py:L83 (get_child_info)
    def get_child_info(self, index: int) -> tuple[str, SamplingParams]:
        """Get child request ID and sampling params."""
        child_req_id = f"{index}_{self.request_id}"
        self.child_requests.add(child_req_id)
        return child_req_id, self._get_child_sampling_params(index)

    @property
    def n(self) -> int:
        # SOURCE: vllm/v1/engine/parallel_sampling.py:L96 (n)
        return self.sampling_params.n

    # SOURCE: vllm/v1/engine/parallel_sampling.py:L100 (get_outputs)
    def get_outputs(
        self,
        child_request_id: str,
        completion_output: CompletionOutput,
    ) -> tuple[list[CompletionOutput], bool]:
        already_finished_and_returned: bool = False
        if completion_output.finished():
            if child_request_id in self.child_requests:
                self.child_requests.remove(child_request_id)
            else:
                # child request ID is not available in child_requests
                # which means the request had finished in previous
                # batch step and returned to the client earlier
                already_finished_and_returned = True

        if self.sampling_params.output_kind != RequestOutputKind.FINAL_ONLY:
            # If streaming, just return the current output
            # DO NOT output finished and already returned child request again
            outputs = [] if already_finished_and_returned else [completion_output]
        else:
            # If not streaming, aggregate the n final outputs.
            self.output_aggregator[completion_output.index] = completion_output
            outputs = [] if self.child_requests else self.output_aggregator

        finished = not self.child_requests
        return outputs, finished

    # SUBTRACTED: observe_num_generation_tokens / observe_finished_request（L128-L151）
    #   —— metrics/IterationStats 统计旁路，不影响子请求 fan-out 与输出聚合主线。
    #   dossier 批准删。
