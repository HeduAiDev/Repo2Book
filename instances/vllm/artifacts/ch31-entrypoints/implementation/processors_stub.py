"""InputProcessor / OutputProcessor 的 in-process 替身。

这两层在前序章已细讲，本章不重讲：
  * InputProcessor.process_inputs（prompt→EngineCoreRequest）—— ch05/ch06。
  * OutputProcessor.process_outputs（EngineCoreOutput→去 token 化→装配 RequestOutput）—— ch08-ch10。

本章焦点是 LLMEngine.step() 把它们串进同步主干的【位置】，所以这里只放够主干跑通的薄替身：
process_inputs 把 (request_id, prompt, params) 包成 EngineCoreRequest；OutputProcessor 维护
RequestState 计未完成数，并在收到 finished 的 EngineCoreOutput 时装配 RequestOutput。
"""
# SUBTRACTED: 真实 InputProcessor 的 renderer 调用/多模态/tokenization/lora 装配（ch05/ch06），
#   OutputProcessor 的增量去 token 化/logprobs/stop string 检测/stats（ch08-ch10）。
#   本章只保留'每请求建 RequestState、finished 时产出 RequestOutput、计未完成数'三件事。
#   原 vllm/v1/engine/input_processor.py / vllm/v1/engine/output_processor.py。

from __future__ import annotations

from dataclasses import dataclass

from messages import (
    EngineCoreOutput,
    EngineCoreRequest,
    PoolingParams,
    PoolingRequestOutput,
    RequestOutput,
    SamplingParams,
)


class InputProcessor:
    # SOURCE: vllm/v1/engine/input_processor.py (InputProcessor) — ch05/ch06 stub
    def __init__(self, vllm_config=None, renderer=None):
        pass

    def process_inputs(self, request_id, prompt, params, **kwargs) -> EngineCoreRequest:
        # SOURCE: vllm/v1/engine/input_processor.py:process_inputs (ch05/ch06 stub)
        # 真实版经 renderer 把 prompt 渲染为 token + 装配 EngineCoreRequest（多模态/lora/采样克隆）。
        return EngineCoreRequest(request_id=request_id, params=params)

    def assign_request_id(self, request: EngineCoreRequest) -> None:
        # SOURCE: vllm/v1/engine/input_processor.py:assign_request_id (ch05/ch06 stub)
        return None


@dataclass
class _ProcessedOutputs:
    # SOURCE: vllm/v1/engine/output_processor.py:OutputProcessorOutput — ch08-ch10 stub
    request_outputs: list
    reqs_to_abort: list


# (above) _ProcessedOutputs 站位真实 OutputProcessorOutput（request_outputs + reqs_to_abort）。


class _RequestState:
    # SOURCE: vllm/v1/engine/output_processor.py:RequestState — ch08-ch10 stub
    def __init__(self, request_id: str, is_pooling: bool):
        # SOURCE: vllm/v1/engine/output_processor.py:RequestState.__init__ — stub
        self.request_id = request_id
        self.is_pooling = is_pooling
        self.finished = False


class OutputProcessor:
    # SOURCE: vllm/v1/engine/output_processor.py (OutputProcessor) — ch08-ch10 stub
    def __init__(self, tokenizer=None, log_stats: bool = False, stream_interval: int = 1,
                 tracing_enabled: bool = False):
        self.request_states: dict[str, _RequestState] = {}

    # SOURCE: vllm/v1/engine/output_processor.py:OutputProcessor.add_request (ch08-ch10 stub)
    def add_request(self, request: EngineCoreRequest, prompt_text, parent_req, index):
        # 真实版建 RequestState（持去 token 化/增量装配状态）。stub 只记 request_id + 是否 pooling。
        is_pooling = isinstance(request.params, PoolingParams)
        self.request_states[request.request_id] = _RequestState(
            request.request_id, is_pooling)

    # SOURCE: vllm/v1/engine/output_processor.py:OutputProcessor.process_outputs (ch08-ch10 stub)
    def process_outputs(self, engine_core_outputs: list[EngineCoreOutput],
                        engine_core_timestamp=None, iteration_stats=None) -> _ProcessedOutputs:
        # 真实版逐 EngineCoreOutput 做增量去 token 化、检 stop string、装配 RequestOutput。
        # stub：finished 的 EngineCoreOutput → 产出对应类型的 *RequestOutput 并标 finished。
        request_outputs: list = []
        for eco in engine_core_outputs:
            state = self.request_states.get(eco.request_id)
            if state is None:
                continue
            if eco.finished:
                state.finished = True
                if state.is_pooling:
                    request_outputs.append(
                        PoolingRequestOutput(request_id=eco.request_id, finished=True))
                else:
                    request_outputs.append(
                        RequestOutput(request_id=eco.request_id, finished=True))
                self.request_states.pop(eco.request_id, None)
        # SUBTRACTED: reqs_to_abort 由 stop string 检测填充（ch09/ch10）；stub 无 stop string → 恒空。
        return _ProcessedOutputs(request_outputs=request_outputs, reqs_to_abort=[])

    def update_scheduler_stats(self, scheduler_stats) -> None:
        # SOURCE: vllm/v1/engine/output_processor.py:update_scheduler_stats (stub no-op)
        return None

    # SOURCE: vllm/v1/engine/output_processor.py:get_num_unfinished_requests (ch08-ch10 stub)
    def get_num_unfinished_requests(self) -> int:
        return len(self.request_states)

    # SOURCE: vllm/v1/engine/output_processor.py:has_unfinished_requests (ch08-ch10 stub)
    def has_unfinished_requests(self) -> bool:
        return len(self.request_states) > 0

    # SOURCE: vllm/v1/engine/output_processor.py:abort_requests (ch08-ch10 stub)
    def abort_requests(self, request_ids: list[str], internal: bool = False) -> list[str]:
        for rid in request_ids:
            self.request_states.pop(rid, None)
        return list(request_ids)


class ParentRequest:
    # SOURCE: vllm/v1/engine/parallel_sampling.py (ParentRequest) — ch04/ch06 stub
    """n>1 并行采样扇出（ch04/ch06 已讲），本章只点其在 add_request 中被复用。"""

    def __init__(self, request: EngineCoreRequest):
        # SOURCE: vllm/v1/engine/parallel_sampling.py:ParentRequest.__init__ — stub
        self.request = request

    def get_child_info(self, idx: int):
        # SOURCE: vllm/v1/engine/parallel_sampling.py:ParentRequest.get_child_info (stub)
        # 真实版给第 idx 个子请求分配 child request_id + 克隆/微调 SamplingParams（seed 等）。
        child_id = f"{self.request.request_id}_{idx}"
        child_params = SamplingParams(
            n=1,
            output_kind=self.request.params.output_kind
            if isinstance(self.request.params, SamplingParams)
            else None,
            max_tokens=getattr(self.request.params, "max_tokens", 1),
        )
        return child_id, child_params
