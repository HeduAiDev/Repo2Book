"""本章用到的最小消息/参数/输出数据结构（站位真实 vLLM 类型）。

这些类型在 ch05/ch06（EngineCoreRequest 装配）、ch08-ch10（RequestOutput 装配/去 token 化）
已细讲，本章不重讲——这里只放够 LLM facade 同步主干跑起来的字段子集，让读者能数值追踪
'渲染→双注册→while step()→排序还原'整条流水。
"""
# SUBTRACTED: 真实 EngineCoreRequest/RequestOutput/PoolingRequestOutput 的完整字段
#   （mm_inputs/lora/sampling 全量参数/logprobs/metrics 等）——ch05/ch06/ch08-ch10 已细讲。
#   本章只保留 request_id + 终止/finished 语义，够体现同步驱动主干即可。
#   原 vllm/v1/engine/__init__.py(EngineCoreRequest) / vllm/outputs.py(RequestOutput)。

from __future__ import annotations

import enum
from dataclasses import dataclass, field


# SOURCE: vllm/sampling_params.py (RequestOutputKind)
class RequestOutputKind(enum.Enum):
    # SUBTRACTED: CUMULATIVE / DELTA 两个流式取值的语义说明——ch04/ch08 流式侧已讲。
    #   本章只需 FINAL_ONLY 这一离线取值（_add_request 强设它）。原 vllm/sampling_params.py。
    CUMULATIVE = 0
    DELTA = 1
    FINAL_ONLY = 2


@dataclass
class SamplingParams:
    # SOURCE: vllm/sampling_params.py (SamplingParams)
    # SUBTRACTED: temperature/top_p/max_tokens/stop/logprobs 等数十个采样字段——ch06 已细讲。
    #   本章只需 n（并行采样扇出）与 output_kind（被 _add_request 强设 FINAL_ONLY）两个字段。
    #   原 vllm/sampling_params.py。
    n: int = 1
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE
    max_tokens: int = 3


@dataclass
class PoolingParams:
    # SOURCE: vllm/pooling_params.py (PoolingParams)
    # SUBTRACTED: dimensions/normalize/activation 等 pooling 字段——pooling 家族细节非本章主线。
    #   原 vllm/pooling_params.py。
    task: str | None = None


@dataclass
class EngineCoreRequest:
    # SOURCE: vllm/v1/engine/__init__.py (EngineCoreRequest)
    request_id: str
    params: SamplingParams | PoolingParams
    # SUBTRACTED: prompt_token_ids/mm_inputs/lora_request/arrival_time/priority 等——ch05/ch06。
    #   原 vllm/v1/engine/__init__.py:EngineCoreRequest。


@dataclass
class EngineCoreOutput:
    # SOURCE: vllm/v1/engine/__init__.py (EngineCoreOutput)
    request_id: str
    new_token_ids: list[int] = field(default_factory=list)
    finished: bool = False


@dataclass
class EngineCoreOutputs:
    # SOURCE: vllm/v1/engine/__init__.py (EngineCoreOutputs)
    # SUBTRACTED: scheduler_stats/timestamp/utility_output/wave_complete 等——ch07/可观测性旁路。
    #   本章 step 第 2/4 步只用到 outputs 列表。原 vllm/v1/engine/__init__.py:EngineCoreOutputs。
    outputs: list[EngineCoreOutput] = field(default_factory=list)


@dataclass
class RequestOutput:
    # SOURCE: vllm/outputs.py (RequestOutput)
    # SUBTRACTED: prompt/prompt_token_ids/CompletionOutput 列表/logprobs/metrics——ch08-ch10 已讲。
    #   本章只需 request_id + finished（_run_engine 收集 finished 并按 request_id 排序）。
    #   原 vllm/outputs.py:RequestOutput。
    request_id: str
    finished: bool = False
    text: str = ""


@dataclass
class PoolingRequestOutput:
    # SOURCE: vllm/outputs.py (PoolingRequestOutput)
    # SUBTRACTED: data（pooled hidden states 张量）字段——pooling 装配非本章主线。
    #   原 vllm/outputs.py:PoolingRequestOutput。
    request_id: str
    finished: bool = False
    data: object = None


@dataclass
class EmbeddingRequestOutput:
    # SOURCE: vllm/outputs.py (EmbeddingRequestOutput)
    request_id: str
    finished: bool = False
    embedding: object = None

    @classmethod
    def from_base(cls, output: PoolingRequestOutput) -> "EmbeddingRequestOutput":
        # SOURCE: vllm/outputs.py (EmbeddingRequestOutput.from_base)
        # SUBTRACTED: 真实从 PoolingRequestOutput.data 抽 embedding 向量并做 dtype/shape 校验。
        #   原 vllm/outputs.py:EmbeddingRequestOutput.from_base。本章只体现 embed=encode 的薄封装。
        return cls(request_id=output.request_id, finished=output.finished,
                   embedding=output.data)
