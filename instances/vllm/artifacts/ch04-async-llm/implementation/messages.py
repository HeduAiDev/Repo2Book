"""跨进程边界的两种消息（精简版）。

本章只展示「流过 IPC 的是什么」——进入引擎方向的 EngineCoreRequest 与返回方向的
EngineCoreOutput。真实 vLLM 用 msgspec.Struct(array_like=True) 做紧凑零拷贝序列化跨进程
传输；精简版用 in-process 队列替代 IPC（见 engine_core_stub.py），故这里用普通 dataclass
即可，字段名/语义与真实 vLLM 一致。
"""

from __future__ import annotations

from dataclasses import dataclass


# SUBTRACTED: msgspec.Struct(array_like=True, omit_defaults=True, gc=False) 基类 —— 那是
#   为跨进程零拷贝序列化服务的；精简版用 in-process 队列传对象，无需序列化，故降为 dataclass。
#   语义不变：仍是「进入引擎方向、已 tokenize 的请求」。原 vllm/v1/engine/__init__.py:L80-L85
@dataclass
class EngineCoreRequest:  # SOURCE: vllm/v1/engine/__init__.py:L80 (class EngineCoreRequest)
    request_id: str
    # 已 tokenize 的 prompt —— Stage1(InputProcessor) 的产物。
    prompt_token_ids: list[int] | None
    # SUBTRACTED: mm_features/pooling_params/lora_request/cache_salt/data_parallel_rank/
    #   prompt_embeds/prompt_is_token_ids/client_index/current_wave/priority/trace_headers/
    #   resumable/external_req_id/reasoning_* 等次要字段 —— 与「三段式如何编排请求」正交。
    #   原 vllm/v1/engine/__init__.py:L88-L123
    sampling_params: SamplingParams | None
    arrival_time: float = 0.0


# SOURCE: vllm/v1/engine/__init__.py:L161 (class EngineCoreOutput)
# SUBTRACTED: msgspec.Struct 基类，同上理由。原 vllm/v1/engine/__init__.py:L161-L166
@dataclass
class EngineCoreOutput:
    request_id: str
    new_token_ids: list[int]
    # SUBTRACTED: new_logprobs/new_prompt_logprobs_tensors/pooling_output/stop_reason/events/
    #   kv_transfer_params/trace_headers/prefill_stats/routed_experts/num_nans_in_logits
    #   等次要字段 —— Stage3 去 tokenize/logprobs 细节留 ch08。
    #   原 vllm/v1/engine/__init__.py:L170-L187
    finish_reason: str | None = None

    @property
    # SOURCE: vllm/v1/engine/__init__.py:L189-L191
    def finished(self) -> bool:
        return self.finish_reason is not None


@dataclass
class EngineCoreOutputs:
    """EngineCore 一次产出的一批 outputs（跨进程边界的批容器）。

    真实 vLLM 还带 scheduler_stats/timestamp/engine_index 等；本章只需 outputs 列表。
    """

    # SOURCE: vllm/v1/engine/__init__.py:L206 (class EngineCoreOutputs)
    # SUBTRACTED: scheduler_stats/timestamp/engine_index/utility_results 等观测/控制字段。
    outputs: list[EngineCoreOutput]
    timestamp: float | None = None


# 下面两个仅作类型占位：精简版不依赖真实 SamplingParams 的内部字段，只用到 n 与 output_kind。
@dataclass
class SamplingParams:  # SOURCE: vllm/sampling_params.py (SamplingParams) — 本章只用 n/output_kind
    n: int = 1
    # DELTA 表示流式增量（消费者跟不上时 RequestOutputCollector 会 merge），
    # 否则为 FINAL/CUMULATIVE 语义。本章只用它选 aggregate。
    output_kind: str = "FINAL"
