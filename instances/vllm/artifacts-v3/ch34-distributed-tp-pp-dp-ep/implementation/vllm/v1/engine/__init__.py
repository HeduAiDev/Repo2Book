# SOURCE: vllm/v1/engine/__init__.py
# ch34 切面（m17/F3/F4）：EngineCoreRequest（client_index/current_wave 双字段）、
# EngineCoreOutputs（wave_complete/start_wave/-1 哨兵载荷）、EngineCoreRequestType
# （START_DP_WAVE 等字面量）。SUBTRACTED：EEP 通知族、EngineCoreEvent 观测、
# logprobs/pooling 字段族——各归其域。保留行对 pin 现核。

from __future__ import annotations

import enum
import time
from typing import Any, Mapping

import msgspec.msgpack
import torch

# SUBTRACTED: LogprobsLists/LogprobsTensors/PrefillStats/UtilityResult 等
#   依赖（ch08/ch29 域）——EngineCoreOutput 的字段子集相应收窄。


# SOURCE: vllm/v1/engine/__init__.py:L43-L65 FinishReason — 逐字（枚举面）
class FinishReason(enum.IntEnum):
    """
    Reason a request finished - stop, length, abort, error, or repetition.

    Int rather than Str for more compact serialization.
    """

    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4


# SOURCE: vllm/v1/engine/__init__.py:L97-L154 EngineCoreRequest —— 逐字（本章
#   双印章字段 client_index L122 / current_wave L127 原注释保留；多模态/
#   观测扩展字段按 SUBTRACTED 裁除）
class EngineCoreRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[Any] | None
    sampling_params: Any | None
    pooling_params: Any | None
    arrival_time: float
    lora_request: Any | None
    cache_salt: str | None
    data_parallel_rank: int | None
    # SUBTRACTED: prompt_embeds / prompt_is_token_ids（L112-L118）——多模态输入面。

    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    client_index: int = 0

    # Used in DP case to indicate which wave of requests this is expected to
    # belong to, to cover a race condition where the request is sent before
    # a wave finished notification is received.
    current_wave: int = 0
    priority: int = 0

    # SUBTRACTED: trace_headers/resumable/external_req_id/reasoning_ended 等
    #   （L130-L139）——观测与请求归属扩展面。


# SOURCE: vllm/v1/engine/__init__.py:L184-L215 EngineCoreOutput —— 字段子集
#   （finish_reason 与 events 观测面保留最小）
class EngineCoreOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    new_token_ids: list[int]

    pooling_output: torch.Tensor | None = None
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    # SUBTRACTED: new_logprobs/new_prompt_logprobs_tensors/events/
    #   kv_transfer_params/ec_transfer_params/trace_headers/prefill_stats/
    #   routed_experts/num_nans_in_logits（L192-L214）——各归其域。

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


# SOURCE: vllm/v1/engine/__init__.py:L218-L227 UtilityOutput —— 逐字（面收窄）
class UtilityOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    call_id: int

    # Non-None implies the call failed, result should be None.
    failure_message: str | None = None
    # SUBTRACTED: result: UtilityResult（L225）——ch09 域的 RPC 结果面。


# SOURCE: vllm/v1/engine/__init__.py:L230-L258 EngineCoreOutputs —— 逐字
#   （wave_complete/start_wave 两字段即 wave 控制面协议）
class EngineCoreOutputs(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    # NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout

    engine_index: int = 0

    # [num_reqs]
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: "SchedulerStats | None" = None
    timestamp: float = 0.0

    utility_output: UtilityOutput | None = None
    finished_requests: set[str] | None = None

    # In DP case, used to signal that the current wave of requests
    # has finished and the engines are paused.
    wave_complete: int | None = None
    # In DP case, used to signal that a request was received for an
    # "old" wave, so the next wave needs to be started in other engines.
    start_wave: int | None = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


# SOURCE: vllm/v1/engine/__init__.py:L261-L274 EngineCoreRequestType —— 逐字
#   （hex 字节字面量——不经编码直接过 socket）
class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """

    ADD = b"\x00"
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"
    # Sentinel used within EngineCoreProc.
    EXECUTOR_FAILED = b"\x04"
    # Sentinel to wake up input_queue.get() during shutdown.
    WAKEUP = b"\x05"


# SUBTRACTED: ReconfigureDistributedRequest / ReconfigureRankType /
#   EngineStatusType（L277-L303）——弹性 EP 与观测面。

from vllm.v1.metrics.stats import SchedulerStats  # noqa: E402,TID252
