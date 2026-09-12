# SOURCE: vllm/v1/engine/__init__.py
# 只做减法的忠实精简版：本章消费面 = FinishReason / EngineCoreEventType /
# EngineCoreEvent / EngineCoreRequest（core.preprocess_add_request 的输入件）
# / EngineCoreOutput（update_from_output 编译失败收账的回执件）。
# msgspec.Struct 基座一律退化 @dataclass（HOST SEAM——精简版无 msgspec
# 序列化面；字段与 docstring 逐字）。
# SUBTRACTED: SPDX 版权头；文件其余（EngineCoreReadyResponse/UtilityOutput/
# EngineCoreOutputs 的 DP wave 面等）。
import enum
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from vllm.sampling_params import SamplingParams
from vllm.pooling_params import PoolingParams


# SOURCE: vllm/v1/engine/__init__.py:L43-L64 FinishReason —— 逐字
class FinishReason(enum.IntEnum):
    """
    Reason a request finished - stop, length, abort, error, or repetition.

    Int rather than Str for more compact serialization.

    stop - a stop string was emitted
    length - max_tokens was consumed, or max_model_len was reached
    abort - aborted by client
    error - retryable request-level internal error (e.g., KV load failure).
            Invariant: always converted to 500 Internal Server Error.
    repetition - repetitive token pattern detected (hallucination)

    """

    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4

    def __str__(self):
        return FINISH_REASON_STRINGS[self.value]


# SUBTRACTED: FINISH_REASON_STRINGS 常量表（真实 vllm/v1/engine/__init__.py
#   的模块级 dict——精简版 __str__ 消费面未进测试；正文以枚举名口径引用）。
FINISH_REASON_STRINGS = {
    FinishReason.STOP: "stop",
    FinishReason.LENGTH: "length",
    FinishReason.ABORT: "abort",
    FinishReason.ERROR: "error",
    FinishReason.REPETITION: "repetition",
}


# SOURCE: vllm/v1/engine/__init__.py:L157-L161 EngineCoreEventType —— 逐字
class EngineCoreEventType(enum.IntEnum):
    """The type of engine core request event."""

    QUEUED = 1
    SCHEDULED = 2
    PREEMPTED = 3


# SOURCE: vllm/v1/engine/__init__.py:L165-L184 EngineCoreEvent —— 字段逐字
#   （msgspec.Struct → @dataclass）
@dataclass
class EngineCoreEvent:
    """A timestamped engine core event associated with a request.

    The timestamp is a monotonic timestamp and is used by the engine
    frontend to calculate intervals between engine core events. These
    timestamps should not be compared with timestamps from other processes.
    """

    type: EngineCoreEventType
    timestamp: float

    @classmethod
    def new_event(
        cls, event_type: EngineCoreEventType, timestamp: float | None = None
    ) -> "EngineCoreEvent":
        # SOURCE: vllm/v1/engine/__init__.py:L177-L184 —— 逐字
        timestamp = time.monotonic() if timestamp is None else timestamp
        return cls(event_type, timestamp)


# SOURCE: vllm/v1/engine/__init__.py:L96-L155 EngineCoreRequest —— 字段与
#   docstring 逐字（msgspec.Struct → @dataclass；HOST SEAM 增设 arrival_time
#   默认值，其余默认值原文）
@dataclass
class EngineCoreRequest:
    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: list | None
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    arrival_time: float = 0.0
    lora_request: Any = None
    cache_salt: str | None = None
    data_parallel_rank: int | None = None
    prompt_embeds: Any = None

    # Per-position mask for mixed-mode inputs (e.g chat completion with
    # prompt_embeds content parts). `True` means the position is a real
    # token ID; `False` means the position uses a pre-computed entry from
    # `prompt_embeds`. `None` for pure-tokens and pure-embeds inputs.
    prompt_is_token_ids: list[bool] | None = None

    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    client_index: int = 0

    # Used in DP case to indicate which wave of requests this is expected
    # to belong to, to cover a race condition where the request is sent before
    # a wave finished notification is received.
    current_wave: int = 0
    priority: int = 0

    trace_headers: Mapping[str, str] | None = None
    resumable: bool = False

    # The user-provided request ID. This field is set internally,
    # copied from the provided request_id that's originally assigned
    # to the request_id field, see InputProcessor.assign_request_id().
    # Used in outputs and to support abort(req_id, internal=False).
    external_req_id: str | None = None

    reasoning_ended: bool | None = None
    reasoning_parser_kwargs: dict[str, Any] | None = None

    # If True, the request should be added to the scheduler's waiting queue
    # and immediately aborted, so connector-side cleanup runs via the standard
    # request_finished hook. Used to free P-side prefill blocks when a
    # KV-transfer request is rejected on the D node before engine admission.
    abort_immediately: bool = False

    @property
    def params(self) -> "SamplingParams | PoolingParams":
        # SOURCE: vllm/v1/engine/__init__.py:L150-L154 —— 逐字
        """Return the processed params (sampling or pooling)."""
        if self.sampling_params is not None:
            return self.sampling_params
        assert self.pooling_params is not None
        return self.pooling_params


# SOURCE: vllm/v1/engine/__init__.py:L184-L226 EngineCoreOutput —— 字段逐字
#   （msgspec.Struct → @dataclass；本章消费位 = update_from_output 编译失败
#   收账 L1957-L1968 的空 token 回执）
@dataclass
class EngineCoreOutput:
    request_id: str
    new_token_ids: list[int]

    new_logprobs: Any = None
    new_prompt_logprobs_tensors: Any = None

    pooling_output: Any = None

    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    events: list[EngineCoreEvent] | None = None
    kv_transfer_params: dict[str, Any] | None = None
    ec_transfer_params: dict[str, Any] | None = None

    trace_headers: Mapping[str, str] | None = None

    prefill_stats: Any = None

    routed_experts: Any = None
    # The number of NaNs in logits.
    # A value greater than 0 indicates that the output is corrupted
    num_nans_in_logits: int = 0

    @property
    def finished(self) -> bool:
        # SOURCE: vllm/v1/engine/__init__.py:L222-L224 —— 逐字
        return self.finish_reason is not None
