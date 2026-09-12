# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/v1/engine/__init__.py —— 忠实承载（消费面子集）：
# PauseMode / FinishReason / EngineCoreRequest 逐字——本章的过线载体
# （serving 层 reasoning_ended 盖进 EngineCoreRequest，abort 投
# FinishReason.ABORT 终态）。EngineCoreOutput(s)/UtilityOutput/
# Reconfigure*/EEP 通知族删——ch5 ZMQ 协议域。
import enum
import time
from collections.abc import Mapping
from typing import Any, Literal

import msgspec
import torch

from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams

# SOURCE: vllm/v1/engine/__init__.py:L27-L31 —— PauseMode 逐字
# Type for pause_generation mode parameter.
# - "abort": Abort all in-flight requests immediately (default).
# - "wait": Wait for in-flight requests to complete before pausing.
# - "keep": Freeze requests in queue; they resume on resume_generation().
PauseMode = Literal["abort", "wait", "keep"]

# SOURCE: vllm/v1/engine/__init__.py:L30-L31
# These are possible values of RequestOutput.finish_reason,
# so form part of the external API.
FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")

# SUBTRACTED: vllm/v1/engine/__init__.py:L33-L95 EEP_NOTIFICATION_CALL_ID /
# FT_STATUS_CALL_ID / EEPNotificationType / EngineCoreReadyResponse——
# ZMQ 跨进程通知族（ch5 域）。


# SOURCE: vllm/v1/engine/__init__.py:L43-L65 —— FinishReason 逐字
class FinishReason(enum.IntEnum):
    """
    Reason a request finished - stop, length, abort, error, or repetition.

    Int rather than Str for more compact serialization.

    stop - a stop string was emitted
    length - max_tokens was consumed, or max_model_len was reached
    abort - aborted by client
    error - retryable request-level internal error (e.g. KV load failure).
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


# SOURCE: vllm/v1/engine/__init__.py:L97-L154 —— EngineCoreRequest 逐字
class EngineCoreRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec] | None
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    arrival_time: float
    lora_request: LoRARequest | None
    cache_salt: str | None
    data_parallel_rank: int | None
    prompt_embeds: torch.Tensor | None = None

    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    client_index: int = 0

    # Used in DP case to indicate which wave of requests this is expected to
    # belong to, to cover a race condition where the request is sent before
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

    # SOURCE: vllm/v1/engine/__init__.py:L148-L154 —— params 属性逐字
    @property
    def params(self) -> SamplingParams | PoolingParams:
        """Return the processed params (sampling or pooling)."""
        if self.sampling_params is not None:
            return self.sampling_params
        assert self.pooling_params is not None
        return self.pooling_params


# SUBTRACTED: vllm/v1/engine/__init__.py:L157-L181 EngineCoreEvent 族；
# L184-L258 EngineCoreOutput/UtilityOutput/EngineCoreOutputs；
# L261-L299 EngineCoreRequestType/ReconfigureDistributedRequest/
# ReconfigureRankType/EngineStatusType——ZMQ 消息族（ch5 域，本章单进程
# 精简环境不跨进程传消息）。
