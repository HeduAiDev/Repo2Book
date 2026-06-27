# SPDX-License-Identifier: Apache-2.0
# 只做减法的忠实子集 —— 对应 vllm/v1/engine/__init__.py（支撑类型）。
# 源码 pin: f3fef123。本模块只保留本章 IPC 边界触及的 enum / msgspec 结构体，
# 字段为真实结构体的子集（# SUBTRACTED 标注删去的字段），名字/类型/语义 1:1。

import enum
import time
from dataclasses import dataclass
from typing import Any

import msgspec

from serial_utils import UtilityResult


# SOURCE: vllm/v1/engine/__init__.py:L243  EngineCoreRequestType
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


@dataclass
# SOURCE: vllm/v1/engine/__init__.py:L67  EngineCoreReadyResponse
class EngineCoreReadyResponse:
    """Sent from EngineCore to each frontend at the end of engine startup.

    Contains post-initialization config that may differ from the original
    values (e.g. max_model_len after KV cache auto-fitting).
    """

    max_model_len: int
    num_gpu_blocks: int
    dp_stats_address: str | None = None


# SOURCE: vllm/v1/engine/__init__.py:L80  EngineCoreRequest
class EngineCoreRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    # SUBTRACTED: mm_features / sampling_params / pooling_params / lora_request /
    # cache_salt / arrival_time / data_parallel_rank / prompt_is_token_ids /
    # current_wave / priority / trace_headers / resumable / external_req_id /
    # reasoning_* —— 这些字段是请求语义负载，与 ZMQ 多帧/字节标签传输协议无关，
    # 删去后 ADD 请求仍能完整编解码、跨"进程"投递。原 vllm/v1/engine/__init__.py:L88-L123。
    prompt_token_ids: list[int] | None = None
    # 大张量字段：演示 _encode_tensor 的 aux_buffers / OOB 两条零拷贝分支。
    prompt_embeds: Any | None = None  # torch.Tensor | None，宽松标注便于 array_like 解码
    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    client_index: int = 0


# SOURCE: vllm/v1/engine/__init__.py:L200  UtilityOutput
class UtilityOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    call_id: int

    # Non-None implies the call failed, result should be None.
    failure_message: str | None = None
    result: UtilityResult | None = None


# SOURCE: vllm/v1/engine/__init__.py:L167  EngineCoreOutput
class EngineCoreOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    new_token_ids: list[int]
    # SUBTRACTED: logprobs / pooling_output / finish_reason / events /
    # kv_transfer_params / trace_headers / prefill_stats / routed_experts /
    # num_nans_in_logits —— 输出语义负载，与跨进程多帧传输无关。
    # 原 vllm/v1/engine/__init__.py:L176-L193。


# SOURCE: vllm/v1/engine/__init__.py:L212  EngineCoreOutputs
class EngineCoreOutputs(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    engine_index: int = 0

    # [num_reqs]
    outputs: list[EngineCoreOutput] = []
    # SUBTRACTED: scheduler_stats（统计旁支，非 IPC 主线）。原 :L219。
    timestamp: float = 0.0

    utility_output: UtilityOutput | None = None
    # SUBTRACTED: finished_requests / wave_complete / start_wave（DP 编排，另章）。
    # 原 vllm/v1/engine/__init__.py:L229-L236。

    def __post_init__(self):
        # SOURCE: vllm/v1/engine/__init__.py:L238  EngineCoreOutputs.__post_init__
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()
