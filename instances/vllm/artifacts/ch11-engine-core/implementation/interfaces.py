# 只做减法的忠实精简版 —— 镜像 vllm/v1/engine/__init__.py、vllm/v1/serial_utils.py、
# vllm/v1/request.py、vllm/v1/core/sched/interface.py 中本章 step()/忙循环/生命周期用到的
# 数据类型（pin f3fef123）。
#
# 与 vLLM 同名、同字段；只删不增。
#
# SUBTRACTED: msgspec.Struct 基类（array_like/omit_defaults/gc=False 等编码选项）一律换成
#             dataclass —— 这些只影响跨 ZMQ 的序列化布局，不改字段语义；本章纯内存路径不发
#             socket。其余多模态/logprobs/stats 等与本章无关的字段一并删去。
import enum
from dataclasses import dataclass, field
from typing import Any, Literal

# SOURCE: vllm/v1/engine/__init__.py:L22-L26
# - "abort": Abort all in-flight requests immediately (default).
# - "wait": Wait for in-flight requests to complete before pausing.
# - "keep": Freeze requests in queue; they resume on resume_generation().
PauseMode = Literal["abort", "wait", "keep"]

# SOURCE: vllm/v1/engine/__init__.py:L30
FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")


# SOURCE: vllm/v1/engine/__init__.py:L42-L64
class FinishReason(enum.IntEnum):
    """Reason a request finished - stop, length, abort, error, or repetition."""

    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4

    def __str__(self):  # SOURCE: vllm/v1/engine/__init__.py:L63-L64
        return FINISH_REASON_STRINGS[self.value]


# SOURCE: vllm/v1/engine/__init__.py:L243-L256
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


# SOURCE: vllm/v1/core/sched/interface.py:L22-L33
class PauseState(enum.IntEnum):
    """Scheduler pause state.

    - UNPAUSED: Normal operation
    - PAUSE_NEW: No new requests are scheduled, requests already in
                 running state are scheduled.
    - PAUSE_ALL: No requests are scheduled
    """

    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2


# SOURCE: vllm/v1/request.py:L316-L329
class RequestStatus(enum.IntEnum):
    # SUBTRACTED: 其余状态枚举值（WAITING/RUNNING/FINISHED_STOPPED 等）——
    #             本章只用到 FINISHED_ABORTED（abort/pause 落地）。
    FINISHED_ABORTED = enum.auto()


# SOURCE: vllm/v1/serial_utils.py:L129
@dataclass
class UtilityResult:
    # SOURCE: vllm/v1/serial_utils.py:L129
    # SUBTRACTED: msgspec 序列化细节；保留"包一个 result 值"的语义。
    result: Any = None


# SOURCE: vllm/v1/engine/__init__.py:L167-L197
@dataclass
class EngineCoreOutput:
    # SOURCE: vllm/v1/engine/__init__.py:L167-L197
    request_id: str
    new_token_ids: list[int]
    # SUBTRACTED: new_logprobs / pooling_output / stop_reason / events /
    #             kv_transfer_params / trace_headers / prefill_stats / routed_experts /
    #             num_nans_in_logits（vllm/v1/engine/__init__.py:L176-L193）——
    #             logprobs/pooling/观测字段属其它章节，不参与 step 编排。
    finish_reason: FinishReason | None = None

    @property
    def finished(self) -> bool:  # SOURCE: vllm/v1/engine/__init__.py:L195-L197
        return self.finish_reason is not None


# SOURCE: vllm/v1/engine/__init__.py:L200-L209
@dataclass
class UtilityOutput:
    # SOURCE: vllm/v1/engine/__init__.py:L200-L209
    call_id: int
    # Non-None implies the call failed, result should be None.
    failure_message: str | None = None
    result: UtilityResult | None = None


# SOURCE: vllm/v1/engine/__init__.py:L212-L226
@dataclass
class EngineCoreOutputs:
    # SOURCE: vllm/v1/engine/__init__.py:L212-L226
    engine_index: int = 0
    # [num_reqs]
    outputs: list = field(default_factory=list)
    # SUBTRACTED: scheduler_stats / timestamp（vllm/v1/engine/__init__.py:L225-L226）。
    finished_requests: Any = None
    utility_output: UtilityOutput | None = None
