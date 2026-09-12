# SOURCE: vllm/v1/request.py
# HOST SEAM：本章消费面一件——RequestStatus 枚举（update_from_output 语法
# 拒绝 → FINISHED_ERROR、is_finished 判据）。逐字（get_finished_reason 与
# _FINISHED_REASON_MAP 归 ch11 生命周期，删）。请求本体 Request 的字段面由
# 测试替身承载（装配归 ch02）。
from __future__ import annotations

import enum


# SOURCE: vllm/v1/request.py:L348-L371 RequestStatus —— 逐字
class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()

    # SOURCE: vllm/v1/request.py:L366-L367 RequestStatus.__str__ —— 逐字
    def __str__(self) -> str:
        return self.name

    @staticmethod
    # SOURCE: vllm/v1/request.py:L369-L371 RequestStatus.is_finished —— 逐字
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED
