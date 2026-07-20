# SOURCE: vllm/v1/request.py
# 只做减法的忠实精简版——只保留与结构化输出相关的切面：RequestStatus 枚举、
# Request.__init__ 里挂 StructuredOutputRequest 并置初始阻塞态、use_structured_output
# 属性。真实 Request 类另有 KV cache block / spec token / 多模态等几十个字段与方法，
# 属调度章（ch13）与投机解码章范围，均不在本章 must_keep 内，一并省略
# （subtraction_plan 未逐条列出——因为它们本就不在 code_spine/must_keep 覆盖范围，
# 而是"未被选入这一章的切面"，不是需要单独批准的删除项）。
import enum
import time
from typing import Any

from so_request import StructuredOutputRequest


class RequestStatus(enum.IntEnum):
    # SOURCE: vllm/v1/request.py:L316-321（枚举其余成员——PREEMPTED/FINISHED_* 一族——
    # 已在调度章 ch13 讲过，非本章 must_keep，省略）。
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()


class Request:
    # SOURCE: vllm/v1/request.py:L60-121（__init__，精简到结构化输出相关字段）
    #
    # SUBTRACTED: pooling_params / lora_request / arrival_time 精确来源 / events /
    # stop_reason / kv_transfer_params / block hashing 等，与结构化输出主控制流无关，
    # 已省略（真实类另见 ch05/ch13）。这里只保留 max_tokens 判定生成模型的分支，
    # 因为 status 的初始置位就发生在这个分支里。
    def __init__(
        self,
        request_id: str,
        sampling_params: Any | None = None,
    ) -> None:
        self.request_id = request_id
        self.sampling_params = sampling_params
        # SOURCE: vllm/v1/request.py:L87-89
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        self.arrival_time = time.time()

        # SOURCE: vllm/v1/request.py:L97
        self.status = RequestStatus.WAITING

        if sampling_params is not None:
            # SOURCE: vllm/v1/request.py:L107-112 —— Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

    @property
    def use_structured_output(self) -> bool:
        # SOURCE: vllm/v1/request.py:L236-238
        return self.structured_output_request is not None
