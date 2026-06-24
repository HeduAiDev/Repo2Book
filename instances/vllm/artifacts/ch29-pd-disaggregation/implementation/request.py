# SPDX-License-Identifier: Apache-2.0
# Subtract-only companion for ch29《PD 分离的抽象与调度器集成》.
# 只做减法：与 vLLM 同名/同结构/同控制流，只删不增。
#
# 本文件是 vllm/v1/request.py 的极小子集，仅保留本章调度集成需要的
# RequestStatus 状态机与 Request 的相关字段（num_computed_tokens /
# num_preemptions / status / prompt token 等）。
import enum


# SOURCE: vllm/v1/request.py:L310 (RequestStatus)
class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    # WAITING_FOR_REMOTE_KVS 是本章核心阻塞态：已分配 block 但 KV 还在远程传输中。
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    # SUBTRACTED: 其余 FINISHED_* 细分原因（LENGTH_CAPPED/ABORTED/IGNORED/ERROR/
    # REPETITION）与本章 KV-connector 调度集成无关，仅保留一个代表 is_finished
    # 判定即可（原 vllm/v1/request.py:L322-326）。

    # SOURCE: vllm/v1/request.py:L328
    def __str__(self) -> str:
        return self.name

    @staticmethod
    # SOURCE: vllm/v1/request.py:L332 (is_finished)
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    # SUBTRACTED: get_finished_reason / _FINISHED_REASON_MAP 是把 FINISHED_* 映射
    # 到对外 FinishReason 的纯展示逻辑，与远程 KV 调度无关（原 L335-352）。


# SOURCE: vllm/v1/request.py (Request — 仅保留本章字段)
class Request:
    """vLLM Request 的极小子集。

    # SUBTRACTED: 真实 Request 持有 sampling/pooling params、output_token_ids、
    # mm_features、lora_request、events、structured_output_request 等数十个字段
    # 与调度无关；这里只保留 KV-connector 调度路径触碰到的：request_id、
    # prompt_token_ids、status、num_computed_tokens、num_preemptions。
    """

    # SOURCE: vllm/v1/request.py (Request.__init__ — 仅本章字段)
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        status: RequestStatus = RequestStatus.WAITING,
    ) -> None:
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.status = status
        # num_computed_tokens：本地+远程命中的已计算 token 数。
        self.num_computed_tokens = 0
        # num_preemptions：被抢占次数，提升时据此区分回 WAITING 还是 PREEMPTED。
        self.num_preemptions = 0

    @property
    # SOURCE: vllm/v1/request.py (Request.num_tokens)
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids)
