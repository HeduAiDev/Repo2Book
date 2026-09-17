# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""请求载体：本章要点是 block_hashes / kv_transfer_params / all_token_ids 三个面。

# SOURCE: vllm/v1/request.py:L50-L340（Request.__init__ / RequestStatus）
# SUBTRACTED: 多模态特征/结构化输出状态机/流式会话/投机解码草稿 token——
#   本章的请求只带 token id、链式块哈希与 kv_transfer_params。
"""

import enum
import time
from typing import Any

from vllm.sampling_params import SamplingParams


# SOURCE: vllm/v1/request.py:L348-L375
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

    # SOURCE: vllm/v1/request.py:L366-L367
    def __str__(self) -> str:
        # SOURCE: vllm/v1/request.py:L366-L367
        return self.name

    # SOURCE: vllm/v1/request.py:L369-L371
    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        # SOURCE: vllm/v1/request.py:L369-L371
        return status > RequestStatus.PREEMPTED


# SOURCE: vllm/v1/request.py:L55-L190（Request.__init__）
class Request:
    # SOURCE: vllm/v1/request.py:L55-L190
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: Any | None = None,
        arrival_time: float | None = None,
    ) -> None:
        # SOURCE: vllm/v1/request.py:L55-L190
        self.request_id = request_id
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        self.status = RequestStatus.WAITING

        # Connector-specific KV transfer parameters（ch37 回执信封同一通道）。
        self.kv_transfer_params: dict[str, Any] | None = None

        if sampling_params is not None:
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )

        # 链式块哈希表：调度器侧 kv_cache_manager 随满块追加（ch15 的算法）。
        # 本章消费它切 chunk 键（scheduler.update_offload_keys）。
        self.block_hashes: list[Any] = []
        # 跳过外部前缀缓存读取的旋钮（--enable-prefix-caching 面的请求级旁路）。
        self.skip_reading_prefix_cache = False

        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = self._prompt_len()
        self._all_token_ids: list[int] = (
            list(self.prompt_token_ids) if self.prompt_token_ids is not None else []
        )
        # Number of computed tokens on this engine。
        self.num_computed_tokens = 0

    # SUBTRACTED: mm_features / lora 登记链（L104-L150）——本章 lora_request 恒 None
    #   （事件面的 lora 载荷字段保留、值恒空）。
    lora_request: Any | None = None

    # SOURCE: vllm/v1/request.py:L138-L148
    def _prompt_len(self) -> int:
        # SOURCE: vllm/v1/request.py:L138-L148
        if self.prompt_token_ids is not None:
            return len(self.prompt_token_ids)
        return 0

    # SOURCE: vllm/v1/request.py:L272-L274
    @property
    def num_tokens(self) -> int:
        # SOURCE: vllm/v1/request.py:L272-L274
        return len(self._all_token_ids)

    # SOURCE: vllm/v1/request.py:all_token_ids 条目（事件面/生态面按 token 切片）
    @property
    def all_token_ids(self) -> list[int]:
        # SOURCE: vllm/v1/request.py:all_token_ids 条目
        return self._all_token_ids

    # SOURCE: vllm/v1/request.py:L300-L340
    def is_finished(self) -> bool:
        # SOURCE: vllm/v1/request.py:L300-L340
        return RequestStatus.is_finished(self.status)


__all__ = ["Request", "RequestStatus"]
