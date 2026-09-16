# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""请求载体：本章要点是 `kv_transfer_params` 如何随请求进引擎。

# SOURCE: vllm/v1/request.py:L50-L340（Request.__init__ / RequestStatus）
# SUBTRACTED: 多模态特征/结构化输出/流式会话/投机解码草稿 token/块哈希回调——
#   本章的 P/D 请求只带 token id 与 kv_transfer_params。
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
    # SOURCE: vllm/v1/request.py:L55-L190（Request.__init__）
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: Any | None = None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: Any | None = None,
        abort_immediately: bool = False,
    ) -> None:
        # SOURCE: vllm/v1/request.py:L55-L190（Request.__init__）
        self.request_id = request_id
        self.client_index = client_index
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        self.status = RequestStatus.WAITING
        self.events: list[Any] = []
        self.stop_reason: int | str | None = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None
        # E/P/D: Connector-specific encoder-cache transfer parameters.
        self.ec_transfer_params: dict[str, Any] | None = None
        # SUBTRACTED: StructuredOutputRequest.from_sampling_params（L90-L96）——
        #   结构化输出状态机属 ch31/ch32；本字段恒 None，下面那条分支永不触发。
        self.structured_output_request = None

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
                self.ec_transfer_params = sampling_params.extra_args.get(
                    "ec_transfer_params"
                )
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        # SUBTRACTED: mm_features / lora_request / cache_salt / trace_headers /
        #   block_hasher 的登记（L104-L150）——connector 面不读它们。
        self.abort_immediately = abort_immediately

        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.num_prompt_tokens = self._prompt_len()
        self._all_token_ids: list[int] = (
            list(self.prompt_token_ids) if self.prompt_token_ids is not None else []
        )
        # Number of computed tokens on this engine（P/D 的交接坐标就是它）。
        self.num_computed_tokens = 0

    # SOURCE: vllm/v1/request.py:L223-L247（from_engine_core_request）
    # SUBTRACTED: mm_features / prompt_is_token_ids / lora / cache_salt / 优先级 /
    #   block_hasher / 流式会话字段——本章的请求只有 token 与 kv_transfer_params。
    # SOURCE: vllm/v1/request.py:L223-L247（from_engine_core_request）
    @classmethod
    def from_engine_core_request(
        cls,
        request: Any,
        block_hasher: Any | None,
    ) -> "Request":
        # SOURCE: vllm/v1/request.py:L223-L247（from_engine_core_request）
        return cls(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            abort_immediately=request.abort_immediately,
        )

    # SOURCE: vllm/v1/request.py:L138-L148（num_prompt_tokens 的求值位）
    def _prompt_len(self) -> int:
        if self.prompt_token_ids is not None:
            return len(self.prompt_token_ids)
        if self.prompt_embeds is not None:
            return int(self.prompt_embeds.shape[0])
        return 0

    # SOURCE: vllm/v1/request.py:L272-L274
    @property
    def num_tokens(self) -> int:
        # SOURCE: vllm/v1/request.py:L272-L274
        return len(self._all_token_ids)

    # SOURCE: vllm/v1/request.py:L300-L340（is_finished / get_finished_reason）
    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)


__all__ = ["Request", "RequestStatus"]
