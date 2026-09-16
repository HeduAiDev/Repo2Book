# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""引擎入参结构：拒绝回执的占位请求就是从这里构造的。

# SOURCE: vllm/v1/engine/__init__.py:L60-L160（EngineCoreRequest / FinishReason）
# SUBTRACTED: 多模态特征/显存报告/DP 归属/结构化输出等字段——本章的占位请求
#   只需要 request_id + prompt + sampling_params + abort_immediately。
"""

import enum
import time
from dataclasses import dataclass
from typing import Any

from vllm.sampling_params import SamplingParams


# SOURCE: vllm/v1/engine/__init__.py:L30-L60
class FinishReason(enum.IntEnum):
    """Reason a request finished."""

    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4


# SOURCE: vllm/v1/engine/__init__.py:L60-L120
@dataclass
class EngineCoreRequest:
    request_id: str
    prompt_token_ids: list[int] | None
    sampling_params: SamplingParams | None
    pooling_params: Any | None = None
    arrival_time: float = 0.0
    mm_features: list[Any] | None = None
    lora_request: Any | None = None
    cache_salt: str | None = None
    data_parallel_rank: int | None = None
    prompt_embeds: Any | None = None
    # 预 abort：请求从未被调度，但 request_finished 钩子照样跑
    # （vllm/v1/engine/core.py:L479-L483）。
    abort_immediately: bool = False

    # SOURCE: vllm/v1/engine/__init__.py:L150-L160（__post_init__）
    def __post_init__(self) -> None:
        if not self.arrival_time:
            self.arrival_time = time.time()


# SOURCE: vllm/v1/engine/__init__.py:L157-L166（EngineCoreEventType）
class EngineCoreEventType(enum.IntEnum):
    """引擎内部事件类型（本章不产事件流，保留类型位）。"""

    # SOURCE: vllm/v1/engine/__init__.py:L157-L166
    QUEUED = 1
    SCHEDULED = 2
    PREEMPTED = 3


# SOURCE: vllm/v1/engine/__init__.py:L184-L214（EngineCoreOutput）
@dataclass
class EngineCoreOutput:
    """P 终局把 connector 产的回执挂在这里随响应体出引擎。"""

    # SOURCE: vllm/v1/engine/__init__.py:L184-L214
    request_id: str
    new_token_ids: list[int]

    new_logprobs: Any | None = None
    new_prompt_logprobs_tensors: Any | None = None
    pooling_output: Any | None = None
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    events: list[Any] | None = None
    kv_transfer_params: dict[str, Any] | None = None
    ec_transfer_params: dict[str, Any] | None = None

    # SOURCE: vllm/v1/engine/__init__.py:L212-L213
    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


__all__ = ["EngineCoreEventType", "EngineCoreOutput", "EngineCoreRequest", "FinishReason"]
