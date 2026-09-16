# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""采样参数载体：回执信封上请求的落点就在这里（extra_args）。

# SOURCE: vllm/sampling_params.py:L200-L600（SamplingParams 字段族）
# SUBTRACTED: 全部采样语义（logprobs/beam/structured output 校验/序列化）——
#   本章只关心 `extra_args` 这条通路：HTTP extra_args → SamplingParams.extra_args
#   → Request 构造时摘出 kv_transfer_params（v1/request.py:L116-L119）。
"""

from dataclasses import dataclass, field
from typing import Any


# SOURCE: vllm/sampling_params.py:L199-L520
@dataclass
class SamplingParams:
    max_tokens: int | None = 16
    extra_args: dict[str, Any] | None = field(default=None)

    # SOURCE: vllm/sampling_params.py:L199-L520
    # SUBTRACTED: n/temperature/top_p 等取值校验——本章的 P/D 流程不依赖它们。
    # SOURCE: vllm/sampling_params.py:L560-L600
    def __post_init__(self) -> None:
        # SOURCE: vllm/sampling_params.py:L560-L600
        if self.max_tokens is None:
            raise ValueError("max_tokens must be set for generative requests")


__all__ = ["SamplingParams"]
