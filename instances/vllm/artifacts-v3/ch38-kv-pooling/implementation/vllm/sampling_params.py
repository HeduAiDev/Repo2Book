# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""采样参数载体（本章消费面：extra_args 携带 kv_transfer_params）。

# SOURCE: vllm/sampling_params.py:L1-L200（SamplingParams 字段族）
# SUBTRACTED: 温度/top_p/惩罚/结构化输出/种子等采样字段——本章只构造请求，
#   不采样。
"""

from typing import Any

from dataclasses import dataclass, field


# SOURCE: vllm/sampling_params.py:L60-L200
@dataclass
class SamplingParams:
    # SOURCE: vllm/sampling_params.py:L60-L200
    max_tokens: int = 16
    extra_args: dict[str, Any] | None = field(default=None)


__all__ = ["SamplingParams"]
