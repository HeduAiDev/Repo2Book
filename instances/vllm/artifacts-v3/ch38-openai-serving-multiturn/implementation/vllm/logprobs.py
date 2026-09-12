# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/logprobs.py —— HOST SEAM（最小承载）：本章消费面（outputs.py
# 的 RequestOutput 字段、base/serving.py 的 clamp_prompt_logprobs）只需要
# Logprob dataclass 与两个类型别名；FlatLogprobs 全量实现归 ch8（logprobs
# 域，dossier delete[1] 已把 logprobs 组装全家从本章精简版删去）。
from dataclasses import dataclass


# SOURCE: vllm/logprobs.py:L13-L24
@dataclass
class Logprob:
    """Infos for supporting OpenAI compatible logprobs and token ranks.

    Attributes:
        logprob: The logprob of chosen token
        rank: The vocab rank of chosen token (>=1)
        decoded_token: The decoded chosen token index
    """

    logprob: float
    rank: int | None = None
    decoded_token: str | None = None


# SOURCE: vllm/logprobs.py:L27
LogprobsOnePosition = dict[int, Logprob]

# SUBTRACTED: vllm/logprobs.py:L30-L155 FlatLogprobs 全量实现（ch8 域，
# 本章 delete[1] logprobs 组装全家）。

# SOURCE: vllm/logprobs.py:L157
PromptLogprobs = list[LogprobsOnePosition | None]

# SOURCE: vllm/logprobs.py:L159
SampleLogprobs = list[LogprobsOnePosition]
