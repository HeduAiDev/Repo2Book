# SUBTRACTED: SPDX 版权头（vllm/v1/sample/metadata.py:L1-L2）。
# 本章的 rejection sampling 只用到 SamplingMetadata 的采样调度子集（温度/全贪心/
# 全随机/top-k/top-p/per-request generators）。完整 SamplingMetadata 见 ch27。
from __future__ import annotations

from dataclasses import dataclass

import torch


# SOURCE: vllm/v1/sample/metadata.py:L14-L55
# SUBTRACTED: max_num_logprobs / no_penalties / prompt_token_ids / *_penalties /
#             output_token_ids / allowed_token_ids_mask / bad_words_token_ids /
#             logitsprocs / logprob_token_ids / spec_token_ids /
#             thinking_budget_state_holder —— 它们服务 logprobs/penalties/bad_words/
#             thinking 等本章已减掉的旁路（见 rejection_sampler.py 的 SUBTRACTED 注释）。
#             rejection_sample 与两个 kernel 只读以下字段。
@dataclass
class SamplingMetadata:
    # SOURCE: vllm/v1/sample/metadata.py:L14-L55
    # [batch_size]：每请求温度；==0 表示该请求贪心（GREEDY_TEMPERATURE）。
    temperature: torch.Tensor | None
    all_greedy: bool
    all_random: bool

    # [batch_size] or None
    top_p: torch.Tensor | None
    top_k: torch.Tensor | None

    # req_index -> torch.Generator（带种子的请求用于可复现的 uniform/exponential 采样）。
    generators: dict[int, torch.Generator]
