# SOURCE: vllm/v1/sample/metadata.py
# ch29 契约载体：SamplingMetadata 冻结快照 dataclass（快照生产者
# InputBatch._make_sampling_metadata 归 ch18，本章只吃快照结果——m14）。
# SUBTRACTED：delete[2] spec_token_ids 字段（L51-L52，投机解码归 ch32/33——
#   非 spec 路径该字段恒 None、无消费者）；
#   delete[1] thinking budget 整链的字段位（L53-L55）与 import（L11）——
#   未设 reasoning_config 时 holder 恒 None、整块跳过，标准 9 步路径
#   逐字节不变（m7 登记轻讲、正文内嵌真源码，不依赖精简版）。
from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors

# SUBTRACTED: vllm/v1/sample/metadata.py:L11
#   `from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder`
#   —— delete[1] 连带 import 修剪（thinking budget 状态机文件整体删除）。


@dataclass
class SamplingMetadata:
    # SOURCE: vllm/v1/sample/metadata.py:L14-L49 SamplingMetadata —— 逐字
    temperature: torch.Tensor | None
    all_greedy: bool
    all_random: bool

    top_p: torch.Tensor | None
    top_k: torch.Tensor | None

    generators: dict[int, torch.Generator]

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: int | None

    no_penalties: bool
    prompt_token_ids: torch.Tensor | None
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

    output_token_ids: list[list[int]]

    # `allowed_token_ids_mask` is a 2D bool tensor of shape (max batch size,
    # vocab size).
    allowed_token_ids_mask: torch.Tensor | None

    # req_index -> bad_words_token_ids
    bad_words_token_ids: dict[int, list[list[int]]]

    # Loaded logits processors
    logitsprocs: LogitsProcessors

    # Specific token IDs to compute logprobs for (more efficient than full vocab)
    # When set, logprobs are computed only for these token IDs using gather
    # req_index -> list of token IDs to get logprobs for
    logprob_token_ids: dict[int, list[int]] | None = None

    # SUBTRACTED: vllm/v1/sample/metadata.py:L51-L52 spec_token_ids 字段
    #   —— delete[2]（投机解码组合分支归 ch32/33；非 spec 路径恒 None）。
    # SUBTRACTED: vllm/v1/sample/metadata.py:L53-L55
    #   thinking_budget_state_holder 字段 —— delete[1]（holder 恒 None 时
    #   整块跳过；字段与状态机文件一并删除）。
