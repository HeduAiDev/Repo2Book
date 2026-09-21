# SOURCE: vllm/v1/sample/metadata.py
# ch34 HOST SEAM（沿 ch30 镜像 + 本章消费面回补）：SamplingMetadata 冻结快照
# dataclass（快照生产者 InputBatch._make_sampling_metadata 归 ch18，本章只吃
# 快照结果）。ch34 消费面回补 spec_token_ids 字段（RejectionSampler 的
# 惩罚/bad_words 吃『草稿当已输出』的历史）与 thinking_budget_state_holder
# 字段（本章 dossier delete[2] 删的是 RejectionSampler 侧 holder 调用块，
# 字段本身保留——Sampler 镜像的 bonus 位路径按名引用它、恒 None 跳过）。
# SUBTRACTED：沿 ch30 镜像的 delete[0]——logprob_token_ids 旁路的 gather 面
#   本章 bonus 调用 replace(max_num_logprobs=-1) 不进（字段保留、恒 None）。
# SUBTRACTED: vllm/v1/sample/metadata.py:L11
#   `from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder`
#   —— thinking budget 状态机文件不载（归 ch30 语境）；字段注解在
#   `from __future__ import annotations` 下为惰性字符串，运行期不求值。
from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors


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

    # SOURCE: vllm/v1/sample/metadata.py:L51-L52 spec_token_ids 字段 —— 逐字
    #   （ch34 消费面回补：惩罚/bad_words 的草稿前缀历史）
    # Speculative token ids
    spec_token_ids: list[list[int]] | None = None
    # SOURCE: vllm/v1/sample/metadata.py:L53-L55 thinking_budget_state_holder
    #   字段 —— 逐字（holder 恒 None 时整块跳过；本章无 thinking 消费位）
    # When non-None, use ``holder.has_tracked_requests()`` to see if this batch applies
    # thinking-token-budget logits (holder may exist with an empty tracking set).
    thinking_budget_state_holder: "ThinkingBudgetStateHolder | None" = None
