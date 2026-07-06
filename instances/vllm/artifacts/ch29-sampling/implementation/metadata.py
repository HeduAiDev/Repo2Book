# 只做减法的忠实精简版 —— 镜像 vllm/v1/sample/metadata.py（pin f3fef123）
# 与 vLLM 同名、同结构、同控制流；只删不增。
#
# SUBTRACTED: SPDX 版权头；`from ... thinking_budget_state import ThinkingBudgetStateHolder`
# import —— 仅服务被减掉的 thinking_budget 旁路字段。
from __future__ import annotations

from dataclasses import dataclass

import torch

from logits_processor import LogitsProcessors


@dataclass
class SamplingMetadata:
    # SOURCE: vllm/v1/sample/metadata.py:L14-55
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

    # SUBTRACTED: logprob_token_ids（generative_scoring API 专用旁路）、spec_token_ids
    # 与 thinking_budget_state_holder 三个字段（metadata.py:L46-55）。它们分别服务
    # 特定 token logprobs 旁路、投机解码、thinking 预算特性，均不在 9 步主流水线上；
    # subtraction_plan.delete 批准，且不在 must_keep。本精简版 sampler 不引用它们。
