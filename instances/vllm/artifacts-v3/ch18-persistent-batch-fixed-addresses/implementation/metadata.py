# SOURCE: vllm/v1/sample/metadata.py
# 本章消费面：SamplingMetadata——InputBatch.refresh_metadata 的产出类型
# （_make_sampling_metadata 装配；_sample 消费）。全文逐字（纯 dataclass）。
from __future__ import annotations

from dataclasses import dataclass

import torch

from .logits_processor import LogitsProcessors
from ._host_seams import ThinkingBudgetStateHolder


# SOURCE: vllm/v1/sample/metadata.py:L15 SamplingMetadata
@dataclass
class SamplingMetadata:
    # SOURCE: vllm/v1/sample/metadata.py:L16-L23
    temperature: torch.Tensor | None
    all_greedy: bool
    all_random: bool

    top_p: torch.Tensor | None
    top_k: torch.Tensor | None

    generators: dict[int, torch.Generator]

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: int | None

    # SOURCE: vllm/v1/sample/metadata.py:L28-L33
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

    # SOURCE: vllm/v1/sample/metadata.py:L46-L55
    # Specific token IDs to compute logprobs for (more efficient than full vocab)
    # When set, logprobs are computed only for these token IDs using gather
    # req_index -> list of token IDs to get logprobs for
    logprob_token_ids: dict[int, list[int]] | None = None

    # Speculative token ids
    spec_token_ids: list[list[int]] | None = None
    # When non-None, use ``holder.has_tracked_requests()`` to see if this batch applies
    # thinking-token-budget logits (holder may exist with an empty tracking set).
    thinking_budget_state_holder: ThinkingBudgetStateHolder | None = None
