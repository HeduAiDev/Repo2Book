# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Minimal faithful stand-ins for the few vLLM types the ch18 reduced companion
# depends on. These mirror the real vLLM definitions (same field names / values)
# so the persistent-batch machinery runs host-side without importing vllm.

import enum
from dataclasses import dataclass, field


# SOURCE: vllm/sampling_params.py:L33  SamplingType
class SamplingType(enum.IntEnum):
    GREEDY = 0
    RANDOM = 1
    RANDOM_SEED = 2


# SOURCE: vllm/sampling_params.py SamplingParams (field subset used by InputBatch)
@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    logprobs: int | None = None
    logprob_token_ids: list[int] | None = None
    allowed_token_ids: list[int] | None = None
    bad_words_token_ids: list[list[int]] | None = None

    # SOURCE: vllm/sampling_params.py:L610  sampling_type
    @property
    def sampling_type(self) -> SamplingType:
        # SOURCE: vllm/sampling_params.py:L610
        if self.temperature < 1e-5:
            return SamplingType.GREEDY
        if self.seed is not None:
            return SamplingType.RANDOM_SEED
        return SamplingType.RANDOM


# SOURCE: vllm/utils/__init__.py:L15  length_from_prompt_token_ids_or_embeds
def length_from_prompt_token_ids_or_embeds(prompt_token_ids, prompt_embeds) -> int:
    prompt_token_len = None if prompt_token_ids is None else len(prompt_token_ids)
    prompt_embeds_len = None if prompt_embeds is None else len(prompt_embeds)
    if prompt_token_len is None:
        if prompt_embeds_len is None:
            raise ValueError("Neither prompt_token_ids nor prompt_embeds were defined.")
        return prompt_embeds_len
    else:
        if prompt_embeds_len is not None and prompt_embeds_len != prompt_token_len:
            raise ValueError("Prompt token ids and prompt embeds had different lengths")
        return prompt_token_len
