"""Subtract-only supporting types for the ch09 detokenization companion.

These are faithful subsets of the real vLLM types that the incremental
detokenizer and the output-processor stop-string loop depend on. They live here
only so the companion runs standalone; every type keeps vLLM's name / fields /
semantics, with orthogonal fields subtracted (and marked).

The detokenizer itself is reproduced verbatim against real vLLM source — only
its *external* dependency surface (EngineCoreRequest, TokenizerLike,
length_from_prompt_token_ids_or_embeds, FinishReason) is narrowed here.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Protocol


# SOURCE: vllm/tokenizers/__init__.py (TokenizerLike protocol — detokenize-relevant subset)
# SUBTRACTED: the full TokenizerLike protocol exposes encode/apply_chat_template/
#   vocab_size/save_pretrained/... — orthogonal to incremental detokenization.
#   Only the methods detokenize_incrementally / convert_prompt_ids_to_tokens call
#   are declared here (subtraction_plan: keep the detokenize path only).
class TokenizerLike(Protocol):
    is_fast: bool

    def __len__(self) -> int:  # SOURCE: vllm/tokenizers/__init__.py TokenizerLike
        ...

    def convert_ids_to_tokens(  # SOURCE: vllm/tokenizers/__init__.py TokenizerLike
        self, ids, skip_special_tokens: bool = False
    ): ...

    def convert_tokens_to_string(  # SOURCE: vllm/tokenizers/__init__.py TokenizerLike
        self, tokens: list[str]
    ) -> str: ...

    def get_added_vocab(self) -> dict:  # SOURCE: vllm/tokenizers/__init__.py TokenizerLike
        ...

    def decode(self, ids, **kwargs) -> str:  # SOURCE: vllm/tokenizers/__init__.py TokenizerLike
        ...

    @property
    def all_special_tokens(  # SOURCE: vllm/tokenizers/__init__.py TokenizerLike
        self,
    ) -> list[str]: ...


# SOURCE: vllm/utils/__init__.py:L15
# SUBTRACTED: torch.Tensor branches in the type hints — host companion uses plain
#   python lists for prompt_token_ids/prompt_embeds; the length logic is verbatim.
def length_from_prompt_token_ids_or_embeds(
    prompt_token_ids: list[int] | None,
    prompt_embeds: list | None,
) -> int:
    # SOURCE: vllm/utils/__init__.py:L15
    """Calculate the request length (in number of tokens) give either
    prompt_token_ids or prompt_embeds.
    """
    prompt_token_len = None if prompt_token_ids is None else len(prompt_token_ids)
    prompt_embeds_len = None if prompt_embeds is None else len(prompt_embeds)

    if prompt_token_len is None:
        if prompt_embeds_len is None:
            raise ValueError("Neither prompt_token_ids nor prompt_embeds were defined.")
        return prompt_embeds_len
    else:
        if prompt_embeds_len is not None and prompt_embeds_len != prompt_token_len:
            raise ValueError(
                "Prompt token ids and prompt embeds had different lengths"
                f" prompt_token_ids={prompt_token_len}"
                f" prompt_embeds={prompt_embeds_len}"
            )
        return prompt_token_len


# SOURCE: vllm/sampling_params.py SamplingParams (detokenize-relevant subset)
# SUBTRACTED: temperature/top_p/max_tokens/logprobs/... — the ~50 sampling knobs
#   are orthogonal to detokenization; only the fields the detokenizer reads are
#   kept (stop / min_tokens / include_stop_str_in_output / skip_special_tokens /
#   spaces_between_special_tokens), per subtraction_plan.
@dataclass
class SamplingParams:
    # SOURCE: vllm/sampling_params.py SamplingParams (detokenize-relevant subset)
    stop: str | list[str] | None = None
    min_tokens: int = 0
    include_stop_str_in_output: bool = False
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True


# SOURCE: vllm/v1/engine/__init__.py EngineCoreRequest (detokenize-relevant subset)
# SUBTRACTED: mm_features / lora_request / scheduler / arrival_time / ... — the
#   detokenizer only reads request_id / sampling_params / prompt_token_ids /
#   prompt_embeds (subtraction_plan).
@dataclass
class EngineCoreRequest:
    # SOURCE: vllm/v1/engine/__init__.py EngineCoreRequest (detokenize-relevant subset)
    request_id: str
    sampling_params: SamplingParams
    prompt_token_ids: list[int] | None = None
    prompt_embeds: list | None = None


# SOURCE: vllm/v1/engine/__init__.py:L42
class FinishReason(enum.IntEnum):
    """Reason a request finished - stop, length, abort, error, or repetition."""

    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4


# SOURCE: vllm/v1/engine/__init__.py EngineCoreOutput (detokenize-relevant subset)
# SUBTRACTED: new_logprobs / prompt_logprobs / pooling_output / prefill_stats /
#   kv_transfer_params / routed_experts / trace_headers — orthogonal subsystems
#   (logprobs ch10, pooling/PD/MoE other chapters); only the fields the
#   stop-string loop reads are kept (subtraction_plan).
@dataclass
class EngineCoreOutput:
    # SOURCE: vllm/v1/engine/__init__.py EngineCoreOutput (detokenize-relevant subset)
    request_id: str
    new_token_ids: list[int] = field(default_factory=list)
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    finished: bool = False


# SOURCE: vllm/v1/engine/output_processor.py OutputProcessorOutput (subset)
# SUBTRACTED: request_outputs assembly (make_request_output) is out of scope for
#   ch09 — only reqs_to_abort (the stop-string back-propagation channel) is kept.
@dataclass
class OutputProcessorOutput:
    # SOURCE: vllm/v1/engine/output_processor.py OutputProcessorOutput (subset)
    reqs_to_abort: list[str] = field(default_factory=list)
