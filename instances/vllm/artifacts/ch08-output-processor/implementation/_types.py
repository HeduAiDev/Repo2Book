"""Subtract-only supporting types for the Stage 3 output-processing companion.

These are faithful subsets of the real vLLM types that the output processor,
detokenizer, logprobs processor and parallel-sampling aggregator depend on.
They live here only so the companion runs without ``import vllm``; every type
keeps vLLM's name/fields/semantics, with orthogonal fields subtracted.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence as GenericSequence
from dataclasses import dataclass, field
from enum import Enum


# SOURCE: vllm/sampling_params.py:L151
class RequestOutputKind(Enum):
    # Return entire output so far in every RequestOutput
    CUMULATIVE = 0
    # Return only deltas in each RequestOutput
    DELTA = 1
    # Do not return intermediate RequestOutput
    FINAL_ONLY = 2


# SOURCE: vllm/v1/engine/__init__.py:L30
FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")


# SOURCE: vllm/v1/engine/__init__.py:L42
class FinishReason(enum.IntEnum):
    """Reason a request finished - stop, length, abort, error, or repetition."""

    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4

    def __str__(self):
        # SOURCE: vllm/v1/engine/__init__.py:L63
        return FINISH_REASON_STRINGS[self.value]


# SOURCE: vllm/v1/engine/__init__.py (EngineCoreOutput, generation-relevant subset)
# SUBTRACTED: pooling_output / prefill_stats(LoRA/metrics) / trace_headers /
#   kv_transfer_params / routed_experts transated as plain optional fields;
#   the many scheduler/metrics fields are out of scope for Stage 3 output path —
#   orthogonal subsystems (vllm/v1/engine/__init__.py EngineCoreOutput).
@dataclass
class PrefillStats:
    # SOURCE: vllm/v1/engine/__init__.py (EngineCoreOutput.prefill_stats subset)
    num_cached_tokens: int = 0


@dataclass
class EngineCoreOutput:
    # SOURCE: vllm/v1/engine/__init__.py EngineCoreOutput
    request_id: str
    new_token_ids: list[int] = field(default_factory=list)
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    finished: bool = False
    new_logprobs: object | None = None
    new_prompt_logprobs_tensors: object | None = None
    prefill_stats: PrefillStats | None = None
    # SUBTRACTED: pooling_output (pooling/embedding product line — subtraction_plan)
    pooling_output = None
    # SUBTRACTED: kv_transfer_params / routed_experts透传 (P/D + MoE orthogonal)
    kv_transfer_params: dict | None = None
    routed_experts = None


# SOURCE: vllm/outputs.py:L22 CompletionOutput
# SUBTRACTED: routed_experts / lora_request fields (MoE + LoRA orthogonal — subtraction_plan)
@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: GenericSequence[int]
    cumulative_logprob: float | None
    logprobs: list | None
    finish_reason: str | None = None
    stop_reason: int | str | None = None

    # SOURCE: vllm/outputs.py:L50
    def finished(self) -> bool:
        return self.finish_reason is not None
