# SOURCE: vllm/v1/sample/logits_processor/interface.py
# 本章消费面：MoveDirectionality / BatchUpdate（swap_states 与 condense 的
# moved 记录、BatchUpdateBuilder.get_and_reset 的产出类型）+ LogitsProcessor
# ABC（LogitsProcessors.all 迭代面）。
# SamplingParams 由 HOST SEAM 提供（vllm/sampling_params.py 全量归 ch08）。
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto

import torch

from .._host_seams import SamplingParams


# SOURCE: vllm/v1/sample/logits_processor/interface.py:L17-L22 MoveDirectionality
class MoveDirectionality(Enum):
    # One-way i1->i2 req move within batch
    # SOURCE: vllm/v1/sample/logits_processor/interface.py:L19
    UNIDIRECTIONAL = auto()
    # Two-way i1<->i2 req swap within batch
    # SOURCE: vllm/v1/sample/logits_processor/interface.py:L21
    SWAP = auto()


# Batch indices of any removed requests.
# SOURCE: vllm/v1/sample/logits_processor/interface.py:L24-L25 RemovedRequest
RemovedRequest = int

# (index, params, prompt_tok_ids, output_tok_ids) tuples for new
# requests added to the batch.
# SOURCE: vllm/v1/sample/logits_processor/interface.py:L27-L29 AddedRequest
AddedRequest = tuple[int, SamplingParams, list[int] | None, list[int]]

# (index 1, index 2, directionality) tuples representing
# one-way moves or two-way swaps of requests in batch
# SOURCE: vllm/v1/sample/logits_processor/interface.py:L31-L33 MovedRequest
MovedRequest = tuple[int, int, MoveDirectionality]


# SOURCE: vllm/v1/sample/logits_processor/interface.py:L36-L37 BatchUpdate
@dataclass(frozen=True)
class BatchUpdate:
    """Persistent batch state change info for logitsprocs"""

    # Current num reqs in batch
    # SOURCE: vllm/v1/sample/logits_processor/interface.py:L40
    batch_size: int

    # Metadata for requests added to, removed from, and moved
    # within the persistent batch.
    #
    # Key assumption: the `output_tok_ids` list (which is an element of each
    # tuple in `added`) is a reference to the request's running output tokens
    # list; via this reference, the logits processors always see the latest
    # list of generated output tokens.
    #
    # NOTE:
    # * Added or moved requests may replace existing requests with the same
    #   index.
    # * Operations should be processed in the following order:
    #   - removed, added, moved
    # SOURCE: vllm/v1/sample/logits_processor/interface.py:L42-L57
    removed: Sequence[RemovedRequest]
    added: Sequence[AddedRequest]
    moved: Sequence[MovedRequest]


# SOURCE: vllm/v1/sample/logits_processor/interface.py:L60 LogitsProcessor ABC
class LogitsProcessor(ABC):
    @classmethod
    # SOURCE: vllm/v1/sample/logits_processor/interface.py:L61-L69 validate_params
    def validate_params(cls, sampling_params: SamplingParams):
        """Validate sampling params for this logits processor.

        Raise ``VLLMValidationError`` (preferred) / ``ValueError`` (backward compatible)
        for invalid params. Bare ``ValueError`` is converted to ``VLLMValidationError``
        at the engine boundary so online serving returns HTTP 400.
        """
        return None

    @abstractmethod
    # SOURCE: vllm/v1/sample/logits_processor/interface.py:L71-L75 __init__
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    # SOURCE: vllm/v1/sample/logits_processor/interface.py:L77-L84 apply
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply LogitsProcessor to batch logits tensor.

        The updated tensor must be returned but may be
        modified in-place.
        """
        raise NotImplementedError

    @abstractmethod
    # SOURCE: vllm/v1/sample/logits_processor/interface.py:L86-L94 is_argmax_invariant
    def is_argmax_invariant(self) -> bool:
        """True if logits processor has no impact on the
        argmax computation in greedy sampling.
        NOTE: may or may not have the same value for all
        instances of a given LogitsProcessor subclass,
        depending on subclass implementation.
        """
        raise NotImplementedError

    @abstractmethod
    # SOURCE: vllm/v1/sample/logits_processor/interface.py:L96-L108 update_state
    def update_state(
        self,
        batch_update: "BatchUpdate | None",
    ) -> None:
        """Called when there are new output tokens, prior
        to each forward pass.

        Args:
            batch_update: Non-None iff there have been changes
                to the batch makeup.
        """
        raise NotImplementedError
