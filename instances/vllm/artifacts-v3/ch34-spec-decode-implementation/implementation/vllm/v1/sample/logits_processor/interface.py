# SOURCE: vllm/v1/sample/logits_processor/interface.py
# ch34 HOST SEAM（沿 ch30 镜像 + 本章消费面回补）：LogitsProcessor 抽象基类
# （apply 与 is_argmax_invariant 是「决定处理器进哪条流水线」的语义承诺）
# 与 BatchUpdate 族类型（MinTokens.update_state 的真实填充面——ch34 测试经
# update_state(BatchUpdate) 装载 min_toks，而非绕过 API 自建内部状态）。
# SUBTRACTED（沿 ch30 镜像）：update_state 抽象方法（L96-L108）——基类面
#   的持久批维护契约归 ch18（MinTokens 自带 update_state 实现按本章消费面
#   回补于 builtin.py）。
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch

from vllm import SamplingParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# SOURCE: vllm/v1/sample/logits_processor/interface.py:L17-L21 MoveDirectionality —— 逐字
class MoveDirectionality(Enum):
    # One-way i1->i2 req move within batch
    UNIDIRECTIONAL = auto()
    # Two-way i1<->i2 req swap within batch
    SWAP = auto()


# SOURCE: vllm/v1/sample/logits_processor/interface.py:L24-L33 批事件三类型别名 —— 逐字
# Batch indices of any removed requests.
RemovedRequest = int

# (index, params, prompt_tok_ids, output_tok_ids) tuples for new
# requests added to the batch.
AddedRequest = tuple[int, SamplingParams, list[int] | None, list[int]]

# (index 1, index 2, directionality) tuples representing
# one-way moves or two-way swaps of requests in batch
MovedRequest = tuple[int, int, MoveDirectionality]


# SOURCE: vllm/v1/sample/logits_processor/interface.py:L36-L57 BatchUpdate —— 逐字
@dataclass(frozen=True)
class BatchUpdate:
    # SOURCE: vllm/v1/sample/logits_processor/interface.py:L36-L57 BatchUpdate（持久批状态变更描述符）
    """Persistent batch state change info for logitsprocs"""

    batch_size: int  # Current num reqs in batch

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
    removed: Sequence[RemovedRequest]
    added: Sequence[AddedRequest]
    moved: Sequence[MovedRequest]


# SOURCE: vllm/v1/sample/logits_processor/interface.py:L60-L94 LogitsProcessor
#   —— 逐字（update_state 抽象方法已按 ch30 镜像减去）
class LogitsProcessor(ABC):
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        # SOURCE: vllm/v1/sample/logits_processor/interface.py:L61-L69 LogitsProcessor.validate_params —— 逐字
        """Validate sampling params for this logits processor.

        Raise ``VLLMValidationError`` (preferred) / ``ValueError`` (backward compatible)
        for invalid params. Bare ``ValueError`` is converted to ``VLLMValidationError``
        at the engine boundary so online serving returns HTTP 400.
        """
        return None

    @abstractmethod
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ) -> None:
        # SOURCE: vllm/v1/sample/logits_processor/interface.py:L71-L75 LogitsProcessor.__init__ —— 逐字
        raise NotImplementedError

    @abstractmethod
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/v1/sample/logits_processor/interface.py:L77-L84 LogitsProcessor.apply —— 逐字
        """Apply LogitsProcessor to batch logits tensor.

        The updated tensor must be returned but may be
        modified in-place.
        """
        raise NotImplementedError

    @abstractmethod
    def is_argmax_invariant(self) -> bool:
        # SOURCE: vllm/v1/sample/logits_processor/interface.py:L86-L94 LogitsProcessor.is_argmax_invariant —— 逐字（章节核心概念的可检测锚点）
        """True if logits processor has no impact on the
        argmax computation in greedy sampling.
        NOTE: may or may not have the same value for all
        instances of a given LogitsProcessor subclass,
        depending on subclass implementation.
        """
        raise NotImplementedError

    # SUBTRACTED: vllm/v1/sample/logits_processor/interface.py:L96-L108
    #   LogitsProcessor.update_state 抽象方法 —— 基类面的持久批维护契约
    #   （归 ch18；MinTokens 的实现按 ch34 消费面回补于 builtin.py）。
