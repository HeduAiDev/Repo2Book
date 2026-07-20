# SOURCE: vllm/v1/structured_output/backend_types.py
# 只做减法的忠实精简版（pin ad7125a4 / v0.21.0）。与 vLLM 同名、同结构、同控制流；只删不增。
#
# SUBTRACTED: SPDX 版权头、`if TYPE_CHECKING: import torch / from vllm.config import
# VllmConfig / from vllm.tokenizers import TokenizerLike` 的 LazyLoader 式类型占位
# （backend_types.py:L9-16）——纯类型标注装饰，无控制流影响（subtraction_plan 批准项7）。
# 精简版直接用 `object` 起名占位，不影响 ABC 的方法契约。
import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass

VllmConfig = object
TokenizerLike = object
Tensor = object  # SUBTRACTED: 真实标注是 "torch.Tensor"（惰性导入避免精简版强依赖 torch）


class StructuredOutputOptions(enum.Enum):
    # SOURCE: vllm/v1/structured_output/backend_types.py:L19-26
    JSON = enum.auto()
    JSON_OBJECT = enum.auto()
    REGEX = enum.auto()
    GRAMMAR = enum.auto()
    CHOICE = enum.auto()
    STRUCTURAL_TAG = enum.auto()


# SOURCE: vllm/v1/structured_output/backend_types.py:L28
StructuredOutputKey = tuple[StructuredOutputOptions, str]


class StructuredOutputGrammar(ABC):
    # SOURCE: vllm/v1/structured_output/backend_types.py:L31-96
    """Request-level backend for structured output requests."""

    @abstractmethod
    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        # SOURCE: vllm/v1/structured_output/backend_types.py:L34-46
        """
        Determines whether the provided tokens are accepted for the
        given request.

        Args:
            request_id (str): The unique identifier for the request.
            tokens (list[int]): A list of token IDs to evaluate.

        Returns:
            bool: True if the tokens are accepted, False otherwise.
        """

    @abstractmethod
    def validate_tokens(self, tokens: list[int]) -> list[int]:
        # SOURCE: vllm/v1/structured_output/backend_types.py:L48-60
        """
        Validates the provided tokens against the grammar.
        Will not advance the FSM.

        Args:
            tokens (list[int]): A list of token IDs to validate.

        Returns:
            list[int]: A list of accepted token IDs. Will be a prefix
                of the input tokens, and empty if none are accepted.
        """

    @abstractmethod
    def rollback(self, num_tokens: int) -> None:
        # SOURCE: vllm/v1/structured_output/backend_types.py:L62-71
        """
        Rolls back the state of the grammar by a specified number of tokens.
        Will also revert counters for the number of processed tokens.

        Args:
            num_tokens (int): The number of tokens to roll back.
        """

    @abstractmethod
    def fill_bitmask(self, bitmask: "Tensor", batch_index: int) -> None:
        # SOURCE: vllm/v1/structured_output/backend_types.py:L73-81
        """
        Fills the bitmask for a specific batch index.

        Args:
            bitmask (torch.Tensor): The bitmask to fill
            batch_index (int): The index in the bitmask to fill
        """

    @abstractmethod
    def is_terminated(self) -> bool:
        # SOURCE: vllm/v1/structured_output/backend_types.py:L83-90
        """
        Checks whether the structured output process has terminated.

        Returns:
            bool: True if the process is terminated, False otherwise.
        """

    @abstractmethod
    def reset(self):
        # SOURCE: vllm/v1/structured_output/backend_types.py:L92-96
        """
        Resets the state of the structured output grammar.
        """


@dataclass
class StructuredOutputBackend(ABC):
    # SOURCE: vllm/v1/structured_output/backend_types.py:L98-136
    """Engine-level backend for structured output requests."""

    vllm_config: VllmConfig
    tokenizer: TokenizerLike
    vocab_size: int

    @abstractmethod
    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        # SOURCE: vllm/v1/structured_output/backend_types.py:L107-121
        """
        Compiles a grammar specification into a structured output grammar.

        Args:
            request_type (StructuredOutputOptions): The type of structured
                output request.
            grammar_spec (str): The grammar specification to compile.

        Returns:
            StructuredOutputGrammar: The compiled structured output grammar.
        """

    @abstractmethod
    def allocate_token_bitmask(self, max_num_seqs: int) -> "Tensor":
        # SOURCE: vllm/v1/structured_output/backend_types.py:L123-131
        """
        Allocates a token bitmask for the specified maximum number of sequences.

        Args:
            max_num_seqs (int): The maximum number of sequences for which
                to allocate the bitmask.
        """

    @abstractmethod
    def destroy(self):
        # SOURCE: vllm/v1/structured_output/backend_types.py:L133-137
        """
        Backend-specific cleanup.
        """
