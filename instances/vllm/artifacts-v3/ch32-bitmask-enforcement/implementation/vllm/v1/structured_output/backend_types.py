# SOURCE: vllm/v1/structured_output/backend_types.py
# 整文件逐字保留（ch30 已立的六方法契约：accept_tokens/validate_tokens/
# rollback/fill_bitmask/is_terminated/reset + allocate_token_bitmask/destroy
# ——本章 must_keep 的 fill_bitmask/is_terminated/accept_tokens/rollback 所在），
# 仅插入本行与各定义上方的 # SOURCE 标记。
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.tokenizers import TokenizerLike
else:
    VllmConfig = object
    TokenizerLike = object


# SOURCE: vllm/v1/structured_output/backend_types.py:L19 StructuredOutputOptions —— 逐字
class StructuredOutputOptions(enum.Enum):
    JSON = enum.auto()
    JSON_OBJECT = enum.auto()
    REGEX = enum.auto()
    GRAMMAR = enum.auto()
    CHOICE = enum.auto()
    STRUCTURAL_TAG = enum.auto()


StructuredOutputKey = tuple[StructuredOutputOptions, str]


# SOURCE: vllm/v1/structured_output/backend_types.py:L31 StructuredOutputGrammar —— 逐字（六方法契约本体）
class StructuredOutputGrammar(ABC):
    """Request-level backend for structured output requests."""

    @abstractmethod
    # SOURCE: vllm/v1/structured_output/backend_types.py:L34-L46 accept_tokens —— 逐字
    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
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
    # SOURCE: vllm/v1/structured_output/backend_types.py:L48-L60 validate_tokens —— 逐字
    def validate_tokens(self, tokens: list[int]) -> list[int]:
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
    # SOURCE: vllm/v1/structured_output/backend_types.py:L62-L70 rollback —— 逐字
    def rollback(self, num_tokens: int) -> None:
        """
        Rolls back the state of the grammar by a specified number of tokens.
        Will also revert counters for the number of processed tokens.

        Args:
            num_tokens (int): The number of tokens to roll back.
        """

    @abstractmethod
    # SOURCE: vllm/v1/structured_output/backend_types.py:L72-L80 fill_bitmask —— 逐字
    def fill_bitmask(self, bitmask: "torch.Tensor", batch_index: int) -> None:
        """
        Fills the bitmask for a specific batch index.

        Args:
            bitmask (torch.Tensor): The bitmask to fill
            batch_index (int): The index in the bitmask to fill
        """

    @abstractmethod
    # SOURCE: vllm/v1/structured_output/backend_types.py:L82-L89 is_terminated —— 逐字
    def is_terminated(self) -> bool:
        """
        Checks whether the structured output process has terminated.

        Returns:
            bool: True if the process is terminated, False otherwise.
        """

    @abstractmethod
    # SOURCE: vllm/v1/structured_output/backend_types.py:L91-L95 reset —— 逐字
    def reset(self):
        """
        Resets the state of the structured output grammar.
        """


@dataclass
# SOURCE: vllm/v1/structured_output/backend_types.py:L98-L99 StructuredOutputBackend —— 逐字
class StructuredOutputBackend(ABC):
    """Engine-level backend for structured output requests."""

    vllm_config: VllmConfig
    tokenizer: TokenizerLike
    vocab_size: int

    @abstractmethod
    # SOURCE: vllm/v1/structured_output/backend_types.py:L106-L120 compile_grammar —— 逐字
    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
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
    # SOURCE: vllm/v1/structured_output/backend_types.py:L122-L130 allocate_token_bitmask —— 逐字
    def allocate_token_bitmask(self, max_num_seqs: int) -> "torch.Tensor":
        """
        Allocates a token bitmask for the specified maximum number of sequences.

        Args:
            max_num_seqs (int): The maximum number of sequences for which
                to allocate the bitmask.
        """

    @abstractmethod
    # SOURCE: vllm/v1/structured_output/backend_types.py:L132-L136 destroy —— 逐字
    def destroy(self):
        """
        Backend-specific cleanup.
        """
