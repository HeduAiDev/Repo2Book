# SOURCE: vllm/v1/structured_output/backend_types.py
# 只做减法的忠实精简版（pin ad7125a4 / v0.21.0）。与 vLLM 同名、同结构、同控制流；只删不增。
# 本章从「语法已就绪」起步（ch31 已讲透六方法契约的完整设计），这里只保留
# StructuredOutputGrammar 六个方法的签名，供本章的 FakeGrammar 测试替身与
# StructuredOutputManager 的类型标注使用——本章的主线是「这些方法怎么被
# grammar_bitmask 调用」，不是它们各自怎么实现（那是 ch31 范围）。
#
# SUBTRACTED: SPDX 版权头、StructuredOutputBackend 的 dataclass 字段
# （vllm_config/tokenizer/vocab_size）与 compile_grammar/destroy 两个方法
# （backend_types.py:L98-147）——后端构造与销毁不在本章控制流内，ch31 已讲。
import enum
from abc import ABC, abstractmethod


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
        """Determines whether the provided tokens are accepted for the
        given request. Advances the FSM if accepted."""

    @abstractmethod
    def validate_tokens(self, tokens: list[int]) -> list[int]:
        # SOURCE: vllm/v1/structured_output/backend_types.py:L48-60
        """Validates the provided tokens against the grammar.
        Will not advance the FSM. Returns a prefix of the input."""

    @abstractmethod
    def rollback(self, num_tokens: int) -> None:
        # SOURCE: vllm/v1/structured_output/backend_types.py:L62-71
        """Rolls back the state of the grammar by num_tokens."""

    @abstractmethod
    def fill_bitmask(self, bitmask: "object", batch_index: int) -> None:
        # SOURCE: vllm/v1/structured_output/backend_types.py:L73-81
        """Fills the bitmask row at batch_index for this grammar's
        current state."""

    @abstractmethod
    def is_terminated(self) -> bool:
        # SOURCE: vllm/v1/structured_output/backend_types.py:L83-90
        """Checks whether the structured output process has terminated."""

    @abstractmethod
    def reset(self):
        # SOURCE: vllm/v1/structured_output/backend_types.py:L92-96
        """Resets the state of the structured output grammar."""
