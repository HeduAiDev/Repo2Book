# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/inputs/__init__.py —— 逐字承载（re-export 面；llm 段按
# HOST SEAM 精简版实际提供的符号裁剪，见 inputs/llm.py 头注）。
from .engine import (
    DecoderOnlyEngineInput,
    EncoderDecoderInput,
    EngineInput,
    MultiModalHashes,
    MultiModalPlaceholders,
    SingletonInput,
    TokensInput,
    tokens_input,
)
from .llm import (
    PromptType,
    SingletonPrompt,
    TokensPrompt,
)

__all__ = [
    "TokensPrompt",
    "PromptType",
    "SingletonPrompt",
    "MultiModalHashes",
    "MultiModalPlaceholders",
    "TokensInput",
    "tokens_input",
    "DecoderOnlyEngineInput",
    "EncoderDecoderInput",
    "SingletonInput",
    "EngineInput",
]
