# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/tool_parsers/gptoss_tool_parser.py —— 逐字承载（49 行全量，
# 无删改）：GptOss 工具解析桩——所有输出解析由 HarmonyParser 承担，此桩
# 只作为能力声明挂在 HarmonyParser.tool_parser_cls 上。
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
)
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike


# SOURCE: vllm/tool_parsers/gptoss_tool_parser.py:L17-L49
class GptOssToolParser(ToolParser):
    """
    Stub tool parser for gpt-oss/harmony models.

    All output parsing is handled by HarmonyParser. This stub exists as a
    capability declaration via HarmonyParser.tool_parser_cls.
    """

    structural_tag_model = "harmony"

    # SOURCE: vllm/tool_parsers/gptoss_tool_parser.py:L27-L29
    def __init__(self, tokenizer: "TokenizerLike", tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

    # SOURCE: vllm/tool_parsers/gptoss_tool_parser.py:L31-L35
    def extract_tool_calls(
        self, model_output, request, **kwargs
    ) -> ExtractedToolCallInformation:
        raise NotImplementedError(
            "GptOssToolParser is a stub. Use HarmonyParser for tool parsing."
        )

    # SOURCE: vllm/tool_parsers/gptoss_tool_parser.py:L37-L49
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request,
    ) -> DeltaMessage | None:
        raise NotImplementedError(
            "GptOssToolParser is a stub. Use HarmonyParser for tool parsing."
        )
