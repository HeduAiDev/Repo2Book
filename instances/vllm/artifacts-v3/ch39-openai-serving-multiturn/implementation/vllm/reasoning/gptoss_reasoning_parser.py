# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/reasoning/gptoss_reasoning_parser.py —— 逐字承载（63 行全量，
# 无删改）：gpt-oss 推理边界桩——harmony 下思考/正文分拣由 HarmonyParser
# 承担，此桩只提供边界检测（is_reasoning_end 恒 True：StreamableParser 的
# 通道切换天然界定边界）。
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest


# SOURCE: vllm/reasoning/gptoss_reasoning_parser.py:L16-L63
class GptOssReasoningParser(ReasoningParser):
    """
    Reasoning parser for GptOss model.

    The GptOss model uses harmony to extract reasoning content and this parser
    is only used for detecting the end of the reasoning content.
    """

    # SOURCE: vllm/reasoning/gptoss_reasoning_parser.py:L24-L25
    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

    # SOURCE: vllm/reasoning/gptoss_reasoning_parser.py:L27-L28
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return True

    # SOURCE: vllm/reasoning/gptoss_reasoning_parser.py:L30-L33
    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return True

    # SOURCE: vllm/reasoning/gptoss_reasoning_parser.py:L35-L39
    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        raise NotImplementedError(
            "GptOssReasoningParser only provides boundary detection. "
            "Use HarmonyParser for output parsing."
        )

    # SOURCE: vllm/reasoning/gptoss_reasoning_parser.py:L41-L53
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        raise NotImplementedError(
            "GptOssReasoningParser only provides boundary detection. "
            "Use HarmonyParser for output parsing."
        )

    # SOURCE: vllm/reasoning/gptoss_reasoning_parser.py:L55-L63
    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest",
    ) -> tuple[str | None, str | None]:
        raise NotImplementedError(
            "GptOssReasoningParser only provides boundary detection. "
            "Use HarmonyParser for output parsing."
        )
