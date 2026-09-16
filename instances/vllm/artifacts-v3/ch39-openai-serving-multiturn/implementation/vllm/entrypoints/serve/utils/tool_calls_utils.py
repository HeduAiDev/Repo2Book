# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/entrypoints/serve/utils/tool_calls_utils.py —— 逐字承载
# （37 行全量，无删改）：maybe_filter_parallel_tool_calls（流式/非流式
# 两生成器的 parallel_tool_calls=False 过滤）。
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TypeVar

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
)

# Used internally
_ChatCompletionResponseChoiceT = TypeVar(
    "_ChatCompletionResponseChoiceT",
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
)


# SOURCE: vllm/entrypoints/serve/utils/tool_calls_utils.py:L19
def maybe_filter_parallel_tool_calls(
    choice: _ChatCompletionResponseChoiceT, request: ChatCompletionRequest
) -> _ChatCompletionResponseChoiceT:
    """Filter to first tool call only when parallel_tool_calls is explicitly False."""

    if request.parallel_tool_calls is not False:
        return choice

    if isinstance(choice, ChatCompletionResponseChoice) and choice.message.tool_calls:
        choice.message.tool_calls = choice.message.tool_calls[:1]
    elif (
        isinstance(choice, ChatCompletionResponseStreamChoice)
        and choice.delta.tool_calls
    ):
        choice.delta.tool_calls = [
            tool_call for tool_call in choice.delta.tool_calls if tool_call.index == 0
        ]

    return choice
