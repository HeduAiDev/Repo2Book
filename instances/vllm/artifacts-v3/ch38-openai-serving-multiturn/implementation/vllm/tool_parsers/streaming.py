# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/tool_parsers/streaming.py —— 忠实承载：
# filter_delta_text/extract_named_tool_call_streaming/
# extract_required_tool_call_streaming 逐字（DelegatingParser.
# _extract_tool_calls_streaming 的 named/required 两臂消费）；named 臂的
# Mistral tokenizer id 生成分支删（delete[5]）。
import json
from typing import TYPE_CHECKING

import regex as re
from partial_json_parser.core.options import Allow

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.tool_parsers.utils import partial_json_loads

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
else:
    TokenizerLike = object

# SUBTRACTED: vllm/tool_parsers/streaming.py:L18 is_mistral_tokenizer 导入
# ——delete[5] Mistral 特例。


# SOURCE: vllm/tool_parsers/streaming.py:L26-L47 —— _bracket_level_state 逐字
def _bracket_level_state(
    s: str, opening: str = "{", closing: str = "}"
) -> tuple[int, bool, bool]:
    level = 0
    in_string = False
    escaped = False
    for char in s:
        if escaped:
            escaped = False
            continue
        if in_string and char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == opening:
                level += 1
            elif char == closing:
                level -= 1
    return level, in_string, escaped


# SOURCE: vllm/tool_parsers/streaming.py:L50-L80 —— filter_delta_text 逐字
def filter_delta_text(
    delta_text: str,
    previous_text: str,
) -> tuple[str, bool]:
    """Trim trailing tool-list delimiters from required-tool streaming text."""
    bracket_level, in_string, escaped = _bracket_level_state(previous_text)
    updated_delta = ""
    passed_zero = False
    for char in delta_text:
        if escaped:
            escaped = False
        elif in_string:
            if char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            bracket_level += 1
            passed_zero = bracket_level == 0
        elif char == "}":
            bracket_level -= 1
            passed_zero = bracket_level == 0

        if bracket_level != 0:
            updated_delta += char
        else:
            if not in_string and char == ",":
                break
    return updated_delta, passed_zero


# SOURCE: vllm/tool_parsers/streaming.py:L83-L124 —— extract_named_tool_call_streaming
# 逐字（Mistral id 生成分支删——delete[5]，普通 tokenizer 走 make_tool_call_id）
def extract_named_tool_call_streaming(
    *,
    delta_text: str,
    function_name: str,
    function_name_returned: bool,
    tool_call_idx: int | None,
    tool_call_id_type: str,
    tokenizer: "TokenizerLike",
    tool_call_array_index: int = 0,
) -> tuple[DeltaMessage | None, bool]:
    """Build a streaming tool-call delta for forced named tool choice."""
    if function_name_returned:
        delta_tool_call = DeltaToolCall(
            function=DeltaFunctionCall(arguments=delta_text),
            index=tool_call_array_index,
        )
    else:
        # SUBTRACTED: vllm/tool_parsers/streaming.py:L100-L104
        # is_mistral_tokenizer 分支的 MistralToolCall.generate_random_id
        # ——delete[5]。
        tool_call_id = make_tool_call_id(
            id_type=tool_call_id_type,
            func_name=function_name,
            idx=tool_call_idx,
        )
        delta_tool_call = DeltaToolCall(
            id=tool_call_id,
            type="function",
            function=DeltaFunctionCall(
                name=function_name,
                arguments=delta_text,
            ),
            index=tool_call_array_index,
        )
        function_name_returned = True
    return (
        DeltaMessage(tool_calls=[delta_tool_call]),
        function_name_returned,
    )


# SOURCE: vllm/tool_parsers/streaming.py:L127-L223 —— extract_required_tool_call_streaming 逐字
def extract_required_tool_call_streaming(
    *,
    previous_text: str,
    current_text: str | None,
    delta_text: str,
    function_name_returned: bool,
    tool_call_idx: int | None,
    tool_call_id_type: str,
) -> tuple[DeltaMessage | None, bool]:
    if current_text is None or current_text == "":
        # if the current text is empty, we cannot parse it
        return None, function_name_returned
    try:
        flags = Allow.ALL
        obj, _ = partial_json_loads(current_text, flags)
    except (
        partial_json_parser.core.exceptions.MalformedJSON,
        json.JSONDecodeError,
    ):
        obj = None

    # check if the current text is a valid array
    # containing a partial tool calling object
    # if not repeat
    if obj is None or not isinstance(obj, list) or not len(obj) > 0:
        function_name_returned = False
        delta_message = None
    else:
        _, finishes_previous_tool = filter_delta_text(delta_text, previous_text)
        # take the last tool call from the generated list
        current_tool_call = obj[-1]

        # once parameters have been generated the name is complete as well
        if not finishes_previous_tool and (
            "name" not in current_tool_call or "parameters" not in current_tool_call
        ):
            function_name_returned = False
            delta_message = None
        else:
            if not function_name_returned:
                # get partly generated arguments from the latest tool call
                param_match = re.search(
                    r'.*"parameters":\s*(.*)', current_text, re.DOTALL
                )
                if param_match:
                    arguments = param_match.group(1)
                    arguments_prefix = current_text[: param_match.start(1)]
                    arguments, _ = filter_delta_text(arguments, arguments_prefix)
                else:
                    arguments = ""

                # if this iteration finishes a previous tool call but a
                # new incomplete tool is already generated, take the
                # previous from the list
                if finishes_previous_tool and "parameters" not in current_tool_call:
                    current_tool_call = obj[-2]

                function_name_returned = True
                tool_call_id = make_tool_call_id(
                    id_type=tool_call_id_type,
                    func_name=current_tool_call["name"],
                    idx=tool_call_idx,
                )
                delta_message = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            id=tool_call_id,
                            function=DeltaFunctionCall(
                                name=current_tool_call["name"], arguments=arguments
                            ),
                            index=len(obj) - 1,
                            type="function",
                        )
                    ]
                )

            else:
                delta_text, _ = filter_delta_text(delta_text, previous_text)

                if delta_text != "":
                    delta_message = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                function=DeltaFunctionCall(
                                    # OpenAI API returns None
                                    # instead of name every time
                                    name=None,
                                    arguments=delta_text,
                                ),
                                index=len(obj) - 1,
                            )
                        ]
                    )
                else:
                    delta_message = None

    return delta_message, function_name_returned
