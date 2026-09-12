# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/tool_parsers/utils.py —— HOST SEAM（消费面子集）：Tool 别名、
# partial_json_loads（required 流式工具调用解析消费）、
# get_json_schema_from_tools 与其 ChatCompletion 主路（ToolParser.
# adjust_request 消费）逐字；pythonic AST 解析族与 Responses 专用装配
# 深层（_get_tool_schema_* / build_responses_tool_call_name_map）删——
# pythonic parser 族与 Responses API 面（delete[13]）非本章主线。
import json
from json import JSONDecodeError, JSONDecoder
from typing import Any, TypeAlias

import partial_json_parser
from openai.types.responses import FunctionTool, NamespaceTool, ToolChoiceFunction
from openai.types.responses.tool import Tool as ResponsesTool
from partial_json_parser.core.options import Allow

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolsParam,
)
from vllm.logger import init_logger

# SOURCE: vllm/tool_parsers/utils.py:L33
Tool: TypeAlias = ChatCompletionToolsParam | ResponsesTool

logger = init_logger(__name__)

# SUBTRACTED: vllm/tool_parsers/utils.py:L38-L131 safe_literal_eval /
# partial_tag_overlap / find_common_prefix / find_common_suffix /
# extract_intermediate_diff——pythonic 工具调用解析族（L436 起的
# UnexpectedAstError 族一并删）。


# SOURCE: vllm/tool_parsers/utils.py:L133-L141 —— partial_json_loads 逐字
def partial_json_loads(input_str: str, flags: Allow) -> tuple[Any, int]:
    try:
        return (partial_json_parser.loads(input_str, flags), len(input_str))
    except JSONDecodeError as e:
        if "Extra data" in e.msg:
            dec = JSONDecoder()
            return dec.raw_decode(input_str)
        raise


# SOURCE: vllm/tool_parsers/utils.py:L143-L149 —— is_complete_json 逐字
def is_complete_json(input_str: str) -> bool:
    try:
        json.loads(input_str)
        return True
    except JSONDecodeError:
        return False


# SOURCE: vllm/tool_parsers/utils.py:L180-L182 —— flat_namespace_tool_name
def flat_namespace_tool_name(namespace: str, name: str) -> str:
    return f"{namespace}__{name}"


# SOURCE: vllm/tool_parsers/utils.py:L184-L201 —— iter_response_function_tool_info 逐字
def iter_response_function_tool_info(
    tool: ResponsesTool,
) -> list[tuple[str, dict[str, Any] | None]]:
    if isinstance(tool, FunctionTool):
        return [(tool.name, tool.parameters)]
    if not isinstance(tool, NamespaceTool):
        return []

    namespace = tool.name
    return [
        (
            flat_namespace_tool_name(namespace, namespaced_tool.name),
            namespaced_tool.parameters,
        )
        for namespaced_tool in tool.tools
        if namespaced_tool.type == "function"
    ]


# SUBTRACTED: vllm/tool_parsers/utils.py:L203-L379 iter_response_function_
# tool_dicts / build_responses_tool_call_name_map / _get_json_schema_from_
# tools 的 Responses 深层装配——delete[13] Responses 面（chat 主线不触达
# NamespaceTool/Responses schema 装配）。

# SOURCE: vllm/tool_parsers/utils.py:L380-L413 —— get_json_schema_from_tools
# 逐字（"required" 分支的 _get_json_schema_from_tools 在精简环境按
# HOST SEAM 退化：仅装配 ChatCompletion 函数工具的 anyOf 骨架——见下位）
def get_json_schema_from_tools(
    tool_choice: str | ToolChoiceFunction | ChatCompletionNamedToolChoiceParam,
    tools: list[Tool] | None,
) -> str | dict | None:
    # tool_choice: "none"
    if tool_choice in ("none", None) or tools is None:
        return None
    # tool_choice: Forced Function (Responses)
    if (not isinstance(tool_choice, str)) and isinstance(
        tool_choice, ToolChoiceFunction
    ):
        tool_name = tool_choice.name
        responses_tool_map: dict[str, dict[str, Any] | None] = {}
        for tool in tools:
            if not isinstance(tool, (FunctionTool, NamespaceTool)):
                continue
            for name, params in iter_response_function_tool_info(tool):
                responses_tool_map[name] = params
                if "__" in name:
                    responses_tool_map.setdefault(name.rsplit("__", 1)[1], params)
        if tool_name not in responses_tool_map:
            raise ValueError(f"Tool '{tool_name}' has not been passed in `tools`.")
        return responses_tool_map[tool_name]
    # tool_choice: Forced Function (ChatCompletion)
    if (not isinstance(tool_choice, str)) and isinstance(
        tool_choice, ChatCompletionNamedToolChoiceParam
    ):
        tool_name = tool_choice.function.name
        chat_tool_map: dict[str, ChatCompletionToolsParam] = {
            tool.function.name: tool
            for tool in tools
            if isinstance(tool, ChatCompletionToolsParam)
        }
        if tool_name not in chat_tool_map:
            raise ValueError(f"Tool '{tool_name}' has not been passed in `tools`.")
        return chat_tool_map[tool_name].function.parameters
    # tool_choice: "required"
    if tool_choice == "required":
        # SUBTRACTED: vllm/tool_parsers/utils.py:L350-L379
        # _get_json_schema_from_tools 的 $defs/strict 装配——按 HOST SEAM
        # 退化（anyOf 骨架 + ChatCompletion 函数工具参数直取）
        fn_tool_schemas: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, ChatCompletionToolsParam) and tool.function:
                fn_tool_schemas.append(
                    {
                        "type": "object",
                        "properties": {},
                        "required": ["name", "parameters"],
                    }
                )
        return {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "anyOf": fn_tool_schemas,
            },
        }
    # tool_choice: "auto"
    return None
