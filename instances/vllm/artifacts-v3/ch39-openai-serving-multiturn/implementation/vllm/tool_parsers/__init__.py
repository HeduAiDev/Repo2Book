# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/tool_parsers/__init__.py —— 承载 re-export + 注册表装配
# （HOST SEAM：真实的 ~40 个 lazy 注册按精简树实际存在的模块裁剪——
# 只注册本章消费面真实承载的 gpt-oss 桩 "openai"→gptoss_tool_parser；
# 其余 parser 文件未随精简版携带，注册项不保留）。
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)

__all__ = ["ToolParser", "ToolParserManager"]


"""
Register a lazy module mapping.

Example:
    ToolParserManager.register_lazy_module(
        name="kimi_k2",
        module_path="vllm.tool_parsers.kimi_k2_tool_parser",
        class_name="KimiK2ToolParser",
    )
"""


# SOURCE: vllm/tool_parsers/__init__.py:L27-L202 —— _TOOL_PARSERS_TO_REGISTER
# （HOST SEAM 裁剪：仅 "openai" → GptOssToolParser 一项；其余 ~39 项的
# parser 实现文件不在精简树内——parser 族各归其章）
_TOOL_PARSERS_TO_REGISTER = {
    "openai": (
        "gptoss_tool_parser",
        "GptOssToolParser",
    ),
}


# SOURCE: vllm/tool_parsers/__init__.py:L204-L208 —— register_lazy_tool_parsers 逐字
def register_lazy_tool_parsers():
    for name, (file_name, class_name) in _TOOL_PARSERS_TO_REGISTER.items():
        module_path = f"vllm.tool_parsers.{file_name}"
        ToolParserManager.register_lazy_module(name, module_path, class_name)


register_lazy_tool_parsers()
