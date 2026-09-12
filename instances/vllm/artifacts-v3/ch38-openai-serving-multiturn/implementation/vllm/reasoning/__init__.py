# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/reasoning/__init__.py —— 承载 re-export + 注册表装配
# （HOST SEAM：真实的 ~35 个 lazy 注册按精简树实际存在的模块裁剪——
# 只注册本章消费面真实承载的 "openai_gptoss"→gptoss_reasoning_parser；
# 其余 parser 实现文件未随精简版携带）。
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser, ReasoningParserManager

__all__ = [
    "ReasoningParser",
    "ReasoningParserManager",
]
"""
Register a lazy module mapping.

Example:
    ReasoningParserManager.register_lazy_module(
        name="qwen3",
        module_path="vllm.reasoning.qwen3_engine_reasoning_parser",
        class_name="Qwen3ParserReasoningAdapter",
    )
"""


# SOURCE: vllm/reasoning/__init__.py:L26-L123 —— _REASONING_PARSERS_TO_REGISTER
# （HOST SEAM 裁剪：仅 "openai_gptoss" → GptOssReasoningParser 一项；其余
# ~34 项的 parser 实现文件不在精简树内——各归其章）
_REASONING_PARSERS_TO_REGISTER = {
    "openai_gptoss": (  # name
        "gptoss_reasoning_parser",  # filename
        "GptOssReasoningParser",  # class_name
    ),
}


# SOURCE: vllm/reasoning/__init__.py:L125-L129 —— register_lazy_reasoning_parsers 逐字
def register_lazy_reasoning_parsers():
    for name, (file_name, class_name) in _REASONING_PARSERS_TO_REGISTER.items():
        module_path = f"vllm.reasoning.{file_name}"
        ReasoningParserManager.register_lazy_module(name, module_path, class_name)


register_lazy_reasoning_parsers()
