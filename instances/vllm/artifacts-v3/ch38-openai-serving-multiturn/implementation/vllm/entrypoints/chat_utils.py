# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/entrypoints/chat_utils.py —— HOST SEAM（最小承载）：真实
# 2000+ 行承载 chat 模板解析全链（ch6 域）；本章消费面只有四个符号：
# ChatTemplateContentFormatOption / ConversationMessage /
# ChatCompletionMessageParam 类型面 + make_tool_call_id /
# get_tool_call_id_type（函数逐字）+ load_chat_template（init_app_state
# 消费）。
from pathlib import Path
from typing import Any, Literal, TypeAlias

from vllm.utils import random_uuid

# SOURCE: vllm/entrypoints/chat_utils.py:L406
ChatTemplateContentFormatOption = Literal["auto", "string", "openai"]

# SOURCE: vllm/entrypoints/chat_utils.py:L376 —— ConversationMessage TypedDict
from typing_extensions import TypedDict


# SOURCE: vllm/entrypoints/chat_utils.py:L376
class ConversationMessage(TypedDict, total=False):
    # SOURCE: vllm/entrypoints/chat_utils.py:L377-L381 字段位
    role: str
    content: str


# SOURCE: vllm/entrypoints/chat_utils.py —— ChatCompletionMessageParam 别名
# 面（真实为各 role message param 的 Union；本章协议层 extra="allow" 下
# 按 dict 总形承载）
ChatCompletionMessageParam: TypeAlias = dict[str, Any]

# SUBTRACTED: vllm/entrypoints/chat_utils.py 其余（apply_hf_chat_template /
# parse_chat_messages / 多模态内容展开…）——ch6 渲染四步流水内部。

# SOURCE: vllm/entrypoints/chat_utils.py:L2040-L2042 —— kimi 模型族判定位
_KIMI_MODEL_TYPES = ("kimi_k2", "kimi_k25", "kimi_k3")


# SOURCE: vllm/entrypoints/chat_utils.py:L2044-L2059 —— get_tool_call_id_type 逐字
def get_tool_call_id_type(model_config) -> str:
    """Return the tool-call ID type for a given model configuration."""
    hf_overrides = getattr(model_config, "hf_overrides", None)
    hf_config = getattr(model_config, "hf_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    model_types = (
        getattr(hf_config, "model_type", None),
        getattr(hf_text_config, "model_type", None),
    )
    if any(model_type in _KIMI_MODEL_TYPES for model_type in model_types) or (
        isinstance(hf_overrides, dict)
        and hf_overrides.get("model_type") in _KIMI_MODEL_TYPES
    ):
        return "kimi_k2"
    return "random"


# SOURCE: vllm/entrypoints/chat_utils.py:L2061-L2067 —— make_tool_call_id 逐字
def make_tool_call_id(id_type: str = "random", func_name=None, idx=None):
    if id_type == "kimi_k2":
        return f"functions.{func_name}:{idx}"
    else:
        # by default return random
        return f"chatcmpl-tool-{random_uuid()}"


# SOURCE: vllm/entrypoints/chat_utils.py:L1397-L1401 —— load_chat_template
# （HOST SEAM：真实经 lru_cache 读模板文件/字面量；精简环境模板恒 None
# 路径——chat_template 默认 None）
def load_chat_template(
    chat_template: Path | str | None,
    *,
    is_literal: bool = False,
) -> str | None:
    if chat_template is None:
        return None
    if is_literal:
        return str(chat_template)
    with open(chat_template, encoding="utf-8") as f:
        return f.read()
