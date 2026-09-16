# SOURCE: vllm/config/structured_outputs.py
# 只做减法的忠实精简版（引擎级结构化输出配置——m20 引擎侧后端冲突检查的
# 对照基准）。六字段与其 docstring 逐字；校验器本体逐字。
# SUBTRACTED: SPDX 版权头；pydantic @config 装饰器与 model_validator(mode=
#   "after")（真实 L66——HOST SEAM 用 __post_init__ 承载同一校验体）；
# compute_hash（L48-L62——配置指纹，编译缓存键的属地归 ch19）。
from typing import Any, Literal

StructuredOutputsBackend = Literal[
    "auto", "xgrammar", "guidance", "outlines", "lm-format-enforcer"
]


# SOURCE: vllm/config/structured_outputs.py:L16-L74 StructuredOutputsConfig
class StructuredOutputsConfig:
    """Dataclass which contains structured outputs config for the engine."""

    backend: StructuredOutputsBackend = "auto"
    """Which engine will be used for structured outputs (e.g. JSON schema,
    regex, etc) by default. With "auto", we will make opinionated choices
    based on request contents and what the backend libraries currently support,
    so the behavior is subject to change in each release."""
    disable_any_whitespace: bool = False
    """If `True`, json output will always be compact without any whitespace.
    If `False`, the model may generate whitespace between JSON fields,
    which is still valid JSON. This is only supported for xgrammar
    and guidance backends."""
    disable_additional_properties: bool = False
    """If `True`, the `guidance` backend will not use `additionalProperties`
    in the JSON schema. This is only supported for the `guidance` backend and
    is used to better align its behaviour with `outlines` and `xgrammar`."""
    reasoning_parser: str = ""
    """Select the reasoning parser depending on the model that you're using.
    This is used to parse the reasoning content into OpenAI API format."""
    reasoning_parser_plugin: str = ""
    """Path to a dynamically reasoning parser plugin that can be dynamically
    loaded and registered."""
    enable_in_reasoning: bool = False
    """Whether to use structured input for reasoning."""

    # SOURCE: vllm/config/structured_outputs.py:L16-L40 字段声明面 —— HOST SEAM 承载
    def __init__(
        self,
        backend: StructuredOutputsBackend = "auto",
        disable_any_whitespace: bool = False,
        disable_additional_properties: bool = False,
        reasoning_parser: str = "",
        reasoning_parser_plugin: str = "",
        enable_in_reasoning: bool = False,
    ) -> None:
        # HOST SEAM：真实是 pydantic @config dataclass（声明式字段）；
        # 精简版用显式 __init__ 承载同一字段面，随后跑同一校验体。
        self.backend = backend
        self.disable_any_whitespace = disable_any_whitespace
        self.disable_additional_properties = disable_additional_properties
        self.reasoning_parser = reasoning_parser
        self.reasoning_parser_plugin = reasoning_parser_plugin
        self.enable_in_reasoning = enable_in_reasoning
        self._validate_structured_output_config()

    # 校验体逐字（HOST SEAM：pydantic model_validator(mode="after") →
    #   __init__ 尾调用承载）
    # SOURCE: vllm/config/structured_outputs.py:L66-L74
    def _validate_structured_output_config(self) -> Any:
        if self.disable_any_whitespace and self.backend not in ("xgrammar", "guidance"):
            raise ValueError(
                "disable_any_whitespace is only supported for "
                "xgrammar and guidance backends."
            )
        if self.disable_additional_properties and self.backend != "guidance":
            raise ValueError(
                "disable_additional_properties is only supported "
                "for the guidance backend."
            )
        return self
