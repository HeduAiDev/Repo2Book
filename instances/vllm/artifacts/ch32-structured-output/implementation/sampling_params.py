# SOURCE: vllm/sampling_params.py
# 只做减法的忠实精简版。ch31 已讲透 StructuredOutputsParams 的六选一互斥校验与
# SamplingParams._validate_structured_outputs 的完整后端选择阶梯；本章从「语法已
# 就绪」起步，不重复这条控制流，只保留 so_request.py 依赖的字段容器。
#
# SUBTRACTED: SamplingParams._validate_structured_outputs 整个方法（后端选择/auto
# 阶梯/choice 就地改写调用，ch31 §31.4-§31.8 已讲）、StructuredOutputsConfig。
# SUBTRACTED: SPDX 版权头。
from dataclasses import dataclass, field


@dataclass
class StructuredOutputsParams:
    # SOURCE: vllm/sampling_params.py:L40-97（字段声明部分；__post_init__ 六选一
    # 互斥校验属 ch31，本章不重复）
    json: "str | dict | None" = None
    regex: "str | None" = None
    choice: "list[str] | None" = None
    grammar: "str | None" = None
    json_object: "bool | None" = None
    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False
    whitespace_pattern: "str | None" = None
    structural_tag: "str | None" = None

    _backend: "str | None" = field(default=None, init=False)
    _backend_was_auto: bool = field(default=False, init=False)

    def all_constraints_none(self) -> bool:
        # SOURCE: vllm/sampling_params.py:L82-96
        return all(
            getattr(self, field_name) is None
            for field_name in (
                "json",
                "regex",
                "choice",
                "grammar",
                "json_object",
                "structural_tag",
            )
        )


class SamplingParams:
    # SOURCE: vllm/sampling_params.py class SamplingParams（精简为结构化输出相关切面）
    def __init__(
        self,
        max_tokens: "int | None" = 16,
        structured_outputs: "StructuredOutputsParams | None" = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.structured_outputs = structured_outputs
