# SOURCE: vllm/sampling_params.py
# 只做减法的忠实精简版。真实 SamplingParams 另有 temperature/top_p/top_k/max_tokens/
# penalties/... 数十个采样相关字段与方法（ch30 范围），与结构化输出的后端选择/校验
# 主控制流正交，精简版 SamplingParams 只保留 _validate_structured_outputs 依赖的
# structured_outputs 字段与 max_tokens（Request.__init__ 用到）。
#
# SUBTRACTED: SPDX 版权头。
import json as json_mod
from dataclasses import dataclass, field

from backend_types import StructuredOutputOptions  # noqa: F401  (供文档引用)


@dataclass
class StructuredOutputsParams:
    # SOURCE: vllm/sampling_params.py:L40-97
    # One of these fields will be used to build a logit processor.
    json: "str | dict | None" = None
    regex: "str | None" = None
    choice: "list[str] | None" = None
    grammar: "str | None" = None
    json_object: "bool | None" = None
    # These are other options that can be set.
    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False
    whitespace_pattern: "str | None" = None
    structural_tag: "str | None" = None

    _backend: "str | None" = field(default=None, init=False)
    """CAUTION: Should only be set by Processor._validate_structured_output"""
    _backend_was_auto: bool = field(default=False, init=False)
    """CAUTION: Should only be set by Processor._validate_structured_output"""

    def __post_init__(self):
        """Validate that some fields are mutually exclusive."""
        # SOURCE: vllm/sampling_params.py:L59-80 —— 六选一互斥，count>1 与 count<1
        # 双向校验。
        count = sum(
            [
                self.json is not None,
                self.regex is not None,
                self.choice is not None,
                self.grammar is not None,
                self.json_object is not None,
                self.structural_tag is not None,
            ]
        )
        if count > 1:
            raise ValueError(
                "You can only use one kind of structured outputs constraint "
                f"but multiple are specified: {self.__dict__}"
            )
        if count < 1:
            raise ValueError(
                "You must use one kind of structured outputs constraint "
                f"but none are specified: {self.__dict__}"
            )

    def all_constraints_none(self) -> bool:
        # SOURCE: vllm/sampling_params.py:L82-96
        """
        Returns True if all structured-output constraint fields are None.
        """
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


@dataclass
class StructuredOutputsConfig:
    # SOURCE: vllm/config/... StructuredOutputsConfig（精简为本章唯一用到的字段）
    #
    # SUBTRACTED: 真实 StructuredOutputsConfig 另有 reasoning_parser /
    # reasoning_parser_plugin / enable_in_reasoning 等 reasoning 相关字段
    # （subtraction_plan 批准项3），以及 disable_any_whitespace /
    # disable_additional_properties 的引擎级默认值字段——本章只关心 backend 选择。
    backend: str = "auto"


class SamplingParams:
    # SOURCE: vllm/sampling_params.py class SamplingParams（精简为结构化输出相关切面）
    def __init__(
        self,
        max_tokens: "int | None" = 16,
        structured_outputs: "StructuredOutputsParams | None" = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.structured_outputs = structured_outputs

    def _validate_structured_outputs(
        self,
        structured_outputs_config: "StructuredOutputsConfig | None",
        tokenizer: object | None,
    ) -> None:
        # SOURCE: vllm/sampling_params.py:L773-907
        if structured_outputs_config is None or self.structured_outputs is None:
            return

        if tokenizer is None:
            raise ValueError(
                "Structured outputs requires a tokenizer so it can't be used "
                "with 'skip_tokenizer_init'"
            )

        backend = structured_outputs_config.backend
        if _backend := self.structured_outputs._backend:
            # Request-level backend selection is not supported.
            # The values may differ if `params` is reused and was set
            # to a specific backend based on `auto` behavior in a previous
            # request. We remember that it was set as a result of `auto`
            # using the `_backend_was_auto` field set in the params.
            if backend != _backend and not (
                backend == "auto" and self.structured_outputs._backend_was_auto
            ):
                raise ValueError(
                    "Request-level structured output backend selection is not "
                    f"supported. The request specified '{_backend}', but vLLM "
                    f"was initialised with '{backend}'. This error can be "
                    "resolved by removing '_backend' from the request."
                )
        else:
            self.structured_outputs._backend = backend

        # Request content validation
        if (
            isinstance(self.structured_outputs.choice, list)
            and not self.structured_outputs.choice
        ):
            # It is invalid for choice to be an empty list
            raise ValueError(
                f"Choice '{self.structured_outputs.choice}' cannot be an empty list"
            )
        # Reject empty string grammar early to avoid engine-side crashes
        if (
            isinstance(self.structured_outputs.grammar, str)
            and self.structured_outputs.grammar.strip() == ""
        ):
            raise ValueError("structured_outputs.grammar cannot be an empty string")

        # SUBTRACTED: `from vllm.v1.structured_output.backend_guidance import
        # has_guidance_unsupported_json_features, validate_guidance_grammar` /
        # `from vllm.v1.structured_output.backend_lm_format_enforcer import
        # validate_structured_output_request_lm_format_enforcer` / `from
        # vllm.v1.structured_output.backend_outlines import
        # validate_structured_output_request_outlines`（延迟导入，避免顶层强依赖）
        # 精简为直接顶层导入 backend_xgrammar 与 backend_guidance 的对应函数。
        from backend_guidance import has_guidance_unsupported_json_features
        from backend_xgrammar import validate_xgrammar_grammar

        if backend.startswith("xgrammar"):
            # xgrammar with no fallback
            validate_xgrammar_grammar(self)
        elif (
            backend.startswith("guidance")
            or backend == "outlines"
            or backend == "lm-format-enforcer"
        ):
            # SUBTRACTED: 三个显式后端分支的真实校验逻辑被删——
            # backend=="guidance" 分支唯一逻辑是 Mistral tokenizer 检查（批准项1，
            # backend_guidance.py:L96-97 手工分支）+ validate_guidance_grammar 调用
            # （该函数本身已整体删除，批准项6："各后端请求校验函数里与 choice 改写
            # 无关的部分"）；backend=="outlines"/"lm-format-enforcer" 分支对应的两个
            # 后端文件本身已整体删除（批准项5）。精简版信任显式选择、不做该分支的
            # 语法可编译性预检——这不改变 auto 阶梯（下面 else 分支）的行为，
            # 只影响用户显式指定这三种后端时的早期报错能力。
            pass
        else:
            # NOTE: backend must be "auto" here, because we have
            # checked supported_backends above.
            # SOURCE: vllm/sampling_params.py:L871-901 —— auto 的先试最优、失败降级
            # 阶梯：xgrammar → guidance → outlines。
            #
            # SUBTRACTED: `skip_guidance = _is_non_tekken_mistral(tokenizer)`
            # （Mistral 检查，批准项1）——精简版 skip_guidance 只由
            # has_guidance_unsupported_json_features 决定。
            try:
                validate_xgrammar_grammar(self)
                self.structured_outputs._backend = "xgrammar"
            except ValueError:
                # The request either failed validation
                # or includes some jsonschema feature(s) that
                # are not supported in xgrammar.
                skip_guidance = False

                so_params = self.structured_outputs
                if so_params.json:
                    if isinstance(so_params.json, str):
                        schema = json_mod.loads(so_params.json)
                    else:
                        schema = so_params.json
                    skip_guidance = has_guidance_unsupported_json_features(schema)

                if skip_guidance:
                    # SUBTRACTED: outlines 全实现被删（批准项5），精简版不含 outlines
                    # 这个回退目标；真实 vLLM 在此调用
                    # validate_structured_output_request_outlines 并把 _backend
                    # 设为 "outlines"。三家阶梯的形状仍然可见——只是这一支没有可运行
                    # 的实现，用异常明确标出边界，而不是悄悄换一种行为。
                    raise NotImplementedError(
                        "auto 阶梯的 outlines 回退分支已随 outlines 后端一起从"
                        "精简版中删除（真实 vLLM 会 fall back 到 outlines）"
                    )
                else:
                    # SUBTRACTED: validate_guidance_grammar(self, tokenizer=...) 调用
                    # 被删——该函数本身经批准整体删除（只做 guidance 语法可编译性
                    # 预检，与 choice 改写无关）。精简版直接信任并设置 _backend。
                    self.structured_outputs._backend = "guidance"
            # Remember that this backend was set automatically
            self.structured_outputs._backend_was_auto = True

        # Run post-init validation. This is also important to ensure subsequent
        # roundtrip serialization/deserialization won't fail.
        self.structured_outputs.__post_init__()
