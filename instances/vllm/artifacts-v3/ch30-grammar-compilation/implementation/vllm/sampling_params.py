# SOURCE: vllm/sampling_params.py
# 只做减法的忠实精简版（pin v0.27.1 / 6e448d0ea）。本章主线 = StructuredOutputs
# Params 六选一 + 前端校验选后端（verify → _validate_structured_outputs 的
# auto 降级阶梯）。同名、同结构、同控制流；只删 dossier.subtraction_plan.delete
# 批准项，must_keep 全保留；每 def/class 标 # SOURCE:（行号对 v0.27.1 现核）。
#
# SUBTRACTED: SPDX 版权头；文件其余（SamplingType/RepetitionDetectionParams/
#   RequestOutputKind 与 SamplingParams 的数百行采样字段面——ch29 的属地，
#   本章 HOST SEAM 只载 structured_outputs 切面）。
import json as json_mod
from dataclasses import dataclass, field
from typing import Any

from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.utils.mistral import is_mistral_tokenizer

logger = init_logger(__name__)


# SOURCE: vllm/sampling_params.py:L70-L142 StructuredOutputsParams —— 逐字
#   （L72 类头注释 "maybe make msgspec?" 亦原文）
# SOURCE: vllm/sampling_params.py:L71-L142（@dataclass+类头；装饰器上一行的锚）
# SOURCE: vllm/sampling_params.py:L71-L142（@dataclass+类头；装饰器上一行的锚）
@dataclass
class StructuredOutputsParams:
    # One of these fields will be used to build a logit processor.
    json: str | dict | None = None
    regex: str | None = None
    choice: list[str] | None = None
    grammar: str | None = None
    json_object: bool | None = None
    # These are other options that can be set.
    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False
    whitespace_pattern: str | None = None
    structural_tag: str | None = None

    _backend: str | None = field(default=None, init=False)
    """CAUTION: Should only be set by Processor._validate_structured_output"""
    _backend_was_auto: bool = field(default=False, init=False)
    """CAUTION: Should only be set by Processor._validate_structured_output"""

    # SOURCE: vllm/sampling_params.py:L90-L111（六选一互斥双向校验）
    def __post_init__(self):
        """Validate that some fields are mutually exclusive."""
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
            raise VLLMValidationError(
                "You can only use one kind of structured outputs constraint "
                f"but multiple are specified: {self.__dict__}"
            )
        if count < 1:
            raise VLLMValidationError(
                "You must use one kind of structured outputs constraint "
                f"but none are specified: {self.__dict__}"
            )

    # SOURCE: vllm/sampling_params.py:L113-L127
    def all_constraints_none(self) -> bool:
        """
        Returns True if all structured-output constraint fields are None.
        """
        return all(
            getattr(self, field) is None
            for field in (
                "json",
                "regex",
                "choice",
                "grammar",
                "json_object",
                "structural_tag",
            )
        )

    # SOURCE: vllm/sampling_params.py:L129-L142
    def all_non_structural_tag_constraints_none(self) -> bool:
        """
        Returns True if all structured-output constraint fields are None.
        """
        return all(
            getattr(self, field) is None
            for field in (
                "json",
                "regex",
                "choice",
                "grammar",
                "json_object",
            )
        )


# SOURCE: vllm/sampling_params.py:L191-L192 _is_non_tekken_mistral —— 逐字
#   （auto 阶梯 skip_guidance 第一判据 L1058 + guidance 显式分支拒单 L1016
#   两处消费；subtraction_plan.delete[0] 明令保留）
# SOURCE: vllm/sampling_params.py:L191-L192
def _is_non_tekken_mistral(tokenizer: TokenizerLike) -> bool:
    return is_mistral_tokenizer(tokenizer) and not tokenizer.is_tekken


# SOURCE: vllm/sampling_params.py:L195-L196 _get_llg_tokenizer —— 逐字
def _get_llg_tokenizer(tokenizer: TokenizerLike) -> Any:
    return tokenizer.llg_tokenizer if is_mistral_tokenizer(tokenizer) else None


# SOURCE: vllm/sampling_params.py:L199 SamplingParams —— HOST SEAM 字段面
#   （真实是 msgspec.Struct 数百字段大类——ch29 的属地；本章只消费
#   structured_outputs 与 max_tokens（Request.__init__ L111-L112 断言读取））
class SamplingParams:
    # SOURCE: vllm/sampling_params.py:L199 SamplingParams 类头 —— HOST SEAM 字段面
    def __init__(
        self,
        structured_outputs: StructuredOutputsParams | None = None,
        max_tokens: int | None = 16,
    ) -> None:
        self.structured_outputs = structured_outputs
        self.max_tokens = max_tokens

    # SOURCE: vllm/sampling_params.py:L755-L770 verify —— 前端校验入口
    #   （调用方 = InputProcessor，vllm/v1/engine/input_processor.py:L104）
    # SOURCE: vllm/sampling_params.py:L755-L770
    def verify(
        self,
        model_config,
        speculative_config,
        structured_outputs_config,
        tokenizer: TokenizerLike | None,
    ) -> None:
        # SUBTRACTED: 六项非结构化校验调用（L762-L767：
        #   _validate_logprobs/_validate_logit_bias/_validate_logits_processors/
        #   _validate_allowed_token_ids/_validate_spec_decode/
        #   _validate_diffusion——本体在 L772-L921，ch29 的 SamplingParams
        #   完整面属地；HOST SEAM 未载其字段面）。
        self._validate_structured_outputs(
            model_config, structured_outputs_config, tokenizer
        )

    # SOURCE: vllm/sampling_params.py:L923-L1086 _validate_structured_outputs
    def _validate_structured_outputs(
        self,
        model_config,
        structured_outputs_config,
        tokenizer: TokenizerLike | None,
    ) -> None:
        if structured_outputs_config is None or self.structured_outputs is None:
            return

        # SOURCE: vllm/sampling_params.py:L932-L942 diffusion 拒单（#45436）
        if model_config.is_diffusion:
            # Diffusion LLMs denoise a whole canvas of tokens in parallel
            # rather than sampling left-to-right, which the grammar FSM
            # requires. Without this check, requests fail mid-generation
            # with an FSM rejection (HTTP 500). See issue #45436.
            raise VLLMValidationError(
                "Structured outputs are not yet supported for diffusion "
                "language models. Remove the structured output constraint "
                "(e.g. `response_format`, `structured_outputs`) from the "
                "request."
            )

        # SOURCE: vllm/sampling_params.py:L944-L947 无 tokenizer 拒单
        if tokenizer is None:
            raise VLLMValidationError(
                "Structured outputs requires a tokenizer so it can't be used with 'skip_tokenizer_init'"  # noqa: E501
            )

        # SOURCE: vllm/sampling_params.py:L949-L966 引擎单后端冲突检查
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
                raise VLLMValidationError(
                    "Request-level structured output backend selection is not "
                    f"supported. The request specified '{_backend}', but vLLM "
                    f"was initialised with '{backend}'. This error can be "
                    "resolved by removing '_backend' from the request."
                )
        else:
            self.structured_outputs._backend = backend

        # Request content validation
        # SOURCE: vllm/sampling_params.py:L968-L998 内容预检
        if (
            isinstance(self.structured_outputs.choice, list)
            and not self.structured_outputs.choice
        ):
            # It is invalid for choice to be an empty list
            raise VLLMValidationError(
                f"Choice '{self.structured_outputs.choice}' cannot be an empty list"  # noqa: E501
            )
        # Reject empty string grammar early to avoid engine-side crashes
        if (
            isinstance(self.structured_outputs.grammar, str)
            and self.structured_outputs.grammar.strip() == ""
        ):
            raise VLLMValidationError(
                "structured_outputs.grammar cannot be an empty string"
            )
        # Reject empty string json schema early to avoid engine-side crashes
        if (
            isinstance(self.structured_outputs.json, str)
            and self.structured_outputs.json.strip() == ""
        ):
            raise VLLMValidationError(
                "structured_outputs.json cannot be an empty string"
            )
        # Reject json_object=False early to avoid engine-side crashes
        if self.structured_outputs.json_object is False:
            raise VLLMValidationError(
                "structured_outputs.json_object must be True if set; omit "
                "structured_outputs to disable structured outputs"
            )

        # SOURCE: vllm/sampling_params.py:L1000-L1003 guidance 校验器 import
        # SUBTRACTED: L1004-L1009 两段 import——backend_lm_format_enforcer 的
        #   validate_structured_output_request_lm_format_enforcer 与
        #   backend_outlines 的 validate_structured_output_request_outlines
        #   （delete[4]：两后端整体删除，防 import 断链一体删）。
        from vllm.v1.structured_output.backend_guidance import (
            has_guidance_unsupported_json_features,
            validate_guidance_grammar,
        )
        # SOURCE: vllm/sampling_params.py:L1010 xgrammar 校验器 import
        from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar

        # SOURCE: vllm/sampling_params.py:L1012-L1014 xgrammar 显式分支
        if backend.startswith("xgrammar"):
            # xgrammar with no fallback
            validate_xgrammar_grammar(self)
        # SOURCE: vllm/sampling_params.py:L1015-L1030 guidance 显式分支
        elif backend.startswith("guidance"):
            if _is_non_tekken_mistral(tokenizer=tokenizer):
                raise VLLMValidationError(
                    "Non-tekken Mistral tokenizers are not supported for the 'guidance'"
                    " structured output backend. Please either use a more recent "
                    "Mistral model, the ['xgrammar', 'outlines'] "
                    "backends or tokenizer_mode='hf' instead."
                )
            # TODO: ideally we would have the LLTokenizer here as Lark syntax
            # allows <|special_token|> and similar, see
            # https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md#special-tokens
            # Without tokenizer these are disallowed in grammars.
            validate_guidance_grammar(
                self,
                tokenizer=_get_llg_tokenizer(tokenizer),
            )
        # SUBTRACTED: L1031-L1042 两个显式后端分支——backend == "outlines" 与
        #   backend == "lm-format-enforcer"（delete[4]：两后端整体删除）。
        #   已知行为差异（正文 m18 对照表给全量真相）：显式配 outlines/LMFE
        #   的引擎在精简版下走 auto 阶梯的 else——真实源码不会。
        else:
            # NOTE: backend must be "auto" here, because we have
            # checked supported_backends above.
            # In this mode, we set opinionated defaults based on what we think
            # will satisfy the most use cases without having to worry about
            # this setting. We include fallback behavior here, but not with any
            # other setting where a specific backend was specified.
            try:
                validate_xgrammar_grammar(self)
                self.structured_outputs._backend = "xgrammar"
            except ValueError:
                # The request either failed validation
                # or includes some jsonschema feature(s) that
                # are not supported in xgrammar.

                # SOURCE: vllm/sampling_params.py:L1058-L1067 skip_guidance 两判据
                #   （_is_non_tekken_mistral + has_guidance_unsupported_json_
                #   features——降级判据的纯计算证据，其值不再分流）
                skip_guidance = _is_non_tekken_mistral(tokenizer)

                # Check if schema has features unsupported by guidance
                so_params = self.structured_outputs
                if not skip_guidance and so_params.json:
                    if isinstance(so_params.json, str):
                        schema = json_mod.loads(so_params.json)
                    else:
                        schema = so_params.json
                    skip_guidance = has_guidance_unsupported_json_features(schema)

                # SUBTRACTED: L1069-L1073 if skip_guidance: 的 outlines 降级支
                #   （delete[4]③：outlines 后端整体删除，else 体提升为直线）。
                #   已知行为差异：非 tekken Mistral 分词器或 guidance 不支持
                #   的 schema（patternProperties），真实源码降 outlines，精简版
                #   经 validate_guidance_grammar 报错拒单。
                # Fall back to guidance by default.
                validate_guidance_grammar(
                    self,
                    tokenizer=_get_llg_tokenizer(tokenizer),
                )
                self.structured_outputs._backend = "guidance"
            # Remember that this backend was set automatically
            self.structured_outputs._backend_was_auto = True

        # Run post-init validation. This is also important to ensure subsequent
        # roundtrip serialization/deserialization won't fail.
        self.structured_outputs.__post_init__()
