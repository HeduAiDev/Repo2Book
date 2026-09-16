# SOURCE: vllm/v1/structured_output/backend_guidance.py
# 只做减法的忠实精简版（pin v0.27.1 / 6e448d0ea）——第二实现，证明两层契约
# 可替换（同契约异实现：rollback_lag/is_terminated 语义分歧的最好例证）。
# 删除项（subtraction_plan.delete）：[1] Mistral ll_tokenizer 三分支；
# [6] serialize_guidance_grammar 的 STRUCTURAL_TAG 分支；[7] logger 调用。
# SUBTRACTED: SPDX 版权头。
import copy
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import LazyLoader
# SUBTRACTED: `from transformers import MistralCommonBackend` 与
#   `from vllm.utils.mistral import is_mistral_tokenizer`（delete[1]——两者
#   只服务已删的 mistral ll_tokenizer 分支）
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.request import get_structured_output_key

if TYPE_CHECKING:
    import llguidance
    import llguidance.hf as llguidance_hf
    import llguidance.torch as llguidance_torch
else:
    llguidance = LazyLoader("llguidance", globals(), "llguidance")
    llguidance_hf = LazyLoader("llguidance.hf", globals(), "llguidance.hf")
    llguidance_torch = LazyLoader("llguidance.torch", globals(), "llguidance.torch")

logger = init_logger(__name__)


# SOURCE: vllm/v1/structured_output/backend_guidance.py:L36-L46 —— 逐字
def _walk_json_for_additional_properties(data: object):
    if isinstance(data, dict):
        for value in data.values():
            _walk_json_for_additional_properties(value)
        if "additionalProperties" not in data and (
            "properties" in data or "patternProperties" in data
        ):
            data["additionalProperties"] = False
    elif isinstance(data, list):
        for item in data:
            _walk_json_for_additional_properties(item)


# SOURCE: vllm/v1/structured_output/backend_guidance.py:L49-L72
#   has_guidance_unsupported_json_features —— 逐字（auto 阶梯 skip_guidance
#   第二判据，sampling_params.py:L1062-L1067 消费）
# SOURCE: vllm/v1/structured_output/backend_guidance.py:L49-L72
def has_guidance_unsupported_json_features(schema: dict[str, Any]) -> bool:
    """Check if JSON schema contains features unsupported by guidance/llguidance."""

    # SOURCE: vllm/v1/structured_output/backend_guidance.py:L52-L70（嵌套递归）
    def check_object(obj: dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False

        # patternProperties is not supported by llguidance
        if "patternProperties" in obj:
            return True

        # Recursively check all nested objects and arrays
        for value in obj.values():
            if isinstance(value, dict):
                if check_object(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True

        return False

    return check_object(schema)


# SOURCE: vllm/v1/structured_output/backend_guidance.py:L75-L84 —— 逐字
def process_for_additional_properties(
    guide_json: str | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(guide_json, str):
        guide_json_obj = json.loads(guide_json)
    else:
        # copy for modifications
        guide_json_obj = copy.deepcopy(guide_json)
    _walk_json_for_additional_properties(guide_json_obj)
    return guide_json_obj


# SOURCE: vllm/v1/structured_output/backend_guidance.py:L87-L139 GuidanceBackend
@dataclass
class GuidanceBackend(StructuredOutputBackend):
    def __post_init__(self):
        # SOURCE: vllm/v1/structured_output/backend_guidance.py:L89-L95
        self.disable_any_whitespace = (
            self.vllm_config.structured_outputs_config.disable_any_whitespace
        )
        self.disable_additional_properties = (
            self.vllm_config.structured_outputs_config.disable_additional_properties
        )

        # SUBTRACTED: vllm/v1/structured_output/backend_guidance.py:L97-L102
        #   mistral ll_tokenizer 三分支（is_mistral_tokenizer → llg_tokenizer /
        #   MistralCommonBackend → from_mistral_tokenizer）——delete[1]。精简版
        #   固定走 from_tokenizer（非 Mistral 模型即此路径）。
        # SOURCE: vllm/v1/structured_output/backend_guidance.py:L103-L106
        self.ll_tokenizer = llguidance_hf.from_tokenizer(
            self.tokenizer, max(self.vocab_size, len(self.tokenizer))
        )

    # SOURCE: vllm/v1/structured_output/backend_guidance.py:L108-L131 compile_grammar
    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        self.serialized_grammar = serialize_guidance_grammar(
            request_type,
            grammar_spec,
            self.disable_any_whitespace,
            self.disable_additional_properties,
        )

        ll_matcher = llguidance.LLMatcher(
            self.ll_tokenizer,
            self.serialized_grammar,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )

        r = GuidanceGrammar(
            ll_matcher=ll_matcher,
            ll_tokenizer=self.ll_tokenizer,
            vocab_size=self.vocab_size,
        )

        r.check_error()
        return r

    # SOURCE: vllm/v1/structured_output/backend_guidance.py:L133-L136
    def allocate_token_bitmask(self, max_num_seqs: int):
        return llguidance_torch.allocate_token_bitmask(
            max_num_seqs, self.ll_tokenizer.vocab_size
        )

    # SOURCE: vllm/v1/structured_output/backend_guidance.py:L138-L139
    def destroy(self):
        pass


# SOURCE: vllm/v1/structured_output/backend_guidance.py:L142-L221 GuidanceGrammar
#   —— 同一契约的第二实现（rollback_lag 是四后端语义分歧的最好例证）
@dataclass
class GuidanceGrammar(StructuredOutputGrammar):
    ll_matcher: llguidance.LLMatcher
    ll_tokenizer: llguidance.LLTokenizer
    vocab_size: int
    printed_error: bool = False
    terminated: bool = False
    rollback_lag: int = 0

    def check_error(self):
        # SOURCE: vllm/v1/structured_output/backend_guidance.py:L151-L156
        if not self.printed_error:
            err = self.ll_matcher.get_error()
            if err:
                self.printed_error = True
                # SUBTRACTED: logger.warning("LLMatcher error: %s", err)
                #   （delete[7]；错误标记位 printed_error 语义原样保留）

    # SOURCE: vllm/v1/structured_output/backend_guidance.py:L158-L184 accept_tokens
    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the parser.

        Returns True if the parser was advanced successfully.
        Returns False if the parser failed to advance.
        """

        if self.ll_tokenizer.eos_token in tokens:
            if self.ll_matcher.is_stopped() and not self.terminated:
                self.rollback_lag = 1
            self.terminated = True

        if self.ll_matcher.is_stopped():
            return True

        # TODO - Add jump decoding support in the future:
        # self.ll_matcher.compute_ff_bytes() - this should always work
        # self.ll_matcher.compute_ff_tokens() - this only works for
        #   "canonical" tokenizers
        # For conversion between the two, see
        # https://github.com/guidance-ai/llguidance/blob/main/docs/fast_forward.md

        r = self.ll_matcher.consume_tokens(tokens)

        self.check_error()

        return r

    # SOURCE: vllm/v1/structured_output/backend_guidance.py:L186-L201 validate_tokens
    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the parser in sequence.
        Will not advance the parser.

        Returns the prefix list of tokens that are accepted by the parser.
        """
        if len(tokens) == 0:
            return []
        if self.ll_matcher.is_stopped():
            return []

        num_tokens = self.ll_matcher.validate_tokens(tokens)

        self.check_error()

        return tokens[:num_tokens]

    # SOURCE: vllm/v1/structured_output/backend_guidance.py:L203-L208 rollback
    def rollback(self, num_tokens: int) -> None:
        if num_tokens > 0:
            self.ll_matcher.rollback(num_tokens - self.rollback_lag)
            self.terminated = False
            self.rollback_lag = 0
            self.check_error()

    # SOURCE: vllm/v1/structured_output/backend_guidance.py:L210-L214 fill_bitmask
    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        # this will automatically return [EOS] mask if the matcher is stopped
        # or otherwise in an error state
        llguidance_torch.fill_next_token_bitmask(self.ll_matcher, bitmask, idx)
        self.check_error()

    # SOURCE: vllm/v1/structured_output/backend_guidance.py:L216-L217 is_terminated
    def is_terminated(self) -> bool:
        return self.terminated

    # SOURCE: vllm/v1/structured_output/backend_guidance.py:L219-L221 reset
    def reset(self):
        # This method may be not needed anymore? TODO
        self.ll_matcher.reset()


# SOURCE: vllm/v1/structured_output/backend_guidance.py:L224-L290 serialize_guidance_
#   grammar —— 六形统一序列化（guidance 原生支持 choice——tp='choice'，与
#   xgrammar 的校验期改写路线对照）
# SOURCE: vllm/v1/structured_output/backend_guidance.py:L224-L290
def serialize_guidance_grammar(
    request_type: StructuredOutputOptions,
    grammar_spec: str | dict[str, Any],
    disable_any_whitespace: bool = False,
    disable_additional_properties: bool = False,
) -> str:
    # SOURCE: vllm/v1/structured_output/backend_guidance.py:L230-L240（嵌套）
    def _process_schema(
        grammar_spec: str | dict[str, Any],
    ) -> str:
        if disable_additional_properties:
            grammar_spec = process_for_additional_properties(grammar_spec)
        return llguidance.LLMatcher.grammar_from_json_schema(
            grammar_spec,
            defaults={
                "whitespace_flexible": not disable_any_whitespace,
            },
        )

    if request_type == StructuredOutputOptions.JSON:
        return _process_schema(grammar_spec)
    elif request_type == StructuredOutputOptions.JSON_OBJECT:
        return llguidance.LLMatcher.grammar_from_json_schema(
            '{"type": "object"}',
            defaults={
                "whitespace_flexible": not disable_any_whitespace,
            },
        )
    else:
        # SUBTRACTED: vllm/v1/structured_output/backend_guidance.py:L258-L282
        #   STRUCTURAL_TAG 分支（StructTag 拆解/trigger 匹配/to_grammar——
        #   delete[6]。guidance 只作契约可替换性的第二实现；structural_tag
        #   主线走 xgrammar，能力矩阵全量真相由 dossier m18 对照表给出）。
        if request_type == StructuredOutputOptions.REGEX:
            tp = "regex"
        elif request_type == StructuredOutputOptions.GRAMMAR:
            tp = "grammar"
        elif request_type == StructuredOutputOptions.CHOICE:
            tp = "choice"
        else:
            # SUBTRACTED: logger.error("Validation should have already
            #   occurred. Please file an issue.")（L284-L286——delete[7]）
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})"
            )
        return llguidance.grammar_from(tp, grammar_spec)


# SOURCE: vllm/v1/structured_output/backend_guidance.py:L293-L303 validate_guidance_
#   grammar —— auto 阶梯唯一兜底（sampling_params.py:L1076）+ guidance 显式分支
#   （L1027）共用
# SOURCE: vllm/v1/structured_output/backend_guidance.py:L293-L303
def validate_guidance_grammar(
    sampling_params: SamplingParams, tokenizer: llguidance.LLTokenizer | None = None
) -> None:
    # if structured output is not enabled, there is nothing to validate
    if sampling_params.structured_outputs is None:
        return
    tp, grm = get_structured_output_key(sampling_params.structured_outputs)
    guidance_grm = serialize_guidance_grammar(tp, grm)
    err = llguidance.LLMatcher.validate_grammar(guidance_grm, tokenizer)
    if err:
        raise ValueError(f"Grammar error: {err}")
