# SOURCE: vllm/v1/structured_output/backend_guidance.py
# 只做减法的忠实精简版（第二实现，证明契约的可替换性：同一份 StructuredOutputGrammar
# 六方法契约，guidance 后端的实现细节与 xgrammar 明显不同——最典型的是 rollback_lag）。
#
# SUBTRACTED: SPDX 版权头；`from vllm.utils.import_utils import LazyLoader` 的
# LazyLoader 惰性导入模式（批准项7）——精简版改用 try/except 顶层导入。
try:
    import llguidance  # type: ignore
    import llguidance.torch as llguidance_torch  # type: ignore
except ImportError:  # pragma: no cover - host 通常无 llguidance，见 INSTANCE.md
    llguidance = None  # type: ignore
    llguidance_torch = None  # type: ignore

from dataclasses import dataclass
from typing import Any

from backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)

# SUBTRACTED: `from vllm.utils.mistral import is_mistral_tokenizer`（批准项1）；
# `import copy` / `_walk_json_for_additional_properties` /
# `process_for_additional_properties`（disable_additional_properties 旁路，
# 与本章"六方法契约 + auto 阶梯"主线正交，未被 must_keep 或 code_spine 提及）。


def has_guidance_unsupported_json_features(schema: dict[str, Any]) -> bool:
    # SOURCE: vllm/v1/structured_output/backend_guidance.py:L48-71（顶层检测）
    """Check if JSON schema contains features unsupported by guidance/llguidance."""
    # SUBTRACTED: 完整版对嵌套 dict/list 的递归下钻（backend_guidance.py:L59-69）——
    # 批准项6"has_*_unsupported_json_features 的递归下钻细节"，精简版只测顶层 schema。
    if not isinstance(schema, dict):
        return False
    # patternProperties is not supported by llguidance
    return "patternProperties" in schema


@dataclass
class GuidanceBackend(StructuredOutputBackend):
    # SOURCE: vllm/v1/structured_output/backend_guidance.py:L86-134
    def __post_init__(self):
        # SOURCE: vllm/v1/structured_output/backend_guidance.py:L88-101
        #
        # SUBTRACTED: `if is_mistral_tokenizer(self.tokenizer): self.ll_tokenizer =
        # self.tokenizer.llg_tokenizer else: ...`（backend_guidance.py:L96-97）——
        # 批准项1，精简版固定走 else 分支。
        # SUBTRACTED: 真实取值路径是 `self.vllm_config.structured_outputs_config.
        # disable_any_whitespace` / `...disable_additional_properties`——VllmConfig
        # 的完整对象图是 ch03 的范围，本章 vllm_config 只是个占位对象；用 getattr
        # 兜底代替精确属性链，语义等价。
        self.disable_any_whitespace = getattr(
            self.vllm_config, "disable_any_whitespace", False
        )
        self.disable_additional_properties = getattr(
            self.vllm_config, "disable_additional_properties", False
        )
        self.ll_tokenizer = llguidance.hf.from_tokenizer(
            self.tokenizer, max(self.vocab_size, len(self.tokenizer))
        )

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        # SOURCE: vllm/v1/structured_output/backend_guidance.py:L103-126
        self.serialized_grammar = serialize_guidance_grammar(
            request_type,
            grammar_spec,
            self.disable_any_whitespace,
            self.disable_additional_properties,
        )

        ll_matcher = llguidance.LLMatcher(
            self.ll_tokenizer,
            self.serialized_grammar,
        )

        r = GuidanceGrammar(
            ll_matcher=ll_matcher,
            ll_tokenizer=self.ll_tokenizer,
            vocab_size=self.vocab_size,
        )

        r.check_error()
        return r

    def allocate_token_bitmask(self, max_num_seqs: int):
        # SOURCE: vllm/v1/structured_output/backend_guidance.py:L128-131
        return llguidance_torch.allocate_token_bitmask(
            max_num_seqs, self.ll_tokenizer.vocab_size
        )

    def destroy(self):
        # SOURCE: vllm/v1/structured_output/backend_guidance.py:L133-134
        pass


@dataclass
class GuidanceGrammar(StructuredOutputGrammar):
    # SOURCE: vllm/v1/structured_output/backend_guidance.py:L137-216
    ll_matcher: Any
    ll_tokenizer: Any
    vocab_size: int
    printed_error: bool = False
    terminated: bool = False
    rollback_lag: int = 0

    def check_error(self):
        # SOURCE: vllm/v1/structured_output/backend_guidance.py:L146-151
        if not self.printed_error:
            err = self.ll_matcher.get_error()
            if err:
                self.printed_error = True
                # SUBTRACTED: logger.warning(...)（可观测性，批准项7）

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        # SOURCE: vllm/v1/structured_output/backend_guidance.py:L153-179
        """Accepts a list of tokens and advances the parser.

        Returns True if the parser was advanced successfully.
        Returns False if the parser failed to advance.
        """
        if self.ll_tokenizer.eos_token in tokens:
            if self.ll_matcher.is_stopped() and not self.terminated:
                # EOS 后回滚要少退一格——guidance 独有的语义差异，与 xgrammar 的
                # rollback 实现不同（xgrammar 没有 rollback_lag 这个概念）。
                self.rollback_lag = 1
            self.terminated = True

        if self.ll_matcher.is_stopped():
            return True

        # SUBTRACTED: jump-forward decoding 的 TODO 注释（backend_guidance.py:
        # L168-173），未实现的功能，纯文档。

        r = self.ll_matcher.consume_tokens(tokens)
        self.check_error()
        return r

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        # SOURCE: vllm/v1/structured_output/backend_guidance.py:L181-196
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

    def rollback(self, num_tokens: int) -> None:
        # SOURCE: vllm/v1/structured_output/backend_guidance.py:L198-203
        if num_tokens > 0:
            self.ll_matcher.rollback(num_tokens - self.rollback_lag)
            self.terminated = False
            self.rollback_lag = 0
            self.check_error()

    def fill_bitmask(self, bitmask, idx: int) -> None:
        # SOURCE: vllm/v1/structured_output/backend_guidance.py:L205-209
        # this will automatically return [EOS] mask if the matcher is stopped
        # or otherwise in an error state
        llguidance_torch.fill_next_token_bitmask(self.ll_matcher, bitmask, idx)
        self.check_error()

    def is_terminated(self) -> bool:
        # SOURCE: vllm/v1/structured_output/backend_guidance.py:L211-212
        return self.terminated

    def reset(self):
        # SOURCE: vllm/v1/structured_output/backend_guidance.py:L214-216
        # This method may be not needed anymore? TODO
        self.ll_matcher.reset()


def serialize_guidance_grammar(
    request_type: StructuredOutputOptions,
    grammar_spec,
    disable_any_whitespace: bool = False,
    disable_additional_properties: bool = False,
) -> str:
    # SOURCE: vllm/v1/structured_output/backend_guidance.py:L219-285
    #
    # SUBTRACTED: disable_additional_properties 分支对 grammar_spec 做
    # process_for_additional_properties 预处理（backend_guidance.py:L228-229）——
    # 依赖已删除的 _walk_json_for_additional_properties，此处不再改写 schema。
    def _process_schema(grammar_spec) -> str:
        # SOURCE: vllm/v1/structured_output/backend_guidance.py:L225-235
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
        if request_type == StructuredOutputOptions.REGEX:
            tp = "regex"
        elif request_type == StructuredOutputOptions.GRAMMAR:
            tp = "grammar"
        elif request_type == StructuredOutputOptions.CHOICE:
            tp = "choice"
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            # SUBTRACTED: structural_tag 序列化分支（backend_guidance.py:L253-277，
            # 按 triggers 匹配 begin 前缀拼 StructTag 列表）——guidance 后端不是本章
            # 深挖对象（xgrammar 才是），且未出现在 must_keep/mechanisms 里。
            raise NotImplementedError(
                "guidance 后端的 structural_tag 序列化未纳入精简版"
                "（未在本章 must_keep 范围内，见 backend_guidance.py:L253-277）"
            )
        else:
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})"
            )
        return llguidance.grammar_from(tp, grammar_spec)
