# SOURCE: vllm/v1/structured_output/backend_xgrammar.py
# 只做减法的忠实精简版（默认后端，正文深挖主线）。
#
# SUBTRACTED: SPDX 版权头；`from vllm.utils.import_utils import LazyLoader` 的
# `xgr = LazyLoader("xgr", globals(), "xgrammar")` 模式（批准项7，工程装饰）——
# 精简版改用 try/except 顶层导入，效果等价（首次真正用到 xgr 属性才会报错，
# 测试里用 monkeypatch 注入假的 xgr 替身）。
try:
    import xgrammar as xgr  # type: ignore
except ImportError:  # pragma: no cover - host 通常无 xgrammar，见 INSTANCE.md
    xgr = None  # type: ignore

import json
from dataclasses import dataclass, field
from typing import Any

from backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from utils import choice_as_grammar

# SUBTRACTED: `from vllm.utils.mistral import is_mistral_tokenizer`（批准项1）；
# `from vllm.v1.structured_output.utils import convert_lark_to_ebnf,
# grammar_is_likely_lark`（Lark→EBNF 方言转换，批准项8，"细节实现"整支不在精简版）。


@dataclass
class XgrammarBackend(StructuredOutputBackend):
    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L34-128
    def __post_init__(self):
        # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L36-75
        #
        # SUBTRACTED: `is_mistral_tokenizer(self.tokenizer)` 分支（backend_xgrammar.py
        # :L41-58，手工拼 TokenizerInfo：RAW/BYTE_FALLBACK vocab_type、stop_token_ids、
        # add_prefix_space）——批准项1，只影响 TokenizerInfo 的构造方式，不改变
        # 编译→状态机→掩码这条主控制流；精简版固定走 else 分支。
        # SUBTRACTED: 真实取值路径是
        # `self.vllm_config.structured_outputs_config.disable_any_whitespace`——
        # VllmConfig 的完整对象图（几十个嵌套 config dataclass）是 ch03 的范围，
        # 本章 vllm_config 只是个占位对象；用 getattr 兜底取代精确属性链，语义等价
        # （字段不存在时按真实默认值 False 处理）。
        self.disable_any_whitespace = getattr(
            self.vllm_config, "disable_any_whitespace", False
        )

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer,
            vocab_size=self.vocab_size,
        )
        self.compiler = xgr.GrammarCompiler(
            tokenizer_info,
            max_threads=8,
            cache_enabled=True,
        )

        # SUBTRACTED: 同上，真实取值路径是
        # `self.vllm_config.speculative_config.num_speculative_tokens`——同样是
        # getattr 兜底代替精确属性链，语义等价。
        self.num_speculative_tokens = 0
        speculative_config = getattr(self.vllm_config, "speculative_config", None)
        if speculative_config is not None:
            self.num_speculative_tokens = getattr(
                speculative_config, "num_speculative_tokens", 0
            )

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L77-122
        #
        # 注意：这里只有五个分支（JSON/JSON_OBJECT/GRAMMAR/REGEX/STRUCTURAL_TAG）——
        # 没有 CHOICE 分支。choice 在校验期（validate_xgrammar_grammar）已被原地
        # 改写成 EBNF grammar，走到这里时已经是 (GRAMMAR, ebnf串)。
        if request_type == StructuredOutputOptions.JSON:
            ctx = self.compiler.compile_json_schema(
                grammar_spec, any_whitespace=not self.disable_any_whitespace
            )
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            ctx = self.compiler.compile_json_schema(
                '{"type": "object"}', any_whitespace=not self.disable_any_whitespace
            )
        elif request_type == StructuredOutputOptions.GRAMMAR:
            ctx = self.compiler.compile_grammar(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            ctx = self.compiler.compile_regex(grammar_spec)
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            # SUBTRACTED: deprecated 的 `"structures" in s_tag` 分支（backend_xgrammar.py
            # :L94-104，把旧格式拆成 StructuralTagItem 列表）——批准项2，同一入口的
            # 旧格式兼容分支，新格式路径 compile_structural_tag(grammar_spec) 已覆盖
            # 语义。
            ctx = self.compiler.compile_structural_tag(grammar_spec)
        else:
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})"
            )

        return XgrammarGrammar(
            matcher=xgr.GrammarMatcher(
                ctx,
                max_rollback_tokens=self.num_speculative_tokens,
            ),
            vocab_size=self.vocab_size,
            ctx=ctx,
        )

    def allocate_token_bitmask(self, max_num_seqs: int):
        # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L124-125
        return xgr.allocate_token_bitmask(max_num_seqs, self.vocab_size)

    def destroy(self):
        # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L127-128
        del self.compiler


@dataclass
class XgrammarGrammar(StructuredOutputGrammar):
    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L131-199
    # NOTE: This would be a generic-enough class for
    # supporting different backends, in the future.
    # For now, just xgrammar.

    vocab_size: int
    matcher: Any = field(hash=False)
    ctx: Any = field(hash=False)
    num_processed_tokens: int = field(
        default_factory=lambda: 0, repr=False, hash=False, init=False
    )
    _is_terminated: bool = field(default=False, repr=False, hash=False)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L148-167
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        if self._is_terminated:
            return False
        for token in tokens:
            if not self.matcher.accept_token(token):
                # SUBTRACTED: logger.error(...)（可观测性，批准项7）
                return False
            self.num_processed_tokens += 1
        self._is_terminated = self.matcher.is_terminated()
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L169-184
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens

    def rollback(self, num_tokens: int) -> None:
        # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L186-189
        self.matcher.rollback(num_tokens)
        self.num_processed_tokens -= num_tokens
        self._is_terminated = self.matcher.is_terminated()

    def fill_bitmask(self, bitmask, idx: int) -> None:
        # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L191-192
        self.matcher.fill_next_token_bitmask(bitmask, idx)

    def is_terminated(self) -> bool:
        # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L194-195
        return self._is_terminated

    def reset(self):
        # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L197-199
        self.num_processed_tokens = 0
        self.matcher.reset()


# SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L200-216
# cf https://github.com/mlc-ai/xgrammar/blob/a32ac892676d2eedc0327416105b9b06edfb94b2/cpp/json_schema_converter.cc
STRING_SUPPORTED_FORMATS = {
    "email",
    "date",
    "time",
    "date-time",
    "duration",
    "ipv4",
    "ipv6",
    "hostname",
    "uuid",
    "uri",
    "uri-reference",
    "uri-template",
    "json-pointer",
    "relative-json-pointer",
}


def has_xgrammar_unsupported_json_features(schema: dict[str, Any]) -> bool:
    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L221-245（顶层三类检测）
    """Check if JSON schema contains features unsupported by xgrammar."""
    # SUBTRACTED: 完整版对 object 的 patternProperties/propertyNames 检查
    # （backend_xgrammar.py:L246-249）与对嵌套 dict/list 的递归下钻
    # （backend_xgrammar.py:L251-259）——批准项6"has_*_unsupported_json_features
    # 的递归下钻细节"，精简版只测顶层 schema，不递归进 properties/items。
    obj = schema
    if not isinstance(obj, dict):
        return False

    # Check for numeric ranges
    if obj.get("type") in ("integer", "number") and ("multipleOf" in obj):
        return True

    # Check for array unsupported keywords
    if obj.get("type") == "array" and any(
        key in obj for key in ("uniqueItems", "contains", "minContains", "maxContains")
    ):
        return True

    # Unsupported keywords for strings
    if (
        obj.get("type") == "string"
        and "format" in obj
        and obj["format"] not in STRING_SUPPORTED_FORMATS
    ):
        return True

    return False


def validate_xgrammar_grammar(sampling_params) -> None:
    # SOURCE: vllm/v1/structured_output/backend_xgrammar.py:L268-354
    """Validate that the request is supported by structured output.

    Raises ValueError if the request is not supported.
    """
    if sampling_params.structured_outputs is None:
        return

    so_params = sampling_params.structured_outputs

    if so_params.regex:
        try:
            xgr.Grammar.from_regex(so_params.regex)
        except Exception as err:
            raise ValueError(
                f"Failed to transform regex into a grammar: {err}"
            ) from err

    if so_params.choice:
        # 【本章关键改写点】choice → EBNF：校验期不只是选后端，还会原地改写请求。
        choice_grammar = choice_as_grammar(so_params.choice)
        try:
            xgr.Grammar.from_ebnf(choice_grammar)
        except Exception as err:
            raise ValueError(
                f"Failed to transform choices into a grammar: {err}"
            ) from err
        so_params.choice = None
        so_params.grammar = choice_grammar
        return

    if so_params.json:
        if isinstance(so_params.json, str):
            try:
                schema = json.loads(so_params.json)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON grammar specification.") from e
        else:
            schema = so_params.json

        if has_xgrammar_unsupported_json_features(schema):
            raise ValueError(
                "The provided JSON schema contains features not supported "
                "by xgrammar."
            )

        try:
            xgr.Grammar.from_json_schema(schema)
        except Exception as err:
            raise ValueError(
                f"Failed to transform json schema into a grammar: {err}"
            ) from err
        return

    if so_params.grammar:
        # SUBTRACTED: `if grammar_is_likely_lark(so_params.grammar): so_params.grammar
        # = convert_lark_to_ebnf(so_params.grammar)`（backend_xgrammar.py:L322-329）——
        # Lark→EBNF 方言转换，批准项8"utils helper 的细节实现"，精简版假定
        # grammar 已是 EBNF，直接试解析。
        try:
            # parse the grammar, but we aren't compiling it.
            xgr.Grammar.from_ebnf(so_params.grammar)
        except Exception as e:
            raise ValueError("Invalid grammar specification.") from e
        return

    if so_params.structural_tag:
        # SUBTRACTED: deprecated 的 `"structures" in s_tag` 分支（同 compile_grammar，
        # 批准项2：解析出 s_tag 后拆成 StructuralTagItem 列表再调用）——精简版只走
        # 新格式路径，但仍保留 json.loads 做合法性预检（真实代码也是先 loads 再判断
        # 走哪条分支，新格式分支最终传给 from_structural_tag 的还是原始字符串）。
        try:
            json.loads(so_params.structural_tag)
            xgr.Grammar.from_structural_tag(so_params.structural_tag)
        except Exception as e:
            raise ValueError("Invalid structural tag specification.") from e
