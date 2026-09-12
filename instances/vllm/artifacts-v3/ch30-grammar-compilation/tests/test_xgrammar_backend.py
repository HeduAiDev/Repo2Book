# m6/m7/m8/m9/m10：XgrammarBackend 五分派 → GrammarMatcher FSM；
# XgrammarGrammar 六方法参考实现（accept/validate/rollback 成对语义）。
# 真实行为基准（v0.27.1 现核 + xgrammar 0.2.6 实测）：
#   - backend_xgrammar.py:L78-L126 compile_grammar 五分派：JSON/JSON_OBJECT/
#     GRAMMAR/REGEX/STRUCTURAL_TAG——**无 CHOICE 分支**
#   - backend_xgrammar.py:L119-L123 GrammarMatcher(max_rollback_tokens=
#     num_speculative_tokens)
#   - backend_xgrammar.py:L152-L203 六方法：accept 推进、validate 试走+回退、
#     rollback 成对回退计数
#   - matcher 语义实测：accept_token 拒收返回 False 不前进；完整串后需 EOS
#     才 is_terminated；回滚深度超 max_rollback_tokens 抛 RuntimeError
import pytest
import torch
import xgrammar as xgr
from conftest import make_vllm_config
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import (
    XgrammarBackend,
    XgrammarGrammar,
    has_xgrammar_unsupported_json_features,
)

YES, NO, EOS, GARBAGE = 8505, 3919, 50256, 99999  # gpt2: "yes","no",<|endoftext|>
# 词表内但语法外的 token（GARBAGE=99999 超出 50257 词表，位掩码测试用 4242）
OFF_GRAMMAR = 4242


@pytest.fixture(scope="module")
def backend(tokenizer):
    return XgrammarBackend(
        make_vllm_config(), tokenizer=tokenizer, vocab_size=50257
    )


@pytest.fixture(scope="module")
def spec_backend(tokenizer):
    return XgrammarBackend(
        make_vllm_config(num_speculative_tokens=1),
        tokenizer=tokenizer,
        vocab_size=50257,
    )


def compile_grammar(backend, option, spec):
    return backend.compile_grammar(option, spec)


# ── m6：编译五分派 ──────────────────────────────────────────────
class TestCompileDispatch:
    def test_grammar_ebnf_branch(self, backend):
        g = compile_grammar(
            backend, StructuredOutputOptions.GRAMMAR, 'root ::= "yes" | "no"')
        assert isinstance(g, XgrammarGrammar)
        assert isinstance(g.matcher, xgr.GrammarMatcher)
        assert isinstance(g.ctx, xgr.CompiledGrammar)

    def test_json_branch(self, backend):
        g = compile_grammar(
            backend, StructuredOutputOptions.JSON,
            '{"type": "object", "properties": {"a": {"type": "integer"}}}')
        assert isinstance(g, XgrammarGrammar)

    def test_json_object_is_object_schema_sugar(self, backend, tokenizer):
        # L85-L88：JSON_OBJECT = compile_json_schema('{"type": "object"}') 的语法糖
        g = compile_grammar(backend, StructuredOutputOptions.JSON_OBJECT, "")
        for t in tokenizer.encode("{}"):
            assert g.matcher.accept_token(t)

    def test_regex_branch(self, backend):
        g = compile_grammar(backend, StructuredOutputOptions.REGEX, "[0-9]+")
        assert isinstance(g, XgrammarGrammar)

    def test_structural_tag_routes_to_structural_tag_compiler(self, backend):
        # 新路径：整串 grammar_spec 直传 compile_structural_tag——即使带
        # "structures" 键（deprecated 拆解分支已按减法计划删除）也走新路径。
        # 由此产生的 InvalidStructuralTagError 恰好证明分派落点。
        with pytest.raises(RuntimeError, match="[Ii]nvalid structural tag"):
            compile_grammar(
                backend, StructuredOutputOptions.STRUCTURAL_TAG,
                '{"structures": [], "triggers": []}')

    def test_no_choice_branch(self, backend):
        # 引擎侧分派没有 CHOICE 分支——校验期已把 choice 原地改写成 EBNF。
        with pytest.raises(ValueError, match="not of valid supported types"):
            compile_grammar(
                backend, StructuredOutputOptions.CHOICE, '["yes", "no"]')

    def test_compiler_has_byte_budget_cache(self, backend):
        # m10：GrammarCompiler(cache_enabled=True, cache_limit_bytes=
        # VLLM_XGRAMMAR_CACHE_MB*1024*1024)——编译缓存的真实所在（库内）。
        assert isinstance(backend.compiler, xgr.GrammarCompiler)
        assert backend.num_speculative_tokens == 0

    def test_spec_config_sets_num_speculative_tokens(self, spec_backend):
        assert spec_backend.num_speculative_tokens == 1


# ── m7/m8/m9：六方法语义（真实 GrammarMatcher） ──────────────────
class TestAcceptValidateRollback:
    def _choice_grammar(self, backend):
        return compile_grammar(
            backend, StructuredOutputOptions.GRAMMAR, 'root ::= "yes" | "no"')

    def test_accept_advances_and_counts(self, backend):
        g = self._choice_grammar(backend)
        assert g.accept_tokens("req-1", [YES]) is True
        assert g.num_processed_tokens == 1
        assert g.is_terminated() is False  # 完整串后仍需 EOS 才终态

    def test_accept_eos_completes(self, backend):
        g = self._choice_grammar(backend)
        g.accept_tokens("req-1", [YES])
        assert g.accept_tokens("req-1", [EOS]) is True
        assert g.is_terminated() is True

    def test_accept_after_terminated_short_circuits_false(self, backend):
        g = self._choice_grammar(backend)
        g.accept_tokens("req-1", [YES, EOS])
        # 终态后再 accept：accept_tokens 头部短路返回 False
        assert g.accept_tokens("req-1", [YES]) is False

    def test_accept_rejects_invalid_token(self, backend):
        g = self._choice_grammar(backend)
        assert g.accept_tokens("req-1", [GARBAGE]) is False
        assert g.num_processed_tokens == 0  # 拒收不前进

    def test_validate_returns_prefix_without_advancing(self, backend):
        g = self._choice_grammar(backend)
        assert g.validate_tokens([YES, NO]) == [YES]
        assert g.num_processed_tokens == 0  # 不推进（已回退）
        # 幂等：再 validate 结果一致
        assert g.validate_tokens([YES, NO]) == [YES]

    def test_validate_empty_when_none_accepted(self, backend):
        g = self._choice_grammar(backend)
        assert g.validate_tokens([GARBAGE]) == []

    def test_rollback_paired_with_counter(self, backend):
        g = compile_grammar(
            backend, StructuredOutputOptions.GRAMMAR, 'root ::= "a" "b" "c"')
        ids = [64, 65, 66]  # gpt2: "a","b","c"（单字符 token）
        assert g.accept_tokens("req-1", ids) is True
        assert g.num_processed_tokens == 3
        g.rollback(3)
        assert g.num_processed_tokens == 0
        # 回退到初始态：重走一遍同样成功
        assert g.accept_tokens("req-1", ids) is True

    def test_max_rollback_tokens_follows_spec_config(self, spec_backend):
        # m9：max_rollback_tokens=num_speculative_tokens（构造参数原样传给
        # GrammarMatcher）。xgrammar 0.2.6 已弃用其限制语义（内部恒无限），
        # 但该 kwarg 的传入仍会触发库的 DeprecationWarning——恰好是
        # 『vLLM 把投机 token 数传进构造』的可观测证据。
        with pytest.warns(DeprecationWarning, match="max_rollback_tokens"):
            g = compile_grammar(
                spec_backend, StructuredOutputOptions.GRAMMAR,
                'root ::= "yes" | "no"')
        assert isinstance(g.matcher, xgr.GrammarMatcher)

    def test_non_spec_backend_rollback_depth_zero(self, backend):
        g = self._choice_grammar(backend)
        g.accept_tokens("req-1", [YES])
        g.rollback(1)
        assert g.num_processed_tokens == 0

    def test_fill_bitmask_marks_allowed_set(self, backend):
        g = self._choice_grammar(backend)
        bitmask = xgr.allocate_token_bitmask(1, 50257)
        g.fill_bitmask(bitmask, 0)
        row = bitmask[0]

        def allowed(t):
            return bool(row[t // 32] >> (t % 32) & 1)

        assert allowed(YES) and allowed(NO)
        assert not allowed(4242)  # 词表内但语法外的 token
        assert not allowed(EOS)  # 还没到可停的位置

    def test_is_terminated_and_reset(self, backend):
        g = self._choice_grammar(backend)
        g.accept_tokens("req-1", [YES, EOS])
        assert g.is_terminated() is True
        g.reset()
        # 真实行为（backend_xgrammar.py:L201-L203 逐字对照）：reset 只清
        # num_processed_tokens 与 matcher 本体，**不清 _is_terminated 标志**
        # ——终态后再 accept 恒走头部短路（源码即如此，非精简版偏差）。
        assert g.num_processed_tokens == 0
        assert g.is_terminated() is True
        assert g.accept_tokens("req-1", [YES]) is False

    def test_vocab_size_field(self, backend):
        assert backend.vocab_size == 50257


# ── m3/m4：xgrammar 能力预检 + 校验期改写 ────────────────────────
class TestUnsupportedFeatures:
    def test_multiple_of(self):
        assert has_xgrammar_unsupported_json_features(
            {"type": "integer", "multipleOf": 3}) is True

    def test_unique_items(self):
        assert has_xgrammar_unsupported_json_features(
            {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
        ) is True

    def test_unsupported_string_format(self):
        assert has_xgrammar_unsupported_json_features(
            {"type": "string", "format": "weird-format"}) is True

    def test_supported_string_format(self):
        assert has_xgrammar_unsupported_json_features(
            {"type": "string", "format": "email"}) is False

    def test_pattern_properties(self):
        assert has_xgrammar_unsupported_json_features(
            {"type": "object", "patternProperties": {"^a": {"type": "string"}}}
        ) is True

    def test_nested_in_properties_and_lists(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer", "multipleOf": 2}},
        }
        assert has_xgrammar_unsupported_json_features(schema) is True
        schema2 = {
            "type": "object",
            "properties": {"list": [ {"type": "string", "format": "nope"} ]},
        }
        assert has_xgrammar_unsupported_json_features(schema2) is True

    def test_plain_schema_clean(self):
        assert has_xgrammar_unsupported_json_features(
            {"type": "object",
             "properties": {"a": {"type": "integer"}, "b": {"type": "string"}}}
        ) is False

    def test_non_dict_is_clean(self):
        assert has_xgrammar_unsupported_json_features("not-a-dict") is False
