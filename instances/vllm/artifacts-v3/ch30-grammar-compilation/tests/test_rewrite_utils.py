# m4 + m19：校验期改写工具箱（choice→EBNF / lark 判定 / lark→EBNF）与
# ReDoS 超时护栏。
# 真实行为基准（v0.27.1 现核）：
#   - utils.py:L553-L561 choice_as_grammar：转义双引号/反斜杠，产出
#     `root ::= "A" | "B"` 一条 EBNF
#   - utils.py:L391-L420 grammar_is_likely_lark：无 ::= 即 Lark
#   - utils.py:L423-L550 convert_lark_to_ebnf：root 规则注入 + 引号风格转换 +
#     未定义引用校验
#   - utils.py:L48-L83 compile_regex_with_timeout：VLLM_REGEX_COMPILATION_TIMEOUT_S
#     默认 5s、<=0 直通、超时 ValueError
import time

import pytest
from vllm.v1.structured_output.utils import (
    choice_as_grammar,
    compile_regex_with_timeout,
    convert_lark_to_ebnf,
    grammar_is_likely_lark,
)


class TestChoiceAsGrammar:
    def test_two_choices(self):
        assert choice_as_grammar(["yes", "no"]) == 'root ::= "yes" | "no"'

    def test_single_choice(self):
        assert choice_as_grammar(["a"]) == 'root ::= "a"'

    def test_escapes_quotes_and_backslashes(self):
        # 转义规则：双引号与反斜杠前加反斜杠（escape_ebnf_string）
        g = choice_as_grammar(['He said "hi"', "back\\slash"])
        assert '"He said \\"hi\\""' in g
        assert '"back\\\\slash"' in g
        assert g.startswith("root ::= ")

    def test_output_is_valid_ebnf_for_xgrammar(self):
        import xgrammar as xgr
        xgr.Grammar.from_ebnf(choice_as_grammar(["yes", "no"]))


class TestLarkDetectionAndConversion:
    def test_lark_detected_by_missing_ebnf_assign(self):
        assert grammar_is_likely_lark("rule: 'abc'") is True
        assert grammar_is_likely_lark('rule ::= "abc"') is False

    def test_comment_lines_ignored_in_detection(self):
        # 注释行（# 或 //）剥掉后为空 → 不影响判定
        assert grammar_is_likely_lark("# just a comment\nrule ::= 'x'") is False
        assert grammar_is_likely_lark("rule: 'abc' // trailing comment") is True

    def test_empty_string_not_lark(self):
        assert grammar_is_likely_lark("") is False

    def test_convert_basic(self):
        assert convert_lark_to_ebnf("rule: 'hello'") == 'root ::= rule\nrule ::= "hello"'

    def test_convert_alternatives(self):
        out = convert_lark_to_ebnf("rule: 'a'\n| 'b'")
        assert out == 'root ::= rule\nrule ::= "a" | "b"'

    def test_convert_quotes_validated(self):
        with pytest.raises(ValueError, match="[Mm]ismatched quotes"):
            convert_lark_to_ebnf("rule: 'abc")

    def test_convert_undefined_reference_rejected(self):
        with pytest.raises(ValueError, match="Referenced rules are not defined"):
            convert_lark_to_ebnf("rule: defined_rule 'a'")

    def test_convert_requires_string(self):
        with pytest.raises(ValueError, match="must be a string"):
            convert_lark_to_ebnf(123)

    def test_convert_empty_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            convert_lark_to_ebnf("   ")

    def test_start_rule_wins_root(self):
        out = convert_lark_to_ebnf("first: 'a'\nstart: 'b'")
        assert out.splitlines()[0] == "root ::= start"

    def test_round_trip_through_xgrammar(self):
        import xgrammar as xgr
        ebnf = convert_lark_to_ebnf("answer: 'yes'")
        xgr.Grammar.from_ebnf(ebnf)


class TestCompileRegexWithTimeout:
    def test_passes_pattern_through(self):
        import xgrammar as xgr
        ctx = compile_regex_with_timeout(xgr.Grammar.from_regex, "[0-9]+")
        assert ctx is not None

    def test_disabled_when_timeout_zero(self, monkeypatch):
        # VLLM_REGEX_COMPILATION_TIMEOUT_S<=0 → 直通 fn（L65-L66）
        monkeypatch.setenv("VLLM_REGEX_COMPILATION_TIMEOUT_S", "0")
        seen = []
        compile_regex_with_timeout(seen.append, "pat")
        assert seen == ["pat"]

    def test_timeout_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("VLLM_REGEX_COMPILATION_TIMEOUT_S", "1")

        def slow(pattern):
            time.sleep(2.5)
            return pattern

        t0 = time.monotonic()
        with pytest.raises(ValueError, match="timed out after 1s"):
            compile_regex_with_timeout(slow, "(a+)+b")
        assert time.monotonic() - t0 < 2.2  # 超时即断，不等慢函数跑完

    def test_fast_compile_within_timeout(self, monkeypatch):
        monkeypatch.setenv("VLLM_REGEX_COMPILATION_TIMEOUT_S", "5")

        def fast(pattern):
            return pattern.upper()

        assert compile_regex_with_timeout(fast, "abc") == "ABC"

    def test_timeout_error_mentions_nested_quantifiers(self, monkeypatch):
        monkeypatch.setenv("VLLM_REGEX_COMPILATION_TIMEOUT_S", "1")

        def slow(pattern):
            time.sleep(2.0)
            return pattern

        with pytest.raises(ValueError, match="nested"):
            compile_regex_with_timeout(slow, "x+")
