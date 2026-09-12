# m4/m18：两个校验器——validate_xgrammar_grammar（试编 + choice→EBNF /
# lark→EBNF 原地改写）与 validate_guidance_grammar（serialize + LLMatcher 预检）。
# 真实行为基准（v0.27.1 现核）：
#   - backend_xgrammar.py:L272-L364：regex 试编、choice 改写（L293-L303）、
#     json 预检+试编、grammar 的 lark 判定+转换+试编、structural_tag 试编
#   - backend_guidance.py:L224-L290 serialize_guidance_grammar 六形统一序列化
#     （guidance 原生支持 choice——tp='choice'，与 xgrammar 的改写路线对照）
#   - backend_guidance.py:L293-L303 validate_guidance_grammar → err 即 ValueError
import pytest
from conftest import make_vllm_config
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output.backend_guidance import (
    GuidanceBackend,
    has_guidance_unsupported_json_features,
    serialize_guidance_grammar,
    validate_guidance_grammar,
)
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar


def _sp(so) -> SamplingParams:
    return SamplingParams(structured_outputs=so)


# ── validate_xgrammar_grammar：试编 + 原地改写 ──────────────────
class TestValidateXgrammarGrammar:
    def test_none_structured_outputs_is_noop(self):
        validate_xgrammar_grammar(SamplingParams())  # 不抛即过

    def test_valid_regex_passes(self):
        sp = _sp(StructuredOutputsParams(regex="[0-9]+"))
        validate_xgrammar_grammar(sp)

    def test_bad_regex_raises(self):
        sp = _sp(StructuredOutputsParams(regex="([unclosed"))
        with pytest.raises(ValueError, match="Failed to transform regex"):
            validate_xgrammar_grammar(sp)

    def test_choice_rewritten_in_place_to_ebnf(self):
        so = StructuredOutputsParams(choice=["yes", "no"])
        sp = _sp(so)
        validate_xgrammar_grammar(sp)
        assert so.choice is None
        assert so.grammar == 'root ::= "yes" | "no"'

    def test_invalid_json_string_raises(self):
        so = StructuredOutputsParams(json="{not valid json")
        with pytest.raises(ValueError, match="Invalid JSON grammar"):
            validate_xgrammar_grammar(_sp(so))

    def test_unsupported_json_features_raise(self):
        so = StructuredOutputsParams(json='{"type": "integer", "multipleOf": 3}')
        with pytest.raises(ValueError, match="not supported by xgrammar"):
            validate_xgrammar_grammar(_sp(so))

    def test_valid_json_schema_passes(self):
        so = StructuredOutputsParams(json='{"type": "object"}')
        validate_xgrammar_grammar(_sp(so))

    def test_json_dict_accepted_directly(self):
        so = StructuredOutputsParams(json={"type": "object"})
        validate_xgrammar_grammar(_sp(so))

    def test_ebnf_grammar_passes_untouched(self):
        so = StructuredOutputsParams(grammar='root ::= "yes" | "no"')
        sp = _sp(so)
        validate_xgrammar_grammar(sp)
        assert so.grammar == 'root ::= "yes" | "no"'  # EBNF 不改写

    def test_lark_grammar_converted_in_place(self):
        so = StructuredOutputsParams(grammar="answer: 'yes'")
        sp = _sp(so)
        validate_xgrammar_grammar(sp)
        assert so.grammar == 'root ::= answer\nanswer ::= "yes"'

    def test_invalid_ebnf_raises(self):
        # 带 ::= 的坏 EBNF：不过 lark 判定，from_ebnf 试编失败
        so = StructuredOutputsParams(grammar="root ::= ((")
        with pytest.raises(ValueError, match="Invalid grammar specification"):
            validate_xgrammar_grammar(_sp(so))

    def test_plain_words_detected_as_lark_conversion_fails(self):
        # 无 ::= 的任意串按 Lark 处理：改写失败报 Lark 转换错（真实行为）
        so = StructuredOutputsParams(grammar="this is not a grammar")
        with pytest.raises(ValueError, match="Lark to EBNF"):
            validate_xgrammar_grammar(_sp(so))

    def test_bad_structural_tag_json_raises(self):
        so = StructuredOutputsParams(structural_tag="{not json")
        with pytest.raises(ValueError, match="Invalid structural tag"):
            validate_xgrammar_grammar(_sp(so))

    def test_structural_tag_deprecated_branch_removed(self):
        # 减法边界：structures/triggers 拆解（deprecated 分支）已删——
        # 带 "structures" 的旧格式走新路径 from_structural_tag(str)，
        # 新路径对旧格式报结构错（不是走 StructuralTagItem 拆解成功）。
        so = StructuredOutputsParams(
            structural_tag='{"structures": [{"begin": "<a>", "schema": '
            '"{\\"type\\": \\"integer\\"}", "end": "</a>"}], "triggers": ["<a>"]}')
        with pytest.raises(ValueError, match="Invalid structural tag"):
            validate_xgrammar_grammar(_sp(so))


# ── serialize_guidance_grammar：六形统一序列化 ──────────────────
class TestSerializeGuidanceGrammar:
    def test_json_schema_serialized(self):
        out = serialize_guidance_grammar(
            StructuredOutputOptions.JSON, '{"type": "object"}')
        assert "json_schema" in out

    def test_json_object_sugar(self):
        out = serialize_guidance_grammar(StructuredOutputOptions.JSON_OBJECT, "")
        assert "json_schema" in out and "object" in out

    def test_choice_is_native_for_guidance(self):
        # 与 xgrammar 的改写路线对照：guidance 原生支持 choice（tp='choice'），
        # 序列化产物就是一条选择语法，不改写请求。
        out = serialize_guidance_grammar(
            StructuredOutputOptions.CHOICE, '["yes", "no"]')
        assert out == 'start: "yes" | "no"'

    def test_regex_serialized(self):
        out = serialize_guidance_grammar(
            StructuredOutputOptions.REGEX, "[0-9]+")
        assert isinstance(out, str) and len(out) > 0

    def test_grammar_serialized(self):
        out = serialize_guidance_grammar(
            StructuredOutputOptions.GRAMMAR, 'start: "yes" | "no"')
        assert out == 'start: "yes" | "no"'

    def test_invalid_option_raises(self):
        # STRUCTURAL_TAG 分支已按减法计划删除——落到 else 报非法类型
        with pytest.raises(ValueError, match="not of valid supported types"):
            serialize_guidance_grammar(
                StructuredOutputOptions.STRUCTURAL_TAG, "{}")


class TestHasGuidanceUnsupportedJsonFeatures:
    def test_pattern_properties_top_level(self):
        assert has_guidance_unsupported_json_features(
            {"type": "object", "patternProperties": {"^a": {}}}) is True

    def test_pattern_properties_nested(self):
        schema = {"type": "object", "properties": {
            "inner": {"type": "object",
                      "patternProperties": {"^x": {"type": "string"}}}}}
        assert has_guidance_unsupported_json_features(schema) is True

    def test_nested_in_list(self):
        schema = {"type": "object", "properties": {
            "l": [{"patternProperties": {}}]}}
        assert has_guidance_unsupported_json_features(schema) is True

    def test_multiple_of_is_fine_for_guidance(self):
        # 与 xgrammar 的能力差异：guidance 支持 multipleOf（auto 阶梯由此降级）
        assert has_guidance_unsupported_json_features(
            {"type": "integer", "multipleOf": 3}) is False

    def test_plain_schema_clean(self):
        assert has_guidance_unsupported_json_features(
            {"type": "object", "properties": {"a": {"type": "string"}}}) is False

    def test_non_dict_is_clean(self):
        assert has_guidance_unsupported_json_features([1, 2]) is False


# ── validate_guidance_grammar：LLMatcher 预检 ────────────────────
class TestValidateGuidanceGrammar:
    def test_none_structured_outputs_is_noop(self):
        validate_guidance_grammar(SamplingParams())

    def test_valid_schema_passes(self):
        sp = _sp(StructuredOutputsParams(regex="[0-9]+"))
        validate_guidance_grammar(sp)

    def test_bad_grammar_raises_with_error_text(self):
        # 真实行为：GRANNAR 形先经 grammar_from('grammar', ...) 的 GBNF→
        # Lark 转换，坏语法在序列化段就报错（"Failed to convert the grammar
        # from GBNF to Lark"）
        sp = _sp(StructuredOutputsParams(grammar="this is (( not a grammar"))
        with pytest.raises(ValueError, match="GBNF to Lark"):
            validate_guidance_grammar(sp)

    def test_choice_route_via_serialize(self):
        # 走 serialize 的 choice 原生路线后 LLMatcher 能接受
        sp = _sp(StructuredOutputsParams(choice=["yes", "no"]))
        validate_guidance_grammar(sp)


# ── m18：guidance 第二实现（同契约异实现） ───────────────────────
class TestGuidanceBackendContract:
    @pytest.fixture(scope="class")
    def backend(self, tokenizer):
        return GuidanceBackend(
            make_vllm_config(), tokenizer=tokenizer, vocab_size=50257
        )

    def test_compile_choice_grammar(self, backend):
        g = backend.compile_grammar(
            StructuredOutputOptions.CHOICE, '["yes", "no"]')
        assert hasattr(g, "rollback_lag")
        assert g.accept_tokens("req-1", [8505]) is True  # "yes"

    def test_rollback_lag_on_eos_after_stop(self, backend):
        # guidance 的 EOS 后回滚少退一格：consume 完整 choice 后 matcher
        # is_stopped；此时 EOS 到达 → terminated=True、rollback_lag=1；
        # 随后 rollback(1) 实际只回退 1-1=0 格，并把 lag 清零。
        g = backend.compile_grammar(
            StructuredOutputOptions.CHOICE, '["yes", "no"]')
        assert g.accept_tokens("req-1", [8505]) is True
        assert g.accept_tokens("req-1", [50256]) is True  # EOS
        assert g.terminated is True
        assert g.rollback_lag == 1
        g.rollback(1)
        assert g.rollback_lag == 0
        assert g.terminated is False

    def test_validate_tokens_returns_prefix(self, backend):
        g = backend.compile_grammar(
            StructuredOutputOptions.CHOICE, '["yes", "no"]')
        assert g.validate_tokens([8505, 3919]) == [8505]

    def test_validate_tokens_empty_and_stopped(self, backend):
        g = backend.compile_grammar(
            StructuredOutputOptions.CHOICE, '["yes", "no"]')
        assert g.validate_tokens([]) == []
        g.accept_tokens("req-1", [8505, 50256])
        # is_stopped 后不再验证（返回 []）
        assert g.validate_tokens([8505]) == []

    def test_accept_rejects_off_grammar_token(self, backend):
        g = backend.compile_grammar(
            StructuredOutputOptions.CHOICE, '["yes", "no"]')
        assert g.accept_tokens("req-1", [99999]) is False

    def test_fill_bitmask_via_llguidance_torch(self, backend):
        import llguidance.torch as llguidance_torch
        g = backend.compile_grammar(
            StructuredOutputOptions.CHOICE, '["yes", "no"]')
        bitmask = llguidance_torch.allocate_token_bitmask(1, 50257)
        g.fill_bitmask(bitmask, 0)
        row = bitmask[0]
        assert bool(row[8505 // 32] >> (8505 % 32) & 1)   # yes 允许
        assert not bool(row[4242 // 32] >> (4242 % 32) & 1)  # 词表内语法外禁

    def test_is_terminated_follows_terminated_flag(self, backend):
        g = backend.compile_grammar(
            StructuredOutputOptions.CHOICE, '["yes", "no"]')
        assert g.is_terminated() is False
        g.accept_tokens("req-1", [8505, 50256])
        assert g.is_terminated() is True

    def test_allocate_token_bitmask_and_destroy(self, backend):
        bm = backend.allocate_token_bitmask(4)
        assert bm.shape[0] == 4
        backend.destroy()
