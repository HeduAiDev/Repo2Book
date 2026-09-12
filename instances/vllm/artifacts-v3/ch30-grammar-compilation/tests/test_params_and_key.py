# 站 1 + m1/m2/m3：StructuredOutputsParams 六选一互斥、structured_output_key
# 归一、前端校验 _validate_structured_outputs 的 auto 降级阶梯与冲突检查。
# 真实行为基准（v0.27.1 现核）：
#   - sampling_params.py:L90-L111 __post_init__ 双向报错（count>1 / count<1）
#   - request.py:L82-L103 get_structured_output_key 的 json.dumps 归一
#   - sampling_params.py:L949-L966 引擎单后端冲突检查 + _backend_was_auto 放行
#   - sampling_params.py:L1043-L1082 auto 阶梯：xgrammar 试编失败 → 降 guidance
import json

import pytest
from conftest import make_vllm_config
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.request import (
    StructuredOutputRequest,
    get_structured_output_key,
)


def _tokenizer():
    from transformers import AutoTokenizer
    from conftest import TOKENIZER_NAME
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


# ── m2：六选一互斥 ──────────────────────────────────────────────
class TestSixWayMutualExclusion:
    def test_multiple_constraints_rejected(self):
        with pytest.raises(VLLMValidationError, match="only use one kind"):
            StructuredOutputsParams(json='{"a": 1}', regex="[0-9]+")

    def test_zero_constraints_rejected(self):
        with pytest.raises(VLLMValidationError, match="must use one kind"):
            StructuredOutputsParams()

    def test_single_constraint_ok(self):
        p = StructuredOutputsParams(regex="[0-9]+")
        assert p.regex == "[0-9]+"
        assert not p.all_constraints_none()

    def test_all_constraints_none_semantics(self):
        # all_constraints_none 被 from_sampling_params 用来判『无约束』；
        # 构造期 count<1 会拒，绕过校验直接验证其语义。
        p = StructuredOutputsParams.__new__(StructuredOutputsParams)
        for f in ("json", "regex", "choice", "grammar", "json_object",
                  "structural_tag"):
            setattr(p, f, None)
        assert p.all_constraints_none() is True
        p.regex = "x"
        assert p.all_constraints_none() is False

    def test_private_backend_fields_default(self):
        p = StructuredOutputsParams(regex="x")
        assert p._backend is None
        assert p._backend_was_auto is False


# ── m2：structured_output_key 归一（dict/list → json.dumps） ──────
class TestStructuredOutputKey:
    def test_json_dict_normalized_via_dumps(self):
        p = StructuredOutputsParams(json={"type": "object"})
        assert get_structured_output_key(p) == (
            StructuredOutputOptions.JSON,
            json.dumps({"type": "object"}),
        )

    def test_json_str_passthrough(self):
        p = StructuredOutputsParams(json='{"type": "object"}')
        assert get_structured_output_key(p) == (
            StructuredOutputOptions.JSON,
            '{"type": "object"}',
        )

    def test_json_object_empty_spec(self):
        p = StructuredOutputsParams(json_object=True)
        assert get_structured_output_key(p) == (StructuredOutputOptions.JSON_OBJECT, "")

    def test_regex(self):
        p = StructuredOutputsParams(regex="[0-9]+")
        assert get_structured_output_key(p) == (
            StructuredOutputOptions.REGEX, "[0-9]+")

    def test_choice_list_normalized_via_dumps(self):
        p = StructuredOutputsParams(choice=["yes", "no"])
        assert get_structured_output_key(p) == (
            StructuredOutputOptions.CHOICE,
            '["yes", "no"]',
        )

    def test_grammar(self):
        p = StructuredOutputsParams(grammar='root ::= "yes" | "no"')
        assert get_structured_output_key(p) == (
            StructuredOutputOptions.GRAMMAR,
            'root ::= "yes" | "no"',
        )

    def test_structural_tag(self):
        p = StructuredOutputsParams(structural_tag='{"type":"stag"}')
        assert get_structured_output_key(p) == (
            StructuredOutputOptions.STRUCTURAL_TAG,
            '{"type":"stag"}',
        )

    def test_key_is_cached_property_per_request(self):
        # m10 纠偏点：structured_output_key 是**每请求** cached_property，
        # 不是跨请求编译缓存键——同一请求两次访问返回同一对象。
        sp = SamplingParams(structured_outputs=StructuredOutputsParams(regex="a+"))
        req = StructuredOutputRequest.from_sampling_params(sp)
        assert req.structured_output_key is req.structured_output_key

    def test_no_valid_param_raises(self):
        p = StructuredOutputsParams.__new__(StructuredOutputsParams)
        for f in ("json", "regex", "choice", "grammar", "json_object",
                  "structural_tag"):
            setattr(p, f, None)
        with pytest.raises(ValueError, match="No valid structured output"):
            get_structured_output_key(p)


# ── 站 2：from_sampling_params ──────────────────────────────────
class TestFromSamplingParams:
    def test_none_params_gives_none(self):
        assert StructuredOutputRequest.from_sampling_params(None) is None

    def test_no_constraints_gives_none(self):
        sp = SamplingParams(structured_outputs=None)
        assert StructuredOutputRequest.from_sampling_params(sp) is None

    def test_with_constraint_gives_request(self):
        sp = SamplingParams(structured_outputs=StructuredOutputsParams(regex="a"))
        req = StructuredOutputRequest.from_sampling_params(sp)
        assert req is not None
        assert req.params is sp.structured_outputs
        assert req._grammar is None
        assert req.reasoning_ended is None
        assert req.reasoning_end_token_index is None
        assert req.reasoner is None


# ── 站 1/m3：前端校验与 auto 阶梯 ────────────────────────────────
GOOD_SCHEMA = '{"type": "object", "properties": {"a": {"type": "integer"}}}'
# multipleOf 是 xgrammar 预检明确不支持的特性（backend_xgrammar.py:L233-L234），
# 但 guidance 支持——auto 阶梯由此降级到 guidance（真实行为）。
MULTIPLEOF_SCHEMA = '{"type": "integer", "multipleOf": 3}'


class TestValidateStructuredOutputs:
    def _validate(self, so, *, backend="auto", is_diffusion=False, tokenizer=None):
        sp = SamplingParams(structured_outputs=so)
        cfg = make_vllm_config(
            backend=backend, is_diffusion=is_diffusion
        )
        sp._validate_structured_outputs(
            cfg.model_config, cfg.structured_outputs_config,
            tokenizer if tokenizer is not None else _tokenizer(),
        )
        return sp

    def test_no_config_no_op(self):
        sp = SamplingParams(structured_outputs=StructuredOutputsParams(regex="a"))
        cfg = make_vllm_config()
        # structured_outputs_config=None：函数第一行直接 return（L929-L930）
        sp._validate_structured_outputs(cfg.model_config, None, _tokenizer())

    def test_no_structured_outputs_no_op(self):
        sp = SamplingParams(structured_outputs=None)
        cfg = make_vllm_config()
        sp._validate_structured_outputs(
            cfg.model_config, cfg.structured_outputs_config, _tokenizer())

    def test_diffusion_rejected(self):
        with pytest.raises(VLLMValidationError, match="not yet supported for diffusion"):
            self._validate(StructuredOutputsParams(regex="a"), is_diffusion=True)

    def test_no_tokenizer_rejected(self):
        so = StructuredOutputsParams(regex="a")
        sp = SamplingParams(structured_outputs=so)
        cfg = make_vllm_config()
        with pytest.raises(VLLMValidationError, match="requires a tokenizer"):
            sp._validate_structured_outputs(
                cfg.model_config, cfg.structured_outputs_config, None)

    def test_empty_choice_rejected(self):
        # choice=[] 能过 __post_init__（choice is not None → count=1），
        # 由内容预检拒绝（L969-L976）。
        with pytest.raises(VLLMValidationError, match="cannot be an empty list"):
            self._validate(StructuredOutputsParams(choice=[]))

    def test_empty_grammar_string_rejected(self):
        with pytest.raises(VLLMValidationError, match="cannot be an empty string"):
            self._validate(StructuredOutputsParams(grammar="   "))

    def test_empty_json_string_rejected(self):
        with pytest.raises(VLLMValidationError, match="cannot be an empty string"):
            self._validate(StructuredOutputsParams(json="  "))

    def test_json_object_false_rejected(self):
        with pytest.raises(VLLMValidationError, match="must be True if set"):
            self._validate(StructuredOutputsParams(json_object=False))

    def test_backend_conflict_rejected(self):
        so = StructuredOutputsParams(regex="a+")
        so._backend = "guidance"
        with pytest.raises(VLLMValidationError, match="Request-level structured"):
            self._validate(so, backend="xgrammar")

    def test_backend_was_auto_allows_reuse(self):
        # 复用 params：上次 auto 选了 xgrammar，这次引擎 auto —— 记账位放行。
        so = StructuredOutputsParams(regex="a+")
        so._backend = "xgrammar"
        so._backend_was_auto = True
        self._validate(so, backend="auto")  # 不抛即过
        assert so._backend in ("xgrammar", "guidance")

    def test_first_time_sets_backend_from_engine_config(self):
        so = StructuredOutputsParams(regex="[0-9]+")
        self._validate(so, backend="xgrammar")
        assert so._backend == "xgrammar"
        assert so._backend_was_auto is False

    def test_auto_ladder_picks_xgrammar_for_plain_schema(self):
        so = StructuredOutputsParams(json=GOOD_SCHEMA)
        self._validate(so, backend="auto")
        assert so._backend == "xgrammar"
        assert so._backend_was_auto is True

    def test_auto_ladder_falls_back_to_guidance_on_multiple_of(self):
        # multipleOf：xgrammar 预检拒 → 真实降 guidance（guidance 支持）。
        so = StructuredOutputsParams(json=MULTIPLEOF_SCHEMA)
        self._validate(so, backend="auto")
        assert so._backend == "guidance"
        assert so._backend_was_auto is True

    def test_auto_ladder_chooses_xgrammar_for_choice_via_rewrite(self):
        # choice 在 xgrammar 校验器里被原地改写成 EBNF grammar——改写成功则
        # 阶梯第一级就命中 xgrammar（choice 不触发降级）。
        so = StructuredOutputsParams(choice=["yes", "no"])
        self._validate(so, backend="auto")
        assert so._backend == "xgrammar"
        assert so.choice is None
        assert so.grammar == 'root ::= "yes" | "no"'

    def test_explicit_xgrammar_rejects_unsupported_schema(self):
        so = StructuredOutputsParams(json=MULTIPLEOF_SCHEMA)
        with pytest.raises(ValueError, match="not supported by xgrammar"):
            self._validate(so, backend="xgrammar")

    def test_post_init_rerun_after_validation(self):
        # L1084-L1086：校验尾重跑 __post_init__——choice 被改写成 grammar 后
        # 仍恰好一条约束（choice=None/grammar=EBNF），不会误报多约束。
        so = StructuredOutputsParams(choice=["a", "b"])
        self._validate(so, backend="auto")
        assert so.grammar == 'root ::= "a" | "b"'
