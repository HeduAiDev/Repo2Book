# ch30 m2/m4/m3 驱动脚本：六选一互斥校验、六形态→structured_output_key 归一
# （json.dumps / choice list 归一）、choice→EBNF 与 lark→EBNF 校验期原地改写、
# auto 阶梯的 _backend 落点（choice→xgrammar；multipleOf→guidance 降级）。
# 环境：host Miniconda CPython 3.11.11 + xgrammar 0.2.6 + llguidance 1.7.6 + gpt2。
# 已知行为差异（impl-notes 口径，explainer 须标注）：精简版删了 outlines 后端，
# 非 tekken Mistral 或 guidance 也不支持的 schema 在真实源码降 outlines、在精简版
# 经 validate_guidance_grammar 报错拒单——本 trace 的 multipleOf 例子是 guidance
# 支持的特性，真实源码同样降 guidance，两条路径一致。
import json
import pathlib
import sys

IMPL = pathlib.Path(__file__).resolve().parents[2] / "implementation"
sys.path.insert(0, str(IMPL))

from transformers import AutoTokenizer

from vllm.config import (ModelConfig, ParallelConfig, SchedulerConfig,
                         StructuredOutputsConfig, VllmConfig)
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import (
    validate_xgrammar_grammar,
)
from vllm.v1.structured_output.request import get_structured_output_key
from vllm.v1.structured_output.utils import (choice_as_grammar,
                                             grammar_is_likely_lark)

TOKENIZER_NAME = "gpt2"
VOCAB = 50257
tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def make_vllm_config(backend="auto"):
    return VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER_NAME, vocab_size=VOCAB),
        parallel_config=ParallelConfig(distributed_executor_backend="mp"),
        scheduler_config=SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=8192),
        structured_outputs_config=StructuredOutputsConfig(backend=backend),
        speculative_config=None,
    )


def run_frontend(so, backend="auto"):
    """真实前端校验路径：InputProcessor 调 params.verify → _validate_structured_outputs。"""
    sp = SamplingParams(structured_outputs=so)
    cfg = make_vllm_config(backend)
    sp._validate_structured_outputs(
        cfg.model_config, cfg.structured_outputs_config, tok)
    return sp


out = {"env": {"tokenizer": TOKENIZER_NAME, "vocab_size": VOCAB}}

# ── m2：六选一互斥（__post_init__ 双向报错，sampling_params.py:L90-L111）──
errs = []
for kw in ({"json": "{}", "regex": "a"}, {}):
    try:
        StructuredOutputsParams(**kw)
    except VLLMValidationError as e:
        errs.append({"input": str(kw), "error_head": str(e).splitlines()[0][:100]})
out["mutual_exclusion"] = errs

# ── m2：六形态→键归一（request.py:L82-L103 get_structured_output_key）──
cases = [
    ("json(dict)", StructuredOutputsParams(json={"a": 1, "b": [2, 3]})),
    ("json(str,同 schema 异空格)", StructuredOutputsParams(json='{"b": [2, 3], "a": 1}')),
    ("json(str,dict 的 dumps 形)", StructuredOutputsParams(json='{"a": 1, "b": [2, 3]}')),
    ("json_object", StructuredOutputsParams(json_object=True)),
    ("regex", StructuredOutputsParams(regex="[0-9]+")),
    ("choice", StructuredOutputsParams(choice=["yes", "no"])),
    ("choice(字符串注入)", StructuredOutputsParams(choice='["yes", "no"]')),
    ("grammar", StructuredOutputsParams(grammar='root ::= "yes"')),
    ("structural_tag", StructuredOutputsParams(structural_tag='{"x": 1}')),
]
norm = []
for name, so in cases:
    opt, spec = get_structured_output_key(so)
    norm.append({"case": name, "enum": opt.name, "spec": spec})
out["key_normalization"] = norm

# 同 schema 两请求：键相等但 structured_output_key 是每请求 cached_property
sp1 = SamplingParams(structured_outputs=StructuredOutputsParams(regex="[0-9]+"))
sp2 = SamplingParams(structured_outputs=StructuredOutputsParams(regex="[0-9]+"))
from vllm.v1.request import Request
r1 = Request(request_id="r1", prompt_token_ids=[1], sampling_params=sp1, pooling_params=None)
r2 = Request(request_id="r2", prompt_token_ids=[1], sampling_params=sp2, pooling_params=None)
k1a, k1b, k2 = r1.structured_output_request.structured_output_key, \
    r1.structured_output_request.structured_output_key, \
    r2.structured_output_request.structured_output_key
out["per_request_key"] = {
    "r1_key": [k1a[0].name, k1a[1]],
    "r1_key_equals_itself_cached": k1a is k1b,
    "r1_key_equals_r2_key": k1a == k2,
    "cached_property_same_object_across_reads": k1a is k1b,
    "distinct_objects_across_requests": k1a is not k2,
}

# ── m4：choice_as_grammar（utils.py:L553-L561，EBNF 生成+转义）──
cg_plain = choice_as_grammar(["yes", "no", "maybe"])
cg_escape = choice_as_grammar(['a"b', "c\\d"])
out["choice_as_grammar"] = {
    "plain": cg_plain,
    "escaped": cg_escape,
}

# ── m4：validate_xgrammar_grammar 原地改写（backend_xgrammar.py:L293-L303）──
so = StructuredOutputsParams(choice=["yes", "no"])
sp = run_frontend(so)  # auto：先试 xgrammar（内部做改写）
out["rewrite_in_place"] = {
    "before": {"choice": ["yes", "no"], "grammar": None},
    "after": {"choice": sp.structured_outputs.choice,
              "grammar": sp.structured_outputs.grammar},
    "grammar_value": sp.structured_outputs.grammar,
    "backend_chosen": sp.structured_outputs._backend,
    "backend_was_auto": sp.structured_outputs._backend_was_auto,
}

# ── m4：lark 判定（utils.py:L391-L420：无 ::= 即 Lark）──
out["lark_detection"] = {
    "ebnf_has_assign": grammar_is_likely_lark('root ::= "yes"'),
    "lark_no_assign": grammar_is_likely_lark('start: "yes"'),
}

# ── m3：auto 阶梯落点 ──
ladder = []
# 正常 choice → xgrammar 直接过
so_a = StructuredOutputsParams(choice=["yes", "no"])
sp_a = run_frontend(so_a)
ladder.append({"case": "choice [yes,no]", "backend": sp_a.structured_outputs._backend,
               "was_auto": sp_a.structured_outputs._backend_was_auto})
# multipleOf：xgrammar 不支持（has_xgrammar_unsupported_json_features）→ 降 guidance
so_b = StructuredOutputsParams(json={"type": "object",
                                     "properties": {"n": {"type": "number", "multipleOf": 5}}})
sp_b = run_frontend(so_b)
ladder.append({"case": "json multipleOf=5（xgrammar 预检不过）",
               "backend": sp_b.structured_outputs._backend,
               "was_auto": sp_b.structured_outputs._backend_was_auto})
# 简单 json schema → xgrammar 直接过
so_c = StructuredOutputsParams(json={"type": "object",
                                     "properties": {"a": {"type": "integer"}}})
sp_c = run_frontend(so_c)
ladder.append({"case": "json {a:integer}", "backend": sp_c.structured_outputs._backend,
               "was_auto": sp_c.structured_outputs._backend_was_auto})
out["auto_ladder"] = ladder

# 显式后端冲突（m20 素材，sampling_params.py:L949-L966）：params 复用且 auto 记账放行
sp_d = run_frontend(StructuredOutputsParams(regex="[0-9]+"))  # auto → xgrammar
try:
    sp_d._validate_structured_outputs(  # 复用同一 params 再发（引擎 auto）
        make_vllm_config().model_config,
        make_vllm_config().structured_outputs_config, tok)
    reuse_ok = True
    reused_backend = sp_d.structured_outputs._backend
except VLLMValidationError as e:
    reuse_ok = False
    reused_backend = str(e).splitlines()[0][:90]
so_e = StructuredOutputsParams(regex="[0-9]+")
so_e._backend = "guidance"  # 手工指定不同后端 = 冲突
sp_e = SamplingParams(structured_outputs=so_e)
conflict = None
try:
    sp_e._validate_structured_outputs(
        make_vllm_config().model_config,
        make_vllm_config().structured_outputs_config, tok)
except VLLMValidationError as e:
    conflict = str(e).splitlines()[0][:110]
out["backend_conflict"] = {
    "auto_reuse_passes": reuse_ok,
    "auto_reuse_backend": reused_backend,
    "explicit_conflict_error_head": conflict,
}

# ── 表格行建议 ──
out["table_rows_m2"] = [
    ["json dict", '{"a": 1, "b": [2, 3]}', "JSON",
     'json.dumps 归一：{"a": 1, "b": [2, 3]}", 键序保留'],
    ["json str（dict 的 dumps 形）", '{"a": 1, "b": [2, 3]}', "JSON", "与 dict 形完全同串——同键"],
    ["json str（异键序）", '{"b": [2, 3], "a": 1}', "JSON", "串不同→键不同（不做 canonical 化）"],
    ["json_object", "（无串）", "JSON_OBJECT", 'spec=空串 ""'],
    ["regex", "[0-9]+", "REGEX", "原样"],
    ["choice", '["yes", "no"]', "CHOICE", "list 也 json.dumps 归一"],
    ["grammar", 'root ::= "yes"', "GRAMMAR", "原样"],
    ["structural_tag", '{"x": 1}', "STRUCTURAL_TAG", "原样（JSON 串）"],
]
out["table_rows_m4"] = [
    ["改写前", "choice=['yes', 'no']", "grammar=None", "validate_xgrammar_grammar 入口"],
    ["试编", 'xgr.Grammar.from_ebnf(root ::= "yes" | "no")', "成功", "backend_xgrammar.py:L296-L301"],
    ["改写后", "choice=None", 'grammar=root ::= "yes" | "no"', "L302-L303 原地改写"],
    ["引擎侧", "compile_grammar 五分派", "GRAMMAR 分支（无 CHOICE 分支）", "改写使引擎侧无需 CHOICE 分支"],
]

p = pathlib.Path(__file__).with_name("trace_m02_m04_params_key.json")
with open(p, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("WROTE", p)
print(json.dumps(out["mutual_exclusion"], ensure_ascii=False, indent=1))
print(json.dumps(out["key_normalization"], ensure_ascii=False, indent=1))
print(json.dumps(out["rewrite_in_place"], ensure_ascii=False, indent=1))
print(json.dumps(out["choice_as_grammar"], ensure_ascii=False, indent=1))
print(json.dumps(out["lark_detection"], ensure_ascii=False))
print(json.dumps(out["auto_ladder"], ensure_ascii=False, indent=1))
print(json.dumps(out["backend_conflict"], ensure_ascii=False, indent=1))
print(json.dumps(out["per_request_key"], ensure_ascii=False, indent=1))
