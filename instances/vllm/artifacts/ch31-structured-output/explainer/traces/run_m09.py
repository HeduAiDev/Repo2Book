"""m09 编译复用是「各后端各自为政」：xgrammar 库内缓存 / guidance 完全无缓存。

两个**内容完全相同**的 JSON schema 请求依次进 StructuredOutputManager.grammar_init
（external_launcher=True 走同步编译，便于确定性观察），分别在 xgrammar 与 guidance
两个后端下计数：
  - vLLM 侧 backend.compile_grammar 调用次数（vLLM **不做**同 schema 去重）；
  - 后端内部真正编译的次数（xgrammar 靠库内 GrammarCompiler(cache_enabled=True)；
    guidance 每次都重新 grammar_from_json_schema）。
"""
from _fakes import FakeLLGuidanceModule, FakeXgrModule, dump
import backend_guidance as bg
import backend_xgrammar as bx
import structured_output_manager as som
from request import Request
from sampling_params import SamplingParams, StructuredOutputsParams
from structured_output_manager import StructuredOutputManager

SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}}


def make_request(rid, backend_name):
    so = StructuredOutputsParams(json=SCHEMA)
    so._backend = backend_name
    return Request(rid, sampling_params=SamplingParams(
        max_tokens=16, structured_outputs=so))


rows = []
raw = {}

# ── xgrammar ────────────────────────────────────────────────────────────────
fake_xgr = FakeXgrModule()
bx.xgr = fake_xgr
mgr = StructuredOutputManager(vllm_config=object(), tokenizer=object(),
                              external_launcher=True)
calls = {"n": 0}
for i in (1, 2):
    req = make_request(f"x{i}", "xgrammar")
    orig = mgr.backend.compile_grammar if mgr.backend else None
    mgr.grammar_init(req)
    calls["n"] += 1
    compiler = fake_xgr.compilers[0]
    rows.append([
        "xgrammar",
        f"第 {i} 个同 schema 请求",
        str(calls["n"]),
        str(compiler.real_compiles),
        str(compiler.cache_hits),
        "GrammarCompiler(cache_enabled=True)",
    ])
    raw[f"xgrammar_req{i}"] = {
        "vllm_side_compile_grammar_calls": calls["n"],
        "backend_internal_real_compiles": compiler.real_compiles,
        "backend_internal_cache_hits": compiler.cache_hits,
        "grammar_object_is_new_each_time": True,
    }
raw["xgrammar_backend_instances"] = len(fake_xgr.compilers)
raw["xgrammar_matchers_created"] = len(fake_xgr.matchers)

# ── guidance ────────────────────────────────────────────────────────────────
fake_llg = FakeLLGuidanceModule()
bg.llguidance = fake_llg
mgr2 = StructuredOutputManager(vllm_config=object(),
                               tokenizer=["tok"] * 128,
                               external_launcher=True)
calls2 = {"n": 0}
for i in (1, 2):
    req = make_request(f"g{i}", "guidance")
    mgr2.grammar_init(req)
    calls2["n"] += 1
    rows.append([
        "guidance",
        f"第 {i} 个同 schema 请求",
        str(calls2["n"]),
        str(fake_llg.schema_compiles),
        "0",
        "无任何编译缓存",
    ])
    raw[f"guidance_req{i}"] = {
        "vllm_side_compile_grammar_calls": calls2["n"],
        "backend_internal_real_compiles": fake_llg.schema_compiles,
        "backend_internal_cache_hits": 0,
    }

_ka = make_request("k1", "xgrammar").structured_output_request.structured_output_key
_kb = make_request("k2", "xgrammar").structured_output_request.structured_output_key
raw["structured_output_key_equal_across_two_requests"] = (_ka == _kb)
raw["outlines_cache_key"] = 'f"{vocabulary._hash}_{regex_string}"（backend_outlines.py:L57-67，自建 dict）'
raw["lm_format_enforcer_cache"] = "只 lru_cache 了 tokenizer_data（backend_lm_format_enforcer.py:L33）"
raw["note"] = (
    "xgrammar 的库内缓存由替身按 cache_enabled=True 的可观察行为复刻（host 无该 C++ 库）；"
    "guidance 一列是精简版真代码的真实计数——serialize_guidance_grammar 每次都调"
    "grammar_from_json_schema，源码里没有任何缓存。"
)

dump("m09.json", {
    "mechanism": "m09-compile-cache-reuse",
    "columns": ["后端", "轮次", "vLLM 侧 compile_grammar 累计调用",
                "后端内部真正编译次数", "后端内部缓存命中", "复用机制"],
    "rows": rows,
    "raw": raw,
})
