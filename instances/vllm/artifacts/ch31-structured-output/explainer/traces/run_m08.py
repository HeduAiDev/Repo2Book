"""m08 structured_output_key：六种约束归一成 (枚举, 字符串) 的编译键。

跑精简版 get_structured_output_key 真代码，六种入口各来一发；再验证
StructuredOutputRequest.structured_output_key 是 functools.cached_property
（每请求只算一次），以及它**不是**跨请求的编译缓存键。
"""
from _fakes import dump
import so_request
from sampling_params import StructuredOutputsParams
from so_request import StructuredOutputRequest, get_structured_output_key

cases = [
    ("json（dict 形态）", dict(json={"type": "object"})),
    ("json（str 形态）", dict(json='{"type": "object"}')),
    ("json_object=True", dict(json_object=True)),
    ("regex", dict(regex="[ab]+")),
    ("choice（list 形态）", dict(choice=["red", "blue"])),
    ("grammar（EBNF）", dict(grammar='root ::= "ab"')),
    ("structural_tag", dict(structural_tag='{"triggers": []}')),
]

rows = []
raw = {"cases": []}
for label, kw in cases:
    params = StructuredOutputsParams(**kw)
    option, spec = get_structured_output_key(params)
    raw_value = list(kw.values())[0]
    normalized = not isinstance(raw_value, str)
    rows.append([
        label,
        option.name,
        f"{option.value}",
        repr(spec),
        str(len(spec)),
        "是（json.dumps）" if normalized and not isinstance(raw_value, bool) else "否",
    ])
    raw["cases"].append({
        "label": label,
        "option": option.name,
        "option_value": option.value,
        "spec": spec,
        "spec_len": len(spec),
        "python_type_of_input": type(raw_value).__name__,
    })

# cached_property：同一请求内只算一次
calls = {"n": 0}
orig = so_request.get_structured_output_key


def counting(params):
    calls["n"] += 1
    return orig(params)


so_request.get_structured_output_key = counting
sor = StructuredOutputRequest(params=StructuredOutputsParams(json={"type": "object"}))
k1 = sor.structured_output_key
k2 = sor.structured_output_key
k3 = sor.structured_output_key
raw["cached_property_reads"] = 3
raw["cached_property_computations"] = calls["n"]
raw["cached_property_keys_equal"] = (k1 == k2 == k3)

# 两个不同请求、同一个 schema：键相等，但各自算一次——vLLM 侧不做跨请求去重
sor_a = StructuredOutputRequest(params=StructuredOutputsParams(json={"type": "object"}))
sor_b = StructuredOutputRequest(params=StructuredOutputsParams(json={"type": "object"}))
ka, kb = sor_a.structured_output_key, sor_b.structured_output_key
raw["two_requests_same_schema_keys_equal"] = (ka == kb)
raw["two_requests_key_computations"] = calls["n"] - 1
so_request.get_structured_output_key = orig

raw["note"] = (
    "键相等不代表 vLLM 会去重：_create_grammar 只是解包这个二元组去调 "
    "backend.compile_grammar，每请求各调一次（见 m09）。"
)

dump("m08.json", {
    "mechanism": "m08-structured-output-key",
    "columns": ["用户写法", "StructuredOutputOptions", "枚举值", "grammar_spec 字符串",
                "字符串长度", "是否被 json.dumps 归一"],
    "rows": rows,
    "raw": raw,
})
