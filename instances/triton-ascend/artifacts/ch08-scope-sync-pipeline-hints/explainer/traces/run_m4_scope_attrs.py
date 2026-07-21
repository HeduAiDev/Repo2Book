#!/usr/bin/env python3
"""M4 驱动脚本：跑 pin 的 `_extract_scope_attributes` + `_build_mlir_attrs_from_scope_attrs`，
看 `with scope(...)` 的关键字实参如何变成 scope.scope 上的 MLIR 属性。

取证方式：直接 import pin 的 third_party/ascend/language/cann/extension/code_generator.py
（模块顶层只 `import ast`，无需昇腾运行时），把真实 `with` 语句解析成 AST 后喂给它；
IR builder 用记录型替身（返回属性的**文本形式**，形如 `#hivm.tcore_type<VECTOR>`）。
分派逻辑、白名单、丢弃规则都是 pin 的真实控制流。

用法：python3 run_m4_scope_attrs.py  → 打印并写出 m4_scope_attrs.json
"""
import ast
import importlib.util
import json
import os
from pathlib import Path

PIN = "2badfc89e70a9b7a5e88463a116c2feddce4b101"
_here = Path(__file__).resolve()
_cand = _here.parents[4] / "source" if len(_here.parents) > 4 else None
SRC = Path(os.environ.get(
    "R2B_SRC",
    str(_cand if (_cand and _cand.exists())
        else "/mnt/e/Laboratory/Repo2Book/instances/triton-ascend/source")))
EXT_CG = SRC / "third_party/ascend/language/cann/extension/code_generator.py"


class MockBuilder:
    def get_unit_attr(self):
        return "#unit"

    def get_str_attr(self, v):
        return f'"{v}"'

    def get_bool_attr(self, v):
        return f"#bool<{str(v).lower()}>"

    def get_int32_attr(self, v):
        return f"{v} : i32"

    def get_i64_array_attr(self, v):
        return f"[{', '.join(str(x) for x in v)}] : i64"

    def get_t_core_type_attr_name(self):
        return "tcore_type"

    def get_t_core_type_cube_attr(self):
        return "#hivm.tcore_type<CUBE>"

    def get_t_core_type_vector_attr(self):
        return "#hivm.tcore_type<VECTOR>"


def load():
    spec = importlib.util.spec_from_file_location("ascend_ext_cg", EXT_CG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load()
    b = MockBuilder()
    cases = []
    samples = [
        '"vector"',                              # 位置参数写法：keywords 为空
        'core_mode="cube"',
        'core_mode="vector"',
        'core_mode="vector", noinline=False',
        'core_mode="vector", disable_auto_sync=True',
        'core_mode="vector", disable_auto_sync=False',
        'core_mode="aicore"',
        'core_mode=mode_var',                    # 非常量：被 _extract 丢掉
        'feature_a=True',                        # docstring 里的写法
        'core_mode="cube", my_hint=3, my_tag="x", my_list=[7, 9]',
    ]
    for s in samples:
        node = ast.parse(f"with scope({s}):\n    pass\n").body[0]
        ctx = node.items[0].context_expr
        extracted = mod._extract_scope_attributes(ctx)
        attrs = mod._build_mlir_attrs_from_scope_attrs(b, extracted)
        cases.append({
            "with_source": f"with scope({s}):",
            "keywords_in_ast": [k.arg for k in ctx.keywords],
            "extracted_constants": extracted,
            "n_keywords": len(ctx.keywords),
            "n_extracted": len(extracted),
            "mlir_attrs": attrs,
            "n_mlir_attrs": len(attrs),
            "has_tcore_type": "tcore_type" in attrs,
            "has_noinline": "noinline" in attrs,
        })

    out = {
        "pin": PIN,
        "mechanism": "M4 (+M18 旁证)",
        "harness": "pin 的 _extract_scope_attributes / _build_mlir_attrs_from_scope_attrs 原样 import 执行；builder 为记录型替身（属性以文本形式呈现）",
        "sources": {
            "_extract_scope_attributes": "third_party/ascend/language/cann/extension/code_generator.py:L62-L69",
            "_py_value_to_mlir_attr": "third_party/ascend/language/cann/extension/code_generator.py:L72-L82",
            "_handle_core_mode_attr": "third_party/ascend/language/cann/extension/code_generator.py:L84-L93",
            "_build_mlir_attrs_from_scope_attrs": "third_party/ascend/language/cann/extension/code_generator.py:L96-L118",
        },
        "cases": cases,
        "summary": {
            "n_cases": len(cases),
            "n_with_tcore_type": sum(1 for c in cases if c["has_tcore_type"]),
            "n_without_noinline": sum(1 for c in cases if not c["has_noinline"]),
            "default_attr_count": 1,
        },
    }
    Path(__file__).with_name("m4_scope_attrs.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in cases:
        print(f"{c['with_source']:62s} extracted={c['extracted_constants']} "
              f"-> {c['mlir_attrs']}")
    print("SUMMARY", out["summary"])


if __name__ == "__main__":
    main()
