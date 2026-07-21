#!/usr/bin/env python3
"""M14 / M15 驱动脚本：跑 pin 的 compile_hint / compile_hint_impl / multibuffer，
看「提示值的类型分派 → annotation.mark」以及 SIMT 门控实际管到谁。

取证方式（宿主无昇腾 NPU / 未编译 ascend_ir pybind）：
  * compile_hint / compile_hint_impl / multibuffer 的函数体 **逐字取自 pin**
    （third_party/ascend/language/cann/extension/aux_ops.py，用 ast 按行号切出后 exec；
    装饰器行 @builtin 不在切片内——它只检查 _builder 是否传了）；
  * builder 换成记录型替身：属性以文本形式呈现，`create_annotation_mark` 记录
    (ptr, key, value)；`is_simt_mode()` 可开关，用来观察 M15 的门控范围。

用法：python3 run_m14_compile_hint.py  → 打印并写出 m14_compile_hint.json
"""
import ast
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
AUX = SRC / "third_party/ascend/language/cann/extension/aux_ops.py"

MARKS = []


class constexpr:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"constexpr({self.value!r})"


def _constexpr_to_value(v):
    return v.value if isinstance(v, constexpr) else v


def _unwrap_if_constexpr(v):
    return v.value if isinstance(v, constexpr) else v


class FakeTensor:
    def __init__(self, handle):
        self.handle = handle


class MockBuilder:
    def __init__(self, simt=False):
        self.simt = simt

    def is_simt_mode(self):
        return self.simt

    def get_unit_attr(self):
        return "#unit"

    def get_bool_attr(self, v):
        return f"#bool<{str(v).lower()}>"

    def get_int32_attr(self, v):
        return f"{v} : i32"

    def get_str_attr(self, v):
        return f'"{v}"'

    def get_i64_array_attr(self, v):
        return f"[{', '.join(str(x) for x in v)}] : i64"

    def create_annotation_mark(self, handle, key, val):
        MARKS.append({"op": "annotation.mark", "ptr": handle, "key": key, "attr": val})


def grab(path, names):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out, spans = {}, {}
    for n in ast.parse(text).body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            out[n.name] = "\n".join(lines[n.lineno - 1:n.end_lineno])
            spans[n.name] = f"L{n.lineno}-L{n.end_lineno}"
    assert set(names) <= set(out), set(names) - set(out)
    return out, spans


def load():
    src, spans = grab(AUX, {"compile_hint_impl", "compile_hint", "multibuffer"})
    g = {"tensor": FakeTensor, "ir": type("ir", (), {"builder": object}),
         "core": type("m", (), {"constexpr": constexpr}),
         "_constexpr_to_value": _constexpr_to_value,
         "_unwrap_if_constexpr": _unwrap_if_constexpr}
    for n in ("compile_hint_impl", "compile_hint", "multibuffer"):
        exec(compile(src[n], "aux_ops.py", "exec"), g)
    return g, spans


def run_case(label, fn, kwargs, simt=False):
    before = len(MARKS)
    rec = {"case": label, "simt_mode": simt,
           "args": {k: repr(v) if isinstance(v, (constexpr, FakeTensor)) else v
                    for k, v in kwargs.items() if k != "_builder"}}
    try:
        fn(**kwargs)
        rec["outcome"] = "OK"
        rec["error"] = None
    except Exception as e:  # noqa: BLE001
        rec["outcome"] = type(e).__name__
        rec["error"] = str(e)
    rec["emitted"] = MARKS[before:]
    rec["n_emitted"] = len(rec["emitted"])
    return rec


def main():
    g, spans = load()
    b = MockBuilder(simt=False)
    bs = MockBuilder(simt=True)
    t = FakeTensor("%buf")
    ch, mb = g["compile_hint"], g["multibuffer"]
    cases = [
        run_case("hint_val 缺省（None）→ unit attr", ch,
                 dict(ptr=t, hint_name="hivm.dont_fuse", _builder=b)),
        run_case("hint_val=True → bool attr", ch,
                 dict(ptr=t, hint_name="hivm.some_flag", hint_val=True, _builder=b)),
        run_case("hint_val=False → bool attr（isinstance(bool) 先判）", ch,
                 dict(ptr=t, hint_name="hivm.some_flag", hint_val=False, _builder=b)),
        run_case("hint_val=0 → 落进 `not hint_val` 分支，变 unit attr", ch,
                 dict(ptr=t, hint_name="hivm.count", hint_val=0, _builder=b)),
        run_case("hint_val=4 → i32 attr", ch,
                 dict(ptr=t, hint_name="hivm.count", hint_val=4, _builder=b)),
        run_case("hint_val=[1, 2] → i64 array attr", ch,
                 dict(ptr=t, hint_name="hivm.tiling", hint_val=[1, 2], _builder=b)),
        run_case("hint_val=1.5（float）→ 报错", ch,
                 dict(ptr=t, hint_name="hivm.ratio", hint_val=1.5, _builder=b)),
        run_case("SIMT 模式下 compile_hint 直接 return（不发 mark）", ch,
                 dict(ptr=t, hint_name="hivm.count", hint_val=4, _builder=bs), simt=True),
        run_case("multibuffer(size=2) → hivm.multi_buffer", mb,
                 dict(src=t, size=2, _builder=b)),
        run_case("SIMT 模式下 multibuffer 仍然发 mark（绕开门控）", mb,
                 dict(src=t, size=2, _builder=bs), simt=True),
        run_case("multibuffer(size=3) → 断言只支持 2", mb,
                 dict(src=t, size=3, _builder=b)),
    ]
    out = {
        "pin": PIN,
        "mechanism": "M14 / M15",
        "harness": "pin 的 compile_hint / compile_hint_impl / multibuffer 逐字 exec；builder 为记录型替身（is_simt_mode 可开关）",
        "sources": {f"aux_ops.py::{k}": v for k, v in spans.items()},
        "cases": cases,
        "summary": {
            "n_cases": len(cases),
            "n_marks_emitted": len(MARKS),
            "n_simt_cases": sum(1 for c in cases if c["simt_mode"]),
            "n_simt_cases_still_emitting": sum(1 for c in cases
                                               if c["simt_mode"] and c["n_emitted"] > 0),
            "n_errors": sum(1 for c in cases if c["outcome"] != "OK"),
        },
    }
    Path(__file__).with_name("m14_compile_hint.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in cases:
        print(f"{c['case']:52s} -> {c['outcome']:15s} {c['emitted']} {(c['error'] or '')[:50]}")
    print("SUMMARY", out["summary"])


if __name__ == "__main__":
    main()
