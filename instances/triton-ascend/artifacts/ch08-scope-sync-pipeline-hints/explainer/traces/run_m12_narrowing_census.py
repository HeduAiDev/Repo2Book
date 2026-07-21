#!/usr/bin/env python3
"""M12 / M13 普查脚本：把 PIPE 与 TCoreType 在「.td 定义 → pybind 导出 → Python 可用」
三层里的成员**逐个数出来**（直接在 pin 源文件上做，不猜、不硬编）。

  * PIPE：HIVMAttrs.td 的 I32EnumAttrCase → ascend_ir.cc 的 py::enum_<hivm::PIPE>
          → core.py 的 class PIPE
  * TCoreType：HIVMAttrs.td 的 HIVM_TCoreTypeEnum 列表 → py::enum_<hivm::TCoreType>
          → core.py 的 class CORE → scope(core_mode=…) 语言层实际接受的取值
          （_handle_core_mode_attr 的白名单，从 extension/code_generator.py 解析）

用法：python3 run_m12_narrowing_census.py  → 打印并写出 m12_narrowing_census.json
"""
import ast
import json
import os
import re
from pathlib import Path

PIN = "2badfc89e70a9b7a5e88463a116c2feddce4b101"
_here = Path(__file__).resolve()
_cand = _here.parents[4] / "source" if len(_here.parents) > 4 else None
SRC = Path(os.environ.get(
    "R2B_SRC",
    str(_cand if (_cand and _cand.exists())
        else "/mnt/e/Laboratory/Repo2Book/instances/triton-ascend/source")))

TD = SRC / "third_party/ascend/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td"
CC = SRC / "third_party/ascend/ascend_ir.cc"
CORE = SRC / "third_party/ascend/language/cann/extension/core.py"
EXT_CG = SRC / "third_party/ascend/language/cann/extension/code_generator.py"


def td_enum_cases(prefix):
    """.td 里 def HIVM_<X> : I32EnumAttrCase<"NAME", k>;"""
    out = []
    for i, line in enumerate(TD.read_text(encoding="utf-8").splitlines(), 1):
        m = re.match(r'\s*def\s+HIVM_\S+\s*:\s*I32EnumAttrCase<"([^"]+)",\s*(\d+)>', line)
        if m and m.group(1).startswith(prefix):
            out.append({"name": m.group(1), "value": int(m.group(2)), "line": i})
    return out


def td_enum_list(enum_def):
    """.td 里 def <enum_def> : HIVM_I32Enum<...[ CASE, CASE, ... ]>"""
    text = TD.read_text(encoding="utf-8")
    i = text.index(f"def {enum_def}")
    body = text[i:text.index("]>", i)]
    start_line = text[:i].count("\n") + 1
    end_line = start_line + body.count("\n") + 1
    names = [n for n in re.findall(r"HIVM_(\w+)", body) if not n.endswith("Enum")]
    return {"cases": names, "count": len(names),
            "lines": f"L{start_line}-L{end_line}"}


def cc_pyenum(cpp_type):
    """ascend_ir.cc 里 py::enum_<hivm::X>(...) 之后连续的 .value("NAME", ...)"""
    lines = CC.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines)
                 if f"py::enum_<hivm::{cpp_type}>" in l)
    out = []
    for i in range(start + 1, len(lines)):
        m = re.match(r'\s*\.value\("([^"]+)"', lines[i])
        if not m:
            if ".export_values()" in lines[i]:
                break
            continue
        out.append({"name": m.group(1), "line": i + 1})
    return {"members": [o["name"] for o in out], "count": len(out),
            "lines": f"L{start + 1}-L{out[-1]['line']}"}


def py_enum(cls_name):
    text = CORE.read_text(encoding="utf-8")
    node = next(n for n in ast.parse(text).body
                if isinstance(n, ast.ClassDef) and n.name == cls_name)
    names = [t.targets[0].id for t in node.body if isinstance(t, ast.Assign)]
    return {"members": names, "count": len(names),
            "lines": f"L{node.lineno}-L{node.end_lineno}"}


def core_mode_whitelist():
    """extension/code_generator.py::_handle_core_mode_attr 里 core_mode 的白名单元组。"""
    text = EXT_CG.read_text(encoding="utf-8")
    node = next(n for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name == "_handle_core_mode_attr")
    cmp_ = next(n for n in ast.walk(node) if isinstance(n, ast.Compare))
    values = [e.value for e in cmp_.comparators[0].elts]
    return {"accepted": values, "count": len(values),
            "lines": f"L{node.lineno}-L{node.end_lineno}"}


def main():
    pipe_td_cases = td_enum_cases("PIPE_") + td_enum_cases("VIRTUAL_PIPE_")
    pipe_enum = td_enum_list("HIVM_PipeEnum")
    pipe_cc = cc_pyenum("PIPE")
    pipe_py = py_enum("PIPE")
    core_td = td_enum_list("HIVM_TCoreTypeEnum")
    core_cc = cc_pyenum("TCoreType")
    core_py = py_enum("CORE")
    wl = core_mode_whitelist()

    out = {
        "pin": PIN,
        "mechanism": "M12 / M13",
        "harness": "直接在 pin 源文件上正则/AST 普查枚举成员，无运行时依赖",
        "PIPE": {
            "td_definition": {"file": "third_party/ascend/AscendNPU-IR/bishengir/include/"
                                      "bishengir/Dialect/HIVM/IR/HIVMAttrs.td",
                              "cases": pipe_td_cases, "count": len(pipe_td_cases),
                              "enum_list": pipe_enum},
            "pybind_export": {"file": "third_party/ascend/ascend_ir.cc", **pipe_cc},
            "python_layer": {"file": "third_party/ascend/language/cann/extension/core.py",
                             **pipe_py},
            "dropped_at_pybind": sorted(set(c["name"] for c in pipe_td_cases)
                                        - set(pipe_cc["members"])),
            "narrowing": [len(pipe_td_cases), pipe_cc["count"], pipe_py["count"]],
        },
        "TCoreType": {
            "td_enum_list": core_td,
            "pybind_export": {"file": "third_party/ascend/ascend_ir.cc", **core_cc},
            "python_layer": {"file": "third_party/ascend/language/cann/extension/core.py",
                             **core_py},
            "reachable_from_scope": wl,
            "unreachable_from_scope": sorted(set(core_py["members"]) -
                                             {v.upper() for v in wl["accepted"]}),
            "narrowing": [core_cc["count"], core_py["count"], wl["count"]],
        },
    }
    Path(__file__).with_name("m12_narrowing_census.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
