#!/usr/bin/env python3
"""ch16 m4 — 忠实重放 _apply_binary_method 反射分派(code_generator.py:L536-L543)+
_method_name_for_bin_op 映射(L554-L567)。纯宿主 Python 逻辑,可直跑。"""
import ast
import json
import re

# L554-L567 的映射表(逐字):
_method_name_for_bin_op = {
    ast.Add: "__add__", ast.Sub: "__sub__", ast.Mult: "__mul__",
    ast.Div: "__truediv__", ast.FloorDiv: "__floordiv__", ast.Mod: "__mod__",
    ast.Pow: "__pow__", ast.LShift: "__lshift__", ast.RShift: "__rshift__",
    ast.BitAnd: "__and__", ast.BitOr: "__or__", ast.BitXor: "__xor__",
}

def apply_binary_method(method_name, lhs_is_tensor, rhs_is_tensor):
    # L536-L543 逐字复刻
    if lhs_is_tensor:                                  # _is_triton_tensor(lhs)
        return f"lhs.{method_name}(rhs, _builder=…)", True
    if rhs_is_tensor:                                  # _is_triton_tensor(rhs)
        rev = re.sub(r"__(.*)__", r"__r\1__", method_name)   # L541
        return f"rhs.{rev}(lhs, _builder=…)", True
    return f"lhs.{method_name}(rhs)", False            # 两边都非 tensor:纯 Python

cases = [
    {"expr": "x + y", "op": ast.Add, "lhs_t": True,  "rhs_t": True},
    {"expr": "x + 1", "op": ast.Add, "lhs_t": True,  "rhs_t": False},
    {"expr": "2 + x", "op": ast.Add, "lhs_t": False, "rhs_t": True},
    {"expr": "a + b", "op": ast.Add, "lhs_t": False, "rhs_t": False},  # 都是 Python 常量
]
trace = []
for c in cases:
    mname = _method_name_for_bin_op[c["op"]]
    call, builds = apply_binary_method(mname, c["lhs_t"], c["rhs_t"])
    trace.append({"expr": c["expr"], "ast_op": c["op"].__name__, "method_name": mname,
                  "lhs_tensor": c["lhs_t"], "rhs_tensor": c["rhs_t"],
                  "actual_call": call, "builds_op": builds})

out = {"n_cases": len(cases), "n_builds_op": sum(1 for t in trace if t["builds_op"]),
       "trace": trace}
print(json.dumps(out, ensure_ascii=False, indent=2))
