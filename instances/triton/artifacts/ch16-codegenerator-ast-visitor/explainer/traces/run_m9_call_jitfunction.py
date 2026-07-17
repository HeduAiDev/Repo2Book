#!/usr/bin/env python3
"""ch16 m9 — 忠实重放 call_JitFunction 的 constexpr/tensor 拆分(code_generator.py:L1050-L1062)。
纯宿主 Python 逻辑,可直跑。mangle_fn 命名细节回指 ch10,此处只示意名字构成。"""
import json

# 被调 @triton.jit 函数:  def _helper(a, b, BLOCK: tl.constexpr)
arg_names = ["a", "b", "BLOCK"]
# 调用点:  _helper(x, y, 1024)   x,y 是 tensor,1024 是 Python int
call_args = [
    {"name": "a", "is_triton_value": True,  "repr": "x(tensor)"},
    {"name": "b", "is_triton_value": True,  "repr": "y(tensor)"},
    {"name": "BLOCK", "is_triton_value": False, "repr": "1024"},
]

# L1052-L1053: 非 triton value 的实参包成 constexpr
args = [dict(a, is_constexpr=(not a["is_triton_value"])) for a in call_args]
# L1056-L1057: constexprs / constants
constexprs = [i for i, a in enumerate(args) if a["is_constexpr"]]
constants = {i: call_args[i]["repr"] for i in constexprs}
# L1059-L1061: args 里 constexpr 位置抹 None,只有非 None 的走 handle
args_after = ["None" if i in constexprs else "arg.handle" for i in range(len(args))]
arg_vals = [f"{call_args[i]['repr']}.handle" for i in range(len(args)) if i not in constexprs]
arg_types = [f"{call_args[i]['repr']}.type" for i in range(len(args)) if i not in constexprs]

trace = []
for i, a in enumerate(args):
    trace.append({
        "i": i, "name": arg_names[i], "arg": call_args[i]["repr"],
        "is_triton_value": call_args[i]["is_triton_value"],
        "is_constexpr": a["is_constexpr"],
        "into_constants": (i in constexprs),
        "into_arg_vals_handle": (i not in constexprs),
        "args_after_L1059": args_after[i],
    })

out = {
    "callee": "_helper(a, b, BLOCK: tl.constexpr)  called as _helper(x, y, 1024)",
    "n_python_args": len(args),
    "constexprs_idx": constexprs,
    "constants": constants,
    "n_ssa_call_operands": len(arg_vals),   # arg_vals = [x.handle, y.handle]
    "arg_vals": arg_vals,
    "mangled_name": "mangle_fn('_helper', [x.type, y.type], {2: 1024})  # 回指 ch10",
    "trace": trace,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
