#!/usr/bin/env python3
"""ch16 m5(f4 回收)— 忠实重放 visit_Call 的四条出口判据顺序(code_generator.py:L1097-L1126)。

真 fn 对象需 triton 运行期(JITFunction/is_builtin 等);此脚本把每个调用点的 fn 身份
事实(是否在 statically_implemented_functions / 是否 JITFunction / 是否 @builtin 或
tensor 方法)显式给出,逐字复刻 L1099-L1126 的判据**顺序**,输出命中分支。
statically_implemented_functions = {static_assert, static_print, int, len}(L1252-L1257)。
"""
import json

# 每个调用点声明它的 fn 身份事实(从源码语义确定,非编造):
call_sites = [
    {"expr": "tl.static_assert(BLOCK % 16 == 0)", "fn": "static_assert",
     "in_static_impl": True,  "is_jitfunction": False, "is_builtin_or_method": False},
    {"expr": "n2 = len(shape)",                    "fn": "len",
     "in_static_impl": True,  "is_jitfunction": False, "is_builtin_or_method": False},
    {"expr": "acc = _helper(x, BLOCK)",            "fn": "_helper(@triton.jit)",
     "in_static_impl": False, "is_jitfunction": True,  "is_builtin_or_method": False},
    {"expr": "v = tl.load(x_ptr + offs)",          "fn": "tl.load(@builtin)",
     "in_static_impl": False, "is_jitfunction": False, "is_builtin_or_method": True},
    {"expr": "y = x.to(tl.float32)",               "fn": "tensor.to(有 __self__=tensor)",
     "in_static_impl": False, "is_jitfunction": False, "is_builtin_or_method": True},
    {"expr": "for i in range(0, N):",              "fn": "range(纯 Python 内置)",
     "in_static_impl": False, "is_jitfunction": False, "is_builtin_or_method": False},
]

STATIC_IMPL = {"static_assert", "static_print", "int", "len"}  # L1252-1257

trace = []
for cs in call_sites:
    # L1099-L1101: static_implementation = statically_implemented_functions.get(fn)
    if cs["in_static_impl"]:
        assert cs["fn"] in STATIC_IMPL or cs["fn"] in ("static_assert", "len")
        branch, action, builds_ir = "①static", "编译期求值,直接返回 constexpr", False
    # L1105-L1107: isinstance(fn, JITFunction)
    elif cs["is_jitfunction"]:
        branch, action, builds_ir = "②JITFunction", "call_JitFunction 内联,建 tt.call", True
    # L1108: hasattr(fn,'__self__') and is_triton_value(fn.__self__) or is_builtin(fn)
    elif cs["is_builtin_or_method"]:
        branch, action, builds_ir = "③builtin", "注入 _builder,在 IR 建 tt.* op", True
    # L1124-L1126: 剩下的纯 Python callable
    else:
        branch, action, builds_ir = "④纯Python", "unwrap constexpr 后宿主直调,不建 op", False
    trace.append({"expr": cs["expr"], "fn": cs["fn"], "branch": branch,
                  "action": action, "builds_ir_op": builds_ir})

out = {
    "statically_implemented_functions": sorted(STATIC_IMPL),
    "dispatch_order": ["①static", "②JITFunction", "③builtin", "④纯Python"],
    "n_call_sites": len(call_sites),
    "n_build_ir": sum(1 for t in trace if t["builds_ir_op"]),
    "trace": trace,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
