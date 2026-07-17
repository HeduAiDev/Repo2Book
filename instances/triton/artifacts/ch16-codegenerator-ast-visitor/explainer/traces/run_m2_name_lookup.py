#!/usr/bin/env python3
"""ch16 m2 — 忠实重放 name_lookup 三级查找 + global_lookup 的 constexpr 全局守卫
(code_generator.py:L270-L302 + L305-L311)。

真守卫会 isinstance(val, JITFunction)/language.dtype 等;此脚本用轻量标签值复刻守卫的
any([...]) 布尔清单**同一条判据顺序**,对一组名字跑出 local->global->builtin 的解析级别。
"""
import json
from types import ModuleType

# ---- 三层作用域(具体、可心算)---------------------------------------------
# lscope: 本函数已可见的名(kernel 参数 + 已赋值局部)
lscope = {"x": "tensor", "offs": "tensor", "BLOCK": "constexpr(1024)"}
# gscope: 模块全局。用标签表达守卫关心的类别。
_fake_module = ModuleType("triton.language")
gscope = {
    "tl":          {"kind": "module"},          # type(val) is ModuleType -> 放行
    "MAX_FUSED":   {"kind": "constexpr_global"}, # _is_constexpr_global -> 放行
    "LOOKUP_TABLE": {"kind": "plain_global"},    # 普通 int/list,非 constexpr -> 拒绝
}
builtin_namespace = {"range": "builtin", "len": "builtin", "min": "builtin"}

def guard_allows(name, v):
    """复刻 global_lookup 的 any([...]) 清单(L278-L292),按类别命中。"""
    kind = v.get("kind")
    checks = {
        "in_builtin_namespace": name in builtin_namespace,
        "is_ModuleType":        kind == "module",
        "is_JITFunction":       kind == "jitfunction",
        "is_triton_builtin":    kind == "triton_builtin",
        "is_triton_language":   kind == "triton_language_member",
        "is_dtype":             kind == "dtype",
        "is_constexpr_global":  kind == "constexpr_global",
    }
    return any(checks.values()), checks

def name_lookup(name):
    # L307-L311: for lookup in local, global, builtin.get
    if name in lscope:                       # 级别 1: local_lookup
        return {"name": name, "level": "①local", "result": lscope[name], "guard": None}
    if name in gscope:                       # 级别 2: global_lookup(带守卫)
        allowed, checks = guard_allows(name, gscope[name])
        if allowed:
            hit = next(k for k, v in checks.items() if v)
            return {"name": name, "level": "②global", "result": gscope[name]["kind"],
                    "guard": f"放行({hit})"}
        else:
            return {"name": name, "level": "②global", "result": "raise NameError",
                    "guard": "拒绝(非 constexpr 全局)"}
    if name in builtin_namespace:            # 级别 3: builtin_namespace.get
        return {"name": name, "level": "③builtin", "result": builtin_namespace[name], "guard": None}
    return {"name": name, "level": "—", "result": "raise NameError(not defined)", "guard": None}

names = ["x", "BLOCK", "tl", "MAX_FUSED", "range", "LOOKUP_TABLE"]
trace = [name_lookup(n) for n in names]
out = {
    "scope_sizes": {"lscope": len(lscope), "gscope": len(gscope),
                    "builtin_namespace": len(builtin_namespace)},
    "lookup_order": ["①local", "②global", "③builtin"],
    "n_rejected": sum(1 for t in trace if "raise" in str(t["result"])),
    "trace": trace,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
