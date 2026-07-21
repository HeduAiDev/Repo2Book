"""m5 —— libdevice 三条实现路径的分层：证据 + 数值。

两部分：
A) 对 **pin 源码本身**做静态盘点（third_party/ascend/language/cann/libdevice.py 与
   extension/math_ops.py）：文件行数、@core.extern 个数、__hmf_ 出现次数与去重符号数、
   纯 IR 逼近路径的函数（同一 def 体内出现 semantic.* 组合而非只有 extern_elementwise）。
   —— 图上的"三条路各占多少"这类数字都从这里来。
B) 跑精简版的 acos 纯 IR 多项式逼近，与 CPU 的 math.acos 逐点比对
   （FakeBuilder 的 create_fadd/fmul/... 是真浮点算术，这段数学不依赖昇腾硬件）。
"""
import math as pymath
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import build_env, dump, CH_DIR  # noqa: E402

# 源码根：instances/<inst>/source（引用时写规范路径，不带该前缀）
SRC_ROOT = CH_DIR.parents[1] / "source"
LIBDEVICE = SRC_ROOT / "third_party/ascend/language/cann/libdevice.py"
MATH_OPS = SRC_ROOT / "third_party/ascend/language/cann/extension/math_ops.py"
CUSTOM_OP = SRC_ROOT / "third_party/ascend/language/cann/extension/custom_op.py"
BUILTIN_OPS = SRC_ROOT / "third_party/ascend/language/cann/extension/builtin_custom_ops.py"


def _split_defs(text):
    """把文件按顶层 def 切成 (名字, 函数体文本)。"""
    lines = text.splitlines()
    marks = [(i, m.group(1)) for i, l in enumerate(lines)
             for m in [re.match(r"def (\w+)\(", l)] if m]
    out = []
    for k, (i, name) in enumerate(marks):
        j = marks[k + 1][0] if k + 1 < len(marks) else len(lines)
        out.append((name, "\n".join(lines[i:j])))
    return out


def main():
    ld_text = LIBDEVICE.read_text(encoding="utf-8")
    mo_text = MATH_OPS.read_text(encoding="utf-8")

    defs = _split_defs(ld_text)
    # 四分类（按「函数体里有几处 extern_elementwise 调用」×「有没有纯 IR 组合」判定）：
    #   A 单菜单：恰 1 处 extern、无纯 IR —— 不按开关分流，直接点 __hmf_ 符号
    #   B 双菜单：>=2 处 extern、无纯 IR —— 按开关/架构换一张符号表，两端都还是符号
    #   C 符号/纯 IR 分流：>=1 处 extern + 有纯 IR —— 一端点符号，另一端自己算
    #   D 全程纯 IR：0 处 extern —— 从不出现 __hmf_
    single_menu, dual_menu, mixed, pure_ir = [], [], [], []
    for name, body in defs:
        n_extern = body.count("extern_elementwise(")
        # 纯 IR 组合的证据：用 semantic.* 算子拼，或直接调 _builder.create_*（如 fast_expf 的 create_exp）
        has_semantic = bool(re.search(r"\bsemantic\.\w+|_builder\.create_\w+", body))
        if n_extern == 0:
            pure_ir.append(name)
        elif has_semantic:
            mixed.append(name)
        elif n_extern >= 2:
            dual_menu.append(name)
        else:
            single_menu.append(name)
    # 旧口径（保留以便对账）：只出现 extern、不含纯 IR 组合的 = A + B
    extern_only = single_menu + dual_menu

    static = {
        "libdevice_lines": len(ld_text.splitlines()),
        "libdevice_top_level_defs": len(defs),
        "core_extern_decorators": ld_text.count("@core.extern"),
        "core_builtin_decorators": ld_text.count("@core.builtin"),
        "hmf_occurrences": len(re.findall(r"__hmf_", ld_text)),
        "hmf_distinct_symbols": len(set(re.findall(r"__hmf_[A-Za-z0-9_]*", ld_text))),
        "extern_elementwise_calls": ld_text.count("extern_elementwise("),
        "single_menu_count": len(single_menu),
        "single_menu_names": single_menu,
        "dual_menu_count": len(dual_menu),
        "dual_menu_names": dual_menu,
        "symbol_or_pure_ir_count": len(mixed),
        "symbol_or_pure_ir_names": mixed,
        "pure_ir_only_count": len(pure_ir),
        "pure_ir_only_names": pure_ir,
        "extern_no_pure_ir_count": len(extern_only),
        "path_counts_sum": len(single_menu) + len(dual_menu) + len(mixed) + len(pure_ir),
        "math_ops_lines": len(mo_text.splitlines()),
        "math_ops_jit_defs": mo_text.count("@jit"),
        "math_ops_def_names": re.findall(r"^def (\w+)", mo_text, re.M),
        "custom_op_lines": len(CUSTOM_OP.read_text(encoding="utf-8").splitlines()),
        "builtin_custom_ops_lines": len(BUILTIN_OPS.read_text(encoding="utf-8").splitlines()),
        "register_custom_op_decorations": len(
            re.findall(r"^@register_custom_op", BUILTIN_OPS.read_text(encoding="utf-8"), re.M)),
    }

    # ---- B) acos 纯 IR 逼近的真实数值 ----
    env, cleanup = build_env(simt_enabled=False)
    try:
        tl = env.tl_core
        ld = env.libdevice
        ld.triton_enable_libdevice_simt = lambda: False
        ld.is_compile_on_910_95 = False
        pts = []
        for xv in (0.0, 0.2, -0.4, 0.55, 0.7, -0.7, 0.85, -0.85):
            b = env.FakeBuilder()
            x = tl.tensor(handle=float(xv), type=tl.float32)
            got = ld.acos(x, _builder=b).handle
            ref = pymath.acos(xv)
            pts.append({
                "x": xv,
                "branch": "center(|x|<0.6)" if abs(xv) < 0.6 else "mid(|x|>=0.6)",
                "acos_reduced": round(got, 6),
                "math_acos": round(ref, 6),
                "abs_err": round(abs(got - ref), 6),
                "used_extern": any(c[0] == "create_extern_elementwise" for c in b.calls),
            })
        max_err = max(p["abs_err"] for p in pts)

        # extern 分支（SIMT+910_95）下 acos 一条算术都不算，直接点符号
        ld.triton_enable_libdevice_simt = lambda: True
        ld.is_compile_on_910_95 = True
        b2 = env.FakeBuilder()
        r = ld.acos(tl.tensor(handle=0.3, type=tl.float32), _builder=b2)
        extern_branch = {
            "symbol": r.handle.symbol,
            "arith_calls": len([c for c in b2.calls if c[0].startswith("create_f")]),
        }
    finally:
        cleanup()

    trace = {
        "mechanism": "m5",
        "source": ["third_party/ascend/language/cann/libdevice.py",
                   "third_party/ascend/language/cann/extension/math_ops.py"],
        "pin": "2badfc89e",
        "static_census": static,
        "acos_points": pts,
        "acos_max_abs_err": max_err,
        "acos_extern_branch": extern_branch,
        "note": ("静态盘点跑在 pin 源码上；acos 数值跑在精简版上（多项式逼近是纯 CPU 可复现的数学）。"
                 "__hmf_ 符号本身的数值需昇腾 NPU/CANN 才能算，本章不伪造真机数值。"),
    }
    dump(trace, "m5.json")


if __name__ == "__main__":
    main()
