#!/usr/bin/env python3
"""m2 worked example：真跑 ContainsReturnChecker（pin v3.2.0）判 if 子树是否含 return。

配方同 INSTANCE.md：triton==3.2.0 venv 前端与 pin 逐字节相同。直接 import 本章枢纽用到的
ContainsReturnChecker（python/triton/compiler/code_generator.py:L104-L189），对真实 AST 子树
run 它，观测 True/False——正是 visit_If(L688) 用来决定走 CFG 还是 scf.if 的判据。

用法：<repo>/instances/triton/v32/bin/python run_m2_return_checker.py
"""
import ast
import json
import inspect
import textwrap
import triton
from triton.compiler.code_generator import ContainsReturnChecker

print(f"# triton {triton.__version__}")


# 一个被 transitively 调用的 jit 函数，体内含 return（用于验证跨函数递归语义）。
@triton.jit
def helper_with_return(x):
    if x > 0:
        return x
    return 0


def first_if(src):
    """取源码里第一个 If 节点（visit_If 的输入单位）。"""
    tree = ast.parse(textwrap.dedent(src))
    fn = tree.body[0]
    for stmt in fn.body:
        if isinstance(stmt, ast.If):
            return stmt
    raise RuntimeError("no If found")


# gscope 需能解析被调 jit 函数名 → JITFunction，_visit_function 才能再 parse 展开。
gscope = {"helper_with_return": helper_with_return}

cases = []

# case A：if 体内直接 return → True → visit_If 走 CFG（visit_if_top_level）
srcA = """
def f(c):
    if c > 0:
        return 1
    x = 2
"""
rA = ContainsReturnChecker(gscope).visit(first_if(srcA))
cases.append(("A_direct_return_in_body", rA, "CFG (visit_if_top_level)"))

# case B：if 只做赋值，无 return → False → visit_If 走 scf.if（visit_if_scf）
srcB = """
def f(c):
    if c > 0:
        x = 1
    else:
        x = 2
"""
rB = ContainsReturnChecker(gscope).visit(first_if(srcB))
cases.append(("B_assign_only", rB, "scf.if (visit_if_scf)"))

# case C：if 体内以裸表达式语句调用体内含 return 的 jit 函数 → True（transitively）→ CFG
srcC = """
def f(c):
    if c > 0:
        helper_with_return(c)
    z = 3
"""
rC = ContainsReturnChecker(gscope).visit(first_if(srcC))
cases.append(("C_transitive_bare_call", rC, "CFG (transitively)"))

# case D：同一 transitive 调用但结果被赋值 y=helper(...) → False！
# visit_Assign 直接 return False，不下探 RHS 的 Call（源 L156-L159 的短路）——真实非对称行为。
srcD = """
def f(c):
    if c > 0:
        y = helper_with_return(c)
    z = 3
"""
rD = ContainsReturnChecker(gscope).visit(first_if(srcD))
cases.append(("D_transitive_call_but_assigned", rD, "scf.if (Assign 短路，未下探 RHS)"))

print("\n================ ContainsReturnChecker on real AST ================")
for name, res, dispatch in cases:
    print(f"  {name:34s} -> contains_return={res!s:5s} -> {dispatch}")

out = {"triton_version": triton.__version__,
       "cases": [{"case": n, "contains_return": bool(r), "dispatch": d} for n, r, d in cases]}
with open(__file__.rsplit("/", 1)[0] + "/m2_return_checker.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n# wrote m2_return_checker.json")
