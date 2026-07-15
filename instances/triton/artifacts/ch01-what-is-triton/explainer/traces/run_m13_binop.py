#!/usr/bin/env python3
"""m13：kernel 体里的 `x + y` 根本不经过 visit_Call——它是 ast.BinOp，走 visit_BinOp
→ _apply_binary_method，由追踪器【主动注入 _builder】，所以 tensor.__add__(@builtin)
拿得到守门票、能建 arith op。这正是 m04 反例的成立前提：第③岔当场执行的普通 Python
函数体里，`x + y` 由 Python 解释器直调 __add__、无人注入 _builder → 报错。

instrument visit_Call 与 visit_BinOp，观测 `x + y` 走的是哪条路。headless，无 GPU。
"""
import ast
import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource, make_backend
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir
from triton.compiler import code_generator as CG

print(f"# triton {triton.__version__}")

CALL_LOG, BINOP_LOG = [], []
_ovc = CG.CodeGenerator.visit_Call
_ovb = CG.CodeGenerator.visit_BinOp


def tvc(self, node):
    try:
        fn = CG._unwrap_if_constexpr(self.visit(node.func))
        CALL_LOG.append(getattr(fn, "__name__", repr(fn)))
    except Exception:
        pass
    return _ovc(self, node)


def tvb(self, node):
    BINOP_LOG.append(type(node.op).__name__)
    return _ovb(self, node)


CG.CodeGenerator.visit_Call = tvc
CG.CodeGenerator.visit_BinOp = tvb


@triton.jit
def addy_kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    out = x + y                       # ← ast.BinOp，不是 ast.Call
    tl.store(out_ptr + offs, out)


target = GPUTarget("cuda", 90, 32)
backend = make_backend(target)
options = backend.parse_options({})
ctx = ir.context()
ir.load_dialects(ctx)
backend.load_dialects(ctx)
src = ASTSource(addy_kernel,
                signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32"},
                constants={"BLOCK_SIZE": 16})
mod = src.make_ir(options, backend.get_codegen_implementation(),
                  backend.get_module_map(), ctx)
txt = str(mod)

print("\n---- visit_Call 记录到的调用（写成 f(...) 的）----")
print("  ", CALL_LOG)
print("\n---- visit_BinOp 记录到的运算符 ----")
print("  ", BINOP_LOG)
print("\n  `+`（ast.Add）是否出现在 visit_Call 日志里:", "add" in [c.lower() for c in CALL_LOG])
print("  `+`（ast.Add）是否出现在 visit_BinOp 日志里:", "Add" in BINOP_LOG)
print("\n---- x + y 追踪成的 IR op（fp32 张量加）----")
import re
print("  arith.addf 出现次数:", len(re.findall(r"\barith\.addf\b", txt)))
