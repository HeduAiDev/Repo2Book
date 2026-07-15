#!/usr/bin/env python3
"""m01：instrument CodeGenerator.visit_Call，记录 kernel 体内每一次 f(...) 调用
落到三岔的哪一岔。载体是一个把三岔都覆盖到的自撰 kernel。

第①岔 = 被调是 JITFunction（如 tl.cdiv）→ call_JitFunction（递归追踪成 tt.func + tt.call）
第②岔 = 被调是 @builtin（如 tl.program_id / tl.arange / tl.store）→ 注入 _builder 建 op
第③岔 = 其余普通 Python 可调用物（如 host triton.cdiv）→ 当场执行

前置截胡（三岔之前）= static_assert/static_print/int/len。

headless，无 GPU。用 triton==3.2.0 venv 跑。
"""
import inspect
import triton
import triton.language as tl
import triton.language.core as core
from triton.compiler.compiler import ASTSource, make_backend
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir
from triton.compiler import code_generator as CG
from triton.runtime.jit import JITFunction

print(f"# triton {triton.__version__}")

LOG = []
_orig_visit_call = CG.CodeGenerator.visit_Call


def traced_visit_Call(self, node):
    # 复算 visit_Call 头部的分类逻辑（不改变行为，只观测）
    try:
        fn = CG._unwrap_if_constexpr(self.visit(node.func))
    except Exception:
        return _orig_visit_call(self, node)
    name = getattr(fn, "__name__", repr(fn))
    if self.statically_implemented_functions.get(fn) is not None:
        branch = "前置截胡"
    elif isinstance(fn, JITFunction):
        branch = "① JITFunction → call_JitFunction"
    elif (hasattr(fn, "__self__") and CG._is_triton_value(fn.__self__)) or core.is_builtin(fn):
        branch = "② @builtin → 注入 _builder 建 op"
    else:
        branch = "③ 普通 Python → 当场执行"
    LOG.append((name, branch))
    return _orig_visit_call(self, node)


CG.CodeGenerator.visit_Call = traced_visit_Call


@triton.jit
def demo_kernel(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    n_blocks = tl.cdiv(n_elements, BLOCK_SIZE)   # ① 被调是 JITFunction
    pid = tl.program_id(axis=0)                  # ② @builtin
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # ② tl.arange @builtin
    tl.store(out_ptr + offs, n_blocks)           # ② tl.store @builtin


target = GPUTarget("cuda", 90, 32)
backend = make_backend(target)
options = backend.parse_options({})
ctx = ir.context()
ir.load_dialects(ctx)
backend.load_dialects(ctx)
src = ASTSource(demo_kernel,
                signature={"out_ptr": "*i32", "n_elements": "i32"},
                constants={"BLOCK_SIZE": 1024})
mod = src.make_ir(options, backend.get_codegen_implementation(),
                  backend.get_module_map(), ctx)

print("\n---- demo_kernel 体内每一次 f(...) 调用落到哪一岔 ----")
for i, (name, branch) in enumerate(LOG, 1):
    print(f"  调用#{i}  {name:14s} → {branch}")

# 统计各岔命中次数
from collections import Counter
c = Counter(b for _, b in LOG)
print("\n---- 各岔命中次数 ----")
for b, n in c.items():
    print(f"  {b}: {n}")

print("\n---- 佐证：tl.cdiv 走①的产物是 IR 里的 tt.call ----")
print("  IR 出现 tt.call @cdiv:", "tt.call @cdiv" in str(mod))
print("  tl.cdiv 是 JITFunction:", isinstance(tl.cdiv, JITFunction))
print("  tl.store is_builtin:", core.is_builtin(tl.store))
print("  tl.program_id is_builtin:", core.is_builtin(tl.program_id))
print("  tl.arange is_builtin:", core.is_builtin(tl.arange))
