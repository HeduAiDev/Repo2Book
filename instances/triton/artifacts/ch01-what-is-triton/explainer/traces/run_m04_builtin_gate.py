#!/usr/bin/env python3
"""m04 反例：kernel 体里调 host 侧 triton.cdiv（纯 Python）→ 落第③岔当场执行
→ 其体内 (x + y - 1) 触发 tl.tensor.__add__（@builtin）→ wrapper 查不到 _builder
→ ValueError → 被 CodeGenerator.visit 兜底包成 CompilationError。

对照：同一 kernel 改用 tl.cdiv（@jit，第①岔）编译通过。

headless，无 GPU，只跑 make_ir。用 triton==3.2.0 venv 跑。
"""
import traceback
import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource, make_backend
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir

print(f"# triton {triton.__version__}")


def build(kernel):
    target = GPUTarget("cuda", 90, 32)
    backend = make_backend(target)
    options = backend.parse_options({})
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    src = ASTSource(kernel,
                    signature={"out_ptr": "*i32", "n_elements": "i32"},
                    constants={"BLOCK_SIZE": 1024})
    return src.make_ir(options, backend.get_codegen_implementation(),
                       backend.get_module_map(), ctx)


# ---- 对照组：tl.cdiv（@jit，第①岔）→ 编译通过 ----
@triton.jit
def good_kernel(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    n_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    tl.store(out_ptr, n_blocks)


print("\n================ 对照组：kernel 体内用 tl.cdiv（@jit 第①岔）================")
try:
    mod = build(good_kernel)
    print("结果：编译【通过】。IR 里出现 tt.call:",
          "tt.call @cdiv" in str(mod))
except Exception as e:
    print("意外失败：", type(e).__name__, e)

# triton.cdiv 与 tl.cdiv 的身份对比
import triton.language.core as core
print("\n---- 身份对比 ----")
print("  type(triton.cdiv)   =", type(triton.cdiv).__name__,
      "  is_builtin =", core.is_builtin(triton.cdiv))
print("  type(tl.cdiv)       =", type(tl.cdiv).__name__,
      "  是 JITFunction:", isinstance(tl.cdiv, triton.runtime.jit.JITFunction))
# host triton.cdiv 是否在 CodeGenerator 的第③岔白名单里？
from triton.compiler.code_generator import CodeGenerator
# builtin_namespace 是实例属性；用类源码里的默认值间接确认——直接查它不在 tl 的原语集合
print("  triton.cdiv is tl.cdiv:", triton.cdiv is tl.cdiv)


# ---- 反例：host 侧 triton.cdiv（纯 Python）落第③岔 ----
@triton.jit
def bad_kernel(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)   # ← host 侧纯 Python，落第③岔
    tl.store(out_ptr, n_blocks)


print("\n================ 反例：kernel 体内用 host triton.cdiv（第③岔当场执行）================")
try:
    mod = build(bad_kernel)
    print("结果：意外编译通过（不该发生）")
except Exception as e:
    print("结果：抛出", type(e).__name__)
    tb = traceback.format_exc()
    # 打印异常链里的关键行
    for line in tb.splitlines():
        if any(k in line for k in ("ValueError", "CompilationError",
                                   "_builder", "Did you forget",
                                   "builtin", "__add__")):
            print("  |", line.strip())
