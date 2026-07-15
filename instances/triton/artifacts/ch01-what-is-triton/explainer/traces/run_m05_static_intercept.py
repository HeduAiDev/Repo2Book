#!/usr/bin/env python3
"""m05 二选一对照实验：前置截胡表里 int 与 static_assert 的命运完全不同。

  A. 对照组（真 pin，两者都截胡）：为假的 static_assert 抛 CompileTimeAssertionFailure。
  B. 摘掉 int 截胡              → kernel 照常编译（tt.make_range 建出）→ 截胡是【便利】。
  C. 摘掉 static_assert 截胡    → 为假的断言【静默通过】，断言被吞 → 截胡是【必需】。

做法：临时改写 CodeGenerator.statically_implemented_functions 这张表，再 make_ir。
headless，无 GPU。用 triton==3.2.0 venv 跑。
"""
import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource, make_backend
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir
from triton.compiler.code_generator import CodeGenerator

print(f"# triton {triton.__version__}")


def build(kernel, sig, consts):
    target = GPUTarget("cuda", 90, 32)
    backend = make_backend(target)
    options = backend.parse_options({})
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    src = ASTSource(kernel, signature=sig, constants=consts)
    return src.make_ir(options, backend.get_codegen_implementation(),
                       backend.get_module_map(), ctx)


ORIG_TABLE = dict(CodeGenerator.statically_implemented_functions)
print("\n---- 前置截胡表的成员（statically_implemented_functions）----")
for fn in ORIG_TABLE:
    print("  ", getattr(fn, "__name__", repr(fn)))


# ============ int 用作 shape：摘掉截胡后是否还能工作？ ============
@triton.jit
def int_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, int(BLOCK_SIZE))   # int(...) 作 shape 值
    tl.store(out_ptr + offs, offs)


SIG = {"out_ptr": "*i32"}
CST = {"BLOCK_SIZE": 16}

print("\n================ A. int：对照组（int 被截胡，真 pin 行为）================")
try:
    mod = build(int_kernel, SIG, CST)
    print("  编译通过；IR 出现 tt.make_range:", "tt.make_range" in str(mod))
except Exception as e:
    print("  失败：", type(e).__name__, e)

print("\n================ B. int：摘掉 int 截胡 ================")
patched = dict(ORIG_TABLE)
for fn in list(patched):
    if fn is int:
        del patched[fn]
CodeGenerator.statically_implemented_functions = patched
try:
    mod = build(int_kernel, SIG, CST)
    print("  编译通过；IR 出现 tt.make_range:", "tt.make_range" in str(mod))
    print("  → 结论：int 截胡是【便利】，不是必需（落第③岔也能工作）")
except Exception as e:
    print("  失败：", type(e).__name__, e)
finally:
    CodeGenerator.statically_implemented_functions = dict(ORIG_TABLE)


# ============ static_assert：为假的断言，摘掉截胡后会怎样？ ============
@triton.jit
def assert_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    tl.static_assert(BLOCK_SIZE == 999, "BLOCK_SIZE must be 999")  # 【为假】：实参是 16
    tl.store(out_ptr, BLOCK_SIZE)


print("\n================ C. static_assert：对照组（被截胡，真 pin 行为）================")
try:
    mod = build(assert_kernel, SIG, CST)
    print("  编译【通过】（不该发生）——断言没有生效")
except Exception as e:
    print(f"  抛出 {type(e).__name__}：{e}")
    print("  → 真 pin：为假的断言如实报错")

print("\n================ D. static_assert：摘掉 static_assert 截胡 ================")
patched = dict(ORIG_TABLE)
for fn in list(patched):
    if getattr(fn, "__name__", "") == "static_assert" or fn is tl.static_assert:
        del patched[fn]
# 精确按 language.core.static_assert 摘除
import triton.language.core as core
patched.pop(core.static_assert, None)
CodeGenerator.statically_implemented_functions = patched
try:
    mod = build(assert_kernel, SIG, CST)
    print("  编译【通过】，为假的断言被【静默吞掉】，没有任何报错")
    print("  IR 里有没有 assert op:", "assert" in str(mod).lower())
    print("  → 结论：static_assert 截胡是【必需】——不截胡则断言落第②岔@builtin 空壳(pass)，丢失")
except Exception as e:
    print(f"  抛出 {type(e).__name__}：{e}")
finally:
    CodeGenerator.statically_implemented_functions = dict(ORIG_TABLE)
