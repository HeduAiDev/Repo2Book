"""ch04 mech static-range-vs-range — 追踪期 IR 实证（pin 3.2.0）。

pin 配方（INSTANCE.md）：pip install triton==3.2.0，前端 + libtriton.so 与 pin 一致，headless 无 GPU。
IR 取自 ASTSource.make_ir(...)（追踪期，任何 pass 之前）。
运行：v32/bin/python run_ranges_ir.py > ranges_ir.txt
"""
import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends import backends
from triton._C.libtriton import ir

backend = backends["nvidia"].compiler(GPUTarget("cuda", 80, 32))
options = backend.parse_options({})


def trace_ir(fn, signature, constants):
    src = ASTSource(fn=fn, signature=signature, constants=constants)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    mod = src.make_ir(options, backend.get_codegen_implementation(),
                      backend.get_module_map(), context)
    return str(mod)


@triton.jit
def k_static(x_ptr, BLOCK_SIZE: tl.constexpr):
    acc = 0
    for i in tl.static_range(4):        # compile-time full unroll
        acc = acc + i
    tl.store(x_ptr, acc)


@triton.jit
def k_range(x_ptr):
    acc = 0
    for i in tl.range(0, 4, num_stages=3, loop_unroll_factor=2):  # runtime loop + hints
        acc = acc + i
    tl.store(x_ptr, acc)


s_static = trace_ir(k_static, {"x_ptr": "*i32", "BLOCK_SIZE": "constexpr"},
                    {"BLOCK_SIZE": 4})
s_range = trace_ir(k_range, {"x_ptr": "*i32"}, {})

print("== static_range(4): compile-time full unroll ==")
print("scf.for count:", s_static.count("scf.for"))           # 0
print("arith.addi count:", s_static.count("arith.addi"))     # 8 (4 iters unrolled)
print("tt.num_stages present:", "tt.num_stages" in s_static)  # False
print(s_static)

print("== range(0, 4, num_stages=3, loop_unroll_factor=2): scf.for + attrs ==")
print("scf.for count:", s_range.count("scf.for"))            # 1
print("arith.addi count:", s_range.count("arith.addi"))      # 2 (loop body once)
print("tt.num_stages = 3 present:", "tt.num_stages = 3" in s_range)                 # True
print("tt.loop_unroll_factor = 2 present:", "tt.loop_unroll_factor = 2" in s_range)  # True
print(s_range)
