"""ch05 mech m07-cast-big-dispatch — cast 大 dispatch 的真 IR 实证（pin 3.2.0）。

pin 配方（INSTANCE.md）：pip install triton==3.2.0，前端 + libtriton.so 与 pin 逐字节一致，
headless 无 GPU。IR 取自 backend add_stages 的 make_ttir 之后（cast 的 arith.* op 在此可见）。
每命中 semantic.cast 的一支就发一个真 IR op —— 这是「每次 cast 是真开销」的字面证据。
运行：v32/bin/python run_cast_ir.py > cast_ir.txt
"""
import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends import backends
from triton._C.libtriton import ir

# capability 90 (Hopper) so every fp8 variant is at least parseable
backend = backends["nvidia"].compiler(GPUTarget("cuda", 90, 32))
options = backend.parse_options({})


def ttir(fn, signature, constants={}):
    src = ASTSource(fn=fn, signature=signature, constants=constants)
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    mod = src.make_ir(options, backend.get_codegen_implementation(),
                      backend.get_module_map(), ctx)
    return str(mod)


@triton.jit
def k_cast(xp, p16, p32, pi, pf):
    x = tl.load(xp)                 # fp32
    a = x.to(tl.float16)            # fp32 -> fp16  : truncate (create_fp_trunc)
    b = a.to(tl.float32)            # fp16 -> fp32  : extend   (create_fp_ext)
    s = x.to(tl.int32)             # fp32 -> int32 : fp_to_si  (create_fp_to_si)
    f = s.to(tl.float32)           # int32 -> fp32 : si_to_fp  (create_si_to_fp)
    tl.store(p16, a)
    tl.store(p32, b)
    tl.store(pi, s)
    tl.store(pf, f)


@triton.jit
def k_bf16_twohop(xp, yp):
    x = tl.load(xp)                 # fp16
    z = x.to(tl.bfloat16)          # fp16 -> bf16 : two-hop via fp32
    tl.store(yp, z)


s_cast = ttir(k_cast, {"xp": "*fp32", "p16": "*fp16", "p32": "*fp32",
                       "pi": "*i32", "pf": "*fp32"})
s_bf = ttir(k_bf16_twohop, {"xp": "*fp16", "yp": "*bf16"})

print("== k_cast: one arith op per cast branch (tracing-period make_ir) ==")
for op in ["arith.truncf", "arith.extf", "arith.fptosi", "arith.sitofp",
           "arith.fptoui", "arith.uitofp"]:
    print(f"{op} count: {s_cast.count(op)}")
print("--- cast op lines ---")
for line in s_cast.splitlines():
    st = line.strip()
    if any(o in st for o in ["truncf", "extf", "fptosi", "sitofp"]):
        print(st)

print()
print("== k_bf16_twohop: fp16->bf16 detours via fp32 (two IR ops) ==")
print("arith.extf count:", s_bf.count("arith.extf"))     # fp16 -> fp32
print("arith.truncf count:", s_bf.count("arith.truncf"))  # fp32 -> bf16
for line in s_bf.splitlines():
    st = line.strip()
    if any(o in st for o in ["extf", "truncf"]):
        print(st)
