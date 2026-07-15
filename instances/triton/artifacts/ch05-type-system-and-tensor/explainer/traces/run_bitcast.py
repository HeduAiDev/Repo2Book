"""ch05 mech m08-bitcast-equal-width — 等宽约束 + 同一堆 bit 重解释（pin 3.2.0 实证）。

pin 配方（INSTANCE.md）：semantic.bitcast 与 pin 源码逐字节一致。等宽通过则发 tt.bitcast；
位宽不等则在 semantic 层 raise ValueError（早于任何 IR op）。
运行：v32/bin/python run_bitcast.py > bitcast.txt
"""
import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends import backends
from triton._C.libtriton import ir

backend = backends["nvidia"].compiler(GPUTarget("cuda", 90, 32))
options = backend.parse_options({})


def ttir(fn, signature, constants={}):
    src = ASTSource(fn=fn, signature=signature, constants=constants)
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    return str(src.make_ir(options, backend.get_codegen_implementation(),
                           backend.get_module_map(), ctx))


@triton.jit
def k_bitcast_ok(xp, yp):
    x = tl.load(xp)                          # fp32 (32 bits)
    y = x.to(tl.int32, bitcast=True)         # fp32 -> int32 : equal width 32, tt.bitcast
    tl.store(yp, y)


@triton.jit
def k_bitcast_16(xp, yp):
    x = tl.load(xp)                          # fp16 (16 bits)
    y = x.to(tl.bfloat16, bitcast=True)      # fp16 -> bf16 : equal width 16, tt.bitcast
    tl.store(yp, y)


@triton.jit
def k_bitcast_bad(xp, yp):
    x = tl.load(xp)                          # fp32 (32 bits)
    y = x.to(tl.float16, bitcast=True)       # fp32 -> fp16 : 32 != 16 -> ValueError
    tl.store(yp, y)


print("== equal-width bitcast fp32<->int32 emits one tt.bitcast ==")
s_ok = ttir(k_bitcast_ok, {"xp": "*fp32", "yp": "*i32"})
print("tt.bitcast count:", s_ok.count("tt.bitcast"))
for line in s_ok.splitlines():
    if "bitcast" in line:
        print(line.strip())

print()
print()
print("== equal-width bitcast fp16<->bf16 also emits one tt.bitcast (16==16) ==")
s_16 = ttir(k_bitcast_16, {"xp": "*fp16", "yp": "*bf16"})
print("tt.bitcast count:", s_16.count("tt.bitcast"))
for line in s_16.splitlines():
    st = line.strip()
    if st.startswith("%1 = tt.bitcast"):
        print(st)

print()
print("primitive_bitwidth fp32 =", tl.float32.primitive_bitwidth,
      " int32 =", tl.int32.primitive_bitwidth,
      " fp16 =", tl.float16.primitive_bitwidth,
      " bf16 =", tl.bfloat16.primitive_bitwidth)

print()
print("== unequal-width bitcast fp32->fp16 raises before any IR op ==")
try:
    ttir(k_bitcast_bad, {"xp": "*fp32", "yp": "*fp16"})
    print("NO ERROR (unexpected)")
except Exception as e:
    # error message carries the two sizes 32 and 16
    msg = str(e)
    print("raised:", type(e).__name__)
    print("message:", msg.splitlines()[-1] if msg else e)
    # also surface the ValueError text explicitly
    import traceback
    tb = traceback.format_exc()
    for line in tb.splitlines():
        if "Cannot bitcast" in line:
            print(line.strip())
