"""ch05 mech m02-fp8-family-encoding — 浮点 (mantissa, bitwidth, exponent_bias) 三元组
取自 pin 3.2.0 的真 dtype 对象；精度↔带宽账的派生量在此就地算出并打印。

pin 配方（INSTANCE.md）：pip install triton==3.2.0，dtype.__init__ 与 pin 源码逐字节一致。
每个三元组即 core.py:L316-L354 的字面量；派生量（相对精度 2^-(mant+1)、相对带宽）由三元组算。
运行：v32/bin/python run_fp8_encoding.py > fp8_encoding.txt
"""
from triton.language.core import dtype

names = ["fp8e4nv", "fp8e5", "fp8e4b15", "fp8e4b8", "fp8e5b16",
         "fp16", "bf16", "fp32", "fp64"]

print("== dtype triple (mantissa, bitwidth, exponent_bias) from pin core.py:L316-L354 ==")
print(f"{'name':10} {'bitwidth':>8} {'mantissa':>8} {'exp_bias':>8} "
      f"{'rel_prec=2^-(m+1)':>18} {'bytes/elem':>10} {'x vs fp32 bw':>12}")
for n in names:
    d = dtype(n)
    bw = d.primitive_bitwidth
    m = d.fp_mantissa_width
    bias = d.exponent_bias
    rel_prec = 2.0 ** (-(m + 1))
    bytes_per = bw // 8
    ratio_vs_fp32 = bw / 32.0
    print(f"{n:10} {bw:8d} {m:8d} {bias:8d} {rel_prec:18.6g} "
          f"{bytes_per:10d} {ratio_vs_fp32:12.3f}")

print()
print("== bandwidth account: fp8 is half of fp16, quarter of fp32 ==")
print("fp8 bitwidth =", dtype("fp8e4nv").primitive_bitwidth,
      " fp16 =", dtype("fp16").primitive_bitwidth,
      " fp32 =", dtype("fp32").primitive_bitwidth)
print("fp8 / fp16 =", dtype("fp8e4nv").primitive_bitwidth / dtype("fp16").primitive_bitwidth)
print("fp8 / fp32 =", dtype("fp8e4nv").primitive_bitwidth / dtype("fp32").primitive_bitwidth)

print()
print("== two 8-bit fp8 trade precision vs range (same 8 bits) ==")
e4 = dtype("fp8e4nv"); e5 = dtype("fp8e5")
print("fp8e4nv: mantissa=%d exp_bits=%d bias=%d -> higher precision, narrower range"
      % (e4.fp_mantissa_width, 8 - 1 - e4.fp_mantissa_width, e4.exponent_bias))
print("fp8e5:   mantissa=%d exp_bits=%d bias=%d -> lower precision, wider range"
      % (e5.fp_mantissa_width, 8 - 1 - e5.fp_mantissa_width, e5.exponent_bias))
print("bf16 vs fp16 (both 16-bit): bf16 mantissa=%d bias=%d (fp32-range), fp16 mantissa=%d bias=%d"
      % (dtype("bf16").fp_mantissa_width, dtype("bf16").exponent_bias,
         dtype("fp16").fp_mantissa_width, dtype("fp16").exponent_bias))
