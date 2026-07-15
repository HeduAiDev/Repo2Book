"""ch04 mech two-tier-triton-surface — triton.cdiv(host) vs tl.cdiv(kernel).

pin 配方（INSTANCE.md）：pip install triton==3.2.0，前端与 pin 源码逐字节相同，headless。
运行：v32/bin/python run_two_tier.py > two_tier.txt
纯前端/host 行为，无需 GPU。
"""
import triton
import triton.language as tl
from triton.language.core import is_builtin

print("== two-tier surface: triton.cdiv vs tl.cdiv ==")
print("type(triton.cdiv):", type(triton.cdiv).__name__)      # function (host helper)
print("type(tl.cdiv):", type(tl.cdiv).__name__)              # JITFunction (traced)
print("is_builtin(tl.cdiv):", is_builtin(tl.cdiv))           # False (@jit standard)

# host-side value: grid computation, plain Python int
n_elements = 1000
BLOCK_SIZE = 256
print("n_elements =", n_elements, " BLOCK_SIZE =", BLOCK_SIZE)
print("1000 + 256 - 1 =", 1000 + 256 - 1)                    # 1255
print("1255 // 256 =", 1255 // 256)                          # 4
print("triton.cdiv(1000, 256) =", triton.cdiv(1000, 256))    # 4  (returns Python int now)

# tl.cdiv is NOT evaluated on host: it is a JITFunction, only traced into IR
# inside an @triton.jit body. Calling it directly does not return an int here.
print("tl.cdiv is a JITFunction (kernel-side); traced into IR, not evaluated on host")
