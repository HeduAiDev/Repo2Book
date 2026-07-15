"""ch04 mech builtin-marker-contract — @builtin 标记与调用契约。

pin 配方（INSTANCE.md）：pip install triton==3.2.0，前端与 pin 逐字节相同，headless。
运行：v32/bin/python run_builtin_contract.py > builtin_contract.txt
纯前端行为，无需 GPU。
"""
import triton.language as tl
from triton.language.core import is_builtin

print("== @builtin marker & call contract ==")
# marker bit set by @builtin (core.py:L37 setattr wrapper __triton_builtin__ True)
print("program_id __triton_builtin__:", getattr(tl.program_id, "__triton_builtin__", False))
print("cdiv __triton_builtin__:", getattr(tl.cdiv, "__triton_builtin__", False))
print("is_builtin(tl.program_id):", is_builtin(tl.program_id))   # True
print("is_builtin(tl.cdiv):", is_builtin(tl.cdiv))               # False

# contract: @builtin function called OUTSIDE @triton.jit -> no _builder -> raise
# (core.py:L30-L34: prints kwargs {} then raises ValueError)
print("-- calling tl.program_id(axis=0) outside @triton.jit --")
try:
    tl.program_id(axis=0)
except ValueError as e:
    print("raised ValueError:", str(e))

# tl.cdiv (@jit, not builtin) needs no _builder; it is a JITFunction, no such guard.
print("tl.cdiv carries no _builder guard (is_builtin False): traced normally")
