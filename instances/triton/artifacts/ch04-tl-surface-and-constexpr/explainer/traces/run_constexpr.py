"""ch04 mech constexpr-dunder-class — constexpr 包装编译期值、重载全部 dunder。

pin 配方（INSTANCE.md）：pip install triton==3.2.0，前端与 pin 逐字节相同，headless。
运行：v32/bin/python run_constexpr.py > constexpr.txt
纯 Python 类行为，无需 GPU。
"""
from triton.language.core import constexpr, _constexpr_to_value, _unwrap_if_constexpr

print("== constexpr: wrap a compile-time value, forward all dunders ==")
c = constexpr(256)                         # __init__ stores self.value = 256
print("constexpr(256) ->", repr(c))        # constexpr[256]

print("c + 8   -> __add__      ->", repr(c + 8))     # 256 + 8   = constexpr[264]
print("c * 4   -> __mul__      ->", repr(c * 4))     # 256 * 4   = constexpr[1024]
print("c // 64 -> __floordiv__ ->", repr(c // 64))   # 256 // 64 = constexpr[4]
print("c == 256 -> __eq__      ->", repr(c == 256))  # constexpr[True]

# __bool__ : lets a constexpr drive a compile-time if branch (回指 ch01 折叠)
print("bool(constexpr(256)) -> __bool__ ->", bool(constexpr(256)))  # True
print("bool(constexpr(0))   -> __bool__ ->", bool(constexpr(0)))    # False

# __index__ : lets a constexpr be a list/shape subscript
L = [10, 11, 12, 13]
c2 = constexpr(2)
print("L =", L)
print("L[constexpr(2)] -> __index__ ->", L[c2])     # 12

# the two shuttle helpers unwrap constexpr back to the raw value
print("_constexpr_to_value(constexpr(256)) ->", _constexpr_to_value(constexpr(256)))  # 256
print("_unwrap_if_constexpr(constexpr(7)) ->", _unwrap_if_constexpr(constexpr(7)))    # 7
print("_unwrap_if_constexpr(42) ->", _unwrap_if_constexpr(42))                        # 42 (passthrough)
