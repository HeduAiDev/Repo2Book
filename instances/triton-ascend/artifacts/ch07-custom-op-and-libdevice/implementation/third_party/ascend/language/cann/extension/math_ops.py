# SOURCE: third_party/ascend/language/cann/extension/math_ops.py
# 昇腾数学算子——libdevice 三条实现路径里的『第三条路』：既非 register_custom_op 注册
# 的自定义算子，也非 @core.extern 直调 __hmf_ 符号，而是 @jit 组合已有原语实现。
# 本文件很小(全文件三个函数)，dossier subtraction_plan 未批准任何删除，原样保留。

from math import pi as math_pi
from triton.language import core, math
from triton.language.core import float32, int1
from ..libdevice import atan, isnan, isinf
from triton.runtime.jit import jit


pi: core.constexpr = math_pi


@core._tensor_member_fn
@jit
@math._add_math_2arg_docstr("atan2")
def atan2(y, x):  # SOURCE: third_party/ascend/language/cann/extension/math_ops.py:L13-29
    _is_int8_type_x: core.constexpr = x.dtype.is_int8()
    core.static_assert(not _is_int8_type_x, "Expected dtype fp16/fp32/bf16, but got int8 or int1")
    _is_int8_type_y: core.constexpr = y.dtype.is_int8()
    core.static_assert(not _is_int8_type_y, "Expected dtype fp16/fp32/bf16, but got int8 or int1")
    _is_floating_type_x: core.constexpr = x.dtype.is_floating()
    core.static_assert(_is_floating_type_x == True, f"Expected dtype fp16/fp32/bf16, but got {core.constexpr(x.dtype)}")
    _is_floating_type_y: core.constexpr = y.dtype.is_floating()
    core.static_assert(_is_floating_type_y == True, f"Expected dtype fp16/fp32/bf16, but got {core.constexpr(y.dtype)}")
    half_pi: core.constexpr = 0.5 * pi
    base = core.where(x == 0, 0.0, atan(y.to(core.dtype("fp32")) / x.to(core.dtype("fp32"))))
    base = core.where((x == 0) & (y > 0), half_pi, base)
    base = core.where((x == 0) & (y < 0), -half_pi, base)

    add_pi = core.where((x < 0) & (y >= 0), pi, 0.0)
    sub_pi = core.where((x < 0) & (y < 0), -pi, 0.0)
    return (base + add_pi + sub_pi).to(x.dtype)


@core._tensor_member_fn
@jit
@math._add_math_1arg_docstr("isfinited")
def isfinited(x):  # SOURCE: third_party/ascend/language/cann/extension/math_ops.py:L32-42
    _is_int8_type: core.constexpr = x.dtype.is_int8()
    core.static_assert(not _is_int8_type, "Expected dtype fp16/fp32/bf16, but got int8 or int1")
    _is_floating_type: core.constexpr = x.dtype.is_floating()
    core.static_assert(_is_floating_type == True, f"Expected dtype fp16/fp32/bf16, but got {core.constexpr(x.dtype)}")
    nan_mask = isnan(x)
    inf_mask = isinf(x)
    return (~nan_mask & ~inf_mask).to(int1)


@core._tensor_member_fn
@jit
@math._add_math_1arg_docstr("finitef")
def finitef(x):  # SOURCE: third_party/ascend/language/cann/extension/math_ops.py:L44-53
    _is_int8_type: core.constexpr = x.dtype.is_int8()
    core.static_assert(not _is_int8_type, "finitef only supports float32, but got int8 or int1")
    core.static_assert(x.dtype == float32, f"finitef only supports float32, but got {core.constexpr(x.dtype)}")
    nan_mask = isnan(x)
    inf_mask = isinf(x)
    return (~nan_mask & ~inf_mask).to(int1)
