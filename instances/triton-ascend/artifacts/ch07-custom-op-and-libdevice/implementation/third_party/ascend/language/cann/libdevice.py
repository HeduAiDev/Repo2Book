# SOURCE: third_party/ascend/language/cann/libdevice.py
# 昇腾数学库——展示 libdevice 三条实现路径中的两条:
#   ① @core.extern 直调华为数学函数库 __hmf_ 符号(reciprocal 最简样例、tanh 双分支样例)
#   ② 无 __hmf_ 符号(或旧架构)时退回纯 triton IR 组合逼近(acos 多项式样例)
# 第三条路(@jit 组合已有原语，如 isfinited)在 extension/math_ops.py。
#
# SUBTRACTED: 真实文件还定义 log1p/relu/isinf/tan/atan/ilogb/ldexp/pow/isnan/div_rz/
# fast_dividef/fast_expf/fmod/float_as_int/atan2/trunc/round/sinh/cosh/acosh/asinh/
# atanh/expm1/nextafter/hypot/cyl_bessel_i0/signbit/erfinv/gamma/lgamma/nearbyint/
# asin/log10/copysign/rint 共约 30 个数学函数(third_party/ascend/language/cann/
# libdevice.py 全文件约 1032 行)。它们都是同一『extern 调 __hmf_ / 无符号退回纯 IR』
# 范式的重复实例——精简版保留 reciprocal(最简 extern)、tanh(双分支范式)、acos(纯 IR
# 逼近范式)三个代表即覆盖全部实现路径。其中 isnan/isinf/atan 三个额外保留，原因见下。
#
# 部分修正(非擅自扩大保留范围，而是修复 subtraction_plan 的一处遗漏):
# extension/math_ops.py(本章 must_keep 符号 isfinited 所在文件)顶部
# `from ..libdevice import atan, isnan, isinf` 是模块级 import，若真删掉这三个函数，
# math_ops.py 会在加载期直接 ImportError，连带炸穿 must_keep 的 isfinited——
# subtraction_plan.delete 列出的"其余数学函数"清单未考虑这处依赖。这里按"不能删到
# must_keep 跑不起来"的原则，额外保留这三个同范式的 extern 函数(每个都不到 10 行)，
# 不影响本章要讲的三分野主线。
# 其余被删除函数中，erfinv/gamma/cyl_bessel_i0 还额外带着数值逼近的系数常量数组与
# Newton 迭代/递推展开体——这些是特定数值分析细节而非本章主线(本章讲『三种实现路径
# 的分野』，不讲某个特殊函数的数值分析)，acos 已作为『纯 IR 逼近』代表保留其核心结构。

from triton.language import core, math, semantic
from triton._C.libtriton import ir
from triton.backends.ascend.utils import triton_enable_libdevice_simt
from triton.tools.get_ascend_devices import is_compile_on_910_95


@core.extern
def reciprocal(arg0, _builder=None):  # SOURCE: third_party/ascend/language/cann/libdevice.py:L28-34
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_recipf", core.dtype("fp32")),
            (core.dtype("fp16"),): ("__hmf_recipDh", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def isinf(arg0, _builder=None):  # SOURCE: third_party/ascend/language/cann/libdevice.py:L54-61
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_isinf", core.dtype("int1")),
            (core.dtype("fp16"),): ("__hmf_isinf", core.dtype("int1")),
            (core.dtype("bf16"),): ("__hmf_isinf", core.dtype("int1")),
        }, is_pure=True, _builder=_builder)


@core.extern
def atan(arg0, _builder=None):  # SOURCE: third_party/ascend/language/cann/libdevice.py:L73-79
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_atanf", core.dtype("fp32")),
            (core.dtype("fp16"),): ("__hmf_atanDh", core.dtype("fp16")),
        }, is_pure=True, _builder=_builder)


@core.extern
def tanh(arg0, _builder=None):  # SOURCE: third_party/ascend/language/cann/libdevice.py:L81-93
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("fp32"), ): ("__hmf_tanh_fp32", core.dtype("fp32")),
            }, is_pure=True, _builder=_builder)
    else:
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("fp32"), ): ("__hmf_tanhf", core.dtype("fp32")),
                (core.dtype("fp16"), ): ("__hmf_tanhDh", core.dtype("fp16")),
            }, is_pure=True, _builder=_builder)


@core.extern
def isnan(arg0, _builder=None):  # SOURCE: third_party/ascend/language/cann/libdevice.py:L127-134
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_isnan", core.dtype("int1")),
            (core.dtype("fp16"),): ("__hmf_isnan", core.dtype("int1")),
            (core.dtype("bf16"),): ("__hmf_isnan", core.dtype("int1")),
        }, is_pure=True, _builder=_builder)


@core.builtin
@math._check_dtype(dtypes=["bf16", "fp16", "fp32"])
@math._add_math_1arg_docstr("acos")
def acos(arg0: core.tensor, _builder: ir.builder):  # SOURCE: third_party/ascend/language/cann/libdevice.py:L215-273
    if triton_enable_libdevice_simt() and is_compile_on_910_95:
        if arg0.dtype == core.dtype("bf16"):
            core.static_print("extern livdevice.acos for dtype bf16 is unspported for now.")
            core.static_assert(False)
        return core.extern_elementwise(
            "", "", [arg0], {
                (core.dtype("fp16"),): ("__hmf_acos_fp16", core.dtype("fp16")),
                (core.dtype("fp32"),): ("__hmf_acos_fp32", core.dtype("fp32")),
            }, is_pure=True, _builder=_builder)
    else:
        pi = 3.1415926536
        pi_half = 1.5707963268
        sqrt2 = 1.4142135624
        eps = 1e-8

        # |x| < 0.5, acos(x) = pi/2 - [x + x*x²*(0.1666667 + x²*(0.075 + x²*(0.0446429 + 0.0303810*x²))]
        arg0 = semantic.to_tensor(arg0, _builder)
        abs_x = math.abs(arg0, _builder=_builder)
        dtype = arg0.dtype
        arg0_2 = semantic.mul(arg0, arg0, True, _builder)
        arg0_4 = semantic.mul(arg0_2, arg0_2, True, _builder)
        arg0_6 = semantic.mul(arg0_4, arg0_2, True, _builder)
        arg0_8 = semantic.mul(arg0_6, arg0_2, True, _builder)
        arg0_10 = semantic.mul(arg0_8, arg0_2, True, _builder)
        poly = semantic.add(1.0, semantic.mul(0.166667, arg0_2, True, _builder), True, _builder)
        poly = semantic.add(poly, semantic.mul(0.075, arg0_4, True, _builder), True, _builder)
        poly = semantic.add(poly, semantic.mul(0.044643, arg0_6, True, _builder), True, _builder)
        poly = semantic.add(poly, semantic.mul(0.030380, arg0_8, True, _builder), True, _builder)
        poly = semantic.add(poly, semantic.mul(0.022372, arg0_10, True, _builder), True, _builder)
        acos_center = semantic.sub(pi_half, semantic.mul(arg0, poly, True, _builder), True, _builder)

        # 0.5<|x|<0.9, acos(x) = 2*arctan(t), t=sqrt((1-abs_x)/(1+abs_x))
        numerator_mid = semantic.sub(1.0, abs_x, True, _builder)
        denom_mid = semantic.add(1.0, abs_x, True, _builder)
        div_mid = semantic.truediv(numerator_mid, denom_mid, _builder)
        t_mid = math.sqrt(div_mid, _builder=_builder)
        t2_mid = semantic.mul(t_mid, t_mid, True, _builder)
        t4_mid = semantic.mul(t2_mid, t2_mid, True, _builder)
        t6_mid = semantic.mul(t4_mid, t2_mid, True, _builder)

        poly_mid1 = semantic.mul(0.1065976, t2_mid, True, _builder)
        poly_mid2 = semantic.add(-0.1420890, poly_mid1, True, _builder)
        poly_mid3 = semantic.mul(poly_mid2, t2_mid, True, _builder)
        poly_mid4 = semantic.add(0.1999341, poly_mid3, True, _builder)
        poly_mid5 = semantic.mul(poly_mid4, t2_mid, True, _builder)
        poly_mid6 = semantic.add(-0.3333310, poly_mid5, True, _builder)
        poly_mid = semantic.add(1.0, semantic.mul(poly_mid6, t2_mid, True, _builder), True, _builder)
        arctan_t = semantic.mul(t_mid, poly_mid, True, _builder)
        acos_mid = semantic.mul(2.0, arctan_t, True, _builder)
        is_neg_mid = semantic.less_than(arg0, 0.0, _builder)
        acos_mid_signed = semantic.where(is_neg_mid, semantic.sub(pi, acos_mid, True, _builder), acos_mid, _builder)

        is_center = semantic.less_than(abs_x, 0.6, _builder)
        res_mid_boundary = semantic.where(is_center, acos_center, acos_mid_signed, _builder)
        return res_mid_boundary
