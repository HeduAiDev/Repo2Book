# 支撑层——triton.language.semantic 的标量算子实现，属基座 Triton、未被昇腾 fork
# 改动。本章保留的 libdevice.acos(唯一"纯 IR 组合逼近"代表)在无 __hmf_ 符号的分支里
# 逐行调用这些函数拼多项式，math_ops.isfinited 也经它比较 tensor；但这层"隐式类型
# 提升/广播/指针运算"本身不是本章要讲的机制(register_custom_op/libdevice 三分野)。
#
# SOURCE: python/triton/language/semantic.py(节选，见每个符号上的行号)
# SUBTRACTED: 真实文件还有 computation_type_impl(隐式类型提升表)、
# binary_op_sanitize_overflow_impl(整数环绕检查)、check_ptr_type_impl(指针合法性)、
# broadcast_impl_value(block 广播)等约 40 个辅助函数/分支，服务"不同 dtype/shape的
# 两个操作数如何对齐"。本章保留的 acos 多项式全程只在标量 fp16/fp32/bf16 上运算、
# 调用点两操作数已同 dtype，从不触发提升/广播/指针路径，故这里的 `_binary` 只做
# "都转成 tensor"这一步，不做类型提升。原：python/triton/language/semantic.py
# (完整文件约 1900 行，本章只圈定 acos 用到的这 8 个函数)。

import triton.language.core as tl


def to_tensor(x, builder, check_type=True):  # SOURCE: python/triton/language/semantic.py:L112-141(节选)
    if isinstance(x, tl.tensor):
        return x
    if isinstance(x, bool):
        return tl.tensor(builder.get_int1(x), tl.int1)
    if isinstance(x, tl.constexpr):
        return to_tensor(x.value, builder)
    # SUBTRACTED: int 按取值范围分派 int32/uint32/int64/uint64 的分支(L117-125)——
    # 本章样例(acos 的多项式系数、0/1 等边界常量)只喂 float 字面量，未触达整数分支。
    if isinstance(x, (int, float)):
        return tl.tensor(builder.get_fp32(float(x)), tl.float32)
    if check_type:
        raise TypeError(f"cannot convert {x} of type {type(x)} to tensor")
    return x


def _binary(input, other, builder):  # SOURCE: binary_op_type_checking_impl 节选，python/triton/language/semantic.py:L164-192
    # SUBTRACTED: 隐式类型提升(computation_type_impl)、指针合法性检查
    # (check_ptr_type_impl)、隐式广播(broadcast_impl_value)——本章两操作数在调用点
    # 已同 dtype(标量 fp16/fp32/bf16)，故直接 to_tensor 后返回。
    return to_tensor(input, builder), to_tensor(other, builder)


def add(input, other, sanitize_overflow, builder):  # SOURCE: python/triton/language/semantic.py:L219-234(节选)
    input, other = _binary(input, other, builder)
    # SUBTRACTED: 指针/整数分支(is_ptr()/is_int()，含 sanitize_overflow 环绕检查)——
    # acos 全程浮点运算，恒进 is_floating() 分支。
    return tl.tensor(builder.create_fadd(input.handle, other.handle), input.type)


def sub(input, other, sanitize_overflow, builder):  # SOURCE: python/triton/language/semantic.py:L246-260(节选)
    input, other = _binary(input, other, builder)
    return tl.tensor(builder.create_fsub(input.handle, other.handle), input.type)


def mul(input, other, sanitize_overflow, builder):  # SOURCE: python/triton/language/semantic.py:L264-276(节选)
    input, other = _binary(input, other, builder)
    return tl.tensor(builder.create_fmul(input.handle, other.handle), input.type)


def truediv(input, other, builder):  # SOURCE: python/triton/language/semantic.py:L279-301(节选)
    input, other = _binary(input, other, builder)
    # SUBTRACTED: int/float 混合的隐式 cast 分支(L285-297)——本章样例双侧恒 fp32。
    return tl.tensor(builder.create_fdiv(input.handle, other.handle), input.type)


def less_than(input, other, builder):  # SOURCE: python/triton/language/semantic.py:L592-604(节选)
    input, other = _binary(input, other, builder)
    return tl.tensor(builder.create_fcmpOLT(input.handle, other.handle), tl.int1)


def where(condition, x, y, builder):  # SOURCE: python/triton/language/semantic.py:L1738-1751(节选)
    # SUBTRACTED: 非 bool condition 的废弃警告、block 广播(L1739-1748)——本章样例
    # condition 恒为 less_than/greater_equal 产出的标量 int1。
    x, y = _binary(x, y, builder)
    return tl.tensor(builder.create_select(condition.handle, x.handle, y.handle), x.type)
