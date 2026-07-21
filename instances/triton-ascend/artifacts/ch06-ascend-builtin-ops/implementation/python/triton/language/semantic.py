# 基座（非 ascend）语义层的最小子集——mem_ops.py 的 real_semantic.{reshape,cast,full,
# to_tensor} 与 vec_ops.py 的 semantic.{to_tensor,bitcast,not_equal,wrap_tensor,
# _str_to_rounding_mode} 都转发到这里。本文件属于基座 Triton、未被昇腾 fork 改动
# （只有 vec_ops.py 自己的 ascend_cast_impl 是 fork 新增的对照版本，见该文件），故
# 按 ch04/ch05 的一贯处理方式只留本章实际调用到的函数，且每个函数只保留会被本章输入
# 触达的分支（大量 fp8/指针/跨秩广播分支未被裁剪进来，因为本章从不构造这类输入）。
#
# SOURCE: python/triton/language/semantic.py（节选，见每个符号上的行号）

from typing import List, Optional

import triton.language.core as tl
from triton._C.libtriton import ir


# SOURCE: python/triton/language/semantic.py:L1760-1766
def wrap_tensor(x, scalar_ty, ret_shape):
    if ret_shape:
        res_ty = tl.block_type(scalar_ty, ret_shape)
    else:
        res_ty = scalar_ty
    return tl.tensor(x, res_ty)


# SOURCE: python/triton/language/semantic.py:L671-686
def full(shape: List[int], value, dtype, builder: ir.builder) -> tl.tensor:
    if isinstance(value, tl.tensor):
        assert value.numel.value == 1, "only accepts size-1 tensor"
        value = cast(value, dtype, builder)
    else:
        if dtype is None:
            raise ValueError("dtype must be specified when value is not a tensor")
        if value == 0:
            value = builder.get_null_value(dtype.to_ir(builder))
        else:
            get_value_fn = getattr(builder, f"get_{dtype.name}")
            value = get_value_fn(value)
        value = tl.tensor(value, dtype)
    return splat(value, shape, builder)


# SOURCE: python/triton/language/semantic.py:L693-697
def splat(value: tl.tensor, shape: List[int], builder: ir.builder) -> tl.tensor:
    assert not value.type.is_block(), "Cannot splat a block tensor"
    if len(shape) == 0:
        return value
    ret_ty = tl.block_type(value.dtype, shape)
    return tl.tensor(builder.create_splat(value.handle, shape), ret_ty)


# SOURCE: python/triton/language/semantic.py:L112-146(节选)
# SUBTRACTED: 真实实现还处理 float/int 的字面量位宽推导表(int128/uint64 边界等)；
# 本章只会喂 Python bool/int/tl.constexpr/tl.tensor 四种，float 字面量分支未触达。
def to_tensor(x, builder, check_type: bool = True):  # SOURCE: python/triton/language/semantic.py:L112-146(节选)
    if isinstance(x, bool):
        return tl.tensor(builder.get_int1(x), tl.int1)
    elif isinstance(x, int):
        if -2**31 <= x < 2**31:
            dtype = tl.int32
        elif -2**63 <= x < 2**63:
            dtype = tl.int64
        else:
            raise ValueError(f'Nonrepresentable integer {x}.')
        return full((), x, dtype=dtype, builder=builder)
    elif isinstance(x, tl.constexpr):
        return to_tensor(x.value, builder)
    elif isinstance(x, tl.tensor):
        return x
    if check_type:
        raise TypeError(f"cannot convert {x} of type {type(x)} to tensor")
    return x


# SOURCE: python/triton/language/semantic.py:L702-709
def reshape(input: tl.tensor, dst_shape: List[int], can_reorder: bool, builder: ir.builder) -> tl.tensor:
    numel = 1
    for s in dst_shape:
        numel *= s
    if input.type.numel != numel:
        raise ValueError("reshape() cannot change total number of elements in tensor")
    ret_ty = tl.block_type(input.type.scalar, dst_shape)
    return tl.tensor(builder.create_reshape(input.handle, dst_shape, can_reorder), ret_ty)


# SOURCE: python/triton/language/semantic.py:L778-793(节选)
# SUBTRACTED: 跨秩(rank 不同)广播分支——本章 not_equal 的两个操作数要么同 shape，
# 要么一个是标量、一个是 block，秩不同的情形未触达。
def broadcast_impl_value(lhs: tl.tensor, rhs: tl.tensor, builder: ir.builder):  # SOURCE: python/triton/language/semantic.py:L778-793(节选)
    lhs_ty, rhs_ty = lhs.type, rhs.type
    if lhs_ty.is_block() and not rhs_ty.is_block():
        rhs_ty = tl.block_type(rhs_ty.scalar, lhs_ty.get_block_shapes())
        rhs = tl.tensor(builder.create_splat(rhs.handle, lhs_ty.get_block_shapes()), rhs_ty)
    elif not lhs_ty.is_block() and rhs_ty.is_block():
        lhs_ty = tl.block_type(lhs_ty.scalar, rhs_ty.get_block_shapes())
        lhs = tl.tensor(builder.create_splat(lhs.handle, rhs_ty.get_block_shapes()), lhs_ty)
    return lhs, rhs


# SOURCE: python/triton/language/semantic.py:L555-559(节选，_bool_like)
def _bool_like(v: tl.tensor):
    if not v.type.is_block():
        return tl.int1
    shape = v.type.get_block_shapes()
    return tl.block_type(tl.int1, shape)


# SOURCE: python/triton/language/semantic.py:L634-643
# SUBTRACTED: 真实实现先经 binary_op_type_checking_impl 做隐式类型提升(int8 vs int32
# 之类)——本章两处调用点(ascend_cast_impl 的 bool 分支)的两个操作数在传入前已经是
# 同 dtype(input.dtype 构造出的 null 值)，只需要广播（标量 vs block），不需要提升。
def not_equal(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:  # SOURCE: python/triton/language/semantic.py:L634-643
    input, other = broadcast_impl_value(input, other, builder)
    scalar_ty = input.type.scalar
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fcmpUNE(input.handle, other.handle), _bool_like(input))
    elif scalar_ty.is_int():
        return tl.tensor(builder.create_icmpNE(input.handle, other.handle), _bool_like(input))
    raise TypeError(f"unexpected type {scalar_ty}")


# SOURCE: python/triton/language/semantic.py:L413-425(节选:bitwise_op_type_checking_impl)
# + L438-440(xor_)
# SUBTRACTED: 真实 bitwise_op_type_checking_impl 会在两操作数整型位宽不同的分支上做
# integer_promote_impl 提升——flip 的 SIMT 回退路径里 `y.__xor__(xor_sum(y, ...))`
# 两个操作数都来自同一个 reshape 之后的 y，dtype 恒相同，提升分支未触达。
def xor_(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:  # SOURCE: python/triton/language/semantic.py:L413-425+L438-440
    input, other = broadcast_impl_value(input, other, builder)
    input_sca_ty = input.type.scalar
    other_sca_ty = other.type.scalar
    if not input_sca_ty.is_int() or not other_sca_ty.is_int():
        raise TypeError(f"xor_ expects integer operands, got {input_sca_ty} and {other_sca_ty}")
    return tl.tensor(builder.create_xor(input.handle, other.handle), input.type)


# SOURCE: python/triton/language/semantic.py:L854-861
def _str_to_rounding_mode(rounding_mode: Optional[str]):
    if rounding_mode is None:
        return None
    if rounding_mode == 'rtne':
        return ir.ROUNDING_MODE.RTNE
    if rounding_mode == 'rtz':
        return ir.ROUNDING_MODE.RTZ
    raise ValueError(f"Invalid rounding mode: {rounding_mode}. Supported rounding modes are 'rtne' and 'rtz'.")


# SOURCE: python/triton/language/semantic.py:L864-880
def bitcast(input: tl.tensor, dst_ty, builder: ir.builder) -> tl.tensor:
    src_ty = input.type
    if src_ty.is_block():
        dst_ty = tl.block_type(dst_ty.scalar, input.type.get_block_shapes())
    if src_ty == dst_ty:
        return input
    src_sca_ty = src_ty.scalar
    dst_sca_ty = dst_ty.scalar
    if src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
        return cast(input, dst_ty, builder)
    src_bits = src_sca_ty.primitive_bitwidth
    dst_bits = dst_sca_ty.primitive_bitwidth
    if src_bits != dst_bits:
        raise ValueError("Cannot bitcast data-type of size " + str(src_bits) + " to "
                         "data-type of size " + str(dst_bits))
    return tl.tensor(builder.create_bitcast(input.handle, dst_ty.to_ir(builder)), dst_ty)


# SOURCE: python/triton/language/semantic.py:L883-950(节选)
# SUBTRACTED: fp8 定制类型分支(builder.codegen_fns["convert_custom_types"])与
# 指针<->整数/指针<->指针分支——本章两处调用点(mem_ops 的 gather `other` 值转换、
# tensor.to() 供 flip SIMT 分支做位重解释)都只在 fp16/bf16/fp32/int* 之间转换，
# 从不构造 fp8 或指针输入。
def cast(input: tl.tensor, dst_ty, builder: ir.builder,  # SOURCE: python/triton/language/semantic.py:L883-950(节选)
         fp_downcast_rounding: Optional[str] = None, overflow_mode: Optional[str] = None) -> tl.tensor:
    src_ty = input.type
    if isinstance(dst_ty, tl.constexpr):
        dst_ty = dst_ty.value
    if isinstance(fp_downcast_rounding, tl.constexpr):
        fp_downcast_rounding = fp_downcast_rounding.value
    if src_ty.is_block():
        dst_ty = tl.block_type(dst_ty.scalar, input.type.get_block_shapes())
    if src_ty == dst_ty:
        return input

    src_sca_ty = src_ty.scalar
    dst_sca_ty = dst_ty.scalar
    if src_sca_ty == dst_sca_ty:
        return input

    fp_downcast_rounding = _str_to_rounding_mode(fp_downcast_rounding)
    if dst_sca_ty.is_floating() and src_sca_ty.is_floating() and \
       dst_sca_ty.primitive_bitwidth < src_sca_ty.primitive_bitwidth:
        if fp_downcast_rounding is None:
            fp_downcast_rounding = ir.ROUNDING_MODE.RTNE
    elif fp_downcast_rounding is not None:
        raise ValueError("fp_downcast_rounding should be set only for truncating fp conversions. "
                         "Source scalar type is " + str(src_sca_ty) + " and destination type is " + str(dst_sca_ty))

    # bf16 <=> (not fp32)：先绕道 fp32
    if (src_sca_ty.is_fp16() and not dst_sca_ty.is_fp32()) or \
       (src_sca_ty.is_bf16() and not dst_sca_ty.is_fp32()):
        return cast(cast(input, tl.float32, builder), dst_sca_ty, builder)

    truncate_fp = src_sca_ty.is_floating() and dst_sca_ty.is_floating() and \
        src_sca_ty.primitive_bitwidth > dst_sca_ty.primitive_bitwidth
    if truncate_fp:
        return tl.tensor(builder.create_fp_trunc(input.handle, dst_ty.to_ir(builder)), dst_ty)

    ext_fp = src_sca_ty.is_floating() and dst_sca_ty.is_floating() and \
        src_sca_ty.primitive_bitwidth < dst_sca_ty.primitive_bitwidth
    if ext_fp:
        return tl.tensor(builder.create_fp_ext(input.handle, dst_ty.to_ir(builder)), dst_ty)

    if src_sca_ty.is_int() and dst_sca_ty.is_int() and \
       (src_sca_ty.int_bitwidth != dst_sca_ty.int_bitwidth or src_sca_ty.int_signedness != dst_sca_ty.int_signedness):
        sign_extend = src_sca_ty.is_int_signed() and not src_sca_ty.is_bool()
        if dst_sca_ty.is_bool():
            from . import semantic as _self
            ty = input.dtype.to_ir(builder)
            _0 = tl.tensor(builder.get_null_value(ty), input.dtype)
            return not_equal(input, _0, builder)
        return tl.tensor(builder.create_int_cast(input.handle, dst_ty.to_ir(builder), sign_extend), dst_ty)

    if src_sca_ty.is_standard_floating() and dst_sca_ty.is_int():
        if dst_sca_ty.is_bool():
            ty = input.dtype.to_ir(builder)
            _0 = tl.tensor(builder.get_null_value(ty), input.dtype)
            return not_equal(input, _0, builder)
        elif dst_sca_ty.is_int_signed():
            return tl.tensor(builder.create_fp_to_si(input.handle, dst_ty.to_ir(builder)), dst_ty)
        else:
            return tl.tensor(builder.create_fp_to_ui(input.handle, dst_ty.to_ir(builder)), dst_ty)

    if src_sca_ty.is_int() and dst_sca_ty.is_standard_floating():
        if src_sca_ty.is_bool() or not src_sca_ty.is_int_signed():
            return tl.tensor(builder.create_ui_to_fp(input.handle, dst_ty.to_ir(builder)), dst_ty)
        else:
            return tl.tensor(builder.create_si_to_fp(input.handle, dst_ty.to_ir(builder)), dst_ty)

    assert False, f'cannot cast {input} to {dst_ty}'
