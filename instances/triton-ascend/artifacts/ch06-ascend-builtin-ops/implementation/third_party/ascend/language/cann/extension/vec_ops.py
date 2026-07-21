# SOURCE: third_party/ascend/language/cann/extension/vec_ops.py
#
# 片上向量算子词汇表：insert_slice/extract_slice（互逆的片上切片对）、get_element
# （tile -> 标量）、flip（SIMD 单算子 vs SIMT log2 步 xor-swap 回退两条路径）、sort
# （只允许末维 + int8/int16 自动挂饱和提示）、cast（overflow_mode 校验 + compile_hint
# 挂载，昇腾定制的 ascend_cast_impl 决策树就住在这里，没有单独文件）。

import triton.language as tl
from triton.language import semantic, core, standard
from triton.language.core import (
    _constexpr_to_value,
    _unwrap_if_constexpr,
    _tensor_member_fn,
    builtin,
    constexpr,
    dtype,
    tensor,
)
from triton.language.semantic import (
    wrap_tensor,
    _str_to_rounding_mode,
    not_equal,
    bitcast,
    to_tensor,
)

from . import is_compile_on_910_95
from .aux_ops import compile_hint_impl

from typing import Optional, List
from triton._C.libtriton import ir


@_tensor_member_fn
@builtin
def insert_slice(ful, sub, offsets, sizes, strides, _builder=None, _generator=None) -> tensor:  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L47-92(insert_slice)
    """Insert a tensor to another tensor at the given offsets/sizes/strides."""

    def insert_slice_impl(ful: tensor, sub: tensor, offsets: List[tensor], sizes: List[int], strides: List[int], builder: ir.builder) -> tensor:  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L61-81(insert_slice_impl)
        assert(len(ful.shape) == len(offsets))
        assert(len(ful.shape) == len(sizes))
        assert(len(ful.shape) == len(strides))
        assert(all([s>=1 for s in sizes]))
        assert(all([s>=0 for s in strides]))
        # Handle both tensor and int offsets (for interpreter mode)
        new_offsets = []
        for o in offsets:
            if isinstance(o, tensor):
                new_offsets.append(o.handle)
            elif isinstance(o, int):
                new_offsets.append(o)
            else:
                new_offsets.append(o.handle if hasattr(o, 'handle') else o)
        ret_type = tl.block_type(ful.type.scalar, ful.shape)
        out = builder.create_insert_slice(ful.handle, sub.handle, new_offsets, sizes, strides)
        return tensor(out, ret_type)

    assert len(ful.shape) > 0
    assert len(ful.shape) == len(sub.shape)
    new_offsets = [
        semantic.to_tensor(o, _builder) if isinstance(o, constexpr) else o
        for o in offsets
    ]
    out = insert_slice_impl(ful, sub, new_offsets, sizes, strides, _builder)
    return out


@_tensor_member_fn
@builtin
def extract_slice(ful, offsets, sizes, strides, _builder=None, _generator=None) -> tensor:  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L94-137(extract_slice)
    """Extract a tensor from another tensor at the given offsets/sizes/strides
    (逐行对称于 insert_slice，唯一区别是返回类型用 sizes 而不是 ful.shape——两者是
    一对互逆算子)。"""

    def extract_slice_impl(ful: tensor, offsets: List[tensor], sizes: List[int], strides: List[int], builder: ir.builder) -> tensor:  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L111-131(extract_slice_impl)
        assert(len(ful.shape) == len(offsets))
        assert(len(ful.shape) == len(sizes))
        assert(len(ful.shape) == len(strides))
        assert(all([s>=1 for s in sizes]))
        assert(all([s>=0 for s in strides]))
        new_offsets = []
        for o in offsets:
            if isinstance(o, tensor):
                new_offsets.append(o.handle)
            elif isinstance(o, int):
                new_offsets.append(o)
            else:
                new_offsets.append(o.handle if hasattr(o, 'handle') else o)
        ret_type = tl.block_type(ful.type.scalar, sizes)
        out = builder.create_extract_slice(ful.handle, new_offsets, sizes, strides)
        return tensor(out, ret_type)

    assert len(ful.shape) > 0
    new_offsets = [
        semantic.to_tensor(o, _builder) if isinstance(o, constexpr) else o
        for o in offsets
    ]
    sub = extract_slice_impl(ful, new_offsets, sizes, strides, _builder)
    return sub


@_tensor_member_fn
@builtin
def get_element(src, indice, _builder=None, _generator=None):  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L139-177(get_element)
    """Read one element out of a ranked tensor by integer indices (indice rank
    must match src rank), lowering to `create_extract_scalar`."""

    def get_element_impl(src: tensor, indice: List[tensor], builder: ir.builder):  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L153-172(get_element_impl)
        if len(src.shape) != len(indice):
            raise ValueError("Indice's rank must be equal to src tensor's rank")

        new_indice = []
        for i in indice:
            if isinstance(i, tensor):
                new_indice.append(i.handle)
            elif isinstance(i, int):
                new_indice.append(i)
            else:
                new_indice.append(i.handle if hasattr(i, 'handle') else i)

        result = builder.create_extract_scalar(src.handle, new_indice)
        return wrap_tensor(result, src.type.scalar, None)

    assert len(src.shape) > 0
    new_indice = [
        semantic.to_tensor(i, _builder) if isinstance(i, constexpr) else i
        for i in indice
    ]
    return get_element_impl(src, new_indice, _builder)


@builtin
def flip(ptr, dim=-1, _builder=None, _generator=None):  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L179-269(flip)

    def flip_impl(ptr: tensor, dim: int, builder: ir.builder, generator=None):  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L182-268(flip_impl)
        """
        Flips a tensor `ptr` along the dimension `dim`.
        """

        def _get_flip_dim(dim, shape):  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L194-200(_get_flip_dim)
            dim = _unwrap_if_constexpr(dim)
            shape = _unwrap_if_constexpr(shape)
            if dim is None:
                dim = len(shape) - 1
            if dim < 0:  # flip doesn't work if dim < 0 because the xor-swap for loop will start/end at the wrong index
                dim += len(shape)
            return constexpr(dim)

        def _log2(i: "core.constexpr"):  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L202-208(_log2)
            log2 = 0
            n = core.constexpr(i).value
            while n > 1:
                n >>= 1
                log2 += 1
            return core.constexpr(log2)

        # SUBTRACTED: 真实 flip_simd 在 shape 为空/rank 未知时还有一段防御式兜底
        # （从 ptr.type.shape 再取一次、rank 仍未知时改走"unknown rank"分支，见
        # third_party/ascend/language/cann/extension/vec_ops.py:L223-233）。本章
        # 精简版只喂 shape 已知的 ranked tensor，兜底分支不会进入；主路径(归一化
        # dim -> create_flip)完整保留。
        def flip_simd(ptr: tensor, dim: int, builder: ir.builder):  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L211-251(flip_simd，节选)
            """Triton flip operation for simd: 一条 ascend.flip 算子搞定。"""
            shape = ptr.shape
            rank = len(shape)
            if rank < 1:
                raise ValueError("ascend.flip requires tensor rank >= 1")
            norm_dim = dim if dim >= 0 else dim + rank
            if not (0 <= norm_dim < rank):
                raise ValueError(
                    f"ascend.flip got invalid dim={dim} for shape {tuple(shape)}"
                )
            dim = norm_dim

            flipped_vals = builder.create_flip(ptr.handle, dim)
            flipped = tensor(flipped_vals, type=ptr.type)
            return flipped

        # If compile_mode is not simt, use the simd implementation
        if not builder.is_simt_mode():
            return flip_simd(ptr, dim, builder)
        core.static_assert(-len(ptr.shape) <= dim and dim < len(ptr.shape), _builder=builder)
        _dim: core.constexpr = _get_flip_dim(dim, ptr.shape)
        core.static_assert(standard._is_power_of_two(ptr.shape[_dim]), _builder=builder)
        steps: core.constexpr = _log2(ptr.shape[_dim])
        # If steps is 0, return the original tensor
        if steps == 0:
            return ptr
        # reshape the swap dimension to (2, 2, ..., 2)
        idtype = core.get_int_dtype(bitwidth=ptr.dtype.primitive_bitwidth, signed=True)
        y = core.reshape(ptr.to(idtype, bitcast=True, _builder=builder), ptr.shape.__getitem__(slice(None, _dim)) + [2] * steps + ptr.shape.__getitem__(slice(_dim + 1, None)), _builder=builder)
        for i in static_range(steps):
            y = y.__xor__(standard.xor_sum(y, _dim + i, True, _builder=builder, _generator=generator), _builder=builder)
        ptr = core.reshape(y, ptr.shape, _builder=builder).to(ptr.dtype, bitcast=True, _builder=builder)
        return ptr

    try:
        dim = int(dim.value) if hasattr(dim, "value") else int(dim)
    except Exception as e:
        raise TypeError(f"dim must be an integer (or tl.constexpr int), got {dim!r}") from e

    dim = len(ptr.shape) - 1 if dim == -1 else dim
    return flip_impl(ptr, dim, _builder, _generator)


class static_range:  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L280-313(static_range)
    """
    Iterator for non-JIT Python functions that need to iterate over constexpr
    values (used by flip's SIMT xor-swap loop).
    """
    def __init__(self, arg1, arg2=None, step=None):  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L285-291(static_range.__init__)
        if step is None:
            self.step = core.constexpr(1)
        else:
            self.step = step
        if arg2 is None:
            self.start = core.constexpr(0)
            self.end = arg1
        else:
            self.start = arg1
            self.end = arg2

    def __iter__(self):  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L293-299(static_range.__iter__)
        start_val = core._constexpr_to_value(self.start)
        end_val = core._constexpr_to_value(self.end)
        step_val = core._constexpr_to_value(self.step)
        self._current = start_val
        self._end = end_val
        self._step = step_val
        return self

    def __next__(self):  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L301-306(static_range.__next__)
        if self._current >= self._end:
            raise StopIteration
        value = self._current
        self._current += self._step
        return value


@builtin
def sort(ptr, dim=-1, descending=False, _builder=None):  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L316-397(sort)
    """Sort the input tensor along `dim` (only the last dimension is allowed);
    int8/int16 results automatically get an `overflow_mode=saturate` compile hint."""

    def sort_impl(ptr: tensor, dim: int, descending, builder: ir.builder):  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L330-366(sort_impl，节选)
        allowed_types = {tl.int8, tl.int16, tl.bfloat16, tl.float16, tl.float32, tl.int32, tl.int64, tl.float8e4nv, tl.float8e5}
        base_ty = ptr.type.scalar if hasattr(ptr.type, "scalar") else ptr.type
        if base_ty not in allowed_types:
            raise TypeError(
                f"ascend.sort only supports int8, int16, bfloat16, float16, float32, int32, int64, float8e4nv, float8e5"
                f"but got {ptr.type}"
            )

        # SUBTRACTED: 真实 sort_impl 在 shape 未知（rank is None）时还有一段兜底
        # （third_party/ascend/language/cann/extension/vec_ops.py:L361-366），要求
        # dim 显式传 -1。本章精简版只喂 shape 已知的 ranked tensor，该分支不会进入；
        # "只允许末维"这条主路径完整保留。
        shape = ptr.shape
        rank = len(shape)
        if rank < 1:
            raise ValueError("ascend.sort requires tensor rank >= 1")
        last_dim = rank - 1
        norm_dim = dim if dim >= 0 else dim + rank
        if norm_dim != last_dim:
            raise ValueError(
                f"ascend.sort only supports sorting along the last dimension "
                f"(dim={last_dim} or -1) for shape {tuple(shape)}, but got dim={dim}"
            )
        dim = last_dim

        if hasattr(descending, "value"):
            descending = bool(descending.value)
        else:
            descending = bool(descending)

        sorted_vals = builder.create_sort(ptr.handle, dim, descending)
        values = tensor(sorted_vals, type=ptr.type)
        return values

    try:
        dim = int(dim.value) if hasattr(dim, "value") else int(dim)
    except Exception as e:
        raise TypeError(f"dim must be an integer (or tl.constexpr int), got {dim!r}. Error: {str(e)}") from e

    if hasattr(descending, "value"):
        descending = bool(descending.value)
    else:
        descending = bool(descending)

    ret = sort_impl(ptr, dim, descending, _builder)
    # interpreter mode not support compile_hint overflow_mode, direct return
    from triton.runtime.interpreter import InterpreterBuilder
    if isinstance(_builder, InterpreterBuilder):
        return ret
    base_ty = ptr.type.scalar if hasattr(ptr.type, "scalar") else ptr.type
    if base_ty.is_int8() or base_ty.is_int16():
        compile_hint_impl(ret, "overflow_mode", constexpr("saturate"), _builder)
    return ret


# SUBTRACTED: ascend_cast_impl 里的 fp8e4b15/convert_custom_types 分支
# （third_party/ascend/language/cann/extension/vec_ops.py:L433-436）与指针相关分支
# （L506-520）——本章精简版只覆盖 float/int 之间的数值转换输入，不构造指针 cast 或
# fp8e4b15 输入，这些分支不会被触达。"拒绝 fp8/fp64（非 910_95 芯片）"这条守卫子句
# 本身不在删除范围内，逐字保留。
def ascend_cast_impl(input: tensor, dst_ty: dtype, builder: ir.builder,  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L400-522(ascend_cast_impl，节选)
         fp_downcast_rounding: Optional[str] = None, overflow_mode: Optional[str] = None) -> tensor:
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

    # For fp downcasting default rounding mode should be RTNE, for all other conversions it should
    # not be set
    fp_downcast_rounding = _str_to_rounding_mode(fp_downcast_rounding)
    use_custom_rounding = False
    if dst_sca_ty.is_floating() and src_sca_ty.is_floating(
    ) and dst_sca_ty.primitive_bitwidth < src_sca_ty.primitive_bitwidth:
        if fp_downcast_rounding is None: fp_downcast_rounding = ir.ROUNDING_MODE.RTNE
        elif fp_downcast_rounding != ir.ROUNDING_MODE.RTNE: use_custom_rounding = True
    else:
        if fp_downcast_rounding is not None:
            raise ValueError("fp_downcast_rounding should be set only for truncating fp conversions. "
                             "Source scalar type is " + str(src_sca_ty) + " and destination type is " + str(dst_sca_ty))
    if not is_compile_on_910_95:
       if (src_sca_ty.is_fp8() or dst_sca_ty.is_fp8()) or (src_sca_ty.is_fp64() or dst_sca_ty.is_fp64()):
            raise ValueError("[fp8, fp64] is unsupported on Ascend for now."
                           "Source scalar type is " + str(src_sca_ty) + " and destination type is " + str(dst_sca_ty))

    # Casting with customized floating types involved: fp8 <=> bf16, fp16, fp32, fp64
    # and non-default rounding modes for downcasting
    if (src_sca_ty.is_fp8() and dst_sca_ty.is_floating()) or \
       (src_sca_ty.is_floating() and dst_sca_ty.is_fp8()) or \
       use_custom_rounding:
        return tensor(builder.create_fp_to_fp(input.handle, dst_ty.to_ir(builder), fp_downcast_rounding), dst_ty)

    # bf16 <=> (not fp32)
    if (src_sca_ty.is_fp16() and not dst_sca_ty.is_fp32()) or \
       (src_sca_ty.is_bf16() and not dst_sca_ty.is_fp32()):
        return ascend_cast_impl(ascend_cast_impl(input, tl.float32, builder), dst_sca_ty, builder)

    # Standard floating types' casting: truncation
    truncate_fp = src_sca_ty.is_floating() and \
        dst_sca_ty.is_floating() and \
        src_sca_ty.primitive_bitwidth > dst_sca_ty.primitive_bitwidth
    if truncate_fp:
        return tensor(builder.create_fp_trunc(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Standard floating types' casting: extension
    ext_fp = src_sca_ty.is_floating() and \
        dst_sca_ty.is_floating() and \
        src_sca_ty.primitive_bitwidth < dst_sca_ty.primitive_bitwidth
    if ext_fp:
        return tensor(builder.create_fp_ext(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Casting between integer types
    if src_sca_ty.is_int() and dst_sca_ty.is_int() and \
       (src_sca_ty.int_bitwidth != dst_sca_ty.int_bitwidth or src_sca_ty.int_signedness != dst_sca_ty.int_signedness):
        sign_extend = src_sca_ty.is_int_signed() and not src_sca_ty.is_bool()
        if dst_sca_ty.is_bool():
            ty = input.dtype.to_ir(builder)
            _0 = tensor(builder.get_null_value(ty), input.dtype)
            return not_equal(input, _0, builder)
        elif overflow_mode == "saturate" and \
             (src_sca_ty.is_int_unsigned() or dst_sca_ty.is_int_unsigned()) and \
             src_sca_ty.int_bitwidth >= dst_sca_ty.int_bitwidth:
            if is_compile_on_910_95:
                result = tensor(builder.create_int_cast(input.handle, dst_ty.to_ir(builder), sign_extend), dst_ty)
                compile_hint_impl(result, "saturate_src_unsigned", src_sca_ty.is_int_unsigned(), builder)
                compile_hint_impl(result, "saturate_dst_unsigned", dst_sca_ty.is_int_unsigned(), builder)
                return result
            else:
                return ascend_cast_impl(ascend_cast_impl(input, tl.float32, builder), dst_sca_ty, builder)
        return tensor(builder.create_int_cast(input.handle, dst_ty.to_ir(builder), sign_extend), dst_ty)

    # Casting standard floating types to integer types
    if src_sca_ty.is_standard_floating() and dst_sca_ty.is_int():
        if dst_sca_ty.is_bool():
            ty = input.dtype.to_ir(builder)
            _0 = tensor(builder.get_null_value(ty), input.dtype)
            return not_equal(input, _0, builder)
        elif dst_sca_ty.is_int_signed():
            return tensor(builder.create_fp_to_si(input.handle, dst_ty.to_ir(builder)), dst_ty)
        else:
            return tensor(builder.create_fp_to_ui(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Casting integer types to standard floating types
    if src_sca_ty.is_int() and dst_sca_ty.is_standard_floating():
        if src_sca_ty.is_bool() or not src_sca_ty.is_int_signed():
            return tensor(builder.create_ui_to_fp(input.handle, dst_ty.to_ir(builder)), dst_ty)
        else:
            return tensor(builder.create_si_to_fp(input.handle, dst_ty.to_ir(builder)), dst_ty)

    assert False, f'cannot cast {input} to {dst_ty}'


@_tensor_member_fn
@builtin
def cast(input, dtype: dtype, fp_downcast_rounding: Optional[str] = None, bitcast: bool = False, overflow_mode: Optional[str] = None, _builder=None):  # SOURCE: third_party/ascend/language/cann/extension/vec_ops.py:L524-562(cast)
    """
    Casts a tensor to the given :code:`dtype`.

    :param dtype: The target data type.
    :type dtype: dtype
    :param fp_downcast_rounding: The rounding mode for downcasting
        floating-point values. This parameter is only used when self is a
        floating-point tensor and dtype is a floating-point type with a
        smaller bitwidth. Supported values are :code:`"rtne"` (round to
        nearest, ties to even) and :code:`"rtz"` (round towards zero).
    :type fp_downcast_rounding: str, optional
    :param bitcast: If true, the tensor is bitcasted to the given
        :code:`dtype`, instead of being numerically casted.
    :type bitcast: bool, optional
    :param overflow_mode: When overflow_mode is not set or is "trunc",
        truncation (cut-off) will be used to handle overflow. When
        overflow_mode is "sautrate", the maximum value of the data type
        will be used to handle overflow.
    :type overflow_mode: string, optional
    """
    overflow_modes = ["trunc", "saturate"]
    input = semantic.to_tensor(input, _builder)
    if isinstance(bitcast, constexpr):
        bitcast = bitcast.value
    if bitcast:
        return semantic.bitcast(input, dtype, _builder)
    ret = ascend_cast_impl(input, dtype, _builder, fp_downcast_rounding, overflow_mode)
    if overflow_mode is not None:
        if overflow_mode in overflow_modes:
            from triton.runtime.interpreter import InterpreterBuilder
            if isinstance(_builder, InterpreterBuilder):
                overflow_mode = constexpr(overflow_mode)
            compile_hint_impl(ret, "overflow_mode", overflow_mode, _builder)
        else:
            raise ValueError(f"Unknown overflow_mode:{overflow_mode} is found.")
    return ret
