# SOURCE: third_party/ascend/language/cann/extension/mem_ops.py
#
# 本章主线：GM 侧只有裸指针(third_party/ascend/ascend_ir.cc:L412-417 只导出 5 个
# address space，GM/Zero 不进 Python)，索引先用普通 tl.load 从 GM 载进 UB 变成
# 整型 tile，这四个 builtin 就是"指针世界 <-> buffer 世界"接缝——GM 侧的参数永远是
# 裸指针（src/ptr），UB 侧的索引/值永远是普通 tensor。
#
# SUBTRACTED: 四个函数的长 docstring 只留一行摘要 + 参数含义，原 Constraints/
# Example 段落（官方用例）删除——纯文档、不参与控制流，正文会把用例单独内嵌讲解。

import numbers
import triton.language as tl
from triton.language import semantic as real_semantic
from triton.language.core import (
    _constexpr_to_value,
    _tensor_member_fn,
    builtin,
    constexpr,
    tensor,
)
from triton.language.semantic import wrap_tensor

from typing import Optional, Tuple, List, Union
from triton._C.libtriton import ir

from ._utils import _convert_elem_to_ir_value


@_tensor_member_fn
@builtin
def index_put(  # SOURCE: third_party/ascend/language/cann/extension/mem_ops.py:L40-177(index_put)
    ptr: tensor,
    index: tensor,
    value: tensor,
    dim: int,
    index_boundary: int,
    end_offset: tuple,
    start_offset: tuple,
    dst_stride: tuple,
    _builder=None
):
    """
    Index put values from a tensor into a destination tensor (GM), scattering
    `value` (in UB) at positions given by `index` (in UB) along `dim`.

    :param ptr: pointer type, the destination tensor pointer (in GM)
    :param index: tensor, a index to scatter (in UB)
    :param value: tensor, a value to store (in UB)
    :param dim: int32, the dimension to scatter along
    :param index_boundary: int64, the upper boundary for index values
    :param end_offset: tuple of int, the offsets of each dimension for the end of the scatter region
    :param start_offset: tuple of int, the offsets of each dimension for the start of the scatter region
    :param dst_stride: tuple of int, the stride of each dimension of destination tensor
    """

    def index_put_impl(  # SOURCE: third_party/ascend/language/cann/extension/mem_ops.py:L124-177(index_put_impl)
        ptr: tl.tensor,
        index: tl.tensor,
        value: tl.tensor,
        dim: int,
        index_boundary: int,
        end_offset: Tuple,
        start_offset: Tuple,
        dst_stride: Tuple,
        _builder: ir.builder
    ):
        assert index.dtype.is_int(), "index must be an integer tensor"
        if not ptr.dtype.element_ty.is_floating():
            raise ValueError(f"Expected dtype fp16/fp32/bf16, but got {ptr.dtype.element_ty}")
        if not isinstance(dim, int):
            raise ValueError("dim must be of type tl.constexpr")

        v_rank = len(value.shape)
        idx_rank = len(index.shape)
        if v_rank < 2 or v_rank > 5:
            raise ValueError(f"value rank must be in [2, 5], got value rank={v_rank}")
        if dim < 0 or dim >= v_rank - 1:
            raise ValueError(f"dim must satisfy 0<=dim<value.rank-1 ({v_rank-1}), got dim={dim}")

        if idx_rank != 1:
            # flatten index to 1D, shape (index.numel,)
            flat_numel = index.numel
            index = real_semantic.reshape(index, (flat_numel,), True, _builder)
            idx_rank = 1

        if value.shape[dim] != index.shape[0]:
            raise ValueError(
                f"index.numel must equal value.shape[dim], "
                f"but got index.numel={index.numel.value}, value.shape[dim]={value.shape[dim].value}"
            )

        # 位宽契约：index_put 用 `require_i64 = index.dtype.is_int64()` 一个开关
        # 同时决定 end_offset/start_offset/dst_stride 三者的位宽——index 为 int32
        # 时 dst_stride 也会退成 i32。这与 gather_out_to_ub/scatter_ub_to_out 的
        # "stride 恒 i64、offset 恒 i32"两套硬编码写法不一致（见 m6/open_questions，
        # 本精简版逐字保留这处不一致，不替它统一）。
        require_i64 = index.dtype.is_int64()
        end_offset = [_convert_elem_to_ir_value(_builder, elem, require_i64) for elem in end_offset]
        start_offset = [_convert_elem_to_ir_value(_builder, elem, require_i64) for elem in start_offset]
        dst_stride = [_convert_elem_to_ir_value(_builder, elem, require_i64) for elem in dst_stride]

        if len(end_offset) != v_rank or len(start_offset) != v_rank or len(dst_stride) != v_rank:
            raise ValueError(f"len(end_offset)==len(start_offset)==len(dst_stride)==value.rank required, "
                            f"got {len(end_offset)}, {len(start_offset)}, {len(dst_stride)}, {v_rank}")

        return tl.tensor(_builder.create_index_put(ptr.handle, index.handle, value.handle, dim,
                                                   index_boundary, end_offset, start_offset, dst_stride), tl.void)

    dim = _constexpr_to_value(dim)
    index_boundary = _constexpr_to_value(index_boundary)

    return index_put_impl(ptr, index, value, dim, index_boundary,
                          end_offset, start_offset, dst_stride, _builder)


@_tensor_member_fn
@builtin
def gather_out_to_ub(  # SOURCE: third_party/ascend/language/cann/extension/mem_ops.py:L180-329(gather_out_to_ub)
    src: tensor,
    index: tensor,
    index_boundary: int,
    dim: int,
    src_stride: tuple,
    end_offset: tuple,
    start_offset: tuple,
    other=None,
    _builder=None
):
    """
    Gather from a source tensor in Global Memory (GM) to Unified Buffer (UB)
    along a specified dimension with out-of-bound handling.

    :param src: pointer type, the source tensor pointer (in GM)
    :param index: tensor, a tensor to gather (in UB)
    :param index_boundary: int64, the upper boundary for index values
    :param dim: int32, the dimension to gather along
    :param src_stride: tuple of int64, the stride of each dimension of src tensor
    :param end_offset: tuple of int32, the end offsets of each dimension for index tensor
    :param start_offset: tuple of int32, the start offsets of each dimension for index tensor
    :param other(Optional): scalar value, the default value when index is out of boundary (in UB)
    :return: tensor, with the same shape as `index.shape` (in UB)
    """

    def gather_out_to_ub_impl(  # SOURCE: third_party/ascend/language/cann/extension/mem_ops.py:L274-329(gather_out_to_ub_impl)
        src: tl.tensor,
        index: tl.tensor,
        index_boundary: int,
        dim: int,
        src_stride: Tuple,
        end_offset: Tuple,
        start_offset: Tuple,
        other: Optional[numbers.Number] = None,
        _builder: ir.builder = None
    ):
        assert index.dtype.is_int(), "index must be an integer tensor"
        if not src.dtype.element_ty.is_floating():
            raise ValueError(f"Expected dtype fp16/fp32/bf16, but got {src.dtype.element_ty}")

        if not isinstance(index_boundary, int):
            raise ValueError("index_boundary must be of type tl.constexpr")
        if not isinstance(dim, int):
            raise ValueError("dim must be of type tl.constexpr")

        idx_rank = len(index.shape)
        if idx_rank < 1 or idx_rank > 5:
            raise ValueError(f"index rank must be in [1, 5], got rank={idx_rank}")
        if dim < 0 or dim >= idx_rank:
            raise ValueError(f"dim must satisfy 0<=dim<index.rank ({idx_rank}), got dim={dim}")

        if other is not None:
            other = real_semantic.cast(other, src.dtype.element_ty, _builder)

        # src stride 恒 i64；end/start offset 恒 i32——与 index_put 的单开关写法不同
        # 的另一套硬编码（见上方 index_put_impl 的注释与 m6）。
        src_stride = [_convert_elem_to_ir_value(_builder, elem, True) for elem in src_stride]
        end_offset = [_convert_elem_to_ir_value(_builder, elem, False) for elem in end_offset]
        start_offset = [_convert_elem_to_ir_value(_builder, elem, False) for elem in start_offset]

        if len(src_stride) != idx_rank or len(end_offset) != idx_rank or len(start_offset) != idx_rank:
            raise ValueError(f"len(src_stride)==len(end_offset)==len(start_offset)==index.rank required, "
                            f"got {len(src_stride)}, {len(end_offset)}, {len(start_offset)}, {idx_rank}")

        ret = _builder.create_gather_out_to_ub(
            src.handle,
            index.handle,
            index_boundary,
            dim,
            src_stride,
            end_offset,
            start_offset,
            other if other else None
        )
        ret_shape = [s.value if isinstance(s, constexpr) else s for s in index.shape]
        return wrap_tensor(ret, src.dtype.element_ty, ret_shape)

    dim = _constexpr_to_value(dim)
    index_boundary = _constexpr_to_value(index_boundary)
    return gather_out_to_ub_impl(src, index, index_boundary, dim,
                                 src_stride, end_offset, start_offset, other, _builder)


@_tensor_member_fn
@builtin
def scatter_ub_to_out(  # SOURCE: third_party/ascend/language/cann/extension/mem_ops.py:L332-482(scatter_ub_to_out)
    ptr: tensor,
    value: tensor,
    index: tensor,
    index_boundary: int,
    dim: int,
    dst_stride: tuple,
    end_offset: tuple,
    start_offset: tuple,
    _builder=None
):
    """
    Scatter a tile from Unified Buffer (UB) into a destination tensor in Global Memory (GM)
    along a specified dimension, with index-boundary checking.

    :param ptr: pointer type, the destination tensor pointer (in GM)
    :param value: tensor, a tile value to store (in UB)
    :param index: tensor, a index to scatter (in UB)
    :param index_boundary: int64, the upper boundary for index values
    :param dim: int32, the dimension to scatter along
    :param dst_stride: tuple of int64, the stride of each dimension of destination tensor
    :param end_offset: tuple of int32, the end offsets of each dimension for index tensor
    :param start_offset: tuple of int32, the start offsets of each dimension for index tensor
    """

    def scatter_ub_to_out_impl(ptr: tl.tensor,  # SOURCE: third_party/ascend/language/cann/extension/mem_ops.py:L422-469(scatter_ub_to_out_impl)
        value: tl.tensor,
        index: tl.tensor,
        index_boundary: int,
        dim: int,
        dst_stride: tuple,
        end_offset: tuple,
        start_offset: tuple,
        _builder=None
    ):
        assert index.dtype.is_int(), "index must be an integer tensor"
        if not ptr.dtype.element_ty.is_floating():
            raise ValueError(f"Expected dtype fp16/fp32/bf16, but got {ptr.dtype.element_ty}")

        if not isinstance(index_boundary, int):
            raise ValueError("index_boundary must be of type tl.constexpr")
        if not isinstance(dim, int):
            raise ValueError("dim must be of type tl.constexpr")

        idx_rank = len(index.shape)
        if idx_rank < 1 or idx_rank > 5:
            raise ValueError(f"index rank must be in [1, 5], got rank={idx_rank}")
        if dim < 0 or dim >= idx_rank:
            raise ValueError(f"dim must satisfy 0<=dim<index.rank (index.rank={idx_rank}), got dim={dim}")

        # dst stride 恒 i64；end/start offset 恒 i32——与 gather_out_to_ub 同一套写法，
        # 两者互为对称操作，唯独 index_put 用了不同的单开关写法（见上，m6）。
        dst_stride = [_convert_elem_to_ir_value(_builder, elem, True) for elem in dst_stride]
        end_offset = [_convert_elem_to_ir_value(_builder, elem, False) for elem in end_offset]
        start_offset = [_convert_elem_to_ir_value(_builder, elem, False) for elem in start_offset]

        if len(dst_stride) != idx_rank or len(end_offset) != idx_rank or len(start_offset) != idx_rank:
            raise ValueError(f"len(dst_stride)==len(end_offset)==len(start_offset)==index.rank required, "
                            f"got {len(dst_stride)}, {len(end_offset)}, {len(start_offset)}, {idx_rank}")

        return tl.tensor(
            _builder.create_scatter_ub_to_out(
                ptr.handle,
                value.handle,
                index.handle,
                index_boundary,
                dim,
                dst_stride,
                end_offset,
                start_offset
            ),
            tl.void
        )

    def _is_ranked_tensor(x):  # SOURCE: third_party/ascend/language/cann/extension/mem_ops.py:L471-472(_is_ranked_tensor)
        return isinstance(x, tensor) and x.shape and len(x.shape) > 0

    dim = _constexpr_to_value(dim)
    index_boundary = _constexpr_to_value(index_boundary)
    value = _constexpr_to_value(value)

    if not _is_ranked_tensor(value) or isinstance(value, constexpr):
        element_ty = ptr.type.scalar.element_ty
        value = real_semantic.full(index.shape, value, element_ty, _builder)
    return scatter_ub_to_out_impl(ptr, value, index, index_boundary, dim,
                                  dst_stride, end_offset, start_offset, _builder)


@_tensor_member_fn
@builtin
def index_select_simd(  # SOURCE: third_party/ascend/language/cann/extension/mem_ops.py:L485-636(index_select_simd)
    src,
    dim,
    index,
    src_shape,
    src_offset,
    read_shape,
    _builder=None
) -> tensor:
    """
    Parallel index_select operation from Global Memory to Unified Buffer (SIMD version).

    Selects data from multiple indices along a specified dimension and loads
    them as tiles from GM directly to UB with zero-copy semantics.

    :param src: Source tensor pointer (in GM)
    :param dim: The dimension along which to select indices
    :param index: 1D tensor of indices to select (in UB)
    :param src_shape: Complete shape of the source tensor (can be int or tensor)
    :param src_offset: Starting offset for reading (can be int or tensor)
    :param read_shape: Size to read (tile shape, can be int or tensor)

    Constraints (unlike the other three mem_ops, this one takes NO index_boundary
    and does NOT check if `index` contains out-of-bounds values — that is the
    price paid for zero-copy tile selection):

    - ``read_shape[dim]`` must be ``-1``
    - ``src_offset[dim]`` can be ``-1`` (will be ignored)
    - Boundary handling: ``src_offset + read_shape > src_shape`` automatically
      truncates to ``src_shape`` boundary
    - Does not check if ``index`` contains out-of-bounds values
    """

    def index_select_simd_impl(  # SOURCE: third_party/ascend/language/cann/extension/mem_ops.py:L563-612(index_select_simd_impl)
        src: tl.tensor,
        dim: int,
        index: tl.tensor,
        src_shape: List[Union[int, tl.tensor]],
        src_offset: List[Union[int, tl.tensor]],
        read_shape: List[Union[int, tl.tensor]],
        _builder: ir.builder
    ) -> tl.tensor:
        # Validate inputs
        ndim = len(src_shape)
        assert len(src_offset) == ndim, \
            f"src_offset length {len(src_offset)} must match src_shape length {ndim}"
        assert len(read_shape) == ndim, \
            f"read_shape length {len(read_shape)} must match src_shape length {ndim}"
        assert 0 <= dim < ndim, \
            f"dim={dim} must be in range [0, {ndim})"
        assert len(index.shape) == 1, \
            f"index must be 1D tensor, got {len(index.shape)}D"
        # 源码未给理由（见 open_questions）："尾轴必须留给连续段" 只是推测。
        assert dim < ndim - 1, \
            f"index_select_simd cannot support trailing dimension as dim={dim}, ndim={ndim}"
        # Handle both tensor and int offsets (for interpreter mode)
        newsrc_shape = []
        for s in src_shape:
            if isinstance(s, tensor):
                newsrc_shape.append(s.handle)
            elif isinstance(s, int):
                newsrc_shape.append(s)
            else:
                newsrc_shape.append(s.handle if hasattr(s, 'handle') else s)
        newsrc_offset = []
        for s in src_offset:
            if isinstance(s, tensor):
                newsrc_offset.append(s.handle)
            elif isinstance(s, int):
                newsrc_offset.append(s)
            else:
                newsrc_offset.append(s.handle if hasattr(s, 'handle') else s)

        # Create output type
        return_shape = [
            index.shape[0] if i == dim else read_shape[i]
            for i in range(ndim)
        ]
        element_ty = src.type.element_ty
        output_ty = tl.block_type(element_ty, return_shape)
        out = _builder.create_index_select_simd(src.handle, index.handle, dim, newsrc_shape, newsrc_offset, read_shape, return_shape)
        return tl.tensor(out, output_ty)

    # SUBTRACTED: process_param（third_party/ascend/language/cann/extension/mem_ops.py:
    # L617-622）是从未被调用的死代码——定义后没有任何调用点，删除不改变任何执行路径。

    dim = _constexpr_to_value(dim)

    newsrc_shape = [
        real_semantic.to_tensor(o, _builder) if isinstance(o, constexpr) else o
        for o in src_shape
    ]
    newsrc_offset = [
        real_semantic.to_tensor(o, _builder) if isinstance(o, constexpr) else o
        for o in src_offset
    ]
    assert len(index.shape) == 1, "index must be a 1D tensor"

    return index_select_simd_impl(
        src, dim, index, newsrc_shape, newsrc_offset, read_shape, _builder
    )
