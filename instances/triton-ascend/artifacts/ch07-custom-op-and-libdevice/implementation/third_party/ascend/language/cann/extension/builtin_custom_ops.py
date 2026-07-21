# SOURCE: third_party/ascend/language/cann/extension/builtin_custom_ops.py
# register_custom_op 的真实内建样例——随包自带的 __builtin_ 前缀算子，免注册表校验、
# 免 symbol/bitcode(见 custom_op._make_attrs 对 __builtin_ 前缀的豁免)。
#
# SUBTRACTED: 真实文件还定义 _index_put/_gather_load/_scatter_store 三个内建算子
# (third_party/ascend/language/cann/extension/builtin_custom_ops.py:L106-220)。四个
# 内建算子结构同构——类字段 name/core/pipe/mode 声明 + __init__ 形状/dtype 断言 +
# self.arg_type 动态定型；保留 _index_select 一个完整样例即展示 register_custom_op
# 的真实用法，其余三个删除不改变任何保留者的控制流。

import triton.language.core as tl
from .custom_op import register_custom_op
from .core import CORE, PIPE, MODE
from ._utils import _is_int_like_elem, _assert_int_like_tuple


@register_custom_op
class _index_select:
    """This operation gathers values from the src GM tensor into the out UB tensor
    at positions with offsets specified by the index UB tensor along the specified
    dimension using a SIMT template. This operation supports 2D-5D.

    Arguments:
    - src: pointer type, the source tensor pointer (in GM)
    - index: tensor, a tensor to gather (in UB)
    - dim: int, the dimension to gather along
    - bound: int, the upper boundary for index
    - end_offset: tuple of int, the end offsets of each dimension for index tensor
    - start_offset: tuple of int, the start offsets of each dimension for src tensor
    - src_stride: tuple of int, the stride of each dimension of src tensor
    - other(Optional): scalar value, the default value when index is out of boundary (in UB)
    - out: the output tensor (in UB)
    """
    # SUBTRACTED: 类 docstring 里 2D~5D 各 rank 的 Reference formula 一节
    # (third_party/ascend/language/cann/extension/builtin_custom_ops.py:L53-73)——
    # 说明算子数学语义，writer 摘 1~2 行即可，不必逐 rank 内嵌。

    # SOURCE: third_party/ascend/language/cann/extension/builtin_custom_ops.py:L74-77
    name = '__builtin_index_select'
    core = CORE.VECTOR
    pipe = PIPE.PIPE_V
    mode = MODE.SIMT

    # SOURCE: third_party/ascend/language/cann/extension/builtin_custom_ops.py:L79-103
    def __init__(self, src, index, dim, bound: tl.int64, end_offset, start_offset, src_stride, other=None, out=None):
        assert src.type.is_ptr() or src.dtype.is_ptr(), f"src should be a pointer, but got {src.type}"
        assert index.dtype.is_int(), "index should be integer tensor"
        src_rank = len(src_stride)
        idx_rank = len(index.shape)
        assert 2 <= src_rank <= 5, f"src rank should in [2, 5], but got {src_rank}"
        assert 1 <= idx_rank <= 2, f"index rank should in [1, 2], but got {idx_rank}"
        assert _is_int_like_elem(dim), "dim should be an integer"
        assert _is_int_like_elem(bound), "bound should be an integer"
        assert 0 <= dim < src_rank, f"dim should in [0, {src_rank - 1}], but got {dim}"
        assert len(start_offset) == len(src_stride), "start_offset and src_stride should have same size"
        assert len(end_offset) == idx_rank + len(start_offset) - 1, \
            "len(end_offset) should be equal to index rank + len(start_offset) - 1"

        _assert_int_like_tuple("end_offset", end_offset)
        _assert_int_like_tuple("start_offset", start_offset)
        _assert_int_like_tuple("src_stride", src_stride)

        assert out, "out is required"
        assert out.dtype == src.dtype.element_ty, "out should have same dtype as src"

        # use index type for end_offset, start_offset and src_stride.
        self.arg_type['end_offset'] = index.dtype
        self.arg_type['start_offset'] = index.dtype
        self.arg_type['src_stride'] = index.dtype
        self.extra_attr = f"src_stride_len={len(src_stride)}"
