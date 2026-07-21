# SOURCE: third_party/ascend/language/cann/extension/_utils.py:L18-33
# SUBTRACTED: 本文件真实还定义 custom_op(sync_block_all/set/wait 的 IR 分发，
# third_party/ascend/language/cann/extension/_utils.py:L5-15)与
# _convert_elem_to_ir_value(block 指针 shape/strides/offsets 到 IR 值的转换，
# :L36-54)。二者分别归"块间同步"与"块指针"章节，与本章 register_custom_op 的
# _index_select 样例校验(只用到下面两个整数类值判定辅助函数)无关。

import triton.language.core as tl


def _is_int_like_elem(x) -> bool:  # SOURCE: third_party/ascend/language/cann/extension/_utils.py:L18-28
    """Accept int / tl.constexpr(int) / tl.tensor(int*)."""
    if isinstance(x, int):
        return True
    if isinstance(x, tl.constexpr):
        return isinstance(x.value, int)
    if isinstance(x, tl.tensor):
        return x.dtype.is_int()
    return False


def _assert_int_like_tuple(name, xs):  # SOURCE: third_party/ascend/language/cann/extension/_utils.py:L31-33
    assert isinstance(xs, (tuple, list)), f"{name} should be a tuple/list, but got {type(xs)}"
    assert all(_is_int_like_elem(x) for x in xs), f"{name} should be integer"
