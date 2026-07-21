# SOURCE: third_party/ascend/language/cann/extension/_utils.py:L36-54
#
# SUBTRACTED: 真实文件顶部还有 custom_op（sync_block_all/set/wait 的分发，服务于
# ch08 的核间同步）与 _is_int_like_elem/_assert_int_like_tuple（未被 mem_ops.py
# 实际调用——mem_ops.py 只 import 了 _convert_elem_to_ir_value 一个符号）。三者与
# 本章"GM↔UB 索引搬运"主线无关，按 dossier 批准的"未被引用的符号"一并删除。

import triton.language.core as tl


def _convert_elem_to_ir_value(builder, elem, require_i64):  # SOURCE: third_party/ascend/language/cann/extension/_utils.py:L36-54(_convert_elem_to_ir_value)
    if isinstance(elem, int):
        elem = tl.constexpr(elem)
    if isinstance(elem, tl.constexpr):
        if require_i64:
            assert -2**63 <= elem.value < 2**63, f"Block pointers only support 64 bit `shape/strides`, " \
                f"got a value {elem.value} which is out of the range"
            return builder.get_int64(elem.value)
        else:
            assert -2**31 <= elem.value < 2**31, f"Block pointers only support 32 bit `offsets/block_shape`, " \
                f"got a value {elem.value} which is out of the range"
            return builder.get_int32(elem.value)
    elif isinstance(elem, tl.tensor):
        if require_i64:
            return builder.create_int_cast(elem.handle, builder.get_int64_ty(), elem.dtype.is_int_signed())
        else:
            return builder.create_int_cast(elem.handle, builder.get_int32_ty(), elem.dtype.is_int_signed())
    else:
        assert False, f"Unsupported element type in shape/strides/offsets: {type(elem)}"
