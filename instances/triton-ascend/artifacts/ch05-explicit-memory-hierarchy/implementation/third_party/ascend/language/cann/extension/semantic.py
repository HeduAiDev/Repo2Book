# SOURCE: third_party/ascend/language/cann/extension/semantic.py（节选，行号见各
# 符号上方）
#
# SUBTRACTED（对应 core.py 同批次删减，dossier code_spine 本就只圈定 L94-L129 与
# L132-L148 两段）：真实文件开头还有 create_address_space（builder.get_target_
# attribute 的一层转发，实际未被任何地方调用，是死代码）、PIPE 枚举、
# create_sync_block_set/create_sync_block_wait（核间同步）、sub_vec_id、
# debug_barrier。这些与本章「地址空间校验 + copy/fixpipe」主线无关，归 ch08，本文件
# 只留三个函数本体。

from typing import Union

import triton.language.core as tl
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl


# SOURCE: third_party/ascend/language/cann/extension/semantic.py:L94-110
def copy_from_ub_to_l1(src: Union[tl.tensor, bl.buffer], dst: Union[tl.tensor, bl.buffer], builder):
    if not builder.is_910_95():
        raise RuntimeError("this feature is only supported on Ascend910_95")
    if isinstance(src, tl.tensor) or isinstance(dst, tl.tensor):
        raise TypeError("tensor not support yet")
    if src.shape != dst.shape:
        raise TypeError("src and dst must have same shape")
    if src.dtype != dst.dtype:
        raise TypeError("src and dst need to have the same type")
    if isinstance(src, bl.buffer) and isinstance(dst, bl.buffer):
        if src.space != al.ascend_address_space.UB:
            raise TypeError("src's AddressSpace must be UB")
        if dst.space != al.ascend_address_space.L1:
            raise TypeError("dst's AddressSpace must be L1")
        builder.create_copy_buffer(src.handle, dst.handle)
    else:
        raise TypeError("src and dst must be tl.tensor or bl.buffer")


# SOURCE: third_party/ascend/language/cann/extension/semantic.py:L113-129
def copy(src: Union[tl.tensor, bl.buffer], dst: Union[tl.tensor, bl.buffer], builder):
    if not builder.is_910_95():
        raise RuntimeError("this feature is only supported on Ascend910_95")
    if isinstance(src, tl.tensor) or isinstance(dst, tl.tensor):
        raise TypeError("tensor not support yet")
    if src.shape != dst.shape:
        raise TypeError("src and dst must have same shape")
    if src.dtype != dst.dtype:
        raise TypeError("src and dst need to have the same type")
    if isinstance(src, bl.buffer) and isinstance(dst, bl.buffer):
        if src.space != al.ascend_address_space.UB:
            raise TypeError("src's AddressSpace must be UB")
        if dst.space not in (al.ascend_address_space.L1, al.ascend_address_space.UB):
            raise TypeError("dst's AddressSpace must be UB or L1")
        builder.create_copy_buffer(src.handle, dst.handle)
    else:
        raise TypeError("src and dst must be tl.tensor or bl.buffer")


# SOURCE: third_party/ascend/language/cann/extension/semantic.py:L132-148
def fixpipe(
    src: tl.tensor,
    dst,
    dma_mode,
    dual_dst_mode,
    pre_quant_mode,
    pre_relu_mode,
    builder,
) -> None:
    builder.create_fixpipe(
        src.handle,
        dst.handle,
        dma_mode.value,
        dual_dst_mode.value,
        pre_quant_mode.value,
        pre_relu_mode.value,
    )
