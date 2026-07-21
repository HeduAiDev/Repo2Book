# SOURCE: third_party/ascend/language/cann/extension/semantic.py
# SUBTRACTED(subtraction_plan.delete 批准): 本章数据流只经过 PIPE、
# create_sync_block_set、create_sync_block_wait 三项；create_address_space/
# sub_vec_id/copy_from_ub_to_l1/copy/fixpipe/debug_barrier 等(原 L44-...)属别章
# (ch05 地址空间/搬运、ch06/ch07 算子)题材，随之连带删除只服务于它们的
# import(triton._C.libtriton 的 ir、triton.language.extra.cann.extension as al、
# triton.extension.buffer.language as bl)。
__all__ = []

import enum

from triton._C.libtriton.ascend import ir as ascend_ir
import triton.language.core as tl

from triton.language import semantic as real_semantic


# SOURCE: third_party/ascend/language/cann/extension/semantic.py:L51-59
class PIPE(enum.Enum):
    PIPE_S = ascend_ir.PIPE.PIPE_S
    PIPE_V = ascend_ir.PIPE.PIPE_V
    PIPE_M = ascend_ir.PIPE.PIPE_M
    PIPE_MTE1 = ascend_ir.PIPE.PIPE_MTE1
    PIPE_MTE2 = ascend_ir.PIPE.PIPE_MTE2
    PIPE_MTE3 = ascend_ir.PIPE.PIPE_MTE3
    PIPE_ALL = ascend_ir.PIPE.PIPE_ALL
    PIPE_FIX = ascend_ir.PIPE.PIPE_FIX


# event_id 三形态(int / constexpr / tensor)统一转成 IR handle，pipe 取 .value 后交给
# builder。
def create_sync_block_set(sender, receiver, event_id, sender_pipe: PIPE, receiver_pipe: PIPE, _builder=None):  # SOURCE: third_party/ascend/language/cann/extension/semantic.py:L62-73
    if isinstance(event_id, int):
        _builder.sync_block_set(sender, receiver,
                                real_semantic.to_tensor(tl.constexpr(event_id), _builder).handle,
                                sender_pipe.value, receiver_pipe.value)
    elif isinstance(event_id, tl.constexpr):
        _builder.sync_block_set(sender, receiver,
                                real_semantic.to_tensor(event_id, _builder).handle,
                                sender_pipe.value, receiver_pipe.value)
    else:
        _builder.sync_block_set(sender, receiver,
                                event_id.handle, sender_pipe.value, receiver_pipe.value)


# 与 create_sync_block_set 逐字同构。
def create_sync_block_wait(sender, receiver, event_id, sender_pipe: PIPE, receiver_pipe: PIPE, _builder=None):  # SOURCE: third_party/ascend/language/cann/extension/semantic.py:L76-87
    if isinstance(event_id, int):
        _builder.sync_block_wait(sender, receiver,
                                 real_semantic.to_tensor(tl.constexpr(event_id), _builder).handle,
                                 sender_pipe.value, receiver_pipe.value)
    elif isinstance(event_id, tl.constexpr):
        _builder.sync_block_wait(sender, receiver,
                                 real_semantic.to_tensor(event_id, _builder).handle,
                                 sender_pipe.value, receiver_pipe.value)
    else:
        _builder.sync_block_wait(sender, receiver,
                                 event_id.handle, sender_pipe.value, receiver_pipe.value)
