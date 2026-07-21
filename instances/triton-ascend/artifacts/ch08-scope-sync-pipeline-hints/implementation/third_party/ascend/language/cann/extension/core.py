# SOURCE: third_party/ascend/language/cann/extension/core.py
# SUBTRACTED(subtraction_plan.delete 批准): IteratorType、四个 Fixpipe* 枚举与
# fixpipe、ascend_address_space_base/group、copy、copy_from_ub_to_l1、sub_vec_id、
# sub_vec_num、int64、SYNC_IN_VF 与 debug_barrier——分别属于 ch05(地址空间/搬运)与
# ch06/ch07(算子)的题材，与 scope/sync_block/PIPE/compile_hint 的控制流不相交；删除后
# create_sync_block 链路仍完整。随之连带删除只服务于这些成员的 import
# (triton.language.core as tl、triton.extension.buffer.language as bl、
# triton.backends.ascend.driver.NPUUtils、triton._C.libtriton 的 ir)与 __all__ 里
# 对应的导出名——builtin/is_builtin(@builtin 装饰器本体，且是 ch04 路由的回指)必须留。

__all__ = [
    "builtin",
    "CORE",
    "is_builtin",
    "MODE",
    "PIPE",
    "sync_block_all",
    "sync_block_set",
    "sync_block_wait",
]

import enum
from functools import wraps
from typing import TypeVar

from triton._C.libtriton.ascend import ir as ascend_ir
from triton.language.core import _constexpr_to_value

from . import semantic as semantic
PIPE = semantic.PIPE  # SOURCE: L61 —— 见下方 class PIPE 的覆盖说明(design_decisions M17)


T = TypeVar("T")

TRITON_BUILTIN = "__triton_builtin__"
ASCEND_BUILTIN = "__ascend_builtin__"


def builtin(fn: T) -> T:  # SOURCE: third_party/ascend/language/cann/extension/core.py:L71-85
    """Mark a function as a buffer language builtin."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):  # SOURCE: third_party/ascend/language/cann/extension/core.py:L75-79
        if "_builder" not in kwargs or kwargs["_builder"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_builder` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    # also set triton_builtin to true so that CodeGenerator will recognize this function
    setattr(wrapper, TRITON_BUILTIN, True)
    setattr(wrapper, ASCEND_BUILTIN, True)

    return wrapper


def is_builtin(fn) -> bool:  # SOURCE: third_party/ascend/language/cann/extension/core.py:L88-90
    """Is this a registered ascend language builtin function?"""
    return getattr(fn, ASCEND_BUILTIN, False)


class CORE(enum.Enum):  # SOURCE: third_party/ascend/language/cann/extension/core.py:L104-108
    VECTOR = ascend_ir.CoreType.VECTOR
    CUBE = ascend_ir.CoreType.CUBE
    CUBE_OR_VECTOR = ascend_ir.CoreType.CUBE_OR_VECTOR
    CUBE_AND_VECTOR = ascend_ir.CoreType.CUBE_AND_VECTOR


# 覆盖上面 `PIPE = semantic.PIPE`(L61)那次绑定：模块对外导出的是这个自定义
# enum.Enum 子类，不是 semantic.PIPE(design_decisions M17：两者成员同名同值，但是
# 两个不同的 Python 类，isinstance 互不成立)。
class PIPE(enum.Enum):  # SOURCE: third_party/ascend/language/cann/extension/core.py:L111-119
    PIPE_S = ascend_ir.PIPE.PIPE_S
    PIPE_V = ascend_ir.PIPE.PIPE_V
    PIPE_M = ascend_ir.PIPE.PIPE_M
    PIPE_MTE1 = ascend_ir.PIPE.PIPE_MTE1
    PIPE_MTE2 = ascend_ir.PIPE.PIPE_MTE2
    PIPE_MTE3 = ascend_ir.PIPE.PIPE_MTE3
    PIPE_ALL = ascend_ir.PIPE.PIPE_ALL
    PIPE_FIX = ascend_ir.PIPE.PIPE_FIX


class MODE(enum.Enum):  # SOURCE: third_party/ascend/language/cann/extension/core.py:L122-125
    SIMD = ascend_ir.MODE.SIMD
    SIMT = ascend_ir.MODE.SIMT
    MIX = ascend_ir.MODE.MIX


# 新代核间同步的公共前端：四条校验 + 两个 pipe 都缺省时的配对 + PIPE 类型检查。
def create_sync_block(sender, receiver, event_id, is_set: bool,  # SOURCE: third_party/ascend/language/cann/extension/core.py:L202-221
                      sender_pipe=None, receiver_pipe=None,
                      _builder=None):
    sender = _constexpr_to_value(sender)
    receiver = _constexpr_to_value(receiver)
    assert isinstance(sender, str) and (sender == "cube" or sender == "vector"), f"ERROR: sender = {sender}, only supports cube/vector"
    assert isinstance(receiver, str) and (receiver == "cube" or receiver == "vector"), f"ERROR: receiver = {receiver}, only supports cube/vector"
    if isinstance(event_id, int):
        assert (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    if sender == receiver:
        raise ValueError(f'Unexpected pair: {sender} -> {receiver}, only supports cube -> vector or vector -> cube')
    if sender_pipe is None and receiver_pipe is None:
        if sender == "cube":
            sender_pipe = PIPE.PIPE_FIX
            receiver_pipe = PIPE.PIPE_MTE2
        if sender == "vector":
            sender_pipe = PIPE.PIPE_MTE3
            receiver_pipe = PIPE.PIPE_MTE2
    if not isinstance(sender_pipe, PIPE) or not isinstance(receiver_pipe, PIPE):
        raise TypeError("sender_pipe and receiver_pipe must be instances of PIPE enum")
    if is_set:
        return semantic.create_sync_block_set(sender, receiver, event_id, sender_pipe, receiver_pipe, _builder)
    return semantic.create_sync_block_wait(sender, receiver, event_id, sender_pipe, receiver_pipe, _builder)


@builtin
def sync_block_set(sender, receiver, event_id, sender_pipe=None, receiver_pipe=None, _builder=None):  # SOURCE: L229-230
    return create_sync_block(sender, receiver, event_id, True, sender_pipe, receiver_pipe, _builder)


@builtin
def sync_block_wait(sender, receiver, event_id, sender_pipe=None, receiver_pipe=None, _builder=None):  # SOURCE: L233-234
    return create_sync_block(sender, receiver, event_id, False, sender_pipe, receiver_pipe, _builder)


@builtin
def sync_block_all(mode, event_id, _builder=None):  # SOURCE: L237-244
    mode = _constexpr_to_value(mode)
    event_id = _constexpr_to_value(event_id)
    assert isinstance(mode, str), f"mode: {mode} is not string"
    assert isinstance(event_id, int) and (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    assert mode in ("all_cube", "all_vector", "all", "all_sub_vector"), f"ERROR: mode = {mode}, only supports all_cube/all_vector/all/all_sub_vector"
    _builder.sync_block_all(mode, event_id)
