# SOURCE: third_party/ascend/language/cann/extension/core.py(节选，见每个符号上的行号)
# SUBTRACTED: 本文件真实还定义 copy_from_ub_to_l1/copy/fixpipe/sync_block_*/
# debug_barrier 等全部 @builtin 算子的完整校验体(ch04 双标记机制已讲 builtin/
# is_builtin 本身，这里原样复用同一实现，不重复展开各算子)、IteratorType
# (Parallel/Broadcast/.../Opaque 共 12 种，与 indexing_map 配套，归 P5)、
# FixpipeDMAMode/FixpipeDualDstMode/FixpipePreQuantMode/FixpipePreReluMode/
# SYNC_IN_VF 等枚举、ascend_address_space 等辅助类型。这些是 hivm 方言其他算子的
# 语义目录，归 ch04(双标记)/ch05(内存层级)/P5(hivm op 语义)，与本章
# 『register_custom_op 的 core/pipe/mode 必填三要素』这条主线无关。这里保留
# register_custom_op 真正用到的三枚举 + int64 逃生类型，以及 builtin/is_builtin
# (custom_op.py 的 `@core.builtin def custom(...)` 直接依赖它，不能一并删掉)。
# 原文件：third_party/ascend/language/cann/extension/core.py（约 368 行）。

import enum
from functools import wraps

TRITON_BUILTIN = "__triton_builtin__"  # SOURCE: third_party/ascend/language/cann/extension/core.py:L66
ASCEND_BUILTIN = "__ascend_builtin__"  # SOURCE: third_party/ascend/language/cann/extension/core.py:L67


def builtin(fn):  # SOURCE: third_party/ascend/language/cann/extension/core.py:L70-85
    """Mark a function as a buffer language builtin."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):  # SOURCE: third_party/ascend/language/cann/extension/core.py:L75-83
        if "_builder" not in kwargs or kwargs["_builder"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_builder` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    setattr(wrapper, TRITON_BUILTIN, True)
    setattr(wrapper, ASCEND_BUILTIN, True)
    return wrapper


def is_builtin(fn):  # SOURCE: third_party/ascend/language/cann/extension/core.py:L88-90
    """Is this a registered ascend language builtin function?"""
    return getattr(fn, ASCEND_BUILTIN, False)


# 真实枚举成员绑的是 C++ 绑定 ascend_ir.CoreType/PIPE/MODE 的值(third_party/ascend/
# AscendNPU-IR 编译产物，host 无昇腾 NPU/CANN 工具链故无法拥有，pin 处
# `VECTOR = ascend_ir.CoreType.VECTOR` 这类绑定在本仓不可复现)。
# SUBTRACTED: 用同名字符串顶替 C++ 绑定值——register_custom_op 只要求 core/pipe/mode
# "isinstance(..., core.CORE)"与相等比较成立，`_make_attrs`/测试也只按值做相等断言
# (`op.core.value`/`C.CORE.CUBE.value`)，从不依赖该值本身的具体类型或属性；用字符串
# 即可让成员"存在且互不相同"，故不为此另造一个 pin 没有的包装类。
class CORE(enum.Enum):  # SOURCE: third_party/ascend/language/cann/extension/core.py:L104-109
    VECTOR = "VECTOR"
    CUBE = "CUBE"
    CUBE_OR_VECTOR = "CUBE_OR_VECTOR"
    CUBE_AND_VECTOR = "CUBE_AND_VECTOR"


class PIPE(enum.Enum):  # SOURCE: third_party/ascend/language/cann/extension/core.py:L111-119
    PIPE_S = "PIPE_S"
    PIPE_V = "PIPE_V"
    PIPE_M = "PIPE_M"
    PIPE_MTE1 = "PIPE_MTE1"
    PIPE_MTE2 = "PIPE_MTE2"
    PIPE_MTE3 = "PIPE_MTE3"
    PIPE_ALL = "PIPE_ALL"
    PIPE_FIX = "PIPE_FIX"


class MODE(enum.Enum):  # SOURCE: third_party/ascend/language/cann/extension/core.py:L122-125
    SIMD = "SIMD"
    SIMT = "SIMT"
    MIX = "MIX"


# SOURCE: third_party/ascend/language/cann/extension/core.py:L93-101
class int64(int):
    """
    For custom op, python int argument will be converted to int32 by default,
    if a device-side int64 is required, you can pass an al.int64(x) to it.
    """

    def __new__(cls, value):  # SOURCE: third_party/ascend/language/cann/extension/core.py:L98-101
        import triton.language.core as tl
        obj = int.__new__(cls, value)
        obj.type = tl.int64
        return obj
