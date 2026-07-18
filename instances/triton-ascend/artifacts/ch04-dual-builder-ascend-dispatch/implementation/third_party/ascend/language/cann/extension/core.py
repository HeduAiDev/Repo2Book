# fork 增量：Ascend 内建算子的注册面 —— 本章只留『双标记』这一件事的证据。
#
# SUBTRACTED（subtraction_plan.delete 批准）：本文件真实还定义 copy_from_ub_to_l1/copy/
# fixpipe/sync_block_set/sync_block_wait/sync_block_all/debug_barrier/sub_vec_num 等
# 全部 @builtin 算子的完整校验体（对齐检查、pipe 推断……），以及 CORE/PIPE/MODE/
# IteratorType/FixpipeDMAMode/FixpipeDualDstMode/FixpipePreQuantMode/FixpipePreReluMode/
# SYNC_IN_VF 等枚举类的全部成员、int64/ascend_address_space 等辅助类型。这些是 hivm
# 方言算子的语义目录，归 Part 5（hivm op 语义）与后续 UB/GM 搬运章，与本章『双 builder
# 分发路由』这条主线无关，删除不影响本章要证明的机制。这里只保留一个代表性例子
# sub_vec_id 展示 @builtin 怎么打双标记；原文件：
# third_party/ascend/language/cann/extension/core.py（约 368 行）。
from functools import wraps
from typing import TypeVar

T = TypeVar("T")

# SOURCE: third_party/ascend/language/cann/extension/core.py:L66-67
TRITON_BUILTIN = "__triton_builtin__"
ASCEND_BUILTIN = "__ascend_builtin__"


# SOURCE: third_party/ascend/language/cann/extension/core.py:L70-85
def builtin(fn: T) -> T:
    """Mark a function as a buffer language builtin."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        # SOURCE: third_party/ascend/language/cann/extension/core.py:L74-79
        if "_builder" not in kwargs or kwargs["_builder"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_builder` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    # also set triton_builtin to true so that CodeGenerator will recognize this function
    setattr(wrapper, TRITON_BUILTIN, True)
    setattr(wrapper, ASCEND_BUILTIN, True)

    return wrapper


# SOURCE: third_party/ascend/language/cann/extension/core.py:L88-90
def is_builtin(fn) -> bool:
    """Is this a registered ascend language builtin function?"""
    return getattr(fn, ASCEND_BUILTIN, False)


# SOURCE: third_party/ascend/language/cann/extension/core.py:L166-171
# SUBTRACTED: 真实实现委托 semantic.sub_vec_id(builder) -> tl.tensor(builder.
# create_get_sub_vec_id(), tl.int64)（把 C++ 侧的 get-sub-vec-id 算子包成一个
# tensor）。tensor/dtype 的构造细节属另一套机制（tl.* 语言层，非本章双 builder
# 分发主线），这里直接返回 builder 的 emit 结果，只为演示 @builtin 打双标记与
# ascend_builder 路由这两件事。原：third_party/.../core.py:L166-171 +
# third_party/.../semantic.py:L90-91。
@builtin
def sub_vec_id(_builder=None):
    """Get the Vector Core index on the AI Core."""
    # SOURCE: third_party/ascend/language/cann/extension/core.py:L166-171
    return _builder.create_get_sub_vec_id()
