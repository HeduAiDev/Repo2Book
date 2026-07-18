# 基座（非 ascend）language 层的最小子集 —— 只留本章需要对照/依赖的几件事：
#   1. TRITON_BUILTIN 标记 + 基座 builtin/is_builtin（与 ascend 侧 al.builtin 对照，双标记机制的另一半）
#   2. constexpr 容器 + _unwrap_if_constexpr / _constexpr_to_value（visit_Call 与 scope 都要用）
#   3. tensor/_value/dtype.SIGNEDNESS 的最小容器子集（供 CodeGenerator 的 SSA 值记录与
#      mangle_ty 的类型名编码使用；本章不涉及张量形状/算术，故大量派生字段与算子重载全略）
#
# 真实源码：python/triton/language/core.py（本仓与上游 triton 共享，未被 ascend fork 改动）。
from enum import Enum
from functools import wraps
from typing import TypeVar

T = TypeVar("T")

# SOURCE: python/triton/language/core.py:L20
TRITON_BUILTIN = "__triton_builtin__"


# SOURCE: python/triton/language/core.py:L25-40
def builtin(fn: T) -> T:
    """Mark a function as a builtin.

    基座版本只打一个标记 __triton_builtin__；对照 ascend 侧
    third_party/ascend/language/cann/extension/core.py 的 @builtin（同时打
    __triton_builtin__ 与 __ascend_builtin__），是本章『双标记』机制的另一半。
    """
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        # SOURCE: python/triton/language/core.py:L29-36
        if "_builder" not in kwargs or kwargs["_builder"] is None:
            print(kwargs)
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_builder` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    setattr(wrapper, TRITON_BUILTIN, True)
    return wrapper


# SOURCE: python/triton/language/core.py:L108-110
def is_builtin(fn) -> bool:
    """Is this a registered triton builtin function?"""
    return getattr(fn, TRITON_BUILTIN, False)


# SOURCE: python/triton/language/core.py:L134-...
# SUBTRACTED: constexpr 真实定义还有 __bool__/__int__/各算术 dunder/__repr__ 等一整套
# Python 运算符重载（供 kernel 里 constexpr 参与算术），与本章『builder 分发』主线无关，
# 这里只留 _unwrap_if_constexpr / _constexpr_to_value 需要的 .value 容器语义。
class constexpr:
    # SOURCE: python/triton/language/core.py:L134-...
    """This class is used to store a value that is known at compile-time."""

    def __init__(self, value):
        # SOURCE: python/triton/language/core.py:L134-...
        self.value = value.value if isinstance(value, constexpr) else value


# SOURCE: python/triton/language/core.py:L270-271
def _unwrap_if_constexpr(o):
    return o.value if isinstance(o, constexpr) else o


# SOURCE: python/triton/language/core.py:L1144-1147
def _constexpr_to_value(v):
    if isinstance(v, constexpr):
        return v.value
    return v


# SOURCE: python/triton/language/core.py:L711-716
class _value:
    """Base class of values that exist in the triton IR (i.e. not constexprs)."""

    def __init__(self, handle):
        # SOURCE: python/triton/language/core.py:L715-716
        self.handle = handle


# SOURCE: python/triton/language/core.py:L724-757
# SUBTRACTED: shape/numel/dtype 派生字段与全部算子重载略去（需要完整 dtype.is_block()/
# .scalar 支持）——本章只把 tensor 当 CodeGenerator 内部 SSA 值容器(handle+type)用，
# 不涉及张量形状运算。
class tensor(_value):
    # SOURCE: python/triton/language/core.py:L724-757
    """Represents an N-dimensional array of values or pointers（本章精简为 handle+type 容器）。"""

    def __init__(self, handle, type):
        # SOURCE: python/triton/language/core.py:L744-754
        super().__init__(handle)
        self.type = type


# SOURCE: python/triton/language/core.py:L288-298
# SUBTRACTED: dtype 真实定义还有 SINT_TYPES/UINT_TYPES/FP_TYPES/KIND 等一整套类型目录与
# is_ptr()/is_int()/is_floating()/is_block()/is_void() 等判定方法；本章的 mangle_ty 演示
# 只需要 SIGNEDNESS 这一枚举（int 的有符号/无符号判据），其余类型判定在测试里用 duck-typed
# 假类型对象提供。
class dtype:
    # SOURCE: python/triton/language/core.py:L288-298
    class SIGNEDNESS(Enum):
        # SOURCE: python/triton/language/core.py:L295-298
        SIGNED = 0
        UNSIGNED = 1
