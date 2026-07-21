# 支撑层——triton.language 的值系统(constexpr/dtype 占位/tensor)与内建标记机制，属基座
# Triton、未被昇腾 fork 改动。本章 scope.py/code_generator.py/aux_ops.py/core.py 都
# `from triton.language.core import ...` 依赖它，但它本身不是本章 dossier 的机制主角
# (scope 的编译器特判 / sync_block 两代协议 / PIPE 口径收窄 / compile_hint 才是)。
#
# SOURCE: python/triton/language/core.py(节选，见每个符号上的行号)
# SUBTRACTED: 真实文件还定义 tensor 的约 80 个运算符重载方法(__add__/__getitem__/...)、
# dtype 的完整类型目录(SINT_TYPES/UINT_TYPES/FP_TYPES/fp8 系列)与 is_ptr()/is_int()/
# is_block() 等约 20 个查询方法、device_print/inline_asm_elementwise 等数十个自由函数、
# extern/extern_elementwise/static_assert 等——这些服务于块级张量运算与类型提升，本章
# 样例(scope 的 SSA 穿线只把 tensor 当 handle+type 容器、sync_block 的参数只有
# str/int/PIPE 枚举、compile_hint 的 hint_val 只按 Python 内建类型 isinstance 判断)从不
# 依赖它们，测试里用 duck-typed 假类型对象提供 `.to_ir(builder)`。
# 原：python/triton/language/core.py(全文件约 3400 行，本文件只留以下符号)。

from functools import wraps
from typing import TypeVar

T = TypeVar("T")

TRITON_BUILTIN = "__triton_builtin__"  # SOURCE: python/triton/language/core.py:L20


# 逐字保留(含调试期遗留的 `print(kwargs)`，"只做减法"不代为清理源码里的既有噪声)。
def builtin(fn: T) -> T:  # SOURCE: python/triton/language/core.py:L25-38
    """Mark a function as a builtin."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):  # SOURCE: python/triton/language/core.py:L29-35
        if "_builder" not in kwargs or kwargs["_builder"] is None:
            print(kwargs)
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_builder` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    setattr(wrapper, TRITON_BUILTIN, True)

    return wrapper


# SUBTRACTED: 真实 _tensor_member_fn(python/triton/language/core.py:L42-64)还会检查
# 被装饰函数的签名(inspect.signature)算出 has_args，并把"也可当 x.foo(...) 调"的说明
# 追加进 docstring；本章 aux_ops.py 的 sync_block_all/set/wait 只被当自由函数调用
# (从不写 x.sync_block_set(...))，故这里只保留装饰器"透传不改变函数"这一半行为。
def _tensor_member_fn(fn: T) -> T:  # SOURCE: python/triton/language/core.py:L42-45(节选)
    return fn


# SUBTRACTED: 真实 constexpr(python/triton/language/core.py:L134-259)还定义 __index__
# 与约 20 个算术/比较 dunder 方法；本章样例(_constexpr_to_value/_unwrap_if_constexpr
# 的 unwrap、parallel 的 arg 透传)只用到 .value 容器语义与构造时的类型归一。
class constexpr:  # SOURCE: python/triton/language/core.py:L134-143
    """This class is used to store a value that is known at compile-time."""

    def __init__(self, value):  # SOURCE: python/triton/language/core.py:L139-143
        if isinstance(value, constexpr):
            self.value = value.value
        else:
            self.value = value

    def __repr__(self) -> str:  # SOURCE: python/triton/language/core.py:L145-146
        return f"constexpr[{self.value}]"


def _unwrap_if_constexpr(o):  # SOURCE: python/triton/language/core.py:L270-271
    return o.value if isinstance(o, constexpr) else o


def _constexpr_to_value(v):  # SOURCE: python/triton/language/core.py:L1144-1147
    if isinstance(v, constexpr):
        return v.value
    return v


class _value:  # SOURCE: python/triton/language/core.py:L711-716
    """Base class of values that exist in the triton IR (i.e. not constexprs)."""

    def __init__(self, handle):  # SOURCE: python/triton/language/core.py:L715-716
        self.handle = handle


# SUBTRACTED: 真实 tensor(python/triton/language/core.py:L724-757)还从 type.is_block()/
# type.shape 派生 shape/numel 字段，并定义约 80 个运算符重载方法。本章 handle_scope_with
# 只把 tensor 当"IR handle + 类型"的容器传递(scope_op 的结果类型列表来自 ty.to_ir(...))，
# 从不做张量算术，故只留构造与 .type/.handle。
class tensor(_value):  # SOURCE: python/triton/language/core.py:L724-736
    """Represents an N-dimensional array of values or pointers."""

    def __init__(self, handle, type):  # SOURCE: python/triton/language/core.py:L744-754(节选)
        super().__init__(handle)
        self.type = type


# SUBTRACTED: 真实 range(python/triton/language/core.py:L2572-2635)的 __init__ 还接受
# disallow_acc_multi_buffer/flatten/warp_specialize/disable_licm 四个关键字并展开
# start/end/step 字段；__iter__/__next__ 均 raise RuntimeError(仅供 @triton.jit 内使用)。
# 本章的 parallel(aux_ops.py)只 super().__init__(arg1, arg2, step, num_stages,
# loop_unroll_factor) 转发前五个位置参数，从不读 start/end/step 字段本身，故这里只留
# 转发所需的构造签名与"仅供 jit 内使用"的报错行为。
class range:  # SOURCE: python/triton/language/core.py:L2572-2596(节选)
    """Iterator that counts upward forever."""

    def __init__(self, arg1, arg2=None, step=None, num_stages=None,  # SOURCE: L2614-2615(节选)
                 loop_unroll_factor=None):
        if step is None:  # SOURCE: python/triton/language/core.py:L2616-2626(节选)
            self.step = constexpr(1)
        else:
            self.step = step
        if arg2 is None:
            self.start = constexpr(0)
            self.end = arg1
        else:
            self.start = arg1
            self.end = arg2
        self.num_stages = num_stages
        self.loop_unroll_factor = loop_unroll_factor

    def __iter__(self):  # SOURCE: python/triton/language/core.py:L2634-2635
        raise RuntimeError("tl.range can only be used in @triton.jit'd functions")

    def __next__(self):  # SOURCE: python/triton/language/core.py:L2637-2638
        raise RuntimeError("tl.range can only be used in @triton.jit'd functions")
