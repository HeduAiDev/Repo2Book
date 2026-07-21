# 支撑层——triton.language 的值系统(constexpr/dtype/tensor)与内建标记机制，属基座
# Triton、未被昇腾 fork 改动。本章 custom_op.py/libdevice.py/math_ops.py 都
# `import triton.language.core as tl` 或 `from triton.language import core`依赖它，
# 但它本身不是本章 dossier 的机制主角(register_custom_op/libdevice 三分野)。
#
# SOURCE: python/triton/language/core.py(节选，见每个符号上的行号)
# SUBTRACTED: 真实文件还定义 tensor 的约 80 个运算符重载方法(__add__/__getitem__/...)、
# block_type/pointer_type/function_type、dtype 的 fp8 系列位宽表与 kind()/
# get_int_max_value() 等约 15 个查询方法、device_print/inline_asm_elementwise 等
# 数十个自由函数与 tensor_descriptor* 系列类——这些服务于块级张量运算与类型提升，
# 本章样例(register_custom_op 的标量/1D 校验、libdevice 的逐元素数学)从不依赖它们。

import inspect

__all__ = [
    "constexpr", "dtype", "tensor",
    "int1", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "bfloat16", "float32", "float64",
    "builtin", "extern", "_tensor_member_fn",
    "static_assert", "static_print", "extern_elementwise",
    "Callable", "TypeVar",
]

from typing import Callable, TypeVar  # noqa: E402  (仅供本章保留的类型注解使用)

TRITON_BUILTIN = "__triton_builtin__"  # SOURCE: python/triton/language/core.py:L19


# SUBTRACTED: 无——constexpr 本身很小，本章只用到构造/取值/相等比较三件事。
# 原：python/triton/language/core.py:L134-259(完整类，运算符重载方法未保留)。
class constexpr:  # SOURCE: python/triton/language/core.py:L134-146
    """This class is used to store a value that is known at compile-time."""

    def __init__(self, value):  # SOURCE: python/triton/language/core.py:L139-143
        if isinstance(value, constexpr):
            self.value = value.value
        else:
            self.value = value

    def __repr__(self):  # SOURCE: python/triton/language/core.py:L145-146
        return f"constexpr[{self.value}]"

    def __eq__(self, other):  # SOURCE: python/triton/language/core.py:L212-213
        return constexpr(self.value == _constexpr_to_value(other))

    # SUBTRACTED: 无对应——pin 的 constexpr 只定义了 __eq__(L212-213)/__ne__(L215-216)/
    # __bool__(L218-219)等运算符重载，从未定义 __hash__。本章样例(custom_op.py 的
    # _unwrap_constexpr、math_ops.py 的 core.constexpr(...))从不把 constexpr 实例当
    # dict key/放进 set，故不需要它可哈希——按"只做减法"原则不补这个 pin 没有的方法。

    def __bool__(self):  # SOURCE: python/triton/language/core.py:L218-219(节选)
        return bool(self.value)


def _unwrap_if_constexpr(o):  # SOURCE: python/triton/language/core.py:L270-271
    return o.value if isinstance(o, constexpr) else o


def _constexpr_to_value(v):  # SOURCE: python/triton/language/core.py:L1144-1147(节选)
    if isinstance(v, constexpr):
        return v.value
    return v


# SUBTRACTED: fp8 系列(fp8e4b15/fp8e4nv/fp8e4b8/fp8e5/fp8e5b16)的尾数宽度/指数偏置表、
# is_fp8*/kind()/get_int_max_value()/get_int_min_value()/codegen_name()/cache_key_part()
# 等约 15 个查询方法。本章样例(custom_op 的 core/pipe/mode 校验、libdevice 的
# fp16/fp32/bf16/int1/int32 逐元素数学)只用到 SINT/UINT/标准 FP 类型与下面保留的
# is_* 判定，故只留这些。原：python/triton/language/core.py:L288-605(节选)。
class dtype:  # SOURCE: python/triton/language/core.py:L288-303(节选)
    SINT_TYPES = ["int8", "int16", "int32", "int64"]
    UINT_TYPES = ["int1", "uint8", "uint16", "uint32", "uint64"]
    FP_TYPES = ["fp16", "bf16", "fp32", "fp64"]

    def __init__(self, name):  # SOURCE: python/triton/language/core.py:L304-352(节选)
        name = _unwrap_if_constexpr(name)
        self.name = name
        assert name in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES, name
        if name in dtype.SINT_TYPES:
            self.int_signedness = "signed"
            self.int_bitwidth = int(name.split("int")[-1])
            self.primitive_bitwidth = self.int_bitwidth
        elif name in dtype.UINT_TYPES:
            self.int_signedness = "unsigned"
            self.int_bitwidth = int(name.split("int")[-1])
            self.primitive_bitwidth = self.int_bitwidth
        elif name == "fp16":
            self.primitive_bitwidth = 16
        elif name == "bf16":
            self.primitive_bitwidth = 16
        elif name == "fp32":
            self.primitive_bitwidth = 32
        elif name == "fp64":
            self.primitive_bitwidth = 64

    def is_int1(self):  # SOURCE: python/triton/language/core.py:L388-389
        return self.name == "int1"

    def is_int8(self):  # SOURCE: python/triton/language/core.py:L391-392
        return self.name == "int8"

    def is_int16(self):  # SOURCE: python/triton/language/core.py:L394-395
        return self.name == "int16"

    def is_int32(self):  # SOURCE: python/triton/language/core.py:L397-398
        return self.name == "int32"

    def is_int64(self):  # SOURCE: python/triton/language/core.py:L400-401
        return self.name == "int64"

    def is_uint8(self):  # SOURCE: python/triton/language/core.py:L403-404
        return self.name == "uint8"

    def is_uint16(self):  # SOURCE: python/triton/language/core.py:L406-407
        return self.name == "uint16"

    def is_uint32(self):  # SOURCE: python/triton/language/core.py:L409-410
        return self.name == "uint32"

    def is_uint64(self):  # SOURCE: python/triton/language/core.py:L412-413
        return self.name == "uint64"

    def is_fp16(self):  # SOURCE: python/triton/language/core.py:L376-377
        return self.name == "fp16"

    def is_bf16(self):  # SOURCE: python/triton/language/core.py:L379-380
        return self.name == "bf16"

    def is_fp32(self):  # SOURCE: python/triton/language/core.py:L382-383
        return self.name == "fp32"

    def is_fp64(self):  # SOURCE: python/triton/language/core.py:L385-386
        return self.name == "fp64"

    def is_floating(self):  # SOURCE: python/triton/language/core.py:L415-416(节选)
        return self.name in dtype.FP_TYPES

    def is_int_signed(self):  # SOURCE: python/triton/language/core.py:L421-422(节选)
        return self.name in dtype.SINT_TYPES

    def is_int_unsigned(self):  # SOURCE: python/triton/language/core.py:L424-425(节选)
        return self.name in dtype.UINT_TYPES

    def is_int(self):  # SOURCE: python/triton/language/core.py:L427-428
        return self.name in dtype.SINT_TYPES + dtype.UINT_TYPES

    @staticmethod
    def is_block():  # SOURCE: python/triton/language/core.py:L466-467
        return False

    @staticmethod
    def is_ptr():  # SOURCE: python/triton/language/core.py:L469-471
        return False

    def __eq__(self, other):  # SOURCE: python/triton/language/core.py:L477-480
        if not isinstance(other, dtype):
            return False
        return self.name == other.name

    def __ne__(self, other):  # SOURCE: python/triton/language/core.py:L482-483
        return not self.__eq__(other)

    def __hash__(self):  # SOURCE: python/triton/language/core.py:L485-486
        return hash((self.name, ))

    @property
    def scalar(self):  # SOURCE: python/triton/language/core.py:L488-490
        return self

    def __str__(self):  # SOURCE: python/triton/language/core.py:L532-533
        return self.name

    def __repr__(self):  # SOURCE: python/triton/language/core.py:L548-550
        return f"triton.language.{self.name}"


class _value:  # SOURCE: python/triton/language/core.py:L711-716
    """Base class of values that exist in the triton IR (i.e. not constexprs)."""

    def __init__(self, handle):  # SOURCE: python/triton/language/core.py:L715-716
        self.handle = handle


# SUBTRACTED: 真实 tensor 还定义约 80 个运算符重载方法(__add__/__sub__/.../
# __getitem__)，全部转发到 triton.language.semantic 的算子实现，供"张量参与算术
# 表达式"。本章 custom_op 只把 tensor 当"handle + dtype/shape 的容器"传递，
# libdevice 的三条实现路径都直接调 semantic.*/math.* 自由函数(不经运算符重载)，
# 故只留构造。原：python/triton/language/core.py:L724-1135。
class tensor(_value):  # SOURCE: python/triton/language/core.py:L724-736
    """Represents an N-dimensional array of values or pointers."""

    def __init__(self, handle, type):  # SOURCE: python/triton/language/core.py:L743-755(节选，去掉 block shape/numel 计算)
        super().__init__(handle)
        self.type = type
        self.dtype = type.scalar
        self.shape = ()

    @staticmethod
    def is_ptr():  # SOURCE: python/triton/language/core.py:L579(节选，本章样例从不构造指针张量)
        return False

    def __str__(self):  # SOURCE: python/triton/language/core.py:L757-759(节选)
        return str(self.dtype)


# scalar types —— 只留本章实际用到的类型。原文件还定义 fp8 系列/void 等。
# SOURCE: python/triton/language/core.py:L665-682(节选)
int1 = dtype("int1")
int8 = dtype("int8")
int16 = dtype("int16")
int32 = dtype("int32")
int64 = dtype("int64")
uint8 = dtype("uint8")
uint16 = dtype("uint16")
uint32 = dtype("uint32")
uint64 = dtype("uint64")
float16 = dtype("fp16")
bfloat16 = dtype("bf16")
float32 = dtype("fp32")
float64 = dtype("fp64")


def builtin(fn):  # SOURCE: python/triton/language/core.py:L25-37
    """Mark a function as a builtin."""
    assert callable(fn)

    def wrapper(*args, **kwargs):  # SOURCE: python/triton/language/core.py:L30-35
        if "_builder" not in kwargs or kwargs["_builder"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_builder` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    setattr(wrapper, TRITON_BUILTIN, True)
    return wrapper


def extern(fn):  # SOURCE: python/triton/language/core.py:L2743-2745
    """A decorator for external functions."""
    return builtin(fn)


# SUBTRACTED: 真实 _tensor_member_fn 还会把被装饰函数动态挂到 tensor 类上、生成
# "也可当 x.foo(...) 调"的成员方法版本(python/triton/language/core.py:L42-64)。本章
# math_ops.py 的 isfinited/atan2/finitef 只被当自由函数调用(从不写 x.isfinited())，
# 故这里只保留装饰器"透传不改变函数"这一半行为。
def _tensor_member_fn(fn):  # SOURCE: python/triton/language/core.py:L42-45(节选)
    return fn


def static_print(*values, sep=" ", end="\n", file=None, flush=False, _builder=None):  # SOURCE: L2295-2307
    pass


# SUBTRACTED: 真实 static_assert 还挂着 @builtin 装饰器(python/triton/language/
# core.py:L2310)，要求调用点传 `_builder=` kwarg，否则报"Did you forget to add
# @triton.jit ?"——这是因为 @jit 函数体内的 `core.static_assert(...)` 调用本来就不是
# 被"直接当 Python 函数调"的，而是被 codegen 在 trace 期自动注入 `_builder`(同
# isnan/isinf 等其他内建)。本章测试直接调 math_ops.isfinited.fn(x) 复现函数体的真实
# 组合逻辑(不驱动完整 codegen trace)，同 ch04 对 JITFunction 的处理方式：静态断言
# 本身函数体是空(pass，不校验任何东西)，这里去掉 @builtin 包装只是让它能在"不经
# codegen trace"的测试环境下被直接调用，不改变它"什么都不做"的真实行为。
def static_assert(cond, msg="", _builder=None):  # SOURCE: python/triton/language/core.py:L2311-2320(节选)
    pass


# SUBTRACTED: 真实 extern_elementwise 还做隐式广播(block 参数的形状对齐)与
# "无精确 dtype 匹配时退化为算术类型提升再查表"的宽松匹配(binary_op_type_checking_impl
# 那一段，L2708-2723)。本章 libdevice 样例(reciprocal/tanh)调用点的实参 dtype 总是与
# arg_type_symbol_dict 的某个 key 精确相等(标量、不分块)，故这里只保留"精确 dtype 元组
# 查表 -> 调 create_extern_elementwise"这条主干，这也是本章要讲的机制本身:
# 『extern 只能调预置符号』。原：python/triton/language/core.py:L2691-L2730。
def extern_elementwise(lib_name, lib_path, args, arg_type_symbol_dict, is_pure, _builder=None):  # SOURCE: python/triton/language/core.py:L2691-L2730
    from . import semantic
    dispatch_args = [semantic.to_tensor(a, _builder) for a in args]
    arg_types = tuple(a.dtype for a in dispatch_args)
    if arg_types not in arg_type_symbol_dict:
        raise KeyError(f"extern_elementwise: no symbol registered for dtypes {arg_types}")
    symbol, ret_dtype = arg_type_symbol_dict[arg_types]
    handles = [a.handle for a in dispatch_args]
    ret_handle = _builder.create_extern_elementwise(symbol, handles)
    return tensor(ret_handle, ret_dtype)
