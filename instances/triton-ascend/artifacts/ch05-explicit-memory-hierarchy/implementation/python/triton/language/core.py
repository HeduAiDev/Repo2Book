# 本文件不是 ch05 的机制主角——它是 buffer 语言 / al.copy / al.fixpipe 依赖的
# triton.language 值系统（tl.constexpr/tl.dtype/tl.tensor/tl.block_type），三者本身
# 属于基座 Triton、未被昇腾 fork 改动。按 dossier code_spine，本章只圈定
# third_party/ascend/language/cann/extension/{core,semantic}.py 与
# python/triton/extension/buffer/language/{core,semantic}.py 四个文件的具体行区间；
# tl.* 在那四个文件之外，这里只取它们真正调用到的最小子集（同 ch04 对本文件的处理
# 方式），其余算子重载/类型目录（tensor 的全部运算符方法、pointer_type、
# function_type、load/store/dot/reshape 等几十个自由函数……)与内存层级/copy/fixpipe
# 无关，不纳入。
#
# SOURCE: python/triton/language/core.py（节选，见每个符号上的行号）

from enum import Enum
from typing import List

from ._utils import validate_block_shape

__all__ = [
    "constexpr", "_unwrap_if_constexpr", "_constexpr_to_value", "_unwrap_shape",
    "dtype", "block_type", "_value", "tensor",
    "int1", "int16", "int32", "float16", "bfloat16", "float32",
]


# SUBTRACTED: 真实 constexpr 还定义 __add__/__sub__/__mul__/__truediv__/.../__pow__/
# __iter__/__call__ 等约 30 个算术与容器协议方法，供"编译期常量参与表达式运算"。
# 本章 tl.constexpr 只用作 shape/dtype 的编译期包装，从不参与算术，故只保留构造、
# 相等比较（`etype == tl.int1` 依赖它）与布尔转换（`if` 语句依赖它）。
# 原：python/triton/language/core.py:L134-259。
class constexpr:  # SOURCE: python/triton/language/core.py:L134-146
    """This class is used to store a value that is known at compile-time."""

    def __init__(self, value):  # SOURCE: python/triton/language/core.py:L139-143
        if isinstance(value, constexpr):
            self.value = value.value
        else:
            self.value = value

    def __repr__(self) -> str:  # SOURCE: python/triton/language/core.py:L145-146
        return f"constexpr[{self.value}]"

    def __eq__(self, other):  # SOURCE: python/triton/language/core.py:L212-213(__eq__)
        return constexpr(self.value == _constexpr_to_value(other))

    def __bool__(self):  # SOURCE: python/triton/language/core.py:L221-222(__bool__)
        return bool(self.value)


def _unwrap_if_constexpr(o):  # SOURCE: python/triton/language/core.py:L270-271
    return o.value if isinstance(o, constexpr) else o


def _constexpr_to_value(v):  # SOURCE: python/triton/language/core.py:L1144-1147
    if isinstance(v, constexpr):
        return v.value
    return v


def _unwrap_shape(shape):  # SOURCE: python/triton/language/core.py:L1206-1208
    shape = _constexpr_to_value(shape)
    return [_constexpr_to_value(s) for s in shape]


# SUBTRACTED: 真实 dtype 还有 is_fp8*/is_int*/is_floating/kind/get_int_max_value/
# get_int_min_value/codegen_name/cache_key_part/__repr__ 等约 25 个查询方法，以及
# fp8 系列 dtype 的位宽/尾数表(fp_mantissa_width/exponent_bias)。这些服务于类型
# 提升(computation_type_impl)与算子分发，本章从不做类型提升，只用 dtype 做"这块
# buffer/tensor 是什么元素类型"的标签 + to_ir 下沉 + primitive_bitwidth(subview 的
# 32-byte 对齐换算要用到位宽)，故只留 SINT/UINT/FP 类型表(供 __init__ 校验与位宽
# 推导)、is_block/is_ptr(tensor.__init__ 会调 is_block)、scalar 属性、
# __eq__/__hash__(buffer_type.__eq__ 用它比较 element_ty)与 to_ir(fixpipe 按
# dst.type.element_ty 判 32b/16b 要用到具体 dtype 实例，to_ir 则是 buffer_type.to_ir
# 下沉链路的一环)。to_ir 的分支只留本章实际出现的六个类型(int1/int16/int32/fp16/
# bf16/fp32)，其余 fp8*/int8/int64 分支删除——原 if/elif 链条 L500-529，此处只删
# 未触达的分支。
class dtype:  # SOURCE: python/triton/language/core.py:L288-358(节选)
    SINT_TYPES = ['int8', 'int16', 'int32', 'int64']
    UINT_TYPES = ['int1', 'uint8', 'uint16', 'uint32', 'uint64']
    FP_TYPES = ['fp8e4b15', 'fp8e4nv', 'fp8e4b8', 'fp8e5', 'fp8e5b16', 'fp16', 'bf16', 'fp32', 'fp64']
    OTHER_TYPES = ['void']

    # 原分支覆盖全部 SINT/UINT/FP 类型，这里只保留本章实际用到的六个具体类型的位宽推导。
    def __init__(self, name):  # SOURCE: python/triton/language/core.py:L304-352(节选)
        name = _unwrap_if_constexpr(name)
        self.name = name
        assert name in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES, name
        if name in dtype.SINT_TYPES or name in dtype.UINT_TYPES:
            self.primitive_bitwidth = int(name.split('int')[-1])
        elif name == 'fp16':
            self.primitive_bitwidth = 16
        elif name == 'bf16':
            self.primitive_bitwidth = 16
        elif name == 'fp32':
            self.primitive_bitwidth = 32

    @staticmethod
    def is_block():  # SOURCE: python/triton/language/core.py:L465-467
        return False

    @staticmethod
    def is_ptr():  # SOURCE: python/triton/language/core.py:L469-471
        return False

    def __eq__(self, other: "dtype"):  # SOURCE: python/triton/language/core.py:L477-480
        if not isinstance(other, dtype):
            return False
        return self.name == other.name

    def __ne__(self, other: "dtype"):  # SOURCE: python/triton/language/core.py:L482-483
        return not self.__eq__(other)

    def __hash__(self):  # SOURCE: python/triton/language/core.py:L485-486
        return hash((self.name, ))

    @property
    def scalar(self):  # SOURCE: python/triton/language/core.py:L488-490
        return self

    # 分支只留本章实际出现的六个类型(int1/int16/int32/fp16/bf16/fp32)，其余
    # fp8*/int8/int64 分支删除。
    def to_ir(self, builder) -> "object":  # SOURCE: python/triton/language/core.py:L492-530(节选)
        if self.name == 'int1':
            return builder.get_int1_ty()
        elif self.name == 'int16':
            return builder.get_int16_ty()
        elif self.name == 'int32':
            return builder.get_int32_ty()
        elif self.name == 'fp16':
            return builder.get_half_ty()
        elif self.name == 'bf16':
            return builder.get_bf16_ty()
        elif self.name == 'fp32':
            return builder.get_float_ty()
        raise ValueError(f'fail to convert {self} to ir type')

    def __str__(self):  # SOURCE: python/triton/language/core.py:L532-533
        return self.name

    def __repr__(self):  # SOURCE: python/triton/language/core.py:L548-550
        return f'triton.language.{self.name}'


# SUBTRACTED: 无——block_type 本身很小，全量保留(shape 校验委托 validate_block_shape)。
class block_type(dtype):  # SOURCE: python/triton/language/core.py:L605-646

    def __init__(self, element_ty: dtype, shape: List):  # SOURCE: python/triton/language/core.py:L607-619
        self.element_ty = element_ty
        # Note that block_type's shape is a list of int
        # while tensor's shape is a list of constexpr.
        self.shape = _unwrap_shape(shape)
        if not self.shape:
            raise TypeError('0d block_type is forbidden')
        self.numel = validate_block_shape(self.shape)
        self.name = f'<{self.shape}, {self.element_ty}>'

    def to_ir(self, builder):  # SOURCE: python/triton/language/core.py:L621-622
        return builder.get_block_ty(self.element_ty.to_ir(builder), self.shape)

    def __str__(self):  # SOURCE: python/triton/language/core.py:L624-625
        return self.name

    def is_block(self):  # SOURCE: python/triton/language/core.py:L630-631
        return True

    def __eq__(self, other: "block_type"):  # SOURCE: python/triton/language/core.py:L636-639
        if not isinstance(other, block_type):
            return False
        return self.element_ty == other.element_ty and self.shape == other.shape

    def __ne__(self, other: "block_type"):  # SOURCE: python/triton/language/core.py:L641-642
        return not self.__eq__(other)

    @property
    def scalar(self):  # SOURCE: python/triton/language/core.py:L644-646
        return self.element_ty


class _value:  # SOURCE: python/triton/language/core.py:L711-716
    """Base class of values that exist in the triton IR (i.e. not constexprs)."""

    def __init__(self, handle):  # SOURCE: python/triton/language/core.py:L715-716
        self.handle = handle


# SUBTRACTED: 真实 tensor 还定义 __add__/__sub__/__mul__/.../__getitem__ 等约 80 个
# 运算符重载方法，全部转发到 triton.language.semantic 的算子实现。本章 al.copy/
# al.fixpipe/bl.to_tensor 只把 tensor 当"handle + dtype/shape 的容器"传递，从不对
# tensor 做算术，故只留构造与打印。原：python/triton/language/core.py:L724-1135。
class tensor(_value):  # SOURCE: python/triton/language/core.py:L724-736
    """Represents an N-dimensional array of values or pointers."""

    def __init__(self, handle, type: dtype):  # SOURCE: python/triton/language/core.py:L743-755
        super().__init__(handle)
        self.shape = type.shape if type.is_block() else ()
        self.numel = 1
        for s in self.shape:
            self.numel *= s
        self.numel = constexpr(self.numel)
        self.type = type
        self.dtype = type.scalar
        self.shape = [constexpr(s) for s in self.shape]

    def __str__(self) -> str:  # SOURCE: python/triton/language/core.py:L757-759
        return str(self.dtype) + '[' + ', '.join(str(s) for s in self.shape) + ']'


# scalar types —— 只留本章实际用到的六个（buffer_type/fixpipe 的 dtype 校验都只
# 涉及这些）。原文件还定义 int8/int32/int64/uint*/fp8*/fp64/void 等约 20 个。
# SOURCE: python/triton/language/core.py:L665-682（节选）
int1 = dtype('int1')
int16 = dtype('int16')
int32 = dtype('int32')
float16 = dtype('fp16')
bfloat16 = dtype('bf16')
float32 = dtype('fp32')
