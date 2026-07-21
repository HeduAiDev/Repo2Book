# 基座（非 ascend）language 层的最小子集——mem_ops.py/vec_ops.py 依赖的 triton.language
# 值系统（constexpr/dtype/pointer_type/block_type/tensor）与双标记机制基座半边
# （TRITON_BUILTIN/builtin/is_builtin/_tensor_member_fn）。三者本身属于基座 Triton、
# 未被昇腾 fork 改动，按 dossier code_spine 本章只圈定 third_party/ascend/.../
# {mem_ops,vec_ops,_utils,aux_ops}.py 的具体行区间；tl.* 在那些文件之外，这里只取
# 它们真正调用到的最小子集（同 ch04/ch05 对本文件的一贯处理方式）。
#
# SOURCE: python/triton/language/core.py（节选，见每个符号上的行号）

from enum import Enum
from typing import List


# SOURCE: python/triton/language/core.py:L20
TRITON_BUILTIN = "__triton_builtin__"


# SOURCE: python/triton/language/core.py:L25-38
def builtin(fn):
    """Mark a function as a builtin."""
    assert callable(fn)

    def wrapper(*args, **kwargs):  # SOURCE: python/triton/language/core.py:L25-38(builtin)/L108-110(is_builtin)
        if "_builder" not in kwargs or kwargs["_builder"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_builder` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    wrapper.__wrapped__ = fn
    setattr(wrapper, TRITON_BUILTIN, True)
    return wrapper


# SOURCE: python/triton/language/core.py:L108-110
def is_builtin(fn) -> bool:
    """Is this a registered triton builtin function?"""
    return getattr(fn, TRITON_BUILTIN, False)


# SOURCE: python/triton/language/core.py:L42-83
# SUBTRACTED: 真实实现额外做了 inspect.signature 的文档拼接与 __signature__ 复制，只
# 为了让生成的 docstring/IDE 签名好看；本章只需要"把自由函数登记成 tensor 的同名方法"
# 这一件事（ch06 lead_addenda 指出的『注入并不齐整』现象，靠的正是这里的 setattr）。
def _tensor_member_fn(fn):  # SOURCE: python/triton/language/core.py:L42-83(_tensor_member_fn)
    """Decorator that adds this free function as a member fn on class tensor."""
    assert callable(fn)

    def wrapper(*args, **kwargs):  # SOURCE: python/triton/language/core.py:L42-83(_tensor_member_fn)
        return fn(*args, **kwargs)

    wrapper.__name__ = fn.__name__
    if is_builtin(fn):
        setattr(wrapper, TRITON_BUILTIN, True)

    setattr(tensor, fn.__name__, wrapper)
    return fn


# SOURCE: python/triton/language/core.py:L134-259（全量保留，未删减）——static_range/
# _log2/_is_power_of_two/flip_impl 的算术全部经由 constexpr 的运算符重载完成，任何
# 裁剪都会在某个用不到的分支上悄悄断掉调用链，故此类不做「只留用到的方法」式精简。
class constexpr:  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
    """This class is used to store a value that is known at compile-time."""

    def __init__(self, value):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        if isinstance(value, constexpr):
            self.value = value.value
        else:
            self.value = value

    def __repr__(self) -> str:  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return f"constexpr[{self.value}]"

    def __index__(self):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return self.value

    def __add__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(self.value + _constexpr_to_value(other))

    def __radd__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(_constexpr_to_value(other) + self.value)

    def __sub__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(self.value - _constexpr_to_value(other))

    def __rsub__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(_constexpr_to_value(other) - self.value)

    def __mul__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(self.value * _constexpr_to_value(other))

    def __mod__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(self.value % _constexpr_to_value(other))

    def __rmul__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(_constexpr_to_value(other) * self.value)

    def __truediv__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(self.value / _constexpr_to_value(other))

    def __rtruediv__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(_constexpr_to_value(other) / self.value)

    def __floordiv__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(self.value // _constexpr_to_value(other))

    def __rfloordiv__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(_constexpr_to_value(other) // self.value)

    def __gt__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(self.value > _constexpr_to_value(other))

    def __ge__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(self.value >= _constexpr_to_value(other))

    def __lt__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(self.value < _constexpr_to_value(other))

    def __le__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(self.value <= _constexpr_to_value(other))

    def __eq__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(self.value == _constexpr_to_value(other))

    def __ne__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(self.value != _constexpr_to_value(other))

    def __bool__(self):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return bool(self.value)

    def __neg__(self):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(-self.value)

    def __and__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(self.value & _constexpr_to_value(other))

    def __or__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(self.value | _constexpr_to_value(other))

    def __xor__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(self.value ^ _constexpr_to_value(other))

    def __pos__(self):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(+self.value)

    def __invert__(self):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(~self.value)

    def __rshift__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(self.value >> _constexpr_to_value(other))

    def __lshift__(self, other):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return constexpr(self.value << _constexpr_to_value(other))

    def __hash__(self):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return hash(self.value)

    def __iter__(self):  # SOURCE: python/triton/language/core.py:L134-259(constexpr)
        return iter(self.value)


CONSTEXPR_0 = constexpr(0)


# SOURCE: python/triton/language/core.py:L270-271
def _unwrap_if_constexpr(o):
    return o.value if isinstance(o, constexpr) else o


# SOURCE: python/triton/language/core.py:L1144-1147
def _constexpr_to_value(v):
    if isinstance(v, constexpr):
        return v.value
    return v


# SOURCE: python/triton/language/core.py:L1206-1208
def _unwrap_shape(shape):
    shape = _constexpr_to_value(shape)
    return [_constexpr_to_value(s) for s in shape]


def validate_block_shape(shape: List[int]):
    # SOURCE: python/triton/language/_utils.py:L17-29（节选:只留 numel 计算，去掉
    # TRITON_MAX_TENSOR_NUMEL 的上限校验——本章测试用的 tile 都很小，不会触达上限）
    numel = 1
    for s in shape:
        numel *= s
    return numel


# SOURCE: python/triton/language/core.py:L288-358(节选)
# SUBTRACTED: fp8 各变体的 fp_mantissa_width/exponent_bias 具体数值表、KIND 枚举、
# get_int_max_value/get_int_min_value、is_dtype/is_void 等静态查询方法——ascend_cast_impl
# 里涉及 fp8 的分支本身已按 subtraction_plan 删除（不构造 fp8 输入），这里只保留让
# 「拒绝 fp8/fp64」这条守卫子句(kept)能求值为 False 所需的 is_fp8()/is_fp64() 判据。
class dtype:  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
    SINT_TYPES = ['int8', 'int16', 'int32', 'int64']
    UINT_TYPES = ['int1', 'uint8', 'uint16', 'uint32', 'uint64']
    FP_TYPES = ['fp8e4nv', 'fp8e5', 'fp16', 'bf16', 'fp32', 'fp64']
    STANDARD_FP_TYPES = ['fp16', 'bf16', 'fp32', 'fp64']
    OTHER_TYPES = ['void']

    class SIGNEDNESS(Enum):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        SIGNED = 0
        UNSIGNED = 1

    def __init__(self, name):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        name = _unwrap_if_constexpr(name)
        self.name = name
        assert name in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES, name
        if name in dtype.SINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.SIGNED
            self.int_bitwidth = int(name.split('int')[-1])
            self.primitive_bitwidth = self.int_bitwidth
        elif name in dtype.UINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.UNSIGNED
            self.int_bitwidth = 1 if name == 'int1' else int(name.split('int')[-1])
            self.primitive_bitwidth = self.int_bitwidth
        elif name in dtype.FP_TYPES:
            self.primitive_bitwidth = {'fp8e4nv': 8, 'fp8e5': 8, 'fp16': 16, 'bf16': 16,
                                       'fp32': 32, 'fp64': 64}[name]
        elif name == 'void':
            self.primitive_bitwidth = 0

    def is_fp8(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return 'fp8' in self.name

    def is_fp16(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return self.name == 'fp16'

    def is_bf16(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return self.name == 'bf16'

    def is_fp32(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return self.name == 'fp32'

    def is_fp64(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return self.name == 'fp64'

    def is_int1(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return self.name == 'int1'

    def is_int8(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return self.name == 'int8'

    def is_int16(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return self.name == 'int16'

    def is_int32(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return self.name == 'int32'

    def is_int64(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return self.name == 'int64'

    def is_floating(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return self.name in dtype.FP_TYPES

    def is_standard_floating(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return self.name in dtype.STANDARD_FP_TYPES

    def is_int_signed(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return self.name in dtype.SINT_TYPES

    def is_int_unsigned(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return self.name in dtype.UINT_TYPES

    def is_int(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return self.name in dtype.SINT_TYPES + dtype.UINT_TYPES

    def is_bool(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return self.is_int1()

    @staticmethod
    def is_block():  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return False

    @staticmethod
    def is_ptr():  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return False

    def __eq__(self, other):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        if not isinstance(other, dtype):
            return False
        return self.name == other.name

    def __ne__(self, other):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return not self.__eq__(other)

    def __hash__(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return hash((self.name, ))

    @property
    def scalar(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return self

    # 分支只留本章实际出现的类型；fp8*/fp64/uint* 的下沉未被本章任何 must_keep 路径
    # 触达（fp8/fp64 在 ascend_cast_impl 里被前置守卫拒绝，见上），予以省略。
    def to_ir(self, builder):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        if self.name == 'int1':
            return builder.get_int1_ty()
        elif self.name == 'int8':
            return builder.get_int8_ty()
        elif self.name == 'int16':
            return builder.get_int16_ty()
        elif self.name == 'int32':
            return builder.get_int32_ty()
        elif self.name == 'int64':
            return builder.get_int64_ty()
        elif self.name == 'fp16':
            return builder.get_half_ty()
        elif self.name == 'bf16':
            return builder.get_bf16_ty()
        elif self.name == 'fp32':
            return builder.get_float_ty()
        raise ValueError(f'fail to convert {self} to ir type')

    def __str__(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return self.name

    def __repr__(self):  # SOURCE: python/triton/language/core.py:L288-358(dtype，节选)
        return f'triton.language.{self.name}'


# SOURCE: python/triton/language/core.py:L559-591(节选)
# SUBTRACTED: const/address_space 的完整语义(is_const/__eq__ 里的 const 比较)与
# nv_tma_desc_type 子类——mem_ops 的 GM 指针从不携带这些属性。
class pointer_type(dtype):  # SOURCE: python/triton/language/core.py:L559-591(pointer_type，节选)

    def __init__(self, element_ty: dtype, address_space: int = 1):  # SOURCE: python/triton/language/core.py:L559-591(pointer_type，节选)
        if not isinstance(element_ty, dtype):
            raise TypeError(f'element_ty has type `{type(element_ty).__name__}`; expected `dtype`.')
        self.element_ty = element_ty
        self.address_space = address_space
        self.name = f'pointer<{element_ty}>'

    def to_ir(self, builder):  # SOURCE: python/triton/language/core.py:L559-591(pointer_type，节选)
        return builder.get_ptr_ty(self.element_ty.to_ir(builder), self.address_space)

    def __str__(self):  # SOURCE: python/triton/language/core.py:L559-591(pointer_type，节选)
        return self.name

    def is_ptr(self):  # SOURCE: python/triton/language/core.py:L559-591(pointer_type，节选)
        return True

    def __eq__(self, other):  # SOURCE: python/triton/language/core.py:L559-591(pointer_type，节选)
        if not isinstance(other, pointer_type):
            return False
        return self.element_ty == other.element_ty and self.address_space == other.address_space

    def __ne__(self, other):  # SOURCE: python/triton/language/core.py:L559-591(pointer_type，节选)
        return not self.__eq__(other)

    @property
    def scalar(self):  # SOURCE: python/triton/language/core.py:L559-591(pointer_type，节选)
        return self


# SOURCE: python/triton/language/core.py:L605-646（全量保留，很小）
class block_type(dtype):

    def __init__(self, element_ty: dtype, shape: List):  # SOURCE: python/triton/language/core.py
        self.element_ty = element_ty
        self.shape = _unwrap_shape(shape)
        if not self.shape:
            raise TypeError('0d block_type is forbidden')
        self.numel = validate_block_shape(self.shape)
        self.name = f'<{self.shape}, {self.element_ty}>'

    def to_ir(self, builder):  # SOURCE: python/triton/language/core.py
        return builder.get_block_ty(self.element_ty.to_ir(builder), self.shape)

    def __str__(self):  # SOURCE: python/triton/language/core.py
        return self.name

    def is_block(self):  # SOURCE: python/triton/language/core.py
        return True

    def __eq__(self, other):  # SOURCE: python/triton/language/core.py
        if not isinstance(other, block_type):
            return False
        return self.element_ty == other.element_ty and self.shape == other.shape

    def __ne__(self, other):  # SOURCE: python/triton/language/core.py
        return not self.__eq__(other)

    def get_block_shapes(self) -> List[int]:  # SOURCE: python/triton/language/core.py
        return self.shape

    @property
    def scalar(self):  # SOURCE: python/triton/language/core.py
        return self.element_ty


# SOURCE: python/triton/language/core.py:L711-716
class _value:
    """Base class of values that exist in the triton IR (i.e. not constexprs)."""

    def __init__(self, handle):  # SOURCE: python/triton/language/core.py:L711-716(_value)
        self.handle = handle


# SOURCE: python/triton/language/core.py:L724-757 + L989-998(.to 方法)
# SUBTRACTED: 真实 tensor 还定义 __add__/__sub__/__getitem__ 等约 80 个运算符重载，
# 全部转发到 triton.language.semantic 的算子实现——本章 mem_ops/vec_ops 从不对 tensor
# 做算术（只做 dtype/shape 查询、.to() 类型转换与 __xor__ 一个位运算，flip 的 SIMT
# 分支要用），故只留这些。
class tensor(_value):
    """Represents an N-dimensional array of values or pointers。"""

    def __init__(self, handle, type: dtype):  # SOURCE: python/triton/language/core.py:L724-757+L989-998(tensor，节选)
        super().__init__(handle)
        self.shape = type.shape if type.is_block() else ()
        self.numel = 1
        for s in self.shape:
            self.numel *= s
        self.numel = constexpr(self.numel)
        self.type = type
        self.dtype = type.scalar
        self.shape = [constexpr(s) for s in self.shape]

    def __str__(self) -> str:  # SOURCE: python/triton/language/core.py:L724-757+L989-998(tensor，节选)
        return str(self.dtype) + '[' + ', '.join(str(s) for s in self.shape) + ']'

    # SOURCE: python/triton/language/core.py:L989-998
    def to(self, dtype, fp_downcast_rounding=None, bitcast: bool = False, overflow_mode=None, _builder=None):
        """Alias for :py:func:`tensor.cast`（flip 的 SIMT 回退分支用它做位重解释）。"""
        from . import semantic
        dtype = _unwrap_if_constexpr(dtype)
        bitcast = _unwrap_if_constexpr(bitcast)
        if bitcast:
            return semantic.bitcast(self, dtype, _builder)
        return semantic.cast(self, dtype, _builder, fp_downcast_rounding, overflow_mode)

    # flip 的 SIMT 分支对 reshape 后的整型 tile 做 `y.__xor__(...)`——这是 xor_sum 的
    # 归约结果与 y 本身做逐元素异或，接的是 tensor 的位运算重载而非 Python int 的。
    def __xor__(self, other, _builder=None):  # SOURCE: python/triton/language/core.py:L724-757+L989-998(tensor，节选)
        from . import semantic
        return semantic.xor_(self, other, _builder)


# scalar types —— 只留本章实际用到的（mem_ops 的 fp16/bf16/fp32/index 的各整型 +
# sort 的 allowed_types 白名单 + cast 的位宽收窄测试）。
int1 = dtype('int1')
int8 = dtype('int8')
int16 = dtype('int16')
int32 = dtype('int32')
int64 = dtype('int64')
uint8 = dtype('uint8')
uint16 = dtype('uint16')
uint32 = dtype('uint32')
float16 = dtype('fp16')
bfloat16 = dtype('bf16')
float32 = dtype('fp32')
float8e4nv = dtype('fp8e4nv')
float8e5 = dtype('fp8e5')
void = dtype('void')


# SOURCE: python/triton/language/core.py:L688-703(节选，只留本章用到的位宽)
def get_int_dtype(bitwidth: int, signed: bool) -> dtype:
    table = {
        (1, True): int1, (1, False): int1,
        (8, True): int8, (8, False): uint8,
        (16, True): int16, (16, False): uint16,
        (32, True): int32, (32, False): uint32,
    }
    if (bitwidth, signed) not in table:
        raise ValueError(f"unsupported (bitwidth={bitwidth}, signed={signed}) in this chapter's subset")
    return table[(bitwidth, signed)]


# SOURCE: python/triton/language/core.py:L86-102（_unwrap_iterable，全量保留）
def _unwrap_iterable(x):
    """Returns x[0] if x has one element and x[0] is iterable.

    真实 flip 的 SIMT 分支拼 shape 时写的是
    `slice1 + [2] * steps + slice2`——`steps` 是 constexpr，`[2] * steps` 触发的
    是 constexpr.__rmul__（list.__mul__ 对非 int 操作数返回 NotImplemented），结果
    整个表达式变成一个"包着 list 的 constexpr"，而不是一个 list。core.reshape 的
    `*shape` 因此拿到的是单元素元组 `(constexpr([2, 2]),)`——这个函数就是为了在这种
    情况下把它"打开"成 `constexpr([2, 2])` 本身（判据正是 constexpr 定义了
    __iter__），交给下面 _unwrap_shape 里的 _constexpr_to_value 继续剥一层拿到
    真正的 `[2, 2]`。少了这一步会得到 `[[2, 2]]`（外层多包一层 list）。
    """
    if len(x) == 1:
        try:
            iter(x[0])
            return x[0]
        except TypeError:
            pass
    return x


# SOURCE: python/triton/language/core.py:L1429-1443(节选)
def reshape(input, *shape, can_reorder=False, _builder=None):
    from . import semantic
    shape = _unwrap_iterable(shape)
    shape = _unwrap_shape(shape)
    return semantic.reshape(input, shape, can_reorder, _builder)


# SOURCE: python/triton/language/core.py:L2311-2320(节选)
def static_assert(cond, msg="", _builder=None):
    if not cond:
        raise AssertionError(msg or "static assertion failed")
