# SOURCE: python/triton/extension/buffer/language/core.py（节选，行号见各符号上方）
#
# 本文件是 buffer 语言扩展的前端：address_space 抽象基类、buffer_type/buffer 两个
# 值/类型对象、以及 alloc/to_buffer/to_tensor/subview 四个用户可见的 builtin 入口。
# 基座 Triton 完全没有这一层——GPU 的 shared memory 由编译器托管，Triton 程序员只
# 写 tl.load/tl.store；NPU 把"在哪级内存开一块缓冲、这块缓冲的地址空间是什么"整体
# 暴露成这一层 Python API。

import importlib
from typing import List

import triton.language.core as tl

__all__ = [
    "address_space",
    "buffer_type",
    "subview",
    "alloc",
    "buffer",
    "to_buffer",
    "to_tensor",
]

TRITON_BUILTIN = "__triton_builtin__"
BUFFER_BUILTIN = "__buffer_builtin__"


def builtin(fn):  # SOURCE: python/triton/extension/buffer/language/core.py:L47-62
    """Mark a function as a buffer language builtin."""
    assert callable(fn)

    def wrapper(*args, **kwargs):  # SOURCE: python/triton/extension/buffer/language/core.py:L52-56
        if "_builder" not in kwargs or kwargs["_builder"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_builder` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    setattr(wrapper, TRITON_BUILTIN, True)
    setattr(wrapper, BUFFER_BUILTIN, True)
    return wrapper


def is_builtin(fn) -> bool:  # SOURCE: python/triton/extension/buffer/language/core.py:L65-67
    """Is this a registered buffer language builtin function?"""
    return getattr(fn, BUFFER_BUILTIN, False)


class address_space:  # SOURCE: python/triton/extension/buffer/language/core.py:L70-79
    """Represents a buffer's address space.

    The :code:`address_space` of a buffer is a target-specific attribute.
    """

    def to_ir(self, builder):  # SOURCE: python/triton/extension/buffer/language/core.py:L76-79
        raise NotImplementedError(
            "Abstract address_space cannot be converted to ir"
        )


# SUBTRACTED: __repr__(转发 __str__)与 __eq__/__ne__ 的孪生实现细节未变，原样保留
# 在下方；仅省略 docstring。
class buffer_type(tl.dtype):  # SOURCE: python/triton/extension/buffer/language/core.py:L82-97

    def __init__(self, element_ty: tl.dtype, shape: List, space: address_space = None,
                 strides: List = None):  # SOURCE: python/triton/extension/buffer/language/core.py:L84-89
        self.element_ty = element_ty
        self.shape = shape if isinstance(shape, list) else list(shape)
        self.space = space
        self.strides = strides if strides is not None else []
        self.name = self._make_name()

    def _make_name(self):  # SOURCE: python/triton/extension/buffer/language/core.py:L91-97
        res = '<buffer ' + 'x'.join(str(s) for s in self.shape) + 'x' + str(self.element_ty)
        if self.strides:
            res += ', strides=[' + ', '.join(str(s) for s in self.strides) + ']'
        if self.space:
            res += ', ' + str(self.space)
        return res + '>'

    def to_ir(self, builder):  # SOURCE: python/triton/extension/buffer/language/core.py:L99-107
        element_ty_ir = self.element_ty.to_ir(builder)
        addr_space_attr = self.space.to_ir(builder) if self.space else builder.get_null_attr()

        # use the method with strides if strides is not empty
        if self.strides:
            return builder.get_buffer_ty_with_strides(self.shape, element_ty_ir, self.strides, addr_space_attr)
        else:
            return builder.get_buffer_ty(self.shape, element_ty_ir, addr_space_attr)

    def __str__(self):  # SOURCE: python/triton/extension/buffer/language/core.py:L109-110
        return self.name

    def __repr__(self):  # SOURCE: python/triton/extension/buffer/language/core.py:L112-113
        return self.__str__()

    def __eq__(self, other) -> bool:  # SOURCE: python/triton/extension/buffer/language/core.py:L115-121
        if not isinstance(other, buffer_type):
            return False
        return (self.element_ty == other.element_ty and
                self.shape == other.shape and
                self.space == other.space and
                self.strides == other.strides)

    def __ne__(self, other) -> bool:  # SOURCE: python/triton/extension/buffer/language/core.py:L123-124
        return not self.__eq__(other)

    @property
    def scalar(self):  # SOURCE: python/triton/extension/buffer/language/core.py:L126-128
        return self.element_ty


# -----------------------
# buffer
# -----------------------


# SUBTRACTED: 类文档字符串(rubric 部分)省略，不影响行为。
class buffer(tl._value):  # SOURCE: python/triton/extension/buffer/language/core.py:L136-152
    """Represents a region of memory.

    :code:`buffer` is the fundamental data structure for Triton programs using
    the buffer language extension.
    """

    def __init__(self, handle, buffer_ty: buffer_type):  # SOURCE: python/triton/extension/buffer/language/core.py:L154-161
        """Not called by user code."""
        super().__init__(handle)
        self.type = buffer_ty
        self.dtype = buffer_ty.element_ty.scalar
        self.shape = buffer_ty.shape
        self.space = buffer_ty.space
        self.strides = buffer_ty.strides

    def __str__(self) -> str:  # SOURCE: python/triton/extension/buffer/language/core.py:L163-169
        # ex. "<16x32xfloat32, address_space>"
        res = '<' + 'x'.join(str(s)
                             for s in self.shape) + 'x' + str(self.dtype)
        if self.space:
            res += ', ' + str(self.space)
        return res + '>'

    @builtin
    def subview(  # SOURCE: python/triton/extension/buffer/language/core.py:L171-179
        self,
        offsets: List[tl.constexpr],
        sizes: List[tl.constexpr],
        strides: List[tl.constexpr],
        _builder=None
    ) -> 'buffer':
        return subview(self, offsets, sizes, strides, _builder=_builder)

    @builtin
    def to_tensor(self, writable=True, target_shape=None,
                  _builder=None):  # SOURCE: python/triton/extension/buffer/language/core.py:L181-184
        """Convert this buffer to a tl.tensor"""
        return to_tensor(self, writable=writable, target_shape=target_shape, _builder=_builder)


# SOURCE: python/triton/extension/buffer/language/core.py:L187
semantic = importlib.import_module(".semantic", package=__package__)


@builtin
def alloc(  # SOURCE: python/triton/extension/buffer/language/core.py:L190-208
    etype: tl.dtype,
    shape: List[tl.constexpr],
    _address_space: address_space = None,
    is_mem_unique: bool = False,
    _builder=None
) -> buffer:
    """
    Allocates a region of local memory with the specified shape and type.

    :param etype: the element type of the buffer.
    :type etype: tl.dtype
    :param shape: A list of non-negative integers representing the shape of the buffer.
    :type shape: List[tl.constexpr]
    :param _address_space: (Optional) backend-specific local memory address space
    :type _address_space: bl.address_space
    """
    return semantic.alloc(etype, shape, _address_space, is_mem_unique, _builder)


@builtin
def to_buffer(  # SOURCE: python/triton/extension/buffer/language/core.py:L211-228
    tensor: tl.tensor,
    space: address_space = None,
    bind_buffer: buffer = None,
    _builder=None
) -> buffer:
    """
    Convert a tensor to a buffer.

    :param tensor: the tensor to convert.
    :type tensor: tl.tensor
    :param space: the address space for the buffer (optional).
    :type space: address_space
    """
    return semantic.to_buffer(
        tensor, space, bind_buffer, _builder
    )


@builtin
def to_tensor(  # SOURCE: python/triton/extension/buffer/language/core.py:L231-246
    memref: buffer,
    writable: bool = True,
    target_shape=None,
    _builder=None
) -> tl.tensor:
    """
    Create a tl.tensor from a bl.buffer.

    :param memref: the input bl.buffer object.
    :memref type: bl.buffer
    :param writable: If set true, the resultant tensor is considered "writable" during bufferization.
    :type writable: bool
    """
    return semantic.to_tensor(memref, writable, _builder, target_shape=target_shape)


# SUBTRACTED（subtraction_plan 批准）：真实 check_subview 的详细文档字符串给了一个
# `memref.subview %arg0[1,1][4,4][2,2]` 的失败案例逐步推导 32-byte 对齐；这段推导
# 与本章内存层级主线无关，故只留一句话说明校验意图，删去逐步推导文本。校验逻辑
# 本身（含真实源码里 length==1 分支引用的 `offset[0]`——形参明明是复数 `offsets`，
# `offset` 在这个函数作用域里从未定义，是上游真实存在的 bug：真实仓库里任何
# rank-1 缓冲调用 subview() 都会在这里炸成 NameError，而不是走到 32-byte 对齐
# 校验；for 循环里 `if isinstance(offsets[i], tl.tensor): return` 同样原样保留——
# 某个 offset 若是运行时张量，静态对齐检查在编译期无法判定，直接放弃校验）
# 逐字原样保留，本章只解读、不"顺手修好"上游的真实缺陷。
def check_subview(src, offsets, sizes, strides):  # SOURCE: python/triton/extension/buffer/language/core.py:L249-296
    """Subview 的 offset/stride 必须 32-byte 对齐、且 stride 全为 1，否则报错。"""
    bytes_per_block = 32
    bits_per_byte = 8
    base_byte = bytes_per_block // (src.dtype.primitive_bitwidth // bits_per_byte)
    result_strides = []
    result_offset = 0
    second_row_start_offset = 0
    length = len(strides)
    src_strides = [1] * length
    if length == 1:
        if offset[0] % base_byte != 0:  # 原样保留：上游真实 bug（见上方说明），`offset` 未定义，触发 NameError
            raise TypeError("all strides should be 1 and the offset value should be 32-bytes aligned.")
        return
    for i in range(length - 2, -1, -1):
        src_strides[i] = src_strides[i + 1] * src.shape[i + 1]
    for i in range(0, length):
        if isinstance(offsets[i], tl.tensor):
            return
        result_strides.append(src_strides[i] * strides[i])
        result_offset = result_offset + offsets[i] * src_strides[i]
    second_row_start_offset = result_offset + src_strides[-2] * strides[-2]
    is_unaligned = False
    if sizes[1] > 1:
        is_unaligned = second_row_start_offset % base_byte != 0
    stride_1 = all(s == 1 for s in strides)
    is_unaligned = result_offset % base_byte != 0 or is_unaligned or not stride_1
    if is_unaligned:
        raise TypeError("all strides should be 1 and the offset value should be 32-bytes aligned.")


# SUBTRACTED（subtraction_plan 批准）：真实实现在校验前还会把 offsets/sizes/strides
# 里的裸 int 逐个转成 tl.constexpr，并把每个 offset 额外转成 tl.tensor（经
# triton.language.semantic.to_tensor，让 offset 也能是运行时张量、而不只是编译期
# 常量）。这层类型规整与"偏移可以是运行时张量"的支持，属于 buffer↔tensor 桥
# （M5）的次要细节而非本章内存层级主线，dossier 批准删除；本精简版要求调用方直接
# 传纯 int（或已经是 int 的 tl.constexpr），subview 本身的骨架——校验 offset 非负、
# 调用 check_subview、再转发到 semantic.subview——原样保留。
@builtin
def subview(  # SOURCE: python/triton/extension/buffer/language/core.py:L299-363
    src: buffer,
    offsets: List[tl.constexpr],
    sizes: List[tl.constexpr],
    strides: List[tl.constexpr],
    _builder=None
) -> buffer:
    '''Creates a subview of the source buffer with the specified offsets, sizes, and strides.'''
    sizes = [tl._constexpr_to_value(s) for s in sizes]
    strides = [tl._constexpr_to_value(s) for s in strides]

    checked_offsets = []
    for offset in offsets:
        offset = tl._constexpr_to_value(offset)
        if offset < 0:
            raise ValueError(f"Offset value must be non-negative, got {offset}")
        checked_offsets.append(offset)

    check_subview(src, checked_offsets, sizes, strides)
    return semantic.subview(src, checked_offsets, sizes, strides, _builder)
