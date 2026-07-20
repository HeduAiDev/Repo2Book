# SOURCE: third_party/ascend/language/cann/extension/core.py（节选，行号见各符号
# 上方）
#
# SUBTRACTED（subtraction_plan.delete 批准，第 1 项）：本文件真实还定义 CORE/PIPE/
# MODE/IteratorType 四个属核类型/流水线/迭代器枚举、sync_block_set/wait/all 与
# create_sync_block（核间同步）、sub_vec_id/sub_vec_num（子核索引/数量查询）、
# debug_barrier、SYNC_IN_VF 枚举、int64 辅助类型，以及本文件自己的 @builtin/
# is_builtin 双标记装饰器机制（TRITON_BUILTIN/ASCEND_BUILTIN 两个标记 + wrapper）。
# 这些属于 ch08（scope/同步/流水线）与 ch04（双 builder 的 @builtin 契约，已在该章
# 讲清"打双标记→CodeGenerator.visit_Call 按标记路由到 ascend_builder"这件事），与
# 本章「内存层级对程序员可见 + 显式 copy/fixpipe」这条主线正交，删除不影响本章
# 可运行演示。因此本章的 copy_from_ub_to_l1/copy/fixpipe 这里不再套 @builtin
# 装饰器——只是少了"校验 _builder 已传入 + 打双标记"这层包装，函数体与真实源码
# 逐字一致。
# SUBTRACTED（第 4 项）：真实文件 L61 还有一行 `PIPE = semantic.PIPE`——在 L111 被
# 同名 `class PIPE` 覆盖前的死赋值，随 PIPE 一并删除。

import enum
from typing import Union

from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
import triton.language.core as tl

import triton.extension.buffer.language as bl

from . import semantic as semantic

__all__ = [
    "ascend_address_space",
    "copy_from_ub_to_l1",
    "copy",
    "fixpipe",
    "FixpipeDMAMode",
    "FixpipeDualDstMode",
    "FixpipePreQuantMode",
    "FixpipePreReluMode",
]


class ascend_address_space_base(bl.address_space):  # SOURCE: third_party/ascend/language/cann/extension/core.py:L143-149
    def __init__(self, address_space_value: ascend_ir.AddressSpace) -> None:  # SOURCE: third_party/ascend/language/cann/extension/core.py:L144-146
        super().__init__()
        self.real_address_space = address_space_value

    def to_ir(self, builder: ir.builder) -> ir.attribute:  # SOURCE: third_party/ascend/language/cann/extension/core.py:L148-149
        return builder.get_target_attribute(self.real_address_space)


class ascend_address_space_group:  # SOURCE: third_party/ascend/language/cann/extension/core.py:L152-163

    def __init__(self):  # SOURCE: third_party/ascend/language/cann/extension/core.py:L154-160
        for k, v in {
            k: v
            for k, v in ascend_ir.AddressSpace.__dict__.items()
            if isinstance(v, ascend_ir.AddressSpace)
        }.items():
            setattr(self, k, ascend_address_space_base(v))


ascend_address_space = ascend_address_space_group()


# SOURCE: third_party/ascend/language/cann/extension/core.py:L174-186
def copy_from_ub_to_l1(src: Union[tl.tensor, bl.buffer], dst: Union[tl.tensor, bl.buffer], _builder=None) -> None:
    """
    Copies data from the Unified Buffer (UB) to the L1 Buffer.

    :param src: The source data located in the Unified Buffer.
    :type src: tl.tensor | bl.buffer
    :param dst: The destination buffer located in L1 memory.
    :type dst: tl.tensor | bl.buffer
    """
    from warnings import warn
    warn("copy_from_ub_to_l1 is deprecated, please use copy instead.")
    return semantic.copy_from_ub_to_l1(src, dst, _builder)


# SOURCE: third_party/ascend/language/cann/extension/core.py:L189-199
def copy(src: Union[tl.tensor, bl.buffer], dst: Union[tl.tensor, bl.buffer], _builder=None) -> None:
    """
    Copies data from the Unified Buffer (UB) to the Unified Buffer (UB) or L1 Buffer.

    :param src: The source data located in the Unified Buffer.
    :type src: tl.tensor | bl.buffer
    :param dst: The destination buffer located Unified Buffer (UB) or L1 memory.
    :type dst: tl.tensor | bl.buffer
    """
    return semantic.copy(src, dst, _builder)


# SOURCE: third_party/ascend/language/cann/extension/core.py:L247-270
class FixpipeDMAMode(enum.Enum):
    NZ2DN = ascend_ir.FixpipeDMAMode.NZ2DN
    NZ2ND = ascend_ir.FixpipeDMAMode.NZ2ND
    NZ2NZ = ascend_ir.FixpipeDMAMode.NZ2NZ


class FixpipeDualDstMode(enum.Enum):  # SOURCE: third_party/ascend/language/cann/extension/core.py:L253-256
    NO_DUAL = ascend_ir.FixpipeDualDstMode.NO_DUAL
    COLUMN_SPLIT = ascend_ir.FixpipeDualDstMode.COLUMN_SPLIT
    ROW_SPLIT = ascend_ir.FixpipeDualDstMode.ROW_SPLIT


class FixpipePreQuantMode(enum.Enum):  # SOURCE: third_party/ascend/language/cann/extension/core.py:L259-263
    NO_QUANT = ascend_ir.FixpipePreQuantMode.NO_QUANT
    F322BF16 = ascend_ir.FixpipePreQuantMode.F322BF16
    F322F16 = ascend_ir.FixpipePreQuantMode.F322F16
    S322I8 = ascend_ir.FixpipePreQuantMode.S322I8


class FixpipePreReluMode(enum.Enum):  # SOURCE: third_party/ascend/language/cann/extension/core.py:L266-270
    LEAKY_RELU = ascend_ir.FixpipePreReluMode.LEAKY_RELU
    NO_RELU = ascend_ir.FixpipePreReluMode.NO_RELU
    NORMAL_RELU = ascend_ir.FixpipePreReluMode.NORMAL_RELU
    P_RELU = ascend_ir.FixpipePreReluMode.P_RELU


# SUBTRACTED: 前端固定传 NO_QUANT/NO_RELU——量化/ReLU 融合枚举在语言层未开放，只保留
# 占位（同真实源码，未做删减）。
def fixpipe(  # SOURCE: third_party/ascend/language/cann/extension/core.py:L273-333
    src: tl.tensor,
    dst: bl.buffer,
    dma_mode: FixpipeDMAMode = FixpipeDMAMode.NZ2ND,
    dual_dst_mode: FixpipeDualDstMode = FixpipeDualDstMode.NO_DUAL,
    _builder=None,
) -> None:
    """
    Directly store a tensor on L0C to a local buffer via fixpipe.
    Fixpipe is pipeline that performing data movement from L0C to other memory hierarchies.
    Currently support:
        - L0C to UB (for Ascend910_95 sereies)

    :param src: the source tensor, Must be located in the l0C memory region.
    :type src: tl.tensor
    :param dst: The destination buffer, Must be located in the UB memory region.
    :type dst: bl.buffer
    :param dma_mode: DMA transfer mode, "nz2nd" enables NZ to ND layout transformation
    :type dma_mode: str
    """
    if not _builder.is_910_95():
        raise RuntimeError("this feature is only supported on Ascend910_95")
    if not isinstance(src, tl.tensor):
        raise TypeError("src is not of tensor type")
    elif not isinstance(dst, bl.buffer):
        raise TypeError("dst is not of buffer type")
    if dst.space != ascend_address_space.UB:
        raise TypeError("dst must be located in the UB memory region")

    if len(dst.shape) == 2 and (
        dst.type.element_ty == tl.float32 or dst.type.element_ty == tl.int32
    ):
        N = dst.shape[1]
        if N % 8 != 0:
            raise ValueError("32b Fixpipe last dim must be aligned to 8")
        if (dma_mode != FixpipeDMAMode.NZ2ND) and (N % 16 != 0):
            raise ValueError("32b non-NZ2ND Fixpipe last dim must be aligned to 16")
        if (dual_dst_mode == FixpipeDualDstMode.COLUMN_SPLIT) and (N % 32 != 0):
            raise ValueError(
                "32b Column split dual Fixpipe last dim must be aligned to 32"
            )
        M = dst.shape[0]
        if (dma_mode == FixpipeDMAMode.NZ2DN) and (M % 8 != 0):
            raise ValueError("32b NZ2DN Fixpipe first dim must be aligned to 8")
    dst16bits = (
        dst.type.element_ty == tl.float16
        or dst.type.element_ty == tl.int16
        or dst.type.element_ty == tl.bfloat16
    )
    if len(dst.shape) == 2 and dst16bits:
        N = dst.shape[1]
        if N % 16 != 0:
            raise ValueError("16b Fixpipe last dim must be aligned to 16")
        M = dst.shape[0]
        if (dma_mode == FixpipeDMAMode.NZ2DN) and (M % 16 != 0):
            raise ValueError("16b NZ2DN Fixpipe first dim must be aligned to 16")

    return semantic.fixpipe(
        src, dst, dma_mode, dual_dst_mode, FixpipePreQuantMode.NO_QUANT, FixpipePreReluMode.NO_RELU, _builder
    )
