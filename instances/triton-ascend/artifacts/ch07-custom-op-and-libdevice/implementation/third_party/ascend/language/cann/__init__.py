# SOURCE: third_party/ascend/language/cann/__init__.py
# cann 包组装——本章机制 m8:libdevice 命名空间在 import 期被动态覆盖挂载。libdevice
# 的一部分符号直接复用基座 triton math(如 exp/log/sqrt)，另一部分被昇腾专属实现
# 覆盖(isfinited/finitef/atan2)——用 import 期属性赋值而非重复定义，既复用基座又
# 插入昇腾差异。`triton_enable_libdevice_simt()` 为真时基座已有原生 SIMT atan2 实现，
# 不需要 math_ops.atan2 覆盖；为假时才覆盖——这条 if 正是 m8 的核心分支。
#
# SUBTRACTED:
#  - `extension.parallel = extension.aux_ops.parallel` 与 `libdevice.flip =
#    extension.flip`(third_party/ascend/language/cann/__init__.py:L27,L32)——分别
#    挂载 aux_ops(并行原语)与 vec_ops(flip)，两族算子都不在本章 code_spine 内
#    (各归各自章节)，本章精简版的 extension 包也未重导出它们(见 extension/
#    __init__.py 的 SUBTRACTED 说明)，故这两行一并省略，不改变 m8 覆盖机制本身。
#  - `libdevice.umulhi/exp/exp2/log/log2/cos/sin/sqrt_rn/rsqrt/div_rn/erf/floor/
#    ceil/fdiv/fma`共 15 行(third_party/ascend/language/cann/__init__.py:L34-50)——
#    与保留的 `libdevice.sqrt`/`libdevice.abs` 是同一"`libdevice.X = math.X`一行
#    复用基座同名函数"模式的重复实例，每行相互独立、无控制流关联，删除不影响 m8
#    "覆盖 vs 复用"这条分支主线。

from triton.language import math
from triton.backends.ascend.utils import triton_enable_libdevice_simt

from . import libdevice
from . import extension

if not triton_enable_libdevice_simt():
    libdevice.atan2 = extension.math_ops.atan2
libdevice.isfinited = extension.math_ops.isfinited
libdevice.finitef = extension.math_ops.finitef

libdevice.sqrt = math.sqrt  # 代表：直接复用基座 triton.language.math 的同名实现
libdevice.abs = math.abs

__all__ = ["libdevice", "extension"]
