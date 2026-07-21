# SOURCE: third_party/ascend/language/cann/extension/__init__.py
# 包组装——只重导出本章需要的名字。
#
# SUBTRACTED: 真实文件还从 .scope(ch08 讲)、.aux_ops/.vec_ops/.mem_ops(其余算子族，
# 各归各自章节)重导出 parallel/compile_hint/multibuffer/insert_slice/flip/
# index_put/... 等约 20 个名字，以及 MLIR affine 绑定(affine_map/AffineExpr/...，
# indexing_map 语义归 P5)、.core 里的 copy/fixpipe/sync_block_*/builtin/is_builtin
# (ch04/ch05 已讲)。本章只圈定 register_custom_op 自定义算子框架 + math_ops 数学库，
# 故只重导出这些。原文件：third_party/ascend/language/cann/extension/__init__.py
# (约 156 行)。

from .core import CORE, PIPE, MODE, int64

from .custom_op import (
    custom,
    custom_semantic,
    register_custom_op,
)

from . import builtin_custom_ops
from . import math_ops
from .math_ops import atan2, isfinited, finitef

__all__ = [
    "CORE", "PIPE", "MODE", "int64",
    "custom", "custom_semantic", "register_custom_op",
    "builtin_custom_ops", "math_ops",
    "atan2", "isfinited", "finitef",
]
