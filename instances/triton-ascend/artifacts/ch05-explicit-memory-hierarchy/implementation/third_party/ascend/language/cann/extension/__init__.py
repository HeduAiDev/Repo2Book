# SOURCE: third_party/ascend/language/cann/extension/__init__.py（节选）
#
# SUBTRACTED（subtraction_plan 批准）：真实 __init__ 还重导出 affine_map 系 MLIR
# 绑定、scope、custom_op、math_ops、aux_ops（含 sync_block_set/wait）、vec_ops、
# mem_ops（index_put/gather_out_to_ub/scatter_ub_to_out/index_select_simd——按已
# 审批大纲归 ch06）等一整套算子入口。本章只挑出内存层级相关的四个名字。
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

from .core import (
    ascend_address_space,
    copy_from_ub_to_l1,
    copy,
    fixpipe,
    FixpipeDMAMode,
    FixpipeDualDstMode,
    FixpipePreQuantMode,
    FixpipePreReluMode,
)
