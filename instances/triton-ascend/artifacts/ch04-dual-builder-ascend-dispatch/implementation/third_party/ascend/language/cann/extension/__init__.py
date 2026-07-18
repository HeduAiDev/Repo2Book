# SOURCE: third_party/ascend/language/cann/extension/__init__.py
# SUBTRACTED（subtraction_plan.delete 批准范围内）：真实 __init__ 还聚合导出
# custom_op（custom/custom_semantic/register_custom_op，自定义算子注册——另一套机制）、
# builtin_custom_ops、math_ops（atan2/isfinited/finitef）、aux_ops、mem_ops、vec_ops
# 等一整批具体 hivm 算子的 Python 入口，以及 affine_expr/AffineMap 等 MLIR affine
# 绑定的重导出。这些是 hivm 方言算子的语义目录（归 Part 5），与本章『@builtin 打双
# 标记 + is_builtin 读标记 + with scope 分发』这条主线无关。这里只重导出本章用到的
# builtin/is_builtin/sub_vec_id/scope 四个名字，使得
# `import triton.language.extra.cann.extension as extension` 之后
# `extension.is_builtin(fn)`（visit_Call 第四岔的选路谓词）能解析到。
from .core import builtin, is_builtin, sub_vec_id
from .scope import scope

__all__ = ["builtin", "is_builtin", "sub_vec_id", "scope"]
