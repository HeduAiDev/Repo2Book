# 精简版 extension 包的最小 __init__——真实文件还重导出 affine_*/scope/custom_op/
# math_ops/parallel/compile_hint/multibuffer 等一整套符号（third_party/ascend/
# language/cann/extension/__init__.py:L1-90），本章只保留 mem_ops/vec_ops 两个
# 词汇表模块与它们唯一的跨文件依赖 is_compile_on_910_95。
#
# SOURCE: third_party/ascend/language/cann/extension/__init__.py:L1（is_compile_on_910_95 一行）

from triton.tools.get_ascend_devices import is_compile_on_910_95  # noqa: F401
