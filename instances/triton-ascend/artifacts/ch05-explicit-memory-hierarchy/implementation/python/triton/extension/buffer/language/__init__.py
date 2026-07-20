# SOURCE: python/triton/extension/buffer/language/__init__.py（全文件，未删——本身
# 只是重导出清单，已按 subtraction_plan 与本章保留的符号一致）
__all__ = [
    "builtin",
    "is_builtin",
    "buffer",
    "address_space",
    "buffer_type",
    "alloc",
    "to_buffer",
    "to_tensor",
    "subview",
]

from .core import (
    builtin, is_builtin, address_space, buffer_type, buffer, alloc, to_buffer, to_tensor, subview,
)
