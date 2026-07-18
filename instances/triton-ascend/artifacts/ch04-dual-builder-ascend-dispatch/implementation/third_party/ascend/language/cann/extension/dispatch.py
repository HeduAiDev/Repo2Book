# SOURCE: third_party/ascend/language/cann/extension/dispatch.py（全文件，未删减）
"""
Dispatch table for Ascend-specific 'with' statement context managers.
"""

from .scope import scope
from .code_generator import handle_scope_with, mangle_ty

__all__ = ["ASCEND_WITH_DISPATCH"]

# Registry of 'with' statement handlers for Ascend extension
ASCEND_WITH_DISPATCH = {
    scope: handle_scope_with,
    "mangle_ty": mangle_ty,
}
