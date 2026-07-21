# SOURCE: third_party/ascend/language/cann/extension/dispatch.py
"""
Dispatch table for Ascend-specific 'with' statement context managers.
"""

from .scope import scope
from .code_generator import handle_scope_with

__all__ = ["ASCEND_WITH_DISPATCH"]

# SOURCE: third_party/ascend/language/cann/extension/dispatch.py:L31-34
# SUBTRACTED: 真实字典还有 "mangle_ty": mangle_ty 一项——mangle_ty 做的是类型名字修饰，
# 与本章 with/scope 语义无关(subtraction_plan.delete 批准：连同 ascend 侧
# code_generator.py 的 mangle_ty 定义一并删除)，删掉后这张表只留 scope 一项，仍能完整
# 演示 with 特判本身。mangle_ty 的 override 钩子机制归 ch04《双 builder 与 Ascend 内建
# 的分发路由》的题材。
# Registry of 'with' statement handlers for Ascend extension
ASCEND_WITH_DISPATCH = {
    scope: handle_scope_with,
}
