# 基座 CodeGenerator 的 with-分派缝——本章只留两件事(subtraction_plan.delete 批准
# 删除"除 WITH_DISPATCH 建表/update 与 visit_With 之外的全部内容"):
#   1. 模块级 WITH_DISPATCH 表 + 无条件 import ASCEND_WITH_DISPATCH 并 update
#   2. visit_With 的查表分发
# must_keep 额外要求保留 enter_sub_region——它是 handle_scope_with 借用的作用域
# 进出工具(liveins 的来源、两趟 visit 之间的符号表切换)，删了 handle_scope_with 就
# 无法运行；同理 set_value/_get_insertion_point_and_loc/_set_insertion_point_and_loc/
# visit_compound_statement 是 handle_scope_with 直接调用的 generator 方法，属于
# "运行 must_keep 符号所必需的最小骨架"，不是本章之外内容的"加法"。
#
# SUBTRACTED: 真实 CodeGenerator.__init__(python/triton/compiler/code_generator.py:
# L210-271)还构造 self.gscope 的归一化、self.ascend_builder/self.buffer_builder 双/三
# builder 与 setup_unified_builder 挂接、attributes/constants/function_name/is_kernel/
# cur_node/noinline/scf_stack/dereference_name 等前端通用状态——那是 ch04《双 builder
# 与 Ascend 内建的分发路由》的机制主线(m1)，本章不重复讲，只保留 handle_scope_with
# 实际读写的 builder/lscope/local_defs 三个字段。
# 同理 visit_compound_statement 真实还会在 stmt 是 ast.Return 时提前 break
# (python/triton/compiler/code_generator.py:L370-380)——这一分支本章的 scope 测试
# body 里从不出现 return 语句，但按"只删批准项"原则，源码里存在的这条分支原样保留
# (不算"加法"，只是没被本章任何测试路径触发)。
import ast
from typing import Any, Callable, Dict, Optional, Union

from triton.language.core import _value, constexpr, tensor


def _is_list_like(o: Any) -> bool:  # SOURCE: python/triton/compiler/code_generator.py:L82-83
    return isinstance(o, (list, tuple))


def _is_triton_value(o: Any) -> bool:  # SOURCE: python/triton/compiler/code_generator.py:L66-67
    return isinstance(o, _value)


def _is_triton_tensor(o: Any) -> bool:  # SOURCE: python/triton/compiler/code_generator.py:L70-71
    return isinstance(o, tensor)


# SOURCE: python/triton/compiler/code_generator.py:L25-31
# Central registry for all 'with' statement handlers
WITH_DISPATCH = {}

# Import and register Ascend extension dispatch handlers
from triton.language.extra.cann.extension.dispatch import ASCEND_WITH_DISPATCH
WITH_DISPATCH.update(ASCEND_WITH_DISPATCH)


# must_keep：liveins 的来源，handle_scope_with 借它暂存/恢复 lscope、local_defs 与
# builder 的插入点。
class enter_sub_region:  # SOURCE: python/triton/compiler/code_generator.py:L99-116

    def __init__(self, generator):  # SOURCE: python/triton/compiler/code_generator.py:L101-102
        self.generator = generator

    def __enter__(self):  # SOURCE: python/triton/compiler/code_generator.py:L104-111
        # record lscope & local_defs in the parent scope
        self.liveins = self.generator.lscope.copy()
        self.prev_defs = self.generator.local_defs.copy()
        self.generator.local_defs = {}
        self.insert_block = self.generator.builder.get_insertion_block()
        self.insert_point = self.generator.builder.get_insertion_point()
        return self.liveins, self.insert_block

    def __exit__(self, *args, **kwargs):  # SOURCE: python/triton/compiler/code_generator.py:L113-116
        self.generator.builder.restore_insertion_point(self.insert_point)
        self.generator.lscope = self.liveins
        self.generator.local_defs = self.prev_defs


class CodeGenerator:
    """AST → IR 的前端翻译器。

    SUBTRACTED: 真实定义是 `class CodeGenerator(ast.NodeVisitor)`，靠 NodeVisitor 的
    通用 `visit(node)` 按节点类型名分派到数十个 visit_Xxx(见上方文件头说明)。本章精简版
    只留 visit_With/visit_compound_statement 两个必需方法，其余 visit_Xxx 全部未定义，
    故不继承 ast.NodeVisitor；测试里按需给 self.visit 打桩，不经过完整 AST 遍历——这正是
    本章要单独摘出来看的一处接缝(scope 的编译器特判)。
    """

    # SOURCE: python/triton/compiler/code_generator.py:L210-266(节选)
    def __init__(self, builder):
        self.builder = builder
        self.lscope: Dict[str, Any] = {}  # SOURCE: L254
        self.local_defs: Dict[str, tensor] = {}  # SOURCE: L266

    # SOURCE: python/triton/compiler/code_generator.py:L344-351
    def set_value(self, name: str, value: Union[tensor, constexpr]) -> None:
        ''' This function:
            called by visit_Assign() & visit_FunctionDef() to store left value (lvalue)
        1. record local defined name (FIXME: should consider control flow)
        2. store tensor in self.lvalue
        '''
        self.lscope[name] = value
        self.local_defs[name] = value

    # SOURCE: python/triton/compiler/code_generator.py:L353-360
    def _get_insertion_point_and_loc(self, builder=None):
        # XXX: this is a hack to get the location of the insertion point.
        # The insertion point's location could be invalid sometimes,
        # so we need to explicitly set the location
        _builder = self.builder if not builder else builder
        loc = _builder.get_loc()
        ip = _builder.get_insertion_point()
        return ip, loc

    # SOURCE: python/triton/compiler/code_generator.py:L362-365
    def _set_insertion_point_and_loc(self, ip, loc, builder=None):
        _builder = self.builder if not builder else builder
        _builder.restore_insertion_point(ip)
        _builder.set_loc(loc)

    # SOURCE: python/triton/compiler/code_generator.py:L370-380
    def visit_compound_statement(self, stmts):
        # Ensure that stmts is iterable
        if not _is_list_like(stmts):
            stmts = [stmts]
        for stmt in stmts:
            self.visit(stmt)
            # Stop parsing as soon as we hit a `return` statement; everything
            # after this is dead code.
            if isinstance(stmt, ast.Return):
                break

    # SOURCE: python/triton/compiler/code_generator.py:L801-813
    def visit_With(self, node):
        """Handle 'with' statements using dispatch pattern."""
        assert len(node.items) == 1
        context = node.items[0].context_expr

        # Check if context is a Call and dispatch to registered handler
        if isinstance(context, ast.Call):
            withitemClass = self.visit(context.func)
            handler = WITH_DISPATCH.get(withitemClass)
            if handler:
                return handler(self, node)

        # Fall back to visiting body for unhandled cases
        return self.visit_compound_statement(node.body)
