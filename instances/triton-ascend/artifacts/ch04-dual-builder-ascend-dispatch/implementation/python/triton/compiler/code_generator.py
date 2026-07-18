# SOURCE: python/triton/compiler/code_generator.py
#
# 本章精简版只保留六处机制（dossier.mechanisms m1-m6）真正需要的东西：
#   模块级 WITH_DISPATCH 表 + mangle_ty override 钩子（m5）
#   CodeGenerator.__init__ 的双 builder 构造块（m1）
#   visit_Call 的第四岔分发（m2/m3）
#   visit_With 的查表分发（m5）
#   _get/_set_insertion_point_and_loc（m6）
# 其余 1384 行里与这条主线无关的 AST visitor 方法体（visit_For/visit_While/visit_If/
# visit_BinOp/visit_Compare 等）、gscope 归一化/module_map/function_ret_types 等基座
# 通用初始化，按 subtraction_plan.delete 批准整体删除，不在本文件出现。
import inspect
from typing import Any, Dict, Optional

import triton.language.extra.cann.extension as extension
from triton import language
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.language.core import _unwrap_if_constexpr, _value
from triton.runtime import JITFunction
from triton.compiler.errors import CompilationError

# SOURCE: python/triton/compiler/code_generator.py:L25-31
# Central registry for all 'with' statement handlers
WITH_DISPATCH = {}

# Import and register Ascend extension dispatch handlers
from triton.language.extra.cann.extension.dispatch import ASCEND_WITH_DISPATCH
from triton.language.extra.cann.extension.builder import setup_unified_builder
WITH_DISPATCH.update(ASCEND_WITH_DISPATCH)


# SOURCE: python/triton/compiler/code_generator.py:L34-49
def mangle_ty(ty):
    if ty.is_ptr():
        return 'P' + mangle_ty(ty.element_ty)
    if ty.is_int():
        SIGNED = language.dtype.SIGNEDNESS.SIGNED
        prefix = 'i' if ty.int_signedness == SIGNED else 'u'
        return prefix + str(ty.int_bitwidth)
    if ty.is_floating():
        return str(ty)
    if ty.is_block():
        elt = mangle_ty(ty.scalar)
        shape = '_'.join(map(str, ty.shape))
        return f'{elt}S{shape}S'
    if ty.is_void():
        return 'V'
    raise TypeError(f'Unsupported type {ty}')


# SOURCE: python/triton/compiler/code_generator.py:L51 —— fork 的 override 钩子：
# ASCEND_WITH_DISPATCH 里 "mangle_ty" 键指向 ascend 版 mangle_ty（third_party/ascend/
# language/cann/extension/code_generator.py），若命中就整体替换掉上面这个基座实现。
mangle_ty = WITH_DISPATCH.get("mangle_ty", mangle_ty)


# SOURCE: python/triton/compiler/code_generator.py:L82-83
def _is_list_like(o: Any) -> bool:
    return isinstance(o, (list, tuple))


# SOURCE: python/triton/compiler/code_generator.py:L66-67
def _is_triton_value(o: Any) -> bool:
    return isinstance(o, _value)


# SOURCE: python/triton/compiler/code_generator.py:L99-116
class enter_sub_region:
    """进出一个子作用域（if/while/scope 等）时暂存并恢复 lscope/local_defs/插入点。
    handle_scope_with（third_party/ascend/.../extension/code_generator.py）用它来
    界定 with al.scope(...) 块自己的符号表。"""

    def __init__(self, generator):
        # SOURCE: python/triton/compiler/code_generator.py:L101-102
        self.generator = generator

    def __enter__(self):
        # SOURCE: python/triton/compiler/code_generator.py:L104-111
        self.liveins = self.generator.lscope.copy()
        self.prev_defs = self.generator.local_defs.copy()
        self.generator.local_defs = {}
        self.insert_block = self.generator.builder.get_insertion_block()
        self.insert_point = self.generator.builder.get_insertion_point()
        return self.liveins, self.insert_block

    def __exit__(self, *args, **kwargs):
        # SOURCE: python/triton/compiler/code_generator.py:L113-116
        self.generator.builder.restore_insertion_point(self.insert_point)
        self.generator.lscope = self.liveins
        self.generator.local_defs = self.prev_defs


class CodeGenerator:
    """AST → IR 的前端翻译器。

    SUBTRACTED: 真实定义是 `class CodeGenerator(ast.NodeVisitor)`，靠 NodeVisitor 的
    通用 `visit(node)` 按节点类型名分派到 visit_Xxx。本章精简版只保留 visit_Call/
    visit_With 两个必需的 visit_Xxx 方法，其余全部未定义，故这里不继承
    ast.NodeVisitor（继承了也用不上其分派机制），测试直接调用 visit_Call/visit_With
    本身，不经过完整 AST 遍历——这正是本章要单独摘出来看的两处接缝。
    """

    # SOURCE: python/triton/compiler/code_generator.py:L210-271
    def __init__(self, context, prototype, gscope, attributes, constants, function_name, jit_fn: JITFunction,
                 options, codegen_fns, module_map, module=None, is_kernel=False,
                 function_types: Optional[Dict] = None, noinline=False, file_name: Optional[str] = None,
                 begin_line=0):
        self.context = context
        # Only NPUOptions has force_simt_only attribute, so check for NPU backend
        if hasattr(options, "force_simt_only") and options.force_simt_only:
            self.builder = ir.builder(context, compile_mode="simt")
        else:
            self.builder = ir.builder(context, compile_mode="simd")
        self.file_name = file_name
        # node.lineno starts from 1, so we need to subtract 1
        self.begin_line = begin_line - 1
        self.builder.set_loc(file_name, begin_line, 0)
        self.builder.options = options

        # Set up unified builder interface with methods from specialized builders
        self.ascend_builder = ascend_ir.ascendnpu_ir_builder(context, getattr(options, "arch", ""))
        self.ascend_builder.set_loc(file_name, begin_line, 0)
        setup_unified_builder(self.builder, self.ascend_builder)
        # SUBTRACTED: self.buffer_builder = buffer_ir.buffer_builder(context) +
        # setup_unified_builder_with_buffer_builder(self.builder, self.buffer_builder)——
        # 又一套 buffer 语言的 builder（dossier embed_excerpts 原话「归别处」），与本章
        # 双 builder(self.builder/self.ascend_builder)这一对无关。原：L229-231。

        self.builder.codegen_fns = codegen_fns
        self.builder.module_map = {} if module_map is None else module_map
        self.module = self.builder.create_module() if module is None else module
        self.jit_fn = jit_fn
        # SUBTRACTED: function_ret_types/prototype 记录、gscope 按 ModuleType/module_map
        # 重定向的归一化循环、attributes/constants/function_name/is_kernel/cur_node/
        # noinline/scf_stack/ret_type/dereference_name/fn/visiting_arg_default_value 等
        # 基座前端通用状态——与『第二个 builder 怎么建、怎么被 setup_unified_builder
        # 挂方法』这条主线无关，删除不影响本章机制的可运行性。只保留 visit_Call/
        # visit_With/handle_scope_with 实际读写的 lscope/local_defs。
        # 原：python/triton/compiler/code_generator.py:L233-271。
        self.lscope = {}
        self.local_defs: Dict[str, Any] = {}

    # SOURCE: python/triton/compiler/code_generator.py:L273-278
    # SUBTRACTED: 真实还有 `.update((('print', language.core.device_print),
    # ('min', language.minimum), ('max', language.maximum)))`，把 triton 语言层的
    # print/min/max 接进来；这三个是 tl.* 语言层的具体实现，与本章双 builder 分发
    # 无关，故只保留最前面这一行纯 Python 内建函数集合。
    builtin_namespace: Dict[str, Any] = {_.__name__: _ for _ in (len, list, range, float, int, isinstance, getattr)}

    # SOURCE: python/triton/compiler/code_generator.py:L1332-1338
    # SUBTRACTED: 真实字典里还有 4 项常量折叠条目
    # {language.core.static_assert: execute_static_assert, language.core.static_print:
    # static_executor(print), int: static_executor(int), len: static_executor(len),
    # extension.int64: static_executor(extension.int64)}，连同支撑它们的
    # execute_static_assert/static_executor 两个方法。这是 visit_Call 的『第①岔』
    # ——编译期常量折叠，dossier 理论小节明确称其与本章『第③/④岔』（builtin 分发/
    # ascend_builder 选路）正交，故整体清空，只保留同名空字典使 visit_Call 首行
    # `self.statically_implemented_functions.get(fn)` 结构不变、恒为 None（真实行为
    # 对『非常量折叠函数』本就是 None，故此简化不改变本章测试路径的可观察行为）。
    statically_implemented_functions: Dict[object, Any] = {}

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

    def set_value(self, name, value):
        # SOURCE: python/triton/compiler/code_generator.py:L344-351
        self.lscope[name] = value
        self.local_defs[name] = value

    # SOURCE: python/triton/compiler/code_generator.py:L370-380
    def visit_compound_statement(self, stmts):
        # Ensure that stmts is iterable
        if not _is_list_like(stmts):
            stmts = [stmts]
        for stmt in stmts:
            self.visit(stmt)
            # Stop parsing as soon as we hit a `return` statement; everything
            # after this is dead code.
            import ast
            if isinstance(stmt, ast.Return):
                break

    # SOURCE: python/triton/compiler/code_generator.py:L801-814 —— fork 新增
    # （基座没有 visit_With；with 语句非 Triton IR 关注点）。
    def visit_With(self, node):
        """Handle 'with' statements using dispatch pattern."""
        # SOURCE: python/triton/compiler/code_generator.py:L801-814 —— fork 新增
        # （基座没有 visit_With；with 语句非 Triton IR 关注点）。
        import ast
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

    # SOURCE: python/triton/compiler/code_generator.py:L1168-1206 —— 核心：
    # fork 在统一 builtin 入口门之后加了『第四岔』选 builder 这一行（L1183）。
    def visit_Call(self, node):
        # SOURCE: python/triton/compiler/code_generator.py:L1168-1206 —— 核心：
        # fork 在统一 builtin 入口门之后加了『第四岔』选 builder 这一行（L1183）。
        fn = _unwrap_if_constexpr(self.visit(node.func))
        static_implementation = self.statically_implemented_functions.get(fn)
        if static_implementation is not None:
            return static_implementation(self, node)

        kws = dict(self.visit(keyword) for keyword in node.keywords)
        args = [self.visit(arg) for arg in node.args]
        if isinstance(fn, JITFunction):
            # NOTE（非源码，说明用）：真实分支这里还有 `_check_fn_args(node, fn, args)`
            # + `return self.call_JitFunction(fn, args, kws)`——递归展开 @triton.jit
            # 组合子（dossier 理论小节的『第②岔』），与本章『第③/④岔』正交，属另一批
            # 章节讲解范围，本精简版不复现其实现；本章的测试与叙事也从不构造
            # JITFunction 实例，故这条分支在本章语境下结构存在但从不被走到。
            _check_fn_args(node, fn, args)
            return self.call_JitFunction(fn, args, kws)
        if (hasattr(fn, '__self__') and _is_triton_value(fn.__self__)) or language.core.is_builtin(fn):
            # Copy builder's location and insertion point.
            ip, last_loc = self._get_insertion_point_and_loc()
            # Use ascend_builder if this function is a builtin extension operation.
            _builder = self.ascend_builder if extension.is_builtin(fn) else self.builder
            self._set_insertion_point_and_loc(ip, last_loc, _builder)
            extra_kwargs = {"_builder": _builder}
            sig = inspect.signature(fn)
            if '_generator' in sig.parameters:
                extra_kwargs['_generator'] = self
            try:
                ret = fn(*args, **extra_kwargs, **kws)
                # Sync the builder's location before return.
                ip, last_loc = self._get_insertion_point_and_loc(_builder)
                self._set_insertion_point_and_loc(ip, last_loc)
                return ret
            except Exception as e:
                # Normally when we raise a CompilationError, we raise it as
                # `from None`, because the original fileline from the exception
                # is not relevant (and often points into code_generator.py
                # itself).  But when calling a function, we raise as `from e` to
                # preserve the traceback of the original error, which may e.g.
                # be in core.py.
                raise CompilationError(self.jit_fn.src, node, repr(e)) from e

        if fn in self.builtin_namespace.values():
            args = map(_unwrap_if_constexpr, args)
        return fn(*args, **kws)
