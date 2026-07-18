"""m5 —— WITH_DISPATCH 注册表：`with al.scope(...)` 的查表分发 + mangle_ty 的
override 钩子，两类扩展点收敛在同一张全局字典里。

对照真实源码：
    python/triton/compiler/code_generator.py:L25-31
        WITH_DISPATCH = {}
        WITH_DISPATCH.update(ASCEND_WITH_DISPATCH)
    python/triton/compiler/code_generator.py:L51
        mangle_ty = WITH_DISPATCH.get("mangle_ty", mangle_ty)
    python/triton/compiler/code_generator.py:L801-814  visit_With
    third_party/ascend/language/cann/extension/dispatch.py
        ASCEND_WITH_DISPATCH = {scope: handle_scope_with, "mangle_ty": mangle_ty}
"""
import ast
import types

from conftest import make_generator


def test_ascend_dispatch_entries_are_merged_into_with_dispatch(env):
    WITH_DISPATCH = env.code_generator.WITH_DISPATCH
    scope_cls = env.ext_scope.scope

    assert WITH_DISPATCH[scope_cls] is env.ext_codegen.handle_scope_with
    assert WITH_DISPATCH["mangle_ty"] is env.ext_codegen.mangle_ty


def test_module_level_mangle_ty_is_overridden_by_ascend_version(env):
    # code_generator.py:L51 用 WITH_DISPATCH.get("mangle_ty", 基座版) 整体替换掉了
    # 模块里刚定义的那个基座 mangle_ty——最终留在模块命名空间里的必须是 ascend 版本。
    assert env.code_generator.mangle_ty is env.ext_codegen.mangle_ty


def test_visit_with_dispatches_via_table_to_registered_handler(env):
    """visit_With 自己的机制：context 是 Call 且其 func 命中 WITH_DISPATCH 的键，
    就把 (self, node) 转交注册的 handler——这里换一个哨兵 handler，把
    『visit_With 怎么查表分发』和『handle_scope_with 内部具体做了什么』分开测。"""
    gen = make_generator(env)
    scope_cls = env.ext_scope.scope

    seen = {}
    original_handler = env.code_generator.WITH_DISPATCH[scope_cls]
    try:
        env.code_generator.WITH_DISPATCH[scope_cls] = lambda g, n: seen.setdefault("called", (g, n))

        gen.visit = lambda node: node.func if hasattr(node, "func") else node
        node = types.SimpleNamespace(
            items=[types.SimpleNamespace(
                context_expr=types.SimpleNamespace(func=scope_cls))],
            body=[],
        )
        # 让 isinstance(context, ast.Call) 成立：context_expr 本身要是个 ast.Call。
        node.items[0].context_expr = ast.Call(func=scope_cls, args=[], keywords=[])

        result = gen.visit_With(node)
        assert seen["called"] == (gen, node)
    finally:
        env.code_generator.WITH_DISPATCH[scope_cls] = original_handler


def test_visit_with_falls_back_when_no_handler_registered(env):
    gen = make_generator(env)
    calls = []
    gen.visit_compound_statement = lambda body: calls.append(body)
    gen.visit = lambda node: "not-a-registered-key"

    node = types.SimpleNamespace(
        items=[types.SimpleNamespace(context_expr=ast.Call(func=object(), args=[], keywords=[]))],
        body=["stmt-marker"],
    )
    gen.visit_With(node)
    assert calls == [["stmt-marker"]]


def test_handle_scope_with_lands_on_create_scope_op_and_scope_return(env):
    """整章要证明的落地关系：with al.scope(core_mode='vector'): ... 最终经
    handle_scope_with 调用挂在主 builder 上的 create_scope_op / scope_return
    （setup_unified_builder 挂上去的 wrapper，真正执行在 ascend_builder 上）。"""
    gen = make_generator(env)

    class FakeType:
        def to_ir(self, builder):
            return "ty-ir"

    def visit(node):
        if node == "ASSIGN_X":
            gen.set_value("x", env.core.tensor("x-handle", FakeType()))
            return None
        return node

    gen.visit = visit

    scope_call = ast.Call(
        func=env.ext_scope.scope,
        args=[],
        keywords=[ast.keyword(arg="core_mode", value=ast.Constant(value="vector"))],
    )
    node = types.SimpleNamespace(
        items=[types.SimpleNamespace(context_expr=scope_call)],
        body=["ASSIGN_X"],
    )

    handle_scope_with = env.code_generator.WITH_DISPATCH[env.ext_scope.scope]
    handle_scope_with(gen, node)

    ascend_calls = [c[0] for c in gen.ascend_builder.calls]
    assert "create_scope_op" in ascend_calls
    assert "scope_return" in ascend_calls

    create_call = next(c for c in gen.ascend_builder.calls if c[0] == "create_scope_op")
    attrs = create_call[1]
    # core_mode='vector' 被转换成了 t_core_type 属性（handle_scope_with 里演示的那一条
    # 属性路径）。
    assert attrs[gen.builder.get_t_core_type_attr_name()] == gen.builder.get_t_core_type_vector_attr()

    # scope 结束后，x 被重建回 lscope（用 scope_op 的 get_result handle）。
    assert gen.lscope["x"].handle == "scope-result-0"
