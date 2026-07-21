"""M1 —— with 语句被编译器特判：WITH_DISPATCH 把 scope 类对象本身当 key 查表接管
`with al.scope(...)`，而不是走 Python 上下文管理器协议。

对照真实源码：
    python/triton/compiler/code_generator.py:L25-31   WITH_DISPATCH 建表 + update
    python/triton/compiler/code_generator.py:L801-813 visit_With
    third_party/ascend/language/cann/extension/dispatch.py:L31-34
        ASCEND_WITH_DISPATCH = {scope: handle_scope_with}
"""
import ast
import types

from conftest import make_generator


def test_ascend_dispatch_entry_is_merged_into_with_dispatch(env):
    WITH_DISPATCH = env.code_generator.WITH_DISPATCH
    scope_cls = env.ext_scope.scope
    assert WITH_DISPATCH[scope_cls] is env.ext_codegen.handle_scope_with
    # SUBTRACTED 已把 "mangle_ty" 那一项砍掉(见 dispatch.py 的注释)：表里只有一项。
    assert len(WITH_DISPATCH) == 1


def test_visit_with_dispatches_via_table_by_class_object_key(env):
    """visit_With 的 key 是『scope 类对象本身』，不是字符串名字、也不是实例——
    context.func 被 self.visit(...) 求值成类对象后直接查表。"""
    gen = make_generator(env)
    scope_cls = env.ext_scope.scope

    seen = {}
    original_handler = env.code_generator.WITH_DISPATCH[scope_cls]
    try:
        env.code_generator.WITH_DISPATCH[scope_cls] = lambda g, n: seen.setdefault("called", (g, n))

        gen.visit = lambda node: node.func if hasattr(node, "func") else node
        node = types.SimpleNamespace(
            items=[types.SimpleNamespace(context_expr=ast.Call(func=scope_cls, args=[], keywords=[]))],
            body=[],
        )
        gen.visit_With(node)
        assert seen["called"] == (gen, node)
    finally:
        env.code_generator.WITH_DISPATCH[scope_cls] = original_handler


def test_visit_with_falls_back_when_context_is_not_a_call(env):
    """context_expr 不是 ast.Call(比如 `with some_name:`)——直接退化为 visit body，
    从不查表。"""
    gen = make_generator(env)
    calls = []
    gen.visit_compound_statement = lambda body: calls.append(body)
    gen.visit = lambda node: "should-not-be-called"

    node = types.SimpleNamespace(
        items=[types.SimpleNamespace(context_expr=ast.Name(id="not_a_call"))],
        body=["stmt-marker"],
    )
    gen.visit_With(node)
    assert calls == [["stmt-marker"]]


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


def test_scope_call_expression_is_never_actually_executed(env):
    """核心反直觉点(M1+M2 的合力)：`with al.scope(core_mode='vector'):` 里的
    `scope(core_mode='vector')` 这个调用表达式，在真正的 with-分派路径下从未被求值——
    visit_With 只对 `context.func`(scope 类对象)调 self.visit，从不对整个 ast.Call
    节点求值，所以 scope.__init__/__enter__ 都不会跑。"""
    gen = make_generator(env)
    scope_cls = env.ext_scope.scope
    init_calls = []
    original_init = scope_cls.__init__

    def spy_init(self, *a, **kw):
        init_calls.append((a, kw))
        return original_init(self, *a, **kw)

    scope_cls.__init__ = spy_init
    try:
        def visit(node):
            if isinstance(node, ast.Call):
                raise AssertionError("visit_With 不应该对整个 Call 节点求值")
            if isinstance(node, ast.Name) and node.id == "scope":
                return scope_cls
            return node

        gen.visit = visit
        gen.visit_compound_statement = lambda body: None

        scope_call = ast.Call(
            func=ast.Name(id="scope"),
            args=[],
            keywords=[ast.keyword(arg="core_mode", value=ast.Constant(value="vector"))],
        )
        node = types.SimpleNamespace(
            items=[types.SimpleNamespace(context_expr=scope_call)],
            body=[],
        )
        gen.visit_With(node)
        assert init_calls == []
    finally:
        scope_cls.__init__ = original_init
