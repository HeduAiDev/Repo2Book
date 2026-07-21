"""M3/M4 —— handle_scope_with 的两趟 visit 与 SSA 穿线，以及 scope 关键字 -> MLIR
属性的翻译规则。

对照真实源码：third_party/ascend/language/cann/extension/code_generator.py:L63-208
"""
import ast
import types

import pytest

from conftest import make_generator


class FakeType:
    """duck-typed 假类型：只提供 handle_scope_with 用到的 .to_ir() 与 ==。"""

    def __init__(self, name):
        self.name = name

    def to_ir(self, builder):
        return f"ty-ir:{self.name}"

    def __eq__(self, other):
        return isinstance(other, FakeType) and self.name == other.name

    def __repr__(self):
        return f"FakeType({self.name})"


def _run_scope(env, gen, keywords, body_assigns, existing_lscope=None):
    """驱动一次 `with al.scope(**keywords): <body_assigns 里的名字各被赋值一次>`。

    body_assigns: dict[name -> FakeType]，模拟 scope 块内被赋值的变量及其类型。
    existing_lscope: 调用前先塞进 gen.lscope 的既有变量(模拟外层同名 liveins)。
    """
    if existing_lscope:
        gen.lscope.update(existing_lscope)

    tensor_cls = env.core.tensor

    def visit(node):
        if node in body_assigns:
            gen.set_value(node, tensor_cls(f"{node}-handle", body_assigns[node]))
            return None
        return node

    gen.visit = visit

    scope_call = ast.Call(
        func=env.ext_scope.scope,
        args=[],
        keywords=[ast.keyword(arg=k, value=ast.Constant(value=v)) for k, v in keywords.items()],
    )
    node = types.SimpleNamespace(
        items=[types.SimpleNamespace(context_expr=scope_call)],
        body=list(body_assigns),
    )
    handle_scope_with = env.code_generator.WITH_DISPATCH[env.ext_scope.scope]
    handle_scope_with(gen, node)
    return node


def test_body_is_visited_twice_dummy_block_erased_first(env):
    """第一趟在 dummy block 上试跑收集 local_defs 后 dummy.erase()；第二趟才在真
    entry_block 里落地。两趟都要真正走 self.visit。"""
    gen = make_generator(env)
    visits = []

    tensor_cls = env.core.tensor

    def visit(node):
        visits.append(node)
        if node == "x":
            gen.set_value("x", tensor_cls("x-handle", FakeType("f32")))
        return None

    gen.visit = visit
    scope_call = ast.Call(func=env.ext_scope.scope, args=[],
                          keywords=[ast.keyword(arg="core_mode", value=ast.Constant(value="vector"))])
    node = types.SimpleNamespace(items=[types.SimpleNamespace(context_expr=scope_call)], body=["x"])

    handle_scope_with = env.code_generator.WITH_DISPATCH[env.ext_scope.scope]
    handle_scope_with(gen, node)

    assert visits == ["x", "x"]  # 两趟，都传的同一个 body 节点
    dummy_erase_calls = [c for c in gen.builder.calls if c[0] == "create_block"]
    assert len(dummy_erase_calls) == 1  # 只建了一个 dummy block(第二趟用的是 create_block_with_parent)


def test_core_mode_translates_to_t_core_type_attr(env):
    gen = make_generator(env)
    _run_scope(env, gen, {"core_mode": "vector"}, {"x": FakeType("f32")})

    create_call = next(c for c in gen.builder.calls if c[0] == "create_scope_op")
    attrs = create_call[1]
    assert attrs[gen.builder.get_t_core_type_attr_name()] == gen.builder.get_t_core_type_vector_attr()
    assert attrs["noinline"] == gen.builder.get_unit_attr()  # noinline 默认打开


def test_core_mode_cube_translates_to_cube_attr(env):
    gen = make_generator(env)
    _run_scope(env, gen, {"core_mode": "cube"}, {"x": FakeType("f32")})
    create_call = next(c for c in gen.builder.calls if c[0] == "create_scope_op")
    attrs = create_call[1]
    assert attrs[gen.builder.get_t_core_type_attr_name()] == gen.builder.get_t_core_type_cube_attr()


def test_noinline_false_removes_default_attr(env):
    gen = make_generator(env)
    _run_scope(env, gen, {"core_mode": "vector", "noinline": False}, {"x": FakeType("f32")})
    create_call = next(c for c in gen.builder.calls if c[0] == "create_scope_op")
    attrs = create_call[1]
    assert "noinline" not in attrs


def test_disable_auto_sync_true_sets_hivm_prefixed_attr(env):
    gen = make_generator(env)
    _run_scope(env, gen, {"core_mode": "vector", "disable_auto_sync": True}, {"x": FakeType("f32")})
    create_call = next(c for c in gen.builder.calls if c[0] == "create_scope_op")
    attrs = create_call[1]
    assert attrs["hivm.disable_auto_sync"] == gen.builder.get_bool_attr(True)


def test_disable_auto_sync_false_is_not_set(env):
    gen = make_generator(env)
    _run_scope(env, gen, {"core_mode": "vector", "disable_auto_sync": False}, {"x": FakeType("f32")})
    create_call = next(c for c in gen.builder.calls if c[0] == "create_scope_op")
    attrs = create_call[1]
    assert "hivm.disable_auto_sync" not in attrs


def test_arbitrary_keyword_is_passed_through_via_py_value_to_mlir_attr(env):
    """docstring 例子里的 feature_a=True 一类『既非 core_mode 又非 noinline/
    disable_auto_sync』的关键字，原样透传给 _py_value_to_mlir_attr。"""
    gen = make_generator(env)
    _run_scope(env, gen, {"core_mode": "vector", "feature_a": True}, {"x": FakeType("f32")})
    create_call = next(c for c in gen.builder.calls if c[0] == "create_scope_op")
    attrs = create_call[1]
    assert attrs["feature_a"] == gen.builder.get_bool_attr(True)


def test_non_constant_keyword_is_silently_dropped(env):
    """_extract_scope_attributes 只认 ast.Constant——非常量关键字实参被静默丢弃，
    不出现在最终属性表里、也不报错。"""
    gen = make_generator(env)
    gen.lscope.setdefault("threshold", env.core.tensor("outer-handle", FakeType("i32")))

    tensor_cls = env.core.tensor

    def visit(node):
        if node == "x":
            gen.set_value("x", tensor_cls("x-handle", FakeType("f32")))
            return None
        return node

    gen.visit = visit
    scope_call = ast.Call(
        func=env.ext_scope.scope, args=[],
        keywords=[
            ast.keyword(arg="core_mode", value=ast.Constant(value="vector")),
            # 非常量：引用一个变量名而不是字面量。
            ast.keyword(arg="threshold", value=ast.Name(id="threshold")),
        ],
    )
    node = types.SimpleNamespace(items=[types.SimpleNamespace(context_expr=scope_call)], body=["x"])
    handle_scope_with = env.code_generator.WITH_DISPATCH[env.ext_scope.scope]
    handle_scope_with(gen, node)

    create_call = next(c for c in gen.builder.calls if c[0] == "create_scope_op")
    attrs = create_call[1]
    assert "threshold" not in attrs


def test_ssa_threading_writes_back_scope_result_to_parent_lscope(env):
    """scope 结束后，块内被赋值的变量在块外可见，且已换成 scope op 的结果值——
    对照官方 UT kernel_scope_escape(test_scope.py:L49-59)。"""
    gen = make_generator(env)
    _run_scope(env, gen, {"core_mode": "vector"}, {"x": FakeType("f32")})

    assert "scope_return" in [c[0] for c in gen.builder.calls]
    assert gen.lscope["x"].handle == "scope-result-0"
    assert gen.lscope["x"].type == FakeType("f32")


def test_ssa_threading_handles_multiple_scope_defined_names(env):
    gen = make_generator(env)
    _run_scope(env, gen, {"core_mode": "cube"}, {"a": FakeType("f32"), "b": FakeType("i32")})

    create_call = next(c for c in gen.builder.calls if c[0] == "create_scope_op")
    result_types = create_call[2]
    assert result_types == ["ty-ir:f32", "ty-ir:i32"]
    assert gen.lscope["a"].handle == "scope-result-0"
    assert gen.lscope["b"].handle == "scope-result-1"


def test_loop_carried_type_change_raises_assertion(env):
    """M3/『_verify_loop_carried_variable』：外层同名变量若在 scope 内被改写成不同
    类型，assert 失败(与 while 的循环携带变量校验同一套规则)。"""
    gen = make_generator(env)
    outer = env.core.tensor("outer-x", FakeType("i32"))
    with pytest.raises(AssertionError, match="Loop-carried variable x has initial type"):
        _run_scope(env, gen, {"core_mode": "vector"}, {"x": FakeType("f32")},
                   existing_lscope={"x": outer})


def test_loop_carried_same_type_passes(env):
    gen = make_generator(env)
    outer = env.core.tensor("outer-x", FakeType("f32"))
    _run_scope(env, gen, {"core_mode": "vector"}, {"x": FakeType("f32")},
              existing_lscope={"x": outer})
    assert gen.lscope["x"].handle == "scope-result-0"


def test_no_scope_defined_names_still_creates_empty_result_scope_op(env):
    """scope 块内什么都不赋值(比如只 tl.store)——scope_defs 为空，仍然要建
    create_scope_op([], [])/scope_return([])，不能因为空而跳过整套流程。"""
    gen = make_generator(env)
    node = _run_scope(env, gen, {"core_mode": "vector"}, {})
    create_call = next(c for c in gen.builder.calls if c[0] == "create_scope_op")
    assert create_call[2] == []
    return_call = next(c for c in gen.builder.calls if c[0] == "scope_return")
    assert return_call[1] == []
