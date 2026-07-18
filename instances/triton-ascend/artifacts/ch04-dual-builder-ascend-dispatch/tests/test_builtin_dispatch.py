"""m2/m3 —— visit_Call 的第四岔 + 双内建标记。

对照真实源码 python/triton/compiler/code_generator.py:L1179-1193：
    if (hasattr(fn, '__self__') and _is_triton_value(fn.__self__)) or language.core.is_builtin(fn):
        ...
        _builder = self.ascend_builder if extension.is_builtin(fn) else self.builder
        ...
        ret = fn(*args, **extra_kwargs, **kws)

即：任何内建（tl.* 或 al.*）都要先过 `language.core.is_builtin`（读 __triton_builtin__）
这道入口门；进门后再用 `extension.is_builtin`（读 __ascend_builtin__）选 builder——
只有 ascend 内建（third_party/ascend/language/cann/extension/core.py 的 @builtin
装饰，同时打两个标记）才会被路由到 ascend_builder。

我们直接调用真实、未改写的 visit_Call 本身（不经完整 AST 遍历——本章精简版没有保留
visit_Name 等其余 visitor，见 code_generator.py 里的类文档说明），把 `self.visit`
换成恒等函数，让 `self.visit(node.func)` 直接返回我们准备好的 fn。这测的是
visit_Call 真实、逐字未改的控制流，不是重新发明的逻辑。
"""
import types

from conftest import make_generator


def _call_node(fn, args=(), kwargs=None):
    kwargs = kwargs or {}
    return types.SimpleNamespace(
        func=fn,
        args=list(args),
        keywords=[types.SimpleNamespace(arg=k, value=v) for k, v in kwargs.items()],
    )


def _identity_visit(gen):
    """visit_Call 内部还会对 node.func / 每个 keyword / 每个 arg 调 self.visit(...)。
    真实 CodeGenerator 靠 ast.NodeVisitor 按节点类型分派；本章精简版只留了
    visit_Call/visit_With 两个方法（见类文档），所以这里用恒等函数代替，只把
    visit_Call 自己的分发逻辑单独摘出来测——这是白盒隔离，不是改写 visit_Call。
    """

    def visit(node):
        if isinstance(node, types.SimpleNamespace) and hasattr(node, "arg"):
            return (node.arg, node.value)
        return node

    gen.visit = visit


def test_ascend_builtin_is_routed_to_ascend_builder(env):
    gen = make_generator(env)
    _identity_visit(gen)

    calls = []

    @env.ext_core.builtin
    def al_copy(_builder=None):
        calls.append(_builder)
        return "copied"

    node = _call_node(al_copy)
    result = gen.visit_Call(node)

    assert result == "copied"
    assert calls == [gen.ascend_builder]  # 第四岔:选中了 ascend_builder，不是主 builder


def test_plain_triton_builtin_stays_on_main_builder(env):
    gen = make_generator(env)
    _identity_visit(gen)

    calls = []

    @env.core.builtin
    def tl_load(_builder=None):
        calls.append(_builder)
        return "loaded"

    node = _call_node(tl_load)
    result = gen.visit_Call(node)

    assert result == "loaded"
    assert calls == [gen.builder]  # 只打了 __triton_builtin__，走标准 builder


def test_extension_is_builtin_reads_ascend_marker_only(env):
    @env.ext_core.builtin
    def al_fn(_builder=None):
        pass

    @env.core.builtin
    def tl_fn(_builder=None):
        pass

    def plain(_builder=None):
        pass

    assert env.extension.is_builtin(al_fn) is True
    assert env.extension.is_builtin(tl_fn) is False  # 只打了基座标记，不是 ascend 标记
    assert env.extension.is_builtin(plain) is False


def test_triton_core_is_builtin_is_entry_gate_for_both_kinds(env):
    """language.core.is_builtin 是统一入口门：tl.* 与 al.* 都要读 __triton_builtin__
    这同一个标记才能通过（al.builtin 同时打两个标记，见 dossier design_decisions）。"""

    @env.ext_core.builtin
    def al_fn(_builder=None):
        pass

    @env.core.builtin
    def tl_fn(_builder=None):
        pass

    assert env.core.is_builtin(al_fn) is True
    assert env.core.is_builtin(tl_fn) is True


def test_non_builtin_callable_falls_through_without_builder_kwarg(env):
    """既不是 JITFunction、也没打 __triton_builtin__ 的普通可调用对象——
    走 visit_Call 最后一条兜底分支，不会被塞 _builder 关键字。"""
    gen = make_generator(env)
    _identity_visit(gen)

    seen = {}

    def plain_python_fn(x):
        seen["x"] = x
        return x * 2

    node = _call_node(plain_python_fn, args=[21])
    result = gen.visit_Call(node)

    assert result == 42
    assert "x" in seen
