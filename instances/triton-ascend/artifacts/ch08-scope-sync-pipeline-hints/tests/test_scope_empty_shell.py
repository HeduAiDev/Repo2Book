"""M2 —— scope 类的空壳性：__enter__/__exit__ 不发任何 IR，__init__ 只做
core_mode 合法性校验。M18 —— docstring 示例 `scope(feature_a=True)` 与签名不一致。

对照真实源码：third_party/ascend/language/cann/extension/scope.py:L28-71
"""
import pytest


def test_core_mode_must_be_cube_or_vector(env):
    scope = env.ext_scope.scope
    with pytest.raises(ValueError, match='core_mode must be "cube" or "vector"'):
        scope(core_mode="both", _builder=object())


def test_core_mode_cube_and_vector_are_accepted(env):
    scope = env.ext_scope.scope
    s1 = scope(core_mode="cube", _builder=object())
    s2 = scope(core_mode="vector", _builder=object())
    assert s1.core_mode == "cube"
    assert s2.core_mode == "vector"


def test_enter_raises_when_no_builder(env):
    """__enter__ 先在 self._builder is None 时 raise RuntimeError，再 return self——
    结论(不发 IR)不变；但由于 with 被编译器特判(见 test_with_dispatch.py)，kernel 内
    __enter__ 根本不会被调用，这里只验证脱离 kernel 直接构造/进入时的行为。"""
    scope = env.ext_scope.scope
    s = scope(core_mode="vector")  # 不传 _builder -> 非 kernel 内路径
    assert s._builder is None
    with pytest.raises(RuntimeError, match="scope can only be used inside a Triton kernel"):
        s.__enter__()


def test_exit_returns_false_and_never_suppresses_or_emits_ir(env):
    scope = env.ext_scope.scope
    s = scope(core_mode="vector", _builder=object())
    # __exit__ 直接 return False，不做任何 IR 操作(没有 builder 调用可记录，因为它连
    # self._builder 都不碰)。
    assert s.__exit__(None, None, None) is False


def test_docstring_example_does_not_match_required_signature(env):
    """M18：docstring 写 `scope(feature_a=True)`，但 core_mode 是必填位置参数——脱离
    kernel(无 with-分派特判)直接这样构造会 TypeError。"""
    scope = env.ext_scope.scope
    with pytest.raises(TypeError):
        scope(feature_a=True)
