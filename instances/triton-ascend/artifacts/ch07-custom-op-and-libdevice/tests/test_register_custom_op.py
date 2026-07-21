"""m1 —— register_custom_op 注册流程。

对照真实源码 third_party/ascend/language/cann/extension/custom_op.py:L324-345：
校验被装饰类带合法 core/pipe/mode 三要素 -> 抽取 __init__ 的 signature -> 以
op.name(缺省取类名)为键写入全局 `_custom_op_registry`。这是昇腾语言层相对基座
triton 多出的『注册自定义算子』能力的入口。
"""
import pytest


def _fresh_op_class(env, name="my_op", core=None, pipe=None, mode=None):
    C = env.ext_core
    core = C.CORE.VECTOR if core is None else core
    pipe = C.PIPE.PIPE_V if pipe is None else pipe
    mode = C.MODE.SIMT if mode is None else mode

    class Op:
        pass

    Op.name = name
    if core is not False:
        Op.core = core
    if pipe is not False:
        Op.pipe = pipe
    if mode is not False:
        Op.mode = mode

    def __init__(self, x, y=1):
        pass

    Op.__init__ = __init__
    return Op


def test_register_populates_global_registry(env):
    reg = env.custom_op._custom_op_registry
    assert "my_op" not in reg

    Op = _fresh_op_class(env, name="my_op")
    returned = env.custom_op.register_custom_op(Op)

    assert returned is Op  # 装饰器原样返回被装饰的类
    assert reg["my_op"] is Op


def test_register_captures_init_signature(env):
    Op = _fresh_op_class(env, name="sig_op")
    env.custom_op.register_custom_op(Op)

    assert hasattr(Op, "signature")
    # inspect.signature(cls) 对类求值时会跟 __init__ 但自动去掉 self
    # (real Python 行为，非本章精简版引入)。
    assert list(Op.signature.parameters.keys()) == ["x", "y"]


def test_register_uses_class_name_when_name_field_absent(env):
    C = env.ext_core

    class NoNameOp:
        core = C.CORE.VECTOR
        pipe = C.PIPE.PIPE_V
        mode = C.MODE.SIMT

        def __init__(self):
            pass

    env.custom_op.register_custom_op(NoNameOp)
    assert NoNameOp.name == "NoNameOp"
    assert env.custom_op._custom_op_registry["NoNameOp"] is NoNameOp


def test_register_rejects_duplicate_name(env):
    Op1 = _fresh_op_class(env, name="dup_op")
    env.custom_op.register_custom_op(Op1)

    Op2 = _fresh_op_class(env, name="dup_op")
    with pytest.raises(AssertionError, match="already used"):
        env.custom_op.register_custom_op(Op2)


@pytest.mark.parametrize("missing", ["core", "pipe", "mode"])
def test_register_requires_core_pipe_mode_fields(env, missing):
    Op = _fresh_op_class(env, name=f"missing_{missing}", **{missing: False})
    with pytest.raises(AssertionError, match="field is required"):
        env.custom_op.register_custom_op(Op)


@pytest.mark.parametrize("field,bad_value", [("core", "not-a-core"), ("pipe", 123), ("mode", None)])
def test_register_type_checks_core_pipe_mode_fields(env, field, bad_value):
    Op = _fresh_op_class(env, name=f"badtype_{field}")
    setattr(Op, field, bad_value)
    with pytest.raises(AssertionError, match="type is required"):
        env.custom_op.register_custom_op(Op)
