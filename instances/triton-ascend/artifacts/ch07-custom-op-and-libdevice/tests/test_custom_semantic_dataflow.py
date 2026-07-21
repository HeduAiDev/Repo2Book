"""m2/m3/m4 —— custom/custom_semantic 调用期数据流 + core/pipe/mode 到 hivm IR 属性
的翻译 + __builtin_ 前缀免注册/免 bitcode 的分野。

对照真实源码 third_party/ascend/language/cann/extension/custom_op.py:
  - custom_semantic(L294-315)：查表(_get_op_class)→实例化校验(_init_op)→
    operand 化(_to_operands/_args_to_operands)→属性化(_make_attrs)→
    `_builder.create_custom_op(name, attrs, inputs, outputs, arg_attrs)` emit IR。
  - _make_attrs(L245-271)：core/pipe/mode -> hivm.tcore_type/hivm.pipe/hivm.vf_mode
    三个 IR 属性；非 __builtin_ 前缀强制要 symbol+bitcode。
  - _get_op_class(L37-51)：注册表未命中时只放行 __builtin_ 前缀，走默认
    VECTOR/PIPE_V/SIMT 的哑类。
"""
import inspect

import pytest


def _make_tensor(env, dtype):
    tl = env.tl_core
    return tl.tensor(handle=f"h-{dtype.name}", type=dtype)


class _FakeTensorInit:
    """给自定义算子当 handle 用的最小 tensor 替身：只需要 .handle/.type/.dtype。"""


def test_custom_registered_op_builds_hivm_attrs_and_emits_ir(env):
    C = env.ext_core
    tl = env.tl_core

    @env.custom_op.register_custom_op
    class Adder:
        name = "adder"
        core = C.CORE.CUBE
        pipe = C.PIPE.PIPE_M
        mode = C.MODE.SIMD
        symbol = "adder_symbol"
        bitcode = None  # 由测试在下方替换成真实存在的文件路径

        def __init__(self, a, b, out=None):
            pass

    import tempfile
    import os
    fd, path = tempfile.mkstemp(suffix=".bc")
    os.close(fd)
    try:
        Adder.bitcode = path

        builder = env.FakeBuilder()
        a = _make_tensor(env, tl.float32)
        b = _make_tensor(env, tl.float32)
        out = _make_tensor(env, tl.float32)

        result = env.custom_op.custom_semantic("adder", a, b, out=out, _builder=builder)

        assert isinstance(result, tl.tensor)
        # m2: create_custom_op 真的被调用一次，name 是注册名
        create_calls = [c for c in builder.calls if c[0] == "create_custom_op"]
        assert len(create_calls) == 1
        _, name, attrs, inputs, outputs, arg_attrs = create_calls[0]
        assert name == "adder"
        assert inputs == [a.handle, b.handle]
        assert outputs == [out.handle]

        # m3: core/pipe/mode 被翻译成三个 hivm 属性，值就是 op.core/pipe/mode.value
        assert attrs["hivm.tcore_type"] == ("core_type_attr", C.CORE.CUBE.value)
        assert attrs["hivm.pipe"] == ("pipe_attr", C.PIPE.PIPE_M.value)
        assert attrs["hivm.vf_mode"] == ("vf_mode_attr", C.MODE.SIMD.value)
        # 非 __builtin_ 前缀：symbol/bitcode 被挂上
        assert attrs["symbol"] == ("str_attr", "adder_symbol")
        assert "bitcode" in attrs
    finally:
        os.remove(path)


def test_non_builtin_op_requires_symbol_and_bitcode(env):
    """m3：_make_attrs 对非 __builtin_ 前缀强制要求 symbol/bitcode 字段存在。"""
    C = env.ext_core

    @env.custom_op.register_custom_op
    class NoSymbolOp:
        name = "no_symbol_op"
        core = C.CORE.VECTOR
        pipe = C.PIPE.PIPE_V
        mode = C.MODE.SIMT
        # 故意不设 symbol/bitcode

        def __init__(self, out=None):
            pass

    builder = env.FakeBuilder()
    with pytest.raises(AssertionError, match="symbol is required"):
        env.custom_op.custom_semantic("no_symbol_op", _builder=builder)


def test_non_builtin_op_bitcode_must_exist_on_disk(env):
    """_add_bitcode_attr 真的用 pathlib.Path.exists() 校验文件存在——这是本章
    register_custom_op 的一处真实、可观察的运行期约束(不是文档说明)。"""
    C = env.ext_core

    @env.custom_op.register_custom_op
    class MissingBitcodeOp:
        name = "missing_bitcode_op"
        core = C.CORE.VECTOR
        pipe = C.PIPE.PIPE_V
        mode = C.MODE.SIMT
        symbol = "sym"
        bitcode = "/definitely/does/not/exist.bc"

        def __init__(self, out=None):
            pass

    builder = env.FakeBuilder()
    with pytest.raises(AssertionError, match="not exist"):
        env.custom_op.custom_semantic("missing_bitcode_op", _builder=builder)


def test_builtin_prefix_bypasses_registry_with_default_core_pipe_mode(env):
    """m4：_get_op_class 对未注册但带 __builtin_ 前缀的名字放行，走默认
    VECTOR/PIPE_V/SIMT 的哑类——免 symbol/bitcode(见 _make_attrs 的前缀豁免)。"""
    C = env.ext_core
    assert "__builtin_never_registered" not in env.custom_op._custom_op_registry

    op_class = env.custom_op._get_op_class("__builtin_never_registered")
    assert op_class.core == C.CORE.VECTOR
    assert op_class.pipe == C.PIPE.PIPE_V
    assert op_class.mode == C.MODE.SIMT

    builder = env.FakeBuilder()
    result = env.custom_op.custom_semantic("__builtin_never_registered", _builder=builder)
    assert result is None  # 无 out -> 无 outputs -> _to_result 返回 None

    _, name, attrs, inputs, outputs, arg_attrs = [c for c in builder.calls if c[0] == "create_custom_op"][0]
    assert name == "__builtin_never_registered"
    assert "symbol" not in attrs and "bitcode" not in attrs  # __builtin_ 前缀免 symbol/bitcode


def test_unregistered_non_builtin_name_raises(env):
    """_get_op_class：既未注册、又不带 __builtin_ 前缀 -> 明确报错，不静默放行。"""
    with pytest.raises(AssertionError, match="not registered"):
        env.custom_op._get_op_class("totally_unknown_op")
