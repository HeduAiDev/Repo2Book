"""_utils.custom_op 手写 dispatcher(与 ch07 的 register_custom_op 注册表划清界限)，
以及 CORE/PIPE/MODE 三个 Python 枚举与 M12/M13 口径收窄(与本仓 ascend_ir 占位对齐)。

对照真实源码：
    third_party/ascend/language/cann/extension/_utils.py:L5-16
    third_party/ascend/language/cann/extension/core.py:L104-125
    third_party/ascend/ascend_ir.cc:L420-436(pybind 导出的 CoreType 4 档 / PIPE 8 档)
"""
import pytest


def test_custom_op_routes_three_known_names(env):
    builder = env.FakeBuilder()
    env.ext_utils.custom_op(builder, "sync_block_all", mode="all", event_id=1)
    env.ext_utils.custom_op(builder, "sync_block_set", sender="cube", event_id=2)
    env.ext_utils.custom_op(builder, "sync_block_wait", sender="vector", event_id=3)
    assert builder.calls == [
        ("create_custom_op_for_inter_core_sync", "sync_block_all", "all", 1),
        ("create_custom_op_for_inter_core_sync", "sync_block_set", "cube", 2),
        ("create_custom_op_for_inter_core_sync", "sync_block_wait", "vector", 3),
    ]


def test_custom_op_rejects_unknown_name(env):
    builder = env.FakeBuilder()
    with pytest.raises(ValueError, match="Unsupported custom op"):
        env.ext_utils.custom_op(builder, "sync_block_frobnicate", event_id=1)


def test_custom_op_is_not_ch07_register_custom_op(env):
    """同名陷阱：本章 custom_op 是 _utils.py 里只认三个 op 名的手写 dispatcher，与
    ch07 的 custom_op.register_custom_op 注册表是完全不同的两个符号。这里只断言
    本章 custom_op 没有注册表(dict/装饰器)相关的公开接口，避免读者把两者混同。"""
    custom_op = env.ext_utils.custom_op
    assert not hasattr(custom_op, "register")
    assert custom_op.__module__.endswith("extension._utils")


# ---------------------------------------------------------------------------
# CORE / PIPE / MODE 枚举与 M12/M13 口径收窄
# ---------------------------------------------------------------------------

def test_core_pipe_mode_enum_sizes(env):
    CORE = env.ext_core.CORE
    PIPE = env.ext_core.PIPE
    MODE = env.ext_core.MODE
    assert {m.name for m in CORE} == {"VECTOR", "CUBE", "CUBE_OR_VECTOR", "CUBE_AND_VECTOR"}
    assert len(list(PIPE)) == 8
    assert {m.name for m in MODE} == {"SIMD", "SIMT", "MIX"}


def test_scope_core_mode_cannot_reach_cube_or_vector_or_cube_and_vector(env):
    """M13：TCoreType 四档在枚举里真实存在(CORE.CUBE_OR_VECTOR/CUBE_AND_VECTOR)，
    但 scope 的 core_mode 校验只接受 "cube"/"vector" 两个字符串——语言层到不了另外
    两档，即便 CORE 枚举本身能表示它们。"""
    scope = env.ext_scope.scope
    with pytest.raises(ValueError):
        scope(core_mode="cube_or_vector", _builder=object())
    with pytest.raises(ValueError):
        scope(core_mode="cube_and_vector", _builder=object())
    # CORE 枚举本身确实有这两档(证明是"语言层够不着"，不是"根本不存在")。
    assert env.ext_core.CORE.CUBE_OR_VECTOR.name == "CUBE_OR_VECTOR"
    assert env.ext_core.CORE.CUBE_AND_VECTOR.name == "CUBE_AND_VECTOR"


def test_handle_core_mode_attr_silently_returns_empty_for_non_cube_vector(env):
    """_handle_core_mode_attr 遇到不是 cube/vector 的值直接 return {}(不报错)——
    这条分支通常走不到(scope.__init__ 会先挡)，但作为翻译规则本身要如实存在。"""
    from conftest import FakeBuilder
    builder = FakeBuilder()
    attrs = env.ext_codegen._handle_core_mode_attr(builder, "cube_or_vector")
    assert attrs == {}
