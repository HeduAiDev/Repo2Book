"""M6-M11, M17 —— 核间同步的两代实现对照：旧代 aux_ops.sync_block_set/wait/all(经
_utils.custom_op 落到通用 CustomOp) vs 新代 core.create_sync_block/sync_block_set/
wait/all(经 semantic.py 落到 hivm.sync_block_set/wait/all，带 pipe 缺省配对)。

对照真实源码：
    third_party/ascend/language/cann/extension/aux_ops.py:L39-96
    third_party/ascend/language/cann/extension/_utils.py:L5-16
    third_party/ascend/language/cann/extension/core.py:L202-244
    third_party/ascend/language/cann/extension/semantic.py:L51-87
"""
import warnings

import pytest


# ---------------------------------------------------------------------------
# 旧代(aux_ops.py)：DeprecationWarning + 四条校验 + custom_op
# ---------------------------------------------------------------------------

def test_old_sync_block_set_warns_deprecated(env):
    builder = env.FakeBuilder()
    with pytest.warns(DeprecationWarning, match="Use al.sync_block_set instead"):
        env.ext_aux_ops.sync_block_set.__wrapped__("cube", "vector", 3, _builder=builder)


def test_old_sync_block_set_rejects_bad_sender(env):
    builder = env.FakeBuilder()
    with pytest.raises(AssertionError, match="only supports cube/vector"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env.ext_aux_ops.sync_block_set.__wrapped__("both", "vector", 3, _builder=builder)


def test_old_sync_block_set_rejects_same_sender_receiver(env):
    builder = env.FakeBuilder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="only supports cube -> vector or vector -> cube"):
            env.ext_aux_ops.sync_block_set.__wrapped__("cube", "cube", 3, _builder=builder)


def test_old_sync_block_set_rejects_event_id_out_of_range(env):
    builder = env.FakeBuilder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(AssertionError, match="should be 0 ~ 15"):
            env.ext_aux_ops.sync_block_set.__wrapped__("cube", "vector", 16, _builder=builder)


def test_old_sync_block_set_calls_custom_op_without_receiver(env):
    """receiver 只参与校验、不下传——custom_op 只收到 sender/event_id。"""
    builder = env.FakeBuilder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env.ext_aux_ops.sync_block_set.__wrapped__("cube", "vector", 3, _builder=builder)
    call = next(c for c in builder.calls if c[0] == "create_custom_op_for_inter_core_sync")
    assert call == ("create_custom_op_for_inter_core_sync", "sync_block_set", "cube", 3)


def test_old_sync_block_all_modes(env):
    builder = env.FakeBuilder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for mode in ("all_cube", "all_vector", "all"):
            env.ext_aux_ops.sync_block_all.__wrapped__(mode, 1, _builder=builder)
    modes_seen = [c[2] for c in builder.calls if c[0] == "create_custom_op_for_inter_core_sync"]
    assert modes_seen == ["all_cube", "all_vector", "all"]


def test_old_sync_block_all_rejects_all_sub_vector(env):
    """旧代只认三档，新代才有 all_sub_vector。"""
    builder = env.FakeBuilder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(AssertionError, match="only supports all_cube/all_vector/all"):
            env.ext_aux_ops.sync_block_all.__wrapped__("all_sub_vector", 1, _builder=builder)


# ---------------------------------------------------------------------------
# 新代(core.py + semantic.py)：pipe 缺省配对 + PIPE 类型检查 + hivm.sync_block_*
# ---------------------------------------------------------------------------

def test_new_sync_block_set_default_pipe_pairing_for_cube_sender(env):
    builder = env.FakeBuilder()
    env.ext_core.sync_block_set.__wrapped__("cube", "vector", 5, _builder=builder)
    call = next(c for c in builder.calls if c[0] == "sync_block_set")
    _, sender, receiver, event_handle, sender_pipe_v, receiver_pipe_v = call
    assert (sender, receiver) == ("cube", "vector")
    assert sender_pipe_v == env.ext_core.PIPE.PIPE_FIX.value
    assert receiver_pipe_v == env.ext_core.PIPE.PIPE_MTE2.value


def test_new_sync_block_set_default_pipe_pairing_for_vector_sender(env):
    builder = env.FakeBuilder()
    env.ext_core.sync_block_set.__wrapped__("vector", "cube", 5, _builder=builder)
    call = next(c for c in builder.calls if c[0] == "sync_block_set")
    _, sender, receiver, event_handle, sender_pipe_v, receiver_pipe_v = call
    assert sender_pipe_v == env.ext_core.PIPE.PIPE_MTE3.value
    assert receiver_pipe_v == env.ext_core.PIPE.PIPE_MTE2.value


def test_new_sync_block_wait_lands_on_wait_builder_method(env):
    builder = env.FakeBuilder()
    env.ext_core.sync_block_wait.__wrapped__("cube", "vector", 5, _builder=builder)
    assert any(c[0] == "sync_block_wait" for c in builder.calls)
    assert not any(c[0] == "sync_block_set" for c in builder.calls)


def test_new_create_sync_block_rejects_same_sender_receiver(env):
    builder = env.FakeBuilder()
    with pytest.raises(ValueError, match="only supports cube -> vector or vector -> cube"):
        env.ext_core.sync_block_set.__wrapped__("cube", "cube", 5, _builder=builder)


def test_new_create_sync_block_event_id_check_only_when_int(env):
    """新代只在 isinstance(event_id, int) 时才查区间——允许 event_id 是非 int(比如
    constexpr/tensor)时跳过区间检查(由 semantic 层的三形态归一负责)。"""
    builder = env.FakeBuilder()
    with pytest.raises(AssertionError, match="should be 0 ~ 15"):
        env.ext_core.sync_block_set.__wrapped__("cube", "vector", 16, _builder=builder)

    class _NotInt:
        handle = "not-int-handle"

    # 非 int 的 event_id：不触发 0~15 断言，直接往下走(即便"16"这种数值语义上过大)。
    env.ext_core.sync_block_set.__wrapped__("cube", "vector", _NotInt(), _builder=builder)
    call = next(c for c in builder.calls if c[0] == "sync_block_set")
    assert call[3] == "not-int-handle"


def test_new_create_sync_block_rejects_non_pipe_explicit_pipe(env):
    builder = env.FakeBuilder()
    with pytest.raises(TypeError, match="must be instances of PIPE enum"):
        env.ext_core.sync_block_set.__wrapped__(
            "cube", "vector", 5, sender_pipe="PIPE_FIX", receiver_pipe=env.ext_core.PIPE.PIPE_MTE2,
            _builder=builder)


def test_new_create_sync_block_accepts_explicit_pipe_pair(env):
    builder = env.FakeBuilder()
    env.ext_core.sync_block_set.__wrapped__(
        "cube", "vector", 5,
        sender_pipe=env.ext_core.PIPE.PIPE_ALL, receiver_pipe=env.ext_core.PIPE.PIPE_V,
        _builder=builder)
    call = next(c for c in builder.calls if c[0] == "sync_block_set")
    assert call[4] == env.ext_core.PIPE.PIPE_ALL.value
    assert call[5] == env.ext_core.PIPE.PIPE_V.value


def test_new_sync_block_all_supports_all_sub_vector_and_skips_custom_op(env):
    """新代 sync_block_all 比旧代多一档 all_sub_vector，且不再经 custom_op、直接
    _builder.sync_block_all。"""
    builder = env.FakeBuilder()
    env.ext_core.sync_block_all.__wrapped__("all_sub_vector", 2, _builder=builder)
    assert ("sync_block_all", "all_sub_vector", 2) in builder.calls
    assert not any(c[0] == "create_custom_op_for_inter_core_sync" for c in builder.calls)


def test_new_sync_block_all_rejects_bad_mode(env):
    builder = env.FakeBuilder()
    with pytest.raises(AssertionError, match="only supports all_cube/all_vector/all/all_sub_vector"):
        env.ext_core.sync_block_all.__wrapped__("all_something_else", 2, _builder=builder)


def test_new_sync_block_all_rejects_event_id_out_of_range(env):
    builder = env.FakeBuilder()
    with pytest.raises(AssertionError, match="should be 0 ~ 15"):
        env.ext_core.sync_block_all.__wrapped__("all", 16, _builder=builder)


# ---------------------------------------------------------------------------
# semantic.py：event_id 三形态(int / constexpr / tensor)归一
# ---------------------------------------------------------------------------

def test_event_id_int_form_goes_through_to_tensor(env):
    builder = env.FakeBuilder()
    env.ext_core.sync_block_set.__wrapped__("cube", "vector", 7, _builder=builder)
    call = next(c for c in builder.calls if c[0] == "sync_block_set")
    assert call[3] == 7  # FakeRealSemantic.to_tensor(constexpr(7), ...).handle == 7


def test_event_id_constexpr_form(env):
    builder = env.FakeBuilder()
    event_id = env.core.constexpr(9)
    env.ext_core.sync_block_set.__wrapped__("cube", "vector", event_id, _builder=builder)
    call = next(c for c in builder.calls if c[0] == "sync_block_set")
    assert call[3] == 9


def test_event_id_tensor_form_uses_handle_directly(env):
    builder = env.FakeBuilder()
    event_tensor = env.core.tensor("event-handle-42", "i32-ty")
    env.ext_core.sync_block_set.__wrapped__("cube", "vector", event_tensor, _builder=builder)
    call = next(c for c in builder.calls if c[0] == "sync_block_set")
    assert call[3] == "event-handle-42"


# ---------------------------------------------------------------------------
# M17 —— core.py 里 PIPE 被绑定两次：core.PIPE(自定义 enum) 覆盖了
# `PIPE = semantic.PIPE`(L61)那次绑定，两者同名同值但不是同一个类。
# ---------------------------------------------------------------------------

def test_core_pipe_is_not_semantic_pipe_but_same_values(env):
    assert env.ext_core.PIPE is not env.ext_semantic.PIPE
    assert env.ext_core.PIPE.PIPE_FIX.value is env.ext_semantic.PIPE.PIPE_FIX.value
    assert not isinstance(env.ext_core.PIPE.PIPE_FIX, env.ext_semantic.PIPE)
