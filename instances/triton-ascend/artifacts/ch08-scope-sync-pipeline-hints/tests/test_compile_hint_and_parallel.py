"""M14/M15/M16 —— compile_hint 的类型分派与 annotation.mark 落地、SIMT 早退的
外层/内层不一致、multibuffer 的具名特例、parallel(bind_sub_block=...)。

对照真实源码：third_party/ascend/language/cann/extension/aux_ops.py:L99-162
"""
import pytest


class FakePtr:
    def __init__(self, handle):
        self.handle = handle


def test_compile_hint_simt_mode_returns_early_without_emitting_mark(env):
    builder = env.FakeBuilder()
    builder._simt_mode = True
    env.ext_aux_ops.compile_hint.__wrapped__(FakePtr("p"), "hint_a", None, _builder=builder)
    assert not any(c[0] == "create_annotation_mark" for c in builder.calls)


def test_compile_hint_no_value_becomes_unit_attr(env):
    builder = env.FakeBuilder()
    env.ext_aux_ops.compile_hint.__wrapped__(FakePtr("p"), "hint_a", None, _builder=builder)
    call = next(c for c in builder.calls if c[0] == "create_annotation_mark")
    assert call == ("create_annotation_mark", "p", "hint_a", builder.get_unit_attr())


def test_compile_hint_int_value_becomes_int32_attr(env):
    builder = env.FakeBuilder()
    env.ext_aux_ops.compile_hint.__wrapped__(FakePtr("p"), "hint_b", 42, _builder=builder)
    call = next(c for c in builder.calls if c[0] == "create_annotation_mark")
    assert call[3] == builder.get_int32_attr(42)


def test_compile_hint_bool_true_becomes_bool_attr_not_unit(env):
    """isinstance(hint_val, bool) 必须先于 `elif not hint_val` 判断，否则 False 会被
    误判成"无值"分支：这里验证 True 至少落到 bool 分支。"""
    builder = env.FakeBuilder()
    env.ext_aux_ops.compile_hint.__wrapped__(FakePtr("p"), "hint_c", True, _builder=builder)
    call = next(c for c in builder.calls if c[0] == "create_annotation_mark")
    assert call[3] == builder.get_bool_attr(True)


def test_compile_hint_bool_false_becomes_bool_attr_not_silently_dropped(env):
    """isinstance(hint_val, bool) 排在 `elif not hint_val` 之前，正是为了让 False
    这个"假值"也能被正确编码成 bool attr，而不是被 `not hint_val` 分支吞成 unit_attr。"""
    builder = env.FakeBuilder()
    env.ext_aux_ops.compile_hint_impl(FakePtr("p"), "hint_c", False, builder)
    call = next(c for c in builder.calls if c[0] == "create_annotation_mark")
    assert call[3] == builder.get_bool_attr(False)


def test_compile_hint_list_value_becomes_i64_array_attr(env):
    builder = env.FakeBuilder()
    env.ext_aux_ops.compile_hint.__wrapped__(FakePtr("p"), "hint_d", [4, 8], _builder=builder)
    call = next(c for c in builder.calls if c[0] == "create_annotation_mark")
    assert call[3] == builder.get_i64_array_attr([4, 8])


def test_compile_hint_constexpr_value_becomes_str_attr(env):
    """compile_hint_impl 的 core.constexpr 分支：值被 .value 取出后编成 str_attr。"""
    builder = env.FakeBuilder()
    hint_val = env.core.constexpr("some-name")
    env.ext_aux_ops.compile_hint_impl(FakePtr("p"), "hint_e", hint_val, builder)
    call = next(c for c in builder.calls if c[0] == "create_annotation_mark")
    assert call[3] == builder.get_str_attr("some-name")


def test_compile_hint_impl_rejects_unsupported_type(env):
    builder = env.FakeBuilder()
    with pytest.raises(ValueError, match="Unsupported hint value type"):
        env.ext_aux_ops.compile_hint_impl(FakePtr("p"), "hint_f", object(), builder)


def test_multibuffer_only_accepts_size_two(env):
    builder = env.FakeBuilder()
    with pytest.raises(AssertionError, match="only support bufferize equals 2"):
        env.ext_aux_ops.multibuffer.__wrapped__(FakePtr("p"), 3, _builder=builder)


def test_multibuffer_sets_hivm_multi_buffer_hint(env):
    builder = env.FakeBuilder()
    env.ext_aux_ops.multibuffer.__wrapped__(FakePtr("p"), 2, _builder=builder)
    call = next(c for c in builder.calls if c[0] == "create_annotation_mark")
    assert call == ("create_annotation_mark", "p", "hivm.multi_buffer", builder.get_int32_attr(2))


def test_multibuffer_bypasses_simt_early_return(env):
    """M15：multibuffer 直接调 compile_hint_impl，不经过 compile_hint 的外层
    `if _builder.is_simt_mode(): return`——SIMT 模式下 multibuffer 仍会落地提示，
    是源码自带的 FIXME 现状(不作对错判断)。"""
    builder = env.FakeBuilder()
    builder._simt_mode = True
    env.ext_aux_ops.multibuffer.__wrapped__(FakePtr("p"), 2, _builder=builder)
    assert any(c[0] == "create_annotation_mark" for c in builder.calls)


def test_parallel_inherits_range_and_records_bind_sub_block(env):
    parallel = env.ext_aux_ops.parallel
    p_default = parallel(0, 10)
    assert p_default.bind_sub_block is False

    p = parallel(0, 10, step=2, num_stages=3, loop_unroll_factor=4, bind_sub_block=True)
    assert p.bind_sub_block is True
    assert p.start == 0
    assert p.end == 10
    assert p.step == 2
    assert p.num_stages == 3
    assert p.loop_unroll_factor == 4


def test_parallel_iter_raises_outside_jit(env):
    parallel = env.ext_aux_ops.parallel
    p = parallel(0, 10)
    with pytest.raises(RuntimeError, match="can only be used in @triton.jit'd functions"):
        iter(p)
