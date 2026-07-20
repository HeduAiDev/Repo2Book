"""M5 —— buffer↔tensor 互转:to_tensor/to_buffer/subview 作为 bl.* 与 tl.* 的桥。

SOURCE: python/triton/extension/buffer/language/semantic.py:L65-158
        (to_buffer / to_tensor / subview)
"""
import pytest

from conftest import FakeBuilder


def test_to_tensor_converts_buffer_to_tensor_same_shape(env):
    tl, bl, al = env.tl, env.bl, env.al
    builder = FakeBuilder()
    buf = bl.alloc(tl.float32, [16, 32], _address_space=al.ascend_address_space.UB, _builder=builder)

    t = bl.to_tensor(buf, _builder=builder)

    assert isinstance(t, tl.tensor)
    assert [s.value for s in t.shape] == [16, 32]
    assert t.dtype == tl.float32
    assert ("to_tensor", buf.handle, True) in builder.calls


def test_to_tensor_with_target_shape_triggers_convert_layout(env):
    tl, bl, al = env.tl, env.bl, env.al
    builder = FakeBuilder()
    buf = bl.alloc(tl.float32, [16, 32], _address_space=al.ascend_address_space.UB, _builder=builder)

    t = bl.to_tensor(buf, target_shape=[32, 16], _builder=builder)

    assert [s.value for s in t.shape] == [32, 16]
    assert any(c[0] == "create_convert_layout" for c in builder.calls)


def test_to_buffer_converts_tensor_to_buffer(env):
    tl, bl, al = env.tl, env.bl, env.al
    builder = FakeBuilder()
    tensor = tl.tensor("t-handle", tl.block_type(tl.float32, [8, 8]))

    buf = bl.to_buffer(tensor, space=al.ascend_address_space.UB, _builder=builder)

    assert isinstance(buf, bl.buffer)
    assert buf.shape == [8, 8]
    assert ("to_buffer", "t-handle", "attr(UB)") in builder.calls


def test_to_buffer_with_bind_buffer_reuses_existing_buffer(env):
    tl, bl, al = env.tl, env.bl, env.al
    builder = FakeBuilder()
    tensor = tl.tensor("t-handle", tl.block_type(tl.float32, [8, 8]))
    existing = bl.alloc(tl.float32, [8, 8], _address_space=al.ascend_address_space.UB, _builder=builder)

    result = bl.to_buffer(tensor, bind_buffer=existing, _builder=builder)

    assert result is existing
    assert ("create_bind_buffer", "t-handle", existing.handle) in builder.calls


def test_subview_aligned_offset_succeeds_and_scales_strides(env):
    tl, bl, al = env.tl, env.bl, env.al
    builder = FakeBuilder()
    # fp32 -> primitive_bitwidth 32 -> base_byte = 32 // (32/8) = 8 elements.
    buf = bl.alloc(tl.float32, [64, 64], _address_space=al.ascend_address_space.UB, _builder=builder)

    sub = buf.subview([8, 0], [16, 16], [1, 1], _builder=builder)

    assert isinstance(sub, bl.buffer)
    assert sub.shape == [16, 16]
    # row-major strides of the [64,64] source are [64,1]; subview stride=1 keeps them.
    assert sub.strides == [64, 1]
    assert any(c[0] == "subview" for c in builder.calls)


def test_subview_unaligned_offset_rejected(env):
    tl, bl, al = env.tl, env.bl, env.al
    builder = FakeBuilder()
    buf = bl.alloc(tl.float32, [64, 64], _address_space=al.ascend_address_space.UB, _builder=builder)

    # last-dim offset=3 is not a multiple of base_byte(8) for fp32 (stride of the
    # last dim is 1, so it contributes directly to the byte offset).
    with pytest.raises(TypeError, match="32-bytes aligned"):
        buf.subview([0, 3], [16, 16], [1, 1], _builder=builder)


def test_subview_rejects_negative_offset(env):
    tl, bl, al = env.tl, env.bl, env.al
    builder = FakeBuilder()
    buf = bl.alloc(tl.float32, [64, 64], _address_space=al.ascend_address_space.UB, _builder=builder)

    with pytest.raises(ValueError, match="non-negative"):
        buf.subview([-8, 0], [16, 16], [1, 1], _builder=builder)


def test_check_subview_rank1_raises_nameerror_matching_upstream_bug(env):
    """真实源码 check_subview 的 length==1 分支引用了从未定义的单数变量 `offset`
    （形参是复数 `offsets`）——这是上游真实存在的缺陷：真实 triton-ascend 仓库里
    任何 rank-1 缓冲调用 subview() 都会在这里炸成 NameError，走不到 32-byte 对齐
    校验。本章忠实保留这个上游怪癖（只解读源码，不代表实现者顺手修好它），因此
    这里断言的是 NameError 而非 TypeError。
    """
    tl, bl, al = env.tl, env.bl, env.al
    builder = FakeBuilder()
    buf = bl.alloc(tl.float32, [64], _address_space=al.ascend_address_space.UB, _builder=builder)

    with pytest.raises(NameError, match="offset"):
        buf.subview([0], [16], [1], _builder=builder)


def test_check_subview_skips_alignment_when_offset_is_runtime_tensor(env):
    """真实源码 check_subview 的 for 循环里，一旦某个 offset 已经是 tl.tensor
    （运行时动态偏移，不是编译期已知的整数），立刻 return——放弃静态 32-byte 对齐
    检查，因为编译期根本不知道运行时值。这条支路只有直接调用 check_subview 才能
    触达：本章精简版的 subview() 前端已按 dossier 批准把 offsets 收窄成纯 int（不
    再产出 tl.tensor 偏移），但 check_subview 自身这条控制流并未获批准删除，必须
    原样保留。
    """
    tl, bl, al = env.tl, env.bl, env.al
    builder = FakeBuilder()
    buf = bl.alloc(tl.float32, [64, 64], _address_space=al.ascend_address_space.UB, _builder=builder)
    dynamic_offset = tl.tensor("off-handle", tl.block_type(tl.float32, [1]))

    # offsets[0] 是 tl.tensor → 第一次 isinstance 命中就 return；若这条早退分支
    # 被误删，offsets[1]=3（不是 base_byte=8 的倍数）会让下面这行改为抛 TypeError。
    result = env.bl_core.check_subview(buf, [dynamic_offset, 3], [16, 16], [1, 1])
    assert result is None
