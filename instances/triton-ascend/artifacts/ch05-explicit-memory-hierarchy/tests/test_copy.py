"""M3 —— al.copy 的地址空间逐条校验(UB→UB/L1)。

SOURCE: third_party/ascend/language/cann/extension/semantic.py:L94-129
        (copy_from_ub_to_l1 / copy)
        third_party/ascend/language/cann/extension/core.py:L174-199 (前端 builtin)
"""
import pytest

from conftest import FakeBuilder


def _bufs(env, builder, src_space, dst_space, shape=(16, 16), dtype=None):
    tl, bl, al = env.tl, env.bl, env.al
    dtype = dtype or tl.float32
    src = bl.alloc(dtype, list(shape), _address_space=src_space, _builder=builder)
    dst = bl.alloc(dtype, list(shape), _address_space=dst_space, _builder=builder)
    return src, dst


def test_copy_from_ub_to_l1_succeeds_ub_to_l1(env):
    al = env.al
    builder = FakeBuilder()
    src, dst = _bufs(env, builder, al.ascend_address_space.UB, al.ascend_address_space.L1)

    al.copy_from_ub_to_l1(src, dst, _builder=builder)

    assert ("create_copy_buffer", src.handle, dst.handle) in builder.calls


def test_copy_from_ub_to_l1_rejects_dst_not_l1(env):
    al = env.al
    builder = FakeBuilder()
    src, dst = _bufs(env, builder, al.ascend_address_space.UB, al.ascend_address_space.UB)

    with pytest.raises(TypeError, match="dst's AddressSpace must be L1"):
        al.copy_from_ub_to_l1(src, dst, _builder=builder)


def test_copy_from_ub_to_l1_rejects_src_not_ub(env):
    al = env.al
    builder = FakeBuilder()
    src, dst = _bufs(env, builder, al.ascend_address_space.L1, al.ascend_address_space.L1)

    with pytest.raises(TypeError, match="src's AddressSpace must be UB"):
        al.copy_from_ub_to_l1(src, dst, _builder=builder)


@pytest.mark.parametrize("dst_space_name", ["UB", "L1"])
def test_copy_allows_dst_ub_or_l1(env, dst_space_name):
    al = env.al
    builder = FakeBuilder()
    dst_space = getattr(al.ascend_address_space, dst_space_name)
    src, dst = _bufs(env, builder, al.ascend_address_space.UB, dst_space)

    al.copy(src, dst, _builder=builder)

    assert ("create_copy_buffer", src.handle, dst.handle) in builder.calls


def test_copy_rejects_dst_l0c(env):
    """copy 的合法搬运方向在语言层逐条 assert，而非硬件默默处理
    （dossier design_decisions 第 2 条）。"""
    al = env.al
    builder = FakeBuilder()
    src, dst = _bufs(env, builder, al.ascend_address_space.UB, al.ascend_address_space.L0C)

    with pytest.raises(TypeError, match="dst's AddressSpace must be UB or L1"):
        al.copy(src, dst, _builder=builder)


def test_copy_rejects_tensor_arguments(env):
    tl, al = env.tl, env.al
    builder = FakeBuilder()
    fake_tensor = tl.tensor("h", tl.block_type(tl.float32, [16, 16]))
    dst = _bufs(env, builder, al.ascend_address_space.UB, al.ascend_address_space.UB)[1]

    with pytest.raises(TypeError, match="tensor not support yet"):
        al.copy(fake_tensor, dst, _builder=builder)


def test_copy_rejects_shape_mismatch(env):
    al = env.al
    builder = FakeBuilder()
    src, dst = _bufs(env, builder, al.ascend_address_space.UB, al.ascend_address_space.UB,
                     shape=(16, 16))
    _, dst2 = _bufs(env, builder, al.ascend_address_space.UB, al.ascend_address_space.UB,
                    shape=(8, 8))

    with pytest.raises(TypeError, match="same shape"):
        al.copy(src, dst2, _builder=builder)


@pytest.mark.parametrize("fn_name", ["copy", "copy_from_ub_to_l1"])
def test_is_910_95_gate_blocks_copy_family(env, fn_name):
    """is_910_95 芯片门禁:copy/fixpipe 都仅 910_95 支持（M6）。"""
    al = env.al
    builder = FakeBuilder(is_910_95=False)
    src, dst = _bufs(env, builder, al.ascend_address_space.UB, al.ascend_address_space.L1)

    fn = getattr(al, fn_name)
    with pytest.raises(RuntimeError, match="only supported on Ascend910_95"):
        fn(src, dst, _builder=builder)
