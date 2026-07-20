"""M4/M7 —— al.fixpipe:L0C→UB 搬运 + NZ2ND 布局变换 + 对齐/dtype/芯片校验，以及
对齐约束的算术(32b:N%8/16b:N%16/NZ2DN 首维/列切分 N%32)。

SOURCE: third_party/ascend/language/cann/extension/core.py:L273-333 (前端 fixpipe)
        third_party/ascend/language/cann/extension/semantic.py:L132-148 (semantic.fixpipe)
"""
import pytest

from conftest import FakeBuilder


def _dst(env, builder, shape, dtype=None):
    tl, bl, al = env.tl, env.bl, env.al
    dtype = dtype or tl.float32
    return bl.alloc(dtype, list(shape), _address_space=al.ascend_address_space.UB, _builder=builder)


def _src(env, shape, dtype=None):
    tl = env.tl
    dtype = dtype or tl.float32
    return tl.tensor("l0c-handle", tl.block_type(dtype, list(shape)))


def test_fixpipe_success_nz2nd_fp32_calls_create_fixpipe(env):
    al = env.al
    builder = FakeBuilder()
    src = _src(env, [64, 128])
    dst = _dst(env, builder, [64, 128])

    al.fixpipe(src, dst, dma_mode=al.FixpipeDMAMode.NZ2ND, _builder=builder)

    fixpipe_calls = [c for c in builder.calls if c[0] == "create_fixpipe"]
    assert len(fixpipe_calls) == 1
    _, src_h, dst_h, dma_mode_val, dual_dst_val, pre_quant_val, pre_relu_val = fixpipe_calls[0]
    assert src_h == src.handle
    assert dst_h == dst.handle
    assert dma_mode_val == al.FixpipeDMAMode.NZ2ND.value
    assert dual_dst_val == al.FixpipeDualDstMode.NO_DUAL.value
    assert pre_quant_val == al.FixpipePreQuantMode.NO_QUANT.value
    assert pre_relu_val == al.FixpipePreReluMode.NO_RELU.value


def test_fixpipe_32b_last_dim_must_be_aligned_to_8(env):
    al = env.al
    builder = FakeBuilder()
    src = _src(env, [64, 100])
    dst = _dst(env, builder, [64, 100])

    with pytest.raises(ValueError, match="32b Fixpipe last dim must be aligned to 8"):
        al.fixpipe(src, dst, dma_mode=al.FixpipeDMAMode.NZ2ND, _builder=builder)


def test_fixpipe_32b_non_nz2nd_last_dim_must_be_aligned_to_16(env):
    al = env.al
    builder = FakeBuilder()
    # N=136: 136 % 8 == 0 (passes first gate) but 136 % 16 == 8 (fails second, non-NZ2ND).
    src = _src(env, [64, 136])
    dst = _dst(env, builder, [64, 136])

    with pytest.raises(ValueError, match="32b non-NZ2ND Fixpipe last dim must be aligned to 16"):
        al.fixpipe(src, dst, dma_mode=al.FixpipeDMAMode.NZ2NZ, _builder=builder)


def test_fixpipe_32b_column_split_last_dim_must_be_aligned_to_32(env):
    al = env.al
    builder = FakeBuilder()
    # N=104: 104 % 8 == 0, but 104 % 32 == 8 (fails column-split gate).
    src = _src(env, [64, 104])
    dst = _dst(env, builder, [64, 104])

    with pytest.raises(ValueError, match="32b Column split dual Fixpipe last dim must be aligned to 32"):
        al.fixpipe(
            src, dst, dma_mode=al.FixpipeDMAMode.NZ2ND,
            dual_dst_mode=al.FixpipeDualDstMode.COLUMN_SPLIT, _builder=builder,
        )


def test_fixpipe_32b_nz2dn_first_dim_must_be_aligned_to_8(env):
    al = env.al
    builder = FakeBuilder()
    # N=128 (%8==0, %16==0 passes non-NZ2ND gate too), M=100 (%8==4 fails NZ2DN gate).
    src = _src(env, [100, 128])
    dst = _dst(env, builder, [100, 128])

    with pytest.raises(ValueError, match="32b NZ2DN Fixpipe first dim must be aligned to 8"):
        al.fixpipe(src, dst, dma_mode=al.FixpipeDMAMode.NZ2DN, _builder=builder)


def test_fixpipe_16b_last_dim_must_be_aligned_to_16(env):
    tl, al = env.tl, env.al
    builder = FakeBuilder()
    src = _src(env, [64, 100], dtype=tl.float16)
    dst = _dst(env, builder, [64, 100], dtype=tl.float16)

    with pytest.raises(ValueError, match="16b Fixpipe last dim must be aligned to 16"):
        al.fixpipe(src, dst, dma_mode=al.FixpipeDMAMode.NZ2ND, _builder=builder)


def test_fixpipe_16b_nz2dn_first_dim_must_be_aligned_to_16(env):
    tl, al = env.tl, env.al
    builder = FakeBuilder()
    src = _src(env, [100, 128], dtype=tl.bfloat16)
    dst = _dst(env, builder, [100, 128], dtype=tl.bfloat16)

    with pytest.raises(ValueError, match="16b NZ2DN Fixpipe first dim must be aligned to 16"):
        al.fixpipe(src, dst, dma_mode=al.FixpipeDMAMode.NZ2DN, _builder=builder)


def test_fixpipe_16b_success_shape_aligned(env):
    tl, al = env.tl, env.al
    builder = FakeBuilder()
    src = _src(env, [64, 128], dtype=tl.float16)
    dst = _dst(env, builder, [64, 128], dtype=tl.float16)

    al.fixpipe(src, dst, dma_mode=al.FixpipeDMAMode.NZ2ND, _builder=builder)

    assert any(c[0] == "create_fixpipe" for c in builder.calls)


def test_fixpipe_rejects_src_not_tensor(env):
    al = env.al
    builder = FakeBuilder()
    dst = _dst(env, builder, [64, 128])

    with pytest.raises(TypeError, match="src is not of tensor type"):
        al.fixpipe(dst, dst, _builder=builder)


def test_fixpipe_rejects_dst_not_buffer(env):
    al = env.al
    builder = FakeBuilder()
    src = _src(env, [64, 128])

    with pytest.raises(TypeError, match="dst is not of buffer type"):
        al.fixpipe(src, src, _builder=builder)


def test_fixpipe_rejects_dst_not_ub(env):
    al = env.al
    builder = FakeBuilder()
    src = _src(env, [64, 128])
    dst = env.bl.alloc(env.tl.float32, [64, 128], _address_space=al.ascend_address_space.L1, _builder=builder)

    with pytest.raises(TypeError, match="dst must be located in the UB memory region"):
        al.fixpipe(src, dst, _builder=builder)


def test_fixpipe_is_910_95_gate(env):
    al = env.al
    builder = FakeBuilder(is_910_95=False)
    src = _src(env, [64, 128])
    dst = _dst(env, builder, [64, 128])

    with pytest.raises(RuntimeError, match="only supported on Ascend910_95"):
        al.fixpipe(src, dst, _builder=builder)
