"""M2 —— buffer 语言:bl.alloc 在指定 address_space 显式开缓冲。

SOURCE: python/triton/extension/buffer/language/core.py:L190-208 (alloc 前端)
        python/triton/extension/buffer/language/semantic.py:L35-62 (alloc 实现)
"""
import pytest

from conftest import FakeBuilder


def test_alloc_returns_buffer_with_shape_dtype_space(env):
    tl, bl, al = env.tl, env.bl, env.al
    builder = FakeBuilder()

    buf = bl.alloc(tl.float32, [64, 128], _address_space=al.ascend_address_space.UB, _builder=builder)

    assert isinstance(buf, bl.buffer)
    assert buf.shape == [64, 128]
    assert buf.dtype == tl.float32
    assert buf.space is al.ascend_address_space.UB


def test_alloc_calls_builder_alloc_with_buffer_ty_carrying_address_space_attr(env):
    tl, bl, al = env.tl, env.bl, env.al
    builder = FakeBuilder()

    bl.alloc(tl.float32, [16, 32], _address_space=al.ascend_address_space.L1, _builder=builder)

    alloc_calls = [c for c in builder.calls if c[0] == "alloc"]
    assert len(alloc_calls) == 1
    memref_ty = alloc_calls[0][1]
    # memref_ty = ("buffer_ty", shape, element_ty_ir, addr_space_attr)
    assert memref_ty[1] == (16, 32)
    assert memref_ty[3] == "attr(L1)"


def test_alloc_marks_effects_annotation_read_write(env):
    tl, bl, al = env.tl, env.bl, env.al
    builder = FakeBuilder()

    bl.alloc(tl.float32, [8, 8], _address_space=al.ascend_address_space.UB, _builder=builder)

    marks = [c for c in builder.calls if c[0] == "create_annotation_mark"]
    effects_marks = [m for m in marks if m[2] == "effects"]
    assert len(effects_marks) == 1
    assert effects_marks[0][3] == ("write", "read")


def test_alloc_is_mem_unique_adds_extra_annotation(env):
    tl, bl, al = env.tl, env.bl, env.al
    builder = FakeBuilder()

    bl.alloc(tl.float32, [8, 8], _address_space=al.ascend_address_space.UB, is_mem_unique=True, _builder=builder)

    marks = [c for c in builder.calls if c[0] == "create_annotation_mark"]
    mem_unique_marks = [m for m in marks if m[2] == "mem_unique"]
    assert len(mem_unique_marks) == 1


def test_alloc_rejects_int1_etype(env):
    tl, bl, al = env.tl, env.bl, env.al
    builder = FakeBuilder()

    with pytest.raises(TypeError):
        bl.alloc(tl.int1, [8], _address_space=al.ascend_address_space.UB, _builder=builder)
