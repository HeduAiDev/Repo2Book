"""insert_slice / extract_slice（互逆的片上切片对，m7）与 get_element（tile -> 标量，m8）。"""
from conftest import FakeBuilder, make_tensor


def test_insert_extract_are_inverse_pair(env):
    mods = env
    b = FakeBuilder()
    ful = make_tensor(mods, [4, 4], mods.core.float32)
    sub = make_tensor(mods, [4, 4], mods.core.float32)

    inserted = mods.vec_ops.insert_slice(
        ful, sub, offsets=(0, 0), sizes=(2, 2), strides=(1, 1), _builder=b)
    assert inserted.type.shape == [4, 4]  # insert_slice 返回类型用 ful.shape

    extracted = mods.vec_ops.extract_slice(
        ful, offsets=(0, 0), sizes=(2, 2), strides=(1, 1), _builder=b)
    assert extracted.type.shape == [2, 2]  # extract_slice 返回类型用 sizes（唯一区别）

    insert_call = [c for c in b.calls if c[0] == "create_insert_slice"][0]
    extract_call = [c for c in b.calls if c[0] == "create_extract_slice"][0]
    assert insert_call[4] == (2, 2) and insert_call[5] == (1, 1)  # sizes, strides
    assert extract_call[3] == (2, 2) and extract_call[4] == (1, 1)


def test_slice_requires_matching_rank_offsets(env):
    mods = env
    b = FakeBuilder()
    ful = make_tensor(mods, [4, 4], mods.core.float32)
    try:
        mods.vec_ops.extract_slice(ful, offsets=(0,), sizes=(2, 2), strides=(1, 1), _builder=b)
        assert False, "should have raised"
    except AssertionError:
        pass


def test_get_element_reads_scalar(env):
    mods = env
    b = FakeBuilder()
    src = make_tensor(mods, [2, 2], mods.core.float32)
    scalar = mods.vec_ops.get_element(src, (0, 1), _builder=b)
    assert scalar.type is mods.core.float32  # 标量类型 = src.type.scalar

    call = [c for c in b.calls if c[0] == "create_extract_scalar"][0]
    assert call[2] == (0, 1)


def test_get_element_rejects_rank_mismatch(env):
    mods = env
    b = FakeBuilder()
    src = make_tensor(mods, [2, 2], mods.core.float32)
    try:
        mods.vec_ops.get_element(src, (0,), _builder=b)
        assert False, "should have raised"
    except ValueError as e:
        assert "rank" in str(e)


def test_manual_gather_baseline_uses_get_element_and_insert_slice(env):
    """official test_index_select.py 里的『手写基线』写法：逐个用 get_element 取
    索引再 insert_slice 拼回——不依赖任何 mem_ops 扩展算子，是 m13 的第一条路。"""
    mods = env
    b = FakeBuilder()
    indices = make_tensor(mods, [2], mods.core.int32)
    tmp_buf = make_tensor(mods, [2, 4], mods.core.float32)

    idx0 = mods.vec_ops.get_element(indices, (0,), _builder=b)
    row = make_tensor(mods, [1, 4], mods.core.float32)  # 对应真实用例里的 val[None, :]
    tmp_buf = mods.vec_ops.insert_slice(
        tmp_buf, row, offsets=(0, 0), sizes=(1, 4), strides=(1, 1), _builder=b)

    assert idx0.type is mods.core.int32
    assert tmp_buf.type.shape == [2, 4]
