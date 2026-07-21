"""flip 双路径（m9）与 sort（m10）。"""
from conftest import FakeBuilder, make_tensor


def test_flip_simd_path_when_not_simt(env):
    """is_simt_mode()==False -> 走 flip_simd，一条 create_flip 算子搞定。"""
    mods = env
    b = FakeBuilder(is_simt=False)
    ptr = make_tensor(mods, [4, 8], mods.core.float32)

    out = mods.vec_ops.flip(ptr, dim=1, _builder=b)

    flip_calls = [c for c in b.calls if c[0] == "create_flip"]
    assert len(flip_calls) == 1
    assert flip_calls[0][2] == 1  # dim
    assert out.type is ptr.type  # 同类型同形状


def test_flip_simd_negative_dim_normalized(env):
    mods = env
    b = FakeBuilder(is_simt=False)
    ptr = make_tensor(mods, [4, 8], mods.core.float32)
    mods.vec_ops.flip(ptr, dim=-1, _builder=b)
    flip_calls = [c for c in b.calls if c[0] == "create_flip"]
    assert flip_calls[0][2] == 1  # -1 -> rank(2) + (-1) = 1


def test_flip_simt_path_does_log2_steps_of_xor_swap(env):
    """is_simt_mode()==True -> 没有 ascend.flip，退回 log2(n) 步 xor-swap：
    reshape 成 (..,2,2,..) -> 每步 xor_sum -> reshape 回原 shape，前后各一次
    bitcast 往返(通过 tensor.to(idtype, bitcast=True))。"""
    mods = env
    b = FakeBuilder(is_simt=True)
    ptr = make_tensor(mods, [4], mods.core.float32)  # 4 = 2**2，两步；float 才会真正触发 bitcast

    out = mods.vec_ops.flip(ptr, dim=0, _builder=b)

    bitcast_calls = [c for c in b.calls if c[0] == "create_bitcast"]
    reduce_calls = [c for c in b.calls if c[0] == "create_reduce"]
    xor_calls = [c for c in b.calls if c[0] == "create_xor"]
    reshape_calls = [c for c in b.calls if c[0] == "create_reshape"]

    # int32 -> get_int_dtype(32, signed=True) == int32 本身，bitcast 之前/之后各一次
    assert len(bitcast_calls) == 2
    # steps = log2(4) = 2 步 xor-swap
    assert len(reduce_calls) == 2
    assert len(xor_calls) == 2
    # reshape 成 (2,2) 再 reshape 回 (4,)
    assert len(reshape_calls) == 2
    assert reshape_calls[0][2] == (2, 2)
    assert reshape_calls[1][2] == (4,)
    assert out.type.shape == [4]


def test_flip_simt_requires_power_of_two_dim(env):
    mods = env
    b = FakeBuilder(is_simt=True)
    ptr = make_tensor(mods, [3], mods.core.int32)  # 3 不是 2 的幂
    try:
        mods.vec_ops.flip(ptr, dim=0, _builder=b)
        assert False, "should have raised"
    except AssertionError:
        pass


def test_sort_only_supports_last_dim(env):
    mods = env
    b = FakeBuilder()
    ptr = make_tensor(mods, [4, 8], mods.core.float32)
    try:
        mods.vec_ops.sort(ptr, dim=0, _builder=b)
        assert False, "should have raised"
    except ValueError as e:
        assert "last dimension" in str(e)


def test_sort_last_dim_ok_and_dtype_whitelisted(env):
    mods = env
    b = FakeBuilder()
    ptr = make_tensor(mods, [4, 8], mods.core.float32)
    out = mods.vec_ops.sort(ptr, dim=1, descending=True, _builder=b)
    call = [c for c in b.calls if c[0] == "create_sort"][0]
    assert call[2] == 1 and call[3] is True
    assert out.type.shape == [4, 8]


def test_sort_rejects_dtype_outside_whitelist(env):
    mods = env
    b = FakeBuilder()
    ptr = make_tensor(mods, [4], mods.core.uint8)  # uint8 不在 allowed_types 白名单
    try:
        mods.vec_ops.sort(ptr, dim=0, _builder=b)
        assert False, "should have raised"
    except TypeError as e:
        assert "only supports" in str(e)


def test_sort_int8_auto_gets_saturate_compile_hint(env):
    """m10: int8/int16 排序结果自动挂 overflow_mode=saturate 编译提示。"""
    mods = env
    b = FakeBuilder()
    ptr = make_tensor(mods, [4], mods.core.int8)
    mods.vec_ops.sort(ptr, dim=0, _builder=b)

    hint_calls = [c for c in b.calls if c[0] == "create_annotation_mark"]
    assert len(hint_calls) == 1
    _, handle, name, value = hint_calls[0]
    assert name == "overflow_mode"
    assert value == ("str_attr", "saturate")


def test_sort_float32_gets_no_compile_hint(env):
    mods = env
    b = FakeBuilder()
    ptr = make_tensor(mods, [4], mods.core.float32)
    mods.vec_ops.sort(ptr, dim=0, _builder=b)
    assert not [c for c in b.calls if c[0] == "create_annotation_mark"]


def test_sort_skips_compile_hint_in_interpreter_mode(env):
    """interpreter 模式(_builder 是 InterpreterBuilder)不支持 compile_hint，
    直接返回，即使 dtype 是 int8。"""
    import sys
    mods = env
    InterpreterBuilder = sys.modules["triton.runtime.interpreter"].InterpreterBuilder

    class FakeInterpBuilder(InterpreterBuilder):
        def __init__(self):
            self.calls = []

        def create_sort(self, handle, dim, descending):
            self.calls.append(("create_sort", handle, dim, descending))
            return "sorted_interp"

    b = FakeInterpBuilder()
    ptr = make_tensor(mods, [4], mods.core.int8)
    out = mods.vec_ops.sort(ptr, dim=0, _builder=b)
    assert not [c for c in b.calls if c[0] == "create_annotation_mark"]
    assert out.handle == "sorted_interp"
