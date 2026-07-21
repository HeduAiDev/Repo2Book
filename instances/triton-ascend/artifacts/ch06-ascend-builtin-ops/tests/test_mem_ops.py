"""mem_ops 四件套：index_put / gather_out_to_ub / scatter_ub_to_out / index_select_simd。

真实行为口径（dossier m1/m2/m3/m4/m5/m6）：
  - 四者共同形态是"GM 裸指针 + UB tile"接缝，但 index_select_simd 与另外三者不同：
    没有 index_boundary，也没有 (start_offset,end_offset,stride) 三元组。
  - 位宽契约不是统一契约：gather/scatter 硬编码 stride=i64、offset=i32；
    index_put 用 index.dtype.is_int64() 一个开关同时决定三者位宽。
"""
from conftest import FakeBuilder, make_tensor, make_gm_ptr


def test_gather_out_to_ub_index_boundary_and_widths(env):
    """m2: gather_out_to_ub 的 stride 恒 i64、offset 恒 i32，index_boundary 原样传给
    C++ builder（越界判断在编译期落到算子里，不是前端 Python 判断）。"""
    mods = env
    b = FakeBuilder()
    src = make_gm_ptr(mods, mods.core.float32, name="src")
    index = make_tensor(mods, [2, 2], mods.core.int32)

    out = mods.mem_ops.gather_out_to_ub(
        src=src, index=index, index_boundary=4, dim=0,
        src_stride=(2, 1), end_offset=(2, 2), start_offset=(0, 0), _builder=b)

    assert out.shape == [mods.core.constexpr(2), mods.core.constexpr(2)]
    assert out.dtype == mods.core.float32

    call = [c for c in b.calls if c[0] == "create_gather_out_to_ub"][0]
    _, src_h, index_h, index_boundary, dim, src_stride, end_offset, start_offset, other_h = call
    assert index_boundary == 4 and dim == 0
    assert other_h is None
    # stride 走 require_i64=True -> constexpr 走 get_int64 常量
    assert src_stride == (("i64_const", 2), ("i64_const", 1))
    # offsets 走 require_i64=False -> get_int32 常量
    assert end_offset == (("i32_const", 2), ("i32_const", 2))
    assert start_offset == (("i32_const", 0), ("i32_const", 0))


def test_gather_out_to_ub_rejects_int_src(env):
    mods = env
    b = FakeBuilder()
    src = make_gm_ptr(mods, mods.core.int32, name="src")
    index = make_tensor(mods, [2], mods.core.int32)
    try:
        mods.mem_ops.gather_out_to_ub(
            src=src, index=index, index_boundary=4, dim=0,
            src_stride=(1,), end_offset=(2,), start_offset=(0,), _builder=b)
        assert False, "should have raised"
    except ValueError as e:
        assert "fp16/fp32/bf16" in str(e)


def test_gather_out_to_ub_dim_out_of_range_rejected(env):
    mods = env
    b = FakeBuilder()
    src = make_gm_ptr(mods, mods.core.float32)
    index = make_tensor(mods, [2, 2], mods.core.int32)
    try:
        mods.mem_ops.gather_out_to_ub(
            src=src, index=index, index_boundary=4, dim=2,  # dim must be in [0, idx_rank)
            src_stride=(1, 1), end_offset=(2, 2), start_offset=(0, 0), _builder=b)
        assert False, "should have raised"
    except ValueError as e:
        assert "dim must satisfy" in str(e)


def test_scatter_ub_to_out_mirrors_gather_widths(env):
    """m3: scatter 与 gather 对称，同样是 stride=i64/offset=i32 硬编码。"""
    mods = env
    b = FakeBuilder()
    ptr = make_gm_ptr(mods, mods.core.float32)
    value = make_tensor(mods, [2, 2], mods.core.float32)
    index = make_tensor(mods, [2, 2], mods.core.int32)

    ret = mods.mem_ops.scatter_ub_to_out(
        ptr=ptr, value=value, index=index, index_boundary=4, dim=0,
        dst_stride=(2, 1), end_offset=(2, 2), start_offset=(0, 0), _builder=b)

    assert ret.type is mods.core.void
    call = [c for c in b.calls if c[0] == "create_scatter_ub_to_out"][0]
    _, ptr_h, value_h, index_h, index_boundary, dim, dst_stride, end_offset, start_offset = call
    assert dst_stride == (("i64_const", 2), ("i64_const", 1))
    assert end_offset == (("i32_const", 2), ("i32_const", 2))


def test_scatter_ub_to_out_broadcasts_scalar_value(env):
    """m3: 标量 value 会被 real_semantic.full 广播成 index.shape 的常量 tile。"""
    mods = env
    b = FakeBuilder()
    ptr = make_gm_ptr(mods, mods.core.float32)
    index = make_tensor(mods, [2, 2], mods.core.int32)

    mods.mem_ops.scatter_ub_to_out(
        ptr=ptr, value=0.0, index=index, index_boundary=4, dim=0,
        dst_stride=(2, 1), end_offset=(2, 2), start_offset=(0, 0), _builder=b)

    # full(shape, 0.0, ...) 内部对 value==0 分支走 get_null_value + create_splat
    splat_calls = [c for c in b.calls if c[0] == "create_splat"]
    assert len(splat_calls) == 1
    assert splat_calls[0][2] == (2, 2)


def test_index_put_flattens_multi_rank_index_and_uses_single_i64_switch(env):
    """m4/m6: index_put 的 index 若非 1D 会被摊平成 1D；require_i64 是单开关，
    int32 index 会让 dst_stride 也退成 i32——与 gather/scatter 的硬编码写法不同。"""
    mods = env
    b = FakeBuilder()
    ptr = make_gm_ptr(mods, mods.core.float32)
    value = make_tensor(mods, [2, 2], mods.core.float32)
    index_2d = make_tensor(mods, [2, 1], mods.core.int32)  # 非 1D，需摊平

    mods.mem_ops.index_put(
        ptr=ptr, index=index_2d, value=value, dim=0, index_boundary=4,
        end_offset=(2, 2), start_offset=(0, 0), dst_stride=(2, 1), _builder=b)

    reshape_calls = [c for c in b.calls if c[0] == "create_reshape"]
    assert reshape_calls, "index should have been reshaped to 1D"
    assert reshape_calls[0][2] == (2,)  # flat_numel == 2

    call = [c for c in b.calls if c[0] == "create_index_put"][0]
    _, ptr_h, index_h, value_h, dim, index_boundary, end_offset, start_offset, dst_stride = call
    # index.dtype 是 int32 -> require_i64=False -> 连 dst_stride 都是 i32 常量
    assert dst_stride == (("i32_const", 2), ("i32_const", 1))
    assert end_offset == (("i32_const", 2), ("i32_const", 2))


def test_index_put_uses_i64_when_index_is_int64(env):
    """同一个开关，index 换成 int64 后 dst_stride 也跟着变成 i64——证明三者真是被
    同一个 require_i64 开关联动，而不是各自独立的位宽契约。"""
    mods = env
    b = FakeBuilder()
    ptr = make_gm_ptr(mods, mods.core.float32)
    value = make_tensor(mods, [2, 2], mods.core.float32)
    index_1d = make_tensor(mods, [2], mods.core.int64)

    mods.mem_ops.index_put(
        ptr=ptr, index=index_1d, value=value, dim=0, index_boundary=4,
        end_offset=(2, 2), start_offset=(0, 0), dst_stride=(2, 1), _builder=b)

    call = [c for c in b.calls if c[0] == "create_index_put"][0]
    dst_stride = call[8]
    assert dst_stride == (("i64_const", 2), ("i64_const", 1))


def test_index_select_simd_has_no_index_boundary_param(env):
    """m5: index_select_simd 的签名里没有 index_boundary，也没有
    (start_offset,end_offset,stride) 三元组——参数是
    (src, dim, index, src_shape, src_offset, read_shape)。"""
    import inspect
    mods = env
    sig = inspect.signature(mods.mem_ops.index_select_simd.__wrapped__
                            if hasattr(mods.mem_ops.index_select_simd, "__wrapped__")
                            else mods.mem_ops.index_select_simd)
    params = list(sig.parameters.keys())
    assert "index_boundary" not in params
    assert "src_stride" not in params and "dst_stride" not in params
    assert params[:6] == ["src", "dim", "index", "src_shape", "src_offset", "read_shape"]


def test_index_select_simd_read_shape_dim_placeholder(env):
    """m5: read_shape[dim]=-1 是占位符，返回 shape 里该维被换成 index 的长度。"""
    mods = env
    b = FakeBuilder()
    src = make_gm_ptr(mods, mods.core.float32)
    index = make_tensor(mods, [4], mods.core.int32)

    out = mods.mem_ops.index_select_simd(
        src=src, dim=1, index=index,
        src_shape=(8, 100, 256), src_offset=(4, -1, 128), read_shape=(4, -1, 128), _builder=b)

    assert [s.value for s in out.shape] == [4, 4, 128]
    call = [c for c in b.calls if c[0] == "create_index_select_simd"][0]
    assert call[3] == 1  # dim
    assert call[6] == (4, -1, 128)  # read_shape passed through unchanged


def test_index_select_simd_rejects_trailing_dim(env):
    """m5: dim < ndim-1 断言——不能选尾轴（源码未给理由，见 open_questions）。"""
    mods = env
    b = FakeBuilder()
    src = make_gm_ptr(mods, mods.core.float32)
    index = make_tensor(mods, [4], mods.core.int32)
    try:
        mods.mem_ops.index_select_simd(
            src=src, dim=2, index=index,  # ndim=3, dim==ndim-1 是尾轴，应拒绝
            src_shape=(8, 100, 256), src_offset=(4, -1, -1), read_shape=(4, -1, -1), _builder=b)
        assert False, "should have raised"
    except AssertionError as e:
        assert "trailing dimension" in str(e)


def test_index_select_simd_rejects_non_1d_index(env):
    mods = env
    b = FakeBuilder()
    src = make_gm_ptr(mods, mods.core.float32)
    index_2d = make_tensor(mods, [2, 2], mods.core.int32)
    try:
        mods.mem_ops.index_select_simd(
            src=src, dim=0, index=index_2d,
            src_shape=(8, 100), src_offset=(-1, 0), read_shape=(-1, 4), _builder=b)
        assert False, "should have raised"
    except AssertionError as e:
        assert "1D" in str(e)
