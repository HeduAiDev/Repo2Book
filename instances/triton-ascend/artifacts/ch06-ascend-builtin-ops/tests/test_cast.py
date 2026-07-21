"""cast()/ascend_cast_impl（m11/m12）：overflow_mode 校验、compile_hint 挂载、
910_95 门控、bf16/fp16 经 fp32 中转、docstring 与校验列表的拼写不一致。"""
from conftest import FakeBuilder, make_tensor


def test_overflow_mode_docstring_typo_vs_real_whitelist(env):
    """m12: docstring 写的是 "sautrate"（拼写错误），真实校验列表是
    ["trunc", "saturate"] —— 照 docstring 抄会被拒。

    NOTE：cast() 是 @builtin，真实调用路径下字符串字面量在到达函数体前已经被
    CodeGenerator 包成 tl.constexpr（ch04 讲过的路由前置动作）；这里绕开 codegen
    直接调用，所以照 codegen 的样子显式传 constexpr(...)，而不是裸 str——同 ch04/
    ch05 对『直接调用 @builtin 函数』这条测试路径的一贯处理方式。
    """
    mods = env
    assert "sautrate" in mods.vec_ops.cast.__doc__
    assert "saturate" not in mods.vec_ops.cast.__doc__.replace("sautrate", "")

    b = FakeBuilder()
    x = make_tensor(mods, [4], mods.core.int32)
    constexpr = mods.core.constexpr
    try:
        mods.vec_ops.cast(x, mods.core.int16, overflow_mode=constexpr("sautrate"), _builder=b)
        assert False, "docstring 里的拼写不应该是合法值"
    except ValueError as e:
        assert "Unknown overflow_mode" in str(e)

    # 真正合法的两个值
    out = mods.vec_ops.cast(x, mods.core.int16, overflow_mode=constexpr("trunc"), _builder=b)
    assert out.dtype is mods.core.int16


def test_overflow_mode_saturate_hooks_compile_hint(env):
    mods = env
    b = FakeBuilder()
    x = make_tensor(mods, [4], mods.core.int32)
    constexpr = mods.core.constexpr
    mods.vec_ops.cast(x, mods.core.int16, overflow_mode=constexpr("saturate"), _builder=b)
    hint_calls = [c for c in b.calls if c[0] == "create_annotation_mark"]
    assert len(hint_calls) == 1
    assert hint_calls[0][2] == "overflow_mode"
    assert hint_calls[0][3] == ("str_attr", "saturate")


def test_cast_bitcast_true_skips_numeric_conversion(env):
    mods = env
    b = FakeBuilder()
    x = make_tensor(mods, [4], mods.core.float32)
    out = mods.vec_ops.cast(x, mods.core.int32, bitcast=True, _builder=b)
    bitcast_calls = [c for c in b.calls if c[0] == "create_bitcast"]
    assert len(bitcast_calls) == 1
    assert out.dtype is mods.core.int32


def test_ascend_cast_impl_rejects_fp8_fp64_off_910_95(env):
    """m11: 非 910_95 芯片上直接拒绝 fp8/fp64（这条守卫子句未被 subtraction_plan
    批准删除，逐字保留）。"""
    mods = env
    b = FakeBuilder()
    x = make_tensor(mods, [4], mods.core.float8e4nv)
    try:
        mods.vec_ops.ascend_cast_impl(x, mods.core.float32, b)
        assert False, "should have raised"
    except ValueError as e:
        assert "unsupported on Ascend" in str(e)


def test_ascend_cast_impl_saturate_int_downcast_hooks_two_hints_on_910_95(env):
    """m11: saturate 整型收窄在 910_95 上走 create_int_cast + 两个 compile_hint
    （saturate_src_unsigned/saturate_dst_unsigned），不绕道 float32。

    NOTE：is_compile_on_910_95 是 ascend_cast_impl 读的模块级硬件探测全局量（不是
    builder 的方法/属性，见 dossier m11 与 python/triton/tools/get_ascend_devices.py
    的桩），这里直接 monkeypatch 该模块全局，等价于真实场景下"这台机器是不是 910_95"。
    """
    mods = env
    b = FakeBuilder()
    mods.vec_ops.is_compile_on_910_95 = True
    try:
        x = make_tensor(mods, [4], mods.core.uint32)
        # uint32 -> int16 收窄，触发 saturate 分支需要 src 或 dst 是 unsigned
        out = mods.vec_ops.ascend_cast_impl(x, mods.core.int16, b, overflow_mode="saturate")
        int_cast_calls = [c for c in b.calls if c[0] == "create_int_cast"]
        hint_calls = [c for c in b.calls if c[0] == "create_annotation_mark"]
        fp_calls = [c for c in b.calls if c[0] in ("create_fp_trunc", "create_fp_ext", "create_si_to_fp", "create_ui_to_fp")]
        assert len(int_cast_calls) == 1
        assert len(hint_calls) == 2
        assert {h[2] for h in hint_calls} == {"saturate_src_unsigned", "saturate_dst_unsigned"}
        assert not fp_calls  # 910_95 上不需要绕道 float32
        assert out.dtype is mods.core.int16
    finally:
        mods.vec_ops.is_compile_on_910_95 = False


def test_ascend_cast_impl_saturate_int_downcast_falls_back_to_float32_off_910_95(env):
    """同一分支，非 910_95 芯片上绕道 float32 往返（先转 fp32 再转目标整型）。"""
    mods = env
    b = FakeBuilder()
    x = make_tensor(mods, [4], mods.core.uint32)
    mods.vec_ops.ascend_cast_impl(x, mods.core.int16, b, overflow_mode="saturate")
    fp_trunc_calls = [c for c in b.calls if c[0] == "create_fp_trunc"]
    hint_calls = [c for c in b.calls if c[0] == "create_annotation_mark"]
    # 绕道 float32：uint32 -> fp32（ui_to_fp）-> fp32 truncate 到自身跳过 -> ... -> int16
    ui_to_fp_calls = [c for c in b.calls if c[0] == "create_ui_to_fp"]
    fp_to_si_calls = [c for c in b.calls if c[0] == "create_fp_to_si"]
    assert ui_to_fp_calls, "应先把 uint32 转成 float32"
    assert fp_to_si_calls, "再从 float32 转回有符号整型 int16"
    assert not hint_calls  # 走的是绕道路径，不挂 compile_hint


def test_ascend_cast_impl_bf16_roundtrips_through_fp32(env):
    """m11 theory: bf16 -> 非 fp32 目标必须先经 float32 中转，一次转换拆成两次。"""
    mods = env
    b = FakeBuilder()
    x = make_tensor(mods, [4], mods.core.bfloat16)
    mods.vec_ops.ascend_cast_impl(x, mods.core.float16, b)
    # bf16 -> fp32 (ext) -> fp32 -> fp16 (trunc)
    ext_calls = [c for c in b.calls if c[0] == "create_fp_ext"]
    trunc_calls = [c for c in b.calls if c[0] == "create_fp_trunc"]
    assert len(ext_calls) == 1
    assert len(trunc_calls) == 1


def test_ascend_cast_impl_bool_dst_uses_not_equal(env):
    mods = env
    b = FakeBuilder()
    x = make_tensor(mods, [4], mods.core.int32)
    out = mods.vec_ops.ascend_cast_impl(x, mods.core.int1, b)
    ne_calls = [c for c in b.calls if c[0] == "create_icmpNE"]
    assert len(ne_calls) == 1
    assert out.dtype is mods.core.int1
