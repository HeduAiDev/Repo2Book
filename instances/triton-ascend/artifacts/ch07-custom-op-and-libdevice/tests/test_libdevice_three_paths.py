"""m5/m6 —— libdevice 数学函数的三种实现路径:
  ① @core.extern 直调华为数学函数库 __hmf_ 符号(reciprocal/tanh)
  ② 无 __hmf_ 符号(或旧架构)时退回纯 triton IR 组合逼近(acos)
  ③ @jit 组合已有原语(math_ops.isfinited —— 见 test 末尾)

对照真实源码 third_party/ascend/language/cann/libdevice.py:
  - reciprocal(L28-34)：(dtype)->__hmf_ 符号 的最简映射。
  - tanh(L81-93)：910_95+SIMT 开关下走单一符号，否则退回多 dtype 映射——展示
    __hmf_ 符号命名随 dtype/架构而变。
  - acos(L215-273)：同一开关下走 extern，否则纯 IR 多项式逼近——酉一个『无
    __hmf_ 时纯 IR 组合』范式代表，且是可在 CPU 上端到端复现数值的部分。
"""
import math as pymath

import pytest


def _t(env, dtype, value):
    """构造一个"handle 就是它自己数值"的标量张量，供 FakeBuilder 的真·浮点算术使用。"""
    return env.tl_core.tensor(handle=float(value), type=dtype)


# --------------------------------------------------------------------------- #
# reciprocal —— 最简 extern 样例：(dtype) -> __hmf_ 符号
# --------------------------------------------------------------------------- #

def test_reciprocal_dispatches_fp32_to_recipf_symbol(env):
    builder = env.FakeBuilder()
    x = _t(env, env.tl_core.float32, 4.0)
    result = env.libdevice.reciprocal(x, _builder=builder)
    assert result.handle.symbol == "__hmf_recipf"


def test_reciprocal_dispatches_fp16_to_recipDh_symbol(env):
    builder = env.FakeBuilder()
    x = _t(env, env.tl_core.float16, 4.0)
    result = env.libdevice.reciprocal(x, _builder=builder)
    assert result.handle.symbol == "__hmf_recipDh"


def test_reciprocal_unsupported_dtype_raises(env):
    """extern 只能调预置符号——不在 arg_type_symbol_dict 里的 dtype 组合直接报错，
    这正是本章要讲的『register_custom_op vs extern』能力分野的另一半:extern 不能
    临时"新增"一个符号来兜底。"""
    builder = env.FakeBuilder()
    x = _t(env, env.tl_core.int32, 4)
    with pytest.raises(KeyError):
        env.libdevice.reciprocal(x, _builder=builder)


# --------------------------------------------------------------------------- #
# tanh —— 双分支范式：910_95+SIMT 开关下走单一符号，否则退回多 dtype 映射
# --------------------------------------------------------------------------- #

def test_tanh_uses_single_fp32_symbol_when_simt_and_910_95(env, monkeypatch):
    monkeypatch.setattr(env.libdevice, "triton_enable_libdevice_simt", lambda: True)
    monkeypatch.setattr(env.libdevice, "is_compile_on_910_95", True)

    builder = env.FakeBuilder()
    x = _t(env, env.tl_core.float32, 0.5)
    result = env.libdevice.tanh(x, _builder=builder)
    assert result.handle.symbol == "__hmf_tanh_fp32"


@pytest.mark.parametrize("simt,is_910_95", [(False, True), (True, False), (False, False)])
def test_tanh_falls_back_to_multi_dtype_symbols_unless_both_flags_set(env, monkeypatch, simt, is_910_95):
    monkeypatch.setattr(env.libdevice, "triton_enable_libdevice_simt", lambda: simt)
    monkeypatch.setattr(env.libdevice, "is_compile_on_910_95", is_910_95)

    builder = env.FakeBuilder()
    fp32_result = env.libdevice.tanh(_t(env, env.tl_core.float32, 0.5), _builder=builder)
    assert fp32_result.handle.symbol == "__hmf_tanhf"

    builder2 = env.FakeBuilder()
    fp16_result = env.libdevice.tanh(_t(env, env.tl_core.float16, 0.5), _builder=builder2)
    assert fp16_result.handle.symbol == "__hmf_tanhDh"


# --------------------------------------------------------------------------- #
# acos —— 『无 __hmf_ 时纯 IR 组合逼近』范式代表：既验证分支选择，也验证多项式
# 数值确实复现 math.acos(在 FakeBuilder 里 create_f* 是真·浮点算术，非哨兵)。
# --------------------------------------------------------------------------- #

def test_acos_extern_path_selected_when_simt_and_910_95(env, monkeypatch):
    monkeypatch.setattr(env.libdevice, "triton_enable_libdevice_simt", lambda: True)
    monkeypatch.setattr(env.libdevice, "is_compile_on_910_95", True)

    builder = env.FakeBuilder()
    x = _t(env, env.tl_core.float32, 0.3)
    result = env.libdevice.acos(x, _builder=builder)

    assert result.handle.symbol == "__hmf_acos_fp32"
    # extern 路径下，不会执行任何纯 IR 多项式算术
    assert not any(c[0] in ("create_fadd", "create_fmul", "create_fsub") for c in builder.calls)


@pytest.mark.parametrize("x_value", [0.0, 0.2, -0.4, 0.55, 0.7, -0.7, 0.85, -0.85])
def test_acos_polynomial_fallback_matches_math_acos(env, monkeypatch, x_value):
    """SIMT/910_95 关闭 -> 走纯 IR 多项式逼近(center: |x|<0.6 / mid: |x|>=0.6 两个
    子分支都覆盖)，FakeBuilder 的算术方法是真浮点运算，可直接和 math.acos 比对。"""
    monkeypatch.setattr(env.libdevice, "triton_enable_libdevice_simt", lambda: False)
    monkeypatch.setattr(env.libdevice, "is_compile_on_910_95", False)

    builder = env.FakeBuilder()
    x = _t(env, env.tl_core.float32, x_value)
    result = env.libdevice.acos(x, _builder=builder)

    assert isinstance(result.handle, float)
    assert result.handle == pytest.approx(pymath.acos(x_value), abs=2e-3)
    # 确认真走了纯 IR 逼近分支，而不是 extern
    assert not any(c[0] == "create_extern_elementwise" for c in builder.calls)


# --------------------------------------------------------------------------- #
# isfinited(math_ops.py)—— 第三条路：@jit 组合 isnan/isinf，既非 extern 也非
# register_custom_op。这里直接调 `.fn`(同 ch04 对 JITFunction 的处理方式)，并把
# isnan/isinf 换成简单的测试替身，隔离出"组合逻辑本身对不对"这一件事。
# --------------------------------------------------------------------------- #

class _BoolTensor:
    """isnan/isinf 的测试替身返回值：只需要支持 isfinited 用到的 ~ 和 & 运算符。"""

    def __init__(self, value):
        self.value = value

    def __invert__(self):
        return _BoolTensor(not self.value)

    def __and__(self, other):
        return _BoolTensor(self.value and other.value)

    def to(self, dtype):
        return self.value


@pytest.mark.parametrize("is_nan,is_inf,expected", [
    (False, False, True),
    (True, False, False),
    (False, True, False),
    (True, True, False),
])
def test_isfinited_combines_isnan_and_isinf(env, monkeypatch, is_nan, is_inf, expected):
    monkeypatch.setattr(env.math_ops, "isnan", lambda x: _BoolTensor(is_nan))
    monkeypatch.setattr(env.math_ops, "isinf", lambda x: _BoolTensor(is_inf))

    x = env.tl_core.tensor(handle="h", type=env.tl_core.float32)
    assert env.math_ops.isfinited.fn(x) == expected
