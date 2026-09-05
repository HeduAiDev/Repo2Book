"""SmoothQuant —— arXiv:2211.10438 §4 Eq.3(Y=(X·diag(s)^{-1})·(diag(s)·W)
严格等价)、Eq.4(s_j = max|X_j|^α / max|W_j|^{1-α})、§5.5 Figure 10(迁移强度
α 的两个极端都崩、0.4-0.6 甜点);§2 Eq.1 的对称 INT8 量化器做 W8A8 模拟。
测试先于实现书写(TDD),每条断言对应论文的一句可复现声明。"""
import numpy as np
import pytest

from smoothquant import (
    apply_smoothing,
    migration_ablation,
    smooth_scale,
    w8a8_output_error,
    w8a8_per_tensor_output,
)


def _toy_layer(seed=0, n_tokens=96, d_in=128, d_out=8, outlier=100.0):
    """合成层:一个输入通道激活离群(~100×,§3 obs.2 的量级)且逐 token 持续
    (§3 obs.3);权重分布平坦(§3 obs.1)、各行 max 归一——α=0 恰为恒等缩放
    的对照(见末测)。线性层的 Frobenius 演示给 ~3× 改善;论文的真实收益
    (PPL 从崩溃恢复)是非线性放大。"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_tokens, d_in))
    x[:, 0] = outlier * rng.uniform(0.9, 1.1, n_tokens)  # 离群通道
    w = rng.standard_normal((d_in, d_out))
    w /= np.abs(w).max(axis=1, keepdims=True)  # 行 max 归一 -> max|W_j| 全等
    w *= 0.1
    return x, w


def test_eq3_smoothing_is_mathematically_equivalent():
    # §4 Eq.3:任意逐通道 s,(X·diag(s)^{-1})·(diag(s)·W) == X·W(浮点精确级)。
    rng = np.random.default_rng(1)
    x = rng.standard_normal((12, 7))
    w = rng.standard_normal((7, 5))
    s = np.abs(rng.standard_normal(7)) + 0.5
    x_hat, w_hat = apply_smoothing(x, w, s)
    np.testing.assert_allclose(x_hat @ w_hat, x @ w, rtol=1e-12, atol=1e-12)


def test_eq4_alpha_half_equalizes_channel_maxima():
    # §4:α=0.5 时「weights and activations at the corresponding channel share
    # a similar maximum value」——精确地:两者都等于 sqrt(max|X_j|·max|W_j|)。
    x, w = _toy_layer()
    s = smooth_scale(x, w, alpha=0.5)
    x_hat, w_hat = apply_smoothing(x, w, s)
    act_max = np.abs(x_hat).max(axis=0)  # max|X_hat_j|
    w_max = np.abs(w_hat).max(axis=1)  # max|W_hat_j|
    np.testing.assert_allclose(act_max, w_max, rtol=1e-12)


def test_eq4_formula_by_hand():
    # §4 Eq.4 逐字:s_j = max|X_j|^α / max|W_j|^{1-α}。
    x = np.array([[1.0, 4.0], [0.5, 2.0]])  # max|X_j| = [1, 4]
    w = np.array([[4.0, 1.0], [1.0, 4.0]])  # max|W_j| = [4, 4]
    s = smooth_scale(x, w, alpha=0.5)
    np.testing.assert_allclose(s, [1.0**0.5 / 4**0.5, 4**0.5 / 4**0.5])


def test_smoothing_reduces_w8a8_error_on_outlier_layer():
    # §1/§4:离群让激活难量化,平滑把难度搬进易量化的权重 -> W8A8 误差大降
    # (线性层 Frobenius 口径 ~3×;非线性 PPL 口径的收益远大于此)。
    x, w = _toy_layer()
    err_smooth = w8a8_output_error(x, w, alpha=0.5)
    # 无平滑的对照:直接 per-tensor W8A8(权重行 max 全等时 α=0 恰为恒等缩放)。
    y_sim = w8a8_per_tensor_output(x, w, num_bits=8)
    err_none = np.linalg.norm(y_sim - x @ w)
    assert err_smooth < err_none / 3


def test_migration_extremes_break_and_sweet_spot_survives():
    # §5.5 Figure 10:α 太小(<0.4)激活难量化、太大(>0.6)权重难量化,
    # 甜点在中间。合成层上:err(0.5) 同时优于 err(0.0) 与 err(1.0),
    # 且网格扫描的最优点落在 [0.3, 0.7]。
    x, w = _toy_layer(seed=2)
    err_0 = w8a8_output_error(x, w, alpha=0.0)
    err_half = w8a8_output_error(x, w, alpha=0.5)
    err_1 = w8a8_output_error(x, w, alpha=1.0)
    assert err_half < err_0
    assert err_half < err_1
    history = migration_ablation(x, w, num_bits=8)
    alphas = [a for a, _ in history]
    errs = [e for _, e in history]
    best = alphas[int(np.argmin(errs))]
    assert 0.3 <= best <= 0.7


def test_alpha_zero_with_uniform_weight_max_is_identity_scaling():
    # 行 max 全等时 max|W_j| 为常数 c:s_j = 1/c,平滑 = 全体乘常数 ——
    # 相对量化难度不变(误差与不平滑一致)。这正是「α=0 把全部难度留给激活」。
    x, w = _toy_layer(seed=3)
    err_alpha0 = w8a8_output_error(x, w, alpha=0.0)
    y_sim = w8a8_per_tensor_output(x, w, num_bits=8)
    err_plain = np.linalg.norm(y_sim - x @ w)
    assert err_alpha0 == pytest.approx(err_plain, rel=1e-9)
