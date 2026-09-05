"""均匀量化底座 —— arXiv:2211.10438 (SmoothQuant) §2 Eq.1(对称式+Δ 定义)、§2
Figure 3(粒度谱:per-tensor / per-token / per-channel)、§3 obs.2(有效量化级数
2^N·m_i/m,RTN 之死的定量版);非对称 min-max 网格 + zero-point 取
arXiv:2210.17323 (GPTQ) §5 Setup("standard uniform per-row asymmetric
quantization on the min-max grid")与 SmoothQuant §2 的 zero-point 提法。
测试先于实现书写(TDD),每条断言对应论文的一句可复现声明。"""
import numpy as np
import pytest

from uniform_quant import (
    dequantize_asymmetric,
    dequantize_symmetric,
    effective_quant_levels,
    quantize_asymmetric,
    quantize_per_channel,
    quantize_per_tensor,
    quantize_per_token,
    quantize_symmetric,
)


def test_symmetric_quantize_matches_eq1_by_hand():
    # §2 Eq.1: X_bar = round(X/Δ), Δ = max|X|/(2^{N-1}-1)。
    # x = [1.0, 0.55, -0.3, 0.9], N=8: Δ = 1/127,逐项 round(x/Δ) = [127, 70, -38, 114]。
    x = np.array([1.0, 0.55, -0.3, 0.9])
    q, delta = quantize_symmetric(x, num_bits=8)
    assert delta == pytest.approx(1.0 / 127)
    np.testing.assert_array_equal(q, [127, 70, -38, 114])
    # 最大值本身落在格点上(Eq.1 用 absmax 定尺,不丢最远点)。
    assert dequantize_symmetric(np.array([127]), delta)[0] == pytest.approx(1.0)


def test_symmetric_round_trip_error_bounded_by_half_step():
    # round-to-nearest 的单点误差 |x - x_hat| <= Δ/2 —— 论文隐含的网格性质。
    rng = np.random.default_rng(0)
    x = rng.standard_normal(512)
    for n_bits in (4, 8):
        q, delta = quantize_symmetric(x, num_bits=n_bits)
        x_hat = dequantize_symmetric(q, delta)
        assert np.max(np.abs(x - x_hat)) <= delta / 2 + 1e-12


def test_asymmetric_minmax_grid_maps_extremes_to_code_range():
    # GPTQ §5 Setup 的非对称 min-max 网格:xmin -> qmin、xmax -> qmax 精确落格;
    # SmoothQuant §2:非对称场景加 zero-point。
    x = np.array([0.0, 0.53, 1.0, 0.73])  # N=4: scale = 1/15, zp = -8
    q, scale, zp = quantize_asymmetric(x, num_bits=4)
    assert scale == pytest.approx(1.0 / 15)
    assert zp == -8
    np.testing.assert_array_equal(q, [-8, 0, 7, 3])
    x_hat = dequantize_asymmetric(q, scale, zp)
    np.testing.assert_allclose(x_hat, [0.0, 8 / 15, 1.0, 11 / 15], atol=1e-12)
    assert np.max(np.abs(x - x_hat)) <= scale / 2 + 1e-12
    # 覆盖负偏置分布:xmin 精确映到 qmin。
    x2 = np.array([-2.0, -1.0, 0.5])
    q2, scale2, zp2 = quantize_asymmetric(x2, num_bits=4)
    assert scale2 == pytest.approx(2.5 / 15)
    np.testing.assert_array_equal(q2, [-8, -2, 7])
    np.testing.assert_allclose(
        dequantize_asymmetric(q2, scale2, zp2), x2, atol=1e-12
    )


def test_per_token_granularity_beats_per_tensor_when_rows_differ():
    # §2 Figure 3 粒度谱:per-tensor 整张矩阵一把尺子——大 token 把尺子撑爆,
    # 小 token 的值全被 round 到 0;per-token(逐行)各拿各的尺子,小 token
    # 完好。大 token 自身的量化误差两种粒度下相同,故只比小 token 的误差。
    rng = np.random.default_rng(1)
    big = rng.standard_normal((1, 32)) * 100.0
    small = rng.standard_normal((3, 32)) * 1.0
    x = np.vstack([big, small])
    q_t, d_t = quantize_per_tensor(x, num_bits=8)
    q_k, d_k = quantize_per_token(x, num_bits=8)
    err_tensor = np.mean(np.abs(small - q_t[1:] * d_t))
    err_token = np.mean(np.abs(small - q_k[1:] * d_k[1:, None]))
    assert err_token < err_tensor / 10
    # per-tensor 下小 token 的值几乎全坍缩到 0(有效级数 << 2)。
    assert np.unique(q_t[1:]).size <= 3


def test_rtn_death_outlier_channel_collapses_per_tensor():
    # §3 obs.2:通道 i 的有效量化级数 = 2^N * m_i / m —— 离群通道(max~70)把
    # 尺子撑爆,普通通道(m~1)只剩 2-3 级;per-channel 一把尺子/通道则全保住。
    assert effective_quant_levels(1.0, 70.0, num_bits=8) == pytest.approx(
        256 / 70
    )
    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 4)) * 1.0
    x[:, 0] = rng.standard_normal(8) * 70.0  # 离群通道,且逐 token 持续大(obs.3)
    q_tensor, d_tensor = quantize_per_tensor(x, num_bits=8)
    q_chan, d_chan = quantize_per_channel(x, num_bits=8)
    # per-tensor:普通通道坍缩到极少数整数档位(有效级数 ~3.7)。
    codes_normal = q_tensor[:, 1:]
    assert np.unique(codes_normal).size <= 8
    # 两种粒度下普通通道(列 1-3)的平均绝对误差:per-tensor 应比 per-channel 差一个量级。
    err_tensor = np.mean(np.abs(x[:, 1:] - q_tensor[:, 1:] * d_tensor))
    err_chan = np.mean(
        np.abs(x[:, 1:] - q_chan[:, 1:] * d_chan[None, 1:])
    )
    assert err_tensor > 10 * err_chan


def test_effective_levels_formula_is_linear_in_channel_max():
    # §3 obs.2 公式本身:2^N * m_i / m,对 m_i 线性、对 N 指数。
    assert effective_quant_levels(70.0, 70.0, num_bits=8) == pytest.approx(256.0)
    assert effective_quant_levels(35.0, 70.0, num_bits=8) == pytest.approx(128.0)
    assert effective_quant_levels(1.0, 70.0, num_bits=4) == pytest.approx(
        16 / 70
    )
