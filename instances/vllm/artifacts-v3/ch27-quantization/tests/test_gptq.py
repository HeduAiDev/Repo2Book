"""GPTQ —— arXiv:2210.17323 §3 Eq.1(层重建目标)/Eq.2-Eq.3(OBQ 逐权重贪心
+ 逆 Hessian 高斯消元)、§4 Step 1(任意固定列序+每列一次 H^{-1} 更新)/Step 2
(lazy batch B=128)/Step 3(Cholesky 重构+dampening 1% 平均对角元)/Algorithm 1
(主循环)、§5(Setup 的 per-row 非对称 min-max 网格、Baselines 的 RTN 对照、
Table 3 的 RTN 崩溃 vs GPTQ 存活)。
测试先于实现书写(TDD),每条断言对应论文的一句可复现声明。"""
import numpy as np
import pytest

from gptq import (
    dampen_hessian,
    dequantize_with_grid,
    gptq_naive_inverse_updates,
    gptq_quantize,
    hessian_update_flops,
    inverse_hessian_cholesky,
    layer_hessian,
    layer_output_error,
    obq_quantize_row,
    quantize_with_grid,
    rtn_quantize,
    row_grid_params,
)


def _toy_xy(seed=3, d_row=8, d_col=12, n_samples=48, corr=0.95):
    """合成层:特征强相关(corr)使 H 病态——GPTQ 补偿的价值所在。"""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((d_row, d_col))
    base = rng.standard_normal((n_samples, 1))
    noise = rng.standard_normal((n_samples, d_col))
    x = corr * base + np.sqrt(1 - corr**2) * noise  # 列间相关 -> H 各向异性
    return w, x


def test_layer_hessian_is_2xtx_by_hand():
    # §3:H_F = 2 X_F X_F^T,只依赖层输入、与权重无关(全行同序的合法性来源)。
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    h = layer_hessian(x)
    np.testing.assert_allclose(h, 2.0 * np.array([[2.0, 1.0], [1.0, 2.0]]))


def test_row_grid_params_per_row_asymmetric_minmax_by_hand():
    # §5 Setup:per-row 非对称 min-max 网格。行 [0.2, -0.5, 0.9, 0.0],N=4:
    # scale = (0.9-(-0.5))/15, zp = -8 - round(-0.5/scale) = -3;
    # codes = [-1, -8, 7, -3](xmin->qmin、xmax->qmax 精确落格)。
    w_rows = np.array([[0.2, -0.5, 0.9, 0.0]])
    scale, zp = row_grid_params(w_rows, num_bits=4)
    assert scale[0] == pytest.approx(1.4 / 15)
    assert zp[0] == -3
    q = quantize_with_grid(w_rows[0], scale, zp, num_bits=4)
    np.testing.assert_array_equal(q, [-1, -8, 7, -3])
    np.testing.assert_allclose(
        dequantize_with_grid(q, scale, zp), [0.2 * 0.93333, -0.466667, 0.933333, 0.0],
        atol=1e-5,
    )


def test_rtn_quantize_rounds_whole_matrix_on_per_row_grid():
    # §5 Baselines:RTN = 在与 GPTQ 完全相同的 per-row 网格上直接取整(无补偿)。
    w, x = _toy_xy()
    q, w_hat = rtn_quantize(w, num_bits=4)
    assert q.shape == w.shape
    assert layer_output_error(w, w_hat, x) > 0
    # 与逐行独立求网格再取整的结果一致(无跨列交互)。
    for r in range(w.shape[0]):
        scale, zp = row_grid_params(w[r : r + 1], num_bits=4)
        np.testing.assert_array_equal(
            q[r], quantize_with_grid(w[r], scale, zp, num_bits=4)
        )


def test_dampen_hessian_adds_one_percent_of_mean_diagonal():
    # §4 Step 3:"adding a small constant λ (we always choose 1% of the average
    # diagonal value) to the diagonal elements of H"。
    h = np.diag([1.0, 2.0, 3.0])
    hd = dampen_hessian(h, damp=0.01)
    lam = 0.01 * 2.0  # mean(diag) = 2
    np.testing.assert_allclose(hd, h + lam * np.eye(3))


def test_cholesky_of_inverse_hessian_survives_rank_deficient_x():
    # Step 3 的动机:无 dampening 时重复 Eq.5 会把 H^{-1} 推成不定阵;秩亏的 X
    # (重复特征列)连 H=2X^T X 都不正定。Algorithm 1 前置行:H^{-1} =
    # (2XX^T + λI)^{-1},再取 Cholesky(H^{-1})^T(上三角)。
    rng = np.random.default_rng(4)
    x_full = rng.standard_normal((16, 4))
    x = np.hstack([x_full, x_full])  # 第 5 列 == 第 4 列 -> 2X^T X 奇异
    h = layer_hessian(x)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(h)
    u = inverse_hessian_cholesky(x, damp=0.01)
    assert np.allclose(u, np.triu(u))  # 上三角(Cholesky(H^{-1})^T)
    # U^T U = H^{-1}(H 已含 dampening)。
    hd = dampen_hessian(h, damp=0.01)
    np.testing.assert_allclose(u.T @ u, np.linalg.inv(hd), atol=1e-8)


def test_gptq_quantize_beats_rtn_on_same_grid():
    # §5 Table 3 / Figure 1:同一副网格,RTN 3-bit 崩溃、GPTQ 3-bit 存活——
    # 差别只在「按什么顺序 round、round 完怎么找补」。
    w, x = _toy_xy()
    _, _, err_gptq = gptq_quantize(w, x, num_bits=3, block_size=8)
    q_rtn, w_hat_rtn = rtn_quantize(w, num_bits=3)
    err_rtn = layer_output_error(w, w_hat_rtn, x)
    assert err_gptq < 0.5 * err_rtn


def test_gptq_cholesky_lazy_batch_equals_naive_per_column_updates():
    # Step 2+Step 3 不改变数学结果(只改执行方式):lazy batch 推迟的更新不影响
    # 当前列取整决策;Cholesky 行 = 对称 H^{-1} 反复高斯消元的稳定等价。
    # 直接按 Eq.2+Eq.3 每列一次更新 H^{-1}(Step 1 的算法)应与 Algorithm 1
    # 给出相同的量化权重。
    w, x = _toy_xy()
    q_a, w_hat_a, _ = gptq_quantize(w, x, num_bits=4, block_size=8)
    q_b, w_hat_b = gptq_naive_inverse_updates(w, x, num_bits=4)
    np.testing.assert_array_equal(q_a, q_b)
    np.testing.assert_allclose(w_hat_a, w_hat_b, atol=1e-6)


def test_lazy_batch_size_does_not_change_result():
    # Step 2:"The final rounding decisions for column i are only affected by
    # updates performed on this very column" —— 分块大小 B 不影响结果。
    w, x = _toy_xy()
    q1, w1, err1 = gptq_quantize(w, x, num_bits=4, block_size=128)
    q2, w2, err2 = gptq_quantize(w, x, num_bits=4, block_size=3)
    np.testing.assert_array_equal(q1, q2)
    assert err1 == pytest.approx(err2)


def test_grouping_recomputes_grid_on_current_weights_and_helps():
    # §5 Additional Tricks:grouping 与 GPTQ 兼容,"the group parameters can be
    # determined during the quantization process ... always using the most
    # current updated weights";更细的组 = 更准(表 5: g1024 -> g128 改善)。
    w, x = _toy_xy(seed=5, d_row=6, d_col=16)
    _, _, err_full = gptq_quantize(w, x, num_bits=3, group_size=16)
    _, _, err_g4 = gptq_quantize(w, x, num_bits=3, group_size=4)
    assert err_g4 < err_full


def test_obq_greedy_row_beats_rtn_and_matches_gptq_scale():
    # §3 Eq.2:OBQ 逐权重贪心(每步选 (quant(w)-w)^2/[H^{-1}]_qq 最小者)+ Eq.3
    # 更新逆 Hessian;§4 Step 1:换成任意固定列序,最终误差相近(不止崩)。
    w, x = _toy_xy(seed=6, d_row=1, d_col=10, n_samples=32)
    h = layer_hessian(x)
    q_obq, w_hat_obq = obq_quantize_row(w[0], h, num_bits=3)
    q_rtn, w_hat_rtn = rtn_quantize(w, num_bits=3)
    err_obq = layer_output_error(w, w_hat_obq[None, :], x)
    err_rtn = layer_output_error(w, w_hat_rtn, x)
    assert err_obq < err_rtn
    _, _, err_gptq = gptq_quantize(w, x, num_bits=3, block_size=8)
    # 固定列序 vs 贪心列序:同数量级(Step 1 "similar" 声明,允许 2 倍内)。
    assert err_gptq < 2.0 * err_obq
    assert err_obq < 2.0 * err_gptq


def test_obq_trace_records_greedy_order():
    # 教学轨迹:OBQ 的贪心顺序是「按代价动态挑列」,一般不是 0,1,2,...
    w, x = _toy_xy(seed=7, d_row=1, d_col=8, n_samples=24)
    trace = []
    obq_quantize_row(w[0], layer_hessian(x), num_bits=3, trace=trace)
    assert len(trace) == 8
    assert sorted(entry[0] for entry in trace) == list(range(8))


def test_hessian_update_flops_matches_paper_complexity():
    # §4 Step 1:OBQ O(d_row * d_col^3) -> GPTQ O(max{d_row*d_col^2, d_col^3}),
    # 提速 min{d_row, d_col} 倍。
    obq, gptq = hessian_update_flops(d_row=4096, d_col=4096)
    assert obq == 4096 * 4096**3
    assert gptq == max(4096 * 4096**2, 4096**3)
    assert obq / gptq == 4096  # min{d_row, d_col}
    obq1, gptq1 = hessian_update_flops(d_row=1, d_col=8)
    assert obq1 == 8**3 and gptq1 == max(8**2, 8**3) == 8**3
