"""AWQ —— arXiv:2306.00978 §3.1(按激活幅度选显著通道)、§3.2 Eq.1(组内
absmax 定尺的量化器)/Eq.2(等价缩放 Q(w·s)(x/s))/Eq.3(两条误差表达式与
(Δ'/Δ)·(1/s) 误差比)/obs.(1)-(3)(RoundErr~0.25、Δ'≈Δ、FP16 中间量无误差)/
Table 2(缩放统计协议)/Eq.4-Eq.5(L(s) 目标与 s=s_X^α 搜索空间)、§4.2
(SIMD-aware weight packing:GPU 上按 w_{0,2,4,6,1,3,5,7} 打包)。
测试先于实现书写(TDD),每条断言对应论文的一句可复现声明。"""
import numpy as np
import pytest

from awq import (
    awq_dequantize,
    awq_group_quantize,
    awq_loss,
    awq_pack,
    awq_quantize_matrix,
    awq_search_scale,
    awq_unpack,
    channel_mean_activation,
    err_Qws_xs,
    err_Qwx,
    round_err,
    salient_channels,
    table2_statistics,
)


def test_eq1_group_quantizer_by_hand():
    # §3.2 Eq.1:Q(w) = Δ·Round(w/Δ), Δ = max|w|/2^{N-1}。
    # 组 [0.9, 9.9], N=4: Δ = 9.9/8 = 1.2375, codes = [1, 8]。
    w = np.array([0.9, 9.9])
    q, delta = awq_group_quantize(w, num_bits=4)
    assert delta == pytest.approx(9.9 / 8)
    np.testing.assert_array_equal(q, [1, 8])
    np.testing.assert_allclose(awq_dequantize(q, delta), [1.2375, 9.9])


def test_eq1_round_trip_error_bounded_by_half_step():
    rng = np.random.default_rng(0)
    w = rng.standard_normal(1024) * 0.05
    q, delta = awq_group_quantize(w, num_bits=4)
    assert np.max(np.abs(w - awq_dequantize(q, delta))) <= delta / 2 + 1e-12


def test_round_err_average_is_quarter():
    # §3.2 obs.(1):round 把浮点映到整数,误差大致均匀分布于 [0, 0.5],
    # 平均误差 0.25:RoundErr(·) ~ 0.25。
    rng = np.random.default_rng(1)
    u = rng.uniform(0.0, 1.0, 200_000)
    assert abs(np.mean(np.abs(round_err(u)))) == pytest.approx(0.25, abs=0.005)


def test_worked_example_salient_weight_9_9_group_s_2():
    # 教学手算(dossier m05):显著权重 w=0.9,组内另有 9.9 定尺,s=2。
    # 未缩放:Q(0.9)=1.2375,误差 0.3375;
    # 缩放后:ws=1.8,Δ'=Δ(obs.2:单元素放大通常不改组内 max),
    # Q(1.8)·(x/s)=0.61875,误差 0.28125 —— 误差比 |0.28125/0.3375|=5/6,
    # 而 (Δ'/Δ)·(1/s)=1/2(Eq.3 两条误差式的比值)。
    group = np.array([0.9, 9.9])
    _, delta = awq_group_quantize(group, num_bits=4)
    x = 1.0
    s = 2.0
    e_plain = err_Qwx(0.9, x, delta)
    assert e_plain == pytest.approx(0.3375)
    e_scaled = err_Qws_xs(0.9, x, s, delta)  # Δ'=Δ
    assert e_scaled == pytest.approx(-0.28125)
    assert abs(e_scaled) < abs(e_plain)
    # Δ' 精确不变:0.9*2=1.8 < 9.9,组 max 仍由非显著权重决定(obs.2)。
    _, delta_prime = awq_group_quantize(group * np.array([s, 1.0]), num_bits=4)
    assert delta_prime == pytest.approx(delta)


def test_table2_statistics_protocol():
    # §3.2 Table 2 协议:给 1% 显著通道(按激活幅度选)乘 s,逐组统计
    # Δ 变化率。s=2 时:Δ'≠Δ 的组比例小(<10%,表值 8.2%)、平均
    # (Δ'/Δ)·(1/s) 落在 0.5 附近(表值 0.519)。
    rng = np.random.default_rng(2)
    d_in, d_out, n_tokens = 128, 32, 64
    w = rng.standard_normal((d_in, d_out)) * 0.05
    x = rng.standard_normal((n_tokens, d_in))
    x[:, :2] *= 20.0  # 2/128 ~ 1.5% 显著通道:激活大
    stats = table2_statistics(w, x, s=2.0, num_bits=4, group_size=16)
    assert stats["proportion_delta_changed"] < 0.10
    assert 0.35 < stats["mean_scaled_error_ratio"] < 0.65


def test_salient_channels_selected_by_activation_magnitude():
    # §3.1:显著与否「看激活不看权重」——按激活幅度(而非权重范数)选通道。
    rng = np.random.default_rng(3)
    x = rng.standard_normal((32, 64))
    x[:, 7] *= 30.0  # 唯一大激活通道
    mask = salient_channels(x, frac=1 / 64)
    assert mask.sum() == 1
    assert mask[7]
    # s_X = 逐输入通道的平均激活幅度(Eq.5 的底数)。
    s_x = channel_mean_activation(x)
    assert s_x.shape == (64,)
    assert s_x[7] > 10 * np.median(s_x)


def test_awq_quantize_matrix_groups_along_input_channel():
    # Eq.1 的组 = 输入通道维上连续 group_size 个权重(每输出通道一列)。
    rng = np.random.default_rng(4)
    w = rng.standard_normal((16, 3)) * 0.1
    w_hat = awq_quantize_matrix(w, num_bits=4, group_size=8)
    # 逐组对照:每列每 8 行一组,组内 Δ = max|·|/8。
    for c in range(3):
        for g in range(0, 16, 8):
            blk = w[g : g + 8, c]
            q, delta = awq_group_quantize(blk, num_bits=4)
            np.testing.assert_allclose(
                w_hat[g : g + 8, c], awq_dequantize(q, delta)
            )


def test_search_finds_nonzero_alpha_that_beats_rtn():
    # §3.2 Eq.4-Eq.5:L(s) = ||Q(W·diag(s))(diag(s)^{-1}·X) - WX||,
    # s = s_X^α、α ∈ [0,1] 网格(附录 C.2:grid size 20);α=0 即 RTN。
    # 有显著激活通道时,内点 α 应严格优于 α=0。
    rng = np.random.default_rng(5)
    d_in, d_out, n_tokens = 64, 32, 96
    w = rng.standard_normal((d_in, d_out)) * 0.05
    x = rng.standard_normal((n_tokens, d_in))
    x[:, :3] *= 15.0  # ~5% 显著通道
    best_alpha, best_s, history = awq_search_scale(
        w, x, num_bits=3, group_size=32, grid_size=20
    )
    losses = dict(history)
    assert len(history) == 20
    assert 0.0 < best_alpha < 1.0
    assert losses[best_alpha] < losses[0.0]  # 打败 RTN(s=1)
    # 最优 s 恰为 s_X^best_alpha(Eq.5 的搜索空间形状)。
    np.testing.assert_allclose(
        best_s, channel_mean_activation(x) ** best_alpha, rtol=1e-12
    )
    # 显著通道拿到的 scale 比非显著通道大(激活感知)。
    assert np.all(best_s[:3] > np.median(best_s))


def test_awq_loss_with_identity_scale_equals_rtn():
    # s ≡ 1(α=0)时 L(s) 就是 RTN 的量化输出误差 —— Eq.4 与 RTN 的衔接。
    rng = np.random.default_rng(6)
    w = rng.standard_normal((32, 8)) * 0.1
    x = rng.standard_normal((16, 32))
    loss_one = awq_loss(w, x, np.ones(32), num_bits=4, group_size=32)
    w_hat = awq_quantize_matrix(w, num_bits=4, group_size=32)
    err = np.linalg.norm(x @ w_hat - x @ w)
    assert loss_one == pytest.approx(err)


def test_awq_pack_interleaves_0246_1357_and_round_trips():
    # §4.2 SIMD-aware packing:"On GPUs, we found it more efficient to pack
    # each 8 weights into w_{0,2,4,6,1,3,5,7}" —— 8 个 4-bit 权重按
    # 偶下标在前重排后压进一个 32-bit 字(第 i 个 nibble 放重排后第 i 个)。
    q = np.arange(16, dtype=np.int64)  # [0..15](只验位布局;有符号码域见下一测)
    packed = awq_pack(q, num_bits=4)
    assert packed.dtype == np.int32
    # 手算:低 8 个权重 [0..7] 重排为 [0,2,4,6,1,3,5,7],逐 nibble 压位。
    expected0 = sum(v << (4 * i) for i, v in enumerate([0, 2, 4, 6, 1, 3, 5, 7]))
    expected1 = sum(v << (4 * i) for i, v in enumerate([8, 10, 12, 14, 9, 11, 13, 15]))
    assert int(packed[0]) == expected0  # 0x75316420
    assert int(packed[1]) == expected1 - (1 << 32)  # 0xFDB9ECA8 的 int32 读数


def test_awq_pack_round_trips_signed_codes():
    rng = np.random.default_rng(7)
    q = rng.integers(-8, 8, size=8 * 5)  # INT4 有符号码域 [-8, 7]
    packed = awq_pack(q, num_bits=4)
    np.testing.assert_array_equal(awq_unpack(packed, num_bits=4), q)
