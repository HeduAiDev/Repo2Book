# -*- coding: utf-8 -*-
"""置信度调度 —— §3.2.1(STS 校准/ECE)+ §3.2.2(Algorithm 1 硬件感知前缀调度器)
+ Appendix A(回顾式选择偏差反例)。

手算基准(论文原数字):
- App A:SPS(1)=1.0/SPS(2)=0.5/SPS(3)=0.45、a_1=0.8 → Θ_0=1.0、Θ_1=0.9(早停在 ℓ=0);
  c_2=0.9 → Θ_2=(1+0.8+0.72)·0.45=1.134(全局最大→ℓ=2);c_2=0 → Θ_2=0.81(→ℓ=0);
  输出偏差 P(Y=A)=0.5·1+0.5·0.7=0.85≠0.7;
- a_{r,j}=∏c_{r,i} 单调不增 ⇒ 全局按 a 排序天然尊重前缀依赖(贪心合法性);
- STS:逐位 1D 网格搜温度,最小化累积乘积 ECE,保序。
"""
import numpy as np
import pytest

from scheduler import (
    hardware_aware_prefix_scheduler,
    retrospective_global_search,
    retrospective_output_distribution,
    expected_calibration_error,
    temperature_scaled_confidence,
    sequential_temperature_scaling,
)
from spec_decode import empirical_output_distribution


SPS_APP_A = {1: 1.0, 2: 0.5, 3: 0.45}


# ── Algorithm 1:硬件感知前缀调度器 ──

def test_algorithm1_appendix_a_early_stop():
    # R=1、c=[0.8]:Θ_0=1·1.0=1.0;准入位 1 后 Θ=(1+0.8)·0.5=0.9<1.0 → break → ℓ*=0
    res = hardware_aware_prefix_scheduler([[0.8]], lambda B: SPS_APP_A[B])
    assert res["ell_star"] == [0]
    assert res["theta_best"] == pytest.approx(1.0)
    assert res["trace"][-1]["event"] == "break"


def test_algorithm1_causal_even_when_retrospective_would_admit():
    # 同一场景、实现 c_2=0.9(Θ_2=1.134 是全局最大):因果版仍在首跌 break → ℓ*=0。
    # 这就是 App A 的点:回顾式会改判 ℓ=2、泄露 x_1
    res = hardware_aware_prefix_scheduler([[0.8, 0.9]], lambda B: SPS_APP_A[B])
    assert res["ell_star"] == [0]


def test_algorithm1_light_load_admits_everything():
    # 轻载:SPS 近平坦(验证近免费)→ 所有正存活位全准入
    def sps(B):
        return 1.0 if B <= 12 else 0.5

    conf = [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]
    res = hardware_aware_prefix_scheduler(conf, sps)
    assert res["ell_star"] == [3, 3]
    a = np.cumprod(conf[0]).sum() + np.cumprod(conf[1]).sum()
    assert res["theta_best"] == pytest.approx((2 + a) * 1.0)


def test_algorithm1_matches_bruteforce_unimodal():
    # 单峰 Θ 下早停贪心 = 全局最优:对拍全枚举
    for seed, lam in [(0, 0.05), (1, 0.05), (2, 0.1), (3, 0.1), (4, 0.2),
                      (5, 0.05), (6, 0.1), (7, 0.15), (8, 0.05), (9, 0.1)]:
        rng = np.random.default_rng(seed)
        R = int(rng.integers(1, 4))
        gamma = int(rng.integers(2, 4))
        conf = [[float(rng.random() * 0.9 + 0.05) for _ in range(gamma)] for _ in range(R)]

        def sps(B, lam=lam):
            return float(np.exp(-lam * B))

        surv = [np.cumprod(c) for c in conf]
        # 全枚举 ℓ_1..ℓ_R
        best = -1.0
        import itertools
        for ell in itertools.product(range(gamma + 1), repeat=R):
            tau = sum(1 + float(surv[r][: ell[r]].sum()) for r in range(R))
            B = sum(1 + ell[r] for r in range(R))
            best = max(best, tau * sps(B))
        res = hardware_aware_prefix_scheduler(conf, sps)
        assert res["theta_best"] == pytest.approx(best), (seed, lam, conf)


def test_prefix_survival_monotone_non_increasing():
    # a_{r,j} ≤ a_{r,j−1}(贪心准入尊重前缀依赖的根据)
    rng = np.random.default_rng(20)
    for _ in range(50):
        c = rng.random(6) * 0.95 + 0.03
        a = np.cumprod(c)
        assert np.all(np.diff(a) <= 1e-15)


# ── Appendix A:回顾式全局搜索 + 分布偏差 ──

def test_retrospective_global_search_app_a():
    ell, thetas = retrospective_global_search(a1=0.8, c2=0.9, sps=lambda B: SPS_APP_A[B])
    assert thetas == pytest.approx([1.0, 0.9, 1.134])
    assert ell == 2  # 全局最大 → 准入 x_1
    ell0, thetas0 = retrospective_global_search(a1=0.8, c2=0.0, sps=lambda B: SPS_APP_A[B])
    assert thetas0 == pytest.approx([1.0, 0.9, 0.81])
    assert ell0 == 0  # 全局最大仍是 Θ_0 → 不准入 x_1


def test_retrospective_distribution_bias_app_a():
    # 回顾式:x_1=A(高 c_2)→ℓ=2、必被接受;x_1=B(低 c_2)→ℓ=0、target 重采
    # ⇒ P(Y=A)=0.5·1+0.5·0.7=0.85 ≠ 0.7(不无损)
    rng = np.random.default_rng(21)
    p_t, p_d = np.array([0.7, 0.3]), np.array([0.5, 0.5])
    emp = retrospective_output_distribution(
        p_t, p_d, c2_high=0.9, c2_low=0.0, sps=lambda B: SPS_APP_A[B], n=40000, rng=rng
    )
    assert np.allclose(emp, [0.85, 0.15], atol=0.015)
    # 对照:标准(恒准入)验证 ≡ p_t —— 无损只在「准入不依赖未来」时成立
    rng2 = np.random.default_rng(22)
    emp_std = empirical_output_distribution(p_t, p_d, n=40000, rng=rng2)
    assert np.allclose(emp_std, p_t, atol=0.015)


# ── §3.2.1 ECE + STS ──

def test_ece_hand():
    preds = [0.9, 0.9, 0.9, 0.3, 0.3]
    outs = [1, 1, 0, 0, 0]
    # 2 bins([0,0.5)/[0.5,1]):bin_lo conf 0.3/acc 0(2 个);bin_hi conf 0.9/acc 2/3(3 个)
    # ECE = 0.4·0.3 + 0.6·|2/3−0.9| = 0.12 + 0.14 = 0.26
    assert expected_calibration_error(preds, outs, num_bins=2) == pytest.approx(0.26)


def test_temperature_scaled_confidence():
    assert temperature_scaled_confidence(2.0, 2.0) == pytest.approx(1.0 / (1.0 + np.exp(-1.0)))
    # T>1 → 拉向 0.5;保序(单调)
    z = np.array([-3.0, 0.0, 3.0])
    c1 = temperature_scaled_confidence(z, 1.0)
    c2 = temperature_scaled_confidence(z, 3.0)
    assert np.all(np.diff(c1) > 0) and np.all(np.diff(c2) > 0)
    assert c2.max() < c1.max()


def test_sts_calibrates_overconfidence():
    # 两上下文混合、统一过自信(z = 2·logit(rate)):STS 应逐位找到 T≈2,
    # 校准后累积乘积均值对齐真实前缀存活 [0.65, 0.51, 0.225],ECE 显著下降
    rng = np.random.default_rng(23)
    rates = np.array([[0.8, 0.9, 0.5], [0.5, 0.6, 0.3]])
    N_half = 6000
    z, outcomes = [], []
    for ctx in (0, 1):
        z.append(2.0 * (np.log(rates[ctx] / (1 - rates[ctx]))))
    z = np.asarray(z)
    alive = np.ones((2, N_half), dtype=bool)
    outs = np.zeros((2, N_half, 3), dtype=bool)
    for ctx in (0, 1):
        for k in range(3):
            alive[ctx] &= rng.random(N_half) < rates[ctx, k]
            outs[ctx, :, k] = alive[ctx]
    conf_logits = np.repeat(z, N_half, axis=0)          # [N, γ]
    prefix_outcomes = np.concatenate(outs, axis=0)       # [N, γ]
    res = sequential_temperature_scaling(conf_logits, prefix_outcomes)
    raw_cum = np.cumprod(1.0 / (1.0 + np.exp(-conf_logits)), axis=1)
    ece_before = expected_calibration_error(raw_cum[:, -1], prefix_outcomes[:, -1])
    ece_after = expected_calibration_error(res["cumulative"][:, -1], prefix_outcomes[:, -1])
    assert ece_after < ece_before
    assert np.allclose(np.mean(res["cumulative"], axis=0), [0.65, 0.51, 0.225], atol=0.02)
    assert np.all(res["temperatures"] > 1.2) and np.all(res["temperatures"] < 3.5)


def test_sts_order_preserving():
    # 温度缩放保序:位 k 上样本间的排序不变(不扰动置信头学到的 token 排序)
    rng = np.random.default_rng(24)
    z = rng.normal(size=(200, 3)) * 2
    outs = rng.random((200, 3)) < 0.5
    res = sequential_temperature_scaling(z, outs)
    raw = 1.0 / (1.0 + np.exp(-z))
    for k in range(3):
        assert np.allclose(np.argsort(raw[:, k]), np.argsort(res["calibrated"][:, k]))
