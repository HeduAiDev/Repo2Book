# -*- coding: utf-8 -*-
"""训练目标三件套 —— §3.3 Eq.(9)-(12)(L_ce/L_tv/L_conf + 位置权重 + 总目标)。

手算基准:
- w_k = exp(−(k−1)/γ):γ=3 → [1, e^{−1/3}, e^{−2/3}];
- L_tv 手算:(0.5,0.5) vs (0.7,0.3) → ‖·‖₁ = 0.4;
- α 默认 0.1/0.9/1.0(Eq.12);
- 下降轨迹:L_tv ↓ ⇒ 逐位接受率 1−½‖p−q‖₁ ↑(§3.3"最小化 L_tv 直接最大化期望接受率")。
"""
import math

import numpy as np
import pytest

from training import (
    position_weights,
    cross_entropy_loss,
    tv_loss,
    confidence_loss,
    total_loss,
    train_toy_drafter,
)


def test_position_weights():
    assert np.allclose(position_weights(3), [1.0, math.exp(-1 / 3), math.exp(-2 / 3)])
    assert position_weights(1)[0] == 1.0
    # 单调不增:前位权重更大(前缀验证下前位贡献期望接受长度更多)
    w = position_weights(7)
    assert np.all(np.diff(w) < 0)


def test_cross_entropy_hand():
    # N=1、γ=1、w_1=1:−log p(x*)
    p = np.array([[[0.7, 0.3]]])
    x = np.array([[0]])
    assert cross_entropy_loss(p, x) == pytest.approx(-math.log(0.7))
    # γ=2 加权:−(1·log p_1(x_1*) + e^{−1/2}·log p_2(x_2*))
    p2 = np.array([[[0.7, 0.3], [0.6, 0.4]]])
    x2 = np.array([[0, 1]])
    expect = -(1.0 * math.log(0.7) + math.exp(-0.5) * math.log(0.4))
    assert cross_entropy_loss(p2, x2) == pytest.approx(expect)


def test_tv_loss_hand():
    q = np.array([[[0.5, 0.5], [0.5, 0.5]]])
    p = np.array([[[0.7, 0.3], [0.6, 0.4]]])
    # 位 1:‖·‖₁=0.4(w=1);位 2:0.2(w=e^{−1/2})
    assert tv_loss(q, p) == pytest.approx(0.4 + math.exp(-0.5) * 0.2)


def test_confidence_loss_hand():
    c = np.array([[0.9]])
    c_star = np.array([[0.8]])
    expect = -(0.8 * math.log(0.9) + 0.2 * math.log(0.1))
    assert confidence_loss(c, c_star) == pytest.approx(expect)


def test_total_loss_composition():
    rng = np.random.default_rng(0)
    p_d = np.array([[[0.6, 0.4], [0.3, 0.7]]])
    p_t = np.array([[[0.5, 0.5], [0.2, 0.8]]])
    x = np.array([[0, 1]])
    c = np.array([[0.7, 0.5]])
    c_star = np.array([[0.6, 0.9]])
    alphas = (0.2, 0.5, 0.3)
    expect = (
        0.2 * cross_entropy_loss(p_d, x)
        + 0.5 * tv_loss(p_d, p_t)
        + 0.3 * confidence_loss(c, c_star)
    )
    assert total_loss(p_d, x, p_t, c, c_star, *alphas) == pytest.approx(expect)
    # 默认权重 = Eq.(12):0.1/0.9/1.0
    default = total_loss(p_d, x, p_t, c, c_star)
    manual = (
        0.1 * cross_entropy_loss(p_d, x)
        + 0.9 * tv_loss(p_d, p_t)
        + 1.0 * confidence_loss(c, c_star)
    )
    assert default == pytest.approx(manual)


def test_train_toy_drafter_raises_acceptance():
    # 目标分布冻结,最小化 Eq.(12):L_tv 下降、逐位接受率(1−½‖p−q‖₁)与 E[τ] 上升
    target = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    trace = train_toy_drafter(target, n_steps=300, lr=0.5, rng=np.random.default_rng(1))
    assert len(trace) == 300
    assert trace[-1]["tv"] < trace[0]["tv"]
    assert trace[-1]["expected_tau"] > trace[0]["expected_tau"]
    assert trace[-1]["loss"] < trace[0]["loss"]
    rate0 = np.mean(trace[0]["accept_rates"])
    rate1 = np.mean(trace[-1]["accept_rates"])
    assert rate1 > rate0
