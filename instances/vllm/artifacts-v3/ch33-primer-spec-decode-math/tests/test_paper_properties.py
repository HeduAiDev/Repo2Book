# -*- coding: utf-8 -*-
"""论文性质补充闸门（tester 站）—— 对 dossier 各机制的论文断言逐条锁性质。

每条性质注明论文锚（arXiv:2607.05147 §/Eq）：
- 分布保持类（无损/bonus 分布/贪心流）→ 固定随机种子的统计检验（宽松阈值防 flaky）；
- 恒等/结构性类（固定 B 贪心最优、位 1 杠杆）→ 数值对照（含暴力枚举对拍）；
- 优化类（Eq.(12) 轨迹忠实）→ 目标函数逐项对照（回归锁：形状广播不得改写 Eq.(11)）。
"""
import itertools

import numpy as np
import pytest

from spec_decode import (
    verify_block,
    analytical_acceptance_rate,
)
from acceptance_length import expected_accepted_length
from training import _loss_of, total_loss
from spec_decode import softmax_lastdim


# ── §1 无损（分布保持类）：随机分布族扫描 ──
# PAPER: §1 — "the acceptance rule preserves the target distribution exactly"
def test_losslessness_random_family_sweep():
    # 5 组随机 4-token 分布（固定种子）：经验输出分布 == p_t
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        raw_t, raw_d = rng.random(4) + 0.1, rng.random(4) + 0.1
        p_t, p_d = raw_t / raw_t.sum(), raw_d / raw_d.sum()
        counts = np.zeros(4)
        n = 20000
        for _ in range(n):
            x = int(rng.choice(4, p=p_d))
            u = float(rng.random())
            if p_d[x] > 0.0 and u * p_d[x] < p_t[x]:
                y = x
            else:
                r = np.maximum(p_t - p_d, 0.0)
                y = int(rng.choice(4, p=r / r.sum()))
            counts[y] += 1
        assert np.allclose(counts / n, p_t, atol=0.02), (seed, counts / n, p_t)


# ── §2.1 周期产出的组成恒等（结构性）：output = accepted + (recovered|bonus) ──
# PAPER: §2.1 + vLLM 术语表（rejection_sampler.py:L38-L59）
def test_verify_block_emission_invariant():
    # 任意试验:发射数 == 接受数 + 1(拒绝位的恢复 token 或全收的 bonus);
    # bonus 存在 ⟺ 无拒绝
    rng = np.random.default_rng(200)
    for _ in range(400):
        gamma = 3
        draft = [int(rng.integers(4)) for _ in range(gamma)]
        q = np.asarray([rng.random(4) + 0.05 for _ in range(gamma)])
        q = q / q.sum(axis=1, keepdims=True)
        tp = np.asarray([rng.random(4) + 0.05 for _ in range(gamma + 1)])
        tp = tp / tp.sum(axis=1, keepdims=True)
        emitted, n_acc, info = verify_block(draft, q, tp, rng)
        assert len(emitted) == n_acc + 1
        assert (info["bonus"] is not None) == (info["rejected_at"] is None)
        if info["rejected_at"] is not None:
            assert len(emitted) == info["rejected_at"] + 1


# PAPER: §2.1 — bonus 只从 target 分布采（"The bonus token is only sampled from
#   the target probabilities",vLLM docstring 同口径）
def test_bonus_sampled_from_target_distribution():
    # 位 0 恒接受(比值 1.4>1)→ bonus 行 [0.7,0.3] ⇒ bonus 经验分布 ≈ [0.7,0.3]
    rng = np.random.default_rng(201)
    target = np.array([[0.7, 0.3], [0.7, 0.3]])
    q = np.array([[0.5, 0.5]])
    counts = np.zeros(2)
    n = 20000
    for _ in range(n):
        emitted, _, info = verify_block([0], q, target, rng)
        assert info["rejected_at"] is None
        counts[info["bonus"]] += 1
    assert np.allclose(counts / n, [0.7, 0.3], atol=0.015)


# ── §4.3.1 位 1 容量优势 × 前缀生存杠杆（结构性,E[τ] 公式的推论）──
# PAPER: §4.3.1 + Fig.2 + Table 1 — 并行 drafter 逐位后段更差、总 τ 却反超:
#   "this initial capacity advantage disproportionately boosts the final
#    accepted length"
def test_position1_leverage_beats_suffix_decay():
    gamma = 7  # 块长 7(§4.1:与 DFlash/DSpark 对齐的 block size)
    # Fig.2 Chat 端点文字数:并行 0.72→0.63(后缀衰减) vs 自回归 0.53→0.74(回升)
    parallel = np.linspace(0.72, 0.63, gamma)
    ar = np.linspace(0.53, 0.74, gamma)
    e_par = expected_accepted_length(parallel)
    e_ar = expected_accepted_length(ar)
    assert e_par > e_ar  # Table 1 Chat 域方向:DFlash ~2.8-3.1 > Eagle3 ~2.3-2.6
    # 结构对照:把自回归末位抬到 0.87(平均条件率 0.70 > 并行 0.675),
    # 位 1 仍是 0.53 → E[τ] 仍输:杠杆在「位 1 × 累积乘积」,不在平均率
    ar_highmean = np.linspace(0.53, 0.87, gamma)
    assert ar_highmean.mean() > parallel.mean()
    assert expected_accepted_length(parallel) > expected_accepted_length(ar_highmean)


# ── §3.2.2 固定 B 时的贪心最优性（恒等类:交换论证的数值对照）──
# PAPER: §3.2.2 — "if the total verification batch size B were fixed, the
#   optimal allocation {ℓ_r} would be determined by greedily selecting the
#   draft tokens with the highest survival probabilities from the global pool"
def test_fixed_budget_greedy_is_optimal():
    rng = np.random.default_rng(202)
    for trial in range(5):
        R, gamma = 3, 4
        conf = [[float(rng.random() * 0.9 + 0.05) for _ in range(gamma)] for _ in range(R)]
        surv = [np.cumprod(c) for c in conf]
        all_a = sorted(
            (float(surv[r][j]) for r in range(R) for j in range(gamma)), reverse=True
        )
        for m in (2, 4, 6):  # 额外验证 token 预算 m(总 B = R + m 固定)
            tau_greedy = R + sum(all_a[:m])  # 全局按 a 取 top-m(单调性⇒前缀自动满足)
            best = -1.0
            for ell in itertools.product(range(gamma + 1), repeat=R):
                if sum(ell) != m:
                    continue
                tau = sum(1 + float(surv[r][: ell[r]].sum()) for r in range(R))
                best = max(best, tau)
            assert tau_greedy == pytest.approx(best), (trial, m)


# ── §3.3 Eq.(11)(12) 轨迹忠实（优化类回归锁）──
# PAPER: §3.3 Eq.(11)(12) — 训练目标 L = α_ce·L_ce + α_tv·L_tv + α_conf·L_conf,
#   L_conf 是逐位 BCE(c_k 对 c*_k)。玩具轨迹的 loss 必须逐值等于正确形状的
#   Eq.(12)——广播形状事故(外积化)会静默改写目标函数,属论文失真。
def test_toy_trajectory_loss_follows_eq12():
    gamma, vocab = 3, 4
    target_probs = np.array([
        [0.7, 0.2, 0.1, 0.0],
        [0.1, 0.7, 0.1, 0.1],
        [0.2, 0.1, 0.6, 0.1],
    ])
    target_tokens = np.array([0, 1, 2])
    rng = np.random.default_rng(203)
    z = rng.normal(0.0, 1.0, size=(gamma, vocab))
    zc = rng.normal(0.0, 1.0, size=gamma)
    alphas = (0.1, 0.9, 1.0)

    got = _loss_of(z, zc, target_probs, target_tokens, alphas)

    # 参考:按 Eq.(9)-(12) 逐式正确形状复算
    p = softmax_lastdim(z)
    c = 1.0 / (1.0 + np.exp(-zc))
    c_star = np.array(
        [analytical_acceptance_rate(target_probs[k], p[k]) for k in range(gamma)]
    )
    want = total_loss(p[None], target_tokens[None], target_probs[None], c[None],
                      c_star[None], *alphas)
    assert got == pytest.approx(want), (
        "train_toy_drafter 的目标函数偏离 Eq.(12):疑似 confidence_loss 内的"
        "形状广播事故(c_star (γ,1) × c (1,γ) → 外积 (1,γ,γ)),L_conf 被静默改写"
    )
