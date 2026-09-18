# -*- coding: utf-8 -*-
"""§2.1 验证规则 + §1 无损 + Eq.(1) 延迟账 —— 论文行为测试(TDD:先于实现书写)。

手算基准全部取自论文原文:
- 接受准则 min(1, p_t/p_d) 与残差 norm(max(0, p_t−p_d)):arXiv:2607.05147 §2.1;
- 无损断言("acceptance rule preserves the target distribution exactly"):§1;
- {A,B} 词表 p_t=(0.7,0.3)/p_d=(0.5,0.5) 手算组:Appendix A 的数字(Σmin=0.8);
- Eq.(1) 延迟账:T_draft=1ms/T_verify=8ms 的算例口径。
"""
import numpy as np
import pytest

from spec_decode import (
    softmax_lastdim,
    acceptance_probability,
    accepted,
    accepted_logspace,
    residual_distribution,
    gumbel_max_argmax,
    sample_recovered,
    acceptance_rate_sum_min,
    analytical_acceptance_rate,
    verify_block,
    verify_block_greedy,
    empirical_output_distribution,
    generate_stream,
    latency_per_token,
    speedup_vs_plain,
)

P_T = np.array([0.7, 0.3])
P_D = np.array([0.5, 0.5])


# ── softmax 算子 ──

def test_softmax_lastdim_rows_and_shift_invariance():
    x = np.array([[0.0, 0.0], [1.0, 3.0]])
    p = softmax_lastdim(x)
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert np.allclose(p[0], [0.5, 0.5])
    # 平移不变:每行加同一常数,分布不变
    assert np.allclose(softmax_lastdim(x + 7.5), p)


# ── 接受准则:§2.1 min(1, p_t/p_d) ──

def test_acceptance_probability_hand():
    # p_t(A)/p_d(A) = 0.7/0.5 = 1.4 → min(1,·)=1;B 位 0.3/0.5 = 0.6
    assert acceptance_probability(P_T, P_D, 0) == 1.0
    assert acceptance_probability(P_T, P_D, 1) == pytest.approx(0.6)


def test_accepted_u_form_min_implicit():
    # u∈[0,1) ⇒ min(1,·) 可省:比值≥1 时恒接受;比值 0.6 时接受 ⟺ u<0.6
    rng = np.random.default_rng(0)
    for _ in range(200):
        u = rng.random()
        assert accepted(0, P_T, P_D, u) is True or accepted(0, P_T, P_D, u)
        assert accepted(1, P_T, P_D, u) == (u < 0.6)
        for x in (0, 1):
            assert accepted(x, P_T, P_D, u) == (u < acceptance_probability(P_T, P_D, x))


def test_accepted_logspace_equivalence():
    # V2 kernel 姿态:log p(x) > log(u) + log q(x) 与概率空间判据逐点等价
    rng = np.random.default_rng(1)
    for _ in range(200):
        raw = rng.random(4) + 0.1
        p_t, p_d = raw / raw.sum(), (rng.random(4) + 0.1) / (rng.random(4) + 0.1).sum()
        x = int(rng.integers(4))
        u = rng.random()
        assert accepted_logspace(x, np.log(p_t), np.log(p_d), u) == accepted(x, p_t, p_d, u)


# ── 残差分布 + Gumbel-max 免归一化采样 ──

def test_residual_distribution_hand_and_identity():
    r = residual_distribution(P_T, P_D)
    assert np.allclose(r, [0.2, 0.0])
    # 拒绝质量 = ½‖p−q‖₁ = 残差总质量(无损证明的两半在此对账)
    rng = np.random.default_rng(2)
    for _ in range(30):
        a, b = rng.random(5) + 0.05, rng.random(5) + 0.05
        a, b = a / a.sum(), b / b.sum()
        assert residual_distribution(a, b).sum() == pytest.approx(0.5 * np.abs(a - b).sum())


def test_gumbel_max_normalization_invariance_and_distribution():
    w = np.array([0.2, 0.5, 0.3])
    rng = np.random.default_rng(3)
    exp_draws = rng.exponential(1.0, size=3)
    # argmax(prob/E) 对正缩放不变 ⇒ 归一化因子 Z 根本不用算
    assert gumbel_max_argmax(w, exp_draws) == gumbel_max_argmax(w / 7.5, exp_draws)
    # 统计等价:与按概率抽样同分布
    counts = np.zeros(3)
    for _ in range(20000):
        e = rng.exponential(1.0, size=3)
        counts[gumbel_max_argmax(w, e)] += 1
    assert np.allclose(counts / 20000, w, atol=0.015)


def test_sample_recovered_forced_cases():
    rng = np.random.default_rng(4)
    # 残差只在 A 上 → 恢复恒为 A(Appendix A:拒绝质量 0.2 全在 A)
    for _ in range(100):
        assert sample_recovered(P_T, P_D, rng) == 0
    # 3-token:残差集中在下标 2
    p3_t, p3_d = np.array([0.2, 0.5, 0.3]), np.array([0.4, 0.1, 0.5])
    assert np.allclose(residual_distribution(p3_t, p3_d), [0.0, 0.4, 0.0])
    assert sample_recovered(p3_t, p3_d, rng) == 1


# ── 一次验证:左到右 + 首拒即停 + 恢复 + bonus ──

def test_verify_block_all_accept_forced():
    # 比值 >1 的位恒接受(与 u 无关);bonus 行 (0,1) → bonus 恒 B
    rng = np.random.default_rng(5)
    target = np.array([[0.9, 0.1], [0.8, 0.2], [0.0, 1.0]])
    draft = [0, 0]
    q = np.array([[0.5, 0.5], [0.5, 0.5]])
    for _ in range(50):
        emitted, n_acc, info = verify_block(draft, q, target, rng)
        assert emitted == [0, 0, 1]
        assert n_acc == 2
        assert info["rejected_at"] is None and info["bonus"] == 1


def test_verify_block_reject_and_recover_forced():
    # 位 0:p_t(A)=0 → 比值 0 → 恒拒;残差 (0, 0.5) → 恢复恒 B
    rng = np.random.default_rng(6)
    target = np.array([[0.0, 1.0], [0.5, 0.5], [0.5, 0.5]])
    draft = [0, 0]
    q = np.array([[0.5, 0.5], [0.5, 0.5]])
    for _ in range(50):
        emitted, n_acc, info = verify_block(draft, q, target, rng)
        assert emitted == [1]
        assert n_acc == 0
        assert info["rejected_at"] == 0 and info["recovered"] == 1


def test_verify_block_first_rejection_truncates():
    # γ=3、位 1 被拒:发射恰 2 个(1 接受 + 1 恢复),位 2 不再评估(前缀语义)
    rng = np.random.default_rng(7)
    target = np.array([[0.9, 0.1], [0.05, 0.95], [0.5, 0.5], [0.5, 0.5]])
    draft = [0, 0, 0]
    q = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    n_trunc = 0
    for _ in range(200):
        emitted, n_acc, info = verify_block(draft, q, target, rng)
        assert len(emitted) in (2, 4)  # 位 1 拒(→2)或全收(→4);位 0 恒收
        if info["rejected_at"] == 1:
            n_trunc += 1
            assert emitted[0] == 0 and n_acc == 1
            assert emitted[1] == info["recovered"]
            assert len(emitted) == 2
    assert n_trunc > 0  # 位 1 比值 0.1,200 次里必有拒绝


# ── 无损:§1 断言的可运行检验 ──

def test_losslessness_single_position():
    # 单位 1-token 周期:经验输出分布 == p_t(Appendix A 的 (0.7,0.3) 组)
    rng = np.random.default_rng(8)
    emp = empirical_output_distribution(P_T, P_D, n=40000, rng=rng)
    assert np.allclose(emp, P_T, atol=0.015)
    # 一组 4-token 随机分布
    rng = np.random.default_rng(9)
    raw_t, raw_d = rng.random(4) + 0.1, rng.random(4) + 0.1
    p_t4, p_d4 = raw_t / raw_t.sum(), raw_d / raw_d.sum()
    emp4 = empirical_output_distribution(p_t4, p_d4, n=40000, rng=rng)
    assert np.allclose(emp4, p_t4, atol=0.015)


def test_losslessness_stream_bigram_and_tau():
    # 整流生成流:2 态 Markov target,qualitative drafter(0.6·target+0.4·uniform)。
    # 无损 ⟹ 流的 bigram 条件分布 == target 转移行;经验 τ == 1+Σ∏α(α 恒 0.92)
    T = np.array([[0.7, 0.3], [0.3, 0.7]])
    rng = np.random.default_rng(10)

    def drafter(prev):
        toks, rows = [], []
        p = prev
        for _ in range(3):
            q = 0.6 * T[p] + 0.4 * np.array([0.5, 0.5])  # 混入 uniform(V=2)
            q = q / q.sum()
            x = int(rng.choice(2, p=q))
            toks.append(x)
            rows.append(q)
            p = x
        return toks, np.asarray(rows)

    stream, stats = generate_stream(
        lambda prev: T[prev], drafter, anchor=0, n_tokens=20000, rng=rng
    )
    stream = np.asarray(stream)
    for c in (0, 1):
        mask = stream[:-1] == c
        nxt = stream[1:][mask]
        emp = np.array([(nxt == 0).mean(), (nxt == 1).mean()])
        assert np.allclose(emp, T[c], atol=0.02)
    # α = Σmin(T_row, q) = 0.92(两行对称;q=0.6·T+0.4·uniform=[0.62,0.38])
    # ⇒ E[每周期发射] = 1+0.92+0.92²+0.92³ ≈ 3.545
    assert stats["mean_emitted_per_cycle"] == pytest.approx(3.545, abs=0.06)
    assert 1.0 <= stats["mean_emitted_per_cycle"] <= 4.0


def test_greedy_stream_matches_target_greedy_rollout():
    # greedy draft(one-hot q)退化:流 ≡ target 的 greedy rollout(逐位 argmax)
    T = np.array([[0.7, 0.3], [0.2, 0.8]])
    rng = np.random.default_rng(11)

    def greedy_drafter(prev):
        toks = []
        p = prev
        for _ in range(3):
            q = 0.6 * T[p] + 0.4 * np.array([0.5, 0.5])
            x = int(np.argmax(q))
            toks.append(x)
            p = x
        return toks, None

    stream, _ = generate_stream(
        lambda prev: T[prev], greedy_drafter, anchor=0, n_tokens=50, rng=rng, greedy=True
    )
    # 期望:target greedy 链
    expect = []
    s = 0
    for _ in range(50):
        s = int(np.argmax(T[s]))
        expect.append(s)
    assert list(stream) == expect


def test_verify_block_greedy_unit():
    # greedy 验证:接受 ⟺ draft == target argmax;输出 ≡ target argmax 链
    rng = np.random.default_rng(12)
    for _ in range(50):
        rows = [rng.random(4) for _ in range(4)]
        rows = np.asarray([r / r.sum() for r in rows])
        q_rows = [rng.random(4) for _ in range(3)]
        draft = [int(np.argmax(q)) for q in q_rows]
        emitted, n_acc, info = verify_block_greedy(draft, rows)
        assert len(emitted) >= 1
        for i, tok in enumerate(emitted):
            assert tok == int(np.argmax(rows[i]))


# ── TV 恒等式 / Eq.(8) 解析接受率 ──

def test_analytical_acceptance_rate_tv_identity():
    # Σ_x min(p,q) = 1 − ½‖p−q‖₁;论文手算:0.8
    assert acceptance_rate_sum_min(P_T, P_D) == pytest.approx(0.8)
    assert analytical_acceptance_rate(P_T, P_D) == pytest.approx(0.8)
    rng = np.random.default_rng(13)
    for _ in range(30):
        a, b = rng.random(6) + 0.05, rng.random(6) + 0.05
        a, b = a / a.sum(), b / b.sum()
        assert acceptance_rate_sum_min(a, b) == pytest.approx(
            analytical_acceptance_rate(a, b), abs=1e-12
        )


# ── Eq.(1) 延迟账 ──

def test_latency_eq1_and_speedup():
    # T_draft=1ms、T_verify=8ms:τ=3 → 3ms/token(2.67x);τ=1.2 → 7.5ms(1.07x)
    assert latency_per_token(1.0, 8.0, 3.0) == pytest.approx(3.0)
    assert latency_per_token(1.0, 8.0, 1.2) == pytest.approx(7.5)
    assert speedup_vs_plain(1.0, 8.0, 3.0) == pytest.approx(8.0 / 3.0)
    assert speedup_vs_plain(1.0, 8.0, 1.2) == pytest.approx(8.0 / 7.5)
