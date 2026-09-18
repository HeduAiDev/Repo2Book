# -*- coding: utf-8 -*-
"""τ 期望接受长度与前缀存活链式法则 —— §2.1/§4.2 Table 1(fn.4 含 bonus 口径)/§3.2.2。

手算基准:
- c=[0.8,0.9,0.5] → a=[0.8,0.72,0.36](§3.2.2 链式法则,论文 §3.2.2 原式 a_{r,j}=∏c_{r,i});
- α=0.8/γ=3 → E[τ]=(1−0.8⁴)/0.2 = 2.952(§2.1/§4.2 恒 α 闭式);
- 条件⇄无条件互推:vLLM v1/spec_decode/utils.py:L598-L601 的 c_i=p_i/p_{i−1}(p_0=1) 同构。
"""
import numpy as np
import pytest

from acceptance_length import (
    prefix_survival_probs,
    expected_accepted_length,
    expected_accepted_length_constant,
    conditional_to_unconditional_rates,
    unconditional_to_conditional_rates,
    simulate_accepted_length,
)


def test_prefix_survival_hand():
    assert np.allclose(prefix_survival_probs([0.8, 0.9, 0.5]), [0.8, 0.72, 0.36])


def test_expected_len_const_alpha_hand():
    # α=0.8、γ=3:1 + 0.8 + 0.64 + 0.512 = 2.952(论文 §4.2/Table 1 口径,fn.4 含 bonus)
    assert expected_accepted_length([0.8, 0.8, 0.8]) == pytest.approx(2.952)
    assert expected_accepted_length_constant(0.8, 3) == pytest.approx(2.952)


def test_expected_len_general_matches_closed_form():
    for alpha in (0.5, 0.7, 0.8, 0.9, 0.95):
        for gamma in (1, 2, 3, 4, 5):
            gen = expected_accepted_length([alpha] * gamma)
            closed = expected_accepted_length_constant(alpha, gamma)
            assert gen == pytest.approx(closed), (alpha, gamma)
    # α=1 边界:恒接受 → 每周期 γ+1 个 token
    assert expected_accepted_length_constant(1.0, 4) == 5


def test_unconditional_conditional_roundtrip():
    # 无条件 → 条件(vLLM unconditional_to_conditional_rates 语义:c_i = p_i/p_{i−1},p_0=1)
    got = unconditional_to_conditional_rates([0.8, 0.72, 0.36])
    assert np.allclose(got, [0.8, 0.9, 0.5])
    # 条件 → 无条件(链式法则正向 = §3.2.2 的 a_{r,j}=∏c_{r,i})
    assert np.allclose(conditional_to_unconditional_rates([0.8, 0.9, 0.5]), [0.8, 0.72, 0.36])
    # 往返恒等
    rng = np.random.default_rng(0)
    for _ in range(20):
        c = rng.random(5) * 0.9 + 0.05
        assert np.allclose(
            unconditional_to_conditional_rates(list(conditional_to_unconditional_rates(c))), c
        )
    # vLLM 边界语义:前一位为 0 → 该位条件率 0(除零守卫)
    assert unconditional_to_conditional_rates([0.5, 0.0, 0.0]) == [0.5, 0.0, 0.0]


def test_monte_carlo_matches_formula():
    # 抽象接受过程(位 k 以 α_k 接受、首拒即停、发射=接受数+1)经验均值 == 公式
    rng = np.random.default_rng(1)
    rates = [0.8, 0.9, 0.5]
    emp = simulate_accepted_length(rates, n_cycles=30000, rng=rng)
    assert emp == pytest.approx(expected_accepted_length(rates), abs=0.03)
    emp_const = simulate_accepted_length([0.8, 0.8, 0.8], n_cycles=30000, rng=rng)
    assert emp_const == pytest.approx(2.952, abs=0.03)
