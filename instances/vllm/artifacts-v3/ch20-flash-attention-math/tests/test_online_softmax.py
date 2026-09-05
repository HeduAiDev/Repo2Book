"""arXiv:1805.02867 (Milakov & Gimelshein) §2-§3 —— naive(两遍扫描,会溢出)→ safe
(三遍扫描,减 max 稳定)→ online(单遍扫描,同样稳定)三版 softmax 收敛到同一结果
(Theorem 1);§3.1 Eq.(3)-(4) —— ⊕ 二元算子的结合律/交换律与分块归并等价性。
测试先于实现书写(TDD),每条断言对应论文的一句可复现声明。"""
import numpy as np
import pytest

from online_softmax import (
    combine_blocks_via_merge,
    naive_softmax,
    online_softmax,
    online_softmax_merge,
    online_softmax_stats,
    safe_softmax,
)


def test_naive_safe_online_agree_on_well_scaled_input():
    # §3:"The algorithm 3 is proved to compute the Softmax function as defined in
    # (2)" —— 数值良好的输入上三版末值恒等(Theorem 1)。
    x = np.array([1.0, 2.0, 3.0, 0.5, -1.0])
    y_naive = naive_softmax(x)
    y_safe = safe_softmax(x)
    y_online = online_softmax(x)
    np.testing.assert_allclose(y_safe, y_naive, rtol=1e-6)
    np.testing.assert_allclose(y_online, y_safe, rtol=1e-6)
    assert np.isclose(y_online.sum(), 1.0)


def test_naive_softmax_overflows_on_large_logits_but_safe_and_online_do_not():
    # §2 原话:"the line 3 of the algorithm 1 can overflow or underflow due to the
    # exponent" —— naive 在 e^1000 处上溢(inf/inf→nan);safe/online 减 max 后不溢出。
    x = np.array([1000.0, 1001.0, 999.0])
    with np.errstate(over="ignore", invalid="ignore"):
        y_naive = naive_softmax(x)
    assert not np.all(np.isfinite(y_naive))

    y_safe = safe_softmax(x)
    y_online = online_softmax(x)
    assert np.all(np.isfinite(y_safe))
    assert np.all(np.isfinite(y_online))
    np.testing.assert_allclose(y_online, y_safe, rtol=1e-6)


def test_online_softmax_stats_match_theorem1_definition():
    # Theorem 1:Alg.3 lines 1-6 compute m_V = max_k x_k and d_V = Σ_j e^{x_j - m_V}。
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    m_v, d_v = online_softmax_stats(x)
    assert np.isclose(m_v, x.max())
    assert np.isclose(d_v, np.exp(x - x.max()).sum())


def test_online_softmax_stats_trajectory_records_running_states():
    # Alg.3 line 4-5 逐元素递推的中间态轨迹(示教素材):每步 (m_j, d_j) 既满足
    # line 5 的 rescale 递推逐字形式,又满足 Theorem 1 的前缀版不变式
    # d_j = Σ_{k<=j} e^{x_k - m_j} ——「旧账折算到新最大值」在轨迹里逐格可验。
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    trace = []
    online_softmax_stats(x, trace=trace)
    assert len(trace) == len(x)
    m_prev, d_prev = -np.inf, 0.0
    for j, (m_j, d_j) in enumerate(trace):
        xj = x[j]
        # line 4: m_j = max(m_{j-1}, x_j)
        assert np.isclose(m_j, max(m_prev, xj))
        # line 5: d_j = d_{j-1} * e^{m_{j-1}-m_j} + e^{x_j-m_j}
        assert np.isclose(
            d_j, d_prev * np.exp(m_prev - m_j) + np.exp(xj - m_j), rtol=1e-12
        )
        # Theorem 1 前缀版:等价于对 x[:j+1] 一次性算 (max, Σ e^{x-m})
        prefix = x[: j + 1]
        assert np.isclose(m_j, prefix.max())
        assert np.isclose(d_j, np.exp(prefix - m_j).sum(), rtol=1e-9)
        m_prev, d_prev = m_j, d_j


def test_online_softmax_running_sum_stays_bounded_1_to_j():
    # §3 数值界原话:"d_j is also bounded: 1 <= d_j <= j, forall j in [1,V]"
    x = np.array([2.0, 7.0, -3.0, 7.0, 0.0, 5.0])
    trace = []
    online_softmax_stats(x, trace=trace)
    for j, (m_j, d_j) in enumerate(trace, start=1):
        assert 1.0 <= d_j <= float(j)


def test_merge_operator_matches_sequential_two_pass_recurrence():
    # §3.1 Eq.(3):"Applying (3) sequentially from left to right is equivalent to
    # running lines 1-6 of the algorithm 3" —— 对拼接向量整体单遍遍历。
    x1 = np.array([2.0, 5.0, -1.0])
    x2 = np.array([0.5, 7.0, 3.0, -2.0])
    m1, d1 = online_softmax_stats(x1)
    m2, d2 = online_softmax_stats(x2)
    m_merged, d_merged = online_softmax_merge((m1, d1), (m2, d2))

    m_full, d_full = online_softmax_stats(np.concatenate([x1, x2]))
    assert np.isclose(m_merged, m_full)
    assert np.isclose(d_merged, d_full)


def test_merge_operator_is_commutative():
    # §3.1:"It is also commutative, which provides the flexibility needed to make
    # parallel implementations more efficient."
    m1, d1 = 3.0, 2.5
    m2, d2 = 5.0, 1.2
    a_then_b = online_softmax_merge((m1, d1), (m2, d2))
    b_then_a = online_softmax_merge((m2, d2), (m1, d1))
    assert np.isclose(a_then_b[0], b_then_a[0])
    assert np.isclose(a_then_b[1], b_then_a[1])


def test_merge_operator_is_associative():
    # §3.1:"The operation ⊕ is associative, which enables parallel evaluation of (3)."
    m1, d1 = 3.0, 2.5
    m2, d2 = 5.0, 1.2
    m3, d3 = -1.0, 0.7
    left = online_softmax_merge(online_softmax_merge((m1, d1), (m2, d2)), (m3, d3))
    right = online_softmax_merge((m1, d1), online_softmax_merge((m2, d2), (m3, d3)))
    assert np.isclose(left[0], right[0])
    assert np.isclose(left[1], right[1])


@pytest.mark.parametrize("block_size", [1, 2, 3, 7])
def test_combine_blocks_via_merge_equals_one_pass_online_softmax(block_size):
    # Eq.(3) 的链式 ⊕ 归并 == Alg.3 单遍顺序遍历(worked example 的核心断言):
    # 任意分块大小下,分块归并得到的 (m_V, d_V) 与一遍扫完逐位一致。
    rng = np.random.default_rng(0)
    x = rng.normal(size=20)
    m_ref, d_ref = online_softmax_stats(x)
    m_blocked, d_blocked = combine_blocks_via_merge(x, block_size)
    assert np.isclose(m_blocked, m_ref)
    assert np.isclose(d_blocked, d_ref, rtol=1e-6)


def test_combine_blocks_out_of_order_still_agrees_with_in_order():
    # 交换律的落地形态:乱序归并块 == 顺序归并块(并行实现的合法性)。
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    blocks = [x[0:2], x[2:4], x[4:6]]
    in_order = online_softmax_merge(
        online_softmax_merge(online_softmax_stats(blocks[0]), online_softmax_stats(blocks[1])),
        online_softmax_stats(blocks[2]),
    )
    shuffled = online_softmax_merge(
        online_softmax_merge(online_softmax_stats(blocks[2]), online_softmax_stats(blocks[0])),
        online_softmax_stats(blocks[1]),
    )
    assert np.isclose(in_order[0], shuffled[0])
    assert np.isclose(in_order[1], shuffled[1])
