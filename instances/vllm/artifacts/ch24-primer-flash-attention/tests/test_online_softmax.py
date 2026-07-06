"""arXiv:1805.02867 §2-3 —— naive/safe/online softmax 三遍→一遍收敛到同一结果,
以及 §3.1 Eq.4 的 ⊕ 合并算子(结合律/交换律)。"""
import numpy as np
import pytest

from online_softmax import (
    naive_softmax,
    safe_softmax,
    online_softmax,
    online_softmax_stats,
    online_softmax_merge,
    combine_blocks_via_merge,
)


def test_naive_safe_online_agree_on_well_scaled_input():
    x = np.array([1.0, 2.0, 3.0, 0.5, -1.0])
    y_naive = naive_softmax(x)
    y_safe = safe_softmax(x)
    y_online = online_softmax(x)
    np.testing.assert_allclose(y_safe, y_naive, rtol=1e-6)
    np.testing.assert_allclose(y_online, y_safe, rtol=1e-6)
    assert np.isclose(y_online.sum(), 1.0)


def test_naive_softmax_overflows_on_large_logits_but_safe_and_online_do_not():
    # 论文 §2:naive 版本(Algorithm 1)在数值范围有限的真实硬件上会上溢/下溢。
    x = np.array([1000.0, 1001.0, 999.0])
    with np.errstate(over="ignore", invalid="ignore"):
        y_naive = naive_softmax(x)
    assert not np.all(np.isfinite(y_naive))  # naive: inf/inf -> nan

    y_safe = safe_softmax(x)
    y_online = online_softmax(x)
    assert np.all(np.isfinite(y_safe))
    assert np.all(np.isfinite(y_online))
    np.testing.assert_allclose(y_online, y_safe, rtol=1e-6)


def test_online_softmax_stats_match_theorem1_definition():
    # Theorem 1: lines 1-6 of Algorithm 3 compute m_V=max(x), d_V=sum(exp(x-m_V))
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    m_v, d_v = online_softmax_stats(x)
    assert np.isclose(m_v, x.max())
    assert np.isclose(d_v, np.exp(x - x.max()).sum())


def test_merge_operator_matches_sequential_two_pass_recurrence():
    # Eq.3: applying (3) sequentially left-to-right == running lines 1-6 of Algorithm 3
    # on the concatenated vector.
    x1 = np.array([2.0, 5.0, -1.0])
    x2 = np.array([0.5, 7.0, 3.0, -2.0])
    m1, d1 = online_softmax_stats(x1)
    m2, d2 = online_softmax_stats(x2)
    m_merged, d_merged = online_softmax_merge((m1, d1), (m2, d2))

    m_full, d_full = online_softmax_stats(np.concatenate([x1, x2]))
    assert np.isclose(m_merged, m_full)
    assert np.isclose(d_merged, d_full)


def test_merge_operator_is_commutative():
    m1, d1 = 3.0, 2.5
    m2, d2 = 5.0, 1.2
    a_then_b = online_softmax_merge((m1, d1), (m2, d2))
    b_then_a = online_softmax_merge((m2, d2), (m1, d1))
    assert np.isclose(a_then_b[0], b_then_a[0])
    assert np.isclose(a_then_b[1], b_then_a[1])


def test_merge_operator_is_associative():
    m1, d1 = 3.0, 2.5
    m2, d2 = 5.0, 1.2
    m3, d3 = -1.0, 0.7
    left = online_softmax_merge(online_softmax_merge((m1, d1), (m2, d2)), (m3, d3))
    right = online_softmax_merge((m1, d1), online_softmax_merge((m2, d2), (m3, d3)))
    assert np.isclose(left[0], right[0])
    assert np.isclose(left[1], right[1])


@pytest.mark.parametrize("block_size", [1, 2, 3, 7])
def test_combine_blocks_via_merge_equals_one_pass_online_softmax(block_size):
    # 分块乱序合并 == 顺序遍历 == 一次性 softmax(worked-example 的核心断言)。
    rng = np.random.default_rng(0)
    x = rng.normal(size=20)
    m_ref, d_ref = online_softmax_stats(x)
    m_blocked, d_blocked = combine_blocks_via_merge(x, block_size)
    assert np.isclose(m_blocked, m_ref)
    assert np.isclose(d_blocked, d_ref, rtol=1e-6)


def test_combine_blocks_out_of_order_still_agrees_with_in_order():
    # 交换律:乱序归并块也得到同一结果。
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
