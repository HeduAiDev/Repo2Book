"""核心注意力 + 滑窗 + 位置细节测试 —— 论文锚：
arXiv:2606.19348 Eq.(18)(19)（共享 KV 的 MQA）、Eq.(24)-(26)（HCA 侧同构）、
§2.3.3（滑窗 n_win、严格因果、attention sink、RoPE 最后 64 维与 position −i 反向旋转）。
"""
import numpy as np
import pytest

from indexer import indexer_queries
from windowed_attention import (
    compressed_entry_count,
    compressed_entry_positions,
    compose_attention_inputs,
    core_attention_mqa,
    core_queries,
    rope_forward,
    rope_inverse,
    sliding_window_slice,
    softmax_with_sink,
)

RNG = np.random.default_rng(1934)


# ── Eq.(18)：主注意力 q 与索引器 q 共用同一份 c^Q（本章隐藏主线） ───────
def test_core_queries_reuses_the_same_latent_vector_as_the_indexer():
    """论文原话：`Note that the latent query vector c_t^Q is shared with that used for
    the indexer queries`——**一次降维、两处消费**。

    判据（数值）：从同一个 `c_q` 出发，索引器支路给 `(n_h^I, c^I)` 的小头 q、核心注意力
    支路给 `(n_h, c)` 的大头 q；两条支路都对**同一个** `c_q` 做线性升维 ⇒ 只要把 `c_q`
    改掉，两边同时变（共用是结构事实，不是两次独立降维碰巧同值）。
    """
    d, d_c = 6, 4
    n_heads_i, c_i = 3, 2      # 索引器：n_h^I 头、c^I 头维（config 口径 64 × 128）
    n_heads, c = 5, 8          # 主注意力：n_h 头、c = head_dim（config 口径 n_h × 512）
    h = RNG.normal(size=(d,)).round(2)
    W_DQ = RNG.normal(size=(d, d_c)).round(2)
    W_IUQ = RNG.normal(size=(d_c, c_i * n_heads_i)).round(2)
    W_UQ = RNG.normal(size=(d_c, c * n_heads)).round(2)

    c_q, q_I = indexer_queries(h, W_DQ, W_IUQ, index_head_dim=c_i)
    q_core = core_queries(c_q, W_UQ, n_heads=n_heads, head_dim=c)

    assert q_I.shape == (n_heads_i, c_i)
    assert q_core.shape == (n_heads, c)
    # 两条支路都是「同一个 c^Q 的线性升维」，各自逐位对上独立算式
    np.testing.assert_allclose(c_q, h @ W_DQ, atol=1e-12)
    np.testing.assert_allclose(q_I, (c_q @ W_IUQ).reshape(n_heads_i, c_i), atol=1e-12)
    np.testing.assert_allclose(q_core, (c_q @ W_UQ).reshape(n_heads, c), atol=1e-12)
    # 共用同一份潜向量：动 c_q（等价于动 W_DQ）⇒ 两支路一起变
    c_q2 = (2.0 * h) @ W_DQ
    assert not np.allclose(c_q2, c_q)
    assert not np.allclose(c_q2 @ W_IUQ, q_I.reshape(-1))
    assert not np.allclose(core_queries(c_q2, W_UQ, n_heads, c), q_core)


# ── Eq.(19)/(26)：共享 KV 的 MQA（一条条目同时当 K 和 V） ───────────────
def test_core_attention_mqa_matches_manual_softmax():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2 个 query 头、head_dim=2
    kv = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 3.0]])  # 3 条压缩条目（K=V）
    o, weights = core_attention_mqa(q, kv)

    logits = q @ kv.T * q.shape[-1] ** -0.5
    w = np.exp(logits - logits.max(axis=1, keepdims=True))
    w = w / w.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(weights, w, atol=1e-12)
    np.testing.assert_allclose(o, w @ kv, atol=1e-12)
    assert o.shape == q.shape


def test_core_attention_mqa_value_equals_key():
    # 「每条压缩 KV 条目同时当 key 和 value」：把 value 换成另一张表结果必然不同
    q = RNG.normal(size=(2, 4)).round(2)
    kv = RNG.normal(size=(5, 4)).round(2)
    o, _ = core_attention_mqa(q, kv)
    other_values = RNG.normal(size=(5, 4)).round(2)
    logits = q @ kv.T * 0.5
    w = np.exp(logits - logits.max(axis=1, keepdims=True))
    w = w / w.sum(axis=1, keepdims=True)
    assert np.allclose(o, w @ kv)
    assert not np.allclose(o, w @ other_values)


def test_masked_entries_do_not_leak():
    # 严格因果：把被 mask 掉的条目改掉，输出必须一字不变
    q = RNG.normal(size=(2, 4)).round(2)
    kv = RNG.normal(size=(4, 4)).round(2)
    mask = np.array([[True, True, False, False], [True, False, False, False]])
    o1, _ = core_attention_mqa(q, kv, mask=mask)
    kv2 = kv.copy()
    kv2[2:] += 100.0
    o2, _ = core_attention_mqa(q, kv2, mask=mask)
    assert np.allclose(o1, o2)


def test_causal_rule_by_block_tail_differs_between_paper_and_impl():
    # 论文 s < Floor(t/m) 与代码 (t+1)//m 只在块尾差一格（dossier m06）
    assert compressed_entry_count(2, 4, rule="impl") == 0
    assert compressed_entry_count(8, 4, rule="impl") == 2
    assert compressed_entry_count(7, 4, rule="impl") == 2
    assert compressed_entry_count(7, 4, rule="paper") == 1


# ── §2.3.3：滑窗 n_win 条未压缩 KV ─────────────────────────────────────
def test_sliding_window_slice_arithmetic():
    # 玩具口径 n_win=3
    assert sliding_window_slice(pos=2, n_win=3) == (0, 3)
    assert sliding_window_slice(pos=8, n_win=3) == (6, 3)
    # 真实 config 口径 n_win=128、同一个 t=8：滑窗有 9 条（窗口长度只看 n_win 与 t）
    assert sliding_window_slice(pos=8, n_win=128) == (0, 9)
    assert sliding_window_slice(pos=200, n_win=128) == (73, 128)


def test_compose_attention_inputs_toy_and_real_widths():
    # dossier m09 的 worked example：m=4、n_win=3（玩具）与 n_win=128（config）
    t2 = compose_attention_inputs(pos=2, m=4, n_win=3, rule="impl")
    assert (t2["swa_start"], t2["n_swa"]) == (0, 3)
    assert t2["n_compressed"] == 0  # (2+1)//4 = 0 ⇒ 没有完成的块，只有滑窗看得见

    t8 = compose_attention_inputs(pos=8, m=4, n_win=3, rule="impl")
    assert (t8["swa_start"], t8["n_swa"]) == (6, 3)
    assert t8["n_compressed"] == 2  # (8+1)//4 = 2
    assert t8["kv_len"] == 5

    t8_real = compose_attention_inputs(pos=8, m=4, n_win=128, rule="impl")
    assert (t8_real["swa_start"], t8_real["n_swa"]) == (0, 9)
    assert t8_real["n_compressed"] == 2
    assert t8_real["kv_len"] == 11

    # ratio<=1 的层（compress_ratios 里的 0）：没有压缩机，滑窗支路就是全部注意力
    only_swa = compose_attention_inputs(pos=8, m=1, n_win=128, rule="impl", has_compressor=False)
    assert only_swa["n_compressed"] == 0
    assert only_swa["kv_len"] == 9


# ── §2.3.3：attention sink —— Exp(z) 只进分母 ──────────────────────────
def test_softmax_with_sink_only_touches_denominator():
    # concepts 卡的玩具例：logits [2.0, 1.0]、sink z'=1.5
    p, mass = softmax_with_sink(np.array([2.0, 1.0]), sink=1.5)
    denom = np.exp(2.0) + np.exp(1.0) + np.exp(1.5)
    np.testing.assert_allclose(p, [np.exp(2.0) / denom, np.exp(1.0) / denom], atol=1e-12)
    np.testing.assert_allclose(p, [0.506480, 0.186324], atol=1e-6)
    assert mass < 1.0  # 总质量 < 1：sink 拿走了 0.307

    # 无 sink 时总质量 = 1（对照）
    p0, mass0 = softmax_with_sink(np.array([2.0, 1.0]), sink=None)
    assert np.allclose(p0.sum(), 1.0) and np.allclose(mass0, 1.0)
    # sink = −inf ⇒ exp(−inf)=0 ⇒ 退化成普通 softmax（pin 的默认初始化等价「无 sink」）
    p_neg, mass_neg = softmax_with_sink(np.array([2.0, 1.0]), sink=-np.inf)
    assert np.allclose(p_neg, p0) and np.allclose(mass_neg, 1.0)


def test_attention_with_sink_reduces_total_mass_but_not_direction():
    q = np.array([[1.0, 0.0]])
    kv = np.array([[1.0, 0.0], [0.0, 1.0]])
    o_free, w_free = core_attention_mqa(q, kv)
    o_sink, w_sink = core_attention_mqa(q, kv, sink=np.array([2.0]))
    assert w_sink.sum() < 1.0
    # sink 没有对应的 value ⇒ 两条目之间的相对比例不变，只是整体缩放
    np.testing.assert_allclose(w_sink[0] / w_sink[0].sum(), w_free[0] / w_free[0].sum(), atol=1e-12)
    assert np.allclose(o_sink, o_free * w_sink.sum() / w_free.sum())


# ── §2.3.3：RoPE 只作用最后 64 维 + 输出侧按 position −i 反旋 ───────────
def test_rope_only_touches_the_trailing_dims():
    x = RNG.normal(size=(3, 8)).round(3)
    y = rope_forward(x, positions=np.array([0, 1, 2]), rope_dim=4, theta=10000.0)
    assert np.allclose(y[:, :-4], x[:, :-4])  # 前 448 维（NoPE）一字不动
    assert not np.allclose(y[:, -4:], x[:, -4:])


def test_rope_forward_inverse_are_a_pair():
    # 「输出侧再按 position −i 旋回来」：正反两次等于没转
    x = RNG.normal(size=(4, 6)).round(3)
    pos = np.array([0, 5, 9, 17])
    fwd = rope_forward(x, positions=pos, rope_dim=6, theta=10000.0)
    back = rope_inverse(fwd, positions=pos, rope_dim=6, theta=10000.0)
    np.testing.assert_allclose(back, x, atol=1e-10)
    # 反旋不是恒等：先反旋再正旋同样回到原点，但单独一步会改变数值
    assert not np.allclose(rope_inverse(x, pos, 6, 10000.0), x)


def test_rope_pair_closed_form():
    # 一对 (x_even, x_odd)、θ 已知：正向 even: x·cos − partner·sin、odd: x·cos + partner·sin
    x = np.array([[1.0, 2.0]])
    pos = np.array([1])
    theta = 10000.0
    f = rope_forward(x, pos, rope_dim=2, theta=theta)
    c, s = np.cos(1.0), np.sin(1.0)  # inv_freq[0] = theta^0 = 1 ⇒ 角度 = pos
    np.testing.assert_allclose(f, [[1.0 * c - 2.0 * s, 2.0 * c + 1.0 * s]], atol=1e-12)
    # 反向：两个符号都翻过来（vLLM 的 fused_inv_rope 即此式）
    b = rope_inverse(x, pos, rope_dim=2, theta=theta)
    np.testing.assert_allclose(b, [[1.0 * c + 2.0 * s, 2.0 * c - 1.0 * s]], atol=1e-12)


def test_compressed_entry_positions_are_block_level():
    # 压缩核的口径：条目位置 = (positions // compress_ratio) * compress_ratio
    np.testing.assert_array_equal(compressed_entry_positions(3, 4), [0, 4, 8])
    np.testing.assert_array_equal(compressed_entry_positions(2, 128), [0, 128])
    # 条目代表整块 ⇒ query 的绝对位置与条目的块级位置必须靠逆旋转对齐
    assert compressed_entry_positions(1, 4)[0] == 0


def test_compose_returns_causal_mask_shape():
    info = compose_attention_inputs(pos=8, m=4, n_win=3, rule="impl")
    assert info["kv_len"] == info["n_swa"] + info["n_compressed"]
    assert info["compressed_indices"] == [0, 1]
