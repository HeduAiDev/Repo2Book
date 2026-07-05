import numpy as np
import pytest

from shared_mqa_grouped_output import (
    attention_output_pipeline,
    attention_queries,
    grouped_output_projection,
    mqa_core_attention,
)


def test_attention_queries_reshape():
    c_q = np.ones(4)
    W_UQ = np.eye(4, 6)
    q = attention_queries(c_q, W_UQ, n_heads=2, head_dim=3)
    assert q.shape == (2, 3)


def test_mqa_core_attention_softmax_normalizes():
    rng = np.random.default_rng(0)
    q_heads = rng.normal(size=(4, 8))
    kv = rng.normal(size=(6, 8))
    out = mqa_core_attention(q_heads, kv)
    assert out.shape == (4, 8)
    # 每个头的注意力权重应归一化为 1(通过重建权重矩阵间接验证:输出必是 kv 的凸组合,
    # 故应落在 kv 各行的凸包内 —— 用范数上界做一个弱但可靠的检验)
    max_norm = np.max(np.linalg.norm(kv, axis=-1))
    assert np.all(np.linalg.norm(out, axis=-1) <= max_norm + 1e-6)


def test_mqa_core_attention_shared_kv_across_heads():
    """MQA 的核心性质:所有头共享同一份 kv_entries 当 key 与 value(不像标准 MHA 每头
    独立)——这里通过"同一个 kv 集合被所有头的输出复用"这一点体现在函数签名本身
    (kv_entries 只传一份,不分头);数值上验证退化情形:只有 1 个 kv 条目时,
    无论 query 是什么,输出必然等于该条目本身(softmax 退化成 1)。"""
    q_heads = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, -1.0]])
    kv = np.array([[5.0, 6.0]])
    out = mqa_core_attention(q_heads, kv)
    for i in range(3):
        np.testing.assert_allclose(out[i], kv[0])


def test_mqa_core_attention_empty_kv_returns_zeros():
    q_heads = np.ones((2, 4))
    kv = np.zeros((0, 4))
    out = mqa_core_attention(q_heads, kv)
    np.testing.assert_allclose(out, np.zeros((2, 4)))


def test_grouped_output_projection_shape_and_value():
    # n_h=4 heads, c=2, g=2 groups -> heads_per_group=2, group_size=4
    o_heads = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    d_g = 3
    rng = np.random.default_rng(0)
    group_weights = [rng.normal(size=(4, d_g)) for _ in range(2)]
    d_out = 5
    W_out = rng.normal(size=(2 * d_g, d_out))
    out = grouped_output_projection(o_heads, group_weights, W_out)
    assert out.shape == (d_out,)
    # 手算校验
    flat0 = o_heads[:2].reshape(-1)
    flat1 = o_heads[2:].reshape(-1)
    inter0 = flat0 @ group_weights[0]
    inter1 = flat1 @ group_weights[1]
    concatenated = np.concatenate([inter0, inter1])
    expected = concatenated @ W_out
    np.testing.assert_allclose(out, expected)


def test_grouped_output_projection_rejects_indivisible_groups():
    o_heads = np.ones((3, 2))
    group_weights = [np.ones((6, 4)), np.ones((6, 4))]  # g=2 but n_h=3 不能整除
    W_out = np.ones((8, 5))
    with pytest.raises(ValueError):
        grouped_output_projection(o_heads, group_weights, W_out)


def test_attention_output_pipeline_end_to_end_shape():
    rng = np.random.default_rng(2)
    d_c = 6
    n_heads, head_dim = 4, 8
    c_q_t = rng.normal(size=d_c)
    W_UQ = rng.normal(scale=0.1, size=(d_c, n_heads * head_dim))
    kv_entries = rng.normal(size=(5, head_dim))
    g = 2
    heads_per_group = n_heads // g
    d_g = 3
    group_weights = [rng.normal(scale=0.1, size=(heads_per_group * head_dim, d_g)) for _ in range(g)]
    d_out = 10
    W_out = rng.normal(scale=0.1, size=(g * d_g, d_out))
    out = attention_output_pipeline(c_q_t, W_UQ, kv_entries, n_heads, head_dim, group_weights, W_out)
    assert out.shape == (d_out,)
