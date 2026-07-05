"""ch31 —— 标准 MHA 基线（§2.1.1 Eq.1-8）的形状/因果性/手算校验。"""
import numpy as np

from mha_baseline import (
    StandardMHA,
    init_mha_weights,
    project_qkv,
    split_heads,
    merge_heads,
    scaled_dot_product_attention,
    output_projection,
)


def test_project_qkv_matches_manual_matmul():
    d, n_h, d_h = 6, 2, 3
    w = init_mha_weights(d, n_h, d_h, seed=1)
    h_seq = np.random.default_rng(2).normal(size=(4, d))
    q_seq, k_seq, v_seq = project_qkv(h_seq, w)
    assert q_seq.shape == (4, n_h * d_h)
    # 逐 token 手算 q_t = W^Q h_t，应与批量矩阵乘完全一致
    for t in range(4):
        np.testing.assert_allclose(q_seq[t], w.W_Q @ h_seq[t])


def test_split_merge_heads_are_inverse():
    x = np.arange(4 * 6).reshape(4, 6).astype(float)
    heads = split_heads(x, n_h=2, d_h=3)
    assert heads.shape == (2, 4, 3)
    back = merge_heads(heads)
    np.testing.assert_allclose(back, x)


def test_attention_is_causal_first_token_only_attends_self():
    n_h, t, d_h = 1, 3, 2
    rng = np.random.default_rng(0)
    q = rng.normal(size=(n_h, t, d_h))
    k = rng.normal(size=(n_h, t, d_h))
    v = rng.normal(size=(n_h, t, d_h))
    o = scaled_dot_product_attention(q, k, v)
    # 第 0 个 token 只能看见自己 -> 输出应恰好等于 v[0]（softmax 退化为 1）
    np.testing.assert_allclose(o[0, 0], v[0, 0])


def test_standard_mha_forward_shape_and_kv_cache_baseline():
    d, n_h, d_h, T = 8, 2, 4, 5
    mha = StandardMHA(d, n_h, d_h, seed=3)
    h_seq = np.random.default_rng(4).normal(size=(T, d))
    u_seq = mha.forward(h_seq)
    assert u_seq.shape == (T, d)
    # KV cache 基线：每 token 每层需要 2*n_h*d_h 个元素（Eq.7-8 讨论段）
    assert 2 * n_h * d_h == 16


def test_output_projection_matches_manual():
    d_h, n_h, d = 3, 2, 5
    rng = np.random.default_rng(5)
    o_heads = rng.normal(size=(n_h, 4, d_h))
    W_O = rng.normal(size=(d, n_h * d_h))
    u = output_projection(o_heads, W_O)
    manual = np.stack([W_O @ merge_heads(o_heads)[t] for t in range(4)])
    np.testing.assert_allclose(u, manual)
