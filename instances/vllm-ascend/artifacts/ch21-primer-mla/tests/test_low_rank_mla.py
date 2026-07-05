"""ch31 —— 低秩 KV 联合压缩（Eq.9-11）+ q 侧低秩（Eq.12-13）+ 权重吸收恒等式（关键测试）。"""
import numpy as np

from low_rank_mla import (
    init_kv_compression_weights,
    init_q_compression_weights,
    kv_joint_compression,
    q_joint_compression,
    precompute_absorbed_query_weights,
    precompute_uv_head_slices,
    score_materialized_nope,
    score_absorbed_nope,
    attention_in_latent_space,
    latent_to_value,
    split_kv_heads,
)
from mha_baseline import split_heads
from numerics import softmax


D, N_H, D_H, D_C, D_C_Q, T = 10, 3, 4, 5, 6, 7


def _setup(seed=0):
    kv_w = init_kv_compression_weights(D, N_H, D_H, D_C, seed=seed)
    q_w = init_q_compression_weights(D, N_H, D_H, D_C_Q, seed=seed + 1)
    h_seq = np.random.default_rng(seed + 2).normal(size=(T, D))
    return kv_w, q_w, h_seq


def test_kv_cache_only_needs_latent_dim():
    kv_w, _, h_seq = _setup()
    c_kv_seq, k_c_seq, v_c_seq = kv_joint_compression(h_seq, kv_w)
    assert c_kv_seq.shape == (T, D_C)          # 推理只需缓存这个
    assert k_c_seq.shape == (T, N_H * D_H)      # 物化出的满维 key（对照组，非缓存对象）
    assert v_c_seq.shape == (T, N_H * D_H)


def test_q_low_rank_reduces_activation_dim_not_kv_cache():
    _, q_w, h_seq = _setup()
    c_q_seq, q_c_seq = q_joint_compression(h_seq, q_w)
    assert c_q_seq.shape == (T, D_C_Q)
    assert q_c_seq.shape == (T, N_H * D_H)


def test_weight_absorption_identity_nope_score():
    """核心恒等式：物化路径打分 == 吸收路径打分（对每个头、每对 (t,j) 都成立）。"""
    kv_w, q_w, h_seq = _setup()
    c_kv_seq, k_c_seq, _ = kv_joint_compression(h_seq, kv_w)
    c_q_seq, q_c_seq = q_joint_compression(h_seq, q_w)
    k_c_heads = split_heads(k_c_seq, N_H, D_H)
    q_c_heads = split_heads(q_c_seq, N_H, D_H)
    w_tildes = precompute_absorbed_query_weights(q_w, kv_w, N_H, D_H)

    for i in range(N_H):
        for t in range(T):
            for j in range(T):
                materialized = score_materialized_nope(q_c_heads[i, t], k_c_heads[i, j])
                absorbed = score_absorbed_nope(c_q_seq[t], c_kv_seq[j], w_tildes[i])
                assert abs(materialized - absorbed) < 1e-8, (i, t, j)


def test_output_side_absorption_identity():
    """核心恒等式：先加权求和 c_j 再乘 W^{UV}  ==  先把每个 c_j 变成 v_j^C 再加权求和。"""
    kv_w, q_w, h_seq = _setup()
    c_kv_seq, k_c_seq, v_c_seq = kv_joint_compression(h_seq, kv_w)
    v_c_heads = split_heads(v_c_seq, N_H, D_H)
    w_uv_heads = precompute_uv_head_slices(kv_w, N_H, D_H)

    rng = np.random.default_rng(42)
    raw_scores = rng.normal(size=T)
    weights = softmax(raw_scores)

    for i in range(N_H):
        materialized = weights @ v_c_heads[i]                      # Σ_j w_j v_j^C
        latent = attention_in_latent_space(weights, c_kv_seq)       # Σ_j w_j c_j（潜空间）
        absorbed = latent_to_value(latent, w_uv_heads[i])           # 再乘 W^{UV}
        np.testing.assert_allclose(materialized, absorbed, atol=1e-8)


def test_split_kv_heads_shapes():
    kv_w, q_w, h_seq = _setup()
    c_kv_seq, k_c_seq, v_c_seq = kv_joint_compression(h_seq, kv_w)
    _, q_c_seq = q_joint_compression(h_seq, q_w)
    k_h, v_h, q_h = split_kv_heads(k_c_seq, v_c_seq, q_c_seq, N_H, D_H)
    assert k_h.shape == v_h.shape == q_h.shape == (N_H, T, D_H)
