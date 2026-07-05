import numpy as np
import pytest

from lightning_indexer_csa import (
    indexer_compressed_keys,
    indexer_head_weights,
    indexer_queries,
    index_score,
    index_scores_for_query,
    low_rank_query_latent,
    topk_sparse_selection,
)


def test_low_rank_query_latent():
    h_t = np.array([1.0, 2.0, 3.0])
    W_DQ = np.eye(3)
    np.testing.assert_allclose(low_rank_query_latent(h_t, W_DQ), h_t)


def test_indexer_queries_reshape():
    c_q = np.ones(4)
    W_IUQ = np.eye(4, 6)   # projects 4 -> 6 = n_heads_idx(2)*head_dim_idx(3)
    q = indexer_queries(c_q, W_IUQ, n_heads_idx=2, head_dim_idx=3)
    assert q.shape == (2, 3)


def test_indexer_head_weights():
    h_t = np.array([1.0, 0.0])
    W_w = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(indexer_head_weights(h_t, W_w), [1.0, 2.0, 3.0])


def test_index_score_relu_zeroes_negative_dot_products():
    # 头 0 的 dot 为负 -> ReLU 清零,不该贡献,即使权重很大
    q_idx = np.array([[-1.0], [1.0]])   # (2 heads, 1 dim)
    k_s = np.array([1.0])
    w = np.array([1000.0, 1.0])
    score = index_score(q_idx, w, k_s)
    # 头0: dot=-1 -> relu=0 -> 贡献0(不管权重多大); 头1: dot=1 -> relu=1 -> 贡献 1*1=1
    assert score == pytest.approx(1.0)


def test_index_scores_for_query_matches_loop():
    rng = np.random.default_rng(0)
    n_heads_idx, dim_idx, S = 3, 4, 5
    q_idx = rng.normal(size=(n_heads_idx, dim_idx))
    w = rng.normal(size=n_heads_idx)
    K = rng.normal(size=(S, dim_idx))
    batch_scores = index_scores_for_query(q_idx, w, K)
    loop_scores = np.array([index_score(q_idx, w, K[s]) for s in range(S)])
    np.testing.assert_allclose(batch_scores, loop_scores)


def test_indexer_compressed_keys_shape():
    n, d, c_idx, m = 8, 5, 3, 4
    rng = np.random.default_rng(1)
    H = rng.normal(size=(n, d))
    W_a_kv = rng.normal(scale=0.1, size=(d, c_idx))
    W_b_kv = rng.normal(scale=0.1, size=(d, c_idx))
    W_a_z = rng.normal(scale=0.1, size=(d, c_idx))
    W_b_z = rng.normal(scale=0.1, size=(d, c_idx))
    B_a = rng.normal(scale=0.1, size=(m, c_idx))
    B_b = rng.normal(scale=0.1, size=(m, c_idx))
    K_IComp = indexer_compressed_keys(H, W_a_kv, W_b_kv, W_a_z, W_b_z, B_a, B_b, m)
    assert K_IComp.shape == (n // m, c_idx)


def test_topk_sparse_selection_picks_highest_scores():
    scores = np.array([0.1, 5.0, 0.2, 3.0, 0.05, 4.0])
    C_comp = np.arange(6).reshape(6, 1).astype(float)
    selected, C_selected = topk_sparse_selection(scores, C_comp, k=3, causal_limit=None)
    # top-3 分数对应下标 1(5.0), 5(4.0), 3(3.0) -> 排序后 [1,3,5]
    np.testing.assert_array_equal(selected, [1, 3, 5])
    np.testing.assert_allclose(C_selected.flatten(), [1.0, 3.0, 5.0])


def test_topk_sparse_selection_respects_causal_limit():
    scores = np.array([0.1, 5.0, 0.2, 3.0, 0.05, 4.0])
    C_comp = np.arange(6).reshape(6, 1).astype(float)
    # 只允许看前 3 个候选块(causal_limit=3),即使全局最高分(下标5)在因果范围外也不可选
    selected, _ = topk_sparse_selection(scores, C_comp, k=2, causal_limit=3)
    assert set(selected.tolist()) <= {0, 1, 2}
    np.testing.assert_array_equal(selected, [1, 2])  # 候选 0..2 里最高的两个是 idx1(5.0), idx2(0.2)...
    # 修正: 候选[0,1,2]分数为[0.1,5.0,0.2],top-2是idx1,idx2


def test_topk_sparse_selection_empty_causal_window():
    scores = np.array([1.0, 2.0, 3.0])
    C_comp = np.arange(3).reshape(3, 1).astype(float)
    selected, C_selected = topk_sparse_selection(scores, C_comp, k=2, causal_limit=0)
    assert selected.shape == (0,)
    assert C_selected.shape == (0, 1)


def test_topk_sparse_selection_k_larger_than_candidates():
    scores = np.array([1.0, 2.0, 3.0])
    C_comp = np.arange(3).reshape(3, 1).astype(float)
    selected, C_selected = topk_sparse_selection(scores, C_comp, k=10, causal_limit=None)
    assert selected.shape == (3,)
    np.testing.assert_array_equal(selected, [0, 1, 2])
