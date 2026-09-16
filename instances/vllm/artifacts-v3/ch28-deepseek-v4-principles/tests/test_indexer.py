"""索引器测试 —— 论文锚：
arXiv:2606.19348 Eq.(13)(14)(15)(16)(17)（压缩块上的打分与 top-k）、§2.3.1、
§2.3.3（因果）；谱系锚 arXiv:2512.02556 Eq.(1)(2)（同形的 token 级版）。
"""
import numpy as np

from indexer import (
    causal_candidate_count,
    compressed_index_keys,
    index_scores,
    indexer_head_weights,
    indexer_queries,
    select_topk_compressed,
    short_context_select_all,
)

RNG = np.random.default_rng(2606)


# ── Eq.(13)(14)：低秩 q ────────────────────────────────────────────────
def test_indexer_queries_low_rank_and_shape():
    d, d_c, n_h, c_i = 6, 3, 2, 2
    h = RNG.normal(size=(d,)).round(2)
    W_DQ = RNG.normal(size=(d, d_c)).round(2)
    W_IUQ = RNG.normal(size=(d_c, c_i * n_h)).round(2)

    c_q, q_I = indexer_queries(h, W_DQ, W_IUQ, index_head_dim=c_i)
    assert c_q.shape == (d_c,)
    assert q_I.shape == (n_h, c_i)
    assert np.allclose(c_q, h @ W_DQ)  # Eq.(13) 逐字
    assert np.allclose(q_I.reshape(-1), c_q @ W_IUQ)  # Eq.(14) 逐字


def test_indexer_queries_share_the_core_attention_latent():
    # Eq.(18) 的注：latent query vector 与索引器共用同一份 c^Q
    d, d_c, n_h, c_i = 5, 2, 3, 2
    h = RNG.normal(size=(d,)).round(2)
    W_DQ = RNG.normal(size=(d, d_c)).round(2)
    W_IUQ = RNG.normal(size=(d_c, c_i * n_h)).round(2)
    W_UQ = RNG.normal(size=(d_c, 4 * 3)).round(2)

    c_q, q_I = indexer_queries(h, W_DQ, W_IUQ, index_head_dim=c_i)
    q_core = c_q @ W_UQ  # 主注意力 q 也从同一份 c^Q 升维而来
    assert np.allclose(c_q, h @ W_DQ)
    assert q_core.shape == (12,)


# ── Eq.(15)(16)：逐头权重 + ReLU 内积 ───────────────────────────────────
def test_index_scores_two_heads_four_blocks_by_hand():
    # dossier m05 的 worked example 口径：2 个索引头 × 4 个候选块
    w = np.array([0.7, 0.3])
    q_I = np.array([[2.0, 1.0], [0.5, 0.0]])  # (n_h=2, c^I=2)
    k_icomp = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.5, 0.5]]
    )  # 4 个压缩块的索引键
    # 头 1 内积 = [2.0, 1.0, -2.0, 1.5]；头 2 内积 = [0.5, 0.0, -0.5, 0.25]
    I = index_scores(q_I, k_icomp, w)
    want = 0.7 * np.maximum([2.0, 1.0, -2.0, 1.5], 0) + 0.3 * np.maximum([0.5, 0.0, -0.5, 0.25], 0)
    np.testing.assert_allclose(I, want, atol=1e-12)
    np.testing.assert_allclose(I, [1.55, 0.7, 0.0, 1.125], atol=1e-12)

    idx, _ = select_topk_compressed(I[None, :], topk=2)
    assert sorted(idx[0].tolist()) == [0, 3]  # top-2 = 块 0 与块 3


def test_index_scores_relu_kills_negative_correlation():
    I = index_scores(
        np.array([[3.0]]), np.array([[1.0], [-1.0]]), np.array([1.0])
    )
    assert I[0] == 3.0 and I[1] == 0.0  # 负相关不打负分


def test_index_scores_implementation_scalings_are_optional():
    # 论文 Eq.(16) 没有缩放；两侧实现各补一个（官方 model.py:L395/L418）
    q_I = np.array([[2.0]])
    k = np.array([[3.0]])
    w = np.array([0.5])
    plain = index_scores(q_I, k, w)
    scaled = index_scores(q_I, k, w, softmax_scale=0.5, weights_scaling=0.5)
    assert np.allclose(plain, [3.0])
    assert np.allclose(scaled, [3.0 * 0.5 * 0.5])


def test_indexer_head_weights_shape():
    h = RNG.normal(size=(5,)).round(2)
    W_w = RNG.normal(size=(5, 3)).round(2)
    w = indexer_head_weights(h, W_w)
    assert w.shape == (3,)
    assert np.allclose(w, h @ W_w)  # Eq.(15) 逐字


# ── Eq.(17) + §2.3.3：top-k 与因果 ─────────────────────────────────────
def test_select_topk_returns_best_blocks_and_minus_one_padding():
    I = np.array([[0.2, 3.1, 0.5, 2.9]])
    idx, valid = select_topk_compressed(I, topk=2)
    assert sorted(idx[0].tolist()) == [1, 3]
    assert valid[0].all()

    idx2, valid2 = select_topk_compressed(I, topk=6)
    assert sorted(int(x) for x in idx2[0] if x >= 0) == [0, 1, 2, 3]
    assert (idx2[0] < 0).sum() == 2  # 候选不足 ⇒ −1 哨兵
    assert valid2[0][:4].all() and not valid2[0][4:].any()


def test_causal_threshold_paper_vs_impl_rule():
    # 论文：s < Floor(t/m)（只能选严格在自己所在块之前的块）
    # 代码：num_compressed = (t+1)//m（已完成的块数，块尾 token 能看自己那块，仍严格因果）
    assert causal_candidate_count(7, 4, rule="paper") == 1
    assert causal_candidate_count(7, 4, rule="impl") == 2
    assert causal_candidate_count(3, 4, rule="paper") == 0
    assert causal_candidate_count(3, 4, rule="impl") == 1  # 差的那一格就在块尾
    for t in range(12):
        assert causal_candidate_count(t, 4, rule="impl") >= causal_candidate_count(t, 4, rule="paper")


def test_causal_mask_keeps_future_blocks_out_of_topk():
    # 4 个候选块、topk=2，但因果只允许前 2 个（t=7,m=4 → impl 口径 2 个）
    I = np.array([[1.0, 9.0, 9.0, 9.0]])
    thresh = causal_candidate_count(7, 4, rule="impl")
    idx, valid = select_topk_compressed(I, topk=2, causal_threshold=thresh)
    assert set(int(x) for x in idx[0]) == {0, 1}
    assert valid[0].all()


def test_short_context_select_all():
    # 官方参考实现（model.py:L409-L410）：max_seq_len // m <= topk 时每个候选都被选中
    assert short_context_select_all(max_seq_len=64, m=4, topk=512)
    assert short_context_select_all(max_seq_len=2048, m=4, topk=512)
    assert not short_context_select_all(max_seq_len=4096, m=4, topk=512)


# ── §2.3.1：索引器键 = 同一套压缩操作，只是头维换成 c^I ────────────────
def test_compressed_index_keys_use_the_same_compression_operation():
    m, n, c_i = 4, 8, 2
    H = RNG.normal(size=(n, 3)).round(2)
    W_kv_a = RNG.normal(size=(3, c_i)).round(2)
    W_kv_b = RNG.normal(size=(3, c_i)).round(2)
    W_z_a = RNG.normal(size=(3, c_i)).round(2)
    W_z_b = RNG.normal(size=(3, c_i)).round(2)
    B_a = RNG.normal(size=(m, c_i)).round(2)
    B_b = RNG.normal(size=(m, c_i)).round(2)

    K_icomp = compressed_index_keys(H, W_kv_a, W_kv_b, W_z_a, W_z_b, B_a, B_b, m=m)
    assert K_icomp.shape == (2, c_i)  # n/m 条，列宽 = 索引头维 c^I
