import numpy as np

from csa import (
    compress,
    core_attention_sparse,
    csa_index_score,
    csa_topk_select,
    indexer_head_weights,
    indexer_query_low_rank,
    main_query_low_rank,
    project_kv_and_gates,
)


def test_project_kv_and_gates_shapes():
    # PAPER: §2.3.1 Eq.(9)-(10)
    rng = np.random.default_rng(0)
    h = rng.standard_normal((6, 8)).astype(np.float32)
    w_akv = rng.standard_normal((8, 3)).astype(np.float32)
    w_bkv = rng.standard_normal((8, 3)).astype(np.float32)
    w_az = rng.standard_normal((8, 3)).astype(np.float32)
    w_bz = rng.standard_normal((8, 3)).astype(np.float32)

    c_a, c_b, z_a, z_b = project_kv_and_gates(h, w_akv, w_bkv, w_az, w_bz)
    assert c_a.shape == c_b.shape == z_a.shape == z_b.shape == (6, 3)


def test_compress_reduces_sequence_length_by_m():
    # PAPER: §2.3.1 Eq.(11)-(12) —— n 个条目压成 n/m 个。
    n, c, m = 6, 2, 3
    c_a = np.arange(n * c, dtype=np.float32).reshape(n, c)
    c_b = np.arange(n * c, dtype=np.float32).reshape(n, c) * 0.5
    z_a = np.zeros((n, c), dtype=np.float32)
    z_b = np.zeros((n, c), dtype=np.float32)
    b_a = np.zeros((m, c), dtype=np.float32)
    b_b = np.zeros((m, c), dtype=np.float32)

    out = compress(c_a, c_b, z_a, z_b, b_a, b_b, m)
    assert out.shape == (n // m, c)


def test_compress_block_zero_ignores_b_side_padding():
    # PAPER: §2.3.1 —— "When i=0, Z^b ... is padded with negative infinity and
    # C^b ... is padded with zeros." 因此 i=0 块的输出应只由 C^a 侧决定
    # （均匀 softmax 权重下即 C^a 窗口的简单平均）。
    n, c, m = 3, 2, 3
    c_a = np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]], dtype=np.float32)
    c_b = np.full((n, c), 999.0, dtype=np.float32)  # 若被误用会显著偏离
    z_a = np.zeros((n, c), dtype=np.float32)  # 均匀权重 -> softmax 后各条目等权
    z_b = np.zeros((n, c), dtype=np.float32)
    b_a = np.zeros((m, c), dtype=np.float32)
    b_b = np.zeros((m, c), dtype=np.float32)

    out = compress(c_a, c_b, z_a, z_b, b_a, b_b, m)
    # z_a 全 0 + z_b 侧 -inf -> softmax 只在 C^a 的 3 个条目间均匀 -> 平均值 (1+3+5)/3=3
    np.testing.assert_allclose(out[0], [3.0, 3.0], atol=1e-5)


def test_indexer_query_low_rank_and_csa_score_matches_eq1_shape():
    # PAPER: §2.3.1 Eq.(13)-(16) —— 低秩出 indexer query，csa_index_score 与
    # lightning_indexer.index_score 同构（Eq.16 == Eq.1，只是 s 换成压缩块）。
    rng = np.random.default_rng(1)
    t, d, d_c, n_hi, c_i = 4, 6, 5, 2, 3
    h = rng.standard_normal((t, d)).astype(np.float32)
    w_dq = rng.standard_normal((d, d_c)).astype(np.float32)
    w_iuq = rng.standard_normal((d_c, c_i * n_hi)).astype(np.float32)
    w_w = rng.standard_normal((d, n_hi)).astype(np.float32)

    c_q, q = indexer_query_low_rank(h, w_dq, w_iuq, n_hi)
    assert c_q.shape == (t, d_c)
    assert q.shape == (t, n_hi, c_i)

    w = indexer_head_weights(h, w_w)
    assert w.shape == (t, n_hi)

    n_blocks = 3
    k_comp = rng.standard_normal((n_blocks, c_i)).astype(np.float32)
    scores = csa_index_score(q, k_comp, w)
    assert scores.shape == (t, n_blocks)

    idx = csa_topk_select(scores, k=2)
    assert idx.shape == (t, 2)


def test_main_query_low_rank_shares_c_q_with_indexer():
    # PAPER: §2.3.1 Eq.(18) —— "the latent query vector c_t^Q is shared with that
    # used for the indexer queries."
    rng = np.random.default_rng(2)
    t, d_c, n_h, c = 3, 5, 4, 6
    c_q = rng.standard_normal((t, d_c)).astype(np.float32)
    w_uq = rng.standard_normal((d_c, n_h * c)).astype(np.float32)

    q = main_query_low_rank(c_q, w_uq, n_h)
    assert q.shape == (t, n_h, c)


def test_core_attention_sparse_only_uses_selected_blocks():
    # PAPER: §2.3.1 Eq.(19) —— Shared Key-Value MQA：每个 query 只在其被选中的
    # 压缩块集合上做注意力（key=value 共享）。
    n_h, c, n_blocks, k = 1, 2, 4, 1
    q = np.array([[[1.0, 0.0]]])  # [T=1, n_h=1, c=2]
    c_comp = np.array([[10.0, 10.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    topk_idx = np.array([[1]])  # 只选中块 1

    out = core_attention_sparse(q, c_comp, topk_idx)
    # 只在块 1 上做注意力（单个 key/value），输出应恰好等于 c_comp[1]
    np.testing.assert_allclose(out[0, 0], c_comp[1], atol=1e-6)
