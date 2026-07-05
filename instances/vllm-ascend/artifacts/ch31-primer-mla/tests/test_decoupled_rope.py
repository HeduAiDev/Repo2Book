"""ch31 —— 解耦 RoPE（§2.1.3 Eq.14-19）。本文件是全章认知悬崖的数值证明：
1) RoPE 的"相对位置"性质本身没错（R_query^T R_key == R_{key-query}）；
2) 但那个 R_{key-query} 夹在 (W^Q)^T 与 W^{UK} 中间，矩阵乘不交换 —— 随相对位置变化，
   "中间那块矩阵" M(delta) 也跟着变，不存在一个位置无关的静态吸收矩阵；
3) 于是论文选择解耦：q^R/k^R 单独走一条不参与吸收的小路，nope 部分保持位置无关、继续可吸收。
"""
import numpy as np

from decoupled_rope import (
    rope_rotation_matrix,
    apply_rope,
    verify_relative_position_property,
    rope_on_compressed_key_score,
    effective_middle_matrix,
    init_decoupled_rope_weights,
    decoupled_query_rope,
    decoupled_key_rope,
    concat_nope_rope_query,
    concat_nope_rope_key,
    decoupled_attention_scores,
)


def test_rope_rotation_matrix_is_orthogonal():
    R = rope_rotation_matrix(pos=3, dim=6)
    np.testing.assert_allclose(R @ R.T, np.eye(6), atol=1e-10)


def test_relative_position_property_holds():
    """RoPE 的教科书性质：这一步没有问题——问题出在下一步的"夹在中间"。"""
    lhs, rhs = verify_relative_position_property(pos_query=5, pos_key=9, dim=8)
    np.testing.assert_allclose(lhs, rhs, atol=1e-10)


def test_rope_on_compressed_key_score_matches_manual_relative_expansion():
    """score = q_t^{rope} . k_j^{rope,C} 应等于展开式 h_t^T @ effective_middle_matrix(delta) @ c_j，
    delta = pos_j - pos_t —— 这条恒等式把"打分"和"中间那块矩阵"严格对应起来，
    后面才能问：这块矩阵是不是与 delta 无关？
    """
    rng = np.random.default_rng(0)
    d_h = 4
    w_q_head = rng.normal(size=(d_h, 6))   # (d_h, d)
    w_uk_head = rng.normal(size=(d_h, 3))  # (d_h, d_c)
    h_t = rng.normal(size=6)
    c_j = rng.normal(size=3)
    pos_t, pos_j = 2, 7

    score = rope_on_compressed_key_score(h_t, pos_t, c_j, pos_j, w_q_head, w_uk_head)
    m_delta = effective_middle_matrix(w_q_head, w_uk_head, delta=pos_j - pos_t)
    manual = float(h_t @ m_delta @ c_j)
    assert abs(score - manual) < 1e-8


def test_effective_middle_matrix_is_not_static_across_relative_positions():
    """这是"不可吸收"的代数证据：M(delta) 随 delta 变化，不存在与 delta 无关的静态矩阵。
    与之对照的是 low_rank_mla.precompute_absorbed_query_weights 里的 W~——那个是真正的常量。
    """
    rng = np.random.default_rng(1)
    d_h = 4
    w_q_head = rng.normal(size=(d_h, 6))
    w_uk_head = rng.normal(size=(d_h, 3))

    m0 = effective_middle_matrix(w_q_head, w_uk_head, delta=0)
    m1 = effective_middle_matrix(w_q_head, w_uk_head, delta=1)
    m10 = effective_middle_matrix(w_q_head, w_uk_head, delta=10)

    assert not np.allclose(m0, m1)
    assert not np.allclose(m1, m10)
    # 且 delta=0 时（query 和 key 同一位置）旋转矩阵退化为单位阵，M(0) 才等于无 rope 时的静态吸收矩阵
    np.testing.assert_allclose(m0, w_q_head.T @ w_uk_head, atol=1e-10)


def test_naive_rope_on_key_breaks_score_shift_invariance_used_by_prefix_caching():
    """工程含义的直接数值版本：论文说"必须为所有 prefix token 重算 key"。用两个不同的
    query 位置 t1、t2 对同一个历史 key 位置 j 打分，若中间矩阵是静态的，两次打分应只相差
    一个与 h 无关的整体因子；但因为 M(delta) 依赖 delta=j-t，两次打分之间没有这种简单关系——
    也就是说，你不能"缓存一份对 j 算好的东西，未来任意新 query t 复用"，每个新 t 都要重新对
    每个历史 j 走一遍完整的旋转+物化。
    """
    rng = np.random.default_rng(2)
    d_h = 4
    w_q_head = rng.normal(size=(d_h, 6))
    w_uk_head = rng.normal(size=(d_h, 3))
    h_t1 = rng.normal(size=6)
    h_t2 = rng.normal(size=6)
    c_j = rng.normal(size=3)
    pos_j = 5

    score_t1 = rope_on_compressed_key_score(h_t1, pos_t=1, c_j=c_j, pos_j=pos_j, w_q_head=w_q_head, w_uk_head=w_uk_head)
    score_t2 = rope_on_compressed_key_score(h_t2, pos_t=8, c_j=c_j, pos_j=pos_j, w_q_head=w_q_head, w_uk_head=w_uk_head)

    # 用"无 rope 静态吸收矩阵"（delta=0 的退化情形）预测出的打分，应显著偏离两次真实打分——
    # 证明静态矩阵在有 rope 的情形下不成立。
    m_static = effective_middle_matrix(w_q_head, w_uk_head, delta=0)
    wrong_predict_t1 = float(h_t1 @ m_static @ c_j)
    wrong_predict_t2 = float(h_t2 @ m_static @ c_j)
    assert abs(score_t1 - wrong_predict_t1) > 1e-6
    assert abs(score_t2 - wrong_predict_t2) > 1e-6


def test_decoupled_pipeline_shapes_and_causal_scores_sum_to_one():
    n_h, d_h_r, d_c_q, d = 3, 2, 5, 6
    T = 4
    w = init_decoupled_rope_weights(d, n_h, d_h_r, d_c_q, seed=7)
    rng = np.random.default_rng(8)
    c_q_seq = rng.normal(size=(T, d_c_q))
    h_seq = rng.normal(size=(T, d))
    positions = list(range(T))

    q_r_heads = decoupled_query_rope(c_q_seq, w.W_QR, positions, n_h, d_h_r)
    k_r_seq = decoupled_key_rope(h_seq, w.W_KR, positions)
    assert q_r_heads.shape == (n_h, T, d_h_r)
    assert k_r_seq.shape == (T, d_h_r)

    d_h_nope = 4
    q_c_heads = rng.normal(size=(n_h, T, d_h_nope))
    k_c_heads = rng.normal(size=(n_h, T, d_h_nope))
    q_full = concat_nope_rope_query(q_c_heads, q_r_heads)
    k_full = concat_nope_rope_key(k_c_heads, k_r_seq)
    assert q_full.shape == (n_h, T, d_h_nope + d_h_r)

    attn_w = decoupled_attention_scores(q_full, k_full)
    assert attn_w.shape == (n_h, T, T)
    # 每行 softmax 应归一
    np.testing.assert_allclose(attn_w.sum(axis=-1), np.ones((n_h, T)), atol=1e-10)
    # 因果：第 0 个 query 只能看到自己 -> weight[.,0,0]==1，其余为 0
    np.testing.assert_allclose(attn_w[:, 0, 0], np.ones(n_h), atol=1e-10)
    np.testing.assert_allclose(attn_w[:, 0, 1:], np.zeros((n_h, T - 1)), atol=1e-10)


def test_shared_key_rope_broadcasts_identically_to_all_heads():
    """k^R 是全头共享的（Eq.15/17 一处生成、广播到每个头），而不是每头各算一份。"""
    n_h, T, d_h_r = 3, 2, 2
    k_c_heads = np.zeros((n_h, T, 4))
    k_r_seq = np.array([[1.0, 2.0], [3.0, 4.0]])
    k_full = concat_nope_rope_key(k_c_heads, k_r_seq)
    for i in range(n_h):
        np.testing.assert_allclose(k_full[i, :, 4:], k_r_seq)
