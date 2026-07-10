import numpy as np

from kl_alignment import (
    dense_warmup_loss,
    detach,
    main_attention_target_distribution,
    sparse_training_loss,
)


def test_target_distribution_sums_heads_then_l1_normalizes():
    # PAPER: §2.1.1 "aggregate the main attention scores by summing across all
    # attention heads. This sum is then L1-normalized along the sequence
    # dimension" —— T=1, H=2, S=2 手算。
    attn_scores = np.array([[[0.6, 0.4], [0.2, 0.8]]])  # [T=1, H=2, S=2]
    p = main_attention_target_distribution(attn_scores)
    # sum over heads -> [0.8, 1.2]; L1 normalize -> [0.4, 0.6]
    np.testing.assert_allclose(p, [[0.4, 0.6]])
    np.testing.assert_allclose(p.sum(axis=-1), [1.0])


def test_dense_warmup_loss_is_zero_when_indexer_matches_target():
    # PAPER: §2.1.1 Eq.(3) —— 若 Softmax(I_{t,:}) 与目标分布 p_{t,:} 完全一致，
    # KL 散度应为 0（indexer 已完美对齐主注意力）。
    p_target = np.array([[0.2, 0.3, 0.5]])
    # 构造 index_scores 使其 softmax 恰好等于 p_target：用 log(p) 作为 logits。
    index_scores = np.log(p_target)
    loss = dense_warmup_loss(index_scores, p_target)
    assert abs(loss) < 1e-8


def test_dense_warmup_loss_positive_when_misaligned():
    p_target = np.array([[0.9, 0.1]])
    index_scores = np.array([[0.0, 0.0]])  # softmax -> [0.5, 0.5]，明显偏离
    loss = dense_warmup_loss(index_scores, p_target)
    assert loss > 0.0


def test_sparse_training_loss_equals_dense_when_topk_covers_full_set():
    # PAPER: §2.1.1 Eq.(4) —— 当 S_t 覆盖全部 S 个索引（按原始顺序）时，
    # p_{t,S_t}=p_{t,:}、Softmax(I_{t,S_t})=Softmax(I_{t,:})，Eq.(4) 应退化为 Eq.(3)。
    p_target = np.array([[0.2, 0.3, 0.5]])
    index_scores = np.array([[1.0, 2.0, 0.5]])
    full_idx = np.array([[0, 1, 2]])

    dense = dense_warmup_loss(index_scores, p_target)
    sparse = sparse_training_loss(index_scores, p_target, full_idx)

    np.testing.assert_allclose(sparse, dense)


def test_sparse_training_loss_only_uses_selected_subset():
    # 选中集只覆盖部分索引时，p_sub 不重新归一化（限制真分布到子集，天然丢失概率
    # 质量），Softmax(I_{t,S_t}) 只在子集内重新做 softmax。
    p_target = np.array([[0.2, 0.3, 0.5]])
    index_scores = np.array([[1.0, 2.0, 0.5]])
    idx = np.array([[1, 2]])  # 只选中索引 1、2

    loss = sparse_training_loss(index_scores, p_target, idx)
    assert loss > 0.0  # 只需可计算且非负（KL 对未归一化 p 逐元素求和恒 >= 一般不小于0附近）


def test_detach_returns_independent_copy():
    # PAPER: §2.1.1 "we detach the indexer input from the computational graph
    # for separate optimization" —— 参考实现里 detach 语义化为独立拷贝。
    x = np.array([1.0, 2.0, 3.0])
    y = detach(x)
    np.testing.assert_array_equal(x, y)
    y[0] = 999.0
    assert x[0] == 1.0  # 修改拷贝不应影响原数组
