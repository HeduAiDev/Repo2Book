import numpy as np
import pytest

from lightning_indexer import IndexerConfig, index_score, topk_select


def test_indexer_config_is_independent_of_main_head_shape():
    # PAPER: §2.1 "Instantiate DSA Under MLA" —— H^I/d^I 与主注意力头数/头维无关，
    # 这里只是确认它是一个自包含的小配置对象（独立小头设计的字面证据）。
    cfg = IndexerConfig(n_heads=2, head_dim=4)
    assert cfg.n_heads == 2
    assert cfg.head_dim == 4


def test_index_score_hand_computed_two_heads():
    # PAPER: §2.1 Eq.(1) —— 手算一个 I_{t,s}：H^I=2, d^I=2, T=1 query, S=2 key。
    q = np.array([[[1.0, 0.0], [0.0, 1.0]]])  # [T=1, H^I=2, d^I=2]
    k = np.array([[2.0, 0.0], [0.0, 3.0]])  # [S=2, d^I=2]
    w = np.array([[1.0, 0.5]])  # [T=1, H^I=2]

    scores = index_score(q, k, w)

    # dot(q_0, k_0)=2, dot(q_1, k_0)=0 -> I_{0,0}=1*relu(2)+0.5*relu(0)=2
    # dot(q_0, k_1)=0, dot(q_1, k_1)=3 -> I_{0,1}=1*relu(0)+0.5*relu(3)=1.5
    np.testing.assert_allclose(scores, [[2.0, 1.5]])


def test_index_score_relu_clips_negative_dot_products():
    # PAPER: §2.1 Eq.(1) —— ReLU(q.k) 把负点积压到 0，这是论文明确写出的激活函数
    # 选择（"We choose ReLU as the activation function for throughput
    # consideration"），不是 softmax。
    q = np.array([[[-1.0, 0.0]]])  # [T=1, H^I=1, d^I=2]
    k = np.array([[2.0, 0.0]])  # [S=1, d^I=2]
    w = np.array([[1.0]])

    scores = index_score(q, k, w)

    assert scores[0, 0] == 0.0  # relu(-2) = 0，不是 -2


def test_index_score_shape_assertions():
    q = np.zeros((2, 3, 4))
    k = np.zeros((5, 4))
    w_bad = np.zeros((2, 2))  # 头数与 q 不匹配
    with pytest.raises(AssertionError):
        index_score(q, k, w_bad)


def test_topk_select_basic():
    # PAPER: §2.1 Eq.(2) —— Top-k(I_{t,:})：给一行分数，手挑 top-2 索引。
    scores = np.array([[5.0, 1.0, 9.0, 3.0]])
    idx = topk_select(scores, k=2)
    np.testing.assert_array_equal(idx, [[2, 0]])  # 9 排第一，5 排第二


def test_topk_select_stable_tie_break():
    # 并列时索引小的优先（argsort(kind='stable') 的行为），与 vllm
    # top_k_per_row_prefill 的稳定选择一致。
    scores = np.array([[5.0, 5.0, 3.0]])
    idx = topk_select(scores, k=2)
    np.testing.assert_array_equal(idx, [[0, 1]])


def test_topk_select_k_larger_than_s_clips():
    scores = np.array([[1.0, 2.0]])
    idx = topk_select(scores, k=10)
    assert idx.shape == (1, 2)
