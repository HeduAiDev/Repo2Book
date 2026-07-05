import numpy as np

from lightning_indexer import indexer_score, indexer_scores_for_query


def test_indexer_score_matches_manual_relu_weighted_sum():
    q_t = np.array([[1.0, 2.0], [-1.0, 3.0], [0.5, 0.5]])  # (H^I=3, d^I=2)
    k_s = np.array([1.0, 1.0])  # (d^I=2,) shared across all indexer heads
    w_t = np.array([0.5, 0.2, 1.0])  # (H^I=3,)

    dots = q_t @ k_s  # [3.0, 2.0, 1.0]
    relu = np.maximum(dots, 0.0)  # all positive here
    expected = float(w_t @ relu)

    assert np.isclose(indexer_score(q_t, k_s, w_t), expected)


def test_relu_zeroes_out_negative_dot_products():
    # PAPER Eq.1: ReLU(q^I . k^I) -- negative dot products contribute exactly 0, not a negative term
    q_t = np.array([[-5.0, -5.0]])  # single head, will dot to a very negative number
    k_s = np.array([1.0, 1.0])
    w_t = np.array([10.0])  # large weight -- if ReLU weren't applied this would swing the score hugely negative
    score = indexer_score(q_t, k_s, w_t)
    assert score == 0.0


def test_indexer_scores_for_query_matches_per_pair_computation():
    rng = np.random.default_rng(5)
    H_I, d_I, t = 4, 8, 6
    q_t = rng.normal(size=(H_I, d_I))
    k_seq = rng.normal(size=(t, d_I))
    w_t = rng.normal(size=(H_I,))

    batched = indexer_scores_for_query(q_t, k_seq, w_t)
    manual = np.array([indexer_score(q_t, k_seq[s], w_t) for s in range(t)])
    assert np.allclose(batched, manual)


def test_indexer_scores_shape_matches_context_length():
    q_t = np.zeros((64, 128))  # vllm_ascend deployment shape: H^I=64, d^I=128
    k_seq = np.zeros((37, 128))
    w_t = np.zeros(64)
    scores = indexer_scores_for_query(q_t, k_seq, w_t)
    assert scores.shape == (37,)
