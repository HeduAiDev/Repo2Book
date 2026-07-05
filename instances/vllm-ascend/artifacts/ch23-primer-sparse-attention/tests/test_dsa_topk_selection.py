import numpy as np

from dsa_topk_selection import (
    indexer_then_sparse_attention,
    sparse_attention_output,
    topk_select,
)
from standard_attention import causal_attention_output


def test_topk_select_returns_k_highest_scoring_indices():
    scores = np.array([0.1, 0.9, 0.3, 0.8, 0.05])
    top2 = topk_select(scores, k=2)
    assert set(top2.tolist()) == {1, 3}


def test_topk_select_clamps_k_to_available_length():
    scores = np.array([0.1, 0.2])
    result = topk_select(scores, k=10)
    assert len(result) == 2


def test_sparse_attention_output_restricts_to_selected_indices():
    rng = np.random.default_rng(6)
    q_t = rng.normal(size=(4,))
    k_seq = rng.normal(size=(8, 4))
    v_seq = rng.normal(size=(8, 6))
    topk_indices = np.array([1, 3, 5])

    out = sparse_attention_output(q_t, k_seq, v_seq, topk_indices, d_k=4)
    expected = causal_attention_output(q_t, k_seq[topk_indices], v_seq[topk_indices], d_k=4)
    assert np.allclose(out, expected)


def test_sparse_attention_equals_dense_when_k_covers_full_sequence():
    # PAPER Eq.2: when the "top-k" set is the entire context, sparse attention degenerates to
    # the standard dense attention of Eq.2 -- sparsity is a strict restriction, not a different algorithm.
    rng = np.random.default_rng(7)
    q_t = rng.normal(size=(4,))
    k_seq = rng.normal(size=(5, 4))
    v_seq = rng.normal(size=(5, 6))
    indexer_scores = rng.normal(size=(5,))

    out, topk_indices = indexer_then_sparse_attention(q_t, indexer_scores, k_seq, v_seq, k=5, d_k=4)
    dense = causal_attention_output(q_t, k_seq, v_seq, d_k=4)
    assert len(topk_indices) == 5
    assert np.allclose(out, dense)


def test_indexer_then_sparse_attention_only_uses_topk_context():
    rng = np.random.default_rng(8)
    q_t = rng.normal(size=(4,))
    k_seq = rng.normal(size=(10, 4))
    v_seq = rng.normal(size=(10, 6))
    indexer_scores = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0])

    out, topk_indices = indexer_then_sparse_attention(q_t, indexer_scores, k_seq, v_seq, k=2, d_k=4)
    assert set(topk_indices.tolist()) == {0, 9}
    expected = causal_attention_output(q_t, k_seq[[0, 9]], v_seq[[0, 9]], d_k=4)
    assert np.allclose(out, expected)
