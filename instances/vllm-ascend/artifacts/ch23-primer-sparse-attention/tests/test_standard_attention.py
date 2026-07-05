import numpy as np
import pytest

from standard_attention import (
    causal_attention_output,
    causal_attention_scores,
    quadratic_attention_flops,
    quadratic_dot_product_count,
)


def test_scores_sum_to_one():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(8,))
    k = rng.normal(size=(5, 8))
    alpha = causal_attention_scores(q, k, d_k=8)
    assert alpha.shape == (5,)
    assert np.isclose(alpha.sum(), 1.0)
    assert np.all(alpha >= 0.0)


def test_scores_uniform_when_query_orthogonal_to_all_keys():
    # q dotted with every key is 0 -> all logits equal -> softmax uniform.
    k = np.eye(4)[:, :2]  # (4, 2), each row has a zero in the last two dims
    q = np.array([0.0, 0.0])
    alpha = causal_attention_scores(q, k, d_k=2)
    assert np.allclose(alpha, 0.25)


def test_causal_attention_output_matches_manual_weighted_sum():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(4,))
    k = rng.normal(size=(3, 4))
    v = rng.normal(size=(3, 6))
    out = causal_attention_output(q, k, v, d_k=4)
    alpha = causal_attention_scores(q, k, d_k=4)
    expected = alpha @ v
    assert np.allclose(out, expected)


@pytest.mark.parametrize("seq_len,expected", [(1, 1), (2, 3), (4, 10), (10, 55)])
def test_quadratic_dot_product_count_matches_triangular_number(seq_len, expected):
    # PAPER: Sum_{t=1}^{L} t = L(L+1)/2 (Eq.2's causal sum, accumulated over all query positions)
    assert quadratic_dot_product_count(seq_len) == expected


def test_quadratic_attention_flops_scales_with_d_k():
    count = quadratic_dot_product_count(100)
    assert quadratic_attention_flops(100, d_k=64) == count * 64
    assert quadratic_attention_flops(100, d_k=128) == count * 128


def test_dot_product_count_grows_quadratically_not_linearly():
    # Doubling L should roughly quadruple the dot-product count -- the "O(L^2) tax".
    small = quadratic_dot_product_count(1000)
    large = quadratic_dot_product_count(2000)
    assert large / small > 3.9  # exact ratio -> 4 as L -> infinity
