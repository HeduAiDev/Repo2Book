import numpy as np

from nsa_framework import (
    branch_attention_output,
    gated_multi_branch_output,
    sparsity_ratio,
    total_remapped_size,
)
from standard_attention import causal_attention_scores


def test_branch_attention_output_empty_branch_is_zero():
    v_tilde = np.zeros((0, 6))
    k_tilde = np.zeros((0, 4))
    q = np.zeros(4)
    out = branch_attention_output(q, k_tilde, v_tilde, d_k=4)
    assert out.shape == (0,) or np.allclose(out, 0.0)


def test_branch_attention_output_matches_causal_attention():
    rng = np.random.default_rng(2)
    q = rng.normal(size=(4,))
    k_tilde = rng.normal(size=(3, 4))
    v_tilde = rng.normal(size=(3, 5))
    out = branch_attention_output(q, k_tilde, v_tilde, d_k=4)
    alpha = causal_attention_scores(q, k_tilde, d_k=4)
    assert np.allclose(out, alpha @ v_tilde)


def test_gated_multi_branch_output_is_weighted_sum_of_branches():
    rng = np.random.default_rng(3)
    q = rng.normal(size=(4,))
    branch_k = {
        "cmp": rng.normal(size=(2, 4)),
        "slc": rng.normal(size=(3, 4)),
        "win": rng.normal(size=(2, 4)),
    }
    branch_v = {
        "cmp": rng.normal(size=(2, 5)),
        "slc": rng.normal(size=(3, 5)),
        "win": rng.normal(size=(2, 5)),
    }
    gates = {"cmp": 0.2, "slc": 0.5, "win": 0.3}

    out = gated_multi_branch_output(q, branch_k, branch_v, gates, d_k=4)

    expected = sum(
        gates[c] * branch_attention_output(q, branch_k[c], branch_v[c], d_k=4) for c in gates
    )
    assert np.allclose(out, expected)


def test_total_remapped_size_sums_branch_sizes():
    branch_k = {"cmp": np.zeros((4, 8)), "slc": np.zeros((16, 8)), "win": np.zeros((8, 8))}
    assert total_remapped_size(branch_k) == 4 + 16 + 8


def test_sparsity_ratio_definition():
    # PAPER: N_t << t is the definition of "high sparsity"
    assert sparsity_ratio(n_t=28, t=131072) < 0.001
    assert sparsity_ratio(n_t=0, t=0) == 0.0
    assert np.isclose(sparsity_ratio(n_t=10, t=100), 0.1)
