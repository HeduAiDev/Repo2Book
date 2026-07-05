import numpy as np
import pytest

from mhc_sinkhorn import (
    apply_constraints,
    apply_sigmoid_constraints,
    dynamic_raw_params,
    hc_residual_update,
    is_doubly_stochastic,
    layer_input,
    rms_norm_flatten,
    sinkhorn_knopp,
)


def test_hc_residual_update_matches_eq1():
    n_hc, d = 3, 2
    X_l = np.arange(n_hc * d).reshape(n_hc, d).astype(float)
    A_l = np.ones((1, n_hc)) / n_hc
    B_l = np.eye(n_hc)
    C_l = np.ones((n_hc, 1))
    F_out = np.array([[1.0, 2.0]])
    out = hc_residual_update(X_l, A_l, B_l, C_l, F_out)
    expected = B_l @ X_l + C_l @ F_out
    np.testing.assert_allclose(out, expected)


def test_layer_input_is_weighted_average():
    X_l = np.array([[1.0, 1.0], [3.0, 3.0]])
    A_l = np.array([[0.5, 0.5]])
    out = layer_input(X_l, A_l)
    np.testing.assert_allclose(out, [[2.0, 2.0]])


def test_is_doubly_stochastic_true_for_identity():
    assert is_doubly_stochastic(np.eye(4))


def test_is_doubly_stochastic_false_for_arbitrary_matrix():
    M = np.array([[0.9, 0.9], [0.1, 0.1]])
    assert not is_doubly_stochastic(M)


def test_rms_norm_flatten_unit_rms():
    X_l = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = rms_norm_flatten(X_l, eps=0.0)
    rms = np.sqrt(np.mean(out ** 2))
    assert rms == pytest.approx(1.0, rel=1e-6)
    assert out.shape == (1, 4)


def test_sinkhorn_knopp_converges_to_doubly_stochastic():
    rng = np.random.default_rng(0)
    B_tilde = rng.normal(size=(5, 5))
    B_l = sinkhorn_knopp(B_tilde, iters=20)
    assert is_doubly_stochastic(B_l, atol=1e-4)


def test_sinkhorn_knopp_more_iterations_converge_more_tightly():
    rng = np.random.default_rng(1)
    B_tilde = rng.normal(size=(6, 6))
    few = sinkhorn_knopp(B_tilde, iters=1)
    many = sinkhorn_knopp(B_tilde, iters=20)

    def deviation(M):
        return np.abs(M.sum(axis=0) - 1).sum() + np.abs(M.sum(axis=1) - 1).sum()

    assert deviation(many) < deviation(few)


def test_apply_sigmoid_constraints_ranges():
    A_tilde = np.array([[-100.0, 0.0, 100.0]])
    C_tilde = np.array([[-100.0], [0.0], [100.0]])
    A_l, C_l = apply_sigmoid_constraints(A_tilde, C_tilde)
    assert np.all(A_l >= 0) and np.all(A_l <= 1)
    assert np.all(C_l >= 0) and np.all(C_l <= 2)
    # 中间值应接近 0.5 / 1.0
    assert A_l[0, 1] == pytest.approx(0.5, abs=1e-6)
    assert C_l[1, 0] == pytest.approx(1.0, abs=1e-6)


def test_dynamic_raw_params_shapes():
    n_hc, d = 3, 4
    X_l = np.random.default_rng(0).normal(size=(n_hc, d))
    W_pre = np.random.default_rng(1).normal(size=(n_hc * d, n_hc))
    W_res = np.random.default_rng(2).normal(size=(n_hc * d, n_hc * n_hc))
    W_post = np.random.default_rng(3).normal(size=(n_hc * d, n_hc))
    S_pre = np.zeros((1, n_hc))
    S_res = np.zeros((n_hc, n_hc))
    S_post = np.zeros((n_hc, 1))
    A_tilde, B_tilde, C_tilde = dynamic_raw_params(
        X_l, W_pre, W_res, W_post, S_pre, S_res, S_post, alpha_pre=0.1, alpha_res=0.1, alpha_post=0.1
    )
    assert A_tilde.shape == (1, n_hc)
    assert B_tilde.shape == (n_hc, n_hc)
    assert C_tilde.shape == (n_hc, 1)


def test_apply_constraints_end_to_end_produces_valid_hc_params():
    n_hc, d = 4, 3
    rng = np.random.default_rng(4)
    X_l = rng.normal(size=(n_hc, d))
    W_pre = rng.normal(size=(n_hc * d, n_hc))
    W_res = rng.normal(size=(n_hc * d, n_hc * n_hc))
    W_post = rng.normal(size=(n_hc * d, n_hc))
    S_pre, S_res, S_post = np.zeros((1, n_hc)), np.zeros((n_hc, n_hc)), np.zeros((n_hc, 1))
    A_tilde, B_tilde, C_tilde = dynamic_raw_params(
        X_l, W_pre, W_res, W_post, S_pre, S_res, S_post, alpha_pre=0.5, alpha_res=0.5, alpha_post=0.5
    )
    A_l, B_l, C_l = apply_constraints(A_tilde, B_tilde, C_tilde, sinkhorn_iters=20)
    assert np.all(A_l >= 0) and np.all(A_l <= 1)
    assert np.all(C_l >= 0) and np.all(C_l <= 2)
    assert is_doubly_stochastic(B_l, atol=1e-4)
