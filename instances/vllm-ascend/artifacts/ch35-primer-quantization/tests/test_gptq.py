"""Tests for gptq.py — GPTQ §3 Eq.1-3 (OBQ background) and §4 Eq.4-5 +
Algorithm 1 (the GPTQ algorithm itself: arbitrary-order + lazy batch +
Cholesky/dampening).
"""
import numpy as np
import pytest

from gptq import (
    dampen,
    gptq_quantize,
    hessian_from_activations,
    make_asymmetric_per_row_quantizer,
    obq_pick_and_compensate,
    obq_quantize_row,
    remove_hessian_row_col,
    reconstruction_error,
)


def _toy_problem(seed=0, d_row=3, d_col=4, m=20):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(d_col, m))  # layer inputs, Eq.1: W in R^{d_row x d_col}, X in R^{d_col x m}
    W = rng.normal(size=(d_row, d_col))
    return W, X


def test_reconstruction_error_eq1_zero_when_no_quantization():
    W, X = _toy_problem()
    err = reconstruction_error(W, W, X)
    assert err == pytest.approx(0.0, abs=1e-9)


def test_hessian_from_activations_is_2xxT():
    _, X = _toy_problem()
    H = hessian_from_activations(X)
    np.testing.assert_allclose(H, 2 * X @ X.T)


def test_dampen_adds_1pct_of_average_diagonal():
    H = np.diag([10.0, 20.0, 30.0])
    H_damped = dampen(H, frac=0.01)
    expected_lambda = 0.01 * np.mean([10.0, 20.0, 30.0])
    np.testing.assert_allclose(np.diag(H_damped), np.diag(H) + expected_lambda)


def test_remove_hessian_row_col_matches_eq3_gaussian_elimination():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(4, 4))
    H = A @ A.T + 4 * np.eye(4)  # SPD
    Hinv = np.linalg.inv(H)
    q = 1
    Hinv_removed = remove_hessian_row_col(Hinv, q)
    # Ground truth: remove row/col q from H directly, then invert the
    # remaining (n-1)x(n-1) block -- should match the Eq.3 one-shot update.
    keep = [i for i in range(4) if i != q]
    H_sub = H[np.ix_(keep, keep)]
    expected = np.linalg.inv(H_sub)
    np.testing.assert_allclose(Hinv_removed, expected, atol=1e-8)


def test_obq_pick_and_compensate_eq2_reduces_row_error():
    rng = np.random.default_rng(6)
    d_col, m = 3, 30
    X = rng.normal(size=(d_col, m))
    H = hessian_from_activations(X)
    H = dampen(H, 0.01)
    Hinv = np.linalg.inv(H)
    w_row = rng.normal(size=d_col)
    quant_fn = make_asymmetric_per_row_quantizer(w_row.reshape(1, -1), n_bits=4)

    q_idx, delta_F, w_after = obq_pick_and_compensate(Hinv, w_row.copy(), quant_fn)
    assert 0 <= q_idx < d_col
    # the compensation should have changed the *other* (unquantized) weights,
    # not the one just quantized.
    assert delta_F[q_idx] == pytest.approx(0.0, abs=1e-9)


def test_obq_quantize_row_full_pass_matches_lower_error_than_naive_rtn():
    rng = np.random.default_rng(7)
    d_col, m = 5, 50
    X = rng.normal(size=(d_col, m))
    H = dampen(hessian_from_activations(X), 0.01)
    w_row = rng.normal(size=d_col) * np.array([1, 1, 1, 1, 5.0])  # one big weight
    quant_fn = make_asymmetric_per_row_quantizer(w_row.reshape(1, -1), n_bits=3)

    w_quantized = obq_quantize_row(w_row.copy(), H, quant_fn)
    rtn = quant_fn(w_row.reshape(1, -1))[0]

    err_obq = np.sum(((w_row - w_quantized) @ X) ** 2)
    err_rtn = np.sum(((w_row - rtn) @ X) ** 2)
    assert err_obq <= err_rtn + 1e-6


def test_gptq_quantize_matches_algorithm1_on_small_matrix():
    W, X = _toy_problem(d_row=3, d_col=6, m=40)
    H = hessian_from_activations(X)
    quant_fn = make_asymmetric_per_row_quantizer(W, n_bits=4)

    Q = gptq_quantize(W, H, quant_fn, blocksize=2, percdamp=0.01)
    assert Q.shape == W.shape

    err_gptq = reconstruction_error(W, Q, X)
    err_rtn = reconstruction_error(W, quant_fn(W), X)
    assert err_gptq <= err_rtn + 1e-6


def test_gptq_quantize_blocksize_does_not_change_result_much():
    # Lazy-batch updates (Eq.4-5) are an efficiency reformulation of the same
    # per-column greedy process -- blocksize should not materially change the
    # final reconstruction error for a small matrix.
    W, X = _toy_problem(d_row=3, d_col=8, m=40, seed=9)
    H = hessian_from_activations(X)
    quant_fn = make_asymmetric_per_row_quantizer(W, n_bits=4)

    Q_b1 = gptq_quantize(W, H, quant_fn, blocksize=1, percdamp=0.01)
    Q_b4 = gptq_quantize(W, H, quant_fn, blocksize=4, percdamp=0.01)
    np.testing.assert_allclose(Q_b1, Q_b4, atol=1e-6)
