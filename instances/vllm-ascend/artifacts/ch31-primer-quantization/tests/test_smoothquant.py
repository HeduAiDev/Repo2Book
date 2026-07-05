"""Tests for smoothquant.py — SmoothQuant §4 Eq.3 (migration) and Eq.4
(difficulty factor s).
"""
import numpy as np
import pytest

from smoothquant import difficulty_factor, migrate, smoothquant_pipeline_error


def _toy_layer():
    # Y = X @ W, X: T x Ci, W: Ci x Co (SmoothQuant §2 convention).
    # Column 0 of X is a systematic activation outlier (~100x), matching
    # SmoothQuant's "outliers persist in fixed channels" observation (§2).
    rng = np.random.default_rng(1)
    T, Ci, Co = 6, 3, 2
    X = rng.normal(scale=1.0, size=(T, Ci))
    X[:, 0] *= 80.0  # outlier input channel
    W = rng.normal(scale=1.0, size=(Ci, Co))
    return X, W


def test_migrate_is_mathematically_equivalent_eq3():
    X, W = _toy_layer()
    s = np.array([2.0, 0.5, 1.5])
    X_hat, W_hat = migrate(X, W, s)
    Y = X @ W
    Y_hat = X_hat @ W_hat
    assert np.allclose(Y, Y_hat, atol=1e-8)


def test_difficulty_factor_extremes_eq4():
    X, W = _toy_layer()
    # alpha=0: s_j = 1 for all j (no migration at all: max(|X_j|)^0=1,
    # max(|W_j|)^1 in the denominator only) -- all difficulty stays on
    # activations.
    s0 = difficulty_factor(X, W, alpha=0.0)
    np.testing.assert_allclose(s0, 1.0 / np.max(np.abs(W), axis=1))
    # alpha=1: s_j = max(|X_j|) -- all difficulty pushed to weights.
    s1 = difficulty_factor(X, W, alpha=1.0)
    np.testing.assert_allclose(s1, np.max(np.abs(X), axis=0))


def test_difficulty_factor_alpha_half_balances_channel_max():
    X, W = _toy_layer()
    s = difficulty_factor(X, W, alpha=0.5)
    X_hat, W_hat = migrate(X, W, s)
    # After alpha=0.5 migration, per-channel activation max and weight max
    # (row-wise, matching channel j) should be much closer than before.
    x_max = np.max(np.abs(X), axis=0)
    w_max = np.max(np.abs(W), axis=1)
    x_hat_max = np.max(np.abs(X_hat), axis=0)
    w_hat_max = np.max(np.abs(W_hat), axis=1)
    before_gap = np.abs(np.log(x_max) - np.log(w_max))
    after_gap = np.abs(np.log(x_hat_max) - np.log(w_hat_max))
    assert np.all(after_gap < before_gap + 1e-9)
    assert np.max(after_gap) < 1e-6  # exact geometric balance at alpha=0.5


def test_smoothquant_reduces_per_tensor_quant_error_vs_raw():
    X, W = _toy_layer()
    err_raw, err_smoothed = smoothquant_pipeline_error(X, W, alpha=0.5, n_bits=8)
    assert err_smoothed < err_raw
