"""Tests for uniform_quant.py — SmoothQuant §2 Eq.1 / AWQ §3.2 Eq.1 / SmoothQuant §3.

These pin down the two competing "symmetric absmax" conventions used across
the paper pack (denominator 2^(N-1)-1 vs 2^(N-1)), the three granularities
from SmoothQuant Fig.3, and the "effective quantization levels" outlier
argument from SmoothQuant §3.
"""
import numpy as np
import pytest

from uniform_quant import (
    awq_scale,
    dequantize,
    effective_quant_levels,
    per_channel_scale,
    per_tensor_scale,
    per_token_scale,
    quantize_awq,
    quantize_smoothquant,
    smoothquant_scale,
)


def test_smoothquant_scale_matches_eq1_denominator():
    # Eq.1: Delta = max(|X|) / (2^(N-1) - 1). For N=8, |x|max=127 -> Delta=1.
    x = np.array([-127.0, 0.0, 64.0, 127.0])
    delta = smoothquant_scale(x, n_bits=8)
    assert delta == pytest.approx(127.0 / 127.0)


def test_awq_scale_matches_eq1_denominator():
    # AWQ Eq.1: Delta = max(|w|) / 2^(N-1). For N=8, |w|max=128 -> Delta=1.
    w = np.array([-128.0, 0.0, 64.0, 128.0])
    delta = awq_scale(w, n_bits=8)
    assert delta == pytest.approx(128.0 / 128.0)


def test_smoothquant_and_awq_scale_differ_by_convention():
    # Same data, same n_bits: the two papers' denominators differ by exactly
    # one code point (2^(N-1)-1 vs 2^(N-1)) -- this is the discrepancy the
    # dossier flags as "not a contradiction, two common conventions".
    x = np.array([-100.0, 50.0, 100.0])
    d_sq = smoothquant_scale(x, n_bits=8)
    d_awq = awq_scale(x, n_bits=8)
    assert d_sq > d_awq
    assert d_sq / d_awq == pytest.approx(128.0 / 127.0)


def test_quantize_dequantize_roundtrip_small_error():
    rng = np.random.default_rng(0)
    x = rng.normal(size=64)
    q, delta = quantize_smoothquant(x, n_bits=8)
    x_hat = dequantize(q, delta)
    # int8 with ~8 bits should keep relative error small on a well-behaved
    # (no-outlier) vector.
    assert np.max(np.abs(x_hat - x)) < 0.05 * np.max(np.abs(x))


def test_quantize_smoothquant_codes_stay_in_int8_symmetric_range():
    x = np.array([-500.0, -1.0, 0.0, 1.0, 500.0])
    q, _ = quantize_smoothquant(x, n_bits=8)
    assert q.min() >= -127
    assert q.max() <= 127


def test_quantize_awq_codes_stay_in_symmetric_range():
    w = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    q, delta = quantize_awq(w, n_bits=4)
    dq = q  # quantize_awq already returns the dequantized Q(w) per Eq.1
    assert np.max(np.abs(dq)) <= 8 * delta + 1e-9  # 2^(N-1)=8 for N=4


def test_per_tensor_scale_is_single_scalar():
    x = np.array([[1.0, 2.0], [3.0, -8.0]])
    delta = per_tensor_scale(x, n_bits=8)
    assert np.isscalar(delta) or delta.shape == ()


def test_per_token_scale_is_one_per_row():
    # X: T x Ci (SmoothQuant Fig.3 convention) -- per-token = per row of X.
    x = np.array([[1.0, 2.0, -4.0], [10.0, -20.0, 5.0]])
    scale = per_token_scale(x, n_bits=8)
    assert scale.shape == (2,)
    assert scale[1] > scale[0]  # row 1 has bigger magnitudes


def test_per_channel_scale_is_one_per_output_channel():
    # W: Ci x Co (SmoothQuant Fig.3 convention) -- per-channel = per column
    # (output-channel dimension Co), the only dimension INT8 GEMM can scale
    # on the weight side.
    w = np.array([[1.0, 100.0], [2.0, -50.0], [-3.0, 10.0]])
    scale = per_channel_scale(w, n_bits=8)
    assert scale.shape == (2,)
    assert scale[1] > scale[0]


def test_effective_quant_levels_outlier_collapse():
    # SmoothQuant §3: effective levels of channel i = 2^N * m_i / m.
    # An outlier channel ~100x larger than the rest should leave the
    # non-outlier channels with only a handful of effective levels.
    x = np.zeros((4, 3))
    x[:, 0] = np.array([1.0, -1.0, 0.5, -0.8])       # normal channel
    x[:, 1] = np.array([1.2, -0.9, 0.7, -1.1])       # normal channel
    x[:, 2] = np.array([100.0, -95.0, 90.0, -110.0])  # outlier channel
    levels = effective_quant_levels(x, n_bits=8)
    assert levels.shape == (3,)
    # outlier channel keeps (close to) the full 2^8 levels
    assert levels[2] == pytest.approx(256.0, rel=0.1)
    # non-outlier channels collapse to a handful of levels
    assert levels[0] < 5
    assert levels[1] < 5
