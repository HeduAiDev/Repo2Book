"""Reference implementation -- uniform (absmax) quantization basics.

Faithful to:
  - SmoothQuant, arXiv:2211.10438, §2 "Preliminaries" Eq.1 (symmetric
    per-tensor quantization), Fig.3 (per-tensor / per-token / per-channel
    granularity), §3 "Review of Quantization Difficulty" (effective
    quantization levels / outlier collapse argument).
  - AWQ, arXiv:2306.00978, §3.2 Eq.1 (the weight-only quantize-dequantize
    function AWQ builds its scaling argument on top of).

Both papers define "the same idea" -- round(x/Delta) on a symmetric absmax
grid -- but with denominators that differ by one code point:
  SmoothQuant Eq.1:  Delta = max(|X|) / (2^(N-1) - 1)   (keeps one code free)
  AWQ        Eq.1:  Delta = max(|w|) / 2^(N-1)          (full symmetric range)
Both conventions are common in the quantization literature; the chapter
narrative points this out explicitly so a reader comparing the two papers
side by side doesn't mistake it for a contradiction. This module keeps both
functions distinct rather than silently picking one.

Landing anchor (vllm_ascend/quantization/methods/w8a8_static.py:L53-L72):
input_scale is a single per-tensor scalar (SmoothQuant per-tensor Eq.1),
weight_scale has shape [output_size, 1] -- one scale per output channel
(SmoothQuant per-channel, Fig.3).
"""
from __future__ import annotations

import numpy as np


def smoothquant_scale(x: np.ndarray, n_bits: int = 8) -> np.ndarray:
    # PAPER: SmoothQuant §2 Eq.1 -- Delta = max(|X|) / (2^(N-1) - 1)
    """Symmetric absmax quantization step (SmoothQuant convention).

    Reserves one code point at the negative extreme (a common asymmetric-
    range trick for a nominally symmetric quantizer): the positive side maps
    onto [0, 2^(N-1)-1].
    """
    max_abs = np.max(np.abs(x))
    qmax = 2 ** (n_bits - 1) - 1
    return max_abs / qmax if max_abs > 0 else 1.0


def awq_scale(w: np.ndarray, n_bits: int = 8) -> np.ndarray:
    # PAPER: AWQ §3.2 Eq.1 -- Delta = max(|w|) / 2^(N-1)
    """Symmetric absmax quantization step (AWQ convention: full range)."""
    max_abs = np.max(np.abs(w))
    qmax = 2 ** (n_bits - 1)
    return max_abs / qmax if max_abs > 0 else 1.0


def quantize_smoothquant(x: np.ndarray, n_bits: int = 8):
    # PAPER: SmoothQuant §2 Eq.1 -- Xbar^INT8 = round(X^FP16 / Delta)
    """Quantize to (clipped) integer codes on the SmoothQuant grid.

    Returns (int_codes, delta). Codes are clipped to
    [-(2^(N-1)-1), 2^(N-1)-1] -- the symmetric range Eq.1 implies.
    """
    delta = smoothquant_scale(x, n_bits)
    qmax = 2 ** (n_bits - 1) - 1
    codes = np.clip(np.round(x / delta), -qmax, qmax)
    return codes, delta


def dequantize(codes: np.ndarray, delta: np.ndarray) -> np.ndarray:
    # PAPER: SmoothQuant §2 Eq.1 (inverse direction) -- x_hat = code * Delta
    """Inverse of quantize_smoothquant: multiply the integer codes back by
    the step size Delta (elementwise inverse of Eq.1's forward direction)."""
    return codes * delta


def quantize_awq(w: np.ndarray, n_bits: int = 8):
    # PAPER: AWQ §3.2 Eq.1 -- Q(w) = Delta * Round(w / Delta)
    """Quantize-dequantize in one step, exactly as AWQ Eq.1 defines Q(.).

    Unlike quantize_smoothquant, AWQ's Eq.1 already folds the dequantize
    step in (Q(w) returns a real number back on the FP16 grid), so this
    returns the *dequantized* value directly, matching how the rest of
    awq.py consumes it.
    """
    delta = awq_scale(w, n_bits)
    qmax = 2 ** (n_bits - 1)
    # AWQ's Eq.1 divisor (2^(N-1), not 2^(N-1)-1) is exactly what's needed for
    # the absmax element to round-trip to +-qmax -- so the code range here is
    # the full symmetric [-qmax, qmax], not a two's-complement [-qmax, qmax-1]
    # register range (this module models the quantization math, not a fixed
    # storage width).
    codes = np.clip(np.round(w / delta), -qmax, qmax)
    return codes * delta, delta


def per_tensor_scale(x: np.ndarray, n_bits: int = 8):
    # PAPER: SmoothQuant §2 Fig.3 -- "per-tensor quantization uses a single
    # step size for the entire matrix"
    """A single scalar Delta for the whole tensor."""
    return smoothquant_scale(x, n_bits)


def per_token_scale(x: np.ndarray, n_bits: int = 8) -> np.ndarray:
    # PAPER: SmoothQuant §2 Fig.3 -- per-token quantization: one scale per
    # row of the activation matrix X (T x Ci convention, token dimension T)
    """One scale per activation row (token). x: (T, Ci)."""
    max_abs = np.max(np.abs(x), axis=1)
    qmax = 2 ** (n_bits - 1) - 1
    return np.where(max_abs > 0, max_abs / qmax, 1.0)


def per_channel_scale(w: np.ndarray, n_bits: int = 8) -> np.ndarray:
    # PAPER: SmoothQuant §2 Fig.3 -- per-channel quantization: one scale per
    # output-channel column of the weight matrix W (Ci x Co convention) --
    # the only weight-side dimension INT8 GEMM can scale after the fact.
    """One scale per weight output-channel (column). w: (Ci, Co)."""
    max_abs = np.max(np.abs(w), axis=0)
    qmax = 2 ** (n_bits - 1) - 1
    return np.where(max_abs > 0, max_abs / qmax, 1.0)


def effective_quant_levels(x: np.ndarray, n_bits: int = 8) -> np.ndarray:
    # PAPER: SmoothQuant §3 -- "the effective quantization levels of channel
    # i is 2^8 * m_i / m", where m_i is channel i's max magnitude and m is
    # the whole matrix's max magnitude (per-tensor quantization case).
    """Per-(input-)channel effective quantization levels under per-tensor
    quantization. x: (T, Ci). Returns one value per column (channel).

    This is the quantitative "why W8A8 is risky" argument: an outlier
    channel (m_i close to m) keeps close to the full 2^N levels, while a
    non-outlier channel (m_i << m) collapses to just a handful of levels.
    """
    m = np.max(np.abs(x))
    m_i = np.max(np.abs(x), axis=0)
    return (2 ** n_bits) * m_i / m
