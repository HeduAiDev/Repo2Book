"""Reference implementation -- SmoothQuant's migration transform.

Faithful to:
  - SmoothQuant, arXiv:2211.10438, §4 "SmoothQuant" Eq.3 (the mathematically
    equivalent activation<->weight migration) and Eq.4 (the migration
    strength alpha that splits quantization difficulty between the two).

Landing anchor: vllm_ascend/quantization/methods/w8a8_static.py:L158-L161
computes `deq_scale = input_scale * weight_scale` -- i.e. it consumes
per-tensor/per-channel scales on *already-migrated* activations/weights;
the migration itself (this module) happens offline during calibration and
is not present as vllm_ascend code, only as the parameters it produces.
"""
from __future__ import annotations

import numpy as np

from uniform_quant import per_tensor_scale, quantize_smoothquant


def difficulty_factor(X: np.ndarray, W: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    # PAPER: SmoothQuant §4 Eq.4 -- s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
    """Per (input-)channel migration-strength factor s.

    X: (T, Ci) activations, W: (Ci, Co) weights. Returns s: (Ci,).
    alpha=0 leaves all difficulty on activations (s_j = 1/max(|W_j|));
    alpha=1 pushes all difficulty to weights (s_j = max(|X_j|));
    alpha=0.5 is the "well-balanced" sweet spot the paper reports for
    OPT/BLOOM models.
    """
    m_x = np.max(np.abs(X), axis=0)     # per input channel, over tokens
    m_w = np.max(np.abs(W), axis=1)     # per input channel, over output channels
    return (m_x ** alpha) / (m_w ** (1.0 - alpha))


def migrate(X: np.ndarray, W: np.ndarray, s: np.ndarray):
    # PAPER: SmoothQuant §4 Eq.3 -- Y = (X diag(s)^-1) (diag(s) W) = X_hat W_hat
    """Apply the mathematically-equivalent migration transform.

    X: (T, Ci), W: (Ci, Co), s: (Ci,). Returns (X_hat, W_hat) with
    X_hat @ W_hat == X @ W (up to floating point) for any s (Eq.3 holds for
    every positive s -- it is an algebraic identity, not an approximation;
    the *choice* of s in Eq.4 is what determines quantization-friendliness).
    """
    X_hat = X / s[np.newaxis, :]
    W_hat = W * s[:, np.newaxis]
    return X_hat, W_hat


def smoothquant_pipeline_error(X: np.ndarray, W: np.ndarray, alpha: float = 0.5, n_bits: int = 8):
    # PAPER: SmoothQuant §4 Eq.3-4 (composition) -- cross-validation helper,
    # not itself a numbered equation; combines migrate()+difficulty_factor()
    """Cross-validation helper (not from the paper's equations directly):
    compare the per-tensor-quantization output error of the raw (X, W) vs.
    the migrated (X_hat, W_hat) -- both dequantized then multiplied, compared
    against the exact FP Y = X @ W.

    This is the small-matrix "does migration actually help" demonstration
    the chapter's numerical walkthrough runs to cross-check the theory.
    """
    Y_exact = X @ W

    q_x_raw, d_x_raw = quantize_smoothquant(X, n_bits)
    q_w_raw, d_w_raw = quantize_smoothquant(W, n_bits)
    Y_raw = (q_x_raw * d_x_raw) @ (q_w_raw * d_w_raw)
    err_raw = float(np.linalg.norm(Y_raw - Y_exact))

    s = difficulty_factor(X, W, alpha)
    X_hat, W_hat = migrate(X, W, s)
    q_x_hat, d_x_hat = quantize_smoothquant(X_hat, n_bits)
    q_w_hat, d_w_hat = quantize_smoothquant(W_hat, n_bits)
    Y_smoothed = (q_x_hat * d_x_hat) @ (q_w_hat * d_w_hat)
    err_smoothed = float(np.linalg.norm(Y_smoothed - Y_exact))

    return err_raw, err_smoothed
