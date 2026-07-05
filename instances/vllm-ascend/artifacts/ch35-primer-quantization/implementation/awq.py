"""Reference implementation -- AWQ's activation-aware weight scaling.

Faithful to:
  - AWQ, arXiv:2306.00978, §3.2 Eq.1 (weight-only quantize-dequantize),
    Eq.2-3 (the scaling argument: multiply a salient weight by s>1, divide
    the matching activation by s -- mathematically equivalent, and reduces
    the salient weight's *relative* quantization error to ~ Delta'/Delta *
    1/s), Eq.4-5 (the search space s = s_X^alpha and the grid search for the
    best alpha).

Landing anchor: the scaling search itself is an *offline* calibration step
(not present in vllm_ascend); the weights it produces land as W4A16
(vllm_ascend/quantization/methods/w4a16.py) group-quantized int4 --
group_size is the AWQ/GPTQ "g128"-style grouping this chapter also
discusses for GPTQ.
"""
from __future__ import annotations

import numpy as np

from uniform_quant import awq_scale, quantize_awq


def round_err(t: np.ndarray) -> float:
    # PAPER: AWQ §3.2 -- "RoundErr(.) ~ 0.25": the expected rounding error is
    # roughly uniform on [0, 0.5], averaging 0.25.
    """Mean absolute rounding error |round(t) - t| for arbitrary reals t."""
    return float(np.mean(np.abs(np.round(t) - t)))


def quantize_dequantize(w: np.ndarray, n_bits: int = 8) -> np.ndarray:
    # PAPER: AWQ §3.2 Eq.1 -- Q(w) = Delta * Round(w / Delta)
    """AWQ's weight-only quantize-dequantize function Q(.)."""
    q, _ = quantize_awq(w, n_bits)
    return q


def scaled_error_ratio(group: np.ndarray, salient_idx: int, s: float, n_bits: int = 4):
    # PAPER: AWQ §3.2 Eq.2-3 -- Q(w*s)*(x/s) vs. Q(w)*x; the ratio of the
    # scaled-element's error to its original error is Delta'/Delta * 1/s.
    """Measure the quantization-error ratio for one "salient" element of a
    weight group when it is scaled up by s (and the matching activation is
    scaled down by 1/s, which is mathematically transparent and doesn't
    appear numerically here -- Eq.2 shows the x/s cancels in the *value*
    Q(w*s)*(x/s), only the *quantization error on w* is affected).

    Returns (naive_predicted_ratio, actual_measured_ratio, delta_ratio):
      naive_predicted_ratio = 1/s -- the paper's simplified conclusion,
        valid under the approximation Delta' ~= Delta (true when scaling
        the single salient element doesn't change the group's absmax).
      delta_ratio = Delta'/Delta -- how much the grid step actually moved;
        the exact error ratio is delta_ratio * naive_predicted_ratio.
      actual_measured_ratio = the true |quant_error_after| / |quant_error_before|.
    """
    group = np.asarray(group, dtype=float)
    delta = awq_scale(group, n_bits)
    original_val = group[salient_idx]
    original_err = abs(quantize_dequantize(group, n_bits)[salient_idx] - original_val)

    scaled_group = group.copy()
    scaled_group[salient_idx] = original_val * s
    delta_scaled = awq_scale(scaled_group, n_bits)
    # Eq.2: dequantized value of the scaled weight, divided back by s, is
    # what actually reaches the output (x/s cancels the w*s).
    scaled_quant = quantize_dequantize(scaled_group, n_bits)[salient_idx] / s
    scaled_err = abs(scaled_quant - original_val)

    naive_predicted_ratio = 1.0 / s
    delta_ratio = delta_scaled / delta
    actual_measured_ratio = scaled_err / original_err if original_err > 0 else 0.0
    return naive_predicted_ratio, actual_measured_ratio, delta_ratio


# PAPER: AWQ §3.2 Eq.4-5 -- s = s_X^alpha; alpha* = argmin_alpha L(s_X^alpha)
# where L(s) = || Q(W diag(s)) (diag(s)^-1 X^T)^T - W X^T ||  (reconstruction
# error of the linear layer's output after scaling+quantizing the weight).
def search_alpha(
    X: np.ndarray,
    W: np.ndarray,
    s_x: np.ndarray,
    n_bits: int = 4,
    alphas: tuple = (0.0, 0.25, 0.5, 0.75, 1.0),
):
    """Grid-search alpha in [0,1] over a fixed candidate set.

    X: (T, Ci) activations. W: (Ci, Co) weights. s_x: (Ci,) average
    per-(input-)channel activation magnitude (the "s_X" of Eq.5).
    Returns (best_alpha, losses) where losses[i] is L(s_X^alphas[i]).
    """
    Y_exact = X @ W
    losses = []
    for alpha in alphas:
        s = s_x ** alpha
        W_scaled = W * s[:, np.newaxis]           # diag(s) @ W
        X_scaled = X / s[np.newaxis, :]           # X @ diag(s)^-1
        W_quant = quantize_dequantize(W_scaled, n_bits)
        Y_hat = X_scaled @ W_quant
        losses.append(float(np.linalg.norm(Y_hat - Y_exact)))
    best_alpha = alphas[int(np.argmin(losses))]
    return best_alpha, losses
