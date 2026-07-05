"""Reference implementation -- GPTQ's layer-wise reconstruction and
second-order compensation.

Faithful to:
  - GPTQ, arXiv:2210.17323, §3 "Background": Eq.1 (layer-wise reconstruction
    objective), the Optimal Brain Quantization (OBQ) background it builds
    on -- Eq.2 (greedy weight pick + compensation) and Eq.3 (Hessian-inverse
    row/column removal via one step of Gaussian elimination).
  - §4 "The GPTQ Algorithm": Eq.4-5 (lazy batch-update reformulation of
    Eq.2-3 over a block of columns Q) and Algorithm 1 (the full pseudocode:
    arbitrary/same column order across all rows + B=128 lazy batching +
    Cholesky-reformulated, dampened Hessian inverse).
  - §5 "Setup": "we perform standard uniform per-row asymmetric quantization
    on the min-max grid" -- the RTN quantization grid GPTQ itself, and this
    module's `make_asymmetric_per_row_quantizer`, use as the fixed grid
    (§3: "we assume that the quantization grid for What is fixed before the
    process").

Landing anchor: the GPTQ algorithm itself is an *offline* calibration
procedure (msmodelslim / llm-compressor-style tooling); vllm_ascend does not
carry this code. It only consumes GPTQ's *output* format: group-quantized
int4 weights, packed 8-per-int32 with a symmetric zero-point
(vllm_ascend/quantization/methods/w4a16.py:L79-L82,L180-L186) -- see the
chapter narrative for how `group_size` there matches GPTQ's grouping
(e.g. the paper's "g128").
"""
from __future__ import annotations

import numpy as np


def reconstruction_error(W: np.ndarray, W_hat: np.ndarray, X: np.ndarray) -> float:
    # PAPER: GPTQ §3 Eq.1 -- argmin_What || W X - What X ||_2^2
    """The Eq.1 objective, evaluated (not minimized) for a candidate What."""
    return float(np.linalg.norm(W @ X - W_hat @ X) ** 2)


def hessian_from_activations(X: np.ndarray) -> np.ndarray:
    # PAPER: GPTQ §3 -- "the Hessian is H_F = 2 X_F X_F^T"
    """H = 2 X X^T for the full (not-yet-quantized) set of columns."""
    return 2.0 * X @ X.T


def dampen(H: np.ndarray, frac: float = 0.01) -> np.ndarray:
    # PAPER: GPTQ §4 "Step 3: Cholesky Reformulation" -- "adding a small
    # constant lambda (we always choose 1% of the average diagonal value)
    # to the diagonal elements of H"
    """H + lambda*I, lambda = frac * mean(diag(H))."""
    lam = frac * float(np.mean(np.diag(H)))
    return H + lam * np.eye(H.shape[0])


def make_asymmetric_per_row_quantizer(W_full: np.ndarray, n_bits: int = 4):
    # PAPER: GPTQ §5 "Setup" -- "we perform standard uniform per-row
    # asymmetric quantization on the min-max grid"; §3 -- the grid is fixed
    # before the greedy/GPTQ process starts.
    """Build a per-row asymmetric min-max quantizer, fixed once from the
    full (pre-quantization) weight matrix.

    Returns quant_fn(w) -> dequantized weights, where w may be a single
    column (shape (d_row,)) or the full matrix (shape (d_row, d_col));
    each row i always uses the same (scale_i, zero_i) pair, computed here.
    """
    minv = np.min(W_full, axis=1, keepdims=True)
    maxv = np.max(W_full, axis=1, keepdims=True)
    qmax = 2 ** n_bits - 1
    scale = (maxv - minv) / qmax
    scale = np.where(scale == 0, 1.0, scale)
    zero = np.round(-minv / scale)

    # PAPER: GPTQ §5 "Setup" -- quant_fn applies the fixed per-row grid built above
    def quant_fn(w):
        w = np.asarray(w, dtype=float)
        col_shape = w.ndim == 1
        if col_shape:
            w = w[:, np.newaxis]
        q = np.clip(np.round(w / scale + zero), 0, qmax)
        dq = (q - zero) * scale
        return dq[:, 0] if col_shape else dq

    return quant_fn


def remove_hessian_row_col(Hinv: np.ndarray, q: int) -> np.ndarray:
    # PAPER: GPTQ §3 Eq.3 (OBQ) -- one step of Gaussian elimination to update
    # H_F^-1 after quantizing w_q, without a full re-inversion:
    #   H_{-q}^-1 = (H^-1 - (1/[H^-1]_qq) H^-1_{:,q} H^-1_{q,:})_{-p}
    """Remove row/col q from Hinv via the Eq.3 rank-1 update, then drop that
    row/col from the result (the `_{-p}` in Eq.3)."""
    n = Hinv.shape[0]
    pivot = Hinv[q, q]
    updated = Hinv - np.outer(Hinv[:, q], Hinv[q, :]) / pivot
    keep = [i for i in range(n) if i != q]
    return updated[np.ix_(keep, keep)]


def obq_pick_and_compensate(Hinv_F: np.ndarray, w_F: np.ndarray, quant_fn):
    # PAPER: GPTQ §3 Eq.2 (OBQ) --
    #   w_q = argmin_q (quant(w_q)-w_q)^2 / [H_F^-1]_qq
    #   delta_F = -(w_q - quant(w_q)) / [H_F^-1]_qq * (H_F^-1)_{:,q}
    """One greedy OBQ step: pick the cheapest-to-quantize remaining weight
    (by squared error weighted by the inverse-Hessian diagonal) and compute
    the compensation applied to every other (still full-precision) weight.

    Returns (q_idx, delta_F, w_after) where w_after = w_F + delta_F, except
    entry q_idx is set to quant(w_F[q_idx]) directly.
    """
    quantized = quant_fn(w_F)
    diag = np.diag(Hinv_F)
    costs = (quantized - w_F) ** 2 / diag
    q = int(np.argmin(costs))
    delta_F = -(w_F[q] - quantized[q]) / Hinv_F[q, q] * Hinv_F[:, q]
    delta_F[q] = 0.0  # the just-quantized weight isn't "compensated"; it's replaced
    w_after = w_F + delta_F
    w_after[q] = quantized[q]
    return q, delta_F, w_after


def obq_quantize_row(w_row: np.ndarray, H: np.ndarray, quant_fn) -> np.ndarray:
    # PAPER: GPTQ §3 Eq.2-3 (OBQ) -- repeated greedy pick+compensate+shrink
    """Full greedy OBQ pass over one row: repeatedly call
    obq_pick_and_compensate and shrink Hinv (Eq.3) until every weight in the
    row has been quantized. Not the GPTQ algorithm itself (each row can pick
    a *different* order) -- this is the "arbitrary order insight" baseline
    GPTQ compares against in §4 Step 1."""
    d = w_row.shape[0]
    Hinv = np.linalg.inv(H)
    w = w_row.copy()
    result = np.zeros(d)
    remaining = list(range(d))
    for _ in range(d):
        q_local, delta_F, w_after = obq_pick_and_compensate(Hinv, w, quant_fn)
        result[remaining[q_local]] = w_after[q_local]
        w = np.delete(w_after, q_local)
        Hinv = remove_hessian_row_col(Hinv, q_local)
        del remaining[q_local]
    return result


def gptq_lazy_batch_compensate(Hinv_F: np.ndarray, w_Q: np.ndarray, quant_Q: np.ndarray, Q_idx):
    # PAPER: GPTQ §4 Eq.4-5 (lazy batch-update) --
    #   delta_F = -(w_Q - quant(w_Q)) ([H_F^-1]_QQ)^-1 (H_F^-1)_{:,Q}
    #   H_{-Q}^-1 = (H^-1 - H^-1_{:,Q} ([H^-1]_QQ)^-1 H^-1_{Q,:})_{-Q}
    """Batched version of obq_pick_and_compensate/remove_hessian_row_col:
    quantize a *whole block* Q of columns/weights at once, using the
    already-inverted Hinv_F block. Returns (delta_F, Hinv_after)."""
    Hqq_inv = np.linalg.inv(Hinv_F[np.ix_(Q_idx, Q_idx)])
    delta_F = -(w_Q - quant_Q) @ Hqq_inv @ Hinv_F[Q_idx, :]
    keep = [i for i in range(Hinv_F.shape[0]) if i not in Q_idx]
    update = Hinv_F[:, Q_idx] @ Hqq_inv @ Hinv_F[Q_idx, :]
    Hinv_after = (Hinv_F - update)[np.ix_(keep, keep)]
    return delta_F, Hinv_after


def gptq_quantize(W: np.ndarray, H: np.ndarray, quant_fn, blocksize: int = 128, percdamp: float = 0.01):
    # PAPER: GPTQ §4 Algorithm 1 -- full pseudocode: dampen + Cholesky-
    # reformulate the inverse Hessian, then process columns left-to-right in
    # blocks of `blocksize`, quantizing one column at a time within a block
    # (updating only that block), and applying one global compensation to
    # the remaining columns at the end of each block.
    """GPTQ's Algorithm 1, applied to the whole (d_row, d_col) matrix W at
    once (all rows share the same column order and the same Hinv, per the
    §4 Step 1 "arbitrary order insight").
    """
    d_row, d_col = W.shape
    H = dampen(H, percdamp)
    Hinv = np.linalg.inv(H)
    # H^-1 <- Cholesky(H^-1)^T : replace the inverse Hessian with the upper-
    # triangular factor U such that Hinv = U^T @ U (np.linalg.cholesky
    # returns the lower-triangular L with Hinv = L @ L^T, so U = L.T).
    Hinv = np.linalg.cholesky(Hinv).T

    Q = np.zeros_like(W)
    W = W.copy()
    for i in range(0, d_col, blocksize):
        j_end = min(i + blocksize, d_col)
        W_block = W[:, i:j_end].copy()
        Q_block = np.zeros_like(W_block)
        E_block = np.zeros_like(W_block)
        Hinv_block = Hinv[i:j_end, i:j_end]
        for j in range(j_end - i):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]
            q_col = quant_fn(w_col)
            Q_block[:, j] = q_col
            err = (w_col - q_col) / d
            E_block[:, j] = err
            # update the *rest of this block* (columns j+1 .. end) in place
            W_block[:, j:] -= np.outer(err, Hinv_block[j, j:])
        Q[:, i:j_end] = Q_block
        W[:, i:j_end] = W_block
        if j_end < d_col:
            # one global compensation of everything after this block
            W[:, j_end:] -= E_block @ Hinv[i:j_end, j_end:]
    return Q
