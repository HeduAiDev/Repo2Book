"""Driver for m3-gptq-second-order.

Uses the paper-faithful reference impl (implementation/gptq.py, GPTQ
arXiv:2210.17323 §3-§4 Alg.1) on a tiny (1x4) weight row so the per-column
second-order compensation is hand-checkable, plus a (2x4)@(4x3) layer to
compare RTN vs GPTQ reconstruction error (Eq.1) and show blocksize-invariance
(Eq.4-5 is an efficiency reformulation, not a different algorithm).
Emits explainer/traces/m3.json.
"""
import json
from pathlib import Path

import numpy as np

from gptq import (dampen, gptq_quantize, hessian_from_activations,
                  make_asymmetric_per_row_quantizer, reconstruction_error)

np.set_printoptions(suppress=True)

# ---------- Part A: per-column compensation on a single 1x4 row ----------
# 3-bit grid so quantization error (and thus compensation) is clearly visible.
W_row = np.array([[0.10, 0.95, -0.40, 0.55]])          # (1,4)
X = np.array([[1.0, 0.5, 0.2],
              [0.4, 1.0, 0.3],
              [0.2, 0.3, 1.0],
              [0.6, 0.2, 0.4]])                          # (4,3) = (d_col, T)
n_bits = 3
quant_fn = make_asymmetric_per_row_quantizer(W_row, n_bits=n_bits)
H = hessian_from_activations(X)
Hd = dampen(H, 0.01)
Hinv = np.linalg.inv(Hd)
Hinv_chol = np.linalg.cholesky(Hinv).T

# Replicate gptq_quantize's inner loop (blocksize>=d_col => one block) with tracing.
d_col = W_row.shape[1]
Wb = W_row.copy()
rows = []
for j in range(d_col):
    w_col = Wb[0, j]
    d = Hinv_chol[j, j]
    q_col = float(quant_fn(np.array([w_col]))[0])
    err = (w_col - q_col) / d
    # remaining columns in this row shift by err * Hinv_chol[j, j:]
    before_rest = Wb[0, j + 1:].copy()
    Wb[0, j:] -= err * Hinv_chol[j, j:]
    after_rest = Wb[0, j + 1:].copy()
    rows.append({
        "col": j, "w_before": round(w_col, 4), "q": round(q_col, 4),
        "raw_err": round(w_col - q_col, 4), "err_scaled": round(err, 4),
        "rest_before": [round(float(v), 4) for v in before_rest],
        "rest_after": [round(float(v), 4) for v in after_rest],
    })

Q_traced = gptq_quantize(W_row, H.copy(), quant_fn, blocksize=d_col)

# ---------- Part B: RTN vs GPTQ reconstruction error + blocksize invariance ----------
rng = np.random.default_rng(0)
W = np.array([[0.10, 0.95, -0.40, 0.55],
              [-0.70, 0.30, 0.80, -0.20]])               # (2,4)
q2 = make_asymmetric_per_row_quantizer(W, n_bits=n_bits)
H2 = hessian_from_activations(X)

W_rtn = q2(W)                                             # plain round-to-nearest, no compensation
err_rtn = reconstruction_error(W, W_rtn, X)

bs_table = []
for bs in (1, 2, 4):
    Q = gptq_quantize(W, H2.copy(), q2, blocksize=bs)
    e = reconstruction_error(W, Q, X)
    bs_table.append([bs, round(e, 4)])

Q_full = gptq_quantize(W, H2.copy(), q2, blocksize=4)
err_gptq = reconstruction_error(W, Q_full, X)

out = {
    "params": {"n_bits": n_bits, "W_row": [round(float(v), 4) for v in W_row[0]],
               "d_col": d_col},
    "hinv_chol_diag": [round(float(Hinv_chol[i, i]), 4) for i in range(d_col)],
    "per_column_rows": rows,
    "rtn_error": round(err_rtn, 4),
    "gptq_error": round(err_gptq, 4),
    "blocksize_invariance": bs_table,
    "improvement_pct": round(100 * (err_rtn - err_gptq) / err_rtn, 2),
}
Path(__file__).with_name("m3.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
