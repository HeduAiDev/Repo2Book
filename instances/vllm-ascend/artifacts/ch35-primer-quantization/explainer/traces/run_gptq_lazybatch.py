"""Driver: GPTQ Algorithm 1 lazy-batch block-size invariance — feeds M5.

GPTQ §4's lazy batch-update (Eq.4-5) is an *efficiency reformulation* of the
per-column OBQ process, not a different algorithm: quantizing in blocks of
`blocksize` columns must give (essentially) the same result as blocksize=1.
This driver runs implementation/gptq.py's gptq_quantize at several block sizes
on one small (d_row x d_col) layer and shows the Eq.1 output error is
invariant to blocksize and never worse than plain RTN.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "implementation"))
from gptq import (  # noqa: E402
    hessian_from_activations, make_asymmetric_per_row_quantizer,
    gptq_quantize, reconstruction_error,
)


def r(x, n=5):
    return round(float(x), n)


# Fixed, correlated calibration data so the Hessian has real off-diagonal
# structure (compensation matters). d_row=3 output rows, d_col=6 weights,
# T=8 calibration samples. Seed pinned for reproducibility.
rng = np.random.default_rng(7)
d_row, d_col, T = 3, 6, 8
shared = rng.normal(size=(1, T))
X = shared + 0.4 * rng.normal(size=(d_col, T)) + rng.uniform(0.5, 1.0, (d_col, 1))
W = rng.uniform(-1.0, 1.0, (d_row, d_col))
H = hessian_from_activations(X)
n_bits = 3
quant_fn = make_asymmetric_per_row_quantizer(W, n_bits)

# RTN baseline: plain rounding on the fixed grid, no compensation.
W_rtn = quant_fn(W)
err_rtn = reconstruction_error(W, W_rtn, X)

rows = []
for bs in (1, 2, 3, 6):
    Q = gptq_quantize(W, H, quant_fn, blocksize=bs)
    err = reconstruction_error(W, Q, X)
    rows.append({"blocksize": bs, "gptq_output_error": r(err),
                 "vs_rtn_ratio": r(err_rtn / err if err > 0 else 0.0, 3)})

out = {
    "shape": {"d_row": d_row, "d_col": d_col, "T": T, "n_bits": n_bits},
    "rtn_output_error": r(err_rtn),
    "blocksize_sweep": rows,
    "max_error_spread_across_blocksizes": r(
        max(x["gptq_output_error"] for x in rows) - min(x["gptq_output_error"] for x in rows), 6),
    "note": "省访存(把逐列更新批量化)不损精度：误差随 blocksize 基本不变，且都优于 RTN。",
}
print(json.dumps(out, indent=2))
