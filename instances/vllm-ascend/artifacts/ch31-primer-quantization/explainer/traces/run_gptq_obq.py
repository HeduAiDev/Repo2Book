"""Driver: GPTQ / OBQ second-order compensation — feeds M4.

Walks the OBQ greedy pick+compensate step (GPTQ Eq.2) on a hand-sized
3-weight problem, one round at a time, then compares the full layer-output
reconstruction error (Eq.1) of OBQ-compensated quantization vs plain RTN
(no compensation). All from implementation/gptq.py.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "implementation"))
from gptq import (  # noqa: E402
    hessian_from_activations, make_asymmetric_per_row_quantizer,
    obq_pick_and_compensate, remove_hessian_row_col, reconstruction_error,
)


def r(x, n=4):
    if isinstance(x, np.ndarray):
        return [r(v, n) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [r(v, n) for v in x]
    return round(float(x), n)


out = {}

# One weight row of 3 entries; a T=4-sample calibration activation X whose
# channels are strongly correlated (one dominant sample) gives a Hessian with
# large off-diagonals, so the second-order compensation genuinely flips a
# rounding decision (verified by search_obq.py: OBQ beats RTN ~3x here, and the
# final codes differ from RTN — not the degenerate "compensation changes
# nothing" tie).
w_row = np.array([-0.60, -0.812, 0.192])
X = np.array([[-0.177, 0.136, 0.051, 1.171],
              [0.138, 0.025, 0.319, 1.341],
              [0.313, 0.221, 0.309, 1.628]])
H = hessian_from_activations(X)
Hinv0 = np.linalg.inv(H)
n_bits = 2  # coarse 2-bit grid (4 levels) so quantization error is large & visible
quant_fn = make_asymmetric_per_row_quantizer(w_row.reshape(1, -1), n_bits)

out["setup"] = {
    "w_row": r(w_row), "n_bits": n_bits,
    "H": r(H), "Hinv_diag": r(np.diag(Hinv0)),
    "rtn_quant_of_w": r(quant_fn(w_row)),  # what plain rounding would produce
}

# --- Round-by-round OBQ (greedy pick + compensate + Hessian shrink) ---
rounds = []
Hinv = Hinv0.copy()
w = w_row.copy()
idx_map = [0, 1, 2]          # original indices still full-precision
final = np.zeros(3)
for rnd in range(1, 4):
    quantized = quant_fn(w)
    diag = np.diag(Hinv)
    costs = (quantized - w) ** 2 / diag
    q_local, delta_F, w_after = obq_pick_and_compensate(Hinv, w, quant_fn)
    picked_orig = idx_map[q_local]
    rounds.append({
        "round": rnd,
        "remaining_orig_idx": list(idx_map),
        "w_remaining": r(w),
        "cost_per_candidate": r(costs, 6),
        "picked_local": q_local,
        "picked_orig_idx": picked_orig,
        "quant_value": r(quantized[q_local]),
        "quant_error": r(float(w[q_local] - quantized[q_local])),
        "delta_F_to_others": r(delta_F),
        "w_after_compensation": r(w_after),
    })
    final[picked_orig] = w_after[q_local]
    # shrink
    w = np.delete(w_after, q_local)
    Hinv = remove_hessian_row_col(Hinv, q_local)
    del idx_map[q_local]
out["obq_rounds"] = rounds
out["obq_final_wq"] = r(final)

# --- Eq.1 objective: OBQ compensation vs plain RTN, full layer output ---
W = w_row.reshape(1, -1)                 # 1x3 layer
W_rtn = quant_fn(w_row).reshape(1, -1)   # plain rounding, no compensation
W_obq = final.reshape(1, -1)             # OBQ-compensated codes
err_rtn = reconstruction_error(W, W_rtn, X)
err_obq = reconstruction_error(W, W_obq, X)
out["reconstruction_eq1"] = {
    "rtn_output_error": r(err_rtn, 5),
    "obq_output_error": r(err_obq, 5),
    "improvement_ratio": r(err_rtn / err_obq if err_obq > 0 else 0.0),
}

print(json.dumps(out, indent=2))
