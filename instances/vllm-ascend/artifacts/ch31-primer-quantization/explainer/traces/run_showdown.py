"""Driver: four-method numerical showdown — feeds M8.

Each advanced method is compared against ITS OWN naive baseline, in ITS OWN
regime (so every row is internally apples-to-apples; the methods target
different bit-widths and are not competing head-to-head):
  - GPTQ  vs plain RTN, W4 weight-only, SAME fixed per-row grid -> isolates
    the second-order compensation effect.
  - AWQ   vs no-scaling (alpha=0), W4 weight-only, SAME absmax quantizer ->
    isolates the activation-aware scaling effect.
  - SmoothQuant vs raw (un-migrated), W8A8 per-tensor -> isolates migration.
All from implementation/. Every number lands in m8_showdown.json.
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
from awq import search_alpha  # noqa: E402
from smoothquant import smoothquant_pipeline_error  # noqa: E402


def r(x, n=4):
    return round(float(x), n)


out = {}

# --- GPTQ vs RTN (W4 weight-only, shared per-row grid) — same layer as M5. ---
rng = np.random.default_rng(7)
d_row, d_col, T = 3, 6, 8
shared = rng.normal(size=(1, T))
Xg = shared + 0.4 * rng.normal(size=(d_col, T)) + rng.uniform(0.5, 1.0, (d_col, 1))
Wg = rng.uniform(-1.0, 1.0, (d_row, d_col))
Hg = hessian_from_activations(Xg)
qfn = make_asymmetric_per_row_quantizer(Wg, 3)
# report as plain L2 norm (reconstruction_error returns the SQUARED norm)
err_rtn_g = np.sqrt(reconstruction_error(Wg, qfn(Wg), Xg))
err_gptq = np.sqrt(reconstruction_error(Wg, gptq_quantize(Wg, Hg, qfn), Xg))
out["gptq"] = {"regime": "W4 weight-only", "n_bits": 3,
               "baseline_rtn_err": r(err_rtn_g), "method_err": r(err_gptq),
               "reduction_x": r(err_rtn_g / err_gptq)}

# --- AWQ scaled vs no-scaling (W4 weight-only) — same layer as M6 alpha search. ---
rng2 = np.random.default_rng(3)
Ta, Cia, Coa = 6, 3, 4
Xa = rng2.normal(scale=0.3, size=(Ta, Cia))
Xa[:, 0] *= 10.0
Wa = rng2.normal(scale=0.5, size=(Cia, Coa))
s_x = np.mean(np.abs(Xa), axis=0)
best_alpha, losses = search_alpha(Xa, Wa, s_x, 4, (0.0, 0.25, 0.5, 0.75, 1.0))
out["awq"] = {"regime": "W4 weight-only", "n_bits": 4, "best_alpha": best_alpha,
              "baseline_noscale_err": r(losses[0]), "method_err": r(min(losses)),
              "reduction_x": r(losses[0] / min(losses))}

# --- SmoothQuant migrated vs raw (W8A8 per-tensor) — same layer as M7. ---
rng3 = np.random.default_rng(7)
Ts, Cis, Cos = 16, 4, 4
Xs = np.abs(rng3.normal(scale=0.15, size=(Ts, Cis)))
Xs[:, 0] = rng3.uniform(6.0, 9.0, Ts)
Ws = rng3.normal(scale=0.35, size=(Cis, Cos))
err_raw, err_smoothed = smoothquant_pipeline_error(Xs, Ws, 0.5, 8)
out["smoothquant"] = {"regime": "W8A8 per-tensor", "n_bits": 8, "alpha": 0.5,
                      "baseline_raw_err": r(err_raw), "method_err": r(err_smoothed),
                      "reduction_x": r(err_raw / err_smoothed)}

# consolidated table view
out["table"] = [
    {"method": "GPTQ", "baseline_err": out["gptq"]["baseline_rtn_err"],
     "method_err": out["gptq"]["method_err"], "reduction_x": out["gptq"]["reduction_x"]},
    {"method": "AWQ", "baseline_err": out["awq"]["baseline_noscale_err"],
     "method_err": out["awq"]["method_err"], "reduction_x": out["awq"]["reduction_x"]},
    {"method": "SmoothQuant", "baseline_err": out["smoothquant"]["baseline_raw_err"],
     "method_err": out["smoothquant"]["method_err"], "reduction_x": out["smoothquant"]["reduction_x"]},
]
print(json.dumps(out, indent=2))
