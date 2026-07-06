"""Driver for m6-dequant-error-worked-example.

Same toy outlier layer, three error-control philosophies each measured
against its OWN matched round-to-nearest (RTN) baseline, by the layer output
reconstruction error ||Y_hat - Y|| (Y = X @ W), grounding the dequantize
round-trip quant_utils.py:L614-L621 (int) / L322-L338 (fp8 amax):

  W8A8 regime  (activations + weights 8-bit per-tensor):
     RTN          -> no mitigation
     SmoothQuant  -> difficulty migrated offline, then per-tensor 8-bit
  W4 weight-only regime (weights 4-bit, activations full precision):
     RTN          -> plain absmax 4-bit
     AWQ          -> activation-aware weight scaling, then 4-bit
Each mitigation beats its matched RTN baseline on the same hard layer.
Emits explainer/traces/m6.json.
"""
import json
from pathlib import Path

import numpy as np

from awq import quantize_dequantize, search_alpha
from smoothquant import smoothquant_pipeline_error
from uniform_quant import quantize_smoothquant

np.set_printoptions(suppress=True)
rng = np.random.default_rng(11)

T, Ci, Co = 6, 4, 3
X = rng.normal(0, 1.0, size=(T, Ci))
X[:, 2] *= 50.0                                    # activation outlier channel
W = rng.normal(0, 0.5, size=(Ci, Co))
Y = X @ W


def rel(err):
    return round(100 * err / float(np.linalg.norm(Y)), 2)


# ---- W8A8 regime: RTN baseline vs SmoothQuant (alpha=0.5) ----
err_rtn_w8a8, err_sq_w8a8 = smoothquant_pipeline_error(X, W, alpha=0.5, n_bits=8)

# ---- W4 weight-only regime: RTN baseline vs AWQ ----
Wq_rtn = quantize_dequantize(W, n_bits=4)          # plain absmax 4-bit, X kept FP
err_rtn_w4 = float(np.linalg.norm(X @ Wq_rtn - Y))

s_x = np.mean(np.abs(X), axis=0)
best_alpha, losses = search_alpha(X, W, s_x, n_bits=4,
                                  alphas=(0.0, 0.25, 0.5, 0.75, 1.0))
s = s_x ** best_alpha
W_scaled = W * s[:, np.newaxis]
X_scaled = X / s[np.newaxis, :]
Wq_awq = quantize_dequantize(W_scaled, n_bits=4)
err_awq_w4 = float(np.linalg.norm(X_scaled @ Wq_awq - Y))

rows = [
    ["RTN", "W8A8", round(err_rtn_w8a8, 4), rel(err_rtn_w8a8), "no mitigation"],
    ["SmoothQuant", "W8A8", round(err_sq_w8a8, 4), rel(err_sq_w8a8), "difficulty migrated"],
    ["RTN", "W4-weight-only", round(err_rtn_w4, 4), rel(err_rtn_w4), "no mitigation"],
    ["AWQ", "W4-weight-only", round(err_awq_w4, 4), rel(err_awq_w4), "act-aware scale"],
]

out = {
    "params": {"T": T, "Ci": Ci, "Co": Co, "outlier_channel": 2,
               "outlier_mult": 50.0, "best_alpha_awq": best_alpha,
               "w8a8_weight_bits": 8, "w8a8_act_bits": 8, "w4_weight_bits": 4,
               "Y_norm": round(float(np.linalg.norm(Y)), 4)},
    "rows": rows,   # [method, regime, abs_err, rel_err_pct, note]
    "w8a8_reduction_pct": round(100 * (err_rtn_w8a8 - err_sq_w8a8) / err_rtn_w8a8, 2),
    "w4_reduction_pct": round(100 * (err_rtn_w4 - err_awq_w4) / err_rtn_w4, 2),
}
Path(__file__).with_name("m6.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
