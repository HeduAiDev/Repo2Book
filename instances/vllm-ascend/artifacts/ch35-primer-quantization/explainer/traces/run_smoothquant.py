"""Driver: SmoothQuant migration difficulty factor s — feeds M7.

SmoothQuant §4 Eq.3 is an algebraic identity: Y = (X diag(s)^-1)(diag(s) W)
for ANY positive s. Eq.4's s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha) is the
*choice* of s that splits quantization difficulty between activation and weight.
This driver builds a toy layer with one ~80x activation-outlier input channel,
shows migration flattens the activation channel-max spread onto the weights,
and sweeps alpha (0 -> all difficulty on activation, 1 -> all on weight) to find
the per-tensor-quant error minimum near alpha=0.5. From implementation/.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "implementation"))
from smoothquant import difficulty_factor, migrate, smoothquant_pipeline_error  # noqa: E402


def r(x, n=4):
    if isinstance(x, np.ndarray):
        return [r(v, n) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [r(v, n) for v in x]
    return round(float(x), n)


out = {}
# Toy linear layer: T=16 tokens, Ci=4 input channels, Co=4 output channels
# (seed pinned). Input channel 0 is a systematic ~40-60x activation outlier
# (the SmoothQuant problem case); weights are well-behaved, unit-scale. Enough
# elements that the per-tensor-quant error landscape is smooth (a clean U in
# alpha), unlike a 3x3 toy where discreteness makes it jagged.
rng = np.random.default_rng(7)
T, Ci, Co = 16, 4, 4
X = np.abs(rng.normal(scale=0.15, size=(T, Ci)))
X[:, 0] = rng.uniform(6.0, 9.0, T)     # activation-outlier channel
W = rng.normal(scale=0.35, size=(Ci, Co))

mx = np.max(np.abs(X), axis=0)       # per input-channel activation max
mw = np.max(np.abs(W), axis=1)       # per input-channel weight max
out["layer"] = {
    "shape": {"T": T, "Ci": Ci, "Co": Co},
    "X_channel_absmax": r(mx),
    "W_channel_absmax": r(mw),
    "activation_outlier_ratio": r(np.max(mx) / np.min(mx)),
    # Eq.3 is an exact algebraic identity for the alpha=0.5 s (checked below).
}
# --- Eq.3 identity check: X_hat @ W_hat == X @ W (migration is loss-free) ---
_s = difficulty_factor(X, W, 0.5)
_Xh, _Wh = migrate(X, W, _s)
out["eq3_identity_residual"] = r(float(np.linalg.norm(_Xh @ _Wh - X @ W)), 8)

# difficulty factor s and migration effect at alpha=0.5
s_half = difficulty_factor(X, W, alpha=0.5)
X_hat, W_hat = migrate(X, W, s_half)
out["migration_alpha0.5"] = {
    "s_per_channel": r(s_half),
    "X_hat_channel_absmax": r(np.max(np.abs(X_hat), axis=0)),   # flattened
    "W_hat_channel_absmax": r(np.max(np.abs(W_hat), axis=1)),   # raised a bit
    "X_spread_before": r(np.max(mx) / np.min(mx)),
    "X_spread_after": r(np.max(np.abs(X_hat)) / np.min(np.max(np.abs(X_hat), axis=0))),
}

# alpha sweep: difficulty factor extremes + per-tensor-quant pipeline error.
rows = []
for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
    s = difficulty_factor(X, W, alpha)
    err_raw, err_smoothed = smoothquant_pipeline_error(X, W, alpha)
    rows.append({
        "alpha": alpha,
        "s_channel0": r(s[0]),          # outlier channel's migration factor
        "err_raw": r(err_raw),          # same regardless of alpha (no migration)
        "err_smoothed": r(err_smoothed),
    })
out["alpha_sweep"] = rows
best = min(rows, key=lambda x: x["err_smoothed"])
out["best_alpha"] = best["alpha"]
out["err_raw_constant"] = rows[0]["err_raw"]
out["err_smoothed_at_best"] = best["err_smoothed"]
out["raw_over_best_ratio"] = r(rows[0]["err_raw"] / best["err_smoothed"])

print(json.dumps(out, indent=2))
