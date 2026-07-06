"""Driver for m4-awq-activation-aware-scaling.

Uses implementation/awq.py (AWQ arXiv:2306.00978 §3.2 Eq.1/2/5):
  (A) search_alpha grid: L(s_X^alpha) over alpha in {0,.25,.5,.75,1} on a toy
      linear layer with one high-activation ("salient") input channel.
  (B) scaled_error_ratio: scaling a salient weight by s reduces its relative
      quantization error ~ 1/s (Eq.2), until s grows the group absmax (Delta'>Delta).
Emits explainer/traces/m4.json.
"""
import json
from pathlib import Path

import numpy as np

from awq import round_err, scaled_error_ratio, search_alpha

np.set_printoptions(suppress=True)
rng = np.random.default_rng(7)

# ---- toy layer: Ci=4 input channels, one of them (channel 0) is high-magnitude
# activation ("salient" -> its weight column matters most for the output) ----
T, Ci, Co = 8, 4, 3
X = rng.normal(0, 1.0, size=(T, Ci))
X[:, 0] *= 12.0                                  # channel 0 = salient (big activation)
W = rng.normal(0, 0.1, size=(Ci, Co))
s_x = np.mean(np.abs(X), axis=0)                 # per-input-channel avg |activation| (Eq.5 s_X)

alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
best_alpha, losses = search_alpha(X, W, s_x, n_bits=4, alphas=alphas)
alpha_rows = [[a, round(l, 4)] for a, l in zip(alphas, losses)]
loss_a0 = losses[0]
loss_best = min(losses)
improvement_pct = round(100 * (loss_a0 - loss_best) / loss_a0, 2)

# ---- Eq.2 error ratio: scale one salient weight by s, average over trials
# (RoundErr~0.25 is an expected-value claim -> average many random groups) ----
def avg_ratio(s, n_trials=4000, group_size=8, n_bits=4):
    ratios, deltas = [], []
    g = np.random.default_rng(123)
    for _ in range(n_trials):
        group = g.normal(0, 1.0, size=group_size)
        _, actual, dratio = scaled_error_ratio(group, salient_idx=0, s=s, n_bits=n_bits)
        ratios.append(actual)
        deltas.append(dratio)
    return round(float(np.mean(ratios)), 4), round(float(np.mean(deltas)), 4)

s_rows = []
for s in (1.0, 2.0, 4.0, 8.0):
    measured, dratio = avg_ratio(s)
    naive = round(1.0 / s, 4)
    s_rows.append([s, naive, measured, dratio])

# RoundErr ~ 0.25 sanity
re = round(round_err(rng.uniform(-100, 100, size=200000)), 4)

out = {
    "params": {"T": T, "Ci": Ci, "Co": Co, "n_bits": 4,
               "salient_channel": 0, "salient_activation_mult": 12.0,
               "s_x_avg_abs": [round(float(v), 4) for v in s_x]},
    "alpha_grid": alpha_rows,          # [alpha, L(s_X^alpha)]
    "best_alpha": best_alpha,
    "loss_alpha0_noscale": round(loss_a0, 4),
    "loss_best": round(loss_best, 4),
    "improvement_pct": improvement_pct,
    "round_err_mean": re,              # ~0.25
    "scale_ratio_rows": s_rows,        # [s, naive 1/s, measured ratio, Delta'/Delta]
}
Path(__file__).with_name("m4.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
