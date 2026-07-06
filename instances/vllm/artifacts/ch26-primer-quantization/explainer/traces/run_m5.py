"""Driver for m5-smoothquant-migration.

Uses implementation/smoothquant.py (SmoothQuant arXiv:2211.10438 §4 Eq.3/4)
on a toy layer with one activation-outlier input channel:
  (A) difficulty_factor s_j = max|X_j|^a / max|W_j|^(1-a), and migrate()
      shows per-channel absmax of X (falls) and W (rises) after X/s, W*s.
  (B) smoothquant_pipeline_error: per-tensor W8A8 output error, raw vs migrated,
      swept over alpha -> the "difficulty migration reduces error" claim.
Emits explainer/traces/m5.json.
"""
import json
from pathlib import Path

import numpy as np

from smoothquant import (difficulty_factor, migrate, smoothquant_pipeline_error)

np.set_printoptions(suppress=True)
rng = np.random.default_rng(3)

# ---- toy layer: Ci=4 input channels; channel 2 is an activation outlier ----
T, Ci, Co = 6, 4, 3
X = rng.normal(0, 1.0, size=(T, Ci))
X[:, 2] *= 60.0                                    # outlier activation channel
W = rng.normal(0, 0.5, size=(Ci, Co))
n_bits = 8

alpha = 0.5
s = difficulty_factor(X, W, alpha=alpha)
X_hat, W_hat = migrate(X, W, s)

mx_before = np.max(np.abs(X), axis=0)
mx_after = np.max(np.abs(X_hat), axis=0)
mw_before = np.max(np.abs(W), axis=1)
mw_after = np.max(np.abs(W_hat), axis=1)

# per-channel migration table: [ch, s_j, max|X| before->after, max|W| before->after]
chan_rows = [[j, round(float(s[j]), 4),
              round(float(mx_before[j]), 4), round(float(mx_after[j]), 4),
              round(float(mw_before[j]), 4), round(float(mw_after[j]), 4)]
             for j in range(Ci)]

# equivalence check: X_hat @ W_hat == X @ W
equiv_max_diff = round(float(np.max(np.abs(X_hat @ W_hat - X @ W))), 8)

# alpha sweep of per-tensor W8A8 output error, raw baseline vs migrated
alpha_rows = []
err_raw_ref = None
for a in (0.0, 0.25, 0.5, 0.75, 1.0):
    err_raw, err_smoothed = smoothquant_pipeline_error(X, W, alpha=a, n_bits=n_bits)
    err_raw_ref = round(err_raw, 4)
    alpha_rows.append([a, round(err_smoothed, 4)])

best = min(alpha_rows, key=lambda r: r[1])
improvement_pct = round(100 * (err_raw_ref - best[1]) / err_raw_ref, 2)

out = {
    "params": {"T": T, "Ci": Ci, "Co": Co, "n_bits": n_bits,
               "outlier_channel": 2, "outlier_mult": 60.0, "alpha": alpha},
    "s_factor": [round(float(v), 4) for v in s],
    "migration_rows": chan_rows,     # [ch, s_j, |X|max before, after, |W|max before, after]
    "equiv_max_diff": equiv_max_diff,        # ~0 : Eq.3 is an exact identity
    "err_raw_pertensor": err_raw_ref,
    "alpha_sweep": alpha_rows,       # [alpha, migrated per-tensor W8A8 error]
    "best_alpha": best[0],
    "best_err": best[1],
    "improvement_pct": improvement_pct,
}
Path(__file__).with_name("m5.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
