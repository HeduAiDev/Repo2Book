"""Driver: AWQ activation-aware scaling — feeds M6.

AWQ §3.2 Eq.2-3: scaling a salient weight by s>1 (and dividing the matching
activation by s) leaves the output value unchanged but cuts the salient
weight's *relative* quantization error to ~ Delta'/Delta * 1/s. Since RoundErr
is an *expected*-value quantity (~0.25), the 1/s law holds in aggregate, not
per single weight — so this driver averages the error over many random salient
weights (ratio of mean errors), the way the paper's claim is meant. It then
shows the "too-large s" failure (s shifts the group absmax -> Delta'>Delta ->
non-salient channels pay) and runs the Eq.4-5 alpha grid search.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "implementation"))
from uniform_quant import awq_scale, quantize_awq  # noqa: E402
from awq import quantize_dequantize, search_alpha  # noqa: E402


def r(x, n=4):
    if isinstance(x, np.ndarray):
        return [r(v, n) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [r(v, n) for v in x]
    return round(float(x), n)


out = {}
n_bits = 4  # 4-bit weights (W4A16 landing regime)

# --- Aggregated 1/s law: group has a large fixed max (3.0). The salient weight
# is *small in value* (AWQ: salience is set by ACTIVATION magnitude, not weight
# magnitude), drawn randomly; scaling it by s keeps it below the group max, so
# Delta is unchanged and the mean error ratio tracks 1/s. ---
rng = np.random.default_rng(0)
N = 20000
group_max = 3.0
delta = awq_scale(np.array([group_max]), n_bits)   # 3.0/8 = 0.375
salient_law = []
for s in (1.0, 2.0, 4.0):
    orig_errs, scaled_errs, shifted = [], [], 0
    for _ in range(N):
        w0 = rng.uniform(-0.5, 0.5)                 # small salient weight
        grp = np.array([group_max, w0])
        oq = quantize_dequantize(grp, n_bits)[1]
        orig_errs.append(abs(oq - w0))
        grp_s = np.array([group_max, w0 * s])
        if abs(w0 * s) > group_max:
            shifted += 1
        sq = quantize_dequantize(grp_s, n_bits)[1] / s
        scaled_errs.append(abs(sq - w0))
    ratio = float(np.mean(scaled_errs)) / float(np.mean(orig_errs))
    salient_law.append({
        "s": s, "naive_1_over_s": r(1.0 / s),
        "mean_orig_err": r(float(np.mean(orig_errs))),
        "mean_scaled_err": r(float(np.mean(scaled_errs))),
        "measured_ratio": r(ratio),
        "frac_absmax_shifted": r(shifted / N),         # ~0 -> Delta unchanged
    })
out["awq_1_over_s_law"] = {"n_trials": N, "group_max": group_max,
                           "delta": r(delta), "rows": salient_law}

# --- Deterministic "too-large s" failure: now the salient weight IS the group
# max-setter after scaling, so Delta grows and every code in the group coarsens. ---
grp = np.array([1.0, 0.6, -0.4, 0.5])   # salient idx 0 = 1.0 (already the max)
delta_base = awq_scale(grp, n_bits)
too_large = []
for s in (2.0, 4.0, 8.0):
    grp_s = grp.copy(); grp_s[0] = grp[0] * s
    delta_s = awq_scale(grp_s, n_bits)
    too_large.append({"s": s, "new_absmax": r(np.max(np.abs(grp_s))),
                      "delta_prime": r(delta_s), "delta_ratio": r(delta_s / delta_base)})
out["awq_too_large_s"] = {"group": r(grp), "delta_base": r(delta_base), "rows": too_large}

# --- Eq.4-5 alpha grid search on a toy layer with an activation-salient channel.
# Ci=3 input channels, channel 0 has ~10x larger activation magnitude. ---
rng2 = np.random.default_rng(3)
T, Ci, Co = 6, 3, 4
X = rng2.normal(scale=0.3, size=(T, Ci))
X[:, 0] *= 10.0                                     # activation-salient channel
W = rng2.normal(scale=0.5, size=(Ci, Co))
s_x = np.mean(np.abs(X), axis=0)                    # AWQ s_X = mean |activation| per channel
alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
best_alpha, losses = search_alpha(X, W, s_x, n_bits, alphas)
out["awq_alpha_search"] = {
    "s_x_per_channel": r(s_x),
    "alphas": list(alphas),
    "losses": r(losses),
    "best_alpha": best_alpha,
    "loss_at_alpha0_no_scaling": r(losses[0]),
    "loss_at_best": r(min(losses)),
    "improvement_vs_no_scaling": r(losses[0] / min(losses)),
}

print(json.dumps(out, indent=2))
