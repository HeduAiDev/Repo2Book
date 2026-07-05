"""Driver: uniform-quantization basics — feeds M1 (motivation/risk),
M2 (scale/zero-point, per-tensor/token/channel), M3 (hardware granularity).

Runs the paper-faithful reference implementation (implementation/uniform_quant.py)
on tiny, hand-checkable inputs and dumps every number the chapter cites to
m1_m2_m3.json. Numbers are rounded to what the narrative/tables quote so the
explainer linter can find each cited value verbatim in the trace.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "implementation"))
from uniform_quant import (  # noqa: E402
    smoothquant_scale, awq_scale, quantize_smoothquant, dequantize, quantize_awq,
    per_tensor_scale, per_token_scale, per_channel_scale, effective_quant_levels,
)
from gptq import make_asymmetric_per_row_quantizer  # noqa: E402


def r(x, n=4):
    if isinstance(x, np.ndarray):
        return [r(v, n) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [r(v, n) for v in x]
    return round(float(x), n)


out = {}

# --- M2: symmetric absmax quantize/dequantize on a tiny vector (n_bits=8) ---
# absmax = 1.27 -> SmoothQuant Delta = 1.27/127 = 0.01 (hand-checkable), but the
# other entries deliberately do NOT sit on the grid, so the round-trip error is
# genuinely non-zero (not the degenerate "lands exactly on a code" case).
w = np.array([1.27, -0.633, 0.307, -0.951])
n_bits = 8
delta_sq = smoothquant_scale(w, n_bits)
codes_sq, _ = quantize_smoothquant(w, n_bits)
deq_sq = dequantize(codes_sq, delta_sq)
err_sq = np.abs(deq_sq - w)
out["M2_vector"] = {
    "input_w": r(w), "n_bits": n_bits,
    "absmax": r(np.max(np.abs(w))),
    "smoothquant_qmax": 2 ** (n_bits - 1) - 1,          # 127
    "smoothquant_delta": r(delta_sq),                    # 0.01
    "w_over_delta": r(w / delta_sq),                     # [127,-63.3,30.7,-95.1]
    "codes": r(codes_sq),                                # [127,-63,31,-95]
    "dequant": r(deq_sq),                                # [1.27,-0.63,0.31,-0.95]
    "abs_error": r(err_sq),                              # [0,0.003,0.003,0.001]
    "max_abs_error": r(np.max(err_sq)),                  # 0.003
}

# --- M2: zero-point (asymmetric) — why it matters for a non-zero-centered vector.
# All-positive activations: a *symmetric* grid wastes the whole negative half;
# an *asymmetric* grid with a zero-point uses every code. n_bits=3 (8 levels) so
# the round-trip error is chunky and hand-checkable.
a = np.array([0.10, 0.55, 0.90, 0.32])
nb = 3
# symmetric (SmoothQuant-style) round-trip
c_sym, d_sym = quantize_smoothquant(a, nb)
deq_sym = dequantize(c_sym, d_sym)
# asymmetric per-row min-max grid with an explicit integer zero-point
qfn = make_asymmetric_per_row_quantizer(a.reshape(1, -1), nb)
deq_asym = qfn(a.reshape(1, -1))[0]
minv, maxv = float(np.min(a)), float(np.max(a))
qmax_asym = 2 ** nb - 1
scale_asym = (maxv - minv) / qmax_asym
zero_asym = round(-minv / scale_asym)
out["M2_zero_point"] = {
    "input_a": r(a), "n_bits": nb,
    "sym_delta": r(d_sym),
    "sym_dequant": r(deq_sym),
    "sym_max_abs_error": r(np.max(np.abs(deq_sym - a))),
    "asym_min": r(minv), "asym_max": r(maxv),
    "asym_levels": qmax_asym + 1,                        # 8
    "asym_scale": r(scale_asym),
    "asym_zero_point": zero_asym,
    "asym_dequant": r(deq_asym),
    "asym_max_abs_error": r(np.max(np.abs(deq_asym - a))),
}

# AWQ convention (full symmetric range, divisor 2^(N-1))
delta_awq = awq_scale(w, n_bits)
out["M2_awq_convention"] = {
    "awq_qmax": 2 ** (n_bits - 1),                       # 128
    "awq_delta": r(delta_awq),                           # 1.27/128 = 0.0099
    "ratio_awq_over_sq": r(delta_awq / delta_sq),        # 127/128 = 0.9922
    "ratio_sq_over_awq": r(delta_sq / delta_awq),        # 128/127 = 1.0079
}

# quantize_awq round-trips the absmax element to +-qmax (=128*delta=absmax)
deq_awq, _ = quantize_awq(w, n_bits)
out["M2_awq_roundtrip"] = {
    "dequant": r(deq_awq),
    "max_abs_error": r(np.max(np.abs(deq_awq - w))),
}

# --- M3 + M1: granularity + outlier collapse on a 2-token x 3-channel matrix ---
# Channel 2 (0-indexed) is a ~50-80x activation outlier (fixed across tokens).
X = np.array([[0.10, 0.20, 10.00],
              [0.15, 0.10,  8.00]])
out["M3_granularity"] = {
    "X": r(X),
    "per_tensor_scale": r(per_tensor_scale(X, n_bits)),          # one scalar, absmax=10
    "per_token_scale": r(per_token_scale(X, n_bits)),            # one per row (token)
    "per_channel_scale_of_X": r(per_channel_scale(X, n_bits)),   # one per column (channel)
}

# M1: effective quantization levels under per-tensor quant (SmoothQuant §3).
lvls = effective_quant_levels(X, n_bits)
out["M1_effective_levels"] = {
    "matrix_absmax_m": r(np.max(np.abs(X))),                     # 10.0
    "channel_absmax_m_i": r(np.max(np.abs(X), axis=0)),          # [0.15,0.20,10.0]
    "full_levels_2^8": 2 ** n_bits,                              # 256
    "effective_levels": r(lvls),                                 # [3.84,5.12,256.0]
    "outlier_channel_levels": r(lvls[2]),                        # 256.0
    "worst_nonoutlier_levels": r(np.min(lvls)),                  # 3.84
}

# A more extreme ~100x outlier to reproduce the paper's "2-3 levels" claim.
X2 = np.array([[0.05, 0.08, 10.00],
               [0.06, 0.05,  9.00]])
lvls2 = effective_quant_levels(X2, n_bits)
out["M1_extreme_outlier"] = {
    "outlier_ratio": r(np.max(np.abs(X2)) / np.max(np.abs(X2[:, :2]))),  # 10/0.08=125x
    "effective_levels": r(lvls2),
    "worst_nonoutlier_levels": r(np.min(lvls2)),                # ~1.28 -> "2-3 levels" regime
}

print(json.dumps(out, indent=2))
