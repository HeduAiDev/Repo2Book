"""Driver for m2-quant-granularity (figure-only).

effective_quant_levels (implementation/uniform_quant.py, SmoothQuant §3):
under a single per-tensor scale, channel i keeps 2^N * m_i/m effective
levels, where m_i is its own absmax and m the whole-matrix absmax. An
outlier channel keeps ~full 256 levels; a non-outlier channel collapses to
a handful -> why per-tensor activation quantization is risky and finer
granularity (per-token / per-channel) is needed.
Emits explainer/traces/m2.json.
"""
import json
from pathlib import Path

import numpy as np

from uniform_quant import effective_quant_levels

np.set_printoptions(suppress=True)
rng = np.random.default_rng(5)

T, Ci = 8, 4
X = rng.normal(0, 1.0, size=(T, Ci))
X[:, 1] *= 100.0                        # channel 1 is a ~100x activation outlier
n_bits = 8

levels = effective_quant_levels(X, n_bits=n_bits)
chan_absmax = np.max(np.abs(X), axis=0)
m = float(np.max(np.abs(X)))

rows = [[j, round(float(chan_absmax[j]), 4), round(float(levels[j]), 2)]
        for j in range(Ci)]

out = {
    "params": {"T": T, "Ci": Ci, "n_bits": n_bits, "full_levels": 2 ** n_bits,
               "outlier_channel": 1, "outlier_mult": 100.0,
               "tensor_absmax_m": round(m, 4)},
    "rows": rows,                        # [channel, absmax m_i, effective levels]
    "outlier_levels": round(float(levels[1]), 2),
    "min_nonoutlier_levels": round(float(np.min(levels)), 2),
}
Path(__file__).with_name("m2.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
