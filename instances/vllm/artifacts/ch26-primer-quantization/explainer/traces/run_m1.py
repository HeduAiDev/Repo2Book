"""Driver for m1-uniform-affine-quant.

Reproduces vllm/model_executor/layers/quantization/utils/quant_utils.py
L586-L621 quantize_weights on a tiny hand-checkable weight vector:
  asymmetric (zero_points=True):  w_s=(max-min)/qmax_u,  zp=round(|min|/w_s)
                                  w_q=clamp(round(w/w_s)+zp, 0, qmax_u)
                                  w_ref=(w_q-zp)*w_s
  symmetric  (zero_points=False): w_s=max(|max|/qmax_s, |min|/qmin_s), zp=0
Both at 4-bit so a reader can hand-verify each row.
Emits explainer/traces/m1.json (every number the m1 table cites lives here).
"""
import json
from pathlib import Path

import numpy as np

np.set_printoptions(suppress=True)

# tiny weight group, chosen so w/scale never lands on a .5 rounding tie
w = np.array([-1.0, -0.32, 0.24, 0.68, 1.36, 2.0])

n_bits = 4
wmin, wmax = float(w.min()), float(w.max())

# ---- asymmetric branch (unsigned 4-bit grid [0,15]), quant_utils L595-L608 ----
qmax_u = 2 ** n_bits - 1                        # 15
scale_asym = (wmax - wmin) / qmax_u             # (2.0 - (-1.0))/15 = 0.2
zp = int(np.round(abs(wmin) / scale_asym))      # round(1.0/0.2) = 5
codes_asym = np.clip(np.round(w / scale_asym) + zp, 0, qmax_u).astype(int)
w_ref_asym = (codes_asym - zp) * scale_asym     # dequantize, L619
err_asym = np.abs(w - w_ref_asym)

# ---- symmetric branch (signed 4-bit grid [-8,7]), quant_utils L602-L608 ----
qmax_s = 2 ** (n_bits - 1) - 1                   # 7
qmin_s = -(2 ** (n_bits - 1))                    # -8
scale_sym = max(abs(wmax) / qmax_s, abs(wmin) / abs(qmin_s))
codes_sym = np.clip(np.round(w / scale_sym), qmin_s, qmax_s).astype(int)
w_ref_sym = codes_sym * scale_sym
err_sym = np.abs(w - w_ref_sym)

r4 = lambda a: [round(float(v), 4) for v in np.atleast_1d(a)]

out = {
    "params": {"n_bits": n_bits, "qmax_unsigned": qmax_u,
               "qmax_signed": qmax_s, "qmin_signed": qmin_s,
               "w": r4(w), "wmin": round(wmin, 4), "wmax": round(wmax, 4)},
    "asymmetric": {
        "scale": round(scale_asym, 4), "zero_point": zp,
        "codes": [int(c) for c in codes_asym],
        "dequant": r4(w_ref_asym), "abs_err": r4(err_asym),
        "max_abs_err": round(float(err_asym.max()), 4),
        "half_scale": round(scale_asym / 2, 4),
    },
    "symmetric": {
        "scale": round(scale_sym, 4), "zero_point": 0,
        "codes": [int(c) for c in codes_sym],
        "dequant": r4(w_ref_sym), "abs_err": r4(err_sym),
        "max_abs_err": round(float(err_sym.max()), 4),
        "half_scale": round(scale_sym / 2, 4),
    },
    # per-element asymmetric table rows: [w, w/scale, round, +zp->code(clamped), dequant, |err|]
    "asym_rows": [
        [round(float(wi), 4), round(float(wi / scale_asym), 4),
         int(np.round(wi / scale_asym)), int(ci),
         round(float(di), 4), round(float(ei), 4)]
        for wi, ci, di, ei in zip(w, codes_asym, w_ref_asym, err_asym)
    ],
}
Path(__file__).with_name("m1.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
