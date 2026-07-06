"""Driver for m8-fp8-e8m0-scale (figure-only).

Reproduces the one-line e8m0 (ue8m0) scale rounding of
vllm/model_executor/layers/quantization/input_quant_fp8.py:L242-L244:
    scales_raw = absmax / _FP8_MAX
    if use_ue8m0: scales_raw = exp2(ceil(log2(scales_raw)))
_FP8_MAX = 448.0 (e4m3fn, from get_fp8_min_max()). e8m0 stores 8 exponent
bits and no mantissa -> the block scale must be an exact power of two, so the
continuous amax-derived scale is rounded UP to the nearest 2^k.
Emits explainer/traces/m8.json.
"""
import json
import math
from pathlib import Path

FP8_MAX = 448.0                                    # input_quant_fp8.py:L24 (e4m3fn)

rows = []
for absmax in (7.0, 100.0, 300.0, 1000.0):
    raw = absmax / FP8_MAX                          # continuous scale
    k = math.ceil(math.log2(raw))                   # exponent after ceil(log2)
    e8m0 = 2.0 ** k                                  # exp2(ceil(log2(raw)))
    overshoot = round(100 * (e8m0 - raw) / raw, 2)   # how much rounding-up inflates the step
    rows.append([round(absmax, 4), round(raw, 6), k, round(e8m0, 6), overshoot])

out = {
    "params": {"FP8_MAX": FP8_MAX, "e8m0_exponent_bits": 8, "e8m0_mantissa_bits": 0},
    "rows": rows,   # [block absmax, raw scale absmax/448, exponent k, e8m0 scale 2^k, overshoot %]
    "note_2power": "e8m0 scale is exactly 2^k; ceil(log2) rounds UP so the FP8 grid never clips the block max",
}
Path(__file__).with_name("m8.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
