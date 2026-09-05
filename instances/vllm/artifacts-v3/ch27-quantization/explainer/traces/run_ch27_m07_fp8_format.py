"""ch27-m07 驱动脚本 —— FP8(e4m3) 格式账:位级枚举格点 + amax 对称量化示例。

跑法(host, 纯 CPU numpy): python run_ch27_m07_fp8_format.py
输出: ch27_m07_fp8_format.json(与本脚本同目录)

素材对应 dossier 机制 ch27-m07(figure-only)。格式数学出自 OCP FP8 规范 /
NVIDIA《FP8 Formats for Deep Learning》(不在论文包,正文一句带过出处);
vLLM 落地对照 vllm/model_executor/layers/quantization/utils/quant_utils.py
的 scaled_quantize(L359-L411:amax 对称,scale = fp8_max/amax)与
get_fp8_min_max(L27-L35:torch.finfo(fp8_dtype) → ±448)。

e4m3fn 逐位枚举:符号 1 + 指数 4(bias 7)+ 尾数 3;e=15&m=7 为 NaN,
其余全有限——max = 1.75×2^8 = 448。e=0 为次正规段(m/8 × 2^-6)。
对照 INT 等距格点:同样 8 bit 铺满 [-448,448] 步长 896/255。
"""
import json
from pathlib import Path

import numpy as np


def r(v, nd=6):
    return round(float(v), nd)


def e4m3_positive_values():
    """逐位枚举 e4m3fn 的全部正有限格点(升序,不含 0)。"""
    vals = []
    for e in range(16):
        for m in range(8):
            if e == 15 and m == 7:
                continue  # e4m3fn: 仅此模式为 NaN
            if e == 0:
                if m == 0:
                    continue  # +0
                v = (m / 8.0) * 2.0**-6  # 次正规
            else:
                v = (1.0 + m / 8.0) * 2.0 ** (e - 7)  # 正规
            vals.append(v)
    return sorted(vals)


def round_to_grid(x, grid):
    grid = np.asarray(grid)
    idx = np.argmin(np.abs(grid - x))
    return float(grid[idx])


def main():
    out = {}
    pos = e4m3_positive_values()
    all_finite = sorted([-v for v in pos] + [0.0] + pos)

    out["bit_layout"] = {
        "sign_bits": 1,
        "exponent_bits": 4,
        "exponent_bias": 7,
        "mantissa_bits": 3,
        "nan_pattern": "e=15 & m=7(仅此一个模式,fn=finite NaN-only)",
        "max_finite": 448.0,
        "min_normal": 0.015625,
        "min_subnormal": 0.001953125,
        "positive_finite_count": len(pos),
        "total_finite_including_neg_and_zero": len(all_finite),
    }

    # ---- 分段结构:正规段每段 8 个等距格点,段间步长翻倍 ----
    boundaries = [2.0**k for k in range(-9, 9)]
    seg_report = []
    for k in range(len(boundaries) - 1):
        lo, hi = boundaries[k], boundaries[k + 1]
        seg = [v for v in pos if lo <= v < hi]
        if seg:
            seg_report.append(
                {
                    "segment": f"[{lo:g}, {hi:g})",
                    "count": len(seg),
                    "step": r(seg[1] - seg[0], 9) if len(seg) >= 2 else None,
                    "values": [r(v, 7) for v in seg],
                }
            )
    out["segments"] = seg_report

    # ---- INT8 对照:同 8 bit 等距铺满 [-448,448] ----
    int8_step = 896.0 / 255.0
    out["int8_contrast"] = {
        "uniform_step": r(int8_step, 4),
        "levels_per_side_plus_zero": 256,
        "claim": "INT 等距:步长恒 3.5137(896/255);FP8 指数分段:步长从 0.001953125 到 32,段间倍增——动态范围换段内精度",
    }

    # ---- amax 对称量化示例(scaled_quantize 的语义) ----
    x = np.array([1.0, 0.55, -0.3, 0.9, 100.0])
    amax = float(np.abs(x).max())
    scale = 448.0 / amax
    x_scaled = x * scale
    xq = np.array([round_to_grid(v, pos if v >= 0 else [-p for p in pos]) for v in x_scaled])
    dequant = xq / scale
    out["amax_scaling_example"] = {
        "x": [r(v, 2) for v in x],
        "amax": r(amax, 2),
        "scale_fp8_max_over_amax": r(scale, 4),
        "x_times_scale": [r(v, 3) for v in x_scaled],
        "nearest_e4m3": [r(v, 4) for v in xq],
        "dequant_x_over_scale": [r(v, 4) for v in dequant],
        "err": [r(a - b, 4) for a, b in zip(x, dequant)],
        "abs_err_max": r(np.abs(x - dequant).max(), 4),
        "note": "无 zero-point:scale=fp8_max/amax 一路到格点(quant_utils.py:L359-L411 scaled_quantize 同式);100.0 精确映到 448",
    }

    # ---- 格点密度分布:靠近 0 密、大数疏 ----
    below_one = [v for v in pos if v < 1.0]
    above_64 = [v for v in pos if v >= 64.0]
    out["density"] = {
        "positive_values_below_1": len(below_one),
        "positive_values_at_or_above_64": len(above_64),
        "values_at_or_above_64": [r(v, 1) for v in above_64],
        "claim": "126 个正格点里 55 个在 (0,1)、23 个 >= 64:小值密、大值疏——离群值天然有格点可落,普通值保相对精度",
    }

    p = Path(__file__).with_name("ch27_m07_fp8_format.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
