"""ch27-m01 驱动脚本 —— 均匀量化底座(scale/zero-point、对称与非对称)的可示教数值轨迹。

跑法(host, 纯 CPU numpy): python run_ch27_m01_uniform_base.py
输出: ch27_m01_uniform_base.json(与本脚本同目录)

素材对应 dossier 机制 ch27-m01,论文 arXiv:2211.10438 §2 Eq.1(对称式
X_bar=round(X/Δ)、Δ=max|X|/(2^(N-1)-1))+ arXiv:2210.17323 §5 Setup(非对称
min-max 网格 + zero-point)。参数:8 元素小向量 INT4 对称量化(读者可心算,
Δ=1/7,误差全部非零、无一落在格点上——非平凡分支),再推两个非对称版本
(ReLU 后正值分布 + 负偏置分布)。np.round 为银行家舍入(0.5 取偶),
选值避开 0.5 平手点。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from uniform_quant import (  # noqa: E402
    dequantize_asymmetric,
    dequantize_symmetric,
    quantize_asymmetric,
    quantize_symmetric,
)


def r(v, nd=4):
    return round(float(v), nd)


def main():
    out = {}

    # ---- ① 对称 INT4:8 元素小向量(dossier m01 worked example) ----
    x = np.array([0.9, -0.4, 0.15, -1.0, 0.62, -0.77, 0.3, 0.05])
    num_bits = 4
    qmax = 2 ** (num_bits - 1) - 1  # 7
    q, delta = quantize_symmetric(x, num_bits)
    x_hat = dequantize_symmetric(q, delta)
    err = x - x_hat
    out["symmetric_int4"] = {
        "x": [r(v) for v in x],
        "num_bits": 4,
        "qmax": 7,
        "absmax": 1.0,
        "delta": r(delta),
        "delta_formula": "max|x|/7 = 1/7",
        "x_over_delta": [r(v / delta, 2) for v in x],
        "codes_q": [int(v) for v in q],
        "dequant_x_hat": [r(v) for v in x_hat],
        "err_x_minus_xhat": [r(v) for v in err],
        "abs_err_max": r(np.abs(err).max()),
        "half_step_delta_over_2": r(delta / 2),
        "all_err_within_half_step": bool(np.all(np.abs(err) <= delta / 2 + 1e-12)),
        "note": "无一元素落在格点上(8 个误差全非 0);最大值 -1.0 精确映到 -7(Δ 用 absmax 定尺,不丢最远点)",
    }

    # ---- ② 非对称 min-max INT4:ReLU 后正值分布(0 不在格点) ----
    x2 = np.array([0.0, 0.53, 1.0, 0.73])
    q2, scale2, zp2 = quantize_asymmetric(x2, num_bits)
    x2_hat = dequantize_asymmetric(q2, scale2, zp2)
    err2 = x2 - x2_hat
    out["asymmetric_int4_relu"] = {
        "x": [r(v) for v in x2],
        "num_bits": 4,
        "xmin": 0.0,
        "xmax": 1.0,
        "qmin": -8,
        "qmax": 7,
        "scale": r(scale2),
        "scale_formula": "(xmax-xmin)/(qmax-qmin) = 1/15",
        "zero_point_zp": int(zp2),
        "zp_formula": "qmin - round(xmin/scale) = -8 - 0 = -8",
        "codes_q": [int(v) for v in q2],
        "x_over_scale": [r(v / scale2, 2) for v in x2],
        "dequant_x_hat": [r(v) for v in x2_hat],
        "err_x_minus_xhat": [r(v) for v in err2],
        "abs_err_max": r(np.abs(err2).max()),
        "half_step_scale_over_2": r(scale2 / 2),
        "note": "xmin=0 精确映到 qmin=-8、xmax=1.0 精确映到 qmax=7(两端落格);中间值 0.53/0.73 有非零误差",
    }

    # ---- ③ 非对称 min-max INT4:负偏置分布(zp 为正) ----
    x3 = np.array([-2.0, -1.0, 0.45])
    q3, scale3, zp3 = quantize_asymmetric(x3, num_bits)
    x3_hat = dequantize_asymmetric(q3, scale3, zp3)
    err3 = x3 - x3_hat
    out["asymmetric_int4_negative_bias"] = {
        "x": [r(v) for v in x3],
        "num_bits": 4,
        "xmin": -2.0,
        "xmax": 0.45,
        "scale": r(scale3),
        "scale_formula": "(0.45-(-2.0))/15 = 2.45/15",
        "zero_point_zp": int(zp3),
        "zp_formula": "qmin - round(xmin/scale) = -8 - round(-2/0.1633) = -8 - (-12) = 4",
        "codes_q": [int(v) for v in q3],
        "dequant_x_hat": [r(v) for v in x3_hat],
        "err_x_minus_xhat": [r(v) for v in err3],
        "abs_err_max": r(np.abs(err3).max()),
        "half_step_scale_over_2": r(scale3 / 2),
        "note": "分布整体偏负:zp=+4 把网格原点搬到 -0.6533;0.45 精确映到 qmax=7",
    }

    # ---- ④ 误差上界 Δ/2 的统计验证(512 随机值 × N=4/8) ----
    rng = np.random.default_rng(0)
    bound_check = {}
    for nb in (4, 8):
        xs = rng.standard_normal(512) * 0.7
        qq, dd = quantize_symmetric(xs, nb)
        e = np.abs(xs - dequantize_symmetric(qq, dd))
        bound_check[f"num_bits_{nb}"] = {
            "num_bits": nb,
            "delta": r(dd),
            "half_step": r(dd / 2),
            "max_abs_err": r(e.max()),
            "within_bound": bool(e.max() <= dd / 2 + 1e-12),
            "mean_abs_err": r(e.mean()),
        }
    out["half_step_bound_check"] = bound_check

    # ---- ⑤ clamp 兜什么:对称式下 |x|/Δ <= qmax 恒成立,clip 只是浮点护栏 ----
    out["clamp_note"] = {
        "claim": "对称式 Δ=max|x|/qmax 保证 |x|/Δ <= qmax,取整后码域恒不越界;clip 防的是浮点除法抖动越过半格",
        "symmetric_max_code": int(np.max(np.abs(q))),
        "qmax": 7,
        "clip_triggered": bool(np.any(np.abs(q) > qmax)),
        "table_row_indices": [1, 2, 3, 4, 5, 6, 7, 8],
    }

    p = Path(__file__).with_name("ch27_m01_uniform_base.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
