"""ch27-m03 驱动脚本 —— RTN 之死:离群值吃掉有效位的数值轨迹。

跑法(host, 纯 CPU numpy): python run_ch27_m03_rtn_death.py
输出: ch27_m03_rtn_death.json(与本脚本同目录)

素材对应 dossier 机制 ch27-m03,论文 arXiv:2211.10438 §3 obs.2(通道 i 的
有效量化级数 = 2^N·m_i/m)。合成 8 token × 4 通道矩阵:通道 0 离群
(逐 token 持续 ~70,§3 obs.3 的离群模式),通道 1-3 普通(~1)。per-tensor
INT8 一把尺子被离群撑爆 → 普通通道坍缩到 {−2..2} 附近;per-channel 一把
尺子/通道全保住。通道 max 用固定显式数组(读者可手算 2^8·m_i/m)。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from uniform_quant import (  # noqa: E402
    effective_quant_levels,
    quantize_per_channel,
    quantize_per_tensor,
)


def r(v, nd=4):
    return round(float(v), nd)


def main():
    out = {}

    # ---- 合成矩阵:通道 0 离群 ~70 且逐 token 持续;通道 1-3 普通 ~1 ----
    ch0 = np.array([70.0, 63.0, 77.0, 66.0, 59.0, 72.0, 68.0, 74.0])
    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 4))
    x[:, 0] = ch0  # 离群通道(固定显式值,可手算)

    num_bits = 8
    q_t, d_t = quantize_per_tensor(x, num_bits)
    q_c, d_c = quantize_per_channel(x, num_bits)
    err_t = np.abs(x - q_t * d_t)
    err_c = np.abs(x - q_c * d_c[None, :])

    tensor_max = float(np.abs(x).max())
    channel_max = np.abs(x).max(axis=0)

    out["params"] = {
        "shape": "8 tokens x 4 channels",
        "num_bits": 8,
        "channel0_values": [int(v) for v in ch0],
        "tensor_max_m": r(tensor_max, 1),
        "note": "通道 0 离群(固定显式值,逐 token 持续 ~70);通道 1-3 普通 ~1(rng(2) 种子)",
    }

    # ---- 逐通道账:有效级数 + 两种粒度的误差 ----
    rows = []
    for j in range(4):
        rows.append(
            {
                "channel": j,
                "channel_max_m_i": r(channel_max[j], 2),
                "effective_levels_256_m_i_over_m": r(
                    effective_quant_levels(channel_max[j], tensor_max, num_bits), 2
                ),
                "per_tensor_mean_abs_err": r(err_t[:, j].mean()),
                "per_channel_mean_abs_err": r(err_c[:, j].mean()),
                "per_tensor_unique_codes": int(np.unique(q_t[:, j]).size),
                "per_channel_unique_codes": int(np.unique(q_c[:, j]).size),
            }
        )
    out["per_channel_account"] = rows

    # ---- 量化前后坍缩对比(普通通道 ch1 的原始值 vs per-tensor 反量化值) ----
    out["collapse_demo_channel1"] = {
        "original_values": [r(v, 2) for v in x[:, 1]],
        "per_tensor_dequant": [r(v, 2) for v in q_t[:, 1] * d_t],
        "per_channel_dequant": [r(v, 2) for v in q_c[:, 1] * d_c[1]],
        "per_tensor_delta": r(d_t),
        "per_channel_delta_ch1": r(d_c[1]),
        "per_tensor_codes": [int(v) for v in q_t[:, 1]],
        "per_channel_codes": [int(v) for v in q_c[:, 1]],
    }

    out["verdict"] = {
        "normal_channels_err_ratio_tensor_over_channel": r(
            err_t[:, 1:].mean() / err_c[:, 1:].mean(), 1
        ),
        "normal_channels_per_tensor_unique_codes_total": int(
            np.unique(q_t[:, 1:]).size
        ),
        "outlier_channel_per_tensor_unique_codes": int(np.unique(q_t[:, 0]).size),
        "claim": "per-tensor 一把尺子:普通通道有效级数 5.0-8.1(= 256·m_i/m),值坍缩到 ±0.61 的少数倍数;per-channel 每 channel 一把,级数回到 256;离群通道自身两种粒度下逐码相同(它就是 per-tensor 尺子的定尺者)",
    }

    # ---- 公式自检(§3 obs.2 原式对 m_i 线性) ----
    out["formula_checks"] = {
        "levels_outlier_channel": r(effective_quant_levels(77.0, 77.0, 8), 1),
        "levels_half_outlier": r(effective_quant_levels(38.5, 77.0, 8), 1),
        "levels_normal_int8": r(effective_quant_levels(1.0, 70.0, 8), 2),
        "levels_normal_int4": r(effective_quant_levels(1.0, 70.0, 4), 3),
    }

    p = Path(__file__).with_name("ch27_m03_rtn_death.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
