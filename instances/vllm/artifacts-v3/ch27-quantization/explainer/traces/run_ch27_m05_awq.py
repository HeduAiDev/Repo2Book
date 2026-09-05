"""ch27-m05 驱动脚本 —— AWQ 激活感知逐通道缩放的可示教数值轨迹。

跑法(host, 纯 CPU numpy): python run_ch27_m05_awq.py
输出: ch27_m05_awq.json(与本脚本同目录)

素材对应 dossier 机制 ch27-m05,论文 arXiv:2306.00978 §3.2 Eq.1-Eq.3(组量化器/
等价缩放/误差表达式与 (Δ'/Δ)·(1/s) 误差比)、obs.(1)-(3)(RoundErr~0.25、
Δ'≈Δ、FP16 中间量无误差)、Table 2(逐组统计协议)、Eq.4-Eq.5(L(s) 与
s=s_X^α 搜索)、§4.2(SIMD-aware packing w_{0,2,4,6,1,3,5,7})。

手推件:组 [0.9, 9.9](显著权重 0.9 + 定尺者 9.9),s=2:x=1 时
未缩放误差 0.3375 → 缩放后 −0.28125;单点误差比 5/6,平均意义的理论比
(Δ'/Δ)·(1/s)=1/2——RoundErr 是随机变量、期望 ~0.25(非平凡分支:
实测比 ≠ 理论比,差异本身是 obs.(1) 的示范)。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from awq import (  # noqa: E402
    awq_dequantize,
    awq_group_quantize,
    awq_loss,
    awq_pack,
    awq_search_scale,
    channel_mean_activation,
    err_Qws_xs,
    err_Qwx,
    round_err,
    salient_channels,
    table2_statistics,
)


def r(v, nd=5):
    return round(float(v), nd)


def main():
    out = {}

    # ---- ① 手推件:显著权重 0.9、组内另有 9.9 定尺,s=2 ----
    group = np.array([0.9, 9.9])
    q, delta = awq_group_quantize(group, num_bits=4)
    x = 1.0
    s = 2.0
    e_plain = err_Qwx(0.9, x, delta)
    ws = 0.9 * s
    _, delta_p = awq_group_quantize(group * np.array([s, 1.0]), num_bits=4)
    e_scaled = err_Qws_xs(0.9, x, s, delta)  # Δ'=Δ
    q_ws = float(np.clip(np.round(ws / delta_p), -8, 8))
    out["hand_example"] = {
        "group": [0.9, 9.9],
        "num_bits": 4,
        "qmax_awq_convention": 8,
        "delta": r(delta),
        "delta_formula": "max|w|/2^(N-1) = 9.9/8(AWQ Eq.1 分母 2^(N-1),与 SmoothQuant Eq.1 的 (2^(N-1)-1) 各自论文约定)",
        "codes": [int(v) for v in q],
        "dequant": [r(v) for v in awq_dequantize(q, delta)],
        "salient_w": 0.9,
        "x": 1.0,
        "s": 2.0,
        "err_unscaled_Qwx": r(e_plain),
        "err_unscaled_detail": "Q(0.9)=Δ·round(0.9/1.2375)=1.2375,误差=1.2375-0.9=0.3375",
        "w_times_s": r(ws),
        "delta_prime": r(delta_p),
        "delta_prime_equals_delta": bool(abs(delta_p - delta) < 1e-12),
        "reason_delta_prime_unchanged": "obs.(2):单元素放大 0.9->1.8 不改组内 max(9.9)",
        "round_ws_over_deltap": r(ws / delta_p, 4),
        "dequant_scaled_contribution": r(delta_p * q_ws / s),
        "err_scaled_Qws_xs": r(e_scaled),
        "err_scaled_detail": "Q(1.8)·(1/s)=1.2375·1/2=0.61875,误差=0.61875-0.9=-0.28125",
        "empirical_error_ratio": r(abs(e_scaled) / abs(e_plain)),
        "abs_err_unscaled": r(abs(e_plain)),
        "abs_err_scaled": r(abs(e_scaled)),
        "round_err_of_ws_over_deltap": r(round_err(ws / delta_p)),
        "theoretical_ratio_delta_prime_over_delta_times_1_over_s": r(delta_p / delta / s),
        "note": "单点误差比 5/6 大于平均意义的理论比 1/2:RoundErr(·) 是随机变量(误差均匀分布于 [0,0.5]、平均 0.25),单点可偏离期望——(Δ'/Δ)(1/s) 描述的是期望之比",
    }

    # ---- ② obs.(1):RoundErr 平均 ~0.25 的统计验证 ----
    rng = np.random.default_rng(1)
    u = rng.uniform(0.0, 1.0, 200_000)
    out["round_err_stats"] = {
        "n_samples": 200000,
        "mean_abs_round_err": r(np.mean(np.abs(round_err(u))), 4),
        "paper_claim": 0.25,
        "err_uniform_on": "[0, 0.5]",
    }

    # ---- ③ Table 2 协议复刻(合成层:~1.5% 显著通道,s=2) ----
    rng = np.random.default_rng(2)
    d_in, d_out, n_tokens = 128, 32, 64
    W = rng.standard_normal((d_in, d_out)) * 0.05
    X = rng.standard_normal((n_tokens, d_in))
    X[:, :2] *= 20.0  # 2/128 ~ 1.5% 显著通道
    stats = table2_statistics(W, X, s=2.0, num_bits=4, group_size=16, salient_frac=0.01)
    out["table2_protocol_synth"] = {
        "shape": "d_in=128, d_out=32, group_size=16",
        "s": 2.0,
        "salient_frac": "1%(按激活平均幅度选,channel 0-1 为 ×20 的大激活通道)",
        "proportion_delta_changed": r(stats["proportion_delta_changed"], 4),
        "mean_delta_ratio": r(stats["mean_delta_ratio"], 4),
        "mean_scaled_error_ratio": r(stats["mean_scaled_error_ratio"], 4),
        "paper_table2_opt67b": {
            "s_values": [1, 1.25, 1.5, 2, 4],
            "proportion_delta_changed": [0.0, 0.028, 0.044, 0.082, 0.212],
            "mean_delta_ratio": [1.0, 1.005, 1.013, 1.038, 1.213],
            "mean_scaled_error_ratio": [1.0, 0.804, 0.676, 0.519, 0.303],
            "wiki2_ppl": [23.54, 12.87, 12.48, 11.92, 12.36],
            "source": "arXiv:2306.00978 §3.2 Table 2(paper-awq.md:L129-L136 逐字转录);s=2 甜点:s=4 时 21.2% 通道 Δ'/Δ>1 反噬非显著通道,PPL 反弹 12.36",
        },
    }

    # ---- ④ Eq.4-Eq.5 搜索:s=s_X^α,α∈[0,1] 网格 20 点 ----
    rng = np.random.default_rng(5)
    d_in2, d_out2, n_tokens2 = 64, 32, 96
    W2 = rng.standard_normal((d_in2, d_out2)) * 0.05
    X2 = rng.standard_normal((n_tokens2, d_in2))
    X2[:, :3] *= 15.0  # ~5% 显著通道
    best_alpha, best_s, history = awq_search_scale(
        W2, X2, num_bits=3, group_size=32, grid_size=20
    )
    losses = {r(a, 2): r(l, 4) for a, l in history}
    out["search_sweep"] = {
        "shape": "d_in=64, d_out=32, INT3-g32(论文主设置)",
        "grid_size": 20,
        "alpha_grid": [r(a, 2) for a, _ in history],
        "loss_history": [r(l, 4) for _, l in history],
        "loss_at_alpha_0_rtn": r(dict(history)[0.0], 4),
        "best_alpha": r(best_alpha, 2),
        "loss_at_best_alpha": r(min(l for _, l in history), 4),
        "improvement_factor": r(dict(history)[0.0] / min(l for _, l in history), 2),
        "shape_of_curve": "U 形:α=0(RTN)最差、内点最优、α=1 反弹(Table 2/Table 3 的形状)",
        "s_x_of_salient_channel0": r(channel_mean_activation(X2)[0], 3),
        "s_x_of_normal_channel10": r(channel_mean_activation(X2)[10], 3),
        "best_s_salient_channel0": r(best_s[0], 4),
        "best_s_normal_channel10": r(best_s[10], 4),
    }

    # ---- ⑤ §4.2 SIMD-aware packing:interleave [0,2,4,6,1,3,5,7] ----
    q16 = np.arange(16, dtype=np.int64)
    packed = awq_pack(q16, num_bits=4)
    out["simd_pack"] = {
        "codes": [int(v) for v in q16[:8]],
        "interleave_order": [0, 2, 4, 6, 1, 3, 5, 7],
        "packed_word0_hex": format(int(packed[0]) & 0xFFFFFFFF, "08X"),
        "packed_word0_int": int(packed[0]),
        "vllm_crossref": "vllm/model_executor/layers/quantization/utils/quant_utils.py:L880-L899 awq_pack 同一 interleave(列维)",
    }

    p = Path(__file__).with_name("ch27_m05_awq.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
