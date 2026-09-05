"""ch27-m06 驱动脚本 —— SmoothQuant 难度迁移的可示教数值轨迹。

跑法(host, 纯 CPU numpy): python run_ch27_m06_smoothquant.py
输出: ch27_m06_smoothquant.json(与本脚本同目录)

素材对应 dossier 机制 ch27-m06,论文 arXiv:2211.10438 §4 Eq.3(等价平滑
X̂·Ŵ = X·W)、Eq.4(s_j = max|X_j|^α/max|W_j|^{1-α})、§5.5 Figure 10(α
两端崩、甜点 0.4-0.6)。合成层:6 token × 4 通道,通道 0 离群(逐 token
持续 ~70,§3 obs.2/obs.3 的量级),固定显式值可手算;α=0.5 时逐通道
max|X̂_j| == max|Ŵ_j| == sqrt(max|X_j|·max|W_j|)(精确配平,浮点 1e-12 验证)。
W8A8 per-tensor(静态对称)模拟 + α 全扫描。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from smoothquant import (  # noqa: E402
    apply_smoothing,
    migration_ablation,
    smooth_scale,
    w8a8_output_error,
    w8a8_per_tensor_output,
)


def r(v, nd=4):
    return round(float(v), nd)


def main():
    out = {}

    # ---- 合成层:6 token × 4 通道,通道 0 离群 ~70(固定显式值) ----
    X = np.array(
        [
            [70.0, 0.9, -1.2, 0.2],
            [63.0, -0.4, 0.7, 0.5],
            [77.0, 0.6, 0.9, -0.7],
            [66.0, -0.8, -0.6, 0.4],
            [59.0, 0.3, 1.1, -0.6],
            [72.0, -0.5, -0.3, 0.8],
        ]
    )
    W = np.array(
        [
            [0.5, -0.3],
            [-2.0, 1.5],
            [1.0, 0.8],
            [-0.6, -1.0],
        ]
    )
    act_max = np.abs(X).max(axis=0)
    w_max = np.abs(W).max(axis=1)

    out["params"] = {
        "X_shape": "6 tokens x 4 channels",
        "W_shape": "4 x 2 (C_i x C_o)",
        "channel0_outlier_values": [70.0, 63.0, 77.0, 66.0, 59.0, 72.0],
        "note": "通道 0 离群且逐 token 持续(§3 obs.3);通道 1-3 ~1;权重分布平坦(§3 obs.1)",
    }

    # ---- α=0.5:逐通道配平表(Eq.4 手算) ----
    s = smooth_scale(X, W, alpha=0.5)
    X_hat, W_hat = apply_smoothing(X, W, s)
    xhat_max = np.abs(X_hat).max(axis=0)
    what_max = np.abs(W_hat).max(axis=1)
    out["alpha_half_table"] = {
        "columns": ["j", "max_X_j", "max_W_j", "s_j_alpha_half", "max_Xhat_j", "max_What_j"],
        "rows": [
            [
                j,
                r(act_max[j], 2),
                r(w_max[j], 2),
                r(s[j]),
                r(xhat_max[j]),
                r(what_max[j]),
            ]
            for j in range(4)
        ],
        "s_formula": "s_j = sqrt(max|X_j| / max|W_j|)",
        "equalized_claim": "max|X̂_j| == max|Ŵ_j| == sqrt(max|X_j|·max|W_j|)——同通道激活与权重难度均分(§4 原话 share a similar maximum value)",
        "equalize_max_rel_diff": r(
            np.max(np.abs(xhat_max - what_max) / what_max), 12
        ),
        "smoothing_ratio_before": r(act_max.max() / act_max.min(), 1),
        "smoothing_ratio_after": r(xhat_max.max() / xhat_max.min(), 1),
    }

    # ---- Eq.3 严格等价(浮点验证) ----
    diff = np.abs(X_hat @ W_hat - X @ W).max()
    out["equivalence_check"] = {
        "max_abs_diff_XhatWhat_minus_XW": r(diff, 12),
        "exactly_equal_up_to_float": bool(diff < 1e-12),
        "claim": "Eq.3:任意逐通道 s,(X·diag(s)^{-1})·(diag(s)·W) == X·W——迁移是免费的",
    }

    # ---- W8A8 per-tensor 误差:α=0(不平) vs 0.5 vs 1(反向爆) ----
    y_fp16 = X @ W
    err0 = w8a8_output_error(X, W, alpha=0.0)
    err_half = w8a8_output_error(X, W, alpha=0.5)
    err1 = w8a8_output_error(X, W, alpha=1.0)
    y_plain = w8a8_per_tensor_output(X, W, num_bits=8)
    err_plain = float(np.linalg.norm(y_plain - y_fp16))
    # 两个极端下的「谁难量化」:平滑后激活/权重各自的 per-tensor 动态范围比
    s1 = smooth_scale(X, W, alpha=1.0)
    X1, W1 = apply_smoothing(X, W, s1)
    out["w8a8_errors"] = {
        "err_no_smooth_per_tensor_int8": r(err_plain),
        "err_alpha_0": r(err0),
        "err_alpha_0_note": "α=0:s_j=1/max|W_j|=[2, 0.5, 1, 1] 只把权重各行归一,激活通道间 96x 的离群差距原封不动留在激活侧(平滑后 X̂ 各通道 max 仍差 48x: [38.5, 1.8, 1.2, 0.8])",
        "err_alpha_half": r(err_half),
        "err_alpha_1": r(err1),
        "err_alpha_1_note": "α=1:激活全平(各通道 max 全等 1.0)但权重爆——W 行 max 变为 [38.5, 1.8, 1.2, 0.8],per-tensor 尺子被权重侧撑爆",
        "W_row_max_after_alpha1": [r(v, 1) for v in np.abs(W1).max(axis=1)],
        "X_col_max_after_alpha1": [r(v, 2) for v in np.abs(X1).max(axis=0)],
        "improvement_ratio_alpha0_over_half": r(err0 / err_half, 1),
        "improvement_ratio_alpha1_over_half": r(err1 / err_half, 1),
    }

    # ---- §5.5 Figure 10 协议:α 0→1 扫描 ----
    history = migration_ablation(X, W, num_bits=8)
    errs = [e for _, e in history]
    best = history[int(np.argmin(errs))][0]
    out["migration_ablation"] = {
        "alphas": [r(a, 2) for a, _ in history],
        "errors": [r(e, 2) for e in errs],
        "best_alpha": r(best, 2),
        "paper_finding": "OPT-175B 上 α<0.4 激活难量化、α>0.6 权重难量化,甜点 0.4-0.6(arXiv:2211.10438 §5.5);OPT/BLOOM 通用甜点 0.5,GLM-130B(离群 ~30%)用 0.75",
    }

    p = Path(__file__).with_name("ch27_m06_smoothquant.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
