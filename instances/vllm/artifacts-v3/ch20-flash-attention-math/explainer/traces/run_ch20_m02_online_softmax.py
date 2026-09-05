"""ch20-m02 驱动脚本 —— online-softmax 单遍递推的可示教数值轨迹。

跑法(host, 纯 CPU numpy): python run_ch20_m02_online_softmax.py
输出: ch20_m02_online_softmax.json(与本脚本同目录)

素材对应 dossier 机制 ch20-m02(online-softmax 单遍递推: running (m,d) 与 rescale 项),
论文 arXiv:1805.02867 §2 Alg.1/Alg.2、§3 Alg.3 + Theorem 1。
参数 x=[1,3,2,5,4]: 5 个一位数整数,读者可心算; max 两次上移(j=2: 1→3、j=4: 3→5),
两次都触发非平凡 rescale 项 e^{m_{j-1}-m_j}=e^{-2}≈0.1353(非退化分支);
两次不上移(j=3、j=5)的步 rescale 项=e^0=1(对照分支)。
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from online_softmax import (  # noqa: E402
    naive_softmax,
    online_softmax,
    online_softmax_stats,
    safe_softmax,
)


def r(v, nd=4):
    return round(float(v), nd)


def main():
    out = {}

    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    out["params"] = {
        "x": [1, 3, 2, 5, 4],
        "V": 5,
        "note": "max 两次上移(j=2:1->3 与 j=4:3->5),rescale=e^{-2};j=3/j=5 max 不变,rescale=e^0=1",
    }

    # ---- 三版末值对照(Theorem 1) ----
    y_naive = naive_softmax(x)
    y_safe = safe_softmax(x)
    y_online = online_softmax(x)
    out["three_versions"] = {
        "naive_softmax": [r(v) for v in y_naive],
        "safe_softmax": [r(v) for v in y_safe],
        "online_softmax": [r(v) for v in y_online],
        "max_abs_diff_naive_vs_safe": r(np.abs(y_naive - y_safe).max(), 12),
        "max_abs_diff_online_vs_safe": r(np.abs(y_online - y_safe).max(), 12),
        "all_close_online_vs_safe": bool(np.allclose(y_online, y_safe, rtol=0, atol=1e-12)),
    }

    # ---- 单遍递推逐轮轨迹(Alg.3 lines 1-6) ----
    trace = []
    m_v, d_v = online_softmax_stats(x, trace=trace)
    rows = []
    m_prev, d_prev = -math.inf, 0.0
    for j, (mj, dj) in enumerate(trace, start=1):
        xj = x[j - 1]
        rows.append({
            "j": j,
            "x_j": int(xj),
            "m_{j-1}": (-math.inf if j == 1 else r(m_prev)),
            "m_j": r(mj),
            "rescale_e^{m_{j-1}-m_j}": (None if j == 1 else r(math.exp(m_prev - mj))),
            "d_{j-1}": (0 if j == 1 else r(d_prev)),
            "old_account_after_rescale": (0 if j == 1 else r(d_prev * math.exp(m_prev - mj))),
            "new_term_e^{x_j-m_j}": r(math.exp(xj - mj)),
            "d_j": r(dj),
            "bound_1_le_d_j_le_j": bool(1.0 <= dj <= j),
            "d_j_full": dj,
        })
        m_prev, d_prev = mj, dj
    out["online_recurrence"] = {
        "init": {"m_0": "-inf", "d_0": 0},
        "steps": rows,
        "final": {"m_V": r(m_v), "d_V": r(d_v), "m_V_exact": m_v, "d_V_exact": d_v},
        "d_V_equals_sum_exp_x_minus_mV": r(d_v - float(np.exp(x - m_v).sum()), 12),
    }

    # ---- safe softmax 侧的同一对末值(对照: 三遍法也算出同一 (m_V,d_V)) ----
    m_safe = float(x.max())
    d_safe = float(np.exp(x - m_safe).sum())
    out["safe_softmax_stats"] = {
        "m_V": r(m_safe),
        "d_V": r(d_safe),
        "passes_over_x": 3,
        "memory_accesses_per_element": 4,
        "source": "arXiv:1805.02867 §2(Algorithm 2):three passes / 4 memory access per element",
    }
    out["online_softmax_stats_meta"] = {
        "passes_over_x_for_stats": 1,
        "memory_accesses_per_element": 3,
        "source": "arXiv:1805.02867 §3:'It reduces memory accesses from 4 down to 3 per vector element'",
    }

    # ---- 溢出对照(naive 在 e^1000 上溢;float64 上 e^1000 同样 = inf) ----
    with np.errstate(over="ignore", invalid="ignore"):
        y_big_naive = naive_softmax(np.array([1000.0, 1001.0]))
        y_big_safe = safe_softmax(np.array([1000.0, 1001.0]))
        y_big_online = online_softmax(np.array([1000.0, 1001.0]))
    out["overflow_demo"] = {
        "x": [1000, 1001],
        "e^1000_in_float64": "inf",
        "naive_softmax": ["nan", "nan"],
        "safe_softmax": [r(v) for v in y_big_safe],
        "online_softmax": [r(v) for v in y_big_online],
        "note": "inf/inf=nan;safe/online 先减 max=1001,最大项 e^{1001-1001}=e^0=1 不溢出",
    }

    p = Path(__file__).parent / "ch20_m02_online_softmax.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"OK {p}")


if __name__ == "__main__":
    main()
