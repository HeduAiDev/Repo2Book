"""ch20-m04 驱动脚本 —— FlashAttention tiling 免物化的 2×2 分块可示教轨迹。

跑法(host, 纯 CPU numpy): python run_ch20_m04_fa_tiling.py
输出: ch20_m04_fa_tiling.json

素材对应 dossier 机制 ch20-m04(FA tiling 免物化: running (m,ℓ,O) 的缩放-累加递推),
论文 arXiv:2205.14135 §3.1 Algorithm 1 + Theorem 1。
参数: N=4, d=2, B_r=B_c=2(2×2 分块, T_r=T_c=2), softmax_scale=1.0(显式传 1,
避开默认 1/sqrt(2) 的无理数中间值,让读者可心算;默认值语义在 m09 讲)。
行块 1 的 max 在第二个 KV 块从 1 上移到 2,rescale 项 e^{1-2}=e^{-1}≈0.3679(非平凡)。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from flash_attention import (  # noqa: E402
    flash_attention_forward,
    standard_attention,
)


def r(v, nd=4):
    return round(float(v), nd)


def main():
    out = {}
    Q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 2.0]])
    V = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    out["params"] = {
        "N": 4, "d": 2, "B_r": 2, "B_c": 2, "T_r": 2, "T_c": 2,
        "softmax_scale": 1.0,
        "Q": [[1, 0], [0, 1], [1, 1], [2, 0]],
        "K": [[1, 0], [0, 1], [1, 1], [0, 2]],
        "V": [[1, 2], [3, 4], [5, 6], [7, 8]],
        "loop_order": "外层 KV 列块 j(Algorithm 1 line 5),内层 Q 行块 i(line 7)",
    }
    # 打分矩阵(仅驱动脚本内部用来核对,FA 从不物化它)
    S_full = (Q @ K.T) * 1.0
    out["S_full_for_verification_only"] = [[r(v) for v in row] for row in S_full]

    trace = []
    O = flash_attention_forward(Q, K, V, block_size_r=2, block_size_c=2,
                                scale=1.0, trace=trace)
    max_shape = flash_attention_forward(Q, K, V, block_size_r=2, block_size_c=2,
                                        scale=1.0, return_max_block_shape=True)
    out["max_S_block_shape_ever_created"] = {
        "shape": list(max_shape),
        "vs_full_NxN": [4, 4],
        "note": "S/P 只以 2x2 块存在于局部变量(代表 SRAM),从不存在 4x4 整表(代表 HBM)",
    }

    # 逐 (j,i) 递推轨迹: 每行块给出 旧 m -> 新 m 的 rescale 因子
    m_hist = {}
    steps = []
    for t in trace:
        i_blk = t["i"]
        m_old = m_hist.get(i_blk, None)
        rows = []
        for ridx in range(len(t["m"])):
            m_o = None if m_old is None else m_old[ridx]
            m_n = t["m"][ridx]
            rows.append({
                "q_row": t["q0"] + ridx,
                "m_old": (None if m_o is None else r(m_o)),
                "m_new": r(m_n),
                "rescale_e^{m_old-m_new}": (None if m_o is None else r(np.exp(m_o - m_n)) if m_o != -np.inf else 0.0),
                "l_new": r(t["l"][ridx]),
                "O_row": [r(v) for v in t["O"][ridx]],
            })
        steps.append({"j": t["j"], "i": t["i"], "q_block": [t["q0"], t["q1"]], "kv_end": t["kv_end"], "rows": rows})
        m_hist[i_blk] = t["m"]
    out["tiling_steps"] = steps

    # 「O_i 每步都是至今为止的正确答案」: 逐 (j,i) 步,该行块的 O
    # == 对 K[:kv_end] 的朴素注意力(只看已见过的 KV 列块)
    checks = []
    for t in trace:
        O_ref = standard_attention(Q[t["q0"]:t["q1"]], K[:t["kv_end"]], V[:t["kv_end"]], scale=1.0)
        checks.append({
            "step": f"(j={t['j']},i={t['i']})",
            "q_rows": [t["q0"], t["q1"]],
            "kv_seen": t["kv_end"],
            "max_abs_diff_vs_naive_on_seen_kv": r(np.abs(t["O"] - O_ref).max(), 12),
            "max_abs_diff_full": float(np.abs(t["O"] - O_ref).max()),
            "allclose_atol_1e-12": bool(np.allclose(t["O"], O_ref, rtol=0, atol=1e-12)),
        })
    out["per_step_correctness_checks"] = checks

    # 终值: FA(全 4 个 K/V) vs 朴素标准注意力
    O_std = standard_attention(Q, K, V, scale=1.0)
    out["final"] = {
        "FA_O": [[r(v) for v in row] for row in O],
        "standard_O": [[r(v) for v in row] for row in O_std],
        "max_abs_diff": r(np.abs(O - O_std).max(), 12),
        "max_abs_diff_full": float(np.abs(O - O_std).max()),
        "allclose_atol_1e-12": bool(np.allclose(O, O_std, rtol=0, atol=1e-12)),
        "note": "Theorem 1: 任意合法分块,输出精确等于 softmax(QK^T)V;实跑为 float64,差异只剩求和顺序的机器精度级",
    }

    p = Path(__file__).parent / "ch20_m04_fa_tiling.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"OK {p}")


if __name__ == "__main__":
    main()
