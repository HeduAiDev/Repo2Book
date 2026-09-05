"""ch20-m06 驱动脚本 —— FA vs FA-2 因果掩码下的块访问账(整块跳过红利)。

跑法(host, 纯 CPU numpy): python run_ch20_m06_causal_blocks.py
输出: ch20_m06_causal_blocks.json

素材对应 dossier 机制 ch20-m06(FA-2 三改),论文 arXiv:2307.08691 §3.1.1
Causal masking("approximately half of the blocks ... skip ... 1.7-1.8x speedup")。
FA 原序实现逐块施掩码不跳块;FA-2 实现整块在因果上侧直接 continue。
两版输出对标准注意力逐位相等(同一份数学、两种调度)。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from flash_attention import (  # noqa: E402
    flash_attention_2_forward,
    flash_attention_forward,
    standard_attention,
)


def r(v, nd=4):
    return round(float(v), nd)


def run_case(N, B):
    rng = np.random.default_rng(20 + N)
    Q, K, V = rng.standard_normal((3, N, 2))
    t_fa, t_fa2 = [], []
    O_fa = flash_attention_forward(Q, K, V, block_size_r=B, block_size_c=B,
                                   causal=True, scale=1.0, trace=t_fa)
    O_fa2, L_fa2 = flash_attention_2_forward(Q, K, V, block_size_r=B, block_size_c=B,
                                             causal=True, scale=1.0, trace=t_fa2)
    O_std = standard_attention(Q, K, V, causal=True, scale=1.0)
    T = -(-N // B)
    return {
        "N": N, "B_r": B, "B_c": B, "T_r": T, "T_c": T,
        "all_blocks_T_r_x_T_c": T * T,
        "FA_visited_blocks": len(t_fa),
        "FA2_visited_blocks": len(t_fa2),
        "FA2_skipped_blocks": T * T - len(t_fa2),
        "compute_ratio_all_over_FA2": round(T * T / len(t_fa2), 4),
        "FA2_vs_standard_max_abs_diff_full": float(np.abs(O_fa2 - O_std).max()),
        "FA_vs_standard_max_abs_diff_full": float(np.abs(O_fa - O_std).max()),
        "FA2_output_allclose_standard": bool(np.allclose(O_fa2, O_std, rtol=1e-6, atol=1e-8)),
        "FA_output_allclose_standard": bool(np.allclose(O_fa, O_std, rtol=1e-6, atol=1e-8)),
        "FA2_L_shape": list(L_fa2.shape),
        "FA2_L_row0": r(L_fa2[0]),
    }


def main():
    out = {"cases": [run_case(8, 2), run_case(64, 8)]}
    out["paper_reference"] = {
        "claim": "for any blocks where all the column indices are more than the row indices (approximately half of the blocks for large sequence length), we can skip the computation of that block. This leads to around 1.7-1.8x speedup",
        "source": "arXiv:2307.08691 §3.1.1 Causal masking (paper-fa2.md:L207)",
        "note": "N=8/B=2 时对角块占比大,16/10=1.6;N=64/B=8 时 64/36≈1.78 已进入论文 1.7-1.8x 区间;N->inf 趋近 2x",
    }
    p = Path(__file__).parent / "ch20_m06_causal_blocks.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"OK {p}")


if __name__ == "__main__":
    main()
