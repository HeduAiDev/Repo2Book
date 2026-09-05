"""ch20-m03 驱动脚本 —— ⊕ 算子(Eq.3/Eq.4)结合律/交换律的分块归并轨迹。

跑法(host, 纯 CPU numpy): python run_ch20_m03_merge_op.py
输出: ch20_m03_merge_op.json

素材对应 dossier 机制 ch20-m03(⊕ 合并算子),论文 arXiv:1805.02867 §3.1
Eq.(3)(链式展开)+Eq.(4)(⊕ 定义);论文自述省略证明,这里用同一向量 [1,3,2,5,4] 的
多种分块/乱序归并验证末值恒等(数值意义上的结合律+交换律实例)。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from online_softmax import (  # noqa: E402
    combine_blocks_via_merge,
    online_softmax_merge,
    online_softmax_stats,
)


def r(v, nd=4):
    return round(float(v), nd)


def main():
    out = {}
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    out["params"] = {"x": [1, 3, 2, 5, 4], "V": 5}

    # 顺序单遍(基准)
    m_seq, d_seq = online_softmax_stats(x)
    out["sequential_one_pass"] = {"m": r(m_seq), "d": r(d_seq)}

    # 分块方案 A: [1,3] | [2,5] | [4],顺序归并 ((b1 ⊕ b2) ⊕ b3)
    blocks_a = [[1.0, 3.0], [2.0, 5.0], [4.0]]
    local_a = [online_softmax_stats(np.array(b)) for b in blocks_a]
    s12 = online_softmax_merge(local_a[0], local_a[1])
    s123 = online_softmax_merge(s12, local_a[2])
    out["split_A_sequential_merge"] = {
        "blocks": [[1, 3], [2, 5], [4]],
        "local_states": [{"block": b, "m": r(m), "d": r(d)} for b, (m, d) in zip(blocks_a, local_a)],
        "step1_b1_oplus_b2": {"m": r(s12[0]), "d": r(s12[1])},
        "step2_oplus_b3": {"m": r(s123[0]), "d": r(s123[1])},
    }

    # 乱序归并: (b3 ⊕ b2) ⊕ b1 —— 交换律
    s32 = online_softmax_merge(local_a[2], local_a[1])
    s321 = online_softmax_merge(s32, local_a[0])
    out["reordered_merge_b3_b2_b1"] = {
        "step1_b3_oplus_b2": {"m": r(s32[0]), "d": r(s32[1])},
        "step2_oplus_b1": {"m": r(s321[0]), "d": r(s321[1])},
    }

    # 不同分块方案 B: [1,3,2] | [5,4](括号位置不同,结合律)
    m_b, d_b = combine_blocks_via_merge(x, 3)
    local_b = [online_softmax_stats(np.array([1.0, 3.0, 2.0])), online_softmax_stats(np.array([5.0, 4.0]))]
    out["split_B_diff_parentheses"] = {
        "blocks": [[1, 3, 2], [5, 4]],
        "local_states": [
            {"block": [1, 3, 2], "m": r(local_b[0][0]), "d": r(local_b[0][1])},
            {"block": [5, 4], "m": r(local_b[1][0]), "d": r(local_b[1][1])},
        ],
        "merged": {"m": r(m_b), "d": r(d_b)},
    }

    # 单块特例: 不分块(块大小 V) == 顺序单遍
    m_c, d_c = combine_blocks_via_merge(x, 5)
    out["no_split_single_block"] = {"m": r(m_c), "d": r(d_c)}

    out["all_equal_to_sequential"] = {
        "split_A": bool(abs(s123[1] - d_seq) < 1e-12 and s123[0] == m_seq),
        "reordered": bool(abs(s321[1] - d_seq) < 1e-12 and s321[0] == m_seq),
        "split_B": bool(abs(d_b - d_seq) < 1e-12 and m_b == m_seq),
        "no_split": bool(abs(d_c - d_seq) < 1e-12 and m_c == m_seq),
    }
    out["max_abs_d_diff_across_orders"] = r(
        max(abs(s123[1] - d_seq), abs(s321[1] - d_seq), abs(d_b - d_seq), abs(d_c - d_seq)), 12
    )

    p = Path(__file__).parent / "ch20_m03_merge_op.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"OK {p}")


if __name__ == "__main__":
    main()
