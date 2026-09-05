"""ch20-m07 驱动脚本 —— LSE 合并(⊕ 在 (lse,output) 上)的 cascade 可示教轨迹。

跑法(host, 纯 CPU numpy): python run_ch20_m07_lse_merge.py
输出: ch20_m07_lse_merge.json

素材对应 dossier 机制 ch20-m07(LSE 合并 merge_attn_states)与 ch20-m08(cascade 落地,
本 trace 同时给 m08 供数),论文 arXiv:1805.02867 §3.1 Eq.(4) + arXiv:2307.08691
§2.3.1(两块表)/§3.1.1 Tweak 2(L=m+log ℓ);vLLM 落地对照
vllm/v1/attention/backends/flash_attn.py:L1638-L1690(cascade 两段调用)
与 vllm/v1/attention/ops/triton_merge_attn_states.py:L259-L322(六步合并)。

场景: 2 条请求共享 4 token 前缀(vLLM cascade: 前缀段 block_table[:1] 一次算完被
全批复用),各带私有后缀(A 3 token、B 2 token);每条请求取最后 2 个 token 作 query
(prefill 尾两拍)。前缀段 causal=False,后缀段 causal=True + query_offset 对齐。
合并权重非 50/50(两段 lse 不同,非退化)。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from flash_attention import standard_attention  # noqa: E402
from lse_merge import attention_with_lse, merge_lse_states  # noqa: E402


def r(v, nd=4):
    return round(float(v), nd)


def main():
    out = {}
    K_p = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    V_p = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    K_sA = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 2.0]])
    V_sA = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    K_sB = np.array([[1.0, 1.0], [0.0, 2.0]])
    V_sB = np.array([[4.0, 2.0], [2.0, 1.0]])
    q_A = np.array([[1.0, 1.0], [0.0, 1.0]])   # A 的全局位置 5,6
    q_B = np.array([[2.0, 0.0], [1.0, 1.0]])   # B 的全局位置 4,5

    out["params"] = {
        "shared_prefix_len": 4,
        "request_A_suffix_len": 3,
        "request_B_suffix_len": 2,
        "queries_per_request": 2,
        "softmax_scale": 1.0,
        "prefix_segment": "causal=False, 一次调用算全批(vLLM: block_table[:1])",
        "suffix_segment": "causal=True + query_offset(右下对齐; vLLM: block_table[:, num_common_kv_blocks:])",
        "K_prefix": [[1, 0], [0, 1], [1, 1], [2, 0]],
        "V_prefix": [[1, 2], [3, 4], [5, 6], [7, 8]],
        "K_suffix_A": [[0, 1], [1, 0], [1, 2]],
        "V_suffix_A": [[2, 0], [0, 3], [1, 1]],
        "K_suffix_B": [[1, 1], [0, 2]],
        "V_suffix_B": [[4, 2], [2, 1]],
        "q_A": [[1, 1], [0, 1]],
        "q_B": [[2, 0], [1, 1]],
    }

    # ---- 段1: 共享前缀,全批一次算(causal=False) ----
    Q_all = np.vstack([q_A, q_B])
    O_pre, lse_pre = attention_with_lse(Q_all, K_p, V_p, causal=False, scale=1.0)
    out["prefix_segment_one_call"] = {
        "rows": ["A_row0", "A_row1", "B_row0", "B_row1"],
        "lse": [r(v) for v in lse_pre],
        "O": [[[r(x) for x in row] for row in O_pre[i:i + 2]] for i in (0, 2)],
    }

    # ---- 段2: 各请求私有后缀(causal=True + offset 右下对齐) ----
    O_sA, lse_sA = attention_with_lse(q_A, K_sA, V_sA, causal=True, scale=1.0,
                                      query_offset=1)   # 全局 c+4 <= r+5 -> c <= r+1
    O_sB, lse_sB = attention_with_lse(q_B, K_sB, V_sB, causal=True, scale=1.0,
                                      query_offset=0)   # 全局 c+4 <= r+4 -> c <= r
    out["suffix_segment_A"] = {"query_offset": 1, "lse": [r(v) for v in lse_sA],
                               "O": [[r(x) for x in row] for row in O_sA],
                               "note": "row0 只见后缀前 2 个 key(第 3 个是未来),row1 见全部 3 个"}
    out["suffix_segment_B"] = {"query_offset": 0, "lse": [r(v) for v in lse_sB],
                               "O": [[r(x) for x in row] for row in O_sB],
                               "note": "row0 只见后缀第 1 个 key,row1 见全部 2 个"}

    # ---- 合并(⊕ 在 (lse,output) 上) + 与一次性对照 ----
    # 一次性参照: 每请求整段 query 一起算(causal 右下对齐, offset=该请求首 query 的
    # 全局位置: A 的 queries 在 5,6 -> offset=5;B 的在 4,5 -> offset=4),再取对应行
    O_oneA, lse_oneA = attention_with_lse(q_A, np.vstack([K_p, K_sA]), np.vstack([V_p, V_sA]),
                                          causal=True, scale=1.0, query_offset=5)
    O_oneB, lse_oneB = attention_with_lse(q_B, np.vstack([K_p, K_sB]), np.vstack([V_p, V_sB]),
                                          causal=True, scale=1.0, query_offset=4)
    out["one_shot_reference"] = {
        "A": {"query_offset": 5, "lse": [r(v) for v in lse_oneA],
              "O": [[r(x) for x in row] for row in O_oneA]},
        "B": {"query_offset": 4, "lse": [r(v) for v in lse_oneB],
              "O": [[r(x) for x in row] for row in O_oneB]},
    }
    rows = []
    for name, Oa, la, Os, ls, O_one, lse_one in [
        ("A_row0", O_pre[0], lse_pre[0], O_sA[0], lse_sA[0], O_oneA[0], lse_oneA[0]),
        ("A_row1", O_pre[1], lse_pre[1], O_sA[1], lse_sA[1], O_oneA[1], lse_oneA[1]),
        ("B_row0", O_pre[2], lse_pre[2], O_sB[0], lse_sB[0], O_oneB[0], lse_oneB[0]),
        ("B_row1", O_pre[3], lse_pre[3], O_sB[1], lse_sB[1], O_oneB[1], lse_oneB[1]),
    ]:
        tr = []
        O_m, lse_m = merge_lse_states(Oa[None], np.array([la]), Os[None], np.array([ls]), trace=tr)
        t = tr[0]
        rows.append({
            "row": name,
            "prefix_lse": r(la),
            "suffix_lse": r(ls),
            "max_lse": r(t["max_lse"][0]),
            "p_se": r(t["p_se"][0]),
            "s_se": r(t["s_se"][0]),
            "out_se": r(t["out_se"][0]),
            "p_scale_weight": r(t["p_scale"][0]),
            "s_scale_weight": r(t["s_scale"][0]),
            "merged_out_lse": r(t["out_lse"][0]),
            "merged_O": [r(v) for v in O_m[0]],
            "one_shot_O": [r(v) for v in O_one],
            "one_shot_lse": r(lse_one),
            "O_max_abs_diff": r(np.abs(O_m[0] - O_one).max(), 12),
            "O_max_abs_diff_full": float(np.abs(O_m[0] - O_one).max()),
            "O_allclose_1e-12": bool(np.allclose(O_m[0], O_one, rtol=0, atol=1e-12)),
            "lse_abs_diff": r(abs(lse_m[0] - lse_one), 12),
        })
    out["merge_rows"] = rows
    out["all_O_close_to_one_shot_atol_1e-12"] = bool(all(row["O_allclose_1e-12"] for row in rows))

    # ---- 「前缀只算一遍」的量化(m08 素材) ----
    out["prefix_reuse_accounting"] = {
        "full_scan_key_elements_two_requests": 7 + 6,
        "cascade_scan_key_elements": 4 + 3 + 2,
        "saved": (7 + 6) - (4 + 3 + 2),
        "saving_ratio": round(((7 + 6) - (4 + 3 + 2)) / (7 + 6), 4),
        "formula": "R 条请求共享前缀 P、后缀 S_i: 全扫 Σ(P+S_i), cascade 只扫 P+ΣS_i, 省 P·(R-1)",
    }

    p = Path(__file__).parent / "ch20_m07_lse_merge.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"OK {p}")


if __name__ == "__main__":
    main()
