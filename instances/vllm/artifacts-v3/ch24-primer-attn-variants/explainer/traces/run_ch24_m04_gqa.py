"""ch24-m04 驱动脚本 —— GQA-g 插值:分组共享 + mean-pool 转换 + 组数取舍。

跑法(host, 纯 CPU numpy): python run_ch24_m04_gqa.py
输出: ch24_m04_gqa.json(与本脚本同目录)

素材对应 dossier 机制 ch24-m04,论文出处 arXiv:2305.13245:
- §2.2 GQA-g 定义(GQA-1=MQA、GQA-H=MHA;连续组映射 idx//(H/G));
- §2.1 uptraining 第一步(mean-pool 优于选单头/随机初始化)+ α 比例步数;
- §3.1(α=0.05,约 600 TPUv3 chip-days)/§3.3 Fig.6(1→8 组温和、近 MHA 递增,选 8)。
主例 Llama-3-70B:64 query 头/8 KV 头(源自模型 config;dossier m04 note)。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from kv_cache_table import gqa_kv_cache_per_token  # noqa: E402
from mha_gqa import (  # noqa: E402
    attention_forward,
    kv_head_of_query_head,
    mha_to_gqa_mean_pool,
)


def r(v, nd=4):
    return round(float(v), nd)


def main():
    out = {}

    # ---- 主例:Llama-3-70B 64 Q 头 / 8 KV 头 ----
    H70, G70, d70 = 64, 8, 128
    out["display_names"] = {"Llama-3-70B": "主例模型(64 query 头/8 KV 头/128 维)"}
    out["llama3_70b_account"] = {
        "num_query_heads": H70, "num_kv_heads": G70, "head_dim": d70,
        "group_width": H70 // G70,                       # 8 —— 每 8 个 query 头一组
        "mha_cache": gqa_kv_cache_per_token(H70, d70),   # 16384 = 2*64*128
        "gqa8_cache": gqa_kv_cache_per_token(G70, d70),  # 2048 = 2*8*128
        "cache_fraction_of_mha": r(gqa_kv_cache_per_token(G70, d70) / gqa_kv_cache_per_token(H70, d70), 4),  # 0.0625 = 1/8
        "expand": {"MHA": "2*64*128", "GQA-8": "2*8*128"},
    }

    # ---- 组映射:H=8、G=2 的连续分组(Fig.2 画法 / HF-vLLM repeat_kv 同款) ----
    kmap = kv_head_of_query_head(8, 2)
    out["group_mapping_H8_G2"] = {
        "num_heads": 8, "num_groups": 2, "group_width": 4,
        "kv_head_map": [int(x) for x in kmap],           # [0,0,0,0,1,1,1,1]
        "reading": "头 0-3 落组 0 吃 KV0,头 4-7 落组 1 吃 KV1(idx // 组宽)",
    }

    # ---- 玩具 mean-pool 转换(H=4→G=2,d_h=2,d=3;手算整数矩阵) ----
    d_toy, n_h_toy, d_h_toy, G_toy = 3, 4, 2, 2
    # 逐头拼接的 W_K (8,3):head0=[[1,0,0],[0,1,0]], head1=[[3,0,0],[0,3,0]],
    # head2=[[0,0,1],[0,0,1]], head3=[[0,0,3],[0,0,3]] —— 组内平均可心算
    W_K = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 3],
        [0, 0, 3],
    ], dtype=float)
    W_K_g, _ = mha_to_gqa_mean_pool(W_K, W_K.copy(), n_h_toy, d_h_toy, G_toy)
    h_vec = np.array([1.0, 2.0, 3.0])
    k_head0 = W_K[0:2] @ h_vec          # 组 0 成员头 0 的 K
    k_head1 = W_K[2:4] @ h_vec          # 组 0 成员头 1 的 K
    k_group0 = W_K_g[0:2] @ h_vec       # 组 0 mean-pool 头的 K
    out["mean_pool_toy"] = {
        "d": d_toy, "num_heads": n_h_toy, "d_h": d_h_toy, "num_groups": G_toy,
        "h": [1, 2, 3],
        "k_head0": [r(v, 2) for v in k_head0],     # [1,2]
        "k_head1": [r(v, 2) for v in k_head1],     # [3,6]
        "k_group0_mean_pooled": [r(v, 2) for v in k_group0],   # [2,4]
        "mean_check": "(k_head0+k_head1)/2 = [(1+3)/2,(2+6)/2] = [2,4] —— 先投再平均 == 先平均再投",
        "linearity_max_diff": r(float(np.abs((k_head0 + k_head1) / 2 - k_group0).max()), 12),  # 0.0
        "W_K_group0_rows": [[r(v, 2) for v in row] for row in W_K_g[0:2]],  # [[2,0,0],[0,2,0]]
        "W_K_group1_rows": [[r(v, 2) for v in row] for row in W_K_g[2:4]],  # [[0,0,2],[0,0,2]]
        "paper_quote": "§2.2 'construct each group key and value head by mean-pooling all the original heads within that group'(优于选单头/随机初始化,§2.1)",
    }

    # ---- 端点与前向共享核验(H=4:G=4 恒等映射 / G=2 组内共享 / G=1 全共享) ----
    T, d = 2, 6
    rng = np.random.default_rng(23)
    h_ep = rng.standard_normal((T, d))
    WQ = rng.standard_normal((4 * 2, d))
    WK_full = rng.standard_normal((4 * 2, d))       # 4 个 KV 头(MHA 配置)
    WV_full = rng.standard_normal((4 * 2, d))
    WO = rng.standard_normal((4 * 2, 4 * 2))
    tr4, tr2, tr1 = {}, {}, {}
    attention_forward(h_ep, WQ, WK_full, WV_full, WO, num_heads=4, num_kv_heads=4, trace=tr4)
    attention_forward(h_ep, WQ, WK_full[:4], WV_full[:4], WO, num_heads=4, num_kv_heads=2, trace=tr2)
    attention_forward(h_ep, WQ, WK_full[:2], WV_full[:2], WO, num_heads=4, num_kv_heads=1, trace=tr1)
    out["endpoint_check"] = {
        "map_G4": [int(x) for x in kv_head_of_query_head(4, 4)],   # [0,1,2,3] 恒等 → MHA
        "map_G2": [int(x) for x in kv_head_of_query_head(4, 2)],   # [0,0,1,1]
        "map_G1": [int(x) for x in kv_head_of_query_head(4, 1)],   # [0,0,0,0] 常值 → MQA
        "gqa2_same_group_k_diff": r(float(np.abs(tr2["k_per_query_head"][:, 0] - tr2["k_per_query_head"][:, 1]).max()), 6),  # 0.0 组内共享同一份
        "gqa2_cross_group_k_diff": r(float(np.abs(tr2["k_per_query_head"][:, 0] - tr2["k_per_query_head"][:, 2]).max()), 6),  # >0 跨组不同
        "gqa2_cross_group_k_diff_value": r(float(np.abs(tr2["k_per_query_head"][:, 0] - tr2["k_per_query_head"][:, 2]).max()), 6),
        "mqa_all_share_diff": r(float(np.abs(tr1["k_per_query_head"][:, 0] - tr1["k_per_query_head"][:, 3]).max()), 6),  # 0.0
        "kv_elements_G4": 2 * 4 * 2,   # 16
        "kv_elements_G2": 2 * 2 * 2,   # 8
        "kv_elements_G1": 2 * 1 * 2,   # 4
        "note": "G=H 恒等映射(每头自己的 KV)、G=1 常值映射(全员一个 KV)——谱系端点在映射与元素账两头合拢",
    }

    # ---- 组数取舍 + uptraining(论文常数,quote 进 trace 供表引用) ----
    out["group_count_tradeoff"] = {
        "selected_groups": 8,
        "fig6_quote": "§3.3 Fig.6:从 1(MQA)加到 8 组只有温和推理开销,越靠近 MHA 代价递增;'We selected 8 groups as a favorable middle ground'",
        "llama_note": "LLaMA-2/3 的 8 KV 头出处即此",
        "uptraining_alpha": 0.05,
        "uptraining_alpha_quote": "§2.1 'pre-trained for a small proportion α of its original training steps';主实验 α=0.05",
        "uptraining_cost_tpuv3_chip_days": 600,
        "uptraining_cost_quote": "§3.1 'For α=0.05, training took approximately 600 TPUv3 chip-days'",
        "gqa_scaling_note": "§2.2 大模型 KV cache 随模型维线性、FLOPs 随平方——GQA 保持带宽/容量随规模同比缩",
    }

    p = Path(__file__).resolve().parent / "ch24_m04_gqa.json"
    with open(p, "w", newline="\n", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
