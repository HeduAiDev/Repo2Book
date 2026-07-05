"""Driver: motivation-kv-flops-account —— 用账本模型演示 dense 注意力的 O(L) 单 token 税
vs CSA 的有界核注意力。head_dim=1 让 FLOPs 代理数 = 条目数,便于心算。"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "implementation"))
from efficiency_account import dense_baseline_layer_cost, csa_layer_cost

HEAD_DIM = 1
M, K, N_WIN = 4, 2, 1
rows = []
for L in (16, 64, 256, 1024):
    dense = dense_baseline_layer_cost(L, HEAD_DIM)
    csa = csa_layer_cost(L, m=M, k=K, n_win=N_WIN, head_dim=HEAD_DIM,
                         indexer_heads=1, indexer_dim=1)
    core_flops = (K + N_WIN) * HEAD_DIM          # 有界:与 L 无关
    indexer_flops = (L / M) * 1 * 1              # indexer 扫全部候选块
    rows.append({
        "L": L,
        "dense_kv": round(dense.kv_entries_stored, 1),
        "dense_flops": round(dense.flops_per_token, 1),
        "csa_kv": round(csa.kv_entries_stored, 1),
        "csa_core_flops": round(core_flops, 1),
        "csa_indexer_flops": round(indexer_flops, 1),
        "csa_total_flops": round(csa.flops_per_token, 1),
    })

out = {"params": {"head_dim": HEAD_DIM, "m": M, "k": K, "n_win": N_WIN}, "rows": rows}
print(json.dumps(out, indent=2, ensure_ascii=False))
with open(os.path.join(os.path.dirname(__file__), "motivation.json"), "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
