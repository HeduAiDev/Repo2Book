"""Driver: efficiency-account-27-10 —— 用账本模型逐项算 CSA/HCA/dense 三种层的 KV 存量与
单 token FLOPs,取混合平均,与 dense/DSA 基线比。诚实声明:这是账本模型,不复现论文的
27%/10%(那需 DeepSeek 未公开的完整逐层配置);示意参数只验证 hybrid << 两条基线。"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "implementation"))
from efficiency_account import (csa_layer_cost, hca_layer_cost,
                                dense_baseline_layer_cost,
                                mixed_precision_kv_bytes, pure_bf16_kv_bytes,
                                worked_example_efficiency)

L = 1_000_000
HEAD_DIM = 128
K, N_WIN = 2048, 1024

csa = csa_layer_cost(L, m=4, k=K, n_win=N_WIN, head_dim=HEAD_DIM, indexer_heads=4, indexer_dim=64)
hca = hca_layer_cost(L, m_prime=128, n_win=N_WIN, head_dim=HEAD_DIM)
dense = dense_baseline_layer_cost(L, HEAD_DIM)

per_layer = {
    "csa":   {"kv": round(csa.kv_entries_stored, 1),   "flops": round(csa.flops_per_token, 1)},
    "hca":   {"kv": round(hca.kv_entries_stored, 1),   "flops": round(hca.flops_per_token, 1)},
    "dense": {"kv": round(dense.kv_entries_stored, 1), "flops": round(dense.flops_per_token, 1)},
}

res = worked_example_efficiency(seq_len=L, k=K, n_win=N_WIN, head_dim=HEAD_DIM,
                                indexer_heads=4, indexer_dim=64, rope_dims=64)
ratios = {
    "hybrid_avg_kv": round(res.hybrid.kv_entries_stored, 1),
    "hybrid_avg_flops": round(res.hybrid.flops_per_token, 1),
    "flops_ratio_vs_dense": round(res.flops_ratio_vs_dense, 4),
    "kv_ratio_vs_dense": round(res.kv_ratio_vs_dense, 4),
    "flops_ratio_vs_dsa": round(res.flops_ratio_vs_dsa, 4),
    "kv_ratio_vs_dsa": round(res.kv_ratio_vs_dsa, 4),
}
precision = {
    "kv_bytes_mixed_per_entry": round(mixed_precision_kv_bytes(1.0, 64, 64), 1),
    "kv_bytes_pure_bf16_per_entry": round(pure_bf16_kv_bytes(1.0, 128), 1),
    "mixed_vs_bf16_ratio": round(mixed_precision_kv_bytes(1.0, 64, 64) / pure_bf16_kv_bytes(1.0, 128), 3),
}

out = {
    "params": {"L": L, "head_dim": HEAD_DIM, "k": K, "n_win": N_WIN,
               "illustrative_compress_ratios": "([4,4,4,128] * 9)"},
    "per_layer_cost": per_layer,
    "hybrid_and_ratios": ratios,
    "precision_account": precision,
    "paper_stated": {"pro_flops_pct": 27, "pro_kv_pct": 10, "flash_flops_pct": 10, "flash_kv_pct": 7},
    "honesty": "账本模型口径,示意参数;非论文 27%/10% 的复现",
}
print(json.dumps(out, indent=2, ensure_ascii=False))
with open(os.path.join(os.path.dirname(__file__), "efficiency.json"), "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
