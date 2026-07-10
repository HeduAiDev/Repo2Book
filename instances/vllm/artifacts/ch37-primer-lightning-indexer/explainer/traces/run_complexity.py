"""驱动脚本：跑参考实现 complexity.py,把"主注意力 O(L^2)->O(Lk)、indexer 自身仍
O(L^2)"这句诚实账换成可代入具体数字。喂给 explainer 的 complexity-honest-account 机制。

小例(L=8,k=2)供逐行心算;大例(L=131072=128k,k=2048)供 worked example 代入真实长上下文。

跑法:python3 run_complexity.py   (纯 Python,host 直接跑)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IMPL = os.path.abspath(os.path.join(HERE, "..", "..", "implementation"))
sys.path.insert(0, IMPL)

from complexity import indexer_ops, main_attention_ops, speedup_ratio  # noqa: E402

INDEXER_COST_RATIO = 0.25  # 少头+FP8/FP4+无反向 -> indexer 每对核算常数远小于主 MLA

# --- 小例:L=8, k=2(读者可逐 t 心算)---
L_small, k_small = 8, 2
small = {
    "L": L_small,
    "k": k_small,
    "main_dense_ops": main_attention_ops(L_small),              # O(L^2)
    "main_sparse_ops": main_attention_ops(L_small, k_small),    # O(Lk)
    "indexer_ops": indexer_ops(L_small, INDEXER_COST_RATIO),    # 仍 O(L^2),常数 0.25x
    "speedup": speedup_ratio(L_small, k_small),
}

# --- 大例:L=128k, k=2048(真实长上下文场景)---
L_big, k_big = 131072, 2048
big = {
    "L": L_big,
    "k": k_big,
    "main_dense_ops": main_attention_ops(L_big),
    "main_sparse_ops": main_attention_ops(L_big, k_big),
    "indexer_ops": indexer_ops(L_big, INDEXER_COST_RATIO),
    "speedup": speedup_ratio(L_big, k_big),
}

out = {"indexer_cost_ratio": INDEXER_COST_RATIO, "small": small, "big": big}
print(json.dumps(out, indent=2, ensure_ascii=False))

with open(os.path.join(HERE, "run_complexity.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
