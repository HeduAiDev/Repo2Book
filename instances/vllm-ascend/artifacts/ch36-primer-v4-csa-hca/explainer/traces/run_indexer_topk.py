"""Driver: csa-lightning-indexer-topk —— lightning indexer 打分(Eq.16)+ top-k 选块(Eq.17)。
2 个 indexer 头(单位基),手造 5 个候选压缩块的 indexer key,展示 ReLU 清零负点积、
每头加权求和、top-k 选出最高分块。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "implementation"))
from lightning_indexer_csa import (index_scores_for_query, topk_sparse_selection)

# 2 个 indexer 头,head_dim=2;单位基让"点积=对应通道分量"便于心算
q_idx_heads = np.array([[1.0, 0.0], [0.0, 1.0]])
w_idx_heads = np.array([1.0, 1.0])
# 5 个候选压缩块的 indexer key(c^I=2)
K_IComp = np.array([
    [2.0, 0.0],    # s0: head0=2 head1=0 -> 2
    [0.0, 3.0],    # s1: head0=0 head1=3 -> 3
    [-1.0, -1.0],  # s2: 两头点积均<0 -> ReLU 清零 -> 0
    [1.0, 1.0],    # s3: 1+1 -> 2
    [4.0, 1.0],    # s4: 4+1 -> 5
])
C_comp = K_IComp.copy()   # 简化:压缩 KV 与 indexer key 同形,仅示教选择

scores = index_scores_for_query(q_idx_heads, w_idx_heads, K_IComp)   # (5,)
K = 2
selected, _ = topk_sparse_selection(scores, C_comp, k=K, causal_limit=5)
selected_set = set(selected.tolist())

rows = []
for s in range(len(scores)):
    dot0 = float(q_idx_heads[0] @ K_IComp[s])
    dot1 = float(q_idx_heads[1] @ K_IComp[s])
    rows.append({
        "block_s": s,
        "head0_dot": round(dot0, 1),
        "head1_dot": round(dot1, 1),
        "relu_head0": round(max(dot0, 0.0), 1),
        "relu_head1": round(max(dot1, 0.0), 1),
        "index_score": round(float(scores[s]), 1),
        "in_topk": 1 if s in selected_set else 0,
    })

out = {"params": {"n_heads_idx": 2, "k": K, "num_candidates": 5},
       "selected_blocks": selected.tolist(),
       "rows": rows}
print(json.dumps(out, indent=2, ensure_ascii=False))
with open(os.path.join(os.path.dirname(__file__), "indexer_topk.json"), "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
