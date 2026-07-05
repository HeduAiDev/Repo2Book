"""Driver: mhc-manifold-hyperconnections —— Sinkhorn-Knopp(Eq.8)把残差映射矩阵 B_l
迭代投影到双随机矩阵流形。展示随迭代次数增加,行/列和偏离 1 的最大值单调趋 0。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "implementation"))
from mhc_sinkhorn import sinkhorn_knopp, is_doubly_stochastic, hc_residual_update

B_tilde = np.array([[0.5, 0.2], [0.1, 0.9]])

rows = []
for iters in (0, 1, 3, 20):
    M = sinkhorn_knopp(B_tilde, iters=iters)
    row_dev = float(np.max(np.abs(M.sum(axis=1) - 1.0)))
    col_dev = float(np.max(np.abs(M.sum(axis=0) - 1.0)))
    rows.append({
        "iters": iters,
        "max_row_dev": round(row_dev, 4),
        "max_col_dev": round(col_dev, 4),
        "doubly_stochastic": 1 if is_doubly_stochastic(M) else 0,
    })

# HC 残差更新数值示例(n_hc=2, d=2):B_l 用收敛后的双随机矩阵
B_l = sinkhorn_knopp(B_tilde, iters=20)
A_l = np.array([[0.6, 0.4]])
C_l = np.array([[0.5], [0.5]])
X_l = np.array([[1.0, 2.0], [3.0, 4.0]])
F_out = np.array([[10.0, 10.0]])
X_next = hc_residual_update(X_l, A_l, B_l, C_l, F_out)
residual_example = {
    "X_next_00": round(float(X_next[0, 0]), 3),
    "X_next_11": round(float(X_next[1, 1]), 3),
}

out = {"B_tilde": [[0.5, 0.2], [0.1, 0.9]], "sinkhorn_rows": rows,
       "residual_update": residual_example}
print(json.dumps(out, indent=2, ensure_ascii=False))
with open(os.path.join(os.path.dirname(__file__), "mhc_sinkhorn.json"), "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
