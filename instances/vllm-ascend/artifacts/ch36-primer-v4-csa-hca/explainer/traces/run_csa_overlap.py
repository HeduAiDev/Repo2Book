"""Driver: csa-overlap-compress —— CSA 重叠 2m 压缩(Eq.11-12)。
n=8, m=4, c=2, 恒等投影(C=Z=H),位置偏置=0,让 softmax 权重与 token 值直接对应,便于心算。
展示:块0 只用自己的 4 个 token(前 4 位被 -inf padding 清零),块1 借上一块 b 值+自己 a 值
共 2m=8 个 token 参与,但相邻块索引交叠 → 净压缩率仍是 1/m。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "implementation"))
from csa_compression import (csa_compress_sequence, overlap_transform,
                             softmax_over_positions)

n, m, c = 8, 4, 2
H = np.array([[float(t + 1), float(t + 1)] for t in range(n)])   # token t 值 = t+1
I2 = np.eye(2)
Ba = np.zeros((m, c))
Bb = np.zeros((m, c))

# 完整压缩
C_comp = csa_compress_sequence(H, I2, I2, I2, I2, Ba, Bb, m)

# 拆看每块窗口(通道0)与 softmax 权重
Z_win = overlap_transform(H, H, m, pad_value=-np.inf)      # (num_blocks, 2m, c)
S = softmax_over_positions(Z_win)                          # 权重

rows = []
num_blocks = n // m
for i in range(num_blocks):
    ch0_window_vals = [None if np.isneginf(v) else round(float(v), 1) for v in Z_win[i, :, 0]]
    ch0_weights = [round(float(w), 3) for w in S[i, :, 0]]
    n_nonzero = int(np.sum(S[i, :, 0] > 1e-9))
    rows.append({
        "block": i,
        "window_ch0_C_values": ch0_window_vals,   # None = -inf padding(权重 0)
        "softmax_weights_ch0": ch0_weights,
        "n_participating_tokens": n_nonzero,       # 实际非零参与数
        "C_comp_ch0": round(float(C_comp[i, 0]), 3),
    })

out = {"params": {"n": n, "m": m, "c": c},
       "note": "块0 前 4 位为 -inf padding(权重 0);块1 的前 4 位借自块0(overlap)",
       "rows": rows}
print(json.dumps(out, indent=2, ensure_ascii=False))
with open(os.path.join(os.path.dirname(__file__), "csa_overlap.json"), "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
