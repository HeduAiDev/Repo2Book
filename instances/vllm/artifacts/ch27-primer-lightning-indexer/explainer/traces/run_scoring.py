"""驱动脚本：跑参考实现的 lightning indexer 打分函数(Eq.1)与 top-k 选块(Eq.2),
产出可示教的数值轨迹。喂给 explainer 的 index-score-formula / topk-selection 两个机制。

参数刻意选小(H^I=2 头,d^I=2,S=4 历史 token,T=2 query),读者可心算对拍。
ReLU 分支被真实触发:多个 q·k 点积为负被 ReLU 归零(非退化)。

跑法:python3 run_scoring.py   (纯 NumPy,host 直接跑)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMPL = os.path.abspath(os.path.join(HERE, "..", "..", "implementation"))
sys.path.insert(0, IMPL)

from lightning_indexer import index_score, topk_select  # noqa: E402
from wiring import TopkIndicesBuffer, v32_indexer_step  # noqa: E402

# --- 小而具体的参数 ---
# q: [T=2, H^I=2, d^I=2]   两个 query token、各 2 个 indexer 头、头维 2
q = np.array(
    [
        [[1.0, 0.0], [0.0, 1.0]],   # t0: head0=[1,0], head1=[0,1]
        [[2.0, 0.0], [0.0, 1.0]],   # t1: head0=[2,0], head1=[0,1]
    ]
)
# k: [S=4, d^I=2]  4 个历史 token 的 indexer key(MQA 式跨头共享)
k = np.array(
    [
        [1.0, 1.0],    # s0
        [-1.0, 2.0],   # s1
        [2.0, -3.0],   # s2
        [0.0, 0.0],    # s3
    ]
)
# w: [T=2, H^I=2]  逐头标量权重
w = np.array(
    [
        [1.0, 2.0],   # t0
        [1.0, 1.0],   # t1
    ]
)
TOPK = 2
BUFFER_TOPK = 3  # buffer 宽度 > k,演示 -1 填充

# --- Eq.(1) 逐步拆解(供教学表逐行核对)---
dots = np.einsum("thd,sd->ths", q, k)   # q_{t,j}^I . k_s^I
relu = np.maximum(dots, 0.0)            # ReLU
I = index_score(q, k, w)               # sum_j w * relu(...)

# --- Eq.(2) top-k 选块 + 写共享 buffer(纯副作用)---
buf = TopkIndicesBuffer(num_tokens=2, topk=BUFFER_TOPK)
idx0 = topk_select(I, TOPK)
# 用 wiring.v32_indexer_step 把"打分->选块->写 buffer"整条副作用链跑一遍
scores_side = v32_indexer_step(q, k, w, buf, TOPK, token_start=0)

out = {
    "params": {"T": 2, "H_I": 2, "d_I": 2, "S": 4, "topk": TOPK, "buffer_topk": BUFFER_TOPK},
    "q": q.tolist(),
    "k": k.tolist(),
    "w": w.tolist(),
    "dots_qk": dots.tolist(),          # [T,H,S] 逐头点积(ReLU 前)
    "relu_qk": relu.tolist(),          # [T,H,S] ReLU 后
    "index_scores_I": I.tolist(),      # [T,S]
    "topk_idx": idx0.tolist(),         # [T,k]
    "topk_buffer": buf.data.tolist(),  # [T, buffer_topk] 含 -1 填充
    "scores_side_channel_equal": bool(np.allclose(scores_side, I)),
}
print(json.dumps(out, indent=2, ensure_ascii=False))

with open(os.path.join(HERE, "run_scoring.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
