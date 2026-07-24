#!/usr/bin/env python3
"""ch32 online-softmax 递推的 host 数值复现（教学素材真相源）。

本章 kind=skip_impl：真 kernel 06-fused-attention.py 需 CANN/NPU 才能编译真跑，
host 无 NPU。但 _attn_fwd_inner 的在线 softmax 递推是纯标量/矩阵算术，与 NPU 无关——
这里用 numpy 忠实复现该递推（严格照搬源码的初值与更新顺序：m_i=-inf, l_i=1.0,
alpha=exp(m_i-m_ij), l_i=l_i*alpha+l_ij, acc=acc*alpha+p·V），并与「一次性物化整张
softmax(QK^T·scale)·V」直接结果对拍，验证在线算法数值等价。

对应源码：06-fused-attention.py L95-L120（更新段）、L211-L212（m/l 初值）、L247-L249（收尾归一）。
产物 run_online_softmax.json 供 explainer.json worked_example 的每个数字溯源。
纯 host，无第三方依赖（除 numpy）。
"""
import json
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# --- 教学参数：单查询行(BLOCK_M=1) / HEAD_DIM=2 / BLOCK_N=2 / 两个 K/V 块(N_CTX=4) ---
# sm_scale=1.0 为便于心算的教学取值（真实 test_06 用 0.5）——已在 explainer 标注。
# 走非因果全序列路径(内层 STAGE=3, 无掩码)，两块合起来即完整 softmax。
q = np.array([1.0, 0.0], dtype=np.float64)          # 查询向量 [HEAD_DIM=2]
sm_scale = 1.0

# 两个 K/V 块，每块 BLOCK_N=2 列。K 行 = key 向量；V 行 = value 向量。
K_blocks = [
    np.array([[1.0, 0.0], [0.0, 1.0]]),   # 块0: k0,k1
    np.array([[2.0, 0.0], [0.5, 1.0]]),   # 块1: k2,k3
]
V_blocks = [
    np.array([[1.0, 0.0], [0.0, 1.0]]),   # 块0: v0,v1
    np.array([[1.0, 1.0], [2.0, 0.0]]),   # 块1: v2,v3
]

# --- 在线 softmax（严格照搬 _attn_fwd_inner 的初值与更新序） ---
m_i = -np.inf          # running max, 源码 L211: m_i = zeros - inf
l_i = 1.0              # running sum, 源码 L212: l_i = zeros + 1.0（首块 alpha=0 会清掉）
acc = np.zeros(2)      # 输出累加器 [HEAD_DIM=2]

trace = {"params": {"q": q.tolist(), "sm_scale": sm_scale,
                    "K_blocks": [b.tolist() for b in K_blocks],
                    "V_blocks": [b.tolist() for b in V_blocks],
                    "m_init": "-inf", "l_init": 1.0}, "rounds": []}

for j, (K, V) in enumerate(zip(K_blocks, V_blocks)):
    # QK^T（Cube 段）：q 与该块每个 key 的点积，再乘 scale
    qk_raw = K @ q                      # [BLOCK_N]  = [q·k for k in block]
    qk = qk_raw * sm_scale
    # softmax 段（Vector 段）：running max → 稳定化 → exp → sum
    blk_max = qk.max()
    m_ij = max(m_i, blk_max)            # 源码 L99: m_ij = maximum(m_i, max(qk))
    qk_stab = qk - m_ij                 # 减 max 稳定化
    p = np.exp(qk_stab)                 # 源码 L103: p = exp(qk)
    l_ij = p.sum()                      # 源码 L113: l_ij = sum(p)
    alpha = np.exp(m_i - m_ij) if m_i != -np.inf else 0.0  # 源码 L115: alpha=exp(m_i-m_ij)
    # PV（Cube 段）+ 在线更新（源码 L116,L119-L120）
    l_i = l_i * alpha + l_ij
    acc = acc * alpha + p @ V           # acc = acc*alpha + p·V
    m_prev = m_i
    m_i = m_ij                          # 源码 L139
    trace["rounds"].append({
        "block": j,
        "qk_raw": [round(float(x), 4) for x in qk_raw],
        "qk_scaled": [round(float(x), 4) for x in qk],
        "block_max": round(float(blk_max), 4),
        "m_prev": ("-inf" if m_prev == -np.inf else round(float(m_prev), 4)),
        "m_ij": round(float(m_ij), 4),
        "alpha": round(float(alpha), 4),
        "p": [round(float(x), 4) for x in p],
        "l_ij": round(float(l_ij), 4),
        "l_i_after": round(float(l_i), 4),
        "acc_after": [round(float(x), 4) for x in acc],
    })

# 收尾归一（源码 L247-L249）：logsumexp 与 acc/l_i
lse = m_i + np.log(l_i)
out_online = acc / l_i
trace["finalize"] = {
    "m_i_final": round(float(m_i), 4),
    "l_i_final": round(float(l_i), 4),
    "logsumexp": round(float(lse), 4),
    "out": [round(float(x), 4) for x in out_online],
}

# --- 参考：一次性物化整张 softmax（对拍，验证在线等价） ---
K_all = np.vstack(K_blocks)
V_all = np.vstack(V_blocks)
scores = (K_all @ q) * sm_scale
w = np.exp(scores - scores.max())
w = w / w.sum()
out_ref = w @ V_all
trace["reference"] = {
    "scores": [round(float(x), 4) for x in scores],
    "softmax_weights": [round(float(x), 4) for x in w],
    "out_ref": [round(float(x), 4) for x in out_ref],
    "max_abs_diff_online_vs_ref": round(float(np.abs(out_online - out_ref).max()), 8),
}

print(json.dumps(trace, ensure_ascii=False, indent=2))
with open(__file__.replace("run_online_softmax.py", "run_online_softmax.json"), "w",
          encoding="utf-8") as f:
    json.dump(trace, f, ensure_ascii=False, indent=2)
