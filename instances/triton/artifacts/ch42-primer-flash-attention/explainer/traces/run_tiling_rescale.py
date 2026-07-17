#!/usr/bin/env python3
"""m03/m04/m07 verification: attention 版三件套 + rescale 恒等性 + epilogue LSE。

纯 host Python 逐字转写 tutorials/06 _attn_fwd_inner 内层循环(L46-74)的三件套更新式 +
epilogue(L185-186),把一行 Q 对 4 个 K/V 切成 2 块增量更新,与全矩阵 softmax 对照。
基底用自然指数 e(与 paper §2.3 公式一致);代码用 exp2 是纯性能变体、数学等价。
用来核对 explainer 手推数字(m03 三件套逐块、m04 带/不带 rescale 对照、m07 LSE 存储)。
"""
import json
import math


def matvec(P, Vblock):
    # P: list[float] (BLOCK_N,), Vblock: list[list[float]] (BLOCK_N, HEAD_DIM)
    hd = len(Vblock[0])
    out = [0.0] * hd
    for p, v in zip(P, Vblock):
        for c in range(hd):
            out[c] += p * v[c]
    return out


# ---- 小而具体的输入(可心算) ----
q = [1.0, 0.0]                       # 一行 Q, HEAD_DIM=2
K = [[1, 0], [0, 1], [1, 1], [2, 0]]  # 4 个 key
V = [[1, 0], [0, 1], [1, 1], [2, 2]]  # 4 个 value
scale = 1.0
scores = [scale * sum(qi * ki for qi, ki in zip(q, k)) for k in K]  # [1,0,1,2]

# ---- 参照:全矩阵 softmax(黄金标准) ----
mx = max(scores)
w = [math.exp(s - mx) for s in scores]
Z = sum(w)
O_full = [sum(w[i] * V[i][c] for i in range(4)) / Z for c in range(2)]

# ---- 分块在线(2 块,每块 2 个 K/V) ----
blocks = [([scores[0], scores[1]], [V[0], V[1]]),
          ([scores[2], scores[3]], [V[2], V[3]])]

m_i = float("-inf")
l_i = 0.0
acc = [0.0, 0.0]
block_rows = []
for jb, (S, Vb) in enumerate(blocks, start=1):
    m_prev = m_i
    m_i = max(m_prev, max(S))
    alpha = 0.0 if math.isinf(m_prev) else math.exp(m_prev - m_i)
    P = [math.exp(s - m_i) for s in S]
    l_ij = sum(P)
    l_i = l_i * alpha + l_ij
    acc = [alpha * a for a in acc]
    pv = matvec(P, Vb)
    acc = [acc[c] + pv[c] for c in range(2)]
    block_rows.append({
        "block": jb, "S_block": S, "rowmax(S)": max(S),
        "m_i(running max)": round(m_i, 6),
        "alpha=e^(m_prev-m_i)": round(alpha, 6),
        "P~=e^(S-m_i)": [round(p, 6) for p in P],
        "l_ij=rowsum(P~)": round(l_ij, 6),
        "l_i(after rescale+add)": round(l_i, 6),
        "acc(unnormalized)": [round(a, 6) for a in acc],
    })

O_blocked = [a / l_i for a in acc]

# ---- m07: epilogue LSE(每行只存 1 个标量) ----
# 自然形态(paper §2.3 直觉):LSE = m + ln(l) = ln(sum e^{scores})
lse_natural = m_i + math.log(l_i)
lse_check = math.log(sum(math.exp(s) for s in scores))
# 代码形态(L163 qk_scale*=1/ln2 使 m_i 是"基-2 缩放后"的 running max,L185 再 += log2(l_i)):
INV_LN2 = 1.44269504  # 06-fused-attention.py:L163
m_i_code = m_i * INV_LN2               # 代码里 running max 走的是 qk*qk_scale 的量纲
lse_base2_code = m_i_code + math.log2(l_i)  # = 代码 L185 存入 M 的值
lse_base2_check = math.log2(sum(math.exp(s) for s in scores))  # 应 == log2(sum e^s)

# ---- m04: 漏掉 rescale 的错误对照(alpha 不乘) ----
m_w = float("-inf")
l_w = 0.0
acc_w = [0.0, 0.0]
for S, Vb in blocks:
    m_prev = m_w
    m_w = max(m_prev, max(S))
    P = [math.exp(s - m_w) for s in S]
    l_w = l_w + sum(P)            # BUG: 不乘 alpha
    pv = matvec(P, Vb)
    acc_w = [acc_w[c] + pv[c] for c in range(2)]  # BUG: 不乘 alpha
O_wrong = [a / l_w for a in acc_w]

out = {
    "inputs": {"q": q, "K": K, "V": V, "scale": scale, "scores": scores},
    "full_matrix": {
        "rowmax": mx, "weights_unnorm": [round(x, 6) for x in w],
        "Z_denominator": round(Z, 6), "O_full": [round(x, 6) for x in O_full],
    },
    "blocked_online": {
        "rows": block_rows,
        "final_l_i": round(l_i, 6),
        "final_acc_unnorm": [round(a, 6) for a in acc],
        "O_blocked_after_norm": [round(x, 6) for x in O_blocked],
    },
    "identity_check": {
        "O_blocked == O_full": all(abs(a - b) < 1e-9 for a, b in zip(O_blocked, O_full)),
        "l_i == Z": abs(l_i - Z) < 1e-9,
    },
    "epilogue_lse": {
        "final_m_i(natural)": round(m_i, 6),
        "final_l_i": round(l_i, 6),
        "LSE_natural = m + ln(l)": round(lse_natural, 6),
        "LSE_check = ln(sum e^s)": round(lse_check, 6),
        "match_natural": abs(lse_natural - lse_check) < 1e-9,
        "m_i_code = m*1.44269504 [L163 scale]": round(m_i_code, 6),
        "M_stored_code = m_i_code + log2(l) [L185]": round(lse_base2_code, 6),
        "M_check = log2(sum e^s)": round(lse_base2_check, 6),
        "match_code": abs(lse_base2_code - lse_base2_check) < 1e-9,
        "stored_scalars_per_row": 1,
        "naive_stored_per_row": len(scores),
    },
    "without_rescale_bug": {
        "l_wrong": round(l_w, 6),
        "acc_wrong_unnorm": [round(a, 6) for a in acc_w],
        "O_wrong": [round(x, 6) for x in O_wrong],
        "note": "漏乘 alpha -> O 偏离正确值,证明 rescale 是恒等性的全部",
    },
}
print(json.dumps(out, ensure_ascii=False, indent=2))
