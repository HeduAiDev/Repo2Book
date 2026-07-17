#!/usr/bin/env python3
"""m02 verification: 在线 softmax 标量递推(Milakov & Gimelshein / paper.md §2.2)。

纯 host Python 逐字转写 tutorials/06 内层三件套的标量原型(L57-63 的 m_ij/l_i 递推,
去掉 attention 的 acc 与分块,只留标量 running max/sum)。用来核对 explainer 手推数字。
基底用自然指数 e(与 paper §2.2 公式一致);代码用 exp2 是纯性能变体、数学等价。
"""
import json
import math

xs = [1.0, 3.0, 2.0, 5.0]  # 一行打分,小到可心算

m = float("-inf")  # running max, m_0
d = 0.0            # running sum(分母), d_0
rows = []
for j, x in enumerate(xs, start=1):
    m_prev, d_prev = m, d
    m = max(m_prev, x)
    refreshed = m > m_prev + 1e-12 and not math.isinf(m_prev)  # 是否刷新(排除首步 -inf)
    rescale = 0.0 if math.isinf(m_prev) else math.exp(m_prev - m)
    old_rescaled = d_prev * (0.0 if math.isinf(m_prev) else math.exp(m_prev - m))
    new_term = math.exp(x - m)
    d = old_rescaled + new_term
    rows.append({
        "j": j, "x_j": x, "m_j": m,
        "refreshed_max": bool(m > m_prev + 1e-12 and j > 1),
        "rescale_factor e^(m_prev-m_j)": round(rescale, 6),
        "old_rescaled d_(j-1)*factor": round(old_rescaled, 6),
        "new_term e^(x_j-m_j)": round(new_term, 6),
        "d_j": round(d, 6),
    })

# 交叉验证:一次性(三遍法)分母
mx = max(xs)
d_oneshot = sum(math.exp(x - mx) for x in xs)

out = {
    "input_xs": xs,
    "recurrence_rows": rows,
    "final_running_denominator_d_N": round(d, 6),
    "oneshot_denominator": round(d_oneshot, 6),
    "match": abs(d - d_oneshot) < 1e-9,
    "note": "d_N(在线一遍过) 与 三遍法 逐位相等,证明 running max/sum 递推正确",
}
print(json.dumps(out, ensure_ascii=False, indent=2))
