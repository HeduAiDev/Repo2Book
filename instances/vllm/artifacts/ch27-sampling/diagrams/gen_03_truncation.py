#!/usr/bin/env python3
"""top-k/top-p 截断（pytorch sort 路径）逐列状态演化。"""
import xml.sax.saxutils as xs
import math

def esc(s):
    return xs.escape(s)

# 一个 6-token 小例子。logits 原始顺序
toks = ["A", "B", "C", "D", "E", "F"]
logits = [3.0, 1.0, 4.0, 0.0, 2.0, -1.0]
k = 3  # top-k=3
p = 0.8  # top-p=0.8

# 升序排序
order = sorted(range(len(logits)), key=lambda i: logits[i])  # ascending
sorted_log = [logits[i] for i in order]
# top-k: 第 (V-k) 大阈值；升序中索引 V-k 处的值
V = len(logits)
thr_idx = V - k  # 3
thr_val = sorted_log[thr_idx]
topk_keep = [sl >= thr_val for sl in sorted_log]  # mask: < thr → -inf
# 应用 topk 后
after_k = [sl if keep else float('-inf') for sl, keep in zip(sorted_log, topk_keep)]
# softmax over after_k
exps = [math.exp(x) if x != float('-inf') else 0.0 for x in after_k]
Z = sum(exps)
probs = [e/Z for e in exps]
cum = []
s = 0.0
for pr in probs:
    s += pr
    cum.append(s)
# top-p mask: cumsum <= 1-p → drop (但至少留最后一个=最大)
topp_drop = [c <= (1-p) for c in cum]
topp_drop[-1] = False
after_p = [a if not d else float('-inf') for a, d in zip(after_k, topp_drop)]

def fmt(x):
    if x == float('-inf'):
        return "-inf"
    return f"{x:.2f}" if isinstance(x, float) and abs(x-round(x)) > 1e-6 else f"{x:.1f}"

cols = [
    ("token (升序后)", [toks[i] for i in order], None),
    ("logits 升序", [fmt(x) for x in sorted_log], None),
    ("top-k mask", ["保留" if keep else "→-inf" for keep in topk_keep], topk_keep),
    ("after top-k", [fmt(x) for x in after_k], None),
    ("probs", [f"{pr:.3f}" for pr in probs], None),
    ("cumsum", [f"{c:.3f}" for c in cum], None),
    ("top-p mask", ["→-inf" if d else "保留" for d in topp_drop], [not d for d in topp_drop]),
    ("最终 logits", [fmt(x) for x in after_p], [x != float('-inf') for x in after_p]),
]

ncol = len(cols)
nrow = V
cw = 122
ch = 44
x0 = 30
y0 = 90
hh = 40
W = x0*2 + cw*ncol
H = y0 + hh + ch*nrow + 70

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="17" font-weight="bold" '
         f'fill="#1e293b">top-k=3, top-p=0.8 截断：sort → 两次 mask → scatter 回原位</text>')
L.append(f'<text x="{W/2}" y="58" text-anchor="middle" font-family="sans-serif" font-size="13" '
         f'fill="#64748b">vocab=6，原始 logits A:3 B:1 C:4 D:0 E:2 F:-1（升序排列后逐列演化）</text>')

# 表头
for j, (title, vals, hl) in enumerate(cols):
    x = x0 + j*cw
    L.append(f'<rect x="{x}" y="{y0}" width="{cw}" height="{hh}" fill="#1e293b"/>')
    L.append(f'<text x="{x+cw/2}" y="{y0+hh/2+5}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="white">{esc(title)}</text>')

# 单元格
for j, (title, vals, hl) in enumerate(cols):
    x = x0 + j*cw
    for i in range(nrow):
        y = y0 + hh + i*ch
        fill = "#f8fafc"
        if hl is not None:
            fill = "#dcfce7" if hl[i] else "#fee2e2"
        L.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="{fill}" stroke="#cbd5e1" stroke-width="1"/>')
        txt = vals[i]
        col = "#b91c1c" if txt in ("-inf", "→-inf") else "#1e293b"
        L.append(f'<text x="{x+cw/2}" y="{y+ch/2+5}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="13" fill="{col}">{esc(txt)}</text>')

L.append(f'<text x="{x0}" y="{H-30}" font-family="sans-serif" font-size="12.5" fill="#475569">'
         f'绿=保留 · 红=置 -inf。top-k 砍掉升序前 V-k 个（最小的）；top-p 在剩下里砍累积概率落在 1-p 以下的，但 [:,-1]=False 保证至少留最大那个。</text>')

L.append('</svg>')
open("03-truncation-trace.svg", "w", encoding="utf-8").write('\n'.join(L))
print("ok")
