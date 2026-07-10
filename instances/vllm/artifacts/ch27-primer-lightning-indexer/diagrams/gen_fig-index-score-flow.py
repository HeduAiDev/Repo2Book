#!/usr/bin/env python3
"""fig-index-score-flow: flow 模板。
展示 index score I_{t,s} 的三步装配:逐头 q·k 点积 -> ReLU 截负 -> 头权重加权求和。
数据来自 traces/run_scoring.json(t0 一行),ReLU 把负值截零的位置高亮标出。
全坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

# ---- 数据(来自 traces/run_scoring.json,严格取 t0 一行) ----
S_LABELS = ["s0", "s1", "s2", "s3"]
HEADS = [
    {
        "name": "head0",
        "dot": [1.0, -1.0, 2.0, 0.0],
        "relu": [1.0, 0.0, 2.0, 0.0],
        "w": 1.0,
        "weighted": [1.0, 0.0, 2.0, 0.0],
    },
    {
        "name": "head1",
        "dot": [1.0, 2.0, -3.0, 0.0],
        "relu": [1.0, 2.0, 0.0, 0.0],
        "w": 2.0,
        "weighted": [2.0, 4.0, 0.0, 0.0],
    },
]
I_T0 = [3.0, 4.0, 2.0, 0.0]

def fmt_vec(v):
    return "[" + ", ".join(str(int(x)) if x == int(x) else str(x) for x in v) + "]"

# ---- 版式常量 ----
COL_W = 250
COL_GAP = 26
ROW_H = 108
ROW_GAP = 46
PAD_L, PAD_T = 78, 78
N_COLS = 3  # 点积 / ReLU / 加权
FINAL_COL_W = 210

W = PAD_L + N_COLS * (COL_W + COL_GAP) + FINAL_COL_W + 60
H = PAD_T + 2 * ROW_H + ROW_GAP + 90

def box(x, y, w, h, fill, stroke, rx=10, sw=1.5):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

def text(x, y, s, size=13, anchor="middle", weight="normal", fill="#0f172a", family="sans-serif"):
    fw = f' font-weight="{weight}"' if weight != "normal" else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{family}" '
            f'font-size="{size}"{fw} fill="{fill}">{esc(s)}</text>')

def arrow(x1, y1, x2, y2, color="#64748b", sw=1.8, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}" marker-end="url(#a)"{d}/>')

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
         '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
         'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
         '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
         'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0f766e"/></marker>'
         '</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(text(W / 2, 30, "index score I_{t0,:} 的装配(t0 · head0/head1)", size=16, weight="bold"))
L.append(text(W / 2, 50, "三步:逐头 q·k 点积 → ReLU 截负 → 头权重加权求和", size=12.5, fill="#475569"))

col_titles = ["① q·k 点积", "② ReLU 截负", "③ × 头权重 w"]
col_x = [PAD_L + c * (COL_W + COL_GAP) for c in range(N_COLS)]
for c in range(N_COLS):
    L.append(text(col_x[c] + COL_W / 2, PAD_T - 14, col_titles[c], size=13, weight="bold", fill="#334155"))

row_y = [PAD_T, PAD_T + ROW_H + ROW_GAP]

for r, head in enumerate(HEADS):
    y = row_y[r]
    L.append(text(PAD_L - 16, y + ROW_H / 2 + 5, head["name"], size=13, weight="bold",
                   anchor="end", fill="#0f172a"))
    # 列1: q·k 点积(负值高亮红框)
    x0 = col_x[0]
    L.append(box(x0, y, COL_W, ROW_H, "#eef2ff", "#6366f1"))
    L.append(text(x0 + COL_W / 2, y + 22, f"t0 · {head['name']}", size=12, fill="#475569"))
    for i, (lab, v) in enumerate(zip(S_LABELS, head["dot"])):
        cx = x0 + 24 + i * ((COL_W - 48) / 3)
        neg = v < 0
        L.append(text(cx, y + 50, lab, size=10.5, fill="#64748b"))
        L.append(text(cx, y + 72, (f"{v:.0f}" if v == int(v) else f"{v}"), size=14,
                       weight="bold", fill=("#dc2626" if neg else "#0f172a")))
        if neg:
            L.append(f'<circle cx="{cx}" cy="{y+66}" r="15" fill="none" stroke="#dc2626" '
                      f'stroke-width="1.5" stroke-dasharray="3,2"/>')
    L.append(arrow(x0 + COL_W, y + ROW_H / 2, col_x[1], y + ROW_H / 2))

    # 列2: ReLU 后(原负值位置标 ->0)
    x1 = col_x[1]
    L.append(box(x1, y, COL_W, ROW_H, "#ecfdf5", "#10b981"))
    L.append(text(x1 + COL_W / 2, y + 22, "ReLU(q·k)", size=12, fill="#475569"))
    for i, (lab, v, orig) in enumerate(zip(S_LABELS, head["relu"], head["dot"])):
        cx = x1 + 24 + i * ((COL_W - 48) / 3)
        was_neg = orig < 0
        L.append(text(cx, y + 50, lab, size=10.5, fill="#64748b"))
        L.append(text(cx, y + 72, (f"{v:.0f}" if v == int(v) else f"{v}"), size=14,
                       weight="bold", fill=("#0f766e" if was_neg else "#0f172a")))
        if was_neg:
            L.append(text(cx, y + 92, "截零", size=9.5, fill="#0f766e", anchor="middle"))
    L.append(arrow(x1 + COL_W, y + ROW_H / 2, col_x[2], y + ROW_H / 2))

    # 列3: × w
    x2 = col_x[2]
    L.append(box(x2, y, COL_W, ROW_H, "#fef3c7", "#d97706"))
    L.append(text(x2 + COL_W / 2, y + 22, f"× w = {head['w']:.0f}", size=12, weight="bold", fill="#92400e"))
    for i, (lab, v) in enumerate(zip(S_LABELS, head["weighted"])):
        cx = x2 + 24 + i * ((COL_W - 48) / 3)
        L.append(text(cx, y + 50, lab, size=10.5, fill="#64748b"))
        L.append(text(cx, y + 72, (f"{v:.0f}" if v == int(v) else f"{v}"), size=14,
                       weight="bold", fill="#0f172a"))

# 汇聚箭头 -> 最终 I(t0)
final_x = col_x[2] + COL_W + 56
mid_y = (row_y[0] + ROW_H / 2 + row_y[1] + ROW_H / 2) / 2
for r in range(2):
    y_start = row_y[r] + ROW_H / 2
    L.append(f'<path d="M {col_x[2]+COL_W+6} {y_start} L {final_x-14} {mid_y}" '
              f'fill="none" stroke="#0f766e" stroke-width="2" marker-end="url(#ag)"/>')

fbox_h = 130
fbox_y = mid_y - fbox_h / 2
L.append(box(final_x, fbox_y, FINAL_COL_W, fbox_h, "#d1fae5", "#0f766e", sw=2.2))
L.append(text(final_x + FINAL_COL_W / 2, fbox_y + 26, "求和", size=12, weight="bold", fill="#0f766e"))
L.append(text(final_x + FINAL_COL_W / 2, fbox_y + 48, "I(t0) =", size=13, fill="#334155"))
for i, (lab, v) in enumerate(zip(S_LABELS, I_T0)):
    cx = final_x + 26 + i * ((FINAL_COL_W - 52) / 3)
    L.append(text(cx, fbox_y + 78, lab, size=10.5, fill="#64748b"))
    L.append(text(cx, fbox_y + 100, f"{v:.0f}", size=15, weight="bold", fill="#0f172a"))
L.append(text(final_x + FINAL_COL_W / 2, fbox_y + 122, "s1 靠 head1(×2)冲到最高分",
               size=10.5, fill="#0f766e"))

# 图例(负相关/ReLU 截零)
ly = H - 44
L.append(f'<circle cx="{PAD_L+8}" cy="{ly}" r="8" fill="none" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="3,2"/>')
L.append(text(PAD_L + 24, ly + 4, "q·k < 0(负相关)", size=11.5, anchor="start", fill="#334155"))
L.append(text(PAD_L + 210, ly + 4, "ReLU 截零后恒为 0,绝不倒扣总分 I", size=11.5, anchor="start", fill="#0f766e"))
L.append(text(PAD_L, H - 16,
               "对应 vLLM 的 fp8_fp4_mqa_logits 核:q·k / ReLU / 逐头加权(softmax_scale·q_scale·n_head^-0.5 预折进 w)",
               size=10.5, anchor="start", fill="#64748b"))

L.append('</svg>')
out = Path(__file__).with_name("fig-index-score-flow.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
