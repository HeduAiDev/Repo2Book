#!/usr/bin/env python3
"""fig-m1-online-softmax-walk: state-table 模板。
列 = Block 0 / Block 1 / 收尾(Epilogue),行 = 在线 softmax 递推的每个中间量。
高亮 alpha 行(全图论点所在):block0 alpha=0 抹掉哨兵,block1 alpha<1 触发重标定。
数字全部来自 explainer/traces/online_softmax.json(spec.numbers)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "在线 softmax 主循环 —— fused-attention 逐块递推(query 行 0)"
SUBTITLE = "qk_scale = sm_scale × 1/ln2 = 0.5 × 1.44269504 = 0.72134752(python/tutorials/06-fused-attention.py:L162-L163)"
COLS = ["Block 0(首块)", "Block 1(重标定)", "收尾 Epilogue"]
ROW_LABELS = ["qk 原始(未缩放)", "rowmax", "m_ij(本轮最高分)", "alpha=exp2(m_i−m_ij)",
              "l_i(累积分母)", "acc(累积输出,4维)", "o = acc / l_i"]
CELLS = {
    "qk 原始(未缩放)":            ["[1.0, 0.0]", "[4.0, 0.0]", "—"],
    "rowmax":                     ["1.0", "4.0", "—"],
    "m_ij(本轮最高分)":            ["0.7213", "2.8854", "—"],
    "alpha=exp2(m_i−m_ij)":       ["0.0", "0.2231", "—"],
    "l_i(累积分母)":               ["1.6065", "1.4938", "—"],
    "acc(累积输出,4维)":           ["[1.0, 0.6065,\n0.0, 0.0]", "[0.2231, 0.1353,\n1.0, 0.1353]", "—"],
    "o = acc / l_i":               ["—", "—", "[0.1494, 0.0906,\n0.6694, 0.0906]"],
}
HIGHLIGHT_ROW = "alpha=exp2(m_i−m_ij)"
STATUS = {HIGHLIGHT_ROW: ["init", "rescale", None]}
COLOR = {"init": ("#fef3c7", "#b45309"), "rescale": ("#fee2e2", "#b91c1c")}
NOTE_ROW = "o = acc / l_i"
NOTE_STATUS = {NOTE_ROW: [None, None, "final"]}
COLOR["final"] = ("#dcfce7", "#15803d")

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 210, 210, 60, 34, 118, 34
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 30

col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS.get(row) or NOTE_STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        lines = CELLS[row][j].split("\n")
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        n = len(lines)
        y0 = ry + ROW_H / 2 - (n - 1) * 8 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+(COL_W-8)/2}" y="{y0+k*16}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12" fill="{text_fill}" '
                      f'{weight_attr}>{esc(line)}</text>')

foot_y = h - PAD + 10
L.append(f'<text x="{PAD}" y="{foot_y-16}" font-family="sans-serif" font-size="11.5" '
          f'fill="#b45309">琥珀 = alpha=0,把 l_i 的哨兵初值 1.0 一乘归零(首块自动生效)</text>')
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#b91c1c">红 = alpha=0.2231&lt;1,把 block 0 的 (l_i, acc) 按新最高分重标定后再合并</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-m1-online-softmax-walk.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
