#!/usr/bin/env python3
"""state-table 模板:期望接受长度 E[#tokens]=(1-a^(g+1))/(1-a) 随 gamma 增大而饱和。
行=gamma 取值,列=闭式值/几何和校验/上界。高亮 gamma=5、10(spec 指定数字)。
数字来自 explainer/traces/expected_length.json(alpha=0.8)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "期望接受 token 数随 gamma 饱和(alpha=0.8)"
SUBTITLE = "E[#tokens] = (1 - alpha^(gamma+1)) / (1 - alpha)  ——  上界 = 1/(1-alpha) = 5.0"
COLS = ["E[#tokens]", "几何和 Sum_k=0^g a^k(校验)", "上界 1/(1-a)"]
ROW_LABELS = ["gamma=1", "gamma=3", "gamma=5", "gamma=10"]
CELLS = {
    "gamma=1":  ["1.8",   "1.8",   "5.0"],
    "gamma=3":  ["2.952", "2.952", "5.0"],
    "gamma=5":  ["3.689", "3.689", "5.0"],
    "gamma=10": ["4.571", "4.571", "5.0"],
}
HIGHLIGHT_ROWS = {"gamma=5", "gamma=10"}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 100, 220, 52, 40, 100, 34
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 60
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    hot = row in HIGHLIGHT_ROWS
    lab_fill, lab_stroke = ("#fef3c7", "#b45309") if hot else ("#f1f5f9", "#64748b")
    L.append(f'<rect x="{PAD}" y="{ry+4}" width="{LABEL_W-10}" height="{ROW_H-8}" rx="4" '
              f'fill="{lab_fill}" stroke="{lab_stroke}" stroke-width="2"/>')
    L.append(f'<text x="{PAD+(LABEL_W-10)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="{lab_stroke}">{esc(row)}</text>')
    for j in range(len(COLS)):
        cx = col_x[j]
        val = CELLS[row][j]
        is_first_col = (j == 0)
        text_fill = "#b45309" if (hot and is_first_col) else "#374151"
        weight = 'font-weight="bold" ' if (hot and is_first_col) else ''
        cell_fill = "#fffbeb" if hot else "none"
        L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" '
                  f'fill="{cell_fill}" stroke="#e2e8f0"/>')
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" fill="{text_fill}" '
                  f'{weight}>{esc(val)}</text>')

foot1_y = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 30
L.append(f'<text x="{PAD}" y="{foot1_y}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#0f172a">gamma=5 -&gt; 3.689 个 token;gamma=10 -&gt; 4.571'
          f' —— 但上界恒为 5.0,永远无法达到</text>')
foot2_y = foot1_y + 22
L.append(f'<text x="{PAD}" y="{foot2_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">gamma 从 5 加到 10(翻倍),收益仅从 3.689 涨到 4.571 —— 几何级数饱和,越往后猜得越不划算</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig33-expected-length-vs-gamma.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
