#!/usr/bin/env python3
"""state-table 模板:墙钟加速比 (1-a^(g+1)) / ((1-a)(gc+1)) —— 复现论文 Table 1(c=0)
并加入现实草稿开销 c=0.05 后的最优 gamma。数字来自 explainer/traces/walltime.json。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "墙钟加速比 = (1 - alpha^(gamma+1)) / ((1-alpha)(gamma*c+1))"
SUBTITLE = "c = 每个草稿 token 相对目标步的相对开销;前两行复现论文 Table 1(c=0),第三行加入现实开销 c=0.05"
COLS = ["alpha", "gamma", "c(草稿开销)", "加速比", "备注"]
ROW_LABELS = ["行 1", "行 2", "行 3"]
CELLS = {
    "行 1": ["0.8", "5",  "0.0",  "3.689X", "Table 1"],
    "行 2": ["0.9", "10", "0.0",  "6.862X", "Table 1"],
    "行 3": ["0.8", "8",  "0.05", "3.092X", "最优 gamma;下界 1.714X"],
}
HIGHLIGHT_ROWS = {"行 3"}

LABEL_W, ROW_H, HEADER_H, TOP, PAD = 46, 54, 40, 100, 34
COL_W_LIST = [110, 100, 130, 130, 230]
w = PAD * 2 + LABEL_W + sum(COL_W_LIST)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 46
col_x = []
xc = PAD + LABEL_W
for cw in COL_W_LIST:
    col_x.append(xc)
    xc += cw
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x, cw = col_x[j], COL_W_LIST[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{cw-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(cw-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    hot = row in HIGHLIGHT_ROWS
    lab_fill, lab_stroke = ("#fef3c7", "#b45309") if hot else ("#f1f5f9", "#64748b")
    L.append(f'<rect x="{PAD}" y="{ry+4}" width="{LABEL_W-10}" height="{ROW_H-8}" rx="4" '
              f'fill="{lab_fill}" stroke="{lab_stroke}" stroke-width="2"/>')
    L.append(f'<text x="{PAD+(LABEL_W-10)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" font-weight="bold" '
              f'fill="{lab_stroke}">{i+1}</text>')
    for j in range(len(COLS)):
        x, cw = col_x[j], COL_W_LIST[j]
        val = CELLS[row][j]
        is_speedup_col = (j == 3)
        text_fill = "#b45309" if (hot and is_speedup_col) else "#374151"
        weight = 'font-weight="bold" ' if is_speedup_col else ''
        cell_fill = "#fffbeb" if hot else "none"
        L.append(f'<rect x="{x}" y="{ry+4}" width="{cw-8}" height="{ROW_H-8}" '
                  f'fill="{cell_fill}" stroke="#e2e8f0"/>')
        L.append(f'<text x="{x+(cw-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="{text_fill}" '
                  f'{weight}>{esc(val)}</text>')

foot1_y = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 26
L.append(f'<text x="{PAD}" y="{foot1_y}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#0f172a">草稿开销可忽略时(c=0):alpha=0.8,gamma=5 -&gt; 3.689X;'
          f'alpha=0.9,gamma=10 -&gt; 6.862X</text>')
foot2_y = foot1_y + 20
L.append(f'<text x="{PAD}" y="{foot2_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">加入现实开销 c=0.05 后最优点移到 gamma=8,得 3.092X,仍远高于 Corollary 3.9 下界 (1+alpha)/(1+c)=1.714X</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig33-walltime-speedup.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
