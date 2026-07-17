#!/usr/bin/env python3
"""state-table 模板:v1 的活跃区间随别名值(v3/v4/v6)逐次并入,min/max 求并演化。
最后一列(v1 当前活跃区间)每行高亮,末行标记最终结果。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "别名合并活跃区间 — v1 的活跃期随 v3/v4/v6 使用区间逐次并入"
SUBTITLE = "resolveAliasBufferLiveness: minId=min(minId,新值), maxId=max(maxId,新值)  (Allocation.cpp:L385-L387)"

COLS = ["并入的别名值", "该值使用区间", "running minId", "running maxId", "v1 当前活跃区间"]
ROWS = [
    ["(v1 自身 alloc 点)", "[2,3)", "2", "3", "[2,3)"],
    ["v3", "[2,5)", "2", "5", "[2,5)"],
    ["v4", "[4,7)", "2", "7", "[2,7)"],
    ["v6", "[6,9)", "2", "9", "[2,9)"],
]
FINAL_NOTE = "终态 [2,9): 宽度 9-2=7 op,涵盖 v3/v4/v6 每一次使用"

COL_W = [150, 110, 110, 110, 130]
ROW_H, HEADER_H, PAD, TOP = 46, 36, 32, 92
LABEL_PAD = 14
w = PAD * 2 + sum(COL_W)
h = TOP + HEADER_H + ROW_H * len(ROWS) + 46 + PAD

col_x = [PAD]
for cw in COL_W[:-1]:
    col_x.append(col_x[-1] + cw)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W[j]-6}" height="{HEADER_H}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W[j]-6)/2}" y="{TOP+HEADER_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROWS):
    ry = TOP + HEADER_H + i * ROW_H
    is_last = (i == len(ROWS) - 1)
    for j, val in enumerate(row):
        x = col_x[j]
        cw = COL_W[j] - 6
        if j == 4:  # last column: highlight, deepen on final row
            fill = "#fef08a" if is_last else "#ecfdf5"
            stroke = "#b45309" if is_last else "#047857"
            L.append(f'<rect x="{x}" y="{ry+4}" width="{cw}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
            text_fill = "#92400e" if is_last else "#047857"
            weight = 'font-weight="bold" '
        else:
            text_fill = "#374151"
            weight = ""
        L.append(f'<text x="{x+cw/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" fill="{text_fill}" '
                  f'{weight}>{esc(val)}</text>')
    # row divider
    L.append(f'<line x1="{PAD}" y1="{ry+ROW_H}" x2="{PAD+sum(COL_W)-6}" y2="{ry+ROW_H}" '
              'stroke="#e2e8f0" stroke-width="1"/>')

foot_y = TOP + HEADER_H + ROW_H * len(ROWS) + 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#92400e">{esc(FINAL_NOTE)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-liveness-alias-merge.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
