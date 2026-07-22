#!/usr/bin/env python3
"""fig-m4-merge-align: addState 双指针拉链归并（state-table 模板）。
行 = 逐 dimIndex 归并轮次 + 收尾（offset/source），列 = [lhs/rhs 该维/兼容判定/
newShape/newStride/前进侧]；末行（收尾）高亮。数据取 %25=addi(%23,%24) 真实推导。
全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "addState 双指针拉链归并（%25 = arith.addi(%23, %24)，PtrAnalysis.cpp:L520-L583）"
SUBTITLE = "同 dimIndex 要求 shape 互为倍数：取 min 作公共段、stride 相加；较大侧 shape 除公共段留待下轮拆分"

COLS = ["lhs(%23) 该维\n(stride,shape)", "rhs(%24) 该维\n(stride,shape)",
        "isMultiple\n兼容判定", "newShape\n=min(shapeL,shapeR)", "newStride\n=strideL+strideR"]
ROWS = [
    ("d0", "(%arg4, 64)", "(0, 64)", "64|64 是", "64", "%arg4+0=%arg4", "两侧同前进", "normal"),
    ("d1", "(0, 256)", "(1, 256)", "256|256 是", "256", "0+1=1", "两侧同前进", "normal"),
    ("收尾", "无剩余", "无剩余", "—", "—",
     "offset: 0+rem(%9,1024)\nsource: ∅?∅→∅", "终态提交", "final"),
]

COLOR = {"normal": (None, None), "final": ("#ecfdf5", "#047857")}

LABEL_W = 76
COL_W = [178, 178, 150, 168, 260]
ROW_H, HEADER_H, TOP, PAD = 74, 54, 108, 30

w = PAD * 2 + LABEL_W + sum(COL_W)
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 46

col_x = [PAD + LABEL_W]
for cw in COL_W[:-1]:
    col_x.append(col_x[-1] + cw)
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROWS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 表头
for j, name in enumerate(COLS):
    x = col_x[j]
    cw = COL_W[j] - 8
    lines = name.split("\n")
    L.append(f'<rect x="{x}" y="{TOP}" width="{cw}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    y0 = TOP + (HEADER_H - 6) / 2 - (len(lines) - 1) * 7 + 4
    for k, line in enumerate(lines):
        L.append(f'<text x="{x+cw/2}" y="{y0+k*14}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11.5" fill="white" '
                  f'font-weight="bold">{esc(line)}</text>')
L.append(f'<rect x="{PAD}" y="{TOP}" width="{LABEL_W-8}" height="{HEADER_H-6}" rx="3" '
          'fill="#1e3a5f"/>')
L.append(f'<text x="{PAD+(LABEL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" fill="white" font-weight="bold">维</text>')

# 行
for i, (dim, lhs, rhs, compat, newshape, newstride, forward, kind) in enumerate(ROWS):
    ry = row_y[i]
    highlight = COLOR[kind][0] is not None
    if highlight:
        fill, stroke = COLOR[kind]
        L.append(f'<rect x="{PAD}" y="{ry+4}" width="{sum(COL_W)+LABEL_W-8}" '
                  f'height="{ROW_H-8}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    text_fill = "#047857" if highlight else "#374151"
    L.append(f'<text x="{PAD+(LABEL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" '
              f'fill="{text_fill}">{esc(dim)}</text>')
    cells = [lhs, rhs, compat, newshape, newstride]
    for j, cell in enumerate(cells):
        cx = col_x[j]
        cw = COL_W[j] - 8
        lines = cell.split("\n")
        n = len(lines)
        y0 = ry + ROW_H / 2 - (n - 1) * 8 + 4
        weight = 'font-weight="bold" ' if highlight else ''
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+cw/2}" y="{y0+k*15}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="11" fill="{text_fill}" '
                      f'{weight}>{esc(line)}</text>')
    if i < len(ROWS) - 1:
        yA = ry + ROW_H
        L.append(f'<line x1="{PAD+LABEL_W-4}" y1="{yA-4}" x2="{PAD+LABEL_W-4}" y2="{yA+2}" '
                  'stroke="#94a3b8" stroke-width="1.2" marker-end="url(#a)"/>')

# 分隔线
tot_h = HEADER_H + len(ROWS) * ROW_H
for i in range(len(ROWS) + 1):
    y = TOP + HEADER_H + i * ROW_H
    L.append(f'<line x1="{PAD}" y1="{y}" x2="{PAD+LABEL_W+sum(COL_W)-8}" y2="{y}" '
              'stroke="#e2e8f0" stroke-width="1"/>')
for j in range(len(COLS) + 2):
    x = PAD if j == 0 else (PAD + LABEL_W if j == 1 else col_x[j-2] + COL_W[j-2] - 8)
    L.append(f'<line x1="{x}" y1="{TOP}" x2="{x}" y2="{TOP+tot_h}" '
              'stroke="#e2e8f0" stroke-width="1"/>')

foot_y = h - PAD + 12
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">绿色 = 收尾行：此例两侧维 shape 逐维恰好相等，是无拆分的对齐加'
          f'（isMultiple 判定：拆维需一侧 shape 严格为另一侧倍数，此例不触发）</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m4-merge-align.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
