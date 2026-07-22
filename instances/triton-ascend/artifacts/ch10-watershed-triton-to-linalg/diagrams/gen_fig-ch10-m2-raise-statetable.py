#!/usr/bin/env python3
"""fig-ch10-m2-raise-statetable: PtrAnalysis 后序还原 add_kernel 的 x 侧地址（state-table 模板）。
行 = 5 个还原轮次（时间顺序），列 = [算子(IR)/还原规则/PtrState/源码依据]；末行（收敛结果）高亮。
全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "PtrAnalysis 还原 add_kernel 的 x 侧地址（BLOCK=4, pid=2）"
SUBTITLE = "从最外层 tt.addptr 后序递归下潜，5 步收敛到 (offset=8, size=4, stride=1)"

COLS = ["当轮算子 (IR)", "还原规则 (函数)", "得到的 PtrState", "源码依据"]
ROWS = [
    ("①", "tt.make_range\n{0,4}", "visitOperandMakeRange",
     "{src=∅, off=0,\n[stride=1,size=4]}", "L778,L786,L787,L788", "normal"),
    ("②", "tt.splat %bs\n(标量 8)", "visitOperandSplat",
     "{src=∅, off=8,\n[stride=0,size=4]}", "L908,L913", "normal"),
    ("③", "arith.addi\n(#2,#1)", "visitOperandAdd\n→addState",
     "{src=∅, off=8,\n[stride=1,size=4]}", "L561,L578", "normal"),
    ("④", "tt.splat %x_ptr\n(指针)", "visitOperandSplat",
     "{src=x_ptr, off=0,\n[stride=0,size=4]}", "L908,L913", "normal"),
    ("⑤", "tt.addptr\n(#4,#3)", "visitOperandAddptr\n→addState",
     "{src=x_ptr, off=8,\n[stride=1,size=4]}", "L279,L561,L578", "final"),
]

COLOR = {"normal": (None, None), "final": ("#ecfdf5", "#047857")}

LABEL_W = 44
COL_W = [150, 176, 226, 158]
ROW_H, HEADER_H, TOP, PAD = 68, 40, 100, 30

w = PAD * 2 + LABEL_W + sum(COL_W)
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 34

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
    L.append(f'<rect x="{x}" y="{TOP}" width="{cw}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+cw/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')
L.append(f'<rect x="{PAD}" y="{TOP}" width="{LABEL_W-8}" height="{HEADER_H-6}" rx="3" '
          'fill="#1e3a5f"/>')
L.append(f'<text x="{PAD+(LABEL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" fill="white" font-weight="bold">轮次</text>')

# 行
for i, (rnd, op, rule, state, src, kind) in enumerate(ROWS):
    ry = row_y[i]
    highlight = COLOR[kind][0] is not None
    if highlight:
        fill, stroke = COLOR[kind]
        L.append(f'<rect x="{PAD}" y="{ry+4}" width="{sum(COL_W)+LABEL_W-8}" '
                  f'height="{ROW_H-8}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{PAD+(LABEL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="15" font-weight="bold" '
              f'fill="{"#047857" if highlight else "#374151"}">{esc(rnd)}</text>')
    cells = [op, rule, state, src]
    for j, cell in enumerate(cells):
        cx = col_x[j]
        cw = COL_W[j] - 8
        lines = cell.split("\n")
        n = len(lines)
        y0 = ry + ROW_H / 2 - (n - 1) * 8 + 4
        text_fill = "#047857" if highlight else "#374151"
        weight = 'font-weight="bold" ' if highlight and j != 3 else ''
        font_size = "11.5" if j != 2 else "11"
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+cw/2}" y="{y0+k*15}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="{font_size}" fill="{text_fill}" '
                      f'{weight}>{esc(line)}</text>')
    if i < len(ROWS) - 1:  # 竖向流箭头：本轮 PtrState 汇入下一轮
        yA = ry + ROW_H
        L.append(f'<line x1="{PAD+LABEL_W-4}" y1="{yA-4}" x2="{PAD+LABEL_W-4}" y2="{yA+2}" '
                  'stroke="#94a3b8" stroke-width="1.2" marker-end="url(#a)"/>')

# 分隔线
for i in range(len(ROWS) + 1):
    y = TOP + HEADER_H + i * ROW_H if i < len(ROWS) else TOP + HEADER_H + len(ROWS) * ROW_H
    L.append(f'<line x1="{PAD}" y1="{y}" x2="{PAD+LABEL_W+sum(COL_W)-8}" y2="{y}" '
              'stroke="#e2e8f0" stroke-width="1"/>')
for j in range(len(COLS) + 2):
    x = PAD if j == 0 else (PAD + LABEL_W if j == 1 else col_x[j-2] + COL_W[j-2] - 8)
    L.append(f'<line x1="{x}" y1="{TOP}" x2="{x}" y2="{TOP+HEADER_H+len(ROWS)*ROW_H}" '
              'stroke="#e2e8f0" stroke-width="1"/>')

foot_y = h - PAD + 8
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">绿色 = 第 5 步收敛结果：(source=x_ptr, offset=8, sizes=[4], strides=[1])——'
          f'即 memref.reinterpret_cast 的输入三元组</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch10-m2-raise-statetable.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
