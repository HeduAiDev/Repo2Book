#!/usr/bin/env python3
"""paper-fig-2-awq: 重绘自 arXiv:2306.00978 Figure 2 —— AWQ 建立"显著权重"直觉的三联对比图。
(a) RTN 量化：W_FP16 逐元素取整到 INT3，PPL=43.2。
(b) 把由激活幅度定位出的 1% 显著通道整行保留 FP16（混合精度理想解），PPL 降到 13.0，
    但混合精度对硬件不友好。
(c) AWQ：不改变数据类型，而是量化前按激活幅度对显著通道整体放大 alpha 倍再量化，
    同样把 PPL 降到 13.0，且全程仍是纯 INT3、硬件友好。
矩阵数值取自原论文图中的示例本身（provenance=原论文），非本章 explainer 数据。
每个 panel 内部固定为「左列（X 或 W_FP16）— 中间过渡带（箭头/符号/标注）— 右列（Q(W)）」，
三个 panel 左右并排，互不重叠。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "AWQ 的核心直觉：给 1% 显著通道“戴放大镜”，比混合精度更硬件友好地达到同等 PPL"
SUBTITLE = "重绘自 arXiv:2306.00978 Figure 2（OPT-6.7B，INT3-g128）"

# 8x4 权重网格数值（原图示例本身）
W_FP16 = [
    ["+1.2", "-0.2", "-2.4", "-3.4"],
    ["-2.5", "-3.5", "+1.9", "+1.4"],
    ["-0.9", "+1.6", "-2.5", "-1.9"],
    ["-3.5", "+1.5", "+0.5", "-0.1"],
    ["+1.8", "-1.6", "-3.2", "-3.4"],
    ["+2.4", "-3.5", "-2.8", "-3.9"],
    ["+0.1", "-3.8", "+2.4", "+3.4"],
    ["+0.9", "+3.3", "-1.9", "-2.3"],
]
Q_INT3 = [
    ["+1", "+0", "-2", "-3"],
    ["-3", "-4", "+2", "+1"],
    ["-1", "+2", "-3", "-2"],
    ["-4", "+2", "+1", "+0"],
    ["+2", "-2", "-3", "-3"],
    ["+2", "-4", "-3", "-4"],
    ["+0", "-4", "+2", "+3"],
    ["+1", "+3", "-2", "-2"],
]
SALIENT_ROW = 1  # 0-indexed：与 X 的显著通道对齐的那一行
SALIENT_COL = 1  # X 里显著通道所在的列（0-indexed）

CELL_W, CELL_H = 32, 24
GRID_W, GRID_H = CELL_W * 4, CELL_H * 8
X_COLS = 8
X_CELL_W, X_CELL_H = GRID_W / X_COLS, 22
X_ROWS = 3
GAP_MID = 64
RIGHT_OFF = GRID_W + GAP_MID

TAN, TAN_STROKE = "#fde3b8", "#c2820f"
BLUE, BLUE_STROKE = "#bfdbfe", "#1d4ed8"
GRID_TEXT = "#1e293b"
X_ROW_COLORS = ["#fde8e8", "#dc2626", "#fdeded", "#fcd8d8", "#f7b3b3", "#fdeaea", "#fef4f4", "#fef6f6"]

PANEL_W = 380
PAD = 36
TOP = 110
PANELS_Y = TOP + 46  # 8x4 网格顶部
X_BOTTOM_Y = PANELS_Y + GRID_H  # X 矩阵与网格底对齐
X_TOP_Y = X_BOTTOM_Y - X_ROWS * X_CELL_H
CAP_Y = PANELS_Y + GRID_H + 30
w = PAD * 4 + PANEL_W * 3
h = CAP_Y + 68


def grid_svg(x, y, values, fills, stroke, font_size=9.5):
    out = []
    for r, row in enumerate(values):
        for c, val in enumerate(row):
            cx = x + c * CELL_W
            cy = y + r * CELL_H
            fill = fills[r][c]
            out.append(f'<rect x="{cx}" y="{cy}" width="{CELL_W}" height="{CELL_H}" '
                       f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
            if val is not None:
                out.append(f'<text x="{cx+CELL_W/2}" y="{cy+CELL_H/2+3.5}" text-anchor="middle" '
                           f'font-family="sans-serif" font-size="{font_size}" '
                           f'fill="{GRID_TEXT}">{esc(val)}</text>')
    return out


def x_matrix_svg(x, y):
    out = []
    for r in range(X_ROWS):
        for c in range(X_COLS):
            cx = x + c * X_CELL_W
            cy = y + r * X_CELL_H
            out.append(f'<rect x="{cx:.1f}" y="{cy}" width="{X_CELL_W:.1f}" height="{X_CELL_H}" '
                       f'fill="{X_ROW_COLORS[c]}" stroke="#94a3b8" stroke-width="0.8"/>')
    out.append(f'<text x="{x-14}" y="{y+X_ROWS*X_CELL_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" '
              f'fill="#0f172a">X</text>')
    return out


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0f172a"/></marker>'
     '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
     '</defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-8}" font-family="sans-serif" font-size="15.5" '
     f'fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+14}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

panel_x = [PAD + i * (PANEL_W + PAD) for i in range(3)]
for i in range(1, 3):
    sx = panel_x[i] - PAD / 2
    L.append(f'<line x1="{sx}" y1="{TOP-4}" x2="{sx}" y2="{h-24}" stroke="#e2e8f0" stroke-width="1"/>')

# ================= panel (a): RTN =================
ax = panel_x[0]
ax_right = ax + RIGHT_OFF
L.append(f'<text x="{ax}" y="{PANELS_Y-10}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="#0f172a">W_FP16</text>')
L.extend(grid_svg(ax, PANELS_Y, W_FP16, [[TAN]*4]*8, TAN_STROKE))
L.append(f'<text x="{ax_right}" y="{PANELS_Y-10}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="#0f172a">Q(W)_INT3</text>')
L.extend(grid_svg(ax_right, PANELS_Y, Q_INT3, [[BLUE]*4]*8, BLUE_STROKE))
mid_y = PANELS_Y + GRID_H / 2
L.append(f'<line x1="{ax+GRID_W+8}" y1="{mid_y}" x2="{ax_right-8}" y2="{mid_y}" '
         f'stroke="#0f172a" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<text x="{ax+GRID_W+GAP_MID/2}" y="{mid_y-8}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="13" fill="#0f172a">Q</text>')
L.append(f'<text x="{ax}" y="{CAP_Y}" font-family="sans-serif" font-size="12.5" '
         f'fill="#334155">(a) RTN 量化（<tspan fill="#dc2626" font-weight="bold">PPL 43.2</tspan>）</text>')

# ================= panel (b): mixed precision =================
bx = panel_x[1]
bx_right = bx + RIGHT_OFF
mp_fills = [[BLUE]*4 for _ in range(8)]
mp_fills[SALIENT_ROW] = [TAN]*4
mp_values = [row[:] for row in Q_INT3]
mp_values[SALIENT_ROW] = W_FP16[SALIENT_ROW]

# top annotation above the right-column grid
ann_x = bx_right + 10
L.append(f'<text x="{ann_x}" y="{TOP-18}" font-family="sans-serif" font-size="11" '
         f'fill="#dc2626">硬件效率差</text>')
L.append(f'<path d="M {ann_x+70} {TOP-22} Q {ann_x+92} {TOP-34} {ann_x+112} {TOP-22}" '
         f'fill="none" stroke="#dc2626" stroke-width="1.3" marker-end="url(#ar)"/>')

L.append(f'<text x="{bx_right}" y="{PANELS_Y-10}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="#0f172a">Q(W)_MixPrec</text>')
L.extend(grid_svg(bx_right, PANELS_Y, mp_values, mp_fills, TAN_STROKE))
fp16_label_y = PANELS_Y + SALIENT_ROW * CELL_H + CELL_H / 2
L.append(f'<text x="{bx_right+GRID_W+8}" y="{fp16_label_y-2}" font-family="sans-serif" '
         f'font-size="10" fill="#b45309">FP16</text>')
L.append(f'<text x="{bx_right+GRID_W+8}" y="{fp16_label_y+11}" font-family="sans-serif" '
         f'font-size="10" fill="#b45309">通道</text>')

L.extend(x_matrix_svg(bx, X_TOP_Y))
star_x = bx + GRID_W + GAP_MID / 2
L.append(f'<text x="{star_x}" y="{X_TOP_Y+X_ROWS*X_CELL_H/2+5}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="14" fill="#0f172a">*</text>')

# dashed arrow: X 的显著列（顶边）→ 网格的显著行（左边）
sx0 = bx + SALIENT_COL * X_CELL_W + X_CELL_W / 2
sy0 = X_TOP_Y
tx0 = bx_right - 6
ty0 = fp16_label_y
L.append(f'<path d="M {sx0:.1f} {sy0} C {sx0:.1f} {sy0-46} {tx0-70:.1f} {ty0+30} {tx0:.1f} {ty0}" '
         f'fill="none" stroke="#0f172a" stroke-width="1.3" stroke-dasharray="4,3" marker-end="url(#a)"/>')
label_y0 = X_TOP_Y - 44
L.append(f'<text x="{bx}" y="{label_y0}" font-family="sans-serif" font-size="10.5" '
         f'fill="#334155">按激活幅度确定显著通道</text>')
L.append(f'<text x="{bx}" y="{CAP_Y}" font-family="sans-serif" font-size="12.5" '
         f'fill="#334155">(b) 显著通道整行保留 FP16（<tspan fill="#16a34a" font-weight="bold">'
         f'PPL 13.0</tspan>）</text>')

# ================= panel (c): AWQ scale-before-quantize =================
cx0 = panel_x[2]
cx_right = cx0 + RIGHT_OFF
awq_fills = [[BLUE]*4 for _ in range(8)]
awq_fills[SALIENT_ROW] = ["#2563eb"]*4
awq_values = [[None]*4 for _ in range(8)]
L.append(f'<text x="{cx_right}" y="{PANELS_Y-10}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="#0f172a">Q(W)_INT3</text>')
L.extend(grid_svg(cx_right, PANELS_Y, awq_values, awq_fills, BLUE_STROKE))

AVG_GAP = 40
avg_y = X_TOP_Y - X_CELL_H - AVG_GAP
for c in range(X_COLS):
    ccx = cx0 + c * X_CELL_W
    L.append(f'<rect x="{ccx:.1f}" y="{avg_y}" width="{X_CELL_W:.1f}" height="{X_CELL_H}" '
             f'fill="{X_ROW_COLORS[c]}" stroke="#94a3b8" stroke-width="0.8"/>')

L.extend(x_matrix_svg(cx0, X_TOP_Y))
star_x_c = cx0 + GRID_W + GAP_MID / 2
L.append(f'<text x="{star_x_c}" y="{X_TOP_Y+X_ROWS*X_CELL_H/2+5}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="14" fill="#0f172a">*</text>')

# average-magnitude arrow: X 顶边 → 均值行底边
avg_arrow_x = cx0 + GRID_W / 2
L.append(f'<line x1="{avg_arrow_x}" y1="{X_TOP_Y-2}" x2="{avg_arrow_x}" y2="{avg_y+X_CELL_H+4}" '
         f'stroke="#0f172a" stroke-width="1.3" marker-end="url(#a)"/>')
L.append(f'<text x="{avg_arrow_x+10}" y="{(X_TOP_Y+avg_y+X_CELL_H)/2+4}" font-family="sans-serif" '
         f'font-size="10" fill="#334155">均值幅度</text>')

# scale arrow: 均值行顶边 → 网格显著行左边，标 α
salient_grid_y = PANELS_Y + SALIENT_ROW * CELL_H + CELL_H / 2
L.append(f'<path d="M {avg_arrow_x:.1f} {avg_y-4} C {avg_arrow_x+40:.1f} {avg_y-40} '
         f'{cx_right-70:.1f} {salient_grid_y+34} {cx_right-6:.1f} {salient_grid_y}" '
         f'fill="none" stroke="#0f172a" stroke-width="1.3" marker-end="url(#a)"/>')
L.append(f'<text x="{avg_arrow_x+20}" y="{avg_y-34}" font-family="sans-serif" font-size="10.5" '
         f'fill="#334155">量化前先按 α 缩放</text>')

L.append(f'<text x="{cx0}" y="{CAP_Y}" font-family="sans-serif" font-size="12.5" '
         f'fill="#334155">(c) AWQ：显著通道量化前先缩放（<tspan fill="#16a34a" font-weight="bold">'
         f'PPL 13.0</tspan>）</text>')

foot_y = CAP_Y + 32
foot_lines = [
    "(a)(b) 权重数值与 (c) 显著通道位置均取自原论文示例本身；(c) 的 Q(W)_INT3 只画色块不画数字，",
    "呼应原图——缩放后全表仍是统一 INT3，不再需要为显著通道单独开一条 FP16 行。",
]
for i, line in enumerate(foot_lines):
    L.append(f'<text x="{PAD}" y="{foot_y+i*18}" font-family="sans-serif" font-size="12" '
             f'fill="#334155">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-2-awq.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
