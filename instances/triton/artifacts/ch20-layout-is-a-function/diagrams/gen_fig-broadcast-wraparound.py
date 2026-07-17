#!/usr/bin/env python3
"""fig-broadcast-wraparound —— L(T) 表里 broadcast(一格多线程)与
wrap-around(一线程多格)并存(m06)。T=2x8,L=4x4。
数据出处:TritonGPUAttrDefs.td:L559-L569(见 explainer/traces/td_layout_tables.md §C)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "broadcast + wrap-around —— 同一张 L(T) 表里的两种映射语义"
SUBTITLE = "张量 T=2x8,布局 L=4x4(TritonGPUAttrDefs.td:L559-L569)"

# T 的 2 行 x 8 列;每格值 = 线程集合(取自 .td 逐字 L(T) 数组)
ROWS = [
    [(0, 8), (1, 9), (2, 10), (3, 11), (0, 8), (1, 9), (2, 10), (3, 11)],
    [(4, 12), (5, 13), (6, 14), (7, 15), (4, 12), (5, 13), (6, 14), (7, 15)],
]
CHECK_CELLS = {  # (row, col) -> provenance
    (0, 0): "TritonGPUAttrDefs.td:L568", (0, 3): "TritonGPUAttrDefs.td:L568",
    (0, 4): "TritonGPUAttrDefs.td:L568", (1, 0): "TritonGPUAttrDefs.td:L569",
    (1, 7): "TritonGPUAttrDefs.td:L569",
}
N_COLS = 8
CELL_W, CELL_H = 130, 70
LABEL_W = 40
PAD, TOP = 40, 150
grid_w = N_COLS * CELL_W
grid_h = len(ROWS) * CELL_H
w = PAD * 2 + LABEL_W + grid_w
h = TOP + grid_h + 190

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

grid_x0 = PAD + LABEL_W
grid_y0 = TOP

# 左半(col 0-3)/右半(col 4-7,wrap 复用)背景区分
L.append(f'<rect x="{grid_x0}" y="{grid_y0}" width="{4*CELL_W}" height="{grid_h}" '
          f'fill="#faf5ff"/>')
L.append(f'<rect x="{grid_x0+4*CELL_W}" y="{grid_y0}" width="{4*CELL_W}" height="{grid_h}" '
          f'fill="#f3e8ff"/>')
sep_x = grid_x0 + 4 * CELL_W
L.append(f'<line x1="{sep_x}" y1="{grid_y0}" x2="{sep_x}" y2="{grid_y0+grid_h}" '
          f'stroke="#7c3aed" stroke-width="2" stroke-dasharray="5,4"/>')

for r, row in enumerate(ROWS):
    ry = grid_y0 + r * CELL_H
    L.append(f'<text x="{grid_x0-10}" y="{ry+CELL_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="12" fill="#475569">'
              f'row {r}</text>')
    for c, (a, b) in enumerate(row):
        cx = grid_x0 + c * CELL_W
        is_check = (r, c) in CHECK_CELLS
        stroke = "#dc2626" if is_check else "#c4b5fd"
        sw = 2.5 if is_check else 1
        L.append(f'<rect x="{cx+3}" y="{ry+4}" width="{CELL_W-6}" height="{CELL_H-8}" rx="6" '
                  f'fill="white" stroke="{stroke}" stroke-width="{sw}"/>')
        weight = 'font-weight="bold" ' if is_check else ''
        fill = "#b91c1c" if is_check else "#334155"
        L.append(f'<text x="{cx+CELL_W/2}" y="{ry+CELL_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="14" {weight}fill="{fill}">'
                  f'{{{a}, {b}}}</text>')

# 列坐标标签
col_label_y = grid_y0 - 8
for c in range(N_COLS):
    cx = grid_x0 + c * CELL_W + CELL_W / 2
    L.append(f'<text x="{cx}" y="{col_label_y}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="#64748b">col {c}</text>')

L.append(f'<text x="{grid_x0+2*CELL_W}" y="{col_label_y-22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#7c3aed">'
          f'原始 4 列(L 的宽度)</text>')
L.append(f'<text x="{grid_x0+6*CELL_W}" y="{col_label_y-22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#7c3aed">'
          f'wrap-around:复用 col 0-3</text>')

# 底部说明条
note_y = grid_y0 + grid_h + 40
L.append(f'<rect x="{PAD}" y="{note_y-24}" width="{w-2*PAD}" height="110" rx="8" '
          f'fill="#faf5ff" stroke="#7c3aed" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+18}" y="{note_y}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#5b21b6">broadcast(行方向):T 高 2 &lt; L 高 4 '
          f'&#8594; 每格是 2 个线程 {{a, a+8}}(如 (0,0)={{0,8}})</text>')
L.append(f'<text x="{PAD+18}" y="{note_y+24}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#5b21b6">wrap-around(列方向):T 宽 8 &gt; L 宽 4 '
          f'&#8594; col 4-7 循环复用 col 0-3 的线程号(如 (0,4)=(0,0)={{0,8}})</text>')
L.append(f'<text x="{PAD+18}" y="{note_y+52}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">红框样点:(0,0)={{0,8}} (0,3)={{3,11}} (0,4)={{0,8}} '
          f'(1,0)={{4,12}} (1,7)={{7,15}} &#8212; broadcast 因子=2(=4/2),'
          f'wrap 因子=2(=8/4)</text>')
L.append(f'<text x="{PAD+18}" y="{note_y+76}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">正解释「正式定义」一节顿悟例中 L(0,0)={{0,4}} 为何是集合而非单点。</text>')

L.append('</svg>')
out = Path(__file__).parent / "fig-broadcast-wraparound.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
