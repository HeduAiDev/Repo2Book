#!/usr/bin/env python3
"""fig-m16n8k16-fragment (layout 模板)
C accumulator (m16n8, FP32) 的逐 lane 线程矩阵——本章黄金 worked example。
16 行(row 0-15) x 8 列(col 0-7)网格,每格标 lane id;lane(row,col) 完全由公式计算,
零手写魔数:lane = (row % 8) * 4 + (col // 2)。
高亮 lane0 的 4 个坐标与 lane31 的 4 个坐标,与源码矩阵逐格核对一致。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

ROWS, COLS = 16, 8

def lane_of(row, col):
    g = row % 8
    h = col // 2
    return g * 4 + h

# 高亮两个 worked-example lane(与源码矩阵逐格核对)
HI = {
    0: ("#2563eb", "#dbeafe"),   # lane 0 -> 蓝
    31: ("#c2410c", "#fed7aa"),  # lane 31 -> 橙
}

CELL_W, CELL_H = 42, 26
PAD = 78
TOP = 162
GRID_W = COLS * CELL_W
GRID_H = ROWS * CELL_H
SIDE_W = 340
GAP = 60

w = PAD + GRID_W + GAP + SIDE_W + PAD
h = TOP + GRID_H + 70

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

# 标题 + 副标题
L.append(f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="19" '
         f'font-weight="bold" fill="#0f172a">{esc("C accumulator(m16n8,FP32)的座位表——逐 lane 完全可核")}</text>')
L.append(f'<text x="{PAD}" y="{PAD+24}" font-family="sans-serif" font-size="13" '
         f'fill="#475569">{esc("源码逐字印着这张矩阵(TritonGPUAttrDefs.td:L1105-L1126)——不是从 PTX 记忆搬来的")}</text>')
L.append(f'<text x="{PAD}" y="{PAD+46}" font-family="sans-serif" font-size="13" '
         f'fill="#475569">{esc("lane(row,col) = (row mod 8) * 4 + (col / 2)   即  g = lane>>2、h = lane&3")}</text>')

grid_x, grid_y = PAD, TOP

# 网格单元
for row in range(ROWS):
    for col in range(COLS):
        lane = lane_of(row, col)
        x = grid_x + col * CELL_W
        y = grid_y + row * CELL_H
        stroke_c, fill_c, sw = "#94a3b8", "#f8fafc", 1
        if lane in HI:
            stroke_c, fill_c, sw = HI[lane][0], HI[lane][1], 2.4
        L.append(f'<rect x="{x}" y="{y}" width="{CELL_W}" height="{CELL_H}" '
                  f'fill="{fill_c}" stroke="{stroke_c}" stroke-width="{sw}"/>')
        text_fill = HI[lane][0] if lane in HI else "#334155"
        fw = "bold" if lane in HI else "normal"
        L.append(f'<text x="{x+CELL_W/2}" y="{y+CELL_H/2+4.5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" font-weight="{fw}" '
                  f'fill="{text_fill}">{lane}</text>')

# 8 行一循环的分隔粗线(第 8 行起从 lane0 重复,+8 行偏移)
mid_y = grid_y + 8 * CELL_H
L.append(f'<line x1="{grid_x-4}" y1="{mid_y}" x2="{grid_x+GRID_W+4}" y2="{mid_y}" '
          'stroke="#0f172a" stroke-width="2.4" stroke-dasharray="7,4"/>')

# 行/列坐标轴标注(行标签只标关键行,留足左侧空间避免裁切)
for col in range(COLS):
    x = grid_x + col * CELL_W + CELL_W / 2
    L.append(f'<text x="{x}" y="{grid_y-14}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#64748b">col {col}</text>')
for row in (0, 7, 8, 15):
    y = grid_y + row * CELL_H + CELL_H / 2 + 4
    L.append(f'<text x="{grid_x-14}" y="{y}" text-anchor="end" font-family="sans-serif" '
              f'font-size="11" fill="#64748b">row {row}</text>')

# 侧栏:worked example + 公式 + 数字核对
side_box_x = grid_x + GRID_W + GAP
side_x = side_box_x + 20
side_y = TOP + 30
L.append(f'<rect x="{side_box_x}" y="{TOP-4}" width="{SIDE_W}" height="{GRID_H-2}" rx="10" '
          'fill="#f8fafc" stroke="#cbd5e1"/>')
L.append(f'<text x="{side_x}" y="{side_y}" font-family="sans-serif" font-size="14" '
          f'font-weight="bold" fill="#0f172a">{esc("公式(g=lane>>2, h=lane&3)")}</text>')
L.append(f'<text x="{side_x}" y="{side_y+24}" font-family="sans-serif" font-size="12.5" '
          f'fill="#334155">{esc("lane 持 4 个 fp32 坐标:")}</text>')
L.append(f'<text x="{side_x}" y="{side_y+44}" font-family="sans-serif" font-size="12.5" '
          f'fill="#334155">{esc("(g,2h) (g,2h+1) (g+8,2h) (g+8,2h+1)")}</text>')

L.append(f'<rect x="{side_x}" y="{side_y+62}" width="14" height="14" rx="2" '
          f'fill="{HI[0][1]}" stroke="{HI[0][0]}" stroke-width="2"/>')
L.append(f'<text x="{side_x+20}" y="{side_y+73}" font-family="sans-serif" font-size="12.5" '
          f'fill="#0f172a">{esc("lane 0: g=0 h=0 -> (0,0)(0,1)(8,0)(8,1)")}</text>')

L.append(f'<rect x="{side_x}" y="{side_y+86}" width="14" height="14" rx="2" '
          f'fill="{HI[31][1]}" stroke="{HI[31][0]}" stroke-width="2"/>')
L.append(f'<text x="{side_x+20}" y="{side_y+97}" font-family="sans-serif" font-size="12.5" '
          f'fill="#0f172a">{esc("lane 31: g=7 h=3 -> (7,6)(7,7)(15,6)(15,7)")}</text>')

L.append(f'<text x="{side_x}" y="{side_y+124}" font-family="sans-serif" font-size="12.5" '
          f'fill="#334155">{esc("每 lane 持 4 个 fp32(=128/32)")}</text>')
L.append(f'<text x="{side_x}" y="{side_y+144}" font-family="sans-serif" font-size="12.5" '
          f'fill="#334155">{esc("2 连续列 = contigPerThread=2")}</text>')
L.append(f'<text x="{side_x}" y="{side_y+164}" font-family="sans-serif" font-size="12.5" '
          f'fill="#334155">{esc("(TritonGPUAttrDefs.td:L1240-L1246)")}</text>')

L.append(f'<text x="{side_x}" y="{side_y+188}" font-family="sans-serif" font-size="12.5" '
          f'fill="#334155">{esc("行 0-7 用满 lane 0-31;行 8-15")}</text>')
L.append(f'<text x="{side_x}" y="{side_y+206}" font-family="sans-serif" font-size="12.5" '
          f'fill="#334155">{esc("从 lane 0 重复(+8 行偏移)")}</text>')

L.append(f'<text x="{side_x}" y="{side_y+232}" font-family="sans-serif" font-size="12" '
          f'fill="#059669">{esc("行 8-15 与行 0-7 逐格一致")}</text>')
L.append(f'<text x="{side_x}" y="{side_y+250}" font-family="sans-serif" font-size="12" '
          f'fill="#059669">{esc("(与源码矩阵逐格核对无误)")}</text>')

# 图注
cap = "16x8 的 C 被 32 个 lane 各持 4 个 fp32 恰好分完——这是从源码矩阵一格一格读出来的,不是转述 PTX 手册。"
L.append(f'<text x="{PAD}" y="{h-14}" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc(cap)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m16n8k16-fragment.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w}x{h}")
