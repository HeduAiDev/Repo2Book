#!/usr/bin/env python3
"""fig13-4 tiling 模板改写为『矩形交叠图』:16x16 网格上,行掩码([0,10)x全)与
列掩码(全x[0,12))的重叠区就是 AND 结果(minStates 逐维取交)。比例按真实 16x16
形状精确绘制(10/16、12/16),不为版式简化牺牲维度。数据取自 explainer m4。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

GRID = 320       # 16x16 网格边长(像素),严格按 16 等分
N = 16
ROW_DIM = 10     # 行掩码保留 [0,10)
COL_DIM = 12     # 列掩码保留 [0,12)
CELL = GRID / N

PAD, TOP = 60, 130
LEGEND_H = 60
w = PAD * 2 + GRID + 260
h = TOP + GRID + LEGEND_H + 90

gx0, gy0 = PAD, TOP

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="{38}" text-anchor="middle" font-family="sans-serif" '
     f'font-size="16" font-weight="bold" fill="#0f172a">'
     f'{esc("andi 两矩形掩码相与 = 逐维区间交(16x16 网格)")}</text>',
     f'<text x="{w/2}" y="{60}" text-anchor="middle" font-family="sans-serif" '
     f'font-size="12" fill="#475569">'
     f'{esc("行掩码 A:dim0<10(offsets=[0,0] dims=[10,16])  ×  列掩码 B:dim1<12(offsets=[0,0] dims=[16,12])")}</text>']

# base grid outline + light gridlines every 2 cells
L.append(f'<rect x="{gx0}" y="{gy0}" width="{GRID}" height="{GRID}" fill="#f8fafc" stroke="#94a3b8"/>')
for i in range(0, N + 1, 2):
    x = gx0 + i * CELL
    L.append(f'<line x1="{x}" y1="{gy0}" x2="{x}" y2="{gy0+GRID}" stroke="#e2e8f0" stroke-width="1"/>')
    y = gy0 + i * CELL
    L.append(f'<line x1="{gx0}" y1="{y}" x2="{gx0+GRID}" y2="{y}" stroke="#e2e8f0" stroke-width="1"/>')

# row mask band A: rows 0..ROW_DIM (dim0 axis = vertical/y), all columns
row_h = ROW_DIM * CELL
L.append(f'<rect x="{gx0}" y="{gy0}" width="{GRID}" height="{row_h}" '
          f'fill="#3b82f6" fill-opacity="0.35"/>')
# col mask band B: cols 0..COL_DIM (dim1 axis = horizontal/x), all rows
col_w = COL_DIM * CELL
L.append(f'<rect x="{gx0}" y="{gy0}" width="{col_w}" height="{GRID}" '
          f'fill="#f97316" fill-opacity="0.35"/>')
# intersection: rows 0..ROW_DIM, cols 0..COL_DIM — outline it distinctly
L.append(f'<rect x="{gx0}" y="{gy0}" width="{col_w}" height="{row_h}" '
          f'fill="#16a34a" fill-opacity="0.45" stroke="#15803d" stroke-width="2.5"/>')

# axis ticks: x axis 0,12,16 ; y axis 0,10,16
for val in (0, COL_DIM, N):
    x = gx0 + val * CELL
    L.append(f'<line x1="{x}" y1="{gy0+GRID}" x2="{x}" y2="{gy0+GRID+6}" stroke="#334155"/>')
    L.append(f'<text x="{x}" y="{gy0+GRID+20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#334155">{val}</text>')
for val in (0, ROW_DIM, N):
    y = gy0 + val * CELL
    L.append(f'<line x1="{gx0-6}" y1="{y}" x2="{gx0}" y2="{y}" stroke="#334155"/>')
    L.append(f'<text x="{gx0-10}" y="{y+4}" text-anchor="end" font-family="sans-serif" '
              f'font-size="11" fill="#334155">{val}</text>')
L.append(f'<text x="{gx0+GRID/2}" y="{gy0+GRID+40}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">{esc("dim1(列)")}</text>')
# 注:避免 rotate(-90) 竖排标签(几何 linter 不识别 transform,按未旋转水平包围盒误判越界)——
# 改用横排标签放在 y 轴上方,同样能标清"这根轴是 dim0(行)"。
L.append(f'<text x="{gx0}" y="{gy0-16}" text-anchor="start" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">{esc("↓ dim0(行)")}</text>')

# side numbers panel
nx = gx0 + GRID + 40
ny = gy0 + 10
rows = [
    ("行掩码 A", "offsets=[0,0] dims=[10,16]", "#3b82f6"),
    ("列掩码 B", "offsets=[0,0] dims=[16,12]", "#f97316"),
    ("交集 A∩B", "offsets=[0,0] dims=[10,12]", "#16a34a"),
]
for i, (name, val, color) in enumerate(rows):
    y = ny + i * 46
    L.append(f'<rect x="{nx}" y="{y}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{nx+22}" y="{y+12}" font-family="sans-serif" font-size="12" '
              f'font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<text x="{nx+22}" y="{y+30}" font-family="monospace" font-size="11" '
              f'fill="#334155">{esc(val)}</text>')
elem_y = ny + len(rows) * 46 + 10
L.append(f'<text x="{nx}" y="{elem_y}" font-family="sans-serif" font-size="12" '
          f'fill="#0f172a">{esc("有效元素 10×12=120 / 256")}</text>')

foot_y = h - 30
L.append(f'<text x="{w/2}" y="{foot_y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#334155">'
          f'{esc("AND = 矩形交,一个 extract_slice 装得下;OR 的并集是 L 形,parseOr 不存在。")}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig13-4.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
