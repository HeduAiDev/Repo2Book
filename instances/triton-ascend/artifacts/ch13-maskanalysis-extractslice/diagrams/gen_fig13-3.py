#!/usr/bin/env python3
"""fig13-3 before-after 模板改写为『数字线多行对比』:同一 range [0,16) 与 bound=10,
五种 cmpi 谓词各熔出不同 (offset,dim) 矩形。每行 16 个格子,保留段高亮、其余段灰淡,
bound=10 处画竖虚线基准。数据取自 explainer m3.figure_specs.numbers(与 model_out.json 对齐)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

N = 16
BOUND = 10
ROWS = [
    ("slt (<)",  0, 10, "[0:10] dim=10"),
    ("sle (<=)", 0, 11, "[0:11] dim=11"),
    ("sge (>=)", 10, 6, "[10:16] off=10 dim=6"),
    ("eq (==)",  10, 1, "[10:11] off=10 dim=1"),
    ("ne (!=0)", 0, 16, "[0:16] dim=16"),
]

CELL, GAP = 34, 2
LABEL_W = 90
RESULT_W = 190
PAD, TOP, ROW_H = 40, 110, 52

w = PAD * 2 + LABEL_W + N * (CELL + GAP) + RESULT_W
h = TOP + len(ROWS) * ROW_H + 130

grid_x0 = PAD + LABEL_W

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="{38}" text-anchor="middle" font-family="sans-serif" '
     f'font-size="16" font-weight="bold" fill="#0f172a">'
     f'{esc("同一 range [0,16),bound=10:五种 cmpi 谓词熔出五种矩形切片")}</text>',
     f'<text x="{w/2}" y="{60}" text-anchor="middle" font-family="sans-serif" '
     f'font-size="12" fill="#475569">'
     f'{esc("蓝色=保留段(将被切出),灰色=丢弃段;红色虚线=bound 基准位置")}</text>']

# column index ruler
ruler_y = TOP - 14
for i in range(N):
    x = grid_x0 + i * (CELL + GAP) + CELL / 2
    if i % 2 == 0:
        L.append(f'<text x="{x}" y="{ruler_y}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="10" fill="#94a3b8">{i}</text>')

bound_x = grid_x0 + BOUND * (CELL + GAP)

for r, (name, off, dim, res) in enumerate(ROWS):
    y = TOP + r * ROW_H
    L.append(f'<text x="{PAD+LABEL_W-14}" y="{y+CELL/2+5}" text-anchor="end" '
              f'font-family="monospace" font-size="13" font-weight="bold" '
              f'fill="#0f172a">{esc(name)}</text>')
    for i in range(N):
        x = grid_x0 + i * (CELL + GAP)
        kept = off <= i < off + dim
        fill = "#3b82f6" if kept else "#f1f5f9"
        stroke = "#1e40af" if kept else "#cbd5e1"
        L.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" rx="3" '
                  f'fill="{fill}" stroke="{stroke}"/>')
    result_x = grid_x0 + N * (CELL + GAP) + 20
    L.append(f'<text x="{result_x}" y="{y+CELL/2+5}" font-family="monospace" font-size="12" '
              f'fill="#1e3a8a">{esc(res)}</text>')

grid_bottom = TOP + len(ROWS) * ROW_H - (ROW_H - CELL)
L.append(f'<line x1="{bound_x}" y1="{TOP-20}" x2="{bound_x}" y2="{grid_bottom}" '
          f'stroke="#dc2626" stroke-width="1.5" stroke-dasharray="5,4"/>')
L.append(f'<text x="{bound_x}" y="{TOP-26}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#dc2626">{esc("bound=10")}</text>')

foot_y = h - 60
L.append(f'<text x="{w/2}" y="{foot_y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#334155">'
          f'{esc("slt/sle 剪尾(改 dim)、sge 抬头(唯一改 offset)、eq 定点(dim=1)、ne 全保。")}</text>')
L.append(f'<text x="{w/2}" y="{foot_y+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#334155">'
          f'{esc("这是唯一把标量 bound 熔进 (offset,dim) 的地方——越界值经 clamp 变成空切片。")}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig13-3.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
