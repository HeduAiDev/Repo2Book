#!/usr/bin/env python3
"""fig-distributed-vs-shared —— before-after 模板改造:两大类布局的分野。
左 distributed:L(i) 只圈出少数几个持有者(复用 m02 顿悟例 8 线程/2 高亮);
右 shared:L(i) = block 内全部线程(num_warps=2 -> 64 线程,8x8 全高亮)。
数据出处:TritonGPUAttrDefs.td:L52(两大类声明)/L158-L161(shared 定义)/L40-L50(distributed 例)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "distributed vs shared —— 两大类布局的分野"

PANEL_W = 420
CELL = 34
GAP = 6
PAD = 40
TOP = 150
LEGEND_H = 40

# 左panel: 8 个方块(线程 0-7),高亮 {0,4}
LEFT_N = 8
LEFT_HL = {0, 4}
# 右panel: 8x8 = 64 个方块(线程 0-63),全部高亮
RIGHT_ROWS, RIGHT_COLS = 8, 8

left_row_w = LEFT_N * (CELL + GAP) - GAP
right_grid_w = RIGHT_COLS * (CELL - 6 + 3) - 3  # 稍紧凑
right_cell = 22
right_gap = 2
right_grid_w = RIGHT_COLS * (right_cell + right_gap) - right_gap
right_grid_h = RIGHT_ROWS * (right_cell + right_gap) - right_gap

panel_gap = 70
w = PAD * 2 + PANEL_W * 2 + panel_gap
h = TOP + right_grid_h + 130

px_left = PAD
px_right = PAD + PANEL_W + panel_gap

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="{PAD}" text-anchor="middle" font-family="sans-serif" '
     f'font-size="18" font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="{PAD+24}" text-anchor="middle" font-family="sans-serif" '
     f'font-size="12" fill="#64748b">Triton 目前实现两大类布局(TritonGPUAttrDefs.td:L52)'
     f'&#8212;同一个函数 L,两种极端形态</text>']

# —— 左 panel: distributed ——
cx = px_left + PANEL_W / 2
L.append(f'<text x="{cx}" y="{TOP-56}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="15" font-weight="bold" fill="#0f172a">distributed</text>')
L.append(f'<text x="{cx}" y="{TOP-36}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#475569">元素分散进各线程寄存器,L(i) 只圈出少数几个持有者</text>')
row_x0 = px_left + (PANEL_W - left_row_w) / 2
for i in range(LEFT_N):
    x = row_x0 + i * (CELL + GAP)
    hl = i in LEFT_HL
    fill = "#3b82f6" if hl else "#e2e8f0"
    stroke = "#1e3a5f" if hl else "#94a3b8"
    text_fill = "white" if hl else "#475569"
    L.append(f'<rect x="{x}" y="{TOP}" width="{CELL}" height="{CELL}" rx="5" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{x+CELL/2}" y="{TOP+CELL/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="{text_fill}">{i}</text>')
L.append(f'<text x="{cx}" y="{TOP+CELL+30}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#1d4ed8">L(i) = {{0, 4}}'
          f'&#8212;典型|L(i)|=2(小集合)</text>')
L.append(f'<text x="{cx}" y="{TOP+CELL+52}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">TritonGPUAttrDefs.td:L40-L50 顿悟例(8 线程块)</text>')

# —— 右 panel: shared ——
cx2 = px_right + PANEL_W / 2
L.append(f'<text x="{cx2}" y="{TOP-56}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="15" font-weight="bold" fill="#0f172a">shared</text>')
L.append(f'<text x="{cx2}" y="{TOP-36}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#475569">元素住共享内存,L(i) = block 内全部线程</text>')
grid_x0 = px_right + (PANEL_W - right_grid_w) / 2
for r in range(RIGHT_ROWS):
    for c in range(RIGHT_COLS):
        x = grid_x0 + c * (right_cell + right_gap)
        y = TOP + r * (right_cell + right_gap)
        L.append(f'<rect x="{x}" y="{y}" width="{right_cell}" height="{right_cell}" rx="3" '
                  f'fill="#f59e0b" stroke="#b45309" stroke-width="1"/>')
right_bottom = TOP + right_grid_h
L.append(f'<text x="{cx2}" y="{right_bottom+30}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#b45309">L(i) = {{0,1,...,63}}'
          f'&#8212;对所有 i 相同</text>')
L.append(f'<text x="{cx2}" y="{right_bottom+52}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">num_warps=2 &#8594; 32&#215;2=64 线程 '
          f'(TritonGPUAttrDefs.td:L158-L161)</text>')

foot_y = h - 20
L.append(f'<text x="{w/2}" y="{foot_y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#64748b">同一个函数 L 的两种极端形态:distributed 服务'
          f'寄存器计算/合并访存/MMA;shared 服务跨线程共享</text>')
L.append('</svg>')
out = Path(__file__).parent / "fig-distributed-vs-shared.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
