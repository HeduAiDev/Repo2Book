#!/usr/bin/env python3
"""fig-layout-as-function-table —— 全章核心顿悟图。
2x2 张量的每个索引 i 映到一个线程集合 L(i);把抽象函数还原成一张可逐格核对的表。
数据来自 TritonGPUAttrDefs.td:L41-L49 顿悟例（见 explainer/traces/td_layout_tables.md §A）。
全坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "布局即函数:L(i) = 线程集合"
SUBTITLE = "2x2 张量,每个索引 i 映到一个线程集合(TritonGPUAttrDefs.td:L41-L44 逐字示例)"

# 行=i0(第一维),列=i1(第二维)
CELLS = [
    [("(0, 0)", "{0, 4}"), ("(0, 1)", "{1, 5}")],
    [("(1, 0)", "{2, 6}"), ("(1, 1)", "{3, 7}")],
]
PROVENANCE = ["TritonGPUAttrDefs.td:L41", "TritonGPUAttrDefs.td:L42",
              "TritonGPUAttrDefs.td:L43", "TritonGPUAttrDefs.td:L44"]

CELL_W, CELL_H, GAP, PAD, TOP = 220, 120, 14, 46, 140
n_rows, n_cols = 2, 2
grid_w = n_cols * (CELL_W + GAP) - GAP
grid_h = n_rows * (CELL_H + GAP) - GAP
w = PAD * 2 + grid_w
h = TOP + grid_h + 176

grid_x0 = PAD
grid_y0 = TOP

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+24}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 轴说明:标题下方一行水平文字(不用旋转文本,避免几何估算对 transform 误判)
L.append(f'<text x="{PAD}" y="{grid_y0-22}" font-family="sans-serif" font-size="12" '
         f'fill="#475569">行 = 第一维索引 i&#8320;(&#8595;方向),'
         f'列 = 第二维索引 i&#8321;(&#8594;方向)</text>')

idx = 0
for r in range(n_rows):
    for c in range(n_cols):
        cx = grid_x0 + c * (CELL_W + GAP)
        cy = grid_y0 + r * (CELL_H + GAP)
        coord, tset = CELLS[r][c]
        L.append(f'<rect x="{cx}" y="{cy}" width="{CELL_W}" height="{CELL_H}" rx="8" '
                  f'fill="#eff6ff" stroke="#3b82f6" stroke-width="2"/>')
        L.append(f'<text x="{cx+CELL_W/2}" y="{cy+30}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" fill="#1e3a5f">'
                  f'i = {esc(coord)}</text>')
        L.append(f'<text x="{cx+CELL_W/2}" y="{cy+64}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="20" font-weight="bold" '
                  f'fill="#1d4ed8">L(i) = {esc(tset)}</text>')
        L.append(f'<text x="{cx+CELL_W/2}" y="{cy+92}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10" fill="#64748b">'
                  f'{esc(PROVENANCE[idx])}</text>')
        idx += 1

# 底部结论条:值域并集 + 每格集合大小(拆两行,避免单行过长溢出画布)
note_y = grid_y0 + grid_h + 42
L.append(f'<rect x="{PAD}" y="{note_y-26}" width="{w-2*PAD}" height="94" rx="8" '
         f'fill="#fefce8" stroke="#ca8a04" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+18}" y="{note_y}" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#854d0e">值域并集 = {{0,1,2,3,4,5,6,7}} 共 8 个线程 '
         f'&#8212; 每格集合大小 = 2</text>')
L.append(f'<text x="{PAD+18}" y="{note_y+24}" font-family="sans-serif" font-size="12" '
         f'fill="#854d0e">普通张量每格 1 个持有者;这里每格是集合'
         f'(&#124;L(i)&#124;=2)</text>')
L.append(f'<text x="{PAD+18}" y="{note_y+46}" font-family="sans-serif" font-size="12" '
         f'fill="#854d0e">&#8212;这正是 GPU 张量与普通张量的分界。</text>')

L.append('</svg>')
out = Path(__file__).parent / "fig-layout-as-function-table.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
