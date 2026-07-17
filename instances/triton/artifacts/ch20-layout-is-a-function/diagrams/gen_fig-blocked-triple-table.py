#!/usr/bin/env python3
"""fig-blocked-triple-table —— Blocked 三元组把四级层次落成一张可核对的
16x16 线程号表(m05)。逐字复刻 TritonGPUAttrDefs.td:L601-L619 的 verbatim 示例
(rows 0-3 与 14-15 显式给出,中间 rows 4-13 用 "..." 省略,与 .td 一致,不外推)。
figure_id 说明:explainer.json 里 m04/m05 两条 figure_specs 撞了同一个
figure_id "fig-four-level-hierarchy"——本图是 m05 那条(state-table/Blocked
16x16),按内容重新命名为 fig-blocked-triple-table,避免 manifest 覆盖。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "Blocked 三元组落地 —— 16x16 网格 / 2 warp / 64 线程"
# 注:字体渲染实测 bold + "量" 字在本环境会缺字(tofu),标题避开"张量"改用"网格"
SUBTITLE = ("sizePerThread={2,2} threadsPerWarp={8,4} warpsPerCTA={1,2}"
            "(TritonGPUAttrDefs.td:L601-L619)")

# 行标签: 0,1,2,3, "...", 14,15  (与 .td 显式给出的行一致,不外推 4-13)
ROW_LABELS = ["0", "1", "2", "3", "...", "14", "15"]
# 每行 16 列(warp0: col0-7, warp1: col8-15),按 .td 给出的模式(每号占 2x2 块)
ROWS = [
    [0, 0, 1, 1, 2, 2, 3, 3,   32, 32, 33, 33, 34, 34, 35, 35],
    [0, 0, 1, 1, 2, 2, 3, 3,   32, 32, 33, 33, 34, 34, 35, 35],
    [4, 4, 5, 5, 6, 6, 7, 7,   36, 36, 37, 37, 38, 38, 39, 39],
    [4, 4, 5, 5, 6, 6, 7, 7,   36, 36, 37, 37, 38, 38, 39, 39],
    None,  # "..." 省略行(.td 原文即省略 rows 4-13,不外推填数)
    [28, 28, 29, 29, 30, 30, 31, 31,   60, 60, 61, 61, 62, 62, 63, 63],
    [28, 28, 29, 29, 30, 30, 31, 31,   60, 60, 61, 61, 62, 62, 63, 63],
]
# 需要核对的样点(来自 dossier worked_example)
CHECK_CELLS = {(0, 0): "TritonGPUAttrDefs.td:L601", (0, 2): "TritonGPUAttrDefs.td:L601",
               (2, 0): "TritonGPUAttrDefs.td:L603", (0, 8): "TritonGPUAttrDefs.td:L601",
               (5, 14): "TritonGPUAttrDefs.td:L618"}  # (row_idx_in_ROWS, col) -> 行14=索引5

N_COLS = 16
CELL_W, CELL_H = 42, 30
LABEL_W = 46
PAD, TOP = 40, 130
grid_w = N_COLS * CELL_W
grid_h = len(ROWS) * CELL_H
w = PAD * 2 + LABEL_W + grid_w
h = TOP + grid_h + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

grid_x0 = PAD + LABEL_W
grid_y0 = TOP

# warp 分区背景(col 0-7 = warp0 浅蓝,col 8-15 = warp1 浅橙)
L.append(f'<rect x="{grid_x0}" y="{grid_y0}" width="{8*CELL_W}" height="{grid_h}" '
          f'fill="#eff6ff"/>')
L.append(f'<rect x="{grid_x0+8*CELL_W}" y="{grid_y0}" width="{8*CELL_W}" height="{grid_h}" '
          f'fill="#fff7ed"/>')
# warp 分隔竖线(对应 .td 的 ";" 分隔)
sep_x = grid_x0 + 8 * CELL_W
L.append(f'<line x1="{sep_x}" y1="{grid_y0}" x2="{sep_x}" y2="{grid_y0+grid_h}" '
          f'stroke="#94a3b8" stroke-width="2" stroke-dasharray="4,3"/>')

for r, row in enumerate(ROWS):
    ry = grid_y0 + r * CELL_H
    L.append(f'<text x="{grid_x0-10}" y="{ry+CELL_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="11" fill="#475569">'
              f'{esc(ROW_LABELS[r])}</text>')
    if row is None:  # 省略行,与 .td 一致不外推
        L.append(f'<text x="{grid_x0+8*CELL_W}" y="{ry+CELL_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="14" fill="#94a3b8">'
                  f'... (rows 4-13,.td 原文省略,不外推) ...</text>')
        continue
    for c, val in enumerate(row):
        cx = grid_x0 + c * CELL_W
        is_check = (r, c) in CHECK_CELLS
        stroke = "#dc2626" if is_check else "#cbd5e1"
        sw = 2 if is_check else 0.7
        L.append(f'<rect x="{cx+1}" y="{ry+1}" width="{CELL_W-2}" height="{CELL_H-2}" '
                  f'fill="none" stroke="{stroke}" stroke-width="{sw}"/>')
        weight = 'font-weight="bold" ' if is_check else ''
        fill = "#b91c1c" if is_check else "#334155"
        L.append(f'<text x="{cx+CELL_W/2}" y="{ry+CELL_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11" {weight}fill="{fill}">'
                  f'{val}</text>')

col_label_y = grid_y0 - 8
L.append(f'<text x="{grid_x0+4*CELL_W}" y="{col_label_y}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#1d4ed8">warp 0(线程 0-31)'
          f'</text>')
L.append(f'<text x="{grid_x0+12*CELL_W}" y="{col_label_y}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#c2410c">warp 1(线程 32-63)'
          f'</text>')

note_y = grid_y0 + grid_h + 34
L.append(f'<rect x="{PAD}" y="{note_y-22}" width="{w-2*PAD}" height="112" rx="8" '
          f'fill="#fef2f2" stroke="#dc2626" stroke-width="1.2"/>')
L.append(f'<text x="{PAD+16}" y="{note_y}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#991b1b">红框 = 核对样点:(0,0)&#8594;0,'
          f'(0,2)&#8594;1,(2,0)&#8594;4</text>')
L.append(f'<text x="{PAD+16}" y="{note_y+22}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#991b1b">(0,8)&#8594;32,(14,14)&#8594;63'
          f'(TritonGPUAttrDefs.td:L601/L603/L618)</text>')
L.append(f'<text x="{PAD+16}" y="{note_y+46}" font-family="sans-serif" font-size="12" '
          f'fill="#991b1b">每个线程号占一个连续 2&#215;2 小块(sizePerThread={{2,2}}),'
          f'故每号在表中出现 4 次;256 元素/64 线程=4元素/线程,严格划分。</text>')
L.append(f'<text x="{PAD+16}" y="{note_y+70}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">block 线程总数 = warpsPerCTA(1&#215;2=2) &#215; '
          f'threadsPerWarp(8&#215;4=32) = 64,与「模块契约」一节 num_warps=2 自洽</text>')

L.append('</svg>')
out = Path(__file__).parent / "fig-blocked-triple-table.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
