#!/usr/bin/env python3
"""ch37-fig-occupancy-contrast: state-table 模板。
同 blockDim=128 下，n_regs 从 28 涨到 212 把 occupancy 从 100% 压到 8.33%——
n_regs 是占用率的直接杠杆；heavy_kernel 的 n_spills=8 另说明 spill 是慢信号。
全部坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "occupancy = min(寄存器限, 线程限, 共享限)：n_regs 是占用率的直接杠杆"
SUBTITLE = "SM 资源上限（真机实测）：寄存器 65536 / 最大线程 1536 / 共享内存 102400；三个真实 kernel 同 blockDim=128 对照"

COLS = ["add_kernel", "mm_kernel", "heavy_kernel"]
ROW_LABELS = ["n_regs", "寄存器限 (blocks/SM)", "线程限 (blocks/SM)", "共享限 (blocks/SM)", "min → occupancy"]
CELLS = {
    "n_regs":               ["28", "212", "26"],
    "寄存器限 (blocks/SM)":  ["floor(65536/(28×128))\n= 18", "floor(65536/(212×128))\n= 2", "（不受限）"],
    "线程限 (blocks/SM)":    ["floor(1536/128) = 12", "floor(1536/128) = 12", "floor(1536/128) = 12"],
    "共享限 (blocks/SM)":    ["∞（shared=0）", "floor(102400/65536)\n= 1", "n_spills=8\n→走 local memory"],
    "min → occupancy":       ["min=12 → 1536/1536\n= 100%", "min=1 → 128/1536\n= 8.33%", "spill>0：每次溢出\n访存一趟高延迟 global"],
}
HIGHLIGHT_ROW = "min → occupancy"
STATUS = {"min → occupancy": ["good", "bad", "warn"]}
COLOR = {"good": ("#ecfdf5", "#047857"), "bad": ("#fee2e2", "#b91c1c"), "warn": ("#fff7ed", "#c2410c")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 190, 250, 62, 34, 108, 30
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 20
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>']

# 副标题换行处理（较长）
sub_words_lines = [SUBTITLE]
for i, line in enumerate(sub_words_lines):
    L.append(f'<text x="{PAD}" y="{PAD+20+i*16}" font-family="sans-serif" font-size="11.5" '
              f'fill="#64748b">{esc(line)}</text>')

for j, name in enumerate(COLS):  # 列头
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="monospace" font-size="12.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):  # 行标签 + 单元格
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        lines = CELLS[row][j].split("\n")
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        n = len(lines)
        y0 = ry + ROW_H / 2 - (n - 1) * 8 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+(COL_W-8)/2}" y="{y0+k*16}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="11.5" fill="{text_fill}" '
                      f'{weight_attr}>{esc(line)}</text>')

foot_y = h - PAD + 10
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">绿=占用率健康,红=占用率受限（共享内存/寄存器挤占）,橙=spill(每次溢出多一趟高延迟访存,慢信号)。</text>')
L.append('</svg>')
out = Path(__file__).with_name("ch37-fig-occupancy-contrast.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} w={w} h={h}")
