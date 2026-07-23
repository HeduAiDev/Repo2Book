#!/usr/bin/env python3
"""tensor-flow 模板:m5 前导维批处理化——两路张量各自 expand_dims+broadcast,
在最外层拼出 size 维,原 8 维内存布局/stride 不动。全坐标计算,零魔数。
box 标题即算子名(含 axis),不再叠加箭头中点文字,避免窄列间距下文字互压。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

ROWS = [
    ("offset 张量(program_id 派生)", "tensor<8>",
     "expand_dims axis=1", "tensor<5x1>",
     "broadcast", "tensor<5x8>"),
    ("range 张量(make_range)", "tensor<8>",
     "expand_dims axis=0", "tensor<1x8>",
     "broadcast", "tensor<5x8>"),
]
SRC_NOTE = ["axis=1: auto_blockify.mlir:L34", "axis=0: auto_blockify.mlir:L37",
            "broadcast: mlir:L35/L38"]
MERGE_LABEL = "相加 → 5×8 偏移张量"
FOOT = "getExpandedType:size(=5) 恒拼在 targetShape 最前(位置 0),原 8 维内存布局与 stride 不变 (Utils.cpp:L33-42)"

BOX_W, BOX_H = 200, 62
COL_GAP = 70
ROW_GAP = 150
PAD, TOP = 40, 96
n_col = 3
cols_x = [PAD + i * (BOX_W + COL_GAP) for i in range(n_col)]
rows_y = [TOP + i * ROW_GAP for i in range(len(ROWS))]
merge_x = cols_x[-1] + BOX_W + COL_GAP + 20
merge_y = (rows_y[0] + rows_y[1]) / 2
w = merge_x + BOX_W + PAD
h = rows_y[-1] + BOX_H + 100

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-24}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">'
     f'{esc("前导维批处理化 — size=5 恒拼在 shape 最前 (auto_blockify.mlir 夹具)")}</text>']

def box(x, y, title, shape, fill, stroke):
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{y+23}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11.5" fill="#334155">{esc(title)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{y+46}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14.5" font-weight="bold" fill="#0f172a">{esc(shape)}</text>')

for r, (name0, sh0, op1, sh1, op2, sh2) in enumerate(ROWS):
    y = rows_y[r]
    box(cols_x[0], y, name0, sh0, "#e2e8f0", "#64748b")
    box(cols_x[1], y, op1, sh1, "#dbeafe", "#1d4ed8")
    box(cols_x[2], y, op2, sh2, "#dbeafe", "#1d4ed8")
    cy = y + BOX_H / 2
    for c in range(2):
        x1 = cols_x[c] + BOX_W
        x2 = cols_x[c + 1]
        L.append(f'<line x1="{x1}" y1="{cy}" x2="{x2-4}" y2="{cy}" '
                  'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    x1 = cols_x[2] + BOX_W
    L.append(f'<line x1="{x1}" y1="{cy}" x2="{merge_x-4}" y2="{merge_y + (18 if r == 0 else -18)}" '
              'stroke="#b45309" stroke-width="1.6" marker-end="url(#b)"/>')

box(merge_x, merge_y - BOX_H / 2, "相加(elementwise add)", "tensor<5x8>", "#fef3c7", "#b45309")
L.append(f'<text x="{merge_x+BOX_W/2}" y="{merge_y - BOX_H/2 - 12}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
          f'fill="#92400e">{esc(MERGE_LABEL)}</text>')

# small source-note strip under the two data rows, above footer
note_y = rows_y[-1] + BOX_H + 34
for i, note in enumerate(SRC_NOTE):
    L.append(f'<text x="{PAD}" y="{note_y + i*15}" font-family="sans-serif" '
              f'font-size="10.6" fill="#94a3b8">{esc(note)}</text>')

L.append(f'<text x="{w/2}" y="{h-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#64748b">{esc(FOOT)}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-m5-leading-dim.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f"wrote {out}")
