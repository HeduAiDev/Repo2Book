#!/usr/bin/env python3
"""state-table 模板:状态逐轮演化/数值追踪表。列=迭代步,行=追踪变量,
一行高亮(每格按 stable/changed 语义上色)。
改造点:COLS(列标题)、ROW_LABELS(行名)、CELLS(每格文本,支持多行 "\n")、
HIGHLIGHT_ROW + STATUS(高亮行每列的语义色)。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "Online Softmax — 3 轮迭代状态追踪"
SUBTITLE = "跟踪单个 Q token 经过 K0V0、K1V1、K2V2;初始 m=-inf, l=0, O_acc=0"
COLS = ["Iter 1 (K0,V0)", "Iter 2 (K1,V1)", "Iter 3 (K2,V2)"]
ROW_LABELS = ["S", "m", "corr", "l", "O_acc"]
CELLS = {
    "S":     ["S0=[2.0,1.0,0.5,3.0]", "S1=[1.5,4.0,2.0,1.0]", "S2=[3.5,2.0,1.5,0.5]"],
    "m":     ["-inf -> 3.0", "3.0 -> 4.0", "4.0 -> 4.0"],
    "corr":  ["= 0(无历史)", "= exp(3.0-4.0)\n= 0.368", "= exp(4.0-4.0)\n= 1.0"],
    "l":     ["0 -> 1.585", "1.585*0.368+sum(P1)\n= 1.850", "1.850*1.0+sum(P2)\n= ..."],
    "O_acc": ["0 -> P0@V0", "0.368*old + P1@V1", "1.0*old + P2@V2"],
}
HIGHLIGHT_ROW = "corr"
STATUS = {"corr": ["stable", "changed", "stable"]}  # 高亮行每列的语义色
COLOR = {"stable": ("#ecfdf5", "#047857"), "changed": ("#fee2e2", "#b91c1c")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 90, 210, 56, 34, 96, 30
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):  # 列头
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):  # 行标签 + 单元格
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" '
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
                      f'font-family="sans-serif" font-size="12" fill="{text_fill}" '
                      f'{weight_attr}>{esc(line)}</text>')

foot_y = h - PAD + 4
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">绿=本轮值不变(stable),红=本轮发生更新(changed)</text>')
L.append('</svg>')
out = Path(__file__).with_name("example-softmax-trace.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
