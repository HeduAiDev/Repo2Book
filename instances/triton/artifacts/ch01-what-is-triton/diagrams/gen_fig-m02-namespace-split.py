#!/usr/bin/env python3
"""fig-m02-namespace-split: state-table 模板改成两列对照表——
tl 命名空间两套实现策略(core.py @builtin 原语 vs standard.py @jit 组合子)
与三岔分发的对应关系。列=两个模块,行=计数口径/去哪一岔。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "tl 命名空间是两套实现混装"
SUBTITLE = "core.py 的 @builtin 原语 vs standard.py 的 @jit 组合子——core.py / standard.py 计数(Triton v3.2.0)"
COLS = ["core.py 的 @builtin 原语", "standard.py 的 @jit 组合子"]
ROW_LABELS = ["模块级计数\n(grep -c '^@X')", "含 tensor 类内方法", "走哪一岔", "举例"]
CELLS = {
    "模块级计数\n(grep -c '^@X')": ["55", "30"],
    "含 tensor 类内方法": ["96\n(多出 41 个算术/访存方法)", "(不适用)"],
    "走哪一岔": ["② 直接建 IR op", "① 被内联进你的 kernel"],
    "举例": ["tl.program_id / tl.arange / tl.store", "tl.cdiv / tl.sort / tl.argmax / tl.cumsum"],
}
HIGHLIGHT_ROW = "走哪一岔"
STATUS = {"走哪一岔": ["changed_b", "changed_a"]}
COLOR = {"changed_b": ("#ecfdf5", "#047857"), "changed_a": ("#eff6ff", "#1d4ed8")}

LABEL_W, COL_W, HEADER_H, TOP, PAD = 190, 340, 40, 108, 34
ROW_HS = [56, 64, 56, 56]
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + sum(ROW_HS) + PAD + 40
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = []
y = TOP + HEADER_H
for rh in ROW_HS:
    row_y.append(y)
    y += rh

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
          'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
         f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
         f'fill="#64748b">{esc(SUBTITLE)}</text>')

for j, name in enumerate(COLS):  # 列头
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    lines = name.split(" ")
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    rh = ROW_HS[i]
    label_lines = row.split("\n")
    n_lbl = len(label_lines)
    y0_lbl = ry + rh / 2 - (n_lbl - 1) * 8 + 4
    for k, line in enumerate(label_lines):
        L.append(f'<text x="{PAD+LABEL_W-16}" y="{y0_lbl+k*16}" text-anchor="end" '
                  f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                  f'fill="#374151">{esc(line)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        lines = CELLS[row][j].split("\n")
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{rh-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        n = len(lines)
        y0 = ry + rh / 2 - (n - 1) * 8 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+(COL_W-8)/2}" y="{y0+k*15}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="11.5" fill="{text_fill}" '
                      f'{weight_attr}>{esc(line)}</text>')
    # 行分隔线
    if i > 0:
        L.append(f'<line x1="{PAD}" y1="{ry}" x2="{w-PAD}" y2="{ry}" '
                  'stroke="#e2e8f0" stroke-width="1"/>')

foot_y = h - PAD + 10
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">第②岔:第①岔 = 原语:组合子 = 55:30(模块级口径);含 tensor 类内方法则 @builtin 共 96</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-m02-namespace-split.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} size={w}x{h}")
