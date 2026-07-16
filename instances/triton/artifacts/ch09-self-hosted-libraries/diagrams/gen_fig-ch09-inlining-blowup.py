#!/usr/bin/env python3
"""state-table 模板改造:tl.sort 内联后 IR 膨胀随块长增长(tt.call 恒为 0)。
数据来自 traces/ir_metrics.json sort_inlining。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "tl.sort 是宏,不是调用:内联后 IR 随块长膨胀"
SUBTITLE = "Triton v3.2.0 headless 精确编译实测(TTIR,make_ttir 之后)"
COLS = ["块长 n=16", "块长 n=64", "块长 n=1024"]
ROW_LABELS = ["log2(n)=阶段数", "arith.select(=CAS数)", "TTIR 行数", "tt.call(真调用)"]
CELLS = {
    "log2(n)=阶段数":       ["4", "6", "10"],
    "arith.select(=CAS数)": ["10", "21", "55"],
    "TTIR 行数":            ["452", "779", "1781"],
    "tt.call(真调用)":       ["0", "0", "0"],
}
HIGHLIGHT_ROW = "tt.call(真调用)"
STATUS = {"tt.call(真调用)": ["stable", "stable", "stable"],
          "arith.select(=CAS数)": ["changed", "changed", "changed"]}
COLOR = {"stable": ("#ecfdf5", "#047857"), "changed": ("#fee2e2", "#b91c1c")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 200, 200, 56, 34, 96, 30
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 34
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

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        text = CELLS[row][j]
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="14" fill="{text_fill}" '
                  f'{weight_attr}>{esc(text)}</text>')

foot_y = h - PAD + 4
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">绿=恒定(tt.call 全 0,完全内联);红=随块长翻倍单调增长(CAS 数)</text>')
L.append(f'<text x="{PAD}" y="{foot_y+18}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">n 从 16 -&gt; 1024(x64)时 CAS 从 10 -&gt; 55(x5.5),TTIR 从 452 -&gt; 1781 行(x3.9)</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch09-inlining-blowup.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
