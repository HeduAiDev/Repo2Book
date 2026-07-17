#!/usr/bin/env python3
"""fig-reduce-butterfly (state-table 模板)
32 车道蝶形归约 5 步:N=16,8,4,2,1,每步 shuffleXor 取对折邻居 combine,
lane0 累积和 16->48->112->240->496,覆盖车道数每步翻倍 2->4->8->16->32。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "warp 内 32 车道蝶形归约(shuffleXor)——5 步覆盖全和"
SUBTITLE = "lane i 初始持值 i(0..31),combine=sum;取邻居原语 shuffleXor(N),N 依次 16,8,4,2,1"

STEPS = ["初始", "步1", "步2", "步3", "步4", "步5"]
ROWS = ["N(shuffleXor 距离)", "lane0 累积 acc", "覆盖车道数"]
DATA = {
    "N(shuffleXor 距离)": ["-", "16", "8", "4", "2", "1"],
    "lane0 累积 acc":     ["0", "16", "48", "112", "240", "496"],
    "覆盖车道数":          ["1", "2", "4", "8", "16", "32"],
}
HIGHLIGHT_COL = 5  # 最后一步:全和达成
HIGHLIGHT_ROWS = {"lane0 累积 acc", "覆盖车道数"}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 168, 128, 52, 36, 96, 34
w = PAD * 2 + LABEL_W + COL_W * len(STEPS)
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 40

col_x = [PAD + LABEL_W + i * COL_W for i in range(len(STEPS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROWS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(STEPS):
    x = col_x[j]
    hot = (j == HIGHLIGHT_COL)
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              f'fill="{"#1d4ed8" if hot else "#3b82f6"}" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROWS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+5}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    for j in range(len(STEPS)):
        cx = col_x[j]
        val = DATA[row][j]
        hot_cell = (j == HIGHLIGHT_COL and row in HIGHLIGHT_ROWS)
        if hot_cell:
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      'fill="#dcfce7" stroke="#059669" stroke-width="2"/>')
        text_fill = "#047857" if hot_cell else "#374151"
        weight_attr = 'font-weight="bold" ' if hot_cell else ''
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="monospace" font-size="13" fill="{text_fill}" '
                  f'{weight_attr}>{esc(val)}</text>')

foot_y = TOP + HEADER_H + ROW_H * len(ROWS) + 26
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc("步数 = log2(32) = 5(warpReduce 循环 N=16..1);全和 496 = Σ(0..31),绿框=达成全和")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-reduce-butterfly.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  {w}x{h}")
