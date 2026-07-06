#!/usr/bin/env python3
"""fig35-1-affine-grid: 非对称仿射量化把权重区间铺满整个整数格，最大误差被钉在半格以内。
state-table：6 列（w / w÷scale / round / code / 反量化 ŵ / |w-ŵ|），6 行数据。
最后一列按误差大小上色；code 列端点 0 与 15 加边框标出"格子用满"。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "非对称 4-bit 仿射量化：scale=0.2, zero_point=5 把 [-1.0, 2.0] 铺满 16 格"
SUBTITLE = "w=[-1.0,-0.32,0.24,0.68,1.36,2.0]；code = round(w/scale) + zero_point，clamp 到 [0,15]"

COLS = ["w", "w / scale", "round()", "code\n(zp=5, clamp[0,15])", "ŵ = (code-zp)*scale", "|w - ŵ|"]
ROWS = [
    ["-1.00", "-5.00", "-5", "0", "-1.00", "0.00"],
    ["-0.32", "-1.60", "-2", "3", "-0.40", "0.08"],
    ["0.24", "1.20", "1", "6", "0.20", "0.04"],
    ["0.68", "3.40", "3", "8", "0.60", "0.08"],
    ["1.36", "6.80", "7", "12", "1.40", "0.04"],
    ["2.00", "10.00", "10", "15", "2.00", "0.00"],
]
ERR_COLOR = {
    "0.00": ("#ecfdf5", "#047857"),
    "0.04": ("#fef9c3", "#b45309"),
    "0.08": ("#fee2e2", "#b91c1c"),
}
CODE_ENDPOINTS = {"0", "15"}

COL_W = [80, 90, 74, 150, 150, 100]
LABEL_X = 40
ROW_H = 40
HEADER_H = 54
TOP = 96
PAD = 40
FOOT_LINES = 3

col_x = []
x = LABEL_X
for wcol in COL_W:
    col_x.append(x)
    x += wcol
table_w = x
w = table_w + PAD * 2
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + FOOT_LINES * 18 + 10

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
     f'fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# header row
for j, name in enumerate(COLS):
    cx = PAD + col_x[j]
    cw = COL_W[j]
    L.append(f'<rect x="{cx}" y="{TOP}" width="{cw-6}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    lines = name.split("\n")
    n = len(lines)
    y0 = TOP + (HEADER_H-6)/2 - (n-1)*7 + 4
    for k, line in enumerate(lines):
        L.append(f'<text x="{cx+(cw-6)/2}" y="{y0+k*14}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11.5" fill="white" '
                  f'font-weight="bold">{esc(line)}</text>')

# data rows
for i, row in enumerate(ROWS):
    ry = TOP + HEADER_H + i * ROW_H
    for j, val in enumerate(row):
        cx = PAD + col_x[j]
        cw = COL_W[j]
        fill, stroke, sw = "#f8fafc", "#cbd5e1", 1
        text_fill = "#1e293b"
        bold = ""
        if j == 5 and val in ERR_COLOR:
            fill, stroke = ERR_COLOR[val]
            sw = 2
            text_fill = stroke
            bold = 'font-weight="bold" '
        if j == 3 and val in CODE_ENDPOINTS:
            stroke = "#1e3a5f"
            sw = 2.5
        L.append(f'<rect x="{cx}" y="{ry+3}" width="{cw-6}" height="{ROW_H-6}" rx="3" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        L.append(f'<text x="{cx+(cw-6)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" fill="{text_fill}" '
                  f'{bold}>{esc(val)}</text>')

foot_y = TOP + HEADER_H + ROW_H * len(ROWS) + 22
foot_lines = [
    "code 列的 0 与 15（粗边框）= 权重两端点恰好落在格子边缘，16 格全部用满，一格不浪费。",
    "最大误差 |w-ŵ| = 0.08（红），未超过半格 half_scale = scale/2 = 0.10（0.08/0.1 = 80%）。",
    "对照同一向量的对称量化：scale = 0.2857，最大误差 0.1429（= 其半格上界）——比非对称大 79%。",
]
for i, line in enumerate(foot_lines):
    L.append(f'<text x="{PAD}" y="{foot_y+i*18}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig35-1-affine-grid.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
