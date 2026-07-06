#!/usr/bin/env python3
"""fig35-7-e8m0-rounding: e8m0 块 scale 只存 8 位指数、无尾数，所以连续 amax scale 必须向上
取整到最近的 2 的幂——scales_raw = exp2(ceil(log2(absmax/448)))。state-table，4 行 absmax 样例。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "e8m0 块 scale：连续 amax/448 必须向上取整到最近的 2 的幂"
SUBTITLE = "scale_raw = absmax / FP8_MAX(448.0)；k = ceil(log2(scale_raw))；scale = 2^k（e8m0：8 位指数，0 位尾数）"

COLS = ["absmax", "scale_raw = absmax/448", "k = ceil(log2)", "scale = 2^k", "overshoot %"]
ROWS = [
    ["7.0", "0.015625", "-6", "0.015625", "0.00"],
    ["100.0", "0.223214", "-2", "0.25", "12.00"],
    ["300.0", "0.669643", "0", "1.0", "49.33"],
    ["1000.0", "2.232143", "2", "4.0", "79.20"],
]
OVERSHOOT_COLOR = {
    "0.00": ("#ecfdf5", "#047857"),
    "12.00": ("#fef9c3", "#b45309"),
    "49.33": ("#ffedd5", "#c2410c"),
    "79.20": ("#fee2e2", "#b91c1c"),
}

COL_W = [90, 190, 130, 120, 130]
LABEL_X = 40
ROW_H = 42
HEADER_H = 56
TOP = 118
PAD = 40

col_x = []
x = LABEL_X
for cw in COL_W:
    col_x.append(x)
    x += cw
table_w = x
w = table_w + PAD * 2
FOOT_N = 3
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + FOOT_N * 18 + 10

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16.5" '
     f'fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    cx = PAD + col_x[j]
    cw = COL_W[j]
    L.append(f'<rect x="{cx}" y="{TOP}" width="{cw-6}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{cx+(cw-6)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white">{esc(name)}</text>')

for i, row in enumerate(ROWS):
    ry = TOP + HEADER_H + i * ROW_H
    for j, val in enumerate(row):
        cx = PAD + col_x[j]
        cw = COL_W[j]
        fill, stroke, sw = "#f8fafc", "#cbd5e1", 1
        text_fill = "#1e293b"
        if j == 4 and val in OVERSHOOT_COLOR:
            fill, stroke = OVERSHOOT_COLOR[val]
            sw = 2
            text_fill = stroke
        L.append(f'<rect x="{cx}" y="{ry+3}" width="{cw-6}" height="{ROW_H-6}" rx="3" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        L.append(f'<text x="{cx+(cw-6)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" fill="{text_fill}">{esc(val)}</text>')

foot_y = TOP + HEADER_H + ROW_H * len(ROWS) + 24
foot_lines = [
    "absmax=7.0 恰好是 2 的幂，overshoot=0%；absmax 越偏离 2 的幂次，取整浪费的余量越大（最高到 79.20%）。",
    "ceil（向上取整）是硬件必须的方向：FP8 格永远不会因为 scale 偏小而裁掉块内最大值。",
    "OCP Microscaling(MX) FP8 的硬件约定在 vllm 的落地；DeepSeek 系 128x128 块量化用的正是这套 e8m0 scale。",
]
for i, line in enumerate(foot_lines):
    L.append(f'<text x="{PAD}" y="{foot_y+i*18}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig35-7-e8m0-rounding.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
