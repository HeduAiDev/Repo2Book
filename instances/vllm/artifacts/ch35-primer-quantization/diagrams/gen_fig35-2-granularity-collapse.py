#!/usr/bin/env python3
"""fig35-2-granularity-collapse: 单一 per-tensor scale 下，离群通道独占几乎全部量化档位，
其余通道被压到只剩个位数有效档位。state-table：4 通道 x (channel absmax m_i / 有效档位)。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "8-bit per-tensor 激活量化：单一 tensor absmax 让离群通道独吞满格"
SUBTITLE = "有效档位 = 256 * m_i / m（m = tensor absmax，256 = 8-bit 满格）；channel 1 是 100x 离群通道"

COLS = ["channel", "channel absmax m_i", "有效档位 = 256*m_i/m"]
ROWS = [
    ["0", "1.136", "1.78"],
    ["1 (outlier, 100x)", "163.4783", "256.00"],
    ["2", "1.643", "2.57"],
    ["3", "1.7321", "2.71"],
]
OUTLIER_ROW = 1
COL_W = [170, 190, 220]
LABEL_X = 40
ROW_H = 46
HEADER_H = 46
TOP = 100
PAD = 40

col_x = []
x = LABEL_X
for wcol in COL_W:
    col_x.append(x)
    x += wcol
table_w = x
w = table_w + PAD * 2
FOOT_LINES = 3
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + FOOT_LINES * 18 + 30

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16.5" '
     f'fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    cx = PAD + col_x[j]
    cw = COL_W[j]
    L.append(f'<rect x="{cx}" y="{TOP}" width="{cw-6}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{cx+(cw-6)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white">{esc(name)}</text>')

for i, row in enumerate(ROWS):
    ry = TOP + HEADER_H + i * ROW_H
    is_outlier = (i == OUTLIER_ROW)
    row_fill = "#dbeafe" if is_outlier else "#fee2e2"
    row_stroke = "#1d4ed8" if is_outlier else "#b91c1c"
    for j, val in enumerate(row):
        cx = PAD + col_x[j]
        cw = COL_W[j]
        L.append(f'<rect x="{cx}" y="{ry+4}" width="{cw-6}" height="{ROW_H-8}" rx="3" '
                  f'fill="{row_fill}" stroke="{row_stroke}" stroke-width="2"/>')
        L.append(f'<text x="{cx+(cw-6)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" fill="{row_stroke}">{esc(val)}</text>')

legend_y = TOP + HEADER_H + ROW_H * len(ROWS) + 24
L.append(f'<rect x="{PAD}" y="{legend_y-12}" width="16" height="16" rx="3" fill="#dbeafe" stroke="#1d4ed8" stroke-width="2"/>')
L.append(f'<text x="{PAD+22}" y="{legend_y+1}" font-family="sans-serif" font-size="12" fill="#334155">离群通道：独占满格 256 档</text>')
L.append(f'<rect x="{PAD+260}" y="{legend_y-12}" width="16" height="16" rx="3" fill="#fee2e2" stroke="#b91c1c" stroke-width="2"/>')
L.append(f'<text x="{PAD+282}" y="{legend_y+1}" font-family="sans-serif" font-size="12" fill="#334155">普通通道：塌缩到个位数档位</text>')

foot_y = legend_y + 30
foot_lines = [
    "tensor absmax m = 163.4783（由 channel 1 决定）；full_levels = 256（8-bit 满格）。",
    "channel 1（离群，100x）保住全部 256 档；其余三个通道最低跌到 1.78 档——不到 2 个刻度。",
    "per-tensor 激活量化因此崩溃：per-token / per-channel 才能救回精度（详见正文）。",
]
for i, line in enumerate(foot_lines):
    L.append(f'<text x="{PAD}" y="{foot_y+i*18}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig35-2-granularity-collapse.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
