#!/usr/bin/env python3
"""state-table 模板：fp8 家族 + 16/32 bit 基准的 (mantissa, exponent_bias) 三元组
对照表——同位宽内精度与量程互换（8 bit 行 vs 16 bit 行 vs fp32 基准行），
颜色按位宽分组。数据同源 traces/fp8_encoding.txt。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "fp8 家族与 16 / 32 bit 基准 —— (mantissa, exponent_bias) 决定精度↔动态范围"
SUBTITLE = "同一条位宽线上，尾数位数与 exponent_bias 此消彼长；数据来自 traces/fp8_encoding.txt"
COLS = ["bitwidth", "mantissa", "exponent_bias", "相对 fp32 带宽"]
ROWS = [
    ("fp8e4nv", ["8", "3", "7", "0.250"], "bit8"),
    ("fp8e5",   ["8", "2", "15", "0.250"], "bit8"),
    ("fp16",    ["16", "10", "15", "0.500"], "bit16"),
    ("bf16",    ["16", "7", "127", "0.500"], "bit16"),
    ("fp32",    ["32", "23", "127", "1.000"], "base"),
]
GROUP_COLOR = {
    "bit8":  ("#fef3c7", "#b45309"),
    "bit16": ("#dbeafe", "#1d4ed8"),
    "base":  ("#f1f5f9", "#475569"),
}
GROUP_LABEL = [("bit8", "8 bit（fp8 变体）"), ("bit16", "16 bit（fp16 / bf16）"), ("base", "fp32 基准（32 bit）")]

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 130, 176, 50, 36, 108, 34
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 30
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROWS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 列头
L.append(f'<rect x="{PAD}" y="{TOP}" width="{LABEL_W-8}" height="{HEADER_H-6}" rx="3" '
          'fill="#334155"/>')
L.append(f'<text x="{PAD+(LABEL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" fill="white" font-weight="bold">'
          f'{esc("dtype")}</text>')
for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

# 行
for i, (dtype, cells, group) in enumerate(ROWS):
    ry = row_y[i]
    fill, stroke = GROUP_COLOR[group]
    L.append(f'<rect x="{PAD}" y="{ry+3}" width="{LABEL_W-8}" height="{ROW_H-6}" rx="4" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{PAD+(LABEL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{stroke}">{esc(dtype)}</text>')
    for j, val in enumerate(cells):
        cx = col_x[j]
        L.append(f'<rect x="{cx}" y="{ry+3}" width="{COL_W-8}" height="{ROW_H-6}" rx="4" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" fill="#1e293b">{esc(val)}</text>')

# 图例
ly = h - 30
lx = PAD
for key, label in GROUP_LABEL:
    fill, stroke = GROUP_COLOR[key]
    L.append(f'<rect x="{lx}" y="{ly}" width="16" height="16" rx="3" fill="{fill}" stroke="{stroke}"/>')
    L.append(f'<text x="{lx+22}" y="{ly+13}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(label)}</text>')
    lx += 22 + 12 * len(label) + 26

L.append('</svg>')
out = Path(__file__).with_name("fig-fp8-tradeoff.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
