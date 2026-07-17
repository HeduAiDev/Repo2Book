#!/usr/bin/env python3
"""state-table 模板：每线程向量宽 = 三道 min 闸门（对齐/连续/硬件宽）取最小。
四行对应不同 divisibility 下的 perThread 结果，i32(elemBits=32)，128-bit cap=4。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "perThread = min(对齐, 连续, 128/bits) —— 三道闸门定每线程向量宽"
SUBTITLE = "i32(elemBits=32)：128-bit cap = 4 个元素；主线 divisibility=16 字节 → perThread=4（绿色行）"

COLS = ["divisibility\n(字节)", "maxMultiple\n=div/4", "maxContig", "128/32\ncap", "perThread\n(向量宽)", "结果"]
ROWS = [
    ("16", "4", "1024", "4", "4", "128-bit (v4)"),
    ("8", "2", "1024", "4", "2", "64-bit (v2)"),
    ("4", "1", "1024", "4", "1", "32-bit 标量"),
    ("1(无提示)", "1", "1024", "4", "1", "32-bit 标量"),
]
HL_ROW_IDX = 0
COLOR = ("#ecfdf5", "#047857")

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 30, 150, 52, 52, 112, 30
NUM_COLS = len(COLS)
w = PAD * 2 + LABEL_W + COL_W * NUM_COLS
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 40

col_x = [PAD + LABEL_W + i * COL_W for i in range(NUM_COLS)]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROWS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    lines = name.split("\n")
    ny0 = TOP + (HEADER_H - 6) / 2 - (len(lines) - 1) * 7 + 4
    for k, ln in enumerate(lines):
        L.append(f'<text x="{x+(COL_W-8)/2}" y="{ny0+k*14}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="white" '
                  f'font-weight="bold">{esc(ln)}</text>')

for i, row in enumerate(ROWS):
    ry = row_y[i]
    is_hl = (i == HL_ROW_IDX)
    for j, val in enumerate(row):
        cx = col_x[j]
        fill, stroke = (COLOR if is_hl else ("#f8fafc", "#cbd5e1"))
        L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if is_hl else 1}"/>')
        fsz = 11 if len(val) > 8 else 14
        text_fill = COLOR[1] if is_hl else "#374151"
        weight_attr = 'font-weight="bold" '
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fsz}" fill="{text_fill}" '
                  f'{weight_attr}>{esc(val)}</text>')

foot_y = h - PAD + 10
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">木桶效应：三道 min 任一约束退化即拉低全局。绿行=主线 %p：divisibility=16 → perThread=4，'
          f'1024 元素/4=256 笔向量事务，较标量 1024 笔省 4 倍。</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-perthread-vec-width.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
