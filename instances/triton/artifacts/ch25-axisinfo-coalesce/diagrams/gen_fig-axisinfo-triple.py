#!/usr/bin/env python3
"""state-table: AxisInfo 三元组——一根轴的三张访存体检报告(全为 2 的幂)。
列 = 三元组分量 + 读法；行 = header 例子 + 主线传播链上的关键值。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "AxisInfo 三元组 — 一根轴的三张访存体检报告"
SUBTITLE = "header 例：[10,11,12,13,18,19,20,21]（AxisInfo.h 头注图例）；主线：x_ptr + tl.arange(0,1024), i32"

COLS = ["contiguity\n(顺读多长)", "divisibility\n(字节对齐)", "constancy\n(重复多长)", "读法"]
ROW_LABELS = ["header 例\n[10..13,18..21]", "%p（主线 addptr\n结果指针张量）"]
CELLS = {
    "header 例\n[10..13,18..21]": ["4", "—", "—", ["前 4 个连号(10-13)", "断在 18，顺读最长 4"]],
    "%p（主线 addptr\n结果指针张量）": ["1024", "16", "1", ["整段连续 1024；起点 16", "字节对齐；无重复"]],
}
HIGHLIGHT_ROW = "%p（主线 addptr\n结果指针张量）"
COLOR = ("#ecfdf5", "#047857")

LABEL_W, COL_W, READ_W, ROW_H, HEADER_H, TOP, PAD = 190, 150, 250, 68, 46, 108, 30
NUM_NUM_COLS = 3
w = PAD * 2 + LABEL_W + COL_W * NUM_NUM_COLS + READ_W
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 34

COL_WIDTHS = [COL_W, COL_W, COL_W, READ_W]
col_x = []
_acc = PAD + LABEL_W
for cw in COL_WIDTHS:
    col_x.append(_acc)
    _acc += cw
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 行标签列头（空白角）
L.append(f'<rect x="{PAD}" y="{TOP}" width="{LABEL_W-8}" height="{HEADER_H-6}" rx="3" '
          'fill="#475569"/>')
L.append(f'<text x="{PAD+(LABEL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" fill="white" font-weight="bold">'
          f'{esc("Value")}</text>')

for j, name in enumerate(COLS):
    x = col_x[j]
    cw = COL_WIDTHS[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{cw-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    lines = name.split("\n")
    ny0 = TOP + (HEADER_H - 6) / 2 - (len(lines) - 1) * 7 + 4
    for k, ln in enumerate(lines):
        L.append(f'<text x="{x+(cw-8)/2}" y="{ny0+k*14}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="white" '
                  f'font-weight="bold">{esc(ln)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    is_hl = (row == HIGHLIGHT_ROW)
    lbl_fill, lbl_stroke = (COLOR[0], COLOR[1]) if is_hl else ("white", "#cbd5e1")
    L.append(f'<rect x="{PAD}" y="{ry+4}" width="{LABEL_W-8}" height="{ROW_H-8}" rx="4" '
              f'fill="{lbl_fill}" stroke="{lbl_stroke}" stroke-width="{2 if is_hl else 1}"/>')
    lbl_lines = row.split("\n")
    lbl_y0 = ry + ROW_H / 2 - (len(lbl_lines) - 1) * 8 + 4
    for k, ln in enumerate(lbl_lines):
        L.append(f'<text x="{PAD+14}" y="{lbl_y0+k*16}" text-anchor="start" '
                  f'font-family="sans-serif" font-size="12" font-weight="bold" '
                  f'fill="{COLOR[1] if is_hl else "#374151"}">{esc(ln)}</text>')
    for j in range(NUM_NUM_COLS + 1):
        cx = col_x[j]
        cw = COL_WIDTHS[j]
        cell = CELLS[row][j]
        fill, stroke = (COLOR if is_hl else ("#f8fafc", "#cbd5e1"))
        L.append(f'<rect x="{cx}" y="{ry+4}" width="{cw-8}" height="{ROW_H-8}" rx="4" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if is_hl else 1}"/>')
        text_fill = COLOR[1] if is_hl else "#374151"
        if isinstance(cell, list):  # 读法列:两行说明文字
            y0 = ry + ROW_H / 2 - (len(cell) - 1) * 8 + 4
            for k, ln in enumerate(cell):
                L.append(f'<text x="{cx+(cw-8)/2}" y="{y0+k*16}" text-anchor="middle" '
                          f'font-family="sans-serif" font-size="12" fill="{text_fill}">'
                          f'{esc(ln)}</text>')
        else:
            L.append(f'<text x="{cx+(cw-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="16" fill="{text_fill}" '
                      f'font-weight="bold">{esc(cell)}</text>')

foot_y = h - PAD + 8
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">绿高亮 = 主线例最终值(1024, 16 字节, 1)：既长又对齐，是 Coalesce 敢向量化的静态凭据。三分量恒为 2 的幂。</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-axisinfo-triple.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
