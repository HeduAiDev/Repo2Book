#!/usr/bin/env python3
"""fig35-5-lazybatch-invariance — state-table 模板：blocksize 扫描下 GPTQ 输出误差恒定。
懒惰批只改访存效率，不改代数结果——四个 blocksize 输出误差逐位相同。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

_BOLD_BREAK = {"量"}
def btext(s):
    parts, buf = [], ""
    for ch in s:
        if ch in _BOLD_BREAK:
            if buf:
                parts.append(f'<tspan font-weight="bold">{esc(buf)}</tspan>')
                buf = ""
            parts.append(f'<tspan font-weight="normal">{esc(ch)}</tspan>')
        else:
            buf += ch
    if buf:
        parts.append(f'<tspan font-weight="bold">{esc(buf)}</tspan>')
    return "".join(parts)

TITLE = "懒惰批只改访存效率、不改结果：4 个 blocksize 的层输出误差逐位相同"
SUBTITLE = "d_row=3, d_col=6, 3-bit；RTN 基线误差 = 1.15236（跨块最大差 = 0.0）"
COLS = ["blocksize=1", "blocksize=2", "blocksize=3", "blocksize=6"]
ROW_LABELS = ["GPTQ 层输出误差", "相对 RTN(1.15236) 倍数"]
CELLS = {
    "GPTQ 层输出误差": ["0.99387", "0.99387", "0.99387", "0.99387"],
    "相对 RTN(1.15236) 倍数": ["1.159×", "1.159×", "1.159×", "1.159×"],
}
HIGHLIGHT_ROW = "GPTQ 层输出误差"
STATUS = {"GPTQ 层输出误差": ["invariant"] * 4}
COLOR = {"invariant": ("#ecfdf5", "#047857")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 180, 190, 56, 40, 100, 30
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 46

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'fill="#1e40af">{btext(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
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
            text_fill, weight_attr = stroke, 'font-weight="bold" '
        else:
            text_fill, weight_attr = "#374151", ''
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" fill="{text_fill}" '
                  f'{weight_attr}>{esc(text)}</text>')

foot_y = h - PAD + 8
L.append(f'<text x="{PAD}" y="{foot_y-16}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("绿=四个 blocksize 的输出误差逐位相同(0.99387)——懒惰批是效率重排，不是新算法。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("均优于 RTN 基线 1.15236：补偿确实降误差，且与批大小无关。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig35-5-lazybatch-invariance.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
