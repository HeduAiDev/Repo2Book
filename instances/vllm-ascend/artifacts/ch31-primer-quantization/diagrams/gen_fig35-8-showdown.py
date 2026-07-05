#!/usr/bin/env python3
"""fig35-8-showdown — state-table 模板：GPTQ/AWQ/SmoothQuant 各自相对朴素基线的误差降幅。
三法各在自己的位宽 regime 内部对照，不横向对打。"""
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

TITLE = "三法各自相对朴素基线降低量化误差（各在其位宽 regime 内部对照）"
SUBTITLE = "GPTQ/AWQ 走 W4 权重-only；SmoothQuant 走 W8A8 per-tensor——行行内部可比，法与法之间不横向对打"

ROWS = [
    ("GPTQ (W4 权重-only)", "1.0735", "0.9969", "1.0768×"),
    ("AWQ (W4 权重-only)", "0.7094", "0.2484", "2.8558×"),
    ("SmoothQuant (W8A8)", "0.082", "0.0233", "3.5177×"),
]
COLS = ["方法(regime)", "朴素基线误差", "方法误差", "降幅"]

LABEL_W, COL_W = [260, 180, 180, 140], None
ROW_H, HEADER_H, TOP, PAD = 58, 40, 110, 34
col_widths = [260, 200, 200, 160]
col_x = [PAD]
for cw in col_widths[:-1]:
    col_x.append(col_x[-1] + cw)
w = PAD * 2 + sum(col_widths)
h = TOP + HEADER_H + ROW_H * len(ROWS) + 80

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'fill="#1e40af">{btext(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x, cw = col_x[j], col_widths[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{cw-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    anchor = "start" if j == 0 else "middle"
    tx = x + 10 if j == 0 else x + (cw - 8) / 2
    L.append(f'<text x="{tx}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="{anchor}" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, (method, base_err, meth_err, reduction) in enumerate(ROWS):
    ry = TOP + HEADER_H + i * ROW_H
    x0, cw0 = col_x[0], col_widths[0]
    L.append(f'<text x="{x0+10}" y="{ry+ROW_H/2+5}" font-family="sans-serif" font-size="13" '
              f'font-weight="bold" fill="#374151">{esc(method)}</text>')
    x1, cw1 = col_x[1], col_widths[1]
    L.append(f'<text x="{x1+(cw1-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="#64748b">{esc(base_err)}</text>')
    x2, cw2 = col_x[2], col_widths[2]
    L.append(f'<rect x="{x2}" y="{ry+6}" width="{cw2-8}" height="{ROW_H-12}" rx="4" '
              'fill="#ecfdf5" stroke="#047857" stroke-width="2"/>')
    L.append(f'<text x="{x2+(cw2-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#047857">{esc(meth_err)}</text>')
    # 箭头：基线 -> 方法误差
    ax1 = x1 + cw1 - 8
    ax2 = x2 - 2
    L.append(f'<line x1="{ax1}" y1="{ry+ROW_H/2}" x2="{ax2}" y2="{ry+ROW_H/2}" '
              'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    x3, cw3 = col_x[3], col_widths[3]
    L.append(f'<text x="{x3+(cw3-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" '
              f'fill="#b45309">{esc(reduction)}</text>')

foot_y = h - 26
L.append(f'<text x="{PAD}" y="{foot_y-16}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("三根降幅箭头都朝下（误差降低），但分属不同位宽赛道——图要点是各自相对基线的改进，而非谁的绝对误差更小。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("落地对应：W4 权重-only → vllm_ascend W4A16；W8A8 → AscendW8A8(static=O3/dynamic=O1)。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig35-8-showdown.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
