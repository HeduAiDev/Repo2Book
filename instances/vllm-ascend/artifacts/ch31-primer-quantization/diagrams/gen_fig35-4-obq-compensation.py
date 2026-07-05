#!/usr/bin/env python3
"""fig35-4-obq-compensation — before-after 模板：RTN(独立四舍五入) vs OBQ(二阶补偿)。
三个权重码几乎一样，唯 w2 因补偿从 0.3347 翻转到 0.0，层输出误差降 2.9534x。"""
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

RTN_CODES = ["-0.6693", "-0.6693", "0.3347"]
OBQ_CODES = ["-0.6693", "-0.6693", "0.0"]
DIFF_IDX = 2  # w2 是唯一不同的
RTN_ERR = "0.13116"
OBQ_ERR = "0.04441"
IMPROV = "2.9534×"

PANELS = [
    ("RTN — 独立四舍五入", RTN_CODES, RTN_ERR, "#dc2626"),
    ("OBQ — 二阶补偿(按 Hessian 摊派)", OBQ_CODES, OBQ_ERR, "#047857"),
]

BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 200, 46, 22, 280, 44, 100
w = PAD * 2 + PANEL_W * 2 + 140
h = TOP + 3 * (BOX_H + VGAP) + 90

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-8}" font-family="sans-serif" font-size="16" '
     f'fill="#1e40af">{btext("二阶补偿不是取整微调：w2 被翻转，层输出误差降到不足 RTN 的三分之一")}</text>',
     f'<text x="{PAD}" y="{PAD+12}" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc("3 个权重、2-bit 网格，w_row = [-0.6, -0.812, 0.192]")}</text>']

for p, (title, codes, err, err_color) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 140)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    for i, code in enumerate(codes):
        y = TOP + i * (BOX_H + VGAP)
        hl = (i == DIFF_IDX)
        fill = "#fef3c7" if hl else "#e2e8f0"
        stroke = "#d97706" if hl else "#64748b"
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{2.5 if hl else 1}"/>')
        L.append(f'<text x="{cx}" y="{y+BOX_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="14" fill="#0f172a">'
                  f'{esc(f"w{i} = {code}")}</text>')
    # 层输出误差
    ey = TOP + 3 * (BOX_H + VGAP) + 6
    L.append(f'<rect x="{cx-BOX_W/2}" y="{ey}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="white" stroke="{err_color}" stroke-width="2"/>')
    L.append(f'<text x="{cx}" y="{ey+BOX_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{err_color}">{esc(f"层输出误差 = {err}")}</text>')
    # 权重框间连线
    for i in range(2):
        y = TOP + i * (BOX_H + VGAP)
        L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                  'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    y2 = TOP + 2 * (BOX_H + VGAP)
    L.append(f'<line x1="{cx}" y1="{y2+BOX_H}" x2="{cx}" y2="{ey-4}" '
              'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

# 中间对比箭头 + 改进倍数
midy = TOP + (3 * (BOX_H + VGAP)) / 2
arrow_x1 = PAD + PANEL_W + 12
arrow_x2 = PAD + PANEL_W + 128
L.append(f'<line x1="{arrow_x1}" y1="{midy}" x2="{arrow_x2}" y2="{midy}" '
          'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(arrow_x1+arrow_x2)/2}" y="{midy-14}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13" font-weight="bold" '
          f'fill="#d97706">{esc(IMPROV)}</text>')
L.append(f'<text x="{(arrow_x1+arrow_x2)/2}" y="{midy+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10" fill="#64748b">{esc("仅 w2 不同")}</text>')

foot_y = h - 24
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("黄框=补偿后发生翻转的权重(w2: 0.3347 → 0.0)；两条码 w0/w1 完全一致——差异只来自 Hessian 相关性驱动的误差再分配。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig35-4-obq-compensation.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
