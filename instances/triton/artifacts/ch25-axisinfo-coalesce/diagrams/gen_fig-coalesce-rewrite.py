#!/usr/bin/env python3
"""before-after 模板：coalesceOp 用 convert_layout 夹层做保语义改写——
操作数转入新布局→造新 op→结果转回原布局→replaceAllUsesWith。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


PANELS = [
    ("改写前：1 个 op", [
        ("%p : L_old", None),
        ("%v = tt.load %p : L_old", "hot"),
    ]),
    ("改写后：1 load + 2 convert_layout", [
        ("%p2 = convert_layout %p", "conv"),
        ("%v2 = tt.load %p2 : L_new(perThread=4)", "hot"),
        ("%v3 = convert_layout %v2 : L_new→L_old", "conv"),
    ]),
]
BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 320, 46, 22, 360, 40, 118
COLOR = {None: ("#e2e8f0", "#64748b"), "hot": ("#fef3c7", "#d97706"),
         "conv": ("#e0f2fe", "#0369a1")}
w = PAD * 2 + PANEL_W * 2 + 90
h = TOP + 3 * (BOX_H + VGAP) + PAD + 130

LEFT_NUMS = [("改写前 op 数", "1")]
RIGHT_NUMS = [("改写后 op 数", "3"), ("新增 convert_layout", "2"), ("load 向量宽提升到", "4")]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="16" font-weight="bold" '
     f'fill="#1e40af">{esc("coalesceOp：用 convert_layout 夹层做保语义改写")}</text>']

for p, (title, steps) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 90)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    n_steps = len(steps)
    # vertically center shorter panel (2 boxes) against 3-box panel
    y_offset = (3 - n_steps) * (BOX_H + VGAP) / 2
    for i, (step, hl) in enumerate(steps):
        y = TOP + y_offset + i * (BOX_H + VGAP)
        fill, stroke = COLOR[hl]
        sw = 2.5 if hl else 1
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        L.append(f'<text x="{cx}" y="{y+BOX_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="#0f172a">{esc(step)}</text>')
        if i < n_steps - 1:
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                      'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

midy = TOP + 1.5 * (BOX_H + VGAP) - VGAP / 2
L.append(f'<line x1="{PAD+PANEL_W+10}" y1="{midy}" x2="{PAD+PANEL_W+80}" y2="{midy}" '
          'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{PAD+PANEL_W+45}" y="{midy-14}" text-anchor="middle" '
          'font-family="sans-serif" font-size="11" fill="#d97706" font-weight="bold">'
          f'{esc("Coalesce 改写")}</text>')

nums_y = TOP + 3 * (BOX_H + VGAP) - VGAP + 30
left_cx = PAD + PANEL_W / 2
right_px = PAD + PANEL_W + 90
right_num_w = PANEL_W / len(RIGHT_NUMS)

for lbl, val in LEFT_NUMS:
    L.append(f'<text x="{left_cx}" y="{nums_y}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="20" font-weight="bold" fill="#1e40af">{esc(val)}</text>')
    L.append(f'<text x="{left_cx}" y="{nums_y+18}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#64748b">{esc(lbl)}</text>')

for i, (lbl, val) in enumerate(RIGHT_NUMS):
    nx = right_px + i * right_num_w + right_num_w / 2
    L.append(f'<text x="{nx}" y="{nums_y}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="20" font-weight="bold" fill="#1e40af">{esc(val)}</text>')
    L.append(f'<text x="{nx}" y="{nums_y+18}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#64748b">{esc(lbl)}</text>')

foot_y = h - 24
FOOT1 = "SSA 下不能原地改类型：操作数 convert 到新布局→造新 load→结果 convert 回原布局→replaceAllUsesWith。"
FOOT2 = "多出的 2 个 convert 由后续化简吸收，换来 load 从 1 升到 perThread=4 的 128-bit 向量访存。"
L.append(f'<text x="{PAD}" y="{foot_y-16}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc(FOOT1)}</text>')
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc(FOOT2)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-coalesce-rewrite.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
