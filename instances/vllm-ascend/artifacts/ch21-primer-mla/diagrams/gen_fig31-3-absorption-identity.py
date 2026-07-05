#!/usr/bin/env python3
"""fig31-3-absorption-identity: before-after 对比——物化路径(算出 full k^C 再内积)
vs 吸收路径(query 先乘 W_UK 落到潜空间、直接和缓存 c_kv 内积),给出逐位相等的打分。
数字全部来自 traces/absorption.json。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

PANELS = [
    ("物化路径(每步都要放大出 full key)",
     ["c_kv (缓存,d_c=4)", "×W_UK → k^C (物化)", "q^C · k^C → 打分 -0.0973"], 2),
    ("吸收路径(query 先落潜空间)",
     ["q ×W_UK(一次,静态吸收)", "q̃ (潜空间 query)", "q̃ · c_kv → 打分 -0.0973"], 2),
]
BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 260, 46, 28, 340, 40, 78
w = PAD * 2 + PANEL_W * 2 + 90
h = TOP + len(PANELS[0][1]) * (BOX_H + VGAP) + 130

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">{esc("权重吸收恒等式:两条路径给出逐位相等的打分")}</text>']

for p, (title, steps, hot) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 90)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="14" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    for i, step in enumerate(steps):
        y = TOP + i * (BOX_H + VGAP)
        hl = (i == hot)
        fill = "#dcfce7" if hl else "#e2e8f0"
        stroke = "#15803d" if hl else "#64748b"
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{2.5 if hl else 1}"/>')
        weight = 'font-weight="bold" ' if hl else ''
        color = "#15803d" if hl else "#0f172a"
        L.append(f'<text x="{cx}" y="{y+BOX_H/2+5}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="13" {weight}fill="{color}">{esc(step)}</text>')
        if i < len(steps) - 1:
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                     'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

midy = TOP + (len(PANELS[0][1]) * (BOX_H + VGAP) - VGAP) / 2
L.append(f'<text x="{PAD+PANEL_W+45}" y="{midy-14}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="#15803d" font-weight="bold">{esc("数值恒等")}</text>')
L.append(f'<line x1="{PAD+PANEL_W+8}" y1="{midy}" x2="{PAD+PANEL_W+82}" y2="{midy}" '
         'stroke="#15803d" stroke-width="2.5" marker-end="url(#a)"/>')

# bottom evidence callout: table of abs diffs -> all 0.0
call_top = TOP + len(PANELS[0][1]) * (BOX_H + VGAP) + 20
call_w = w - 2 * PAD
L.append(f'<rect x="{PAD}" y="{call_top}" width="{call_w}" height="86" rx="10" '
         'fill="#fef3c7" stroke="#d97706" stroke-width="2"/>')
L.append(f'<text x="{PAD+call_w/2}" y="{call_top+30}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="#92400e">'
         f'{esc("4 对 (查询 t, 键 j) 的两路打分逐位相等,最大绝对差 = 0.0")}</text>')
L.append(f'<text x="{PAD+call_w/2}" y="{call_top+58}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" fill="#92400e">'
         f'{esc("如 (t=1, j=0):物化 -0.0973 == 吸收 -0.0973 —— 精确恒等,不是近似")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig31-3-absorption-identity.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
