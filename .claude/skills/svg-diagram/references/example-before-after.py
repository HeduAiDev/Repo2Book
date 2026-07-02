#!/usr/bin/env python3
"""before-after 模板:优化前后双态对比。同构双面板,仅差异处高亮——读者视线只被差异吸引。
改造点:PANELS(标题,步骤,高亮下标)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

PANELS = [
    ("优化前:每步同步", ["forward()", "sync 等待采样", "下一步调度"], None),
    ("优化后:异步流水", ["forward()", "异步取回(不等)", "下一步调度"], 1),
]
BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 240, 44, 26, 320, 40, 70
w = PAD * 2 + PANEL_W * 2 + 80
h = TOP + len(PANELS[0][1]) * (BOX_H + VGAP) + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']
for p, (title, steps, hot) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 80)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="14" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    for i, step in enumerate(steps):
        y = TOP + i * (BOX_H + VGAP)
        hl = (i == hot)
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                 f'fill="{"#fef3c7" if hl else "#e2e8f0"}" '
                 f'stroke="{"#d97706" if hl else "#64748b"}" stroke-width="{2 if hl else 1}"/>')
        L.append(f'<text x="{cx}" y="{y+BOX_H/2+5}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="13" fill="#0f172a">{esc(step)}</text>')
        if i < len(steps) - 1:  # 箭头端点取自框边缘
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                     'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
midy = TOP + (len(PANELS[0][1]) * (BOX_H + VGAP) - VGAP) / 2
L.append(f'<line x1="{PAD+PANEL_W+8}" y1="{midy}" x2="{PAD+PANEL_W+68}" y2="{midy}" '
         'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append('</svg>')
out = Path(__file__).with_name("example-before-after.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
