#!/usr/bin/env python3
"""fig36-1-token-vs-feature: before-after 模板。
左:token 层直接自回归(Medusa 量级);右:特征层自回归再过共享 LM Head(EAGLE)。
数字来自 explainer.json fig36-1 numbers(paper.md L26/L32)。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

PANELS = [
    ("token 层自回归（Medusa 量级）",
     ["token embedding", "直接对 token 自回归", "输出 token"],
     None,
     "草稿准确率 ≈ 0.6　加速比 1.5x"),
    ("特征层自回归（EAGLE）",
     ["token → embedding", "特征自回归（Autoregression Head）", "共享 LM Head → token"],
     1,
     "草稿准确率 ≈ 0.8　加速比 1.9x"),
]
BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 260, 44, 26, 340, 40, 96
w = PAD * 2 + PANEL_W * 2 + 80
h = TOP + len(PANELS[0][1]) * (BOX_H + VGAP) + 56 + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="{PAD}" text-anchor="middle" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#0f172a">{esc("特征层 vs token 层草稿：为何 EAGLE 更准")}</text>']

for p, (title, steps, hot, metric) in enumerate(PANELS):
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
        if i < len(steps) - 1:
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                     'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    metric_y = TOP + len(steps) * (BOX_H + VGAP) - VGAP + 40
    metric_fill = "#fef3c7" if p == 1 else "#f1f5f9"
    metric_stroke = "#d97706" if p == 1 else "#94a3b8"
    L.append(f'<rect x="{cx-BOX_W/2}" y="{metric_y-24}" width="{BOX_W}" height="34" rx="6" '
             f'fill="{metric_fill}" stroke="{metric_stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{cx}" y="{metric_y-3}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="#92400e">{esc(metric)}</text>')

midy = TOP + (len(PANELS[0][1]) * (BOX_H + VGAP) - VGAP) / 2
L.append(f'<line x1="{PAD+PANEL_W+8}" y1="{midy}" x2="{PAD+PANEL_W+68}" y2="{midy}" '
         'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{PAD+PANEL_W+38}" y="{midy-10}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#92400e">{esc("EAGLE 的第一大观察")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig36-1-token-vs-feature.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
