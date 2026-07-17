#!/usr/bin/env python3
"""before-after 模板：控制流汇合处 join=逐轴 gcd，一支对齐退化把合并后的向量宽拖到 1。
左：分支 A 单独（divisibility=16 → perThread=4，128-bit）。
右：A、B 在汇合点 join 后（divisibility=gcd(16,4)=4 → perThread=1，标量）。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


PANELS = [
    ("分支 A 单独：divisibility=16", [
        ("branch A: div=16 字节", None),
        ("perThread = min(16/4,1024,4)", None),
        ("= 4 （128-bit 向量）", "hot"),
    ]),
    ("A、B 在汇合点 join 后", [
        ("branch B: div=4 字节", None),
        ("join(A,B) = gcd(16,4) = 4", "hot"),
        ("perThread = min(4/4,1024,4) = 1（标量）", "hot2"),
    ]),
]
BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 300, 46, 24, 340, 40, 128
COLOR = {None: ("#e2e8f0", "#64748b"), "hot": ("#fef3c7", "#d97706"),
         "hot2": ("#fee2e2", "#b91c1c")}
w = PAD * 2 + PANEL_W * 2 + 90
h = TOP + len(PANELS[0][1]) * (BOX_H + VGAP) + PAD + 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="16" font-weight="bold" '
     f'fill="#1e40af">{esc("join = 逐轴 gcd：一支对齐退化就把合并向量宽拖到 1")}</text>']

for p, (title, steps) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 90)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    for i, (step, hl) in enumerate(steps):
        y = TOP + i * (BOX_H + VGAP)
        fill, stroke = COLOR[hl]
        sw = 2.5 if hl else 1
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        L.append(f'<text x="{cx}" y="{y+BOX_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" fill="#0f172a">{esc(step)}</text>')
        if i < len(steps) - 1:
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                      'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

midy = TOP + (len(PANELS[0][1]) * (BOX_H + VGAP) - VGAP) / 2
L.append(f'<line x1="{PAD+PANEL_W+10}" y1="{midy}" x2="{PAD+PANEL_W+80}" y2="{midy}" '
          'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{PAD+PANEL_W+45}" y="{midy-12}" text-anchor="middle" '
          'font-family="sans-serif" font-size="11" fill="#d97706" font-weight="bold">'
          f'{esc("在此汇合")}</text>')

foot_y = h - 24
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("A 单独能 4 元素/线程(128-bit)；一旦与 4 字节对齐的 B 汇合，divisibility=gcd(16,4)=4，向量宽塌回 1——提示被路径 join 掉了。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-join-gcd-collapse.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
