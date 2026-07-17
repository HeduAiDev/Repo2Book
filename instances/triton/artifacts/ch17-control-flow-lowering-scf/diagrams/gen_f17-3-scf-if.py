#!/usr/bin/env python3
"""f17-3-scf-if: 无 return 的 if 下降成 scf.if——源码只写 then,下降后 else 分支自动
补出并 yield 进入前的 livein。before-after 双面板:左=源码,右=下降后的 scf.if 双区域。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

PANELS = [
    ("源码:只写了 then", ["if c > 0:", "    x = x + 1", "(没写 else)"], None),
    ("下降后:scf.if 双区域全出", ["%5 = scf.if %4 -> (tensor<8xf32>)",
                                 "then: scf.yield %9  (x+1)",
                                 "else: scf.yield %3  (livein x 原值)"], 2),
]
BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 300, 46, 26, 380, 40, 76
w = PAD * 2 + PANEL_W * 2 + 100
h = TOP + len(PANELS[1][1]) * (BOX_H + VGAP) + PAD + 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']
L.append(f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="15" font-weight="bold" '
          f'fill="#0f172a">{esc("无 return 的 if 走 scf.if:未写的 else 被自动补出并 yield livein 原值")}</text>')

for p, (title, steps, hot) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 100)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="14" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    n = len(steps)
    y_offset = 0 if n == len(PANELS[1][1]) else (len(PANELS[1][1]) - n) * (BOX_H + VGAP) / 2
    for i, step in enumerate(steps):
        y = TOP + y_offset + i * (BOX_H + VGAP)
        hl = (i == hot)
        fill = "#dcfce7" if (p == 1 and hl) else ("#e2e8f0" if p == 0 else "#dbeafe")
        stroke = "#15803d" if (p == 1 and hl) else ("#64748b" if p == 0 else "#1d4ed8")
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if hl else 1.4}"/>')
        L.append(f'<text x="{cx}" y="{y+BOX_H/2+5}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="12.5" fill="#0f172a">{esc(step)}</text>')
        if i < n - 1:
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                     'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

midy = TOP + (len(PANELS[1][1]) * (BOX_H + VGAP) - VGAP) / 2
L.append(f'<line x1="{PAD+PANEL_W+8}" y1="{midy}" x2="{PAD+PANEL_W+92}" y2="{midy}" '
         'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{PAD+PANEL_W+50}" y="{midy-10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#b45309">{esc("下降")}</text>')

# 补出标注(放在 else 框正下方,避免右侧越界)
hl_y = TOP + 2 * (BOX_H + VGAP)
hl_cx = PAD + PANEL_W + 100 + PANEL_W / 2
L.append(f'<text x="{hl_cx}" y="{hl_y+BOX_H+18}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#15803d">{esc("↑ 源码没写,下降后自动补出")}</text>')

foot_y = h - 20
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("traces/ch17_traces.json -> ir.K2_if_scf(scf.if 计数=1, scf.yield 计数=2)")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("f17-3-scf-if.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} size={w}x{h}")
