#!/usr/bin/env python3
"""before-after 模板改造:无分支异或交换。左面板 cond 真(交换),右面板 cond 假(保持)。
数据来自 traces/algorithms.json compare_and_swap.ascending_pairs[0]/[1]。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

PANELS = [
    ("cond=真 (left=3 > right=1)", [
        "left=3, right=1", "delta = left^right = 2", "out = left^(cond?delta:0)",
        "out_left=1, out_right=3",
    ], 3),
    ("cond=假 (left=1 < right=3)", [
        "left=1, right=3", "delta = left^right = 2", "out = left^(cond?delta:0)",
        "out_left=1, out_right=3",
    ], None),
]
BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 260, 44, 22, 320, 40, 140
w = PAD * 2 + PANEL_W * 2 + 80
h = TOP + len(PANELS[0][1]) * (BOX_H + VGAP) + PAD + 38

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{40}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">无分支异或交换:一条 select 取代 if</text>',
     f'<text x="{PAD}" y="{60}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">standard.py:L334-L339;delta 恒 = left^right,cond 只决定异或 delta 还是异或 0</text>']

for p, (title, steps, hot_val) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 80)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="14" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    for i, step in enumerate(steps):
        y = TOP + i * (BOX_H + VGAP)
        hl = (i == 3)  # 结果行统一高亮
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                 f'fill="{"#fef3c7" if hl else "#e2e8f0"}" '
                 f'stroke="{"#d97706" if hl else "#64748b"}" stroke-width="{2 if hl else 1}"/>')
        L.append(f'<text x="{cx}" y="{y+BOX_H/2+5}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="13" fill="#0f172a">{esc(step)}</text>')
        if i < len(steps) - 1:
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                     'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
midy = TOP + (len(PANELS[0][1]) * (BOX_H + VGAP) - VGAP) / 2
L.append(f'<line x1="{PAD+PANEL_W+8}" y1="{midy}" x2="{PAD+PANEL_W+68}" y2="{midy}" '
         'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{PAD+PANEL_W+40}" y="{midy-10}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="#d97706">同一条公式</text>')
foot_y = h - PAD + 10
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">两侧 delta 都 = 3^1 = 2;真则异或 delta 完成互换,假则异或 0 原样保持——全程无 if</text>')
L.append(f'<text x="{PAD}" y="{foot_y+18}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">IR 里一次 CAS = select:cmpf:xori = 1:1:2(实测 n=16 时 10:10:20)——1 个 select 顶 1 个 if</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch09-xor-swap.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
