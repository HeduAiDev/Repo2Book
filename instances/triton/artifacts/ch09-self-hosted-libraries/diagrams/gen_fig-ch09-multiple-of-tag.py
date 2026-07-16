#!/usr/bin/env python3
"""before-after 模板:tl.multiple_of 只打标记,不产生计算 op。左面板 plain 版
make_range 无 tt.divisibility;右面板加提示后多出 tt.divisibility=dense<128>。
数据来自 traces/ir_metrics.json multiple_of_hint。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

PANELS = [
    ("plain(无提示)", [
        "off = tl.arange(0, 128)",
        "tt.make_range 0:128",
        "属性:无 tt.divisibility",
        "新增计算 op = 0",
    ], None),
    ("加 tl.multiple_of(off,128)", [
        "off = tl.arange(0, 128)",
        "tt.make_range 0:128",
        "属性:tt.divisibility=dense<128>",
        "新增计算 op = 0(仍是纯标记)",
    ], 2),
]
BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 300, 46, 20, 340, 40, 140
w = PAD * 2 + PANEL_W * 2 + 80
h = TOP + len(PANELS[0][1]) * (BOX_H + VGAP) + PAD + 50

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="40" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">multiple_of 只盖戳,不计算:向量化留给后端 AxisInfo</text>',
     f'<text x="{PAD}" y="60" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">core.py:L2200-L2229;Triton v3.2.0 headless 编译对比 TTIR make_range 属性</text>']

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
                 f'font-family="sans-serif" font-size="12.5" fill="#0f172a">{esc(step)}</text>')
        if i < len(steps) - 1:
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                     'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
midy = TOP + (len(PANELS[0][1]) * (BOX_H + VGAP) - VGAP) / 2
L.append(f'<line x1="{PAD+PANEL_W+8}" y1="{midy}" x2="{PAD+PANEL_W+68}" y2="{midy}" '
         'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{PAD+PANEL_W+40}" y="{midy-10}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="#d97706">加一句提示</text>')
foot_y = h - PAD + 20
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">multiple_of 只在追踪期校验 constexpr[int] 并改张量的 divisibility 元信息;'
          f'向量化/合并 load 由后端 AxisInfo 消费这个戳(第 25 章会讲)</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch09-multiple-of-tag.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
