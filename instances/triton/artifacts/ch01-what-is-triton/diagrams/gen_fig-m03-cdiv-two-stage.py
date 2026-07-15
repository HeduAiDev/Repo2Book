#!/usr/bin/env python3
"""fig-m03-cdiv-two-stage: before-after 模板——同一个 tl.cdiv 在追踪期(阶段一)与
make_ttir 之后(阶段二)的 IR 形态对比。同构双面板,仅差异处高亮。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

PANELS = [
    ("阶段一:追踪期(make_ir 输出)", [
        "tt.func private @cdiv__i32__1cconstexpr_1024_",
        "签名 (%arg0: i32) → i32,1024 不在签名里",
        "体内 25 个 arith op:addi2/subi2/divsi1/\nextsi4/cmpi4/andi2/constant10",
        "全模块 tt.call = 1,tt.func = 2",
    ], 3),
    ("阶段二:make_ttir 之后(add_inliner+canonicalizer)", [
        "tt.func 被内联抹平,无独立 @cdiv",
        "canonicalizer 折出常量 1024 与 1023",
        "全模块逐 op:constant2/addi1/divsi1/\nstore1/return1/func1",
        "全模块 tt.call = 0",
    ], 3),
]
BOX_W, VGAP, PANEL_W, PAD, TOP = 300, 22, 380, 40, 84

def lines_of(step):
    return step.split("\n")

def box_h(step):
    return 30 + 16 * (len(lines_of(step)) - 1)

heights = [box_h(s) for s in PANELS[0][1]]
w = PAD * 2 + PANEL_W * 2 + 90
h = TOP + sum(heights) + VGAP * (len(heights) - 1) + PAD + 20

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{TOP-52}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e293b">同一个 tl.cdiv:追踪期建 IR,make_ttir 一跑就被内联折叠</text>']

for p, (title, steps, hot) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 90)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    y = TOP
    for i, step in enumerate(steps):
        lines = lines_of(step)
        bh = box_h(step)
        hl = (i == hot) if hot is not None else False
        fill = "#fef3c7" if hl else "#e2e8f0"
        stroke = "#d97706" if hl else "#64748b"
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{bh}" rx="8" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if hl else 1}"/>')
        n = len(lines)
        y0 = y + bh / 2 - (n - 1) * 8 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx}" y="{y0+k*16}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12" fill="#0f172a">{esc(line)}</text>')
        if i < len(steps) - 1:
            L.append(f'<line x1="{cx}" y1="{y+bh}" x2="{cx}" y2="{y+bh+VGAP-4}" '
                      'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
        y += bh + VGAP

midy = TOP + sum(heights[:2]) + VGAP + heights[2] / 2 + heights[1] / 2
L.append(f'<line x1="{PAD+PANEL_W+10}" y1="{midy}" x2="{PAD+PANEL_W+80}" y2="{midy}" '
          'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{PAD+PANEL_W+45}" y="{midy-10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#a16207">add_inliner</text>')

foot_y = h - 16
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">追踪期 mangled 名 cdiv__i32__1cconstexpr_1024_;'
          f'TRITON_KERNEL_DUMP 的 .ttir 是阶段二产物,想看阶段一得自己调 make_ir</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-m03-cdiv-two-stage.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} size={w}x{h}")
