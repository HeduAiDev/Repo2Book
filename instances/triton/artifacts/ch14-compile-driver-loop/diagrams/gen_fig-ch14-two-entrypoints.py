#!/usr/bin/env python3
"""before-after 模板:两个入口(ASTSource vs IRSource)同构对比——同 4 行属性、
只有 first_stage 一行(数值 0 vs 2)高亮差异。改造点:PANELS。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

PANELS = [
    ("入口 A: ASTSource(@jit 源码)", [
        "src.ext = \"ttir\"",
        "身份=按调用特化(签名+constexpr+attrs)",
        "make_ir 跑前端:ast_to_ttir",
        "first_stage = 0(从 ttir 起步,全 5 级跑)",
    ], None),
    ("入口 B: IRSource(一份 .ttgir 文件)", [
        "src.ext = \"ttgir\"(取自文件后缀)",
        "身份=文件内容 sha256(不依赖调用现场)",
        "make_ir 直接 parse_mlir_module",
        "first_stage = 2(index(ttgir)+1,跳 2 级)",
    ], 3),
]
BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 320, 52, 20, 360, 40, 118
w = PAD * 2 + PANEL_W * 2 + 100
h = TOP + len(PANELS[0][1]) * (BOX_H + VGAP) + PAD + 96

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="36" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">{esc("两种入口共用同一条降级链，只差 ext/身份/make_ir 三点——决定了 first_stage 起步级")}</text>']

for p, (title, steps, hot) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 100)
    cx = px + PANEL_W / 2
    title_fill = "#b45309" if hot is not None else "#1e40af"
    L.append(f'<text x="{cx}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="{title_fill}">{esc(title)}</text>')
    for i, step in enumerate(steps):
        y = TOP + i * (BOX_H + VGAP)
        hl = (i == hot)
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                  f'fill="{"#fef3c7" if hl else "#e2e8f0"}" '
                  f'stroke="{"#d97706" if hl else "#64748b"}" stroke-width="{2 if hl else 1}"/>')
        L.append(f'<text x="{cx}" y="{y+BOX_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" '
                  f'fill="{"#92400e" if hl else "#0f172a"}">{esc(step)}</text>')
        if i < len(steps) - 1:
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                      'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

midy = TOP + 3 * (BOX_H + VGAP) + BOX_H / 2  # 对齐两侧 first_stage 行
L.append(f'<line x1="{PAD+PANEL_W+10}" y1="{midy}" x2="{PAD+PANEL_W+90}" y2="{midy}" '
          'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{PAD+PANEL_W+50}" y="{midy-12}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="#d97706">{esc("同一驱动循环")}</text>')

foot_y = TOP + len(PANELS[0][1]) * (BOX_H + VGAP) - VGAP + 44
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="#1e293b" font-weight="bold">'
          f'{esc("结论:first_stage = index(src.ext)+1(IR 入口)——恰好跳过把 IR 降成它自己这一级的 pass。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">'
          f'{esc("IRSource 用文件内容 sha256 作身份,是它能拿一份 .ttgir 直接做 IR 级实验、绕过前端迭代的前提。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch14-two-entrypoints.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size {w}x{h}")
