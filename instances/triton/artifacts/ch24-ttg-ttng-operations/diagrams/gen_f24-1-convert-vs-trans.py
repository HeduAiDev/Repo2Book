#!/usr/bin/env python3
"""f24-1-convert-vs-trans: before-after 模板。
左panel = tt.trans/reshape(改名,零跨线程搬运);右panel = convert_layout(真跨线程搬运)。
底部信息条给 traits + 折叠后残留 cvt 条数(spec.numbers 全覆盖)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

PANELS = [
    ("tt.trans / tt.reshape：改名", [
        "输入 tensor<A, #enc>",
        "重排『每个线程持有元素』的下标",
        "输出 tensor<A', #enc>（同一份数据）",
    ], None, "纯元数据：零跨线程搬运"),
    ("convert_layout：真搬运", [
        "输入 tensor<A, #enc1>",
        "跨线程重新分布（经共享内存中转）",
        "输出 tensor<A, #enc2>",
    ], 1, "唯一真花线程间通信成本的 op"),
]
BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 300, 46, 30, 340, 50, 100
w = PAD * 2 + PANEL_W * 2 + 100
h = TOP + len(PANELS[0][1]) * (BOX_H + VGAP) + 140 + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="ah" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

L.append(f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" '
         f'font-size="17" font-weight="bold" fill="#0f172a">改名 vs 搬运：谁在线程间真的移动了数据</text>')

for p, (title, steps, hot, tag) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 100)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-40}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="15" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
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
    tag_y = TOP + len(steps) * (BOX_H + VGAP) - VGAP + 24
    tag_color = "#b45309" if hot is not None else "#334155"
    L.append(f'<text x="{cx}" y="{tag_y}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="{tag_color}">{esc(tag)}</text>')

# 底部信息条:traits + 折叠后残留 cvt 条数(numbers 全覆盖,手工按可用宽度预折行)
info_top = TOP + len(PANELS[0][1]) * (BOX_H + VGAP) + 20
info_lines = [
    "convert_layout 的 traits：SameOperandsAndResultShape +",
    "SameOperandsAndResultElementType + Pure（只改 encoding，shape / elementType 均不变）",
    "canonicalizer 折叠后 dump 残留 cvt 条数：1（#mma→#dot；循环外 mma→blocked1 回写另 1 条）",
]
info_h = 24 + len(info_lines) * 24 + 10
L.append(f'<rect x="{PAD}" y="{info_top}" width="{w-2*PAD}" height="{info_h}" rx="8" '
         'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1"/>')
for i, line in enumerate(info_lines):
    L.append(f'<text x="{PAD+18}" y="{info_top+26+i*24}" font-family="sans-serif" '
             f'font-size="12" fill="#334155">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("f24-1-convert-vs-trans.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
