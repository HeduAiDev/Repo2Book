#!/usr/bin/env python3
"""fig-m9-expand: before-after 模板。expandInterleaveMemRefType 把 memref
末维 shape ×2、stride 归 1，offset 静态则归 0——stride=2 的交错视图被
还原成一段整段连续 2N 的描述。全坐标由循环/常量计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

FIELDS = [
    ("末维 shape", "4", "8", True),
    ("末维 stride", "2", "1", True),
    ("offset（静态时）", "off", "0", True),
]

PANEL_W = 260
GAP = 260
ROW_H = 60
TOP = 130
PAD = 40
W = PAD * 2 + PANEL_W * 2 + GAP
h = TOP + len(FIELDS) * ROW_H + 140

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>',
     f'<rect width="{W}" height="{h}" fill="white"/>']

L.append(f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" font-weight="bold" '
         f'fill="#0f172a">{esc("expandInterleaveMemRefType：末维 shape ×2、stride 归 1")}</text>')
L.append(f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" fill="#64748b">'
         f'{esc("InterleaveOptimization.cpp:L49-L68")}</text>')

left_x = PAD
right_x = W - PAD - PANEL_W
left_cx = left_x + PANEL_W / 2
right_cx = right_x + PANEL_W / 2

L.append(f'<text x="{left_cx}" y="{TOP-20}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="#334155">{esc("Before：原 reinterpret_cast 类型")}</text>')
L.append(f'<text x="{right_cx}" y="{TOP-20}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="#b45309">{esc("After：expandInterleaveMemRefType 结果")}</text>')

mid_cx = (left_x + PANEL_W + right_x) / 2
for i, (name, before, after, changed) in enumerate(FIELDS):
    y = TOP + i * ROW_H
    fill_b, stroke_b = ("#e2e8f0", "#64748b")
    fill_a, stroke_a = ("#fef3c7", "#b45309") if changed else ("#e2e8f0", "#64748b")
    bh = 44
    by = y + (ROW_H - bh) / 2
    L.append(f'<rect x="{left_x}" y="{by}" width="{PANEL_W}" height="{bh}" rx="8" '
             f'fill="{fill_b}" stroke="{stroke_b}" stroke-width="1.5"/>')
    L.append(f'<text x="{left_cx}" y="{by+bh/2+5}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="14" fill="#334155">{esc(before)}</text>')
    L.append(f'<rect x="{right_x}" y="{by}" width="{PANEL_W}" height="{bh}" rx="8" '
             f'fill="{fill_a}" stroke="{stroke_a}" stroke-width="{2.5 if changed else 1.5}"/>')
    L.append(f'<text x="{right_cx}" y="{by+bh/2+5}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="14" font-weight="{"bold" if changed else "normal"}" '
             f'fill="{"#7c2d12" if changed else "#334155"}">{esc(after)}</text>')
    # arrow between panels, field label above the arrow (no overlap with boxes)
    L.append(f'<text x="{mid_cx}" y="{by+bh/2-12}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="#475569">{esc(name)}</text>')
    L.append(f'<line x1="{left_x+PANEL_W+10}" y1="{by+bh/2}" x2="{right_x-10}" y2="{by+bh/2}" '
             'stroke="#d97706" stroke-width="2" marker-end="url(#a)"/>')

foot_y = TOP + len(FIELDS) * ROW_H + 40
L.append(f'<rect x="{PAD}" y="{foot_y-24}" width="{W-2*PAD}" height="70" rx="8" '
         'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
L.append(f'<text x="{PAD+16}" y="{foot_y-2}" font-family="sans-serif" font-size="12.5" '
         f'fill="#334155">{esc("示例：末维 shape 4→8、stride 2→1——跨步 2 的交错视图变成描述")}</text>')
L.append(f'<text x="{PAD+16}" y="{foot_y+18}" font-family="sans-serif" font-size="12.5" '
         f'fill="#334155">{esc("整段连续 2N 的 memref，好让 deinterleave/interleave 用一次连续搬运替代跨步访问。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m9-expand.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
