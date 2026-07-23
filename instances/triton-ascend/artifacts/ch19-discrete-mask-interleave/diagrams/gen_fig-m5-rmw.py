#!/usr/bin/env python3
"""fig-m5-rmw: flow 模板。离散 store 的读-改-写：load 目标原值 origin →
select(mask, src, origin) 拼出 written → store 整段回写。一次逻辑散点写
实付两趟 8 元素全量 DMA。全坐标由循环/常量计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

MASK = [True, True, False, False, False, False, True, True]
SRC = [10, 11, 12, 13, 14, 15, 16, 17]
ORIGIN = [90, 91, 92, 93, 94, 95, 96, 97]
WRITTEN = [10, 11, 92, 93, 94, 95, 16, 17]

W = 1040
CELL, CGAP = 46, 4
STRIP_W = len(MASK) * (CELL + CGAP) - CGAP
PAD = 40
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 620">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="620" fill="white"/>']

L.append(f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
         f'font-weight="bold" fill="#0f172a">{esc("离散 store：读-改-写，一次逻辑散点写付两趟全量 DMA")}</text>')
L.append(f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" fill="#64748b">'
         f'{esc("DiscreteMaskStoreConversion fallback，DiscreteMaskAccessConversionPass.cpp:L206-L226")}</text>')

CX = W / 2
strip_x0 = CX - STRIP_W / 2

def strip(y, values, label, color_fn, label_x=None):
    for i, v in enumerate(values):
        x = strip_x0 + i * (CELL + CGAP)
        fill, stroke = color_fn(i, v)
        L.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="34" rx="4" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        L.append(f'<text x="{x+CELL/2}" y="{y+22}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="12" fill="{stroke}">{esc(str(v))}</text>')
    lx = label_x if label_x is not None else strip_x0 - 12
    L.append(f'<text x="{lx}" y="{y+22}" text-anchor="end" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="#374151">{esc(label)}</text>')

# mask strip
y_mask = 92
def mask_color(i, v):
    return ("#dbeafe", "#1d4ed8") if v else ("#f1f5f9", "#94a3b8")
strip(y_mask, ["T" if m else "F" for m in MASK], "mask", mask_color)
L.append(f'<text x="{strip_x0+STRIP_W+12}" y="{y_mask+22}" font-family="sans-serif" '
         f'font-size="11.5" fill="#64748b">{esc("(idx<2) ∨ (idx>5)")}</text>')

# src strip
y_src = y_mask + 60
strip(y_src, SRC, "src（待写新值）", lambda i, v: ("#e0f2fe", "#0369a1"))

# origin strip
y_origin = y_src + 60
strip(y_origin, ORIGIN, "origin（dst 原值）", lambda i, v: ("#fef3c7", "#b45309"))

# arrow: load origin (DMA①)
y_arrow1 = y_origin + 70
L.append(f'<line x1="{CX}" y1="{y_origin+34}" x2="{CX}" y2="{y_arrow1}" '
         'stroke="#b45309" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{CX+14}" y="{y_origin+34+22}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="#b45309">{esc("① load origin — DMA 第 1 趟，8 元素")}</text>')

# select box
y_select = y_arrow1 + 14
sel_w, sel_h = 380, 40
L.append(f'<rect x="{CX-sel_w/2}" y="{y_select}" width="{sel_w}" height="{sel_h}" rx="8" '
         'fill="#ede9fe" stroke="#6d28d9" stroke-width="2"/>')
L.append(f'<text x="{CX}" y="{y_select+sel_h/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#5b21b6">{esc("select(mask, src, origin)")}</text>')

# arrow src -> select (side)
L.append(f'<line x1="{strip_x0-4}" y1="{y_src+17}" x2="{CX-sel_w/2-14}" y2="{y_select+sel_h/2}" '
         'stroke="#0369a1" stroke-width="1.3" stroke-dasharray="3,2" marker-end="url(#a)"/>')

# arrow select -> written
y_written = y_select + sel_h + 44
L.append(f'<line x1="{CX}" y1="{y_select+sel_h}" x2="{CX}" y2="{y_written}" '
         'stroke="#6d28d9" stroke-width="2" marker-end="url(#a)"/>')

def written_color(i, v):
    return ("#dbeafe", "#1d4ed8") if MASK[i] else ("#fce7f3", "#a21caf")
strip(y_written, WRITTEN, "written（选中→src / 否则→origin）", written_color)

# arrow: store (DMA②)
y_arrow2 = y_written + 34 + 26
L.append(f'<line x1="{CX}" y1="{y_written+34}" x2="{CX}" y2="{y_arrow2}" '
         'stroke="#1d4ed8" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{CX+14}" y="{y_written+34+22}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="#1d4ed8">{esc("② store 整段回写 — DMA 第 2 趟，8 元素")}</text>')

# final target box
y_final = y_arrow2 + 8
fin_w, fin_h = 320, 46
L.append(f'<rect x="{CX-fin_w/2}" y="{y_final}" width="{fin_w}" height="{fin_h}" rx="8" '
         'fill="#0369a1" stroke="#0c4a6e" stroke-width="2"/>')
L.append(f'<text x="{CX}" y="{y_final+fin_h/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="white">{esc("dst 内存写回完成")}</text>')

# caption
foot_y = y_final + fin_h + 46
L.append(f'<rect x="{PAD}" y="{foot_y-30}" width="{W-2*PAD}" height="60" rx="8" '
         'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
L.append(f'<text x="{PAD+16}" y="{foot_y-8}" font-family="sans-serif" font-size="12.5" '
         f'fill="#334155">{esc("实际改动 4 个元素（idx∈{0,1,6,7}），却付出 2 趟 8 元素全量 DMA（load origin + store）")}</text>')
L.append(f'<text x="{PAD+16}" y="{foot_y+12}" font-family="sans-serif" font-size="12.5" '
         f'fill="#334155">{esc("+ 1 次 select——逻辑上 O(4) 的写，实际成本 O(8) 且翻倍。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m5-rmw.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
