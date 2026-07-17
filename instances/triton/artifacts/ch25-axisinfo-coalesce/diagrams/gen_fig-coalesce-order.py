#!/usr/bin/env python3
"""layout 模板：argSort(contiguity) 把最连续的轴排到 order[0]（最内层）。
用「轴排序卡片 + 内存地址条」示意：正确 order=[1,0] 让相邻 lane 落在相邻地址；
误序 order=[0,1] 让相邻 lane 跳步 64，访存放大。不逐格画满 32x64（不可读），
标题已注明"示意条带，取代表性 lane"。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "argSort(contiguity) 定 order：最连续的轴排到最内层"
SUBTITLE = "2D 张量 32x64（行主序）：axis0 contiguity=1，axis1 contiguity=64——示意条带，取 8 个代表性 lane"

PAD, TOP = 40, 108
CELL, GAP = 60, 6
N_LANE = 8

w = PAD * 2 + max(N_LANE * (CELL + GAP), 1080)
h = TOP + 2 * (150) + 170


def lane_strip(y0, addrs, colors, panel_title, title_color, note):
    parts = [f'<text x="{PAD}" y="{y0-14}" font-family="sans-serif" font-size="14" '
             f'font-weight="bold" fill="{title_color}">{esc(panel_title)}</text>']
    for i in range(N_LANE):
        x = PAD + i * (CELL + GAP)
        parts.append(f'<rect x="{x}" y="{y0}" width="{CELL}" height="46" rx="6" '
                      f'fill="{colors[i]}" stroke="#475569" stroke-width="1"/>')
        parts.append(f'<text x="{x+CELL/2}" y="{y0+19}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="11" fill="#0f172a">'
                      f'{esc("lane "+str(i))}</text>')
        parts.append(f'<text x="{x+CELL/2}" y="{y0+37}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12" font-weight="bold" '
                      f'fill="#0f172a">{esc("addr "+str(addrs[i]))}</text>')
        if i < N_LANE - 1:
            parts.append(f'<line x1="{x+CELL+2}" y1="{y0+23}" x2="{x+CELL+GAP-2}" y2="{y0+23}" '
                          'stroke="#94a3b8" stroke-width="1.5" marker-end="url(#a)"/>')
    parts.append(f'<text x="{PAD}" y="{y0+70}" font-family="sans-serif" font-size="12" '
                  f'fill="#374151">{esc(note)}</text>')
    return parts


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="16" font-weight="bold" '
     f'fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="54" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc(SUBTITLE)}</text>']

# Panel A: order=[1,0] correct — addresses step by 1 (axis1 contiguous inner)
addrs_a = [16 + i for i in range(N_LANE)]
colors_a = ["#93c5fd"] * N_LANE
L.extend(lane_strip(TOP, addrs_a, colors_a,
                     "正确：order=[1,0]（axis1 内层，contiguity=64 排 order[0]）",
                     "#047857",
                     "相邻 lane 地址步长 = 1 → 8 个 lane 落在同一合并访存事务"))

# Panel B: order=[0,1] wrong — addresses step by 64 (axis0 outer wrongly inner)
y2 = TOP + 150
addrs_b = [16 + i * 64 for i in range(N_LANE)]
colors_b = ["#fca5a5"] * N_LANE
L.extend(lane_strip(y2, addrs_b, colors_b,
                     "误序：order=[0,1]（axis0 误放内层，contiguity=1）",
                     "#b91c1c",
                     "相邻 lane 地址步长 = 64 → 8 个 lane 落进 8 个不同事务，放大 64 倍"))

# summary table
tbl_y = y2 + 100
rows = [
    ("axis0 contiguity", "1"),
    ("axis1 contiguity", "64"),
    ("argSort([1,64]) 降序 → order", "[1,0]"),
    ("误序 order=[0,1] 访存事务放大倍数", "64"),
]
row_h = 30
tbl_w = 700
label_w = tbl_w * 0.7
for i, (lbl, val) in enumerate(rows):
    ry = tbl_y + i * row_h
    fill = "#f8fafc" if i % 2 == 0 else "white"
    L.append(f'<rect x="{PAD}" y="{ry}" width="{tbl_w}" height="{row_h}" fill="{fill}" '
              'stroke="#e2e8f0" stroke-width="1"/>')
    L.append(f'<text x="{PAD+14}" y="{ry+row_h/2+5}" font-family="sans-serif" font-size="12" '
              f'fill="#374151">{esc(lbl)}</text>')
    L.append(f'<text x="{PAD+label_w+14}" y="{ry+row_h/2+5}" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="#1e40af">{esc(val)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-coalesce-order.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
