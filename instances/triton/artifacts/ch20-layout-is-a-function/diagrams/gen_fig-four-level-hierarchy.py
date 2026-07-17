#!/usr/bin/env python3
"""fig-four-level-hierarchy —— distributed 的 L 用四级嵌套计算层次算出来。
左:CGA/CTA/Warp/Thread/Value 五级同构嵌套矩形(自顶向下切分);
右:shape=[4,4]/order=[0,1] 的具体 linear-id 例子(列优先填号),首列/末列高亮。
数据出处:TritonGPUAttrDefs.td:L470-L471(四级层次)/L473-L481(linear-id 例)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "四级计算层次生成 L —— CGA/CTA/Warp/Thread/Value 同构嵌套"

# —— 左:嵌套矩形 ——
LEVELS = [
    ("CGA", "1 个 CGA(cluster)"),
    ("CTA", "CTAs Per CGA"),
    ("Warp", "Warps Per CTA"),
    ("Thread", "Threads Per Warp"),
    ("Value", "Values Per Thread"),
]
NEST = 34  # 每层向内缩进量
OUTER_W, OUTER_H = 340, 300
LEFT_PAD, TOP = 40, 150

# —— 右:4x4 linear-id 网格 ——
GRID_SHAPE = [
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15],
]
GCELL = 56
GRID_W = GRID_H = 4 * GCELL
RIGHT_PAD = LEFT_PAD + OUTER_W + 120

w = RIGHT_PAD + GRID_W + 260
h = TOP + OUTER_H + 140

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="{40}" text-anchor="middle" font-family="sans-serif" '
     f'font-size="17" font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="{64}" text-anchor="middle" font-family="sans-serif" '
     f'font-size="12" fill="#64748b">TritonGPUAttrDefs.td:L470-L471:自顶向下切分,'
     f'上两级(CTA/Warp)线程号沿 order 连续填号</text>']

COLORS = ["#1e3a5f", "#1d4ed8", "#3b82f6", "#60a5fa", "#93c5fd"]
n = len(LEVELS)
for i, (name, desc) in enumerate(LEVELS):
    x = LEFT_PAD + i * NEST
    y = TOP + i * NEST
    ww = OUTER_W - i * NEST * 2
    hh = OUTER_H - i * NEST * 2
    fill = "none" if i < n - 1 else COLORS[i]
    L.append(f'<rect x="{x}" y="{y}" width="{ww}" height="{hh}" rx="10" '
              f'fill="{fill}" stroke="{COLORS[i]}" stroke-width="2.2"/>')
    text_fill = "white" if i == n - 1 else COLORS[i]
    L.append(f'<text x="{x+10}" y="{y+20}" font-family="sans-serif" font-size="13" '
              f'font-weight="bold" fill="{text_fill}">{esc(name)}</text>')

# 右侧图例:四级 "per" 关系列表(与嵌套框同色对应)
legend_x = LEFT_PAD
legend_y = TOP + OUTER_H + 34
for i in range(1, n):
    ly = legend_y + (i - 1) * 20
    L.append(f'<rect x="{legend_x}" y="{ly-11}" width="14" height="14" rx="3" '
              f'fill="{COLORS[i]}"/>')
    L.append(f'<text x="{legend_x+22}" y="{ly}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{i}. {esc(LEVELS[i][0])}: {esc(LEVELS[i][1])}</text>')

# —— 右:linear-id 4x4 网格 ——
grid_top_label_y = TOP - 40
L.append(f'<text x="{RIGHT_PAD+GRID_W/2}" y="{grid_top_label_y}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13" font-weight="bold" fill="#0f172a">'
          f'shape=[4,4], order=[0,1] &#8594; linear-id(TritonGPUAttrDefs.td:L473-L481)</text>')
FIRST_COL = {0, 1, 2, 3}
LAST_COL = {12, 13, 14, 15}
for r in range(4):
    for c in range(4):
        val = GRID_SHAPE[r][c]
        x = RIGHT_PAD + c * GCELL
        y = TOP + r * GCELL
        if val in FIRST_COL:
            fill, stroke = "#dbeafe", "#1d4ed8"
        elif val in LAST_COL:
            fill, stroke = "#fed7aa", "#c2410c"
        else:
            fill, stroke = "#f1f5f9", "#94a3b8"
        L.append(f'<rect x="{x}" y="{y}" width="{GCELL-4}" height="{GCELL-4}" rx="5" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        L.append(f'<text x="{x+(GCELL-4)/2}" y="{y+(GCELL-4)/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="15" font-weight="bold" '
                  f'fill="#0f172a">{val}</text>')

note_y = TOP + GRID_H + 30
L.append(f'<text x="{RIGHT_PAD}" y="{note_y}" font-family="sans-serif" font-size="12" '
          f'fill="#1d4ed8">首列(蓝){{0,1,2,3}} = linear-id 首列</text>')
L.append(f'<text x="{RIGHT_PAD}" y="{note_y+20}" font-family="sans-serif" font-size="12" '
          f'fill="#c2410c">末列(橙){{12,13,14,15}} = linear-id 末列</text>')

foot_y = h - 24
L.append(f'<text x="{w/2}" y="{foot_y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#64748b">底两级(Thread/Value)因子类而异 &#8212; '
          f'最直白的落地就是下一张 Blocked 图</text>')
L.append('</svg>')
out = Path(__file__).parent / "fig-four-level-hierarchy.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
