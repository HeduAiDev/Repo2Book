#!/usr/bin/env python3
"""flow 模板:三类共享内存 buffer 来源(Explicit/Scratch/Virtual)各自的字节数公式,
汇入同一个 first-fit 定址器。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "共享内存 buffer 的三类来源 → 统一 first-fit 定址"

SOURCES = [
    {
        "name": "Explicit",
        "sub": "local_alloc",
        "lines": [
            "字节数 = product(shapePerCTA) x bitWidth / 8",
            "例:16x16 f16 = 256 x 16 / 8 = 512 字节",
            "默认对齐:512B>256 -> 1024",
        ],
        "src": "lib/Analysis/Allocation.cpp:L203-L205",
        "fill": "#dbeafe", "stroke": "#1e40af",
    },
    {
        "name": "Scratch",
        "sub": "convert_layout",
        "lines": [
            "字节数 = elems x max(8, bitWidth) / 8",
            "对齐 scratchAlignment = 128 字节",
            "指针型 scratch: kPtrBitWidth = 64",
        ],
        "src": "lib/Analysis/Allocation.cpp:L276-L278, L239, L47",
        "fill": "#dcfce7", "stroke": "#15803d",
    },
    {
        "name": "Virtual",
        "sub": "函数调用",
        "lines": [
            "字节数 = 被调函数的 sharedMemorySize",
            "调用者无需展开被调函数",
            "即为其预留整块空间",
        ],
        "src": "lib/Analysis/Allocation.cpp:L298-L302",
        "fill": "#fef3c7", "stroke": "#b45309",
    },
]

CARD_W, CARD_H, GAP, PAD, TOP = 340, 168, 46, 40, 96
n = len(SOURCES)
w = PAD * 2 + CARD_W * n + GAP * (n - 1)
FUNNEL_H = 90
SINK_H = 74
h = TOP + CARD_H + FUNNEL_H + SINK_H + PAD + 40

card_x = [PAD + i * (CARD_W + GAP) for i in range(n)]
sink_w = CARD_W * n + GAP * (n - 1)
sink_x = PAD
sink_y = TOP + CARD_H + FUNNEL_H

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-10}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>']

for i, s in enumerate(SOURCES):
    x = card_x[i]
    L.append(f'<rect x="{x}" y="{TOP}" width="{CARD_W}" height="{CARD_H}" rx="10" '
              f'fill="{s["fill"]}" stroke="{s["stroke"]}" stroke-width="2"/>')
    L.append(f'<text x="{x+18}" y="{TOP+28}" font-family="sans-serif" font-size="15" '
              f'font-weight="bold" fill="#0f172a">{esc(s["name"])}</text>')
    L.append(f'<text x="{x+CARD_W-18}" y="{TOP+28}" text-anchor="end" font-family="sans-serif" '
              f'font-size="12" fill="#475569">{esc(s["sub"])}</text>')
    for k, line in enumerate(s["lines"]):
        L.append(f'<text x="{x+18}" y="{TOP+54+k*22}" font-family="sans-serif" font-size="12.5" '
                  f'fill="#1e293b">{esc(line)}</text>')
    L.append(f'<text x="{x+18}" y="{TOP+CARD_H-14}" font-family="sans-serif" font-size="10.5" '
              f'fill="#64748b">{esc(s["src"])}</text>')
    # funnel line: card bottom-center -> sink top-center-ish
    cx = x + CARD_W / 2
    fy1 = TOP + CARD_H
    fy2 = sink_y
    tx = sink_x + sink_w / 2
    L.append(f'<line x1="{cx}" y1="{fy1}" x2="{tx}" y2="{fy2}" '
              'stroke="#334155" stroke-width="1.5" marker-end="url(#a)" opacity="0.75"/>')

L.append(f'<rect x="{sink_x}" y="{sink_y}" width="{sink_w}" height="{SINK_H}" rx="10" '
          'fill="#ede9fe" stroke="#6d28d9" stroke-width="2.5"/>')
L.append(f'<text x="{sink_x+sink_w/2}" y="{sink_y+30}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="15" font-weight="bold" '
          f'fill="#4c1d95">{esc("first-fit 定址(冲突图 + 贪心染色)")}</text>')
L.append(f'<text x="{sink_x+sink_w/2}" y="{sink_y+52}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" '
          f'fill="#5b21b6">{esc("三类字节数统一进同一张冲突图,算出整个 call graph 的 sharedMemorySize")}</text>')

foot_y = h - 12
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc("Explicit 按张量形状 x 位宽;Scratch 按每轮最大访问 x 位宽(128B 对齐避 bank conflict);Virtual 直接借用被调函数的整块需求。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-buffer-kinds.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
