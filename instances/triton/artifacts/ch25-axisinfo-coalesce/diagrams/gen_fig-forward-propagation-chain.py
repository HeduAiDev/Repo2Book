#!/usr/bin/env python3
"""tensor-flow 模板：三元组沿 make_range→splat→addptr 前向传播，每个 op 一张
visitor 配方卡，边上标注三元组变化。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "稀疏前向数据流：每个 op 一张 visitor 配方卡"
SUBTITLE = "make_range(0,1024) → splat(x_ptr) → addptr —— 三元组 (contiguity, divisibility字节, constancy) 逐 op 更新"

NODES = [
    {
        "op": "make_range 0..1024",
        "visitor": "MakeRangeOpAxisInfoVisitor",
        "triple": "(1024, 1073741824, 1)",
        "note": "造源头：contiguity=end-start",
    },
    {
        "op": "splat x_ptr",
        "visitor": "SplatOpAxisInfoVisitor",
        "triple": "(1, 16, 1024)",
        "note": "标量吹成张量：contiguity→1，constancy→1024",
    },
    {
        "op": "addptr %b, %r",
        "visitor": "AddSubOpAxisInfoVisitor<AddPtrOp>",
        "triple": "(1024, 16, 1)",
        "note": "叠偏移：静态真相汇成于此",
    },
]

BOX_W, BOX_H, GAP, PAD, TOP = 300, 110, 90, 40, 96
w = PAD * 2 + BOX_W * len(NODES) + GAP * (len(NODES) - 1)
h = TOP + BOX_H + 90

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#1e3a5f"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="16" font-weight="bold" '
     f'fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="50" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc(SUBTITLE)}</text>']

xs_ = [PAD + i * (BOX_W + GAP) for i in range(len(NODES))]
box_y = TOP

for i, node in enumerate(NODES):
    x = xs_[i]
    is_last = (i == len(NODES) - 1)
    fill = "#ecfdf5" if is_last else "#eff6ff"
    stroke = "#047857" if is_last else "#3b82f6"
    L.append(f'<rect x="{x}" y="{box_y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{box_y+24}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" '
              f'fill="#0f172a">{esc(node["op"])}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{box_y+44}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="#64748b" '
              f'font-style="italic">{esc(node["visitor"])}</text>')
    L.append(f'<rect x="{x+16}" y="{box_y+54}" width="{BOX_W-32}" height="26" rx="4" '
              f'fill="white" stroke="{stroke}" stroke-width="1"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{box_y+71}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{stroke}">{esc(node["triple"])}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{box_y+BOX_H-12}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="#334155">{esc(node["note"])}</text>')
    if not is_last:
        ax1 = x + BOX_W
        ax2 = ax1 + GAP
        ay = box_y + BOX_H / 2
        L.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2-6}" y2="{ay}" '
                  'stroke="#1e3a5f" stroke-width="2" marker-end="url(#a)"/>')

L.append(f'<text x="{PAD}" y="{TOP+BOX_H+50}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">(contiguity, divisibility字节, constancy)——三元组沿箭头前向传播；'
          f'source→splat→addptr 一趟直链即到不动点，无需回环。</text>')
L.append(f'<text x="{PAD}" y="{TOP+BOX_H+68}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">绿框 = 最终汇成的静态真相 (1024, 16 字节, 1)，交给 Coalesce 消费。</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-forward-propagation-chain.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
