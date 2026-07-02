#!/usr/bin/env python3
"""layout 模板:内存/块表/KV 页布局。示例:8 个 KV block,3 个请求占用 + 空闲。
改造点:SLOTS(占用者列表,None=空闲)与 LEGEND。颜色即语义,>2 色必有图例。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

SLOTS = ["req1", "req1", "req2", None, "req3", "req3", "req3", None]  # block0..7
COLORS = {"req1": "#93c5fd", "req2": "#86efac", "req3": "#fcd34d", None: "#f1f5f9"}
LEGEND = [("req1", "请求 1(2 块)"), ("req2", "请求 2(1 块)"),
          ("req3", "请求 3(3 块)"), (None, "空闲")]
CELL, GAP, PAD, TOP = 84, 10, 40, 64
w = PAD * 2 + len(SLOTS) * (CELL + GAP) - GAP
h = TOP + CELL + 110

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{TOP-28}" font-family="sans-serif" font-size="14" '
     f'font-weight="bold" fill="#0f172a">{esc("KV cache 块池(block_size=16 token/块)")}</text>']
for i, owner in enumerate(SLOTS):
    x = PAD + i * (CELL + GAP)
    L.append(f'<rect x="{x}" y="{TOP}" width="{CELL}" height="{CELL}" rx="8" '
             f'fill="{COLORS[owner]}" stroke="#64748b"/>')
    L.append(f'<text x="{x+CELL/2}" y="{TOP+CELL/2-6}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="12" fill="#334155">block {i}</text>')
    L.append(f'<text x="{x+CELL/2}" y="{TOP+CELL/2+14}" text-anchor="middle" '
             'font-family="sans-serif" font-size="12" font-weight="bold" '
             f'fill="#0f172a">{esc(owner or "空闲")}</text>')
ly = TOP + CELL + 40  # 图例:>2 种语义色必有
for j, (key, label) in enumerate(LEGEND):
    lx = PAD + j * 180
    L.append(f'<rect x="{lx}" y="{ly}" width="16" height="16" rx="3" '
             f'fill="{COLORS[key]}" stroke="#64748b"/>')
    L.append(f'<text x="{lx+24}" y="{ly+13}" font-family="sans-serif" font-size="12" '
             f'fill="#334155">{esc(label)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("example-layout.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
