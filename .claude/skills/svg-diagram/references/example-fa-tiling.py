#!/usr/bin/env python3
"""tiling 模板:分块/many-to-many 连接。左列源分块(各自独立配色),
右列目标分块(统一配色),源到每个目标各画一条箭头(同源同色)。
改造点:SRC(源分块名)、DST(目标分块名)、SRC_COLORS(每个源的颜色)。
全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "FlashAttention Tiling — Q 分块 × KV 分块"
SRC = ["Q0  tokens 0-3", "Q1  tokens 4-7", "Q2  tokens 8-11"]
DST = ["K0V0", "K1V1", "K2V2"]
CAPTION = f"每个 Q 分块只加载一次;KV 分块被重复读取 {len(SRC)} 次,以换取避免 O(L^2) 的中间态显存往返。"
SRC_COLORS = ["#2563eb", "#7c3aed", "#059669"]      # 每个源分块自己的箭头/边框色
SRC_FILLS = ["#3b82f6", "#60a5fa", "#93c5fd"]        # 左列分块本身的填充色(渐浅)
DST_FILL, DST_STROKE = "#fef3c7", "#b45309"

BOX_W, BOX_H, VGAP, PAD, TOP, GAP_H = 210, 46, 22, 40, 60, 140
n_src, n_dst = len(SRC), len(DST)
w = PAD * 2 + BOX_W * 2 + GAP_H
h = TOP + max(n_src, n_dst) * (BOX_H + VGAP) + PAD + 30
src_x, dst_x = PAD, PAD + BOX_W + GAP_H
src_y = [TOP + i * (BOX_H + VGAP) for i in range(n_src)]
dst_y = [TOP + j * (BOX_H + VGAP) for j in range(n_dst)]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-14}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>']

for i, name in enumerate(SRC):  # 左列:源分块
    y = src_y[i]
    L.append(f'<rect x="{src_x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="4" '
              f'fill="{SRC_FILLS[i % len(SRC_FILLS)]}" stroke="#1e3a5f" stroke-width="2"/>')
    L.append(f'<text x="{src_x+BOX_W/2}" y="{y+BOX_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="white">{esc(name)}</text>')
for j, name in enumerate(DST):  # 右列:目标分块(统一配色)
    y = dst_y[j]
    L.append(f'<rect x="{dst_x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="4" '
              f'fill="{DST_FILL}" stroke="{DST_STROKE}" stroke-width="2"/>')
    L.append(f'<text x="{dst_x+BOX_W/2}" y="{y+BOX_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="#92400e">{esc(name)}</text>')

for i in range(n_src):  # 箭头:每个源 -> 每个目标,同源同色,端点在盒内错开避免重叠
    color = SRC_COLORS[i % len(SRC_COLORS)]
    for j in range(n_dst):
        y1 = src_y[i] + BOX_H / 2 - (n_dst - 1) * 2 + j * 4
        y2 = dst_y[j] + BOX_H / 2 - (n_src - 1) * 2 + i * 4
        L.append(f'<line x1="{src_x+BOX_W}" y1="{y1}" x2="{dst_x}" y2="{y2}" '
                  f'stroke="{color}" stroke-width="1.5" marker-end="url(#a)" opacity="0.6"/>')

foot_y = h - 12
L.append(f'<text x="{w/2}" y="{foot_y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#64748b">{esc(CAPTION)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("example-fa-tiling.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
