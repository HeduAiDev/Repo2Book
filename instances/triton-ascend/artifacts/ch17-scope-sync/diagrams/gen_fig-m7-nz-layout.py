#!/usr/bin/env python3
"""layout 模板:fp16 memref[32,64] 重排为 4D nz(4,2,16,16),最内维 16*2B=32B 对齐。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK = "#0f172a"
GRAY = "#64748b"
BLUE = "#1e40af"
BLUE_BG = "#dbeafe"
AMBER = "#b45309"
AMBER_BG = "#fef3c7"

TITLE = "cube 读 CBUF 要 32B 对齐:2D memref → 4D nz 分形"
SUB = "fp16, M=32, N=64: blk=32/2=16 ⇒ nz shape=(N/blk,M/16,16,blk)=(4,2,16,16), 最内维=16×2B=32B (DAGSync.cpp:L386-418)"

PAD, TOP = 40, 130
W = 1400
H = TOP + 560

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="{INK}">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+24}" font-family="sans-serif" font-size="12.5" fill="{GRAY}">{esc(SUB)}</text>']

# ---- LEFT: 2D logical matrix [32,64], drawn as grid of 16-row bands x 16-col bands (coarse) ----
L.append(f'<text x="{PAD+170}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="{INK}">2D 逻辑 memref[M=32, N=64]</text>')
cell, gap = 20, 2
rows2d, cols2d = 32, 64
# downscale visually: draw as 2 row-bands (each 16 rows) x 4 col-bands (each 16 cols)
band_w, band_h = 78, 78
bx0, by0 = PAD + 10, TOP + 10
for r in range(2):
    for c in range(4):
        x = bx0 + c * (band_w + 6)
        y = by0 + r * (band_h + 6)
        L.append(f'<rect x="{x}" y="{y}" width="{band_w}" height="{band_h}" rx="4" '
                  f'fill="{BLUE_BG}" stroke="{BLUE}" stroke-width="1"/>')
        L.append(f'<text x="{x+band_w/2}" y="{y+band_h/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11" fill="{BLUE}">16×16</text>')
L.append(f'<text x="{bx0+ (band_w+6)*4/2 - 3}" y="{by0+(band_h+6)*2+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" fill="{GRAY}">2 行带 × 4 列带,每带 16×16 元素(共 32×64=2048)</text>')

# arrow to right
ax1 = bx0 + (band_w + 6) * 4 + 30
ay = by0 + (band_h + 6)
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
         'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>')
L.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax1+70}" y2="{ay}" stroke="#334155" '
         f'stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{ax1+35}" y="{ay-12}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="{INK}">重排</text>')
L.append(f'<text x="{ax1+35}" y="{ay+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="9.5" fill="{GRAY}">newCbubAllocShape</text>')

# ---- RIGHT: 4D nz shape (4,2,16,16) drawn as 4 outer blocks x 2 sub-blocks, each a 16-row x blk-col cell ----
rx0 = ax1 + 100
L.append(f'<text x="{rx0+280}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="{INK}">4D nz shape=(4, 2, 16, 16)</text>')
outer_w, outer_h = 130, 170
for o in range(4):  # N/blk = 4 outer blocks (arranged 2x2 for compactness)
    ox = rx0 + (o % 2) * (outer_w + 16)
    oy = by0 + (o // 2) * (outer_h + 16)
    L.append(f'<rect x="{ox}" y="{oy}" width="{outer_w}" height="{outer_h}" rx="6" '
              f'fill="#f8fafc" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>')
    L.append(f'<text x="{ox+outer_w/2}" y="{oy-6}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="{GRAY}">块[{o}] (N/blk 维)</text>')
    for s in range(2):  # M/16 = 2 sub-blocks stacked
        sx, sy = ox + 8, oy + 8 + s * 80
        L.append(f'<rect x="{sx}" y="{sy}" width="{outer_w-16}" height="72" rx="4" '
                  f'fill="{AMBER_BG}" stroke="{AMBER}" stroke-width="1.3"/>')
        L.append(f'<text x="{sx+(outer_w-16)/2}" y="{sy+30}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
                  f'fill="{AMBER}">16 行 × 16</text>')
        L.append(f'<text x="{sx+(outer_w-16)/2}" y="{sy+48}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="9.5" fill="{AMBER}">最内维 16×2B=32B</text>')

legend_y = by0 + 2 * (outer_h + 16) + 20
L.append(f'<rect x="{PAD}" y="{legend_y}" width="16" height="16" rx="3" fill="{BLUE_BG}" stroke="{BLUE}"/>')
L.append(f'<text x="{PAD+22}" y="{legend_y+13}" font-family="sans-serif" font-size="11.5" '
          f'fill="{INK}">2D 逻辑分带(未对齐)</text>')
L.append(f'<rect x="{PAD+240}" y="{legend_y}" width="16" height="16" rx="3" fill="{AMBER_BG}" stroke="{AMBER}"/>')
L.append(f'<text x="{PAD+262}" y="{legend_y+13}" font-family="sans-serif" font-size="11.5" '
          f'fill="{INK}">4D nz 分形块(每块最内维恰 32B)</text>')

# numbers table
tbl_y = legend_y + 46
L.append(f'<text x="{PAD}" y="{tbl_y}" font-family="sans-serif" font-size="12.5" font-weight="bold" '
          f'fill="{INK}">M=32, N=64, fp16 elem_bytes=2, blk=32/2=16</text>')
L.append(f'<text x="{PAD}" y="{tbl_y+22}" font-family="sans-serif" font-size="12.5" '
          f'fill="{INK}">nz shape=(N/blk, M/16, 16, blk)=(4, 2, 16, 16); 最内维字节=blk×elem_bytes=16×2=32</text>')

CAP = "2D 逻辑矩阵 → cube 要的 4D 分形 nz：16 是 cube 分形块高、blk 是 32B 对齐后的最内维宽。这就是 VECTOR→CUBE 搬运里 copy 之前那步 reshape 的目标形状。"
cap_y = tbl_y + 50
L.append(f'<text x="{PAD}" y="{cap_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="{INK}">{esc(CAP)}</text>')

L.append('</svg>')
out = Path(__file__).parent / "fig-m7-nz-layout.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
