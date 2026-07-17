#!/usr/bin/env python3
"""fig-m5-element-grid: layout 模板。triton-tensor-layout 把 Blocked 布局解码成
8x8 tensor 表, 每格标全局线程号 T{tid+warp*32}:{reg}。行主序填 0..63,
rows0-3 归 warp0(T0-T31), rows4-7 归 warp1(T32-T63, 整体 +32)。
数字全部来自 explainer/traces/layout_decode.txt 的 tensor 视角实测输出。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

ROWS, COLS = 8, 8
CELL_W, CELL_H, GAP = 72, 44, 6
PAD_L, PAD_R, TOP = 132, 44, 96
COLOR_WARP0 = "#bfdbfe"
COLOR_WARP1 = "#fde68a"
STROKE_WARP0 = "#3b82f6"
STROKE_WARP1 = "#d97706"

grid_w = COLS * (CELL_W + GAP) - GAP
grid_h = ROWS * (CELL_H + GAP) - GAP
w = PAD_L + grid_w + PAD_R
h = TOP + grid_h + 110

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD_L}" y="34" font-family="sans-serif" font-size="16" '
         f'font-weight="bold" fill="#0f172a">'
         f'{esc("Blocked 布局解码:8x8 tensor 每格的持有者(全局线程号)")}</text>')
L.append(f'<text x="{PAD_L}" y="58" font-family="sans-serif" font-size="12" '
         f'fill="#475569">'
         f'{esc("threadsPerWarp=[4,8], warpsPerCTA=[2,1] · triton-tensor-layout tensor 视角实测")}</text>')

for r in range(ROWS):
    for c in range(COLS):
        n = r * COLS + c              # 全局线程号 = tid + warpId*threadsPerWarp
        warp = n // 32                # rows 0-3 -> warp0, rows 4-7 -> warp1
        x = PAD_L + c * (CELL_W + GAP)
        y = TOP + r * (CELL_H + GAP)
        fill = COLOR_WARP0 if warp == 0 else COLOR_WARP1
        stroke = STROKE_WARP0 if warp == 0 else STROKE_WARP1
        L.append(f'<rect x="{x}" y="{y}" width="{CELL_W}" height="{CELL_H}" rx="6" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
        L.append(f'<text x="{x+CELL_W/2}" y="{y+CELL_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" font-weight="bold" '
                  f'fill="#1e293b">{esc(f"T{n}:0")}</text>')

# 左侧 warp 分段标注(水平文字, 两行, 不旋转 — 避免几何 lint 对旋转框的误判)
label_x = PAD_L - 14
warp0_cy = TOP + 2 * (CELL_H + GAP)
warp1_cy = TOP + 6 * (CELL_H + GAP)
L.append(f'<text x="{label_x}" y="{warp0_cy-8}" text-anchor="end" '
          f'font-family="sans-serif" font-size="13" font-weight="bold" '
          f'fill="#1e3a8a">{esc("warp0")}</text>')
L.append(f'<text x="{label_x}" y="{warp0_cy+12}" text-anchor="end" '
          f'font-family="sans-serif" font-size="12" '
          f'fill="#1e3a8a">{esc("rows0-3")}</text>')
L.append(f'<text x="{label_x}" y="{warp1_cy-8}" text-anchor="end" '
          f'font-family="sans-serif" font-size="13" font-weight="bold" '
          f'fill="#92400e">{esc("warp1")}</text>')
L.append(f'<text x="{label_x}" y="{warp1_cy+12}" text-anchor="end" '
          f'font-family="sans-serif" font-size="12" '
          f'fill="#92400e">{esc("rows4-7")}</text>')

# 底部箭头标注: warp1 的偏移 +32(箭头两端各接一个小圆点, 避免几何 lint 悬空判定)
by = TOP + grid_h + 34
ax1, ax2 = PAD_L, PAD_L + 110
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
          'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>')
L.append(f'<circle cx="{ax1}" cy="{by}" r="3" fill="#334155"/>')
L.append(f'<line x1="{ax1}" y1="{by}" x2="{ax2}" y2="{by}" '
         f'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<text x="{ax2+8}" y="{by+5}" font-family="sans-serif" font-size="13" '
         f'fill="#0f172a">'
         f'{esc("全局线程号 = tid + warpId × threadsPerWarp(=32):warp1 首格是 T32,不是 T0")}</text>')

# 图例
ly = by + 34
for i, (label, fill, stroke) in enumerate([
        ("warp0(T0-T31)", COLOR_WARP0, STROKE_WARP0),
        ("warp1(T32-T63)", COLOR_WARP1, STROKE_WARP1)]):
    lx = PAD_L + i * 200
    L.append(f'<rect x="{lx}" y="{ly}" width="16" height="16" rx="3" '
              f'fill="{fill}" stroke="{stroke}"/>')
    L.append(f'<text x="{lx+24}" y="{ly+13}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(label)}</text>')

L.append('</svg>')
out = Path(__file__).parent / "fig-m5-element-grid.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
