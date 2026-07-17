#!/usr/bin/env python3
"""fig-warps-per-tile (tiling 模板)
128x128 输出 tile 被 warpsPerTileV2 贪心切成 warpsPerCTA=[4,2] 片(8 个 warp),
每片内再铺 instrShape=[16,8] 的 mma 砖(2x8=16 砖/warp)。全部坐标由循环计算。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

M, N = 128, 128
WARPS_M, WARPS_N = 4, 2          # warpsPerCTA = [4,2]
INSTR_M, INSTR_N = 16, 8         # instrShape = [16,8]
SCALE = 3.4                      # px per unit

grid_w = N * SCALE
grid_h = M * SCALE
PAD = 60
TOP = 150
SIDE_W = 300
GAP = 50

w = PAD + grid_w + GAP + SIDE_W + PAD
h = TOP + grid_h + 90

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>']

L.append(f'<text x="{PAD}" y="48" font-family="sans-serif" font-size="19" '
         f'font-weight="bold" fill="#0f172a">{esc("warpsPerTile:把 16x8 的 mma 砖平铺满 128x128 输出 tile")}</text>')
L.append(f'<text x="{PAD}" y="72" font-family="sans-serif" font-size="13" '
         f'fill="#475569">{esc("warpsPerTileV2 贪心翻倍(AccelerateMatmul.cpp:L82-L104)——每 warp 分到方正一片、片内再迭代多块砖")}</text>')

gx, gy = PAD, TOP

warp_w = grid_w / WARPS_N   # 每 warp 覆盖的宽度(N 方向)
warp_h = grid_h / WARPS_M   # 每 warp 覆盖的高度(M 方向)
brick_w = INSTR_N * SCALE
brick_h = INSTR_M * SCALE
bricks_per_warp_n = int(round(warp_w / brick_w))
bricks_per_warp_m = int(round(warp_h / brick_h))

WARP_FILL = "#e2e8f0"
HI_FILL, HI_STROKE = "#93c5fd", "#1d4ed8"
HI_WARP = (0, 0)

# 外框
L.append(f'<rect x="{gx}" y="{gy}" width="{grid_w}" height="{grid_h}" fill="white" stroke="#0f172a" stroke-width="1"/>')

# warp 分片(粗线)+ 高亮一片 + 片内 instrShape 细网格
for wm in range(WARPS_M):
    for wn in range(WARPS_N):
        wx = gx + wn * warp_w
        wy = gy + wm * warp_h
        is_hi = (wm, wn) == HI_WARP
        fill = HI_FILL if is_hi else WARP_FILL
        L.append(f'<rect x="{wx}" y="{wy}" width="{warp_w}" height="{warp_h}" '
                  f'fill="{fill}" fill-opacity="{0.9 if is_hi else 0.5}" '
                  f'stroke="#0f172a" stroke-width="2"/>')
        # instrShape 细网格线(片内)
        for bm in range(1, bricks_per_warp_m):
            by = wy + bm * brick_h
            L.append(f'<line x1="{wx}" y1="{by}" x2="{wx+warp_w}" y2="{by}" '
                      f'stroke="{"#1d4ed8" if is_hi else "#94a3b8"}" stroke-width="0.8" opacity="0.7"/>')
        for bn in range(1, bricks_per_warp_n):
            bx = wx + bn * brick_w
            L.append(f'<line x1="{bx}" y1="{wy}" x2="{bx}" y2="{wy+warp_h}" '
                      f'stroke="{"#1d4ed8" if is_hi else "#94a3b8"}" stroke-width="0.8" opacity="0.7"/>')
        L.append(f'<text x="{wx+warp_w/2}" y="{wy+warp_h/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" font-weight="bold" '
                  f'fill="#0f172a">{esc(f"warp({wm},{wn})")}</text>')

# 高亮 warp 的砖数标注(箭头指向高亮块)
hi_x = gx + HI_WARP[1] * warp_w
hi_y = gy + HI_WARP[0] * warp_h
L.append(f'<rect x="{hi_x}" y="{hi_y}" width="{warp_w}" height="{warp_h}" '
          f'fill="none" stroke="{HI_STROKE}" stroke-width="3"/>')

# 轴标注:M/N 总长度
L.append(f'<text x="{gx-14}" y="{gy+grid_h/2}" text-anchor="end" font-family="sans-serif" '
         f'font-size="13" fill="#334155" transform="rotate(-90 {gx-14} {gy+grid_h/2})">'
         f'{esc("M = 128")}</text>')
L.append(f'<text x="{gx+grid_w/2}" y="{gy+grid_h+22}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="13" fill="#334155">{esc("N = 128")}</text>')

# 侧栏
side_box_x = gx + grid_w + GAP
side_x = side_box_x + 20
side_y = TOP + 26
L.append(f'<rect x="{side_box_x}" y="{TOP-24}" width="{SIDE_W}" height="{grid_h+24}" rx="10" '
          'fill="#f8fafc" stroke="#cbd5e1"/>')
lines = [
    ("numWarps = 8", True),
    ("warpsPerCTA = [4, 2]", True),
    ("(warpsPerTileV2 贪心结果)", False),
    ("instrShape = [16, 8]", True),
    ("= shapePerWarp,单条 mma 的砖", False),
    ("", False),
    (f"高亮 warp(0,0) 覆盖:", False),
    (f"M 方向 128/16/4 = 2 砖", False),
    (f"N 方向 128/8/2 = 8 砖", False),
    (f"-> 每 warp 迭代 2x8 = 16 砖", True),
    ("", False),
    ("MMAv3 最小不可分单元 (4,1)", False),
    ("= 一个 warpgroup(4 warps)", False),
    ("(AccelerateMatmul.cpp:L119-L120)", False),
]
yy = side_y
for text, bold in lines:
    if text:
        L.append(f'<text x="{side_x}" y="{yy}" font-family="sans-serif" font-size="12.5" '
                  f'font-weight="{"bold" if bold else "normal"}" '
                  f'fill="{"#0f172a" if bold else "#475569"}">{esc(text)}</text>')
    yy += 22

cap = "128x128 tile 由贪心分配表切成 4x2=8 个 warp 片;每片(如 warp(0,0))再迭代 16 块 16x8 的 instrShape 砖。"
L.append(f'<text x="{PAD}" y="{h-16}" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc(cap)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-warps-per-tile.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w}x{h}  bricks/warp: {bricks_per_warp_m}x{bricks_per_warp_n}")
