#!/usr/bin/env python3
"""fig-warps-per-tile (state-table + tiling 混合模板)
左:warpsPerTileV2 贪心翻倍的 4 轮迭代轨迹(iter/ret/prod/分支)。
右:128x128 输出 tile 最终被切成 4x2=8 个 warp,每 warp 独占 32x64 子块。
(与第 27 章 fig-warps-per-tile 不同焦点:那张图画的是砖 instrShape=[16,8] 在
单个 warp 内的铺贴;这张图画的是 warpsPerTileV2 贪心循环本身怎么产生 [4,2] 这个切分。)
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

# ---- 左:迭代轨迹表 ----
ITER_ROWS = [
    ("1", "[1,1]", "1", "8", "8", "LHS>=RHS 且 ret0<8\n-> M x2", "[2,1]"),
    ("2", "[2,1]", "2", "4", "8", "LHS<RHS -> N x2", "[2,2]"),
    ("3", "[2,2]", "4", "4", "4", "LHS>=RHS 且 ret0<8\n-> M x2", "[4,2]"),
    ("4", "[4,2]", "8", "-", "-", "prod>=numWarps -> break", "[4,2] (终)"),
]
COLS = ["iter", "ret 前", "prod", "LHS", "RHS", "分支", "ret 后"]
COL_W = [46, 62, 46, 46, 46, 148, 70]
ROW_H = 46
HEAD_H = 30
TAB_TOP = 150
TAB_X = 50

tab_w = sum(COL_W)
tab_h = HEAD_H + ROW_H * len(ITER_ROWS)

# ---- 右:最终网格 ----
M, N = 128, 128
WARPS_M, WARPS_N = 4, 2
SCALE = 2.35
grid_w = N * SCALE
grid_h = M * SCALE
GRID_X = TAB_X + tab_w + 140
GRID_TOP = TAB_TOP

w = GRID_X + grid_w + 260
h = TAB_TOP + max(tab_h, grid_h) + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#1d4ed8"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

L.append(f'<text x="{TAB_X}" y="46" font-family="sans-serif" font-size="19" '
          f'font-weight="bold" fill="#0f172a">{esc("warpsPerTileV2:4 轮贪心翻倍,把 8 个 warp 分成 4x2")}</text>')
L.append(f'<text x="{TAB_X}" y="70" font-family="sans-serif" font-size="12.5" '
          f'fill="#475569">{esc("每轮往『剩余空间更大的轴』翻倍该维 warp 数,直到 prod>=numWarps=8 触发 break(AccelerateMatmul.cpp:L82-L104)")}</text>')

# --- 左表 ---
L.append(f'<text x="{TAB_X}" y="{TAB_TOP-14}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#1e3a5f">{esc("迭代轨迹(shapePerWarp=[16,8], M=N=128, numWarps=8)")}</text>')
cx0 = TAB_X
col_x = []
for cw in COL_W:
    col_x.append(cx0)
    cx0 += cw

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TAB_TOP}" width="{COL_W[j]-4}" height="{HEAD_H-4}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1"/>')
    L.append(f'<text x="{x+(COL_W[j]-4)/2}" y="{TAB_TOP+(HEAD_H-4)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="white" font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ITER_ROWS):
    ry = TAB_TOP + HEAD_H + i * ROW_H
    is_last = (i == len(ITER_ROWS) - 1)
    for j, val in enumerate(row):
        x = col_x[j]
        fill = "#ecfdf5" if is_last and j == 6 else ("#f8fafc" if i % 2 == 0 else "white")
        stroke = "#16a34a" if is_last and j == 6 else "#cbd5e1"
        L.append(f'<rect x="{x}" y="{ry+2}" width="{COL_W[j]-4}" height="{ROW_H-4}" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{1.6 if (is_last and j==6) else 0.8}"/>')
        lines = val.split("\n")
        y0 = ry + ROW_H / 2 - (len(lines) - 1) * 6 + 3
        fw = 'font-weight="bold" ' if (is_last and j == 6) else ''
        fill_txt = "#166534" if (is_last and j == 6) else "#334155"
        for k, ln in enumerate(lines):
            L.append(f'<text x="{x+(COL_W[j]-4)/2}" y="{y0+k*13}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="10" {fw}fill="{fill_txt}">{esc(ln)}</text>')

# 左表下方小结箭头 → 指向右侧网格
sum_y = TAB_TOP + HEAD_H + len(ITER_ROWS) * ROW_H + 34
L.append(f'<text x="{TAB_X}" y="{sum_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="#0f172a">{esc("循环进 4 次(log2(8)=3 次翻倍 + 1 次 break 检查)-> warpsPerTile=[4,2]")}</text>')
arrow_y = sum_y + 30
L.append(f'<line x1="{TAB_X + tab_w - 40}" y1="{arrow_y}" x2="{GRID_X-16}" y2="{arrow_y}" '
          'stroke="#1d4ed8" stroke-width="2.2" marker-end="url(#a)"/>')

# --- 右网格 ---
L.append(f'<text x="{GRID_X}" y="{GRID_TOP-14}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#1e3a5f">{esc("最终结果:128x128 输出 tile 切成 4x2=8 个 warp")}</text>')
L.append(f'<rect x="{GRID_X}" y="{GRID_TOP}" width="{grid_w}" height="{grid_h}" '
          'fill="white" stroke="#0f172a" stroke-width="1.5"/>')

warp_w = grid_w / WARPS_N
warp_h = grid_h / WARPS_M
HI_FILL, HI_STROKE = "#93c5fd", "#1d4ed8"
NORM_FILL = "#e2e8f0"
for wm in range(WARPS_M):
    for wn in range(WARPS_N):
        wx = GRID_X + wn * warp_w
        wy = GRID_TOP + wm * warp_h
        is_hi = (wm, wn) == (0, 0)
        fill = HI_FILL if is_hi else NORM_FILL
        L.append(f'<rect x="{wx}" y="{wy}" width="{warp_w}" height="{warp_h}" '
                  f'fill="{fill}" fill-opacity="{0.95 if is_hi else 0.55}" '
                  f'stroke="#0f172a" stroke-width="1.6"/>')
        idx = wm * WARPS_N + wn
        L.append(f'<text x="{wx+warp_w/2}" y="{wy+warp_h/2-6}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" font-weight="bold" '
                  f'fill="#0f172a">{esc(f"warp{idx}")}</text>')
        L.append(f'<text x="{wx+warp_w/2}" y="{wy+warp_h/2+12}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" '
                  f'fill="#334155">{esc("32x64")}</text>')

# 坐标轴标注
L.append(f'<text x="{GRID_X+grid_w/2}" y="{GRID_TOP+grid_h+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" fill="#64748b">{esc("N = 128 (沿此轴切 2 份)")}</text>')
m_label_x = GRID_X - 100
L.append(f'<text x="{m_label_x}" y="{GRID_TOP+grid_h/2-8}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" fill="#64748b">{esc("M = 128")}</text>')
L.append(f'<text x="{m_label_x}" y="{GRID_TOP+grid_h/2+10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" fill="#64748b">{esc("(沿此轴切 4 份)")}</text>')

# 侧注:shapePerWarp 参考单元
note_x = GRID_X + grid_w + 30
L.append(f'<rect x="{note_x}" y="{GRID_TOP}" width="200" height="96" rx="8" '
          'fill="#f8fafc" stroke="#cbd5e1"/>')
L.append(f'<text x="{note_x+14}" y="{GRID_TOP+22}" font-family="sans-serif" font-size="11.5" '
          f'font-weight="bold" fill="#0f172a">{esc("参考单元")}</text>')
L.append(f'<text x="{note_x+14}" y="{GRID_TOP+42}" font-family="sans-serif" font-size="11" '
          f'fill="#334155">{esc("shapePerWarp = [16,8]")}</text>')
L.append(f'<text x="{note_x+14}" y="{GRID_TOP+60}" font-family="sans-serif" font-size="11" '
          f'fill="#334155">{esc("(贪心每步的最小单位)")}</text>')
L.append(f'<text x="{note_x+14}" y="{GRID_TOP+80}" font-family="sans-serif" font-size="11" '
          f'fill="#334155">{esc("每 warp 子块 = 128/4 x 128/2")}</text>')

cap = "8 个 warp 沿 M 分 4、沿 N 分 2,每 warp 管 32x64——warp 数与 tile 形状不匹配时,这里就是 occupancy 损失的源头。"
L.append(f'<text x="{TAB_X}" y="{h-16}" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc(cap)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-warps-per-tile.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w}x{h}")
