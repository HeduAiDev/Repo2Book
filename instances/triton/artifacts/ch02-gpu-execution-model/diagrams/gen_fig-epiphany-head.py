#!/usr/bin/env python3
"""fig-epiphany-head — 本章顿悟头图(layout 模板)。
左:执行层次(grid->block->warp->lane,聚焦一个 warp 的 32 lane)。
右:内存延迟金字塔(register ~1 -> shared -> L2 -> global ~400-800 cycle)。
底部:同一个 warp 的 32 次访存,连续对齐 = 1 次事务,跨步 gather = 32 次事务——
这就是 occupancy/coalescing/spill 三把尺共同的落差起点。
全坐标由循环/常量计算,零手写魔数。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

W = 1220
PAD = 40
TOP = 100

DEFS = ('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
        'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
        '<marker id="ah" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" '
        'markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>')

L = []  # body elements only; svg/rect wrapper written after H is known

L.append(f'<text x="{PAD}" y="{PAD-8}" font-family="sans-serif" font-size="19" '
          f'font-weight="bold" fill="#0f172a">{esc("同一个 warp 的 32 次访存,落在哪一层、落得连不连续,决定快慢")}</text>')
L.append(f'<text x="{PAD}" y="{PAD+18}" font-family="sans-serif" font-size="13" '
          f'fill="#475569">{esc("左:执行层次定位到哪个 warp/lane。右:内存延迟金字塔定位到哪一层。中间一支箭把两者接起来。")}</text>')

# =========================================================================
# 左面板:执行层次(压缩版,聚焦到 warp -> 32 lane)
# =========================================================================
panel_top = TOP
left_x = PAD
left_w = 430

L.append(f'<text x="{left_x}" y="{panel_top-10}" font-family="sans-serif" font-size="14.5" '
          f'font-weight="bold" fill="#1e40af">{esc("① 执行层次:grid → block → warp → lane")}</text>')

row_h = 46
gap = 14
grid_y = panel_top + 32
block_y = grid_y + row_h + gap
warp_y = block_y + row_h + gap
lane_y = warp_y + row_h + gap + 6

# grid row: 4 block boxes
bw, bgap = 88, 10
for i in range(4):
    bx = left_x + i * (bw + bgap)
    fill = "#bfdbfe" if i == 0 else "#dbeafe"
    L.append(f'<rect x="{bx}" y="{grid_y}" width="{bw}" height="{row_h-8}" rx="5" '
              f'fill="{fill}" stroke="#2563eb" stroke-width="1.3"/>')
    L.append(f'<text x="{bx+bw/2}" y="{grid_y+(row_h-8)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="#1e3a5f">{esc(f"block{i}")}</text>')
L.append(f'<text x="{left_x+left_w}" y="{grid_y-6}" text-anchor="end" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">{esc("grid（发射）")}</text>')

# arrow grid -> block
zx = left_x + bw / 2
L.append(f'<line x1="{zx}" y1="{grid_y+row_h-8}" x2="{zx}" y2="{block_y}" '
          'stroke="#2563eb" stroke-width="1.4" stroke-dasharray="4,3" marker-end="url(#a)"/>')

# block row: one wide box
block_w = left_w
L.append(f'<rect x="{left_x}" y="{block_y}" width="{block_w}" height="{row_h-8}" rx="5" '
          'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.8"/>')
L.append(f'<text x="{left_x+block_w/2}" y="{block_y+(row_h-8)/2+4}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="white">'
          f'{esc("block 0 — BLOCK_SIZE=1024 个逻辑 lane")}</text>')
L.append(f'<text x="{left_x+left_w}" y="{block_y-6}" text-anchor="end" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">{esc("block（CTA）—— tl.program_id 停在这层")}</text>')

L.append(f'<line x1="{left_x+block_w/2}" y1="{block_y+row_h-8}" x2="{left_x+block_w/2}" y2="{warp_y}" '
          'stroke="#64748b" stroke-width="1.4" stroke-dasharray="4,3" marker-end="url(#a)"/>')

# warp row: several warp boxes, highlight warp0
ww, wgap = 60, 8
n_warp_show = 6
for i in range(n_warp_show):
    wx = left_x + i * (ww + wgap)
    hot = (i == 0)
    if i == n_warp_show - 2:
        label, fill, stroke = "...", "#f1f5f9", "#94a3b8"
    elif i == n_warp_show - 1:
        label, fill, stroke = "warp31", "#c7d2fe", "#6366f1"
    else:
        label = f"warp{i}"
        fill = "#fde68a" if hot else "#c7d2fe"
        stroke = "#d97706" if hot else "#6366f1"
    L.append(f'<rect x="{wx}" y="{warp_y}" width="{ww}" height="{row_h-8}" rx="5" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{2.2 if hot else 1.3}"/>')
    L.append(f'<text x="{wx+ww/2}" y="{warp_y+(row_h-8)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#312e81">{esc(label)}</text>')
L.append(f'<text x="{left_x+left_w}" y="{warp_y-6}" text-anchor="end" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">{esc("warp（32 lane 锁步，编译器+硬件划分）")}</text>')

w0x = left_x + ww / 2
L.append(f'<line x1="{w0x}" y1="{warp_y+row_h-8}" x2="{w0x}" y2="{lane_y}" '
          'stroke="#d97706" stroke-width="1.8" marker-end="url(#ah)"/>')

# lane row: 32 lanes of warp0, all shown compactly
n_lanes = 32
lane_box = (left_w - (n_lanes - 1) * 2) / n_lanes
for i in range(n_lanes):
    lx = left_x + i * (lane_box + 2)
    L.append(f'<rect x="{lx}" y="{lane_y}" width="{lane_box}" height="30" rx="2" '
              f'fill="#fef3c7" stroke="#d97706" stroke-width="1"/>')
L.append(f'<text x="{left_x+left_w}" y="{lane_y-6}" text-anchor="end" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">{esc("lane（32 个，同一 warp 内锁步执行同一条指令）")}</text>')
L.append(f'<text x="{left_x+left_w/2}" y="{lane_y+50}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#92400e">{esc("这个 warp 的 32 个 lane —— 它们摸的地址连不连续，决定下面这次访问要几次事务")}</text>')

left_panel_bottom = lane_y + 74

# =========================================================================
# 右面板:内存延迟金字塔
# =========================================================================
right_x = 700
right_w = 480
pyr_top = panel_top - 4
pyr_row_h = 62
pyr_gap = 8

L.append(f'<text x="{right_x}" y="{pyr_top-14}" font-family="sans-serif" font-size="14.5" '
          f'font-weight="bold" fill="#1e40af">{esc("② 内存延迟金字塔（Ampere 级数量级，架构相关）")}</text>')

TIERS = [
    ("寄存器 register", "~1 cycle", 0.42, "#93c5fd", "#1e3a5f"),
    ("共享内存 shared", "~20-30 cycle", 0.60, "#60a5fa", "#1e3a5f"),
    ("L2 缓存", "~200 cycle", 0.80, "#3b82f6", "white"),
    ("全局显存 global (HBM)", "~400-800 cycle", 1.00, "#1d4ed8", "white"),
]
pyr_center = right_x + right_w / 2
for i, (name, lat, frac, fill, tcolor) in enumerate(TIERS):
    y = pyr_top + i * (pyr_row_h + pyr_gap)
    tw = right_w * frac
    tx = pyr_center - tw / 2
    L.append(f'<rect x="{tx}" y="{y}" width="{tw}" height="{pyr_row_h}" rx="6" '
              f'fill="{fill}" stroke="#1e3a5f" stroke-width="1.3"/>')
    L.append(f'<text x="{pyr_center}" y="{y+pyr_row_h/2-4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{tcolor}">{esc(name)}</text>')
    L.append(f'<text x="{pyr_center}" y="{y+pyr_row_h/2+16}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="{tcolor}">{esc(lat)}</text>')

pyr_bottom_y = pyr_top + len(TIERS) * (pyr_row_h + pyr_gap) - pyr_gap
L.append(f'<text x="{right_x}" y="{pyr_bottom_y+26}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("寄存器 ↔ 全局显存量级差 ≥ 100×——每往下一层，大致慢一个数量级")}</text>')

right_panel_bottom = pyr_bottom_y + 26

# =========================================================================
# 中间连接箭:从 lane 组指向金字塔底层(load/store 摸的正是最慢那层)
# =========================================================================
bridge_y1 = lane_y + 15
bridge_x1 = left_x + left_w + 10
bridge_x2 = right_x - 10
bridge_y2 = pyr_top + 3 * (pyr_row_h + pyr_gap) + pyr_row_h / 2
L.append(f'<path d="M {bridge_x1} {bridge_y1} C {(bridge_x1+bridge_x2)/2} {bridge_y1}, '
         f'{(bridge_x1+bridge_x2)/2} {bridge_y2}, {bridge_x2} {bridge_y2}" '
         'fill="none" stroke="#d97706" stroke-width="2.4" marker-end="url(#ah)"/>')
mid_x = (bridge_x1 + bridge_x2) / 2
L.append(f'<text x="{mid_x}" y="{(bridge_y1+bridge_y2)/2 - 14}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#92400e">'
          f'{esc("tl.load / tl.store")}</text>')
L.append(f'<text x="{mid_x}" y="{(bridge_y1+bridge_y2)/2 + 4}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#92400e">'
          f'{esc("摸的正是最慢这层")}</text>')

# =========================================================================
# 底部:合并访存对比条(连续 vs 跨步)
# =========================================================================
bottom_top = max(left_panel_bottom, right_panel_bottom) + 46
L.append(f'<text x="{PAD}" y="{bottom_top-14}" font-family="sans-serif" font-size="14.5" '
          f'font-weight="bold" fill="#1e40af">{esc("③ 落地差别:32 个地址连续 vs 分散——事务数差 32 倍")}</text>')

CASES = [
    ("连续对齐（vector-add 的 offsets）", "32 lane × 4B = 128B，同落 1 个对齐段", "1 次事务", "满带宽", "#dcfce7", "#16a34a"),
    ("跨步 gather（stride=32）", "32 个地址散落 32 个不同对齐段", "32 次事务", "带宽 1/32", "#fee2e2", "#dc2626"),
]
case_w = (W - 2 * PAD - 40) / 2
case_h = 118
for i, (title, detail, txn, bw_label, fill, stroke) in enumerate(CASES):
    cx0 = PAD + i * (case_w + 40)
    L.append(f'<rect x="{cx0}" y="{bottom_top}" width="{case_w}" height="{case_h}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    L.append(f'<text x="{cx0+18}" y="{bottom_top+26}" font-family="sans-serif" font-size="13" '
              f'font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    L.append(f'<text x="{cx0+18}" y="{bottom_top+48}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(detail)}</text>')
    L.append(f'<text x="{cx0+18}" y="{bottom_top+82}" font-family="sans-serif" font-size="20" '
              f'font-weight="bold" fill="{stroke}">{esc(txn)}</text>')
    L.append(f'<text x="{cx0+case_w-18}" y="{bottom_top+82}" text-anchor="end" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" '
              f'fill="{stroke}">{esc(bw_label)}</text>')

H = bottom_top + case_h + 30

full = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">', DEFS,
        f'<rect width="{W}" height="{H}" fill="white"/>'] + L + ['</svg>']
svg_text = '\n'.join(full)
out = Path(__file__).with_name("fig-epiphany-head.svg")
out.write_text(svg_text, encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
