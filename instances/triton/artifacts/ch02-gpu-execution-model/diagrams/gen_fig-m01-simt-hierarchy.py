#!/usr/bin/env python3
"""fig-m01-simt-hierarchy — layout 模板。
四层 SIMT 执行层次自上而下:grid -> block(CTA) -> warp(32 lane) -> lane。
程序员在 Triton 只显式写到 block/tile 一层(program_id);再往下由编译器+硬件划分。
全坐标由循环/常量计算,零手写魔数。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

WARP_SIZE = 32
BLOCK_THREADS = 1024
NUM_WARPS = BLOCK_THREADS // WARP_SIZE  # 32

PAD = 40
W = 1180

# ---- band geometry -----------------------------------------------------
TOP = 96
BAND_LABEL_W = 190
BAND_GAP = 18

grid_h = 74
block_h = 60
warp_h = 60
lane_h = 60

grid_y = TOP
block_y = grid_y + grid_h + BAND_GAP
warp_y = block_y + block_h + BAND_GAP
lane_y = warp_y + warp_h + BAND_GAP

content_x = PAD + BAND_LABEL_W
content_w = W - content_x - PAD

H = lane_y + lane_h + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{PAD}" y="{PAD-8}" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">{esc("SIMT 执行层次:grid → block(CTA) → warp(32 lane) → lane")}</text>')
L.append(f'<text x="{PAD}" y="{PAD+16}" font-family="sans-serif" font-size="12.5" '
          f'fill="#475569">{esc("Triton 显式写到 block/tile 这一层(tl.program_id);再往下由编译器与硬件替你划分")}</text>')

# 背景分区:程序员可见区(浅蓝) vs 编译器/硬件划分区(浅灰)
visible_bg_y0 = grid_y - 14
visible_bg_h = (block_y + block_h) - grid_y + 24
hidden_bg_y0 = warp_y - 14
hidden_bg_h = (lane_y + lane_h) - warp_y + 24
L.append(f'<rect x="{PAD-6}" y="{visible_bg_y0}" width="{W-2*(PAD-6)}" height="{visible_bg_h}" '
          'rx="10" fill="#eff6ff" stroke="none"/>')
L.append(f'<rect x="{PAD-6}" y="{hidden_bg_y0}" width="{W-2*(PAD-6)}" height="{hidden_bg_h}" '
          'rx="10" fill="#f1f5f9" stroke="none"/>')

def band_label(y, h, text, sub):
    cy = y + h / 2
    L.append(f'<text x="{PAD}" y="{cy-4}" font-family="sans-serif" font-size="14" '
              f'font-weight="bold" fill="#0f172a">{esc(text)}</text>')
    L.append(f'<text x="{PAD}" y="{cy+14}" font-family="sans-serif" font-size="11" '
              f'fill="#64748b">{esc(sub)}</text>')

# ---- Band 1: grid --------------------------------------------------------
band_label(grid_y, grid_h, "grid（发射）", "launch grid")
n_show = 5
box_w = 76
gap = 14
total_w = n_show * box_w + (n_show - 1) * gap
bx0 = content_x
for i in range(n_show):
    bx = bx0 + i * (box_w + gap)
    label = f"block {i}" if i < n_show - 1 else "..."
    fill = "#bfdbfe" if i < n_show - 1 else "#f1f5f9"
    stroke = "#2563eb" if i < n_show - 1 else "#94a3b8"
    L.append(f'<rect x="{bx}" y="{grid_y+8}" width="{box_w}" height="{grid_h-16}" rx="6" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{bx+box_w/2}" y="{grid_y+grid_h/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="#1e3a5f">{esc(label)}</text>')
L.append(f'<text x="{bx0+total_w+18}" y="{grid_y+grid_h/2+5}" font-family="sans-serif" '
          f'font-size="12" fill="#475569">{esc("program_id 逐个区分身份")}</text>')

# arrow grid -> block (zoom into block 0)
zx = bx0 + box_w / 2
L.append(f'<line x1="{zx}" y1="{grid_y+grid_h-8}" x2="{zx}" y2="{block_y+8}" '
          'stroke="#2563eb" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#a)"/>')

# ---- Band 2: block (CTA) -------------------------------------------------
band_label(block_y, block_h, "block（CTA）", f"tl.program_id 停在这层")
block_w = 560
L.append(f'<rect x="{content_x}" y="{block_y+6}" width="{block_w}" height="{block_h-12}" rx="6" '
          'fill="#3b82f6" stroke="#1e3a5f" stroke-width="2"/>')
L.append(f'<text x="{content_x+block_w/2}" y="{block_y+block_h/2+5}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13" font-weight="bold" fill="white">'
          f'{esc(f"block 0 — BLOCK_SIZE={BLOCK_THREADS} 个逻辑线程")}</text>')

# divider annotation between visible/hidden
div_y = (block_y + block_h + warp_y) / 2
L.append(f'<line x1="{PAD-6}" y1="{div_y}" x2="{W-(PAD-6)}" y2="{div_y}" '
          'stroke="#94a3b8" stroke-width="1" stroke-dasharray="6,4"/>')
L.append(f'<text x="{W-PAD-4}" y="{div_y-6}" text-anchor="end" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="#1d4ed8">{esc("↑ 程序员显式写到这里")}</text>')
L.append(f'<text x="{W-PAD-4}" y="{div_y+16}" text-anchor="end" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="#64748b">{esc("↓ 编译器 + 硬件划分(不可见)")}</text>')

# arrow block -> warp strip (zoom in)
L.append(f'<line x1="{content_x+block_w/2}" y1="{block_y+block_h-6}" '
          f'x2="{content_x+block_w/2}" y2="{warp_y+8}" '
          'stroke="#64748b" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#a)"/>')

# ---- Band 3: warp strip ---------------------------------------------------
band_label(warp_y, warp_h, "warp（32 lane 锁步）", f"{NUM_WARPS} 个 warp / block")
n_warp_show = 6
warp_box_w = 76
warp_gap = 6
wx0 = content_x
for i in range(n_warp_show):
    wx = wx0 + i * (warp_box_w + warp_gap)
    if i < n_warp_show - 2:
        label = f"warp{i}"
        fill = "#c7d2fe"
        stroke = "#6366f1"
    elif i == n_warp_show - 2:
        label = "..."
        fill = "#f1f5f9"
        stroke = "#94a3b8"
    else:
        label = f"warp{NUM_WARPS-1}"
        fill = "#c7d2fe"
        stroke = "#6366f1"
    L.append(f'<rect x="{wx}" y="{warp_y+8}" width="{warp_box_w}" height="{warp_h-16}" rx="5" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{wx+warp_box_w/2}" y="{warp_y+warp_h/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" fill="#312e81">{esc(label)}</text>')
warp_strip_end = wx0 + n_warp_show * (warp_box_w + warp_gap) - warp_gap
L.append(f'<text x="{warp_strip_end+16}" y="{warp_y+warp_h/2+5}" font-family="sans-serif" '
          f'font-size="12" fill="#475569">{esc(f"BLOCK_SIZE={BLOCK_THREADS} ÷ {WARP_SIZE} = {NUM_WARPS} 个 warp")}</text>')

# arrows warp0 / warp1 -> lane band (zoom into first two warps)
w0x = wx0 + warp_box_w / 2
w1x = wx0 + (warp_box_w + warp_gap) + warp_box_w / 2
for wx_c in (w0x, w1x):
    L.append(f'<line x1="{wx_c}" y1="{warp_y+warp_h-8}" x2="{wx_c}" y2="{lane_y+8}" '
              'stroke="#6366f1" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#a)"/>')

# ---- Band 4: lane cells for warp0 + warp1 ---------------------------------
band_label(lane_y, lane_h, "lane（SIMT 执行单元）", "32 lane 同一 warp 内锁步")
lane_box = 24
lane_gap = 3
lx0 = content_x
groups = [("warp0", 0, 31, "#93c5fd"), ("warp1", 32, 63, "#a5b4fc")]
gx = lx0
for gname, lo, hi, color in groups:
    n_cells_show = 8
    L.append(f'<text x="{gx}" y="{lane_y-4}" font-family="sans-serif" font-size="11" '
              f'fill="#334155">{esc(f"{gname}: lane {lo}..{hi}")}</text>')
    for k in range(n_cells_show):
        cx = gx + k * (lane_box + lane_gap)
        if k < n_cells_show - 1:
            lbl = str(lo + k)
        else:
            lbl = "..."
        L.append(f'<rect x="{cx}" y="{lane_y+8}" width="{lane_box}" height="{lane_h-24}" rx="3" '
                  f'fill="{color}" stroke="#1e3a5f" stroke-width="1"/>')
        fs = 9 if lbl != "..." else 10
        L.append(f'<text x="{cx+lane_box/2}" y="{lane_y+8+(lane_h-24)/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs}" fill="#1e3a5f">{esc(lbl)}</text>')
    gx += n_cells_show * (lane_box + lane_gap) + 36

# caption
cap_y = lane_y + lane_h + 44
cap_lines = [
    "一个 BLOCK_SIZE=1024 的 block 被硬件切成 32 个 warp,同 warp 内 32 个 lane 锁步执行同一条指令;",
    "tl.program_id 只给到 block 层(降级为 PTX %ctaid),warp/lane 划分完全不由 kernel 代码指定。",
    "warp 内 lane 范围: warp0 = 0..31, warp1 = 32..63 —— 硬件按 32 一组顺序切分,不跳号不重叠。",
]
for i, line in enumerate(cap_lines):
    L.append(f'<text x="{PAD}" y="{cap_y + i * 20}" font-family="sans-serif" font-size="12.5" '
              f'fill="#334155">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m01-simt-hierarchy.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
