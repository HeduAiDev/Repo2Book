#!/usr/bin/env python3
"""fig-m02-spmd-tiling — tiling 模板(改为一维向量的连续切分,非多对多)。
一条长度 256 的向量被切成 4 个连续 tile,program_id=0..3 各认领一块并列箭头指认。
全坐标由循环/常量计算,零手写魔数。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

N = 256
BLOCK = 64
GRID = N // BLOCK  # 4
COLORS = ["#3b82f6", "#7c3aed", "#059669", "#d97706"]

PAD = 40
W = 1080
TOP = 96

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 1">']
L = []
DEFS = ('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
        'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')

L.append(f'<text x="{PAD}" y="{PAD-8}" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">{esc(f"SPMD tile 切分:向量长度 N={N}、BLOCK_SIZE={BLOCK} → {GRID} 个 program")}</text>')
L.append(f'<text x="{PAD}" y="{PAD+16}" font-family="sans-serif" font-size="12.5" '
          f'fill="#475569">{esc("一个 program 处理一个 tile，不是一个线程处理一个元素——四块首尾相接、互不重叠、恰好铺满")}</text>')

# vector strip: 256 cells grouped into 4 colored tiles
strip_y = TOP
strip_w = W - 2 * PAD
cell_w = strip_w / N
strip_h = 46
for pid in range(GRID):
    x0 = PAD + pid * BLOCK * cell_w
    tile_w = BLOCK * cell_w
    L.append(f'<rect x="{x0}" y="{strip_y}" width="{tile_w}" height="{strip_h}" '
              f'fill="{COLORS[pid]}" stroke="white" stroke-width="1.5" opacity="0.85"/>')
L.append(f'<rect x="{PAD}" y="{strip_y}" width="{strip_w}" height="{strip_h}" fill="none" stroke="#1e293b" stroke-width="2"/>')
# tick marks at tile boundaries with offset labels
for pid in range(GRID + 1):
    x = PAD + pid * BLOCK * cell_w
    L.append(f'<line x1="{x}" y1="{strip_y}" x2="{x}" y2="{strip_y+strip_h}" stroke="#1e293b" stroke-width="1.5"/>')
    lbl = str(pid * BLOCK)
    anchor = "middle"
    L.append(f'<text x="{x}" y="{strip_y+strip_h+18}" text-anchor="{anchor}" font-family="sans-serif" '
              f'font-size="11.5" fill="#334155">{esc(lbl)}</text>')
L.append(f'<text x="{PAD}" y="{strip_y-8}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc(f"完整向量 [0, {N})，元素下标沿横轴递增")}</text>')

# per-program cards below, each connected up to its tile with an arrow
card_top = strip_y + strip_h + 44
card_w = strip_w / GRID - 16
card_h = 108
ROWS = [
    (0, 0, "[0:64)", "第 0 个 program 包下前 64 个元素"),
    (1, 64, "[64:128)", "工号一变，起点整体平移 BLOCK"),
    (2, 128, "[128:192)", "四份 tile 首尾相接、互不重叠"),
    (3, 192, "[192:256)", "并集恰好铺满 [0:256)"),
]
for pid, block_start, offs, note in ROWS:
    cx0 = PAD + pid * (strip_w / GRID) + 8
    tile_center = PAD + (pid + 0.5) * BLOCK * cell_w
    card_center = cx0 + card_w / 2
    # connecting arrow from tile down to its card
    L.append(f'<line x1="{tile_center}" y1="{strip_y+strip_h}" x2="{card_center}" y2="{card_top}" '
              f'stroke="{COLORS[pid]}" stroke-width="1.6" marker-end="url(#a)"/>')
    L.append(f'<rect x="{cx0}" y="{card_top}" width="{card_w}" height="{card_h}" rx="8" '
              f'fill="white" stroke="{COLORS[pid]}" stroke-width="2"/>')
    L.append(f'<rect x="{cx0}" y="{card_top}" width="{card_w}" height="28" rx="8" '
              f'fill="{COLORS[pid]}"/>')
    L.append(f'<rect x="{cx0}" y="{card_top+14}" width="{card_w}" height="14" fill="{COLORS[pid]}"/>')
    L.append(f'<text x="{card_center}" y="{card_top+19}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="white">{esc(f"program_id={pid}")}</text>')
    L.append(f'<text x="{card_center}" y="{card_top+48}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" fill="#0f172a">{esc(f"block_start = {pid}×{BLOCK} = {block_start}")}</text>')
    L.append(f'<text x="{card_center}" y="{card_top+68}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="#0f172a">{esc(f"offsets = {offs}")}</text>')
    L.append(f'<text x="{card_center}" y="{card_top+92}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#475569">{esc(note)}</text>')

foot_y = card_top + card_h + 34
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="#334155">{esc(f"程序员写的是 tile 级代码(block_start=pid×BLOCK, offsets=block_start+arange(0,BLOCK));编译器负责把 tile 铺到 warp/lane。")}</text>')

H = foot_y + 24
full = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">', DEFS,
        f'<rect width="{W}" height="{H}" fill="white"/>'] + L + ['</svg>']
out = Path(__file__).with_name("fig-m02-spmd-tiling.svg")
out.write_text('\n'.join(full), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
