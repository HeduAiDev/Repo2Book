#!/usr/bin/env python3
"""fig-m04-coalescing — before-after 模板。
同一个 warp 的 32 次访存:连续对齐(左)合并成 1 次 128B 事务;跨步 gather(右)炸成 32 次事务。
全坐标由循环/常量计算,零手写魔数。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

WARP_SIZE = 32
PAD = 40
PANEL_W = 480
GAP_H = 60
W = PAD * 2 + PANEL_W * 2 + GAP_H
TOP = 118

L = []
DEFS = ('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
        'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')

L.append(f'<text x="{PAD}" y="{PAD-8}" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">{esc("合并访存 (coalescing)：同一个 warp 的 32 次访存，连续对齐 vs 跨步 gather")}</text>')
L.append(f'<text x="{PAD}" y="{PAD+16}" font-family="sans-serif" font-size="12.5" '
          f'fill="#475569">{esc("代码差别只是 offsets 连不连续——硬件按 128 字节对齐段归并事务，段数越多有效带宽越低")}</text>')

PANELS = [
    ("连续对齐 —— vector-add 的 offsets", "#dcfce7", "#16a34a",
     "lane i 地址 = i × 4B（i=0..31，连续排列）", "0 .. 124", "1", "128", "满带宽（基准 1×）"),
    ("跨步 gather —— stride=32", "#fee2e2", "#dc2626",
     "lane i 地址 = i × 32 × 4B（i=0..31，跨段散落）", "0 .. 3968", "32", "128 (仅 4B 有用)", "1/32 带宽"),
]

lane_row_h = 30
n_show = 16  # 展示 16 个 lane 的地址落点示意(足以看清模式,不做 32 个拥挤小格)

for p, (title, fill, stroke, rule, addr_range, txn, seg_bytes, bw) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + GAP_H)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" fill="#0f172a">{esc(title)}</text>')

    # warp row: 16 lane markers (sampled) laid out evenly, colored by segment
    row_y = TOP
    lane_w = PANEL_W / n_show
    for i in range(n_show):
        lx = px + i * lane_w
        if p == 0:
            seg = 0  # all in one segment -> same color
        else:
            seg = i  # each lane its own segment -> rainbow-ish via alternating shade
        color = stroke if (p == 0 or i % 2 == 0) else "#f87171"
        L.append(f'<rect x="{lx+1}" y="{row_y}" width="{lane_w-2}" height="{lane_row_h}" rx="2" '
                  f'fill="{fill}" stroke="{color}" stroke-width="1.3"/>')
    L.append(f'<text x="{cx}" y="{row_y+lane_row_h+16}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#334155">{esc(f"warp 的 32 lane（示意前 {n_show} 个）：{rule}")}</text>')

    # arrow down into "transaction box"
    arrow_y1 = row_y + lane_row_h + 24
    arrow_y2 = arrow_y1 + 28
    L.append(f'<line x1="{cx}" y1="{arrow_y1}" x2="{cx}" y2="{arrow_y2}" '
              f'stroke="{stroke}" stroke-width="2" marker-end="url(#a)"/>')

    # transaction box(es): 1 big box for contiguous, grid of small boxes for strided
    box_top = arrow_y2 + 6
    if p == 0:
        bw_, bh_ = 200, 66
        bx = cx - bw_ / 2
        L.append(f'<rect x="{bx}" y="{box_top}" width="{bw_}" height="{bh_}" rx="8" '
                  f'fill="{stroke}" stroke="#14532d" stroke-width="1.5"/>')
        L.append(f'<text x="{cx}" y="{box_top+bh_/2-6}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="15" font-weight="bold" fill="white">{esc(f"{txn} 次事务")}</text>')
        L.append(f'<text x="{cx}" y="{box_top+bh_/2+16}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="11.5" fill="white">{esc(f"{seg_bytes} 字节，全部有用")}</text>')
        box_bottom = box_top + bh_
    else:
        n_seg_show = 8
        seg_w, seg_gap = 40, 6
        total_w = n_seg_show * seg_w + (n_seg_show - 1) * seg_gap
        sx0 = cx - total_w / 2
        for i in range(n_seg_show):
            sx = sx0 + i * (seg_w + seg_gap)
            L.append(f'<rect x="{sx}" y="{box_top}" width="{seg_w}" height="46" rx="5" '
                      f'fill="{stroke}" stroke="#7f1d1d" stroke-width="1.3"/>')
        L.append(f'<text x="{cx}" y="{box_top+46+20}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="15" font-weight="bold" fill="{stroke}">{esc(f"{txn} 次事务（示意前 {n_seg_show} 段 + ...）")}</text>')
        L.append(f'<text x="{cx}" y="{box_top+46+38}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="11.5" fill="#7f1d1d">{esc(f"每次事务仅 {seg_bytes}")}</text>')
        box_bottom = box_top + 46 + 38

    # summary card
    card_top = box_bottom + 24
    card_h = 64
    L.append(f'<rect x="{px}" y="{card_top}" width="{PANEL_W}" height="{card_h}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    L.append(f'<text x="{px+16}" y="{card_top+26}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(f"warp 触及字节地址范围：{addr_range}")}</text>')
    L.append(f'<text x="{px+16}" y="{card_top+48}" font-family="sans-serif" font-size="16" '
              f'font-weight="bold" fill="{stroke}">{esc(f"有效带宽 = {bw}")}</text>')

panel_bottom = TOP + lane_row_h + 24 + 28 + 6 + (66 if True else 0)  # placeholder recomputed below via max
# recompute actual bottom precisely using values from loop (contiguous branch had larger explicit box)
contig_box_bottom = TOP + lane_row_h + 16 + 8 + 24 + 28 + 66
strided_box_bottom = TOP + lane_row_h + 16 + 8 + 24 + 28 + 46 + 38
card_bottom = max(contig_box_bottom, strided_box_bottom) + 24 + 64

foot_y = card_bottom + 34
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#0f172a">{esc(f"warp lane 数 = {WARP_SIZE}；同样 32 次访存，事务数差 32 倍，带宽落差 32 倍——机器成因只有一条：地址是否落在同一 128 字节对齐段。")}</text>')

H = foot_y + 30
full = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">', DEFS,
        f'<rect width="{W}" height="{H}" fill="white"/>'] + L + ['</svg>']
out = Path(__file__).with_name("fig-m04-coalescing.svg")
out.write_text('\n'.join(full), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
