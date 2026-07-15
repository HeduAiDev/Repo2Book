#!/usr/bin/env python3
"""fig-m03-latency-pyramid — layout 模板。
内存延迟金字塔:寄存器(~1) -> 共享内存(~20-30) -> L2(~200) -> 全局显存(~400-800 cycle)。
越靠顶端越快越小、越往底端越慢越大,每下一层大致慢一个数量级。
全坐标由循环/常量计算,零手写魔数。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TIERS = [
    ("寄存器 register", "~1 cycle", "1", 0.30, "#93c5fd", "#1e3a5f"),
    ("共享内存 shared memory (SMEM)", "~20-30 cycle", "20-30", 0.52, "#60a5fa", "#1e3a5f"),
    ("L2 缓存", "~200 cycle", "200", 0.76, "#3b82f6", "white"),
    ("全局显存 global (HBM/DRAM)", "~400-800 cycle", "400-800", 1.00, "#1d4ed8", "white"),
]

PAD = 40
W = 860
TOP = 108
ROW_H = 76
GAP = 10

L = []
DEFS = ('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
        'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')

L.append(f'<text x="{PAD}" y="{PAD-8}" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">{esc("内存延迟金字塔——越靠上越快越小，越往下越慢越大")}</text>')
L.append(f'<text x="{PAD}" y="{PAD+16}" font-family="sans-serif" font-size="12.5" '
          f'fill="#475569">{esc("Ampere 级数量级，绝对 cycle 数随架构变化；写快 kernel = 数据尽量留在上层、下层访问尽量少且合并")}</text>')

pyr_center = PAD + (W - 2 * PAD) / 2
max_w = W - 2 * PAD - 260  # leave room for right-side cycle labels

for i, (name, lat, num, frac, fill, tcolor) in enumerate(TIERS):
    y = TOP + i * (ROW_H + GAP)
    tw = max_w * frac
    tx = pyr_center - tw / 2
    L.append(f'<rect x="{tx}" y="{y}" width="{tw}" height="{ROW_H}" rx="7" '
              f'fill="{fill}" stroke="#1e3a5f" stroke-width="1.4"/>')
    L.append(f'<text x="{pyr_center}" y="{y+ROW_H/2-6}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14.5" font-weight="bold" '
              f'fill="{tcolor}">{esc(name)}</text>')
    L.append(f'<text x="{pyr_center}" y="{y+ROW_H/2+18}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13.5" fill="{tcolor}">{esc(lat)}</text>')
    # right-side annotation: "~10x" step arrow to next tier
    if i < len(TIERS) - 1:
        ax = pyr_center + tw / 2 + 46
        ay1 = y + ROW_H
        ay2 = ay1 + GAP
        L.append(f'<line x1="{ax}" y1="{ay1-4}" x2="{ax}" y2="{ay2+4}" '
                  'stroke="#d97706" stroke-width="1.6" marker-end="url(#a)"/>')

# 逐层真实倍数(按本层区间的 min/max 除以上一层区间的 max/min 算出的下上界):
# register(1,1) -> smem(20,30): 20/1..30/1 = 20-30x
# smem(20,30) -> L2(200,200): 200/30..200/20 = 6.7-10x (取 7-10 便于对读)
# L2(200,200) -> global(400,800): 400/200..800/200 = 2-4x
STEP_LABELS = ["~20-30× 慢", "~7-10× 慢", "~2-4× 慢"]
step_label_x = pyr_center + max_w / 2 + 70
for i in range(len(TIERS) - 1):
    y = TOP + i * (ROW_H + GAP) + ROW_H + GAP / 2
    L.append(f'<text x="{step_label_x}" y="{y+4}" font-family="sans-serif" font-size="11" '
              f'fill="#92400e">{esc(STEP_LABELS[i])}</text>')

pyr_bottom = TOP + len(TIERS) * (ROW_H + GAP) - GAP
foot_y = pyr_bottom + 40
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#0f172a">{esc("寄存器 ↔ 全局显存量级差 ≥ 100×")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="12.5" '
          f'fill="#334155">{esc("vector-add 的 tl.load / tl.store 摸的正是最底那层——既要合并成尽量少事务，又要靠 occupancy 藏住这几百 cycle 的延迟。")}</text>')

H = foot_y + 46
full = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">', DEFS,
        f'<rect width="{W}" height="{H}" fill="white"/>'] + L + ['</svg>']
out = Path(__file__).with_name("fig-m03-latency-pyramid.svg")
out.write_text('\n'.join(full), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
