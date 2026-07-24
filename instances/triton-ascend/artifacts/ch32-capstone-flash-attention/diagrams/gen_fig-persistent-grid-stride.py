#!/usr/bin/env python3
"""tiling 模板改写为柱状分布图:20 个物理核(program)各领到的逻辑块数,
呈现 grid-stride range(pid, NUM_BLOCKS, 20) 造成的近乎均衡负载(16 核×13 + 4 核×12 = 256)。
全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

NUM_CORES = 20
NUM_BLOCKS = 256
STRIDE = 20
counts = [len(range(pid, NUM_BLOCKS, STRIDE)) for pid in range(NUM_CORES)]  # 13 x16 + 12 x4
assert counts[0] == 13 and counts[19] == 12 and sum(counts) == NUM_BLOCKS

TITLE = "持久化网格 — num_cores=20 个 program,grid-stride 领块几乎均衡"
SUBTITLE = "示例 Z,H,N_CTX,BM=4,32,64,32(test_06_fused_attention.py:L325);NUM_BLOCKS_M=2,NUM_BLOCKS=NUM_BLOCKS_M·Z·H=256"

PAD, TOP = 50, 120
BAR_W, BAR_GAP = 46, 12
BAR_MAX_H = 220
LABEL_H = 30
w = PAD * 2 + NUM_CORES * (BAR_W + BAR_GAP)
h = TOP + BAR_MAX_H + LABEL_H + 190

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-14}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+6}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

base_y = TOP + BAR_MAX_H
max_count = max(counts)
for i, c in enumerate(counts):
    x = PAD + i * (BAR_W + BAR_GAP)
    bh = c / max_count * (BAR_MAX_H - 30)
    y = base_y - bh
    hi = i in (0, NUM_CORES - 1)
    fill = "#2563eb" if i == 0 else ("#f59e0b" if i == NUM_CORES - 1 else "#93c5fd")
    stroke = "#1e3a5f" if hi else "#60a5fa"
    L.append(f'<rect x="{x}" y="{y}" width="{BAR_W}" height="{bh}" rx="3" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if hi else 1}"/>')
    L.append(f'<text x="{x+BAR_W/2}" y="{y-6}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" font-weight="bold" fill="#1e293b">{c}</text>')
    L.append(f'<text x="{x+BAR_W/2}" y="{base_y+16}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10" fill="#64748b">核{i}</text>')

L.append(f'<line x1="{PAD}" y1="{base_y}" x2="{w-PAD}" y2="{base_y}" stroke="#94a3b8" stroke-width="1.5"/>')

# 核0 / 核19 详细领块索引标注
ann_y = base_y + 44
L.append(f'<rect x="{PAD}" y="{ann_y}" width="{w-2*PAD}" height="34" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.2"/>')
L.append(f'<text x="{PAD+14}" y="{ann_y+22}" font-family="sans-serif" font-size="12" '
          f'fill="#1e3a8a">核0(pid=0)领块索引: 0,20,40,…,240 — 共 13 块(range(0,256,20))</text>')
ann_y2 = ann_y + 42
L.append(f'<rect x="{PAD}" y="{ann_y2}" width="{w-2*PAD}" height="34" rx="6" fill="#fffbeb" stroke="#b45309" stroke-width="1.2"/>')
L.append(f'<text x="{PAD+14}" y="{ann_y2+22}" font-family="sans-serif" font-size="12" '
          f'fill="#92400e">核19(pid=19)领块索引: 19,39,59,…,239 — 共 12 块(range(19,256,20))</text>')

foot_y = ann_y2 + 34 + 34
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#374151">16 核 × 13 块 + 4 核 × 12 块 = 256 块 = NUM_BLOCKS —— 逻辑核数(256)远超物理核数(20)时,'
          f'grid-stride(步长 20)把它们摊平,负载差 ≤1 块。</text>')
foot_y2 = foot_y + 22
L.append(f'<text x="{PAD}" y="{foot_y2}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">对照标本 09-persistent-matmul.py:tiles_per_sm = num_tiles // NUM_SMS(同一「逻辑核数贴物理核数」模式)</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-persistent-grid-stride.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
