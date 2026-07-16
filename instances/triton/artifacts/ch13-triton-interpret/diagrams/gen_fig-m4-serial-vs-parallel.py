#!/usr/bin/env python3
"""fig-m4-serial-vs-parallel: swimlane 变体。上：GPU 本应把 grid 的 2 个 program
并行铺开在同一时刻；下：GridExecutor 三重 for 把它们摊成 2 次顺序调用（CPU 替身实测）。
全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

W = 940
PAD = 40
LANE_LABEL_W = 150
TRACK_W = 690
BOX_W, BOX_H = 220, 62
GAP = 170  # horizontal gap between the two program boxes — wide enough for the arrow label

TOP1 = 90     # GPU panel
TOP2 = 320    # CPU panel (extra headroom reserved for the order badges)
H = 620

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="15" '
     f'font-weight="bold" fill="#0f172a">'
     f'{esc("grid=(2,) 补齐 (2,1,1)：GPU 并行铺开 vs GridExecutor 串行喂入")}</text>']

track_x0 = PAD + LANE_LABEL_W

# --- Panel 1: GPU (本应，对照，灰化) ---
L.append(f'<text x="{PAD}" y="{TOP1-14}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#64748b">{esc("GPU（本应，对照）")}</text>')
L.append(f'<line x1="{track_x0}" y1="{TOP1-4}" x2="{track_x0+TRACK_W}" y2="{TOP1-4}" '
          'stroke="#cbd5e1" stroke-width="1"/>')
gpu_y = TOP1 + 10
gpu_x_positions = [track_x0 + 40, track_x0 + 40 + BOX_W + GAP]
for i, gx in enumerate(gpu_x_positions):
    L.append(f'<rect x="{gx}" y="{gpu_y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,3"/>')
    L.append(f'<text x="{gx+BOX_W/2}" y="{gpu_y+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="#475569">{esc(f"program_id={i}")}</text>')
    L.append(f'<text x="{gx+BOX_W/2}" y="{gpu_y+40}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#64748b">{esc("与另一 program 同时跑")}</text>')
L.append(f'<text x="{track_x0+TRACK_W/2}" y="{gpu_y+BOX_H+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#94a3b8">'
          f'{esc("同一时刻，不同 SM/warp——两个框在时间轴上并排")}</text>')

# --- Panel 2: CPU (实际，高亮) ---
L.append(f'<text x="{PAD}" y="{TOP2-14}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#1e40af">{esc("CPU 替身（实测，GridExecutor 三重 for）")}</text>')
L.append(f'<line x1="{track_x0}" y1="{TOP2-4}" x2="{track_x0+TRACK_W}" y2="{TOP2-4}" '
          'stroke="#3b82f6" stroke-width="1.5"/>')
cpu_y = TOP2 + 40
cpu_x_positions = [track_x0 + 40, track_x0 + 40 + BOX_W + GAP]
labels = [
    ("program_id=0", "offs=[0,1,2,3]", "y=[0,2,4,6]"),
    ("program_id=1", "offs=[4,5,6,7]", "y=[8,10,12,14]"),
]
for i, (cx, lab) in enumerate(zip(cpu_x_positions, labels)):
    L.append(f'<rect x="{cx}" y="{cpu_y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              'fill="#dbeafe" stroke="#2563eb" stroke-width="2.5"/>')
    L.append(f'<text x="{cx+BOX_W/2}" y="{cpu_y+20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="#1e3a8a">{esc(lab[0])}</text>')
    L.append(f'<text x="{cx+BOX_W/2}" y="{cpu_y+37}" text-anchor="middle" font-family="monospace" '
              f'font-size="10.5" fill="#1e40af">{esc(lab[1])}</text>')
    L.append(f'<text x="{cx+BOX_W/2}" y="{cpu_y+53}" text-anchor="middle" font-family="monospace" '
              f'font-size="10.5" font-weight="bold" fill="#1d4ed8">{esc(lab[2])}</text>')
    # order number: centered above each box, clear of the mid-height arrow lane
    badge_cx = cx + BOX_W / 2
    badge_cy = cpu_y - 20
    L.append(f'<circle cx="{badge_cx}" cy="{badge_cy}" r="13" fill="#2563eb" stroke="white" stroke-width="2"/>')
    L.append(f'<text x="{badge_cx}" y="{badge_cy+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" fill="white">{esc(str(i+1))}</text>')
# arrow between the two sequential boxes (串行: 0 整段跑完才轮到 1)
arrow_y = cpu_y + BOX_H / 2
L.append(f'<line x1="{cpu_x_positions[0]+BOX_W}" y1="{arrow_y}" '
          f'x2="{cpu_x_positions[1]}" y2="{arrow_y}" '
          'stroke="#1d4ed8" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(cpu_x_positions[0]+BOX_W+cpu_x_positions[1])/2}" y="{arrow_y-14}" '
          f'text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="bold" '
          f'fill="#1d4ed8">{esc("整段跑完才轮到下一个")}</text>')
L.append(f'<text x="{track_x0+TRACK_W/2}" y="{cpu_y+BOX_H+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#1e40af">'
          f'{esc("同一时间轴上，两个框首尾相接——单线程 numpy，无并发")}</text>')

# code anchor + numbers footer
footer_y = cpu_y + BOX_H + 55
L.append(f'<rect x="{PAD}" y="{footer_y}" width="{W-2*PAD}" height="58" rx="8" '
          'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
L.append(f'<text x="{PAD+16}" y="{footer_y+22}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">'
          f'{esc("grid (2,)→补齐(2,1,1)；串行调用点：for x/y/z: set_grid_idx; self.fn(**args)（interpreter.py:L1093-1102）")}</text>')
L.append(f'<text x="{PAD+16}" y="{footer_y+42}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">'
          f'{esc("program_id 不是硬件算出来的，是 GridExecutor 一个个塞进来的——这就是『串行 ≠ 并行』的画面")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m4-serial-vs-parallel.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
