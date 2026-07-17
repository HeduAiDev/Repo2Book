#!/usr/bin/env python3
"""fig-scan-kogge-stone (state-table 模板)
8 车道 Kogge-Stone 前缀和 3 步(i=1,2,4):shuffleUp(i)+mask(lane>=i)+select,
车道向量逐步演化为完整前缀和 [1,2,3,4,5,6,7,8]。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "warp 内 8 车道 Kogge-Stone 前缀和(shuffleUp)——3 步得完整前缀和"
SUBTITLE = "全 1 初始输入,combine=sum;取前驱原语 shuffleUp(i)+mask(lane>=i),i 依次 1,2,4"

STEPS = ["初始", "步1 (i=1)", "步2 (i=2)", "步3 (i=4)"]
VECTORS = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 2, 2, 2, 2, 2, 2],
    [1, 2, 3, 4, 4, 4, 4, 4],
    [1, 2, 3, 4, 5, 6, 7, 8],
]
I_SEQ = ["-", "1", "2", "4"]
LANE_N = 8

PAD, TOP = 40, 96
LABEL_W = 96
STEP_BLOCK_W = 300
GAP = 26
LANE_W = STEP_BLOCK_W / LANE_N
HEADER_H = 34
LANE_ROW_H = 46
I_ROW_H = 30

w = PAD * 2 + LABEL_W + len(STEPS) * STEP_BLOCK_W + (len(STEPS) - 1) * GAP
h = TOP + HEADER_H + I_ROW_H + LANE_ROW_H + PAD + 44

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

block_x = [PAD + LABEL_W + i * (STEP_BLOCK_W + GAP) for i in range(len(STEPS))]

# 表头
for j, name in enumerate(STEPS):
    x = block_x[j]
    hot = (j == len(STEPS) - 1)
    L.append(f'<rect x="{x}" y="{TOP}" width="{STEP_BLOCK_W}" height="{HEADER_H}" rx="4" '
              f'fill="{"#1d4ed8" if hot else "#3b82f6"}" stroke="#1e3a5f" stroke-width="1.3"/>')
    L.append(f'<text x="{x+STEP_BLOCK_W/2}" y="{TOP+HEADER_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

# i 行标签(左侧)
i_row_y = TOP + HEADER_H
L.append(f'<text x="{PAD+LABEL_W-14}" y="{i_row_y+I_ROW_H/2+5}" text-anchor="end" '
          f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
          f'fill="#374151">{esc("shuffleUp 距离 i")}</text>')
for j in range(len(STEPS)):
    x = block_x[j]
    L.append(f'<text x="{x+STEP_BLOCK_W/2}" y="{i_row_y+I_ROW_H/2+5}" text-anchor="middle" '
              f'font-family="monospace" font-size="13" fill="#374151">{esc(I_SEQ[j])}</text>')

# 车道向量行
lane_row_y = i_row_y + I_ROW_H
L.append(f'<text x="{PAD+LABEL_W-14}" y="{lane_row_y+LANE_ROW_H/2+5}" text-anchor="end" '
          f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
          f'fill="#374151">{esc("acc[0..7]")}</text>')
for j in range(len(STEPS)):
    x = block_x[j]
    vec = VECTORS[j]
    prev = VECTORS[j-1] if j > 0 else None
    for lane in range(LANE_N):
        lx = x + lane * LANE_W
        changed = prev is not None and vec[lane] != prev[lane]
        fill = "#dcfce7" if changed else "#f1f5f9"
        stroke = "#059669" if changed else "#94a3b8"
        L.append(f'<rect x="{lx+2}" y="{lane_row_y+4}" width="{LANE_W-4}" height="{LANE_ROW_H-8}" rx="4" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
        text_fill = "#047857" if changed else "#334155"
        weight = 'font-weight="bold" ' if changed else ''
        L.append(f'<text x="{lx+LANE_W/2}" y="{lane_row_y+LANE_ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="monospace" font-size="12.5" fill="{text_fill}" {weight}>'
                  f'{esc(str(vec[lane]))}</text>')
    # lane 索引小标(仅第一列画一次,置于块下方)
L.append(f'<text x="{block_x[0]+STEP_BLOCK_W/2}" y="{lane_row_y+LANE_ROW_H+18}" '
          f'text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#64748b">'
          f'{esc("每格 = lane0..lane7,绿=本步发生累加")}</text>')

foot_y = lane_row_y + LANE_ROW_H + 42
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" fill="#64748b">'
          f'{esc("步数 = log2(8) = 3(warpScan 循环 i=1..4);第 3 步后 [1,2,3,4,5,6,7,8] = 完整前缀和")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-scan-kogge-stone.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  {w}x{h}")
