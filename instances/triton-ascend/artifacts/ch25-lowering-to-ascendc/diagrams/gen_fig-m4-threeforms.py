#!/usr/bin/env python3
"""fig-m4-threeforms — flow 模板改造：49 个 HIVM->Standard 硬件 op 全落在三种
形态（直接调库 / rank 门控 / 按轴拆循环），三条支线殊途同归到同一个 createLibCall 出口。
坐标全部由常量/循环计算，箭头端点取自元素边缘。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "49 个 HIVM 硬件 op 被设为 illegal：三种形态，殊途同归"

# 每条支线: (标题, 标题色, 填充色, 边框色, [ (行文字, 行号) , ... ])
LANES = [
    ("形态 A：直接调库（无 rank 门控）", "#7c2d12", "#ffedd5", "#c2410c", [
        ("MmadL1Op（NoMaxRank）", "L345"),
        ("ND2NZOp / FixpipeOp（NoMaxRank）", "L388 / L421"),
    ]),
    ("形态 B：rank 门控（超 maxRank 才拆循环）", "#0c4a6e", "#e0f2fe", "#0369a1", [
        ("VAddOp maxRank=3（rank<=3 直调，否则拆）", "L920"),
        ("LoadOp / StoreOp / CopyOp maxRank=3", "L736"),
    ]),
    ("形态 C：按轴拆循环（沿语义轴）", "#4c1d95", "#ede9fe", "#6d28d9", [
        ("VReduce / VTranspose / VBrc（InferMaxRank）", "L1252 / L1414 / L1162"),
    ]),
]

EXIT_LABEL = "createLibCall：getOrInsert 声明 + emit func.call"
EXIT_NUM = "HIVMToStandard.cpp:L106-L168"
CAPTION_LINES = [
    "49 个硬件 op 只有三种形态：矩阵/格式转换/fixpipe 不设 rank 上限直接调库；"
    "普通向量/搬运 op 先比 rank 再决定要不要拆循环；",
    "规约/转置/广播必须沿指定语义轴拆循环。三条路最终都汇到同一个 createLibCall，把 op 变成 func.call。",
]

FONT = "sans-serif"
NUM_FILL = "#b91c1c"

PAD = 44
TOP = 96
LANE_W = 360
LANE_GAP = 40
LANE_HEAD_H = 46
ROW_H = 50
N = len(LANES)
W = PAD * 2 + LANE_W * N + LANE_GAP * (N - 1)

lane_x = [PAD + i * (LANE_W + LANE_GAP) for i in range(N)]
lane_h = [LANE_HEAD_H + len(rows) * ROW_H + 14 for (_, _, _, _, rows) in LANES]
max_lane_h = max(lane_h)

EXIT_Y = TOP + max_lane_h + 70
EXIT_W, EXIT_H = 620, 56
exit_x = W / 2 - EXIT_W / 2

H = EXIT_Y + EXIT_H + 40 + 20 * len(CAPTION_LINES) + 24

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="38" text-anchor="middle" font-family="{FONT}" font-size="17" '
          f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')

lane_bottom_centers = []
for i, (head, head_color, fill, stroke, rows) in enumerate(LANES):
    x = lane_x[i]
    h_i = lane_h[i]
    L.append(f'<rect x="{x}" y="{TOP}" width="{LANE_W}" height="{h_i}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{x+LANE_W/2}" y="{TOP+26}" text-anchor="middle" font-family="{FONT}" '
              f'font-size="12.5" font-weight="bold" fill="{head_color}">{esc(head)}</text>')
    L.append(f'<line x1="{x+14}" y1="{TOP+LANE_HEAD_H}" x2="{x+LANE_W-14}" y2="{TOP+LANE_HEAD_H}" '
              f'stroke="{stroke}" stroke-width="1" stroke-dasharray="3,3"/>')
    for j, (txt, num) in enumerate(rows):
        ry = TOP + LANE_HEAD_H + 22 + j * ROW_H
        L.append(f'<text x="{x+16}" y="{ry}" font-family="{FONT}" font-size="11.5" '
                  f'fill="{head_color}">{esc("· " + txt)}</text>')
        L.append(f'<text x="{x+LANE_W-14}" y="{ry+18}" text-anchor="end" font-family="{FONT}" '
                  f'font-size="10.5" font-weight="bold" fill="{NUM_FILL}">{esc(num)}</text>')
    cx = x + LANE_W / 2
    lane_bottom_centers.append((cx, TOP + h_i))

# 汇聚箭头：每条支线底部中点 -> exit 框顶部（按等距落点，避免重叠）
exit_top_y = EXIT_Y
n_lanes = len(lane_bottom_centers)
for k, (cx, cy) in enumerate(lane_bottom_centers):
    tx = exit_x + EXIT_W * (k + 1) / (n_lanes + 1)
    L.append(f'<line x1="{cx}" y1="{cy}" x2="{tx}" y2="{exit_top_y}" '
              f'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')

# exit 框
L.append(f'<rect x="{exit_x}" y="{EXIT_Y}" width="{EXIT_W}" height="{EXIT_H}" rx="10" '
          f'fill="#dcfce7" stroke="#15803d" stroke-width="2.5"/>')
L.append(f'<text x="{W/2}" y="{EXIT_Y+24}" text-anchor="middle" font-family="monospace" '
          f'font-size="13" font-weight="bold" fill="#14532d">{esc(EXIT_LABEL)}</text>')
L.append(f'<text x="{W/2}" y="{EXIT_Y+44}" text-anchor="middle" font-family="{FONT}" '
          f'font-size="11" font-weight="bold" fill="{NUM_FILL}">{esc(EXIT_NUM)}</text>')

# 图注（多行，避免单行超宽被裁切）
caption_start_y = H - 16 - (len(CAPTION_LINES) - 1) * 20
for i, line in enumerate(CAPTION_LINES):
    L.append(f'<text x="{W/2}" y="{caption_start_y + i*20}" text-anchor="middle" '
              f'font-family="{FONT}" font-size="12" fill="#475569">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m4-threeforms.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={W} h={H}")
