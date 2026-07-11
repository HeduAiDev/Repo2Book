#!/usr/bin/env python3
"""重绘自 arXiv:2602.06036 Figure 4(DFlash training attention)。
布局对齐原图(ref_x4.png,已下载核对):左侧 target 特征网格(浅蓝,对应 prompt p1-p4/
response 干净 token r1-r3),右侧 mask block 网格——每个 anchor(橙,干净 response token)
之后跟 3 个 mask(绿),块内双向可见、块间(白=不可见)互相屏蔽,并与对应的 target 特征列
对齐可见。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "DFlash 训练注意力掩码(重绘自 arXiv:2602.06036 Fig.4)"
SUBTITLE = "块内(锚点+3 mask)双向可见 + 对应 target 特征列可见,块间互相屏蔽——训练与推理共用同一套 KV 注入通路"

N_ROWS = 7  # 简化到 7 行(原图省略号后还有更多行,结构一致)
LEFT_COLS = 6  # p1 p2 p3 p4 r1 r2 (context feature)
RIGHT_COLS = 12  # r1 <m><m><m> r2 <m><m><m> r3 <m><m><m>
ANCHOR_COL = [0, 4, 8]  # r1, r2, r3 起始列(0-indexed within right grid)

CELL = 30
GAP = 0

BLUE = "#bfe3f0"
BLUE_STROKE = "#3a8fb0"
GREEN = "#8fd98f"
GREEN_STROKE = "#2f8f2f"
ORANGE = "#f5b942"
ORANGE_STROKE = "#c98a1a"
WHITE_STROKE = "#94a3b8"

PAD = 50
LEFT_X = PAD
TOP = 130
GAP_MID = 70
RIGHT_X = LEFT_X + LEFT_COLS * CELL + GAP_MID

grid_w = RIGHT_X + RIGHT_COLS * CELL + PAD
# legend 总宽须先算出再定画布宽度(否则末项会被裁)
_legend_items_len = [
    len("Target Context Feature"), len("Mask Token"),
    len("Clean Token(锚点)"), len("Invisible Token"),
]
_legend_w = sum(26 + 9 * n + 34 for n in _legend_items_len) + PAD
w = max(grid_w, _legend_w)
h = TOP + N_ROWS * CELL + 200

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="15.5" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>',
     f'<text x="{LEFT_X+LEFT_COLS*CELL/2}" y="{TOP-14}" text-anchor="middle" '
     f'font-family="sans-serif" font-size="13" font-weight="bold" fill="#0f172a">From Target Model</text>',
     f'<text x="{RIGHT_X+RIGHT_COLS*CELL/2}" y="{TOP-14}" text-anchor="middle" '
     f'font-family="sans-serif" font-size="13" font-weight="bold" fill="#0f172a">Mask Blocks</text>']

# left grid: each row i has 4 "prompt/context" blue cells fixed + progressively 1 more blue
# 对齐原图:每行左侧全蓝(该行能看到的 target 上下文特征随行数增多,前 4 行 4 格蓝,后 3 行 5 格蓝)
LEFT_BLUE_COUNT = [4, 4, 4, 4, 5, 5, 5]
for r in range(N_ROWS):
    for c in range(LEFT_COLS):
        y = TOP + r * CELL
        x = LEFT_X + c * CELL
        is_blue = c < LEFT_BLUE_COUNT[r]
        if is_blue:
            L.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                      f'fill="{BLUE}" stroke="{BLUE_STROKE}" stroke-width="1"/>')
        else:
            L.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                      f'fill="white" stroke="{WHITE_STROKE}" stroke-width="1"/>')
# left column bottom labels
left_labels = ["p1", "p2", "p3", "p4", "r1", "r2"]
for c, lab in enumerate(left_labels):
    x = LEFT_X + c * CELL + CELL/2
    L.append(f'<text x="{x}" y="{TOP+N_ROWS*CELL+18}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="#334155">{esc(lab)}</text>')

# right grid: 3 anchor blocks, each = 1 orange + 3 green; rows i correspond to positions within block
# block b occupies rows depending on which anchor being predicted this row -- simplify: rows 0-3 attend block0,
# rows correspond structurally like original (each row is one training example focusing on one anchor block)
block_row_ranges = [(0, 4), (4, 3)]  # not used directly; build directly per original: 4 rows for block0(after r1),
# then rows 4-6 for block1 (after r2) as in original figure (it shows fewer rows for later blocks due to truncation)
row_active_block = [0, 0, 0, 0, 1, 1, 1]  # which anchor block this row's visible green/orange belongs to

right_labels = ["r1", "<m>", "<m>", "<m>", "r2", "<m>", "<m>", "<m>", "r3", "<m>", "<m>", "<m>"]

for r in range(N_ROWS):
    y = TOP + r * CELL
    active_block = row_active_block[r]
    block_start = ANCHOR_COL[active_block]
    for c in range(RIGHT_COLS):
        x = RIGHT_X + c * CELL
        in_active_block = (block_start <= c < block_start + 4)
        if in_active_block:
            if c == block_start:
                fill, stroke = ORANGE, ORANGE_STROKE
            else:
                fill, stroke = GREEN, GREEN_STROKE
            L.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
        else:
            L.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                      f'fill="white" stroke="{WHITE_STROKE}" stroke-width="1"/>')

for c, lab in enumerate(right_labels):
    x = RIGHT_X + c * CELL + CELL/2
    L.append(f'<text x="{x}" y="{TOP+N_ROWS*CELL+18}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="#334155">{esc(lab)}</text>')

# ellipsis between left and right grids (visual separation marker, matches original "..." between blocks)
L.append(f'<text x="{(LEFT_X+LEFT_COLS*CELL + RIGHT_X)/2}" y="{TOP+N_ROWS*CELL/2}" '
          f'text-anchor="middle" font-family="sans-serif" font-size="18" fill="#94a3b8">...</text>')

# grid outer borders
L.append(f'<rect x="{LEFT_X}" y="{TOP}" width="{LEFT_COLS*CELL}" height="{N_ROWS*CELL}" '
          'fill="none" stroke="#0f172a" stroke-width="1.6"/>')
L.append(f'<rect x="{RIGHT_X}" y="{TOP}" width="{RIGHT_COLS*CELL}" height="{N_ROWS*CELL}" '
          'fill="none" stroke="#0f172a" stroke-width="1.6"/>')

# legend
leg_y = TOP + N_ROWS*CELL + 60
items = [("Target Context Feature", BLUE, BLUE_STROKE),
         ("Mask Token", GREEN, GREEN_STROKE),
         ("Clean Token(锚点)", ORANGE, ORANGE_STROKE),
         ("Invisible Token", "white", WHITE_STROKE)]
lx = PAD
for label, fill, stroke in items:
    L.append(f'<rect x="{lx}" y="{leg_y}" width="18" height="18" rx="3" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
    L.append(f'<text x="{lx+26}" y="{leg_y+14}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(label)}</text>')
    lx += 26 + 9 * len(label) + 34

foot_y = leg_y + 46
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">输入 = 干净 prompt token p + 干净 response token r;每块内随机采样的锚点(橙)之后跟 mask 占位(绿),块内双向可见。</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">白格(Invisible Token)强制块间互相看不见,防止跨块信息泄漏——训练与推理用的是同一套 target 特征 KV 注入通路。</text>')

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-4.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
