#!/usr/bin/env python3
"""fig-m3-ranksplit — tensor-flow 模板改造：
rank=5 的 vadd 超过库 maxRank=3，reduceMemrefsToNestedFor 把外 2 轴（尺寸 2、3）
拆成 2 层嵌套 scf.for，循环体内对 rank-3 subview 调库，共 2x3=6 条 func.call。
坐标全部由常量/循环计算，箭头端点取自元素边缘。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "reduceMemrefsToNestedFor：rank=5 的 vadd 怎么拆循环喂给 rank<=3 的库"

IN_LABEL = "输入：memref<2x3x4x8x16xf16>（rank=5）"
GATE_LABEL = "库 maxRank（VAddOp）= 3    rank(5) > maxRank(3) → 需要拆循环"
GATE_NUM = "LibraryFunctionOpInterfaceImpl.cpp:L1122"
SPLIT_LABEL = "reduceMemrefsToNestedFor(..., 0, rank-maxRank=2)：被拆轴数 = 2 → 生成 2 层嵌套 scf.for"
SPLIT_NUM = "HIVMToStandard.cpp:L936-940"

OUTER_LABEL = "for i0 in [0, 2)"
OUTER_NUM = "外轴 0 尺寸 = 2"
INNER_LABEL = "for i1 in [0, 3)"
INNER_NUM = "外轴 1 尺寸 = 3"

# 6 次迭代：(i0, i1, offset 展示串)
ITERS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

CALL_LABEL = "func.call @vadd_3d_half(subview)"
CALL_SHAPE = "subview rank = maxRank = 3 → memref<4x8x16xf16>"
CALL_SHAPE_NUM = "HIVMToStandard.cpp:L229-L338（rank-reduced subview）"

TOTAL_LABEL = "总调用次数 = 外轴尺寸之积 = 2×3 = 6"

CAPTION = ("库只认 <=3 维：编译期用 2 层 scf.for 把外 2 轴（尺寸 2、3）逐个拉开，"
           "每次拉开留下一个 rank-3 的 subview，对它调一次 vadd_3d_half——"
           "6 次调用后，rank=5 的输入被处理完，库本身从未见过 5 维。")

FONT = "sans-serif"
NUM_FILL = "#b91c1c"

PAD = 44
W = 1180

# ---- 顶部输入框 ----
IN_Y, IN_H = 84, 46
IN_W = 460
in_x = W / 2 - IN_W / 2

# ---- gate（maxRank 判定）----
GATE_Y = IN_Y + IN_H + 46
GATE_W, GATE_H = 760, 40
gate_x = W / 2 - GATE_W / 2

# ---- split 说明框 ----
SPLIT_Y = GATE_Y + GATE_H + 40
SPLIT_W, SPLIT_H = 820, 40
split_x = W / 2 - SPLIT_W / 2

# ---- 嵌套循环外框 ----
LOOP_Y = SPLIT_Y + SPLIT_H + 56
OUTER_PAD = 26
INNER_PAD = 24
CELL_W, CELL_H, CELL_GAP = 150, 60, 16
GRID_COLS, GRID_ROWS = 3, 2
grid_w = GRID_COLS * CELL_W + (GRID_COLS - 1) * CELL_GAP
grid_h = GRID_ROWS * CELL_H + (GRID_ROWS - 1) * CELL_GAP
inner_w = grid_w + INNER_PAD * 2
inner_h = grid_h + INNER_PAD * 2 + 26   # +26 给 inner 标题留空间
outer_w = inner_w + OUTER_PAD * 2
outer_h = inner_h + OUTER_PAD * 2 + 26  # +26 给 outer 标题留空间
outer_x = W / 2 - outer_w / 2
outer_y = LOOP_Y
inner_x = outer_x + OUTER_PAD
inner_y = outer_y + OUTER_PAD + 26
grid_x = inner_x + INNER_PAD
grid_y = inner_y + INNER_PAD + 26

# ---- 汇聚调用框 ----
CALL_Y = outer_y + outer_h + 60
CALL_W, CALL_H = 620, 78
call_x = W / 2 - CALL_W / 2

H = CALL_Y + CALL_H + 100

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4.4" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0369a1"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="38" text-anchor="middle" font-family="{FONT}" font-size="17" '
          f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')

# 输入框
L.append(f'<rect x="{in_x}" y="{IN_Y}" width="{IN_W}" height="{IN_H}" rx="8" '
          f'fill="#f1f5f9" stroke="#334155" stroke-width="2"/>')
L.append(f'<text x="{W/2}" y="{IN_Y+IN_H/2+5}" text-anchor="middle" font-family="monospace" '
          f'font-size="13" fill="#0f172a">{esc(IN_LABEL)}</text>')

# 箭头 输入 -> gate
L.append(f'<line x1="{W/2}" y1="{IN_Y+IN_H}" x2="{W/2}" y2="{GATE_Y}" '
          f'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')

# gate 框
L.append(f'<rect x="{gate_x}" y="{GATE_Y}" width="{GATE_W}" height="{GATE_H}" rx="8" '
          f'fill="#fef9c3" stroke="#a16207" stroke-width="2"/>')
L.append(f'<text x="{W/2}" y="{GATE_Y+GATE_H/2+5}" text-anchor="middle" font-family="{FONT}" '
          f'font-size="12.5" fill="#713f12">{esc(GATE_LABEL)}</text>')
L.append(f'<text x="{gate_x+GATE_W}" y="{GATE_Y-8}" text-anchor="end" font-family="{FONT}" '
          f'font-size="10.5" font-weight="bold" fill="{NUM_FILL}">{esc(GATE_NUM)}</text>')

# 箭头 gate -> split
L.append(f'<line x1="{W/2}" y1="{GATE_Y+GATE_H}" x2="{W/2}" y2="{SPLIT_Y}" '
          f'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')

# split 框
L.append(f'<rect x="{split_x}" y="{SPLIT_Y}" width="{SPLIT_W}" height="{SPLIT_H}" rx="8" '
          f'fill="#ede9fe" stroke="#6d28d9" stroke-width="2"/>')
L.append(f'<text x="{W/2}" y="{SPLIT_Y+SPLIT_H/2+5}" text-anchor="middle" font-family="{FONT}" '
          f'font-size="12.5" fill="#4c1d95">{esc(SPLIT_LABEL)}</text>')
L.append(f'<text x="{split_x+SPLIT_W}" y="{SPLIT_Y-8}" text-anchor="end" font-family="{FONT}" '
          f'font-size="10.5" font-weight="bold" fill="{NUM_FILL}">{esc(SPLIT_NUM)}</text>')

# 箭头 split -> 嵌套循环外框
L.append(f'<line x1="{W/2}" y1="{SPLIT_Y+SPLIT_H}" x2="{W/2}" y2="{outer_y}" '
          f'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')

# 外层循环框 (i0)
L.append(f'<rect x="{outer_x}" y="{outer_y}" width="{outer_w}" height="{outer_h}" rx="10" '
          f'fill="none" stroke="#0369a1" stroke-width="2" stroke-dasharray="8,5"/>')
L.append(f'<text x="{outer_x+14}" y="{outer_y+22}" font-family="monospace" font-size="13" '
          f'font-weight="bold" fill="#0369a1">{esc(OUTER_LABEL)}</text>')
L.append(f'<text x="{outer_x+outer_w-14}" y="{outer_y+22}" text-anchor="end" font-family="{FONT}" '
          f'font-size="11" fill="#0369a1">{esc(OUTER_NUM)}</text>')

# 内层循环框 (i1)
L.append(f'<rect x="{inner_x}" y="{inner_y}" width="{inner_w}" height="{inner_h}" rx="10" '
          f'fill="none" stroke="#0891b2" stroke-width="2" stroke-dasharray="5,4"/>')
L.append(f'<text x="{inner_x+14}" y="{inner_y+22}" font-family="monospace" font-size="13" '
          f'font-weight="bold" fill="#0e7490">{esc(INNER_LABEL)}</text>')
L.append(f'<text x="{inner_x+inner_w-14}" y="{inner_y+22}" text-anchor="end" font-family="{FONT}" '
          f'font-size="11" fill="#0e7490">{esc(INNER_NUM)}</text>')

# 6 个迭代格
cell_centers = []
for idx, (i0, i1) in enumerate(ITERS):
    row, col = i0, i1
    cx0 = grid_x + col * (CELL_W + CELL_GAP)
    cy0 = grid_y + row * (CELL_H + CELL_GAP)
    cell_centers.append((cx0 + CELL_W / 2, cy0 + CELL_H))
    L.append(f'<rect x="{cx0}" y="{cy0}" width="{CELL_W}" height="{CELL_H}" rx="6" '
              f'fill="#e0f2fe" stroke="#0369a1" stroke-width="1.5"/>')
    L.append(f'<text x="{cx0+CELL_W/2}" y="{cy0+22}" text-anchor="middle" font-family="{FONT}" '
              f'font-size="11" fill="#0c4a6e">{esc(f"i0={i0}, i1={i1}")}</text>')
    L.append(f'<text x="{cx0+CELL_W/2}" y="{cy0+42}" text-anchor="middle" font-family="monospace" '
              f'font-size="10.5" fill="#0c4a6e">{esc(f"offset=[{i0},{i1},·]")}</text>')

# 箭头 汇聚循环外框 -> call 框
L.append(f'<line x1="{W/2}" y1="{outer_y+outer_h}" x2="{W/2}" y2="{CALL_Y}" '
          f'stroke="#334155" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{W/2+14}" y="{(outer_y+outer_h+CALL_Y)/2+4}" font-family="{FONT}" '
          f'font-size="11.5" fill="#334155">{esc("6 个格子 -> 同一个库函数，仅 subview 不同")}</text>')

# call 框
L.append(f'<rect x="{call_x}" y="{CALL_Y}" width="{CALL_W}" height="{CALL_H}" rx="10" '
          f'fill="#dcfce7" stroke="#15803d" stroke-width="2.5"/>')
L.append(f'<text x="{W/2}" y="{CALL_Y+24}" text-anchor="middle" font-family="monospace" '
          f'font-size="13.5" font-weight="bold" fill="#14532d">{esc(CALL_LABEL)}</text>')
L.append(f'<text x="{W/2}" y="{CALL_Y+44}" text-anchor="middle" font-family="{FONT}" '
          f'font-size="11" fill="#14532d">{esc(CALL_SHAPE)}</text>')
L.append(f'<text x="{W/2}" y="{CALL_Y+62}" text-anchor="middle" font-family="{FONT}" '
          f'font-size="10.5" font-weight="bold" fill="{NUM_FILL}">{esc(CALL_SHAPE_NUM)}</text>')

# 总次数标注
L.append(f'<text x="{W/2}" y="{CALL_Y+CALL_H+26}" text-anchor="middle" font-family="{FONT}" '
          f'font-size="13" font-weight="bold" fill="#0f172a">{esc(TOTAL_LABEL)}</text>')

# 图注
L.append(f'<text x="{W/2}" y="{H-16}" text-anchor="middle" font-family="{FONT}" font-size="12" '
          f'fill="#475569">{esc(CAPTION)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m3-ranksplit.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={W} h={H}")
