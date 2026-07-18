#!/usr/bin/env python3
"""fig-m3-block-mask-tiling: tiling 模板变体——上排=97 个逻辑 program(压缩显示首/尾),
下排=对应内存分块条带,最后一块(pid=96)被 mask 截断成『有效/越界』两段。
全坐标由循环/常量计算,零手写魔数。numbers 全部来自 explainer figure_specs。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "grid=cdiv(98432, 1024)=97 个 program，只有最后一块被 mask 截断"

GRID = 97
FULL_BLOCKS = 96
BLOCK_SIZE = 1024
N_ELEMENTS = 98432
TAIL_VALID = 128
TAIL_MASKED = 896

# 上排:program 盒子(压缩表示)
PROG_BOXES = [
    {"label": "pid=0", "kind": "normal"},
    {"label": "pid=1", "kind": "normal"},
    {"label": "⋯ pid=2..94 ⋯", "kind": "ellipsis"},
    {"label": "pid=95", "kind": "normal"},
    {"label": "pid=96", "kind": "tail"},
]

PBOX_W, PBOX_H = 140, 46
PBOX_GAP = 26
PAD, TOP = 44, 78
N = len(PROG_BOXES)
prog_row_w = N * PBOX_W + (N - 1) * PBOX_GAP
W = PAD * 2 + prog_row_w

prog_xs = [PAD + i * (PBOX_W + PBOX_GAP) for i in range(N)]
prog_y = TOP

# 下排:内存条带。左侧大块代表 96 个满块(0..95),右侧尾块拆成 valid/masked 两段。
BAR_Y = prog_y + PBOX_H + 90
BAR_H = 56
FULL_BAR_W = prog_xs[3] + PBOX_W - prog_xs[0]          # 对齐 pid=0..pid=95 的横向跨度
TAIL_TOTAL_W = PBOX_W                                    # 与 pid=96 盒子对齐宽度
tail_x = prog_xs[4]
valid_w = max(34, TAIL_TOTAL_W * TAIL_VALID / BLOCK_SIZE)
masked_w = TAIL_TOTAL_W - valid_w

full_bar_x = prog_xs[0]

H = BAR_Y + BAR_H + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs>',
     '<marker id="arrow" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>',
     '<pattern id="hatch" width="8" height="8" patternTransform="rotate(45)" '
     'patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#e2e8f0"/>'
     '<line x1="0" y1="0" x2="0" y2="8" stroke="#94a3b8" stroke-width="3"/></pattern>',
     '</defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>']

L.append(f'<text x="{W/2}" y="{PAD-8}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="17" font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')

# 上排小标题
L.append(f'<text x="{PAD}" y="{prog_y-12}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#475569">① 逻辑 program（grid={GRID}）</text>')

COLORS = {
    "normal": ("#e0f2fe", "#0284c7", "#075985"),
    "ellipsis": ("white", "#94a3b8", "#64748b"),
    "tail": ("#ffedd5", "#ea580c", "#9a3412"),
}
for box, x in zip(PROG_BOXES, prog_xs):
    fill, stroke, text = COLORS[box["kind"]]
    dash = ' stroke-dasharray="5,4"' if box["kind"] == "ellipsis" else ""
    L.append(f'<rect x="{x}" y="{prog_y}" width="{PBOX_W}" height="{PBOX_H}" rx="6" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"{dash}/>')
    L.append(f'<text x="{x+PBOX_W/2}" y="{prog_y+PBOX_H/2+5}" text-anchor="middle" '
              f'font-family="monospace" font-size="13.5" fill="{text}">{esc(box["label"])}</text>')

# 下排小标题
L.append(f'<text x="{PAD}" y="{BAR_Y-12}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#475569">② 对应内存分块（offsets）</text>')

# 大块:96 个满块
L.append(f'<rect x="{full_bar_x}" y="{BAR_Y}" width="{FULL_BAR_W}" height="{BAR_H}" rx="6" '
          f'fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>')
L.append(f'<text x="{full_bar_x+FULL_BAR_W/2}" y="{BAR_Y+BAR_H/2-2}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13" font-weight="bold" '
          f'fill="#166534">{FULL_BLOCKS} 个满块（pid=0..95）</text>')
L.append(f'<text x="{full_bar_x+FULL_BAR_W/2}" y="{BAR_Y+BAR_H/2+16}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" '
          f'fill="#166534">{FULL_BLOCKS}×{BLOCK_SIZE} = {FULL_BLOCKS*BLOCK_SIZE} 个元素，全部有效</text>')

# 尾块:valid + masked
L.append(f'<rect x="{tail_x}" y="{BAR_Y}" width="{valid_w}" height="{BAR_H}" '
          f'fill="#bbf7d0" stroke="#16a34a" stroke-width="2"/>')
L.append(f'<rect x="{tail_x+valid_w}" y="{BAR_Y}" width="{masked_w}" height="{BAR_H}" '
          f'fill="url(#hatch)" stroke="#94a3b8" stroke-width="2"/>')

# valid 段标注(外部引出到上方,短数字避免碰撞)
callout_y = BAR_Y - 20
valid_label_x = tail_x - 8   # 段太窄,标注锚点略偏左、左对齐,避免与右侧 masked 标注重叠
L.append(f'<line x1="{tail_x+valid_w/2}" y1="{BAR_Y}" x2="{valid_label_x}" y2="{callout_y+6}" '
          f'stroke="#16a34a" stroke-width="1.2" stroke-dasharray="3,3"/>')
L.append(f'<text x="{valid_label_x}" y="{callout_y}" text-anchor="end" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" '
          f'fill="#166534">{TAIL_VALID} 有效</text>')

# masked 段标注放在条带下方,与 valid 标注(上方)不共线,彻底避免重叠
label_y1 = BAR_Y + BAR_H + 20
label_y2 = label_y1 + 18
L.append(f'<text x="{tail_x+TAIL_TOTAL_W/2}" y="{label_y1}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" '
          f'fill="#334155">{TAIL_MASKED} 越界（mask 挡）</text>')
L.append(f'<text x="{tail_x+TAIL_TOTAL_W/2}" y="{label_y2}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">block 96：offsets 98304..99327</text>')

# 箭头:pid=0..95 组 -> 大块;pid=96 -> 尾块
group_cx = (prog_xs[0] + prog_xs[3] + PBOX_W) / 2
L.append(f'<line x1="{group_cx}" y1="{prog_y+PBOX_H+4}" x2="{full_bar_x+FULL_BAR_W/2}" '
          f'y2="{BAR_Y-4}" stroke="#16a34a" stroke-width="1.8" marker-end="url(#arrow)"/>')
tail_cx = prog_xs[4] + PBOX_W / 2
L.append(f'<line x1="{tail_cx}" y1="{prog_y+PBOX_H+4}" x2="{tail_cx}" y2="{BAR_Y-4}" '
          f'stroke="#ea580c" stroke-width="1.8" marker-end="url(#arrow)"/>')

# 图注
cap_y = H - 46
CAPTION_LINES = [
    f"size={N_ELEMENTS} 特意取非 {BLOCK_SIZE} 整数倍——最后一个 program(pid=96) 的 offsets 里，",
    f"只有前 {TAIL_VALID} 个 <{N_ELEMENTS} 有效，其余 {TAIL_MASKED} 个越界偏移被 mask=offsets<n_elements 逐个挡掉。",
]
for i, line in enumerate(CAPTION_LINES):
    L.append(f'<text x="{W/2}" y="{cap_y+i*18}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="#475569">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m3-block-mask-tiling.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
