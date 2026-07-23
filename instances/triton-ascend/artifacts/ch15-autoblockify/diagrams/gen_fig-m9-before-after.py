#!/usr/bin/env python3
"""before-after 模板:m9 收益量化——size=5 折叠把调度块数从 G 降到 ceil(G/size)、
把 5 条标量 store 折成 1 条向量 store。左panel=6个未折叠物理块各1条store;
右panel=2个折叠后物理块(block0覆盖5实例1条向量store,尾块覆盖1实例)。
全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "折叠收益(size=5,grid=3×2×1,G=6)"
LEFT_TITLE = "未折叠:6 个调度块"
RIGHT_TITLE = "折叠后:2 个调度块 (⌈6/5⌉)"
LEFT_CELL_W, LEFT_CELL_H = 96, 64
RIGHT_CELL_W, RIGHT_CELL_H = 220, 90

PAD, TOP = 40, 108
LEFT_COLS = 6
left_w = LEFT_COLS * LEFT_CELL_W + (LEFT_COLS - 1) * 10
right_gap = 30
right_w = 2 * RIGHT_CELL_W + right_gap
panel_gap = 90
w = PAD * 2 + left_w + panel_gap + right_w
h = TOP + max(LEFT_CELL_H, RIGHT_CELL_H) + 200

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="{PAD}" text-anchor="middle" font-family="sans-serif" '
     f'font-size="16" font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>']

# LEFT PANEL: 6 blocks in a row, each 1 逻辑实例 tensor<8> + 1 store
lx0 = PAD
L.append(f'<text x="{lx0 + left_w/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="#0f172a">{esc(LEFT_TITLE)}</text>')
for i in range(LEFT_COLS):
    x = lx0 + i * (LEFT_CELL_W + 10)
    L.append(f'<rect x="{x}" y="{TOP}" width="{LEFT_CELL_W}" height="{LEFT_CELL_H}" rx="7" '
              f'fill="#e2e8f0" stroke="#64748b" stroke-width="1.3"/>')
    L.append(f'<text x="{x+LEFT_CELL_W/2}" y="{TOP+22}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="#334155">{esc(f"块{i}")}</text>')
    L.append(f'<text x="{x+LEFT_CELL_W/2}" y="{TOP+40}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.3" fill="#0f172a">{esc("tensor<8>")}</text>')
    L.append(f'<text x="{x+LEFT_CELL_W/2}" y="{TOP+56}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.3" fill="#0f172a">{esc("1 store")}</text>')
L.append(f'<text x="{lx0 + left_w/2}" y="{TOP+LEFT_CELL_H+26}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" fill="#475569">'
          f'{esc("6 个逻辑实例各占 1 调度块 · 各发 1 条 tensor<8> store · 共 6 条 store")}</text>')

# ARROW between panels
mid_y = TOP + max(LEFT_CELL_H, RIGHT_CELL_H) / 2
ax1 = lx0 + left_w + 14
ax2 = ax1 + panel_gap - 28
L.append(f'<line x1="{ax1}" y1="{mid_y}" x2="{ax2}" y2="{mid_y}" '
          'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{mid_y-12}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="#b45309">{esc("size=5 折叠")}</text>')

# RIGHT PANEL: 2 blocks
rx0 = lx0 + left_w + panel_gap
L.append(f'<text x="{rx0 + right_w/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="#0f172a">{esc(RIGHT_TITLE)}</text>')
RIGHT_BLOCKS = [
    ("块0(blockId=0)", "覆盖 5 个逻辑实例(0..4)", "tensor<5x8>", "1 条向量 store"),
    ("尾块(blockId=5)", "覆盖 1 个逻辑实例(5)", "-", "1 条 store"),
]
for i, (name, cover, shape, store) in enumerate(RIGHT_BLOCKS):
    x = rx0 + i * (RIGHT_CELL_W + right_gap)
    hl = (i == 0)
    L.append(f'<rect x="{x}" y="{TOP}" width="{RIGHT_CELL_W}" height="{RIGHT_CELL_H}" rx="8" '
              f'fill="{"#fef3c7" if hl else "#e2e8f0"}" '
              f'stroke="{"#d97706" if hl else "#64748b"}" stroke-width="{2 if hl else 1.3}"/>')
    L.append(f'<text x="{x+RIGHT_CELL_W/2}" y="{TOP+20}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.3" font-weight="bold" '
              f'fill="#0f172a">{esc(name)}</text>')
    L.append(f'<text x="{x+RIGHT_CELL_W/2}" y="{TOP+40}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.3" fill="#334155">{esc(cover)}</text>')
    if shape != "-":
        L.append(f'<text x="{x+RIGHT_CELL_W/2}" y="{TOP+60}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" font-weight="bold" '
                  f'fill="#0f172a">{esc(shape)}</text>')
    L.append(f'<text x="{x+RIGHT_CELL_W/2}" y="{TOP+RIGHT_CELL_H-10}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="#475569">{esc(store)}</text>')
L.append(f'<text x="{rx0 + right_w/2}" y="{TOP+RIGHT_CELL_H+26}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" fill="#475569">'
          f'{esc("调度块数 6→2、覆盖 5 实例的 5 条 store 折成 1 条 tensor<5x8> 向量 store")}</text>')

# bottom summary strip
sy = TOP + max(LEFT_CELL_H, RIGHT_CELL_H) + 70
rows = [
    ("调度块数", "6", "2 (⌈6/5⌉)"),
    ("覆盖 5 实例所需 store 指令数", "5", "1"),
    ("前导张量形状(该批 5 实例)", "tensor<8>", "tensor<5x8>"),
    ("driver 物理核 clamp", "-", "blockNum=std::min(blockNum,num_physical_blocks) (driver.py:L788)"),
]
col_x = [PAD, PAD + 330, PAD + 500]
L.append(f'<line x1="{PAD}" y1="{sy-14}" x2="{w-PAD}" y2="{sy-14}" stroke="#cbd5e1" stroke-width="1"/>')
headers = ["指标", "未折叠", "折叠 size=5"]
for cx0, htext in zip(col_x, headers):
    L.append(f'<text x="{cx0}" y="{sy+6}" font-family="sans-serif" font-size="11.5" '
              f'font-weight="bold" fill="#475569">{esc(htext)}</text>')
for i, (a, b, c) in enumerate(rows):
    ry = sy + 30 + i * 26
    L.append(f'<text x="{col_x[0]}" y="{ry}" font-family="sans-serif" font-size="12" '
              f'fill="#0f172a">{esc(a)}</text>')
    L.append(f'<text x="{col_x[1]}" y="{ry}" font-family="sans-serif" font-size="12" '
              f'fill="#0f172a">{esc(b)}</text>')
    L.append(f'<text x="{col_x[2]}" y="{ry}" font-family="sans-serif" font-size="12" '
              f'fill="#0f172a">{esc(c)}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-m9-before-after.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f"wrote {out}")
