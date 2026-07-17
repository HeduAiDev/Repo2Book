#!/usr/bin/env python3
"""tiling 模板(自定义):BlockedEncoding 的 sizePerThread/threadsPerWarp/warpsPerCTA
三元组如何把 16x16 张量切给 64 个线程。8x8 个"块"网格,每块=一个线程占的 2x2
连续元素(sizePerThread);左半 4 列块=warp0,右半 4 列块=warp1(warpsPerCTA=[1,2])。
线程号公式据 .td Example 1 的字面网格核验:
  左半(col_block<4): tid = row_block*4 + col_block
  右半(col_block>=4): tid = 32 + row_block*4 + (col_block-4)
全坐标由循环计算,零手写魔数;宽度按最长文本估算留够,不裁字。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

def text_w(s, size):
    cjk = sum(1 for ch in s if ord(ch) > 0x2e7f)
    other = len(s) - cjk
    return cjk * size * 1.0 + other * size * 0.56

TITLE = "BlockedEncoding 三元组把 16x16 张量切给 64 个线程"
SUBTITLE = "sizePerThread=[2,2]  threadsPerWarp=[8,4]  warpsPerCTA=[1,2]  (.td Example 1)"
N_BLOCK = 8          # 8x8 个"块",每块 = 一个线程的 2x2 连续元素
BLOCK = 46
PAD_LEFT, TOP = 70, 150

def tid(rb, cb):
    if cb < 4:
        return rb * 4 + cb
    return 32 + rb * 4 + (cb - 4)

HIGHLIGHT = {(0, 0): "thread0", (0, 1): "thread1", (1, 0): "thread4", (0, 4): "thread32"}

WARP0_FILL, WARP0_STROKE = "#dbeafe", "#1d4ed8"
WARP1_FILL, WARP1_STROKE = "#dcfce7", "#15803d"
HL_FILL, HL_STROKE = "#fde68a", "#b45309"

LEGEND = [(WARP0_FILL, WARP0_STROKE, "warp0 覆盖 (cols 0-7)"),
          (WARP1_FILL, WARP1_STROKE, "warp1 覆盖 (cols 8-15)"),
          (HL_FILL, HL_STROKE, "worked_example 标注的 4 个线程")]
NOTE_LINES = [
    "每块 = 一个线程占的 2x2 连续元素(sizePerThread),",
    "相邻线程沿列方向拿相邻块(T0->T1)——这正是",
    "访存合并的编码层前提(第 7 章硬件成因)。",
]
CAPTION_LINES = [
    "把 .td Example 1 的线程号网格图化:每块是一个线程的 2x2 连续元素,相邻线程 0,1,2,3 沿列拿相邻块;",
    "左半(cols 0-7)归 warp0(线程 0-31),右半(cols 8-15)归 warp1(线程 32-63)。",
]

grid_w = N_BLOCK * BLOCK
grid_h = N_BLOCK * BLOCK
right_col_w = max([text_w(lbl, 12) + 26 for _, _, lbl in LEGEND] +
                  [text_w(s, 12) for s in NOTE_LINES])
PAD_RIGHT = 46
W = int(PAD_LEFT + grid_w + 30 + right_col_w + PAD_RIGHT)
cap_w = max(text_w(s, 12) for s in CAPTION_LINES)
W = int(max(W, PAD_LEFT + cap_w + PAD_RIGHT, PAD_LEFT + text_w(TITLE, 17) + PAD_RIGHT))
H = TOP + grid_h + 120

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD_LEFT}" y="32" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD_LEFT}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for rb in range(N_BLOCK):
    for cb in range(N_BLOCK):
        x = PAD_LEFT + cb * BLOCK
        y = TOP + rb * BLOCK
        t = tid(rb, cb)
        key = (rb, cb)
        if key in HIGHLIGHT:
            fill, stroke, sw = HL_FILL, HL_STROKE, 2.5
        elif cb < 4:
            fill, stroke, sw = WARP0_FILL, WARP0_STROKE, 1
        else:
            fill, stroke, sw = WARP1_FILL, WARP1_STROKE, 1
        L.append(f'<rect x="{x}" y="{y}" width="{BLOCK-2}" height="{BLOCK-2}" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        # 块内 2x2 子格虚线,示意"这块是 sizePerThread=2x2 的连续元素"
        gx = x + (BLOCK - 2) / 2
        gy = y + (BLOCK - 2) / 2
        L.append(f'<line x1="{gx}" y1="{y}" x2="{gx}" y2="{y+BLOCK-2}" '
                  'stroke="#cbd5e1" stroke-width="0.8" stroke-dasharray="2,2"/>')
        L.append(f'<line x1="{x}" y1="{gy}" x2="{x+BLOCK-2}" y2="{gy}" '
                  'stroke="#cbd5e1" stroke-width="0.8" stroke-dasharray="2,2"/>')
        label = str(t) if key not in HIGHLIGHT else f"T{t}"
        weight = 'font-weight="bold" ' if key in HIGHLIGHT else ''
        fs = 13 if key in HIGHLIGHT else 10
        color = "#92400e" if key in HIGHLIGHT else "#475569"
        L.append(f'<text x="{x+(BLOCK-2)/2}" y="{y+(BLOCK-2)/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs}" {weight}'
                  f'fill="{color}">{esc(label)}</text>')

# warp 分界竖线(col_block=4 之前)
boundary_x = PAD_LEFT + 4 * BLOCK
L.append(f'<line x1="{boundary_x}" y1="{TOP-8}" x2="{boundary_x}" y2="{TOP+grid_h+8}" '
          'stroke="#0f172a" stroke-width="2.5"/>')

# warp 标注
L.append(f'<text x="{PAD_LEFT+2*BLOCK}" y="{TOP-38}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#1d4ed8">{esc("warp0 (线程 0-31)")}</text>')
L.append(f'<text x="{boundary_x+2*BLOCK}" y="{TOP-38}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#15803d">{esc("warp1 (线程 32-63)")}</text>')

# 行/列坐标轴标注(张量下标,每块跨 2 行/2 列)
for cb in range(N_BLOCK):
    x = PAD_LEFT + cb * BLOCK + (BLOCK - 2) / 2
    L.append(f'<text x="{x}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="9" fill="#94a3b8">{esc(f"c{cb*2}-{cb*2+1}")}</text>')
for rb in range(N_BLOCK):
    y = TOP + rb * BLOCK + (BLOCK - 2) / 2 + 3
    L.append(f'<text x="{PAD_LEFT-8}" y="{y}" text-anchor="end" font-family="sans-serif" '
              f'font-size="9" fill="#94a3b8">{esc(f"r{rb*2}-{rb*2+1}")}</text>')

# 图例
legend_x = PAD_LEFT + grid_w + 30
legend_y = TOP + 10
for i, (fill, stroke, label) in enumerate(LEGEND):
    ly = legend_y + i * 30
    L.append(f'<rect x="{legend_x}" y="{ly}" width="18" height="18" rx="3" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{legend_x+26}" y="{ly+14}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(label)}</text>')

note_y = legend_y + len(LEGEND) * 30 + 20
for i, line in enumerate(NOTE_LINES):
    L.append(f'<text x="{legend_x}" y="{note_y+i*18}" font-family="sans-serif" font-size="12" '
              f'fill="#475569">{esc(line)}</text>')

foot_y0 = H - 20 - (len(CAPTION_LINES) - 1) * 18
for i, line in enumerate(CAPTION_LINES):
    L.append(f'<text x="{PAD_LEFT}" y="{foot_y0+i*18}" font-family="sans-serif" font-size="12" '
              f'fill="#64748b">{esc(line)}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-blocked-triple.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
