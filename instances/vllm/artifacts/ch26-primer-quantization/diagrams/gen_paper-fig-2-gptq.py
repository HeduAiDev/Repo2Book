#!/usr/bin/env python3
"""paper-fig-2-gptq: 重绘自 arXiv:2210.17323 Figure 2 —— GPTQ 逐块量化的整体图景。
左：逆 Hessian 的 Cholesky 形式（阶梯状上三角结构，灰色=与当前/未来块无关，
蓝色=仍保留的二阶信息，粗框标出当前块对应的行带，白色行=正在处理的那一行）。
右：权重矩阵按连续列分块——米色/橙色=已量化列，白色=正在被量化的列，
深蓝/浅紫=块内待更新的剩余列。两图之间用双箭头呼应同一个"块 i"。
非像素复制，仅对齐原图的信息结构；配色套本书视觉语言，文字译中。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "GPTQ 逐块量化：块内白色列正在被量化，蓝色列是待更新的剩余权重"
SUBTITLE = "重绘自 arXiv:2210.17323 Figure 2 的信息结构（Cholesky 形式的逆 Hessian ←→ 权重矩阵分块）"

PAD = 40
TOP = 96

# ---------- left panel: staircase Hessian ----------
N = 10
CELL = 26
GRID = N * CELL  # 260
LEFT_X = PAD
LEFT_Y = TOP
BAND_ROWS = (3, 4)          # 当前块对应的行带（0-indexed，闭区间）
WHITE_ROW = 3               # 行带内“正在处理”的那一行

GRAY = "#94a3b8"
BLUE_LIGHT = "#bfdbfe"
BAND_STROKE = "#0f172a"

left_cells = []
for r in range(N):
    for c in range(N):
        x = LEFT_X + c * CELL
        y = LEFT_Y + r * CELL
        fill = BLUE_LIGHT if r <= c else GRAY
        if r == WHITE_ROW:
            fill = "#ffffff"
        left_cells.append((x, y, fill, r, c))

# ---------- right panel: weight matrix column bands ----------
COL_W = {"done": 68, "cur_done": 42, "white": 18, "cur_pending": 42, "rest": 88}
RIGHT_X = LEFT_X + GRID + 90
RIGHT_Y = TOP
RIGHT_H = GRID
cols = [
    ("done", COL_W["done"], "#fde3b8", "#c2820f", "已量化（历史列）"),
    ("cur_done", COL_W["cur_done"], "#f59e0b", "#b45309", "当前块内已量化"),
    ("white", COL_W["white"], "#ffffff", "#0f172a", "正在被量化"),
    ("cur_pending", COL_W["cur_pending"], "#4338ca", "#312e81", "当前块内待更新"),
    ("rest", COL_W["rest"], "#c7d2fe", "#4f46e5", "块内其余待更新列"),
]
RIGHT_W = sum(c[1] for c in cols)

w = RIGHT_X + RIGHT_W + PAD
h = TOP + GRID + 190

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-6}" font-family="sans-serif" font-size="16" '
     f'fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+16}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# left panel title
L.append(f'<text x="{LEFT_X}" y="{LEFT_Y-14}" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#0f172a">逆层 Hessian（Cholesky 形式）</text>')

for x, y, fill, r, c in left_cells:
    L.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" fill="{fill}" '
             f'stroke="#e2e8f0" stroke-width="0.5"/>')

# bold band around the current block's row range
band_y0 = LEFT_Y + BAND_ROWS[0] * CELL
band_y1 = LEFT_Y + (BAND_ROWS[1] + 1) * CELL
L.append(f'<rect x="{LEFT_X}" y="{band_y0}" width="{GRID}" height="{band_y1-band_y0}" '
         f'fill="none" stroke="{BAND_STROKE}" stroke-width="2.5"/>')
L.append(f'<rect x="{LEFT_X}" y="{LEFT_Y}" width="{GRID}" height="{GRID}" '
         f'fill="none" stroke="#334155" stroke-width="1.5"/>')
L.append(f'<text x="{LEFT_X}" y="{LEFT_Y+GRID+20}" font-family="sans-serif" font-size="11.5" '
         f'fill="#334155">初始一次性算好（computed initially）</text>')
L.append(f'<text x="{LEFT_X}" y="{LEFT_Y+GRID+38}" font-family="sans-serif" font-size="11.5" '
         f'fill="#334155">粗框 = 当前块对应的行带；白行 = 正被处理的那一行</text>')

# double arrow between panels
ay = LEFT_Y + GRID / 2
ax0 = LEFT_X + GRID + 14
ax1 = RIGHT_X - 14
L.append(f'<line x1="{ax0}" y1="{ay}" x2="{ax1}" y2="{ay}" stroke="#64748b" '
         f'stroke-width="2" marker-end="url(#a)" marker-start="url(#a)"/>')

# right panel title
L.append(f'<text x="{RIGHT_X}" y="{RIGHT_Y-14}" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#0f172a">权重矩阵 / 分块</text>')

cx = RIGHT_X
for key, cw, fill, stroke, label in cols:
    stroke_w = 2.5 if key == "white" else 1.2
    L.append(f'<rect x="{cx}" y="{RIGHT_Y}" width="{cw}" height="{RIGHT_H}" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}"/>')
    cx += cw

L.append(f'<rect x="{RIGHT_X}" y="{RIGHT_Y}" width="{RIGHT_W}" height="{RIGHT_H}" '
         f'fill="none" stroke="#334155" stroke-width="1.5"/>')
L.append(f'<text x="{RIGHT_X}" y="{RIGHT_Y+GRID+20}" font-family="sans-serif" font-size="11.5" '
         f'fill="#334155">块 i 在块内逐列递归量化（column-by-column）</text>')

# legend (纵向堆叠，避免右侧文字被画布边界裁切)
leg_y = RIGHT_Y + GRID + 56
leg_items = [
    ("#fde3b8", "#f59e0b", "已量化权重"),
    ("#4338ca", "#c7d2fe", "块内待更新的未量化权重"),
]
for i, (c1, c2, label) in enumerate(leg_items):
    ly = leg_y + i * 26
    L.append(f'<rect x="{RIGHT_X}" y="{ly}" width="16" height="16" fill="{c1}" '
             f'stroke="#334155" stroke-width="1"/>')
    L.append(f'<rect x="{RIGHT_X+16}" y="{ly}" width="16" height="16" fill="{c2}" '
             f'stroke="#334155" stroke-width="1"/>')
    L.append(f'<text x="{RIGHT_X+40}" y="{ly+13}" font-family="sans-serif" font-size="11.5" '
             f'fill="#334155">{esc(label)}</text>')

foot_y = leg_y + len(leg_items) * 26 + 16
foot_lines = [
    "对应正文 Algorithm 1：先对块内当前列取整量化（白→橙），再把该列的量化误差",
    "立刻摊到块内其余未量化列（蓝/浅紫）上做二阶补偿，逆 Hessian 只需算一次。",
]
for i, line in enumerate(foot_lines):
    L.append(f'<text x="{PAD}" y="{foot_y+i*18}" font-family="sans-serif" font-size="12" '
             f'fill="#334155">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-2-gptq.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
