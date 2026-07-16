#!/usr/bin/env python3
"""before-after 模板:permute 只换 stride 视角,不搬数据。上方共享同一条线性内存
(6 个元素 a0..a5,行主序存储);左侧叠一个 (2,3) 视图,右侧叠一个 permute(1,0)
后的 (3,2) 视图——每个元素用同一颜色连回同一块内存格,证明内存原地不动。
数字全部来自 dossier m4-permute-stride-view。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "permute = 换 stride 视角，内存原地不动"
SUBTITLE = "输入形状 (2,3) → permute(1,0)(校验 sorted((1,0))==[0,1] 是排列) → 输出形状 (3,2)；落地算子 create_trans"

ELEMS = ["a0", "a1", "a2", "a3", "a4", "a5"]
COLORS = ["#fca5a5", "#fdba74", "#fde047", "#86efac", "#93c5fd", "#c4b5fd"]

MEM_CELL, MEM_GAP, PAD = 78, 6, 40
mem_w = len(ELEMS) * (MEM_CELL + MEM_GAP) - MEM_GAP
TOP_GRID = 80
GRID_H = 130
MEM_Y = TOP_GRID + GRID_H + 90
w = PAD * 2 + max(mem_w, 900)
h = MEM_Y + 120

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-8}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+14}" font-family="sans-serif" font-size="12.5" '
     f'fill="#475569">{esc(SUBTITLE)}</text>']

# ---- 左视图: (2,3) 行主序 ----
CELL = 66
left_w = 3 * CELL
left_x = PAD + 40
L.append(f'<text x="{left_x+left_w/2}" y="{TOP_GRID-8}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13" font-weight="bold" '
          f'fill="#0f172a">{esc("原视图 shape=(2,3)")}</text>')
left_centers = {}
for r in range(2):
    for c in range(3):
        idx = r * 3 + c
        x = left_x + c * CELL
        y = TOP_GRID + r * CELL
        L.append(f'<rect x="{x}" y="{y}" width="{CELL-4}" height="{CELL-4}" rx="6" '
                  f'fill="{COLORS[idx]}" stroke="#334155" stroke-width="1.3"/>')
        L.append(f'<text x="{x+(CELL-4)/2}" y="{y+(CELL-4)/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="14" font-weight="bold" '
                  f'fill="#1e293b">{esc(ELEMS[idx])}</text>')
        left_centers[idx] = (x + (CELL - 4) / 2, y + (CELL - 4))

# ---- 右视图: (3,2) permute(1,0) ----
right_w = 2 * CELL
right_x = w - PAD - 40 - right_w
L.append(f'<text x="{right_x+right_w/2}" y="{TOP_GRID-8}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13" font-weight="bold" '
          f'fill="#0f172a">{esc("permute(1,0) 后 shape=(3,2)")}</text>')
right_centers = {}
for r in range(3):
    for c in range(2):
        # 转置:new[r][c] = old[c][r] -> idx = c*3 + r
        idx = c * 3 + r
        x = right_x + c * CELL
        y = TOP_GRID + r * CELL
        L.append(f'<rect x="{x}" y="{y}" width="{CELL-4}" height="{CELL-4}" rx="6" '
                  f'fill="{COLORS[idx]}" stroke="#334155" stroke-width="1.3"/>')
        L.append(f'<text x="{x+(CELL-4)/2}" y="{y+(CELL-4)/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="14" font-weight="bold" '
                  f'fill="#1e293b">{esc(ELEMS[idx])}</text>')
        right_centers[idx] = (x + (CELL - 4) / 2, y + (CELL - 4))

# 中间箭头 + create_trans 标注
mid_y = TOP_GRID + GRID_H / 2 - 20
ax1 = left_x + left_w + 14
ax2 = right_x - 14
L.append(f'<line x1="{ax1}" y1="{mid_y}" x2="{ax2}" y2="{mid_y}" '
          'stroke="#0f172a" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{mid_y-10}" text-anchor="middle" font-family="monospace" '
          f'font-size="12.5" font-weight="bold" fill="#b45309">create_trans(dims=(1,0))</text>')

# ---- 共享内存条 ----
mem_x0 = PAD + (w - PAD * 2 - mem_w) / 2
mem_title_text = (f'<text x="{mem_x0}" y="{MEM_Y-14}" font-family="sans-serif" font-size="13" '
                   f'font-weight="bold" fill="#0f172a">{esc("共享的同一条线性内存(行主序存储,未被搬动)")}</text>')
mem_centers = {}
for i, e in enumerate(ELEMS):
    x = mem_x0 + i * (MEM_CELL + MEM_GAP)
    L.append(f'<rect x="{x}" y="{MEM_Y}" width="{MEM_CELL}" height="52" rx="6" '
              f'fill="{COLORS[i]}" stroke="#334155" stroke-width="1.3"/>')
    L.append(f'<text x="{x+MEM_CELL/2}" y="{MEM_Y+22}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#1e293b">{esc(e)}</text>')
    L.append(f'<text x="{x+MEM_CELL/2}" y="{MEM_Y+40}" text-anchor="middle" '
              f'font-family="monospace" font-size="10" fill="#475569">offset {i}</text>')
    mem_centers[i] = (x + MEM_CELL / 2, MEM_Y)

# 从左视图底部连到内存(虚线细,只画代表性的 a0 与 a5 两条,避免拥挤)
for idx in [0, 5]:
    (lx, ly) = left_centers[idx]
    (mx, my) = mem_centers[idx]
    L.append(f'<path d="M {lx} {ly} C {lx} {ly+40}, {mx} {my-40}, {mx} {my}" '
              f'fill="none" stroke="{COLORS[idx]}" stroke-width="2" stroke-dasharray="3,3"/>')
for idx in [0, 5]:
    (rx, ry) = right_centers[idx]
    (mx, my) = mem_centers[idx]
    L.append(f'<path d="M {rx} {ry} C {rx} {ry+40}, {mx} {my-40}, {mx} {my}" '
              f'fill="none" stroke="{COLORS[idx]}" stroke-width="2" stroke-dasharray="3,3"/>')

# 内存条标题最后画,盖在连接线之上,避免斜线穿字
L.append(mem_title_text)

foot_y0 = h - 24
L.append(f'<text x="{PAD}" y="{foot_y0}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">结论:两个视图共享下方同一条内存(a0..a5 位置不变),permute 只是换了'
          f'"先按行还是先按列"读取的 stride 视角——转置形状变了,数据没有搬。</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch07-permute-stride.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
