#!/usr/bin/env python3
"""fig-m6-struct-vs-gather: MemAccType 分岔的两条落地路径对比（before-after
模板）。左=结构化（2 条 op，O(1)）；右=非结构化 gather 回退（N=4 时 12 条 op，O(N)）。
worked_example: idx_tensor=[10,3,7,1], base=100 → combinedOffset=[110,103,107,101]。
全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "结构化 vs 非结构化 gather 回退：同一块访存的两种落地路径"
SUBTITLE = "触发点：间接 load 派生地址 → UnstrucMemAcc（merge=max 全链染色）；循环上界 = blockSizes（本例 4）"

PANEL_W, PAD = 380, 40
GAP = 100
TITLE_Y, SUBTITLE_Y = 26, 50
TOP = SUBTITLE_Y + 56

w = PAD * 2 + PANEL_W * 2 + GAP
LEFT_X = PAD
RIGHT_X = PAD + PANEL_W + GAP

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} 520">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     '<rect width="{}" height="520" fill="white"/>'.format(w),
     f'<text x="{PAD}" y="{TITLE_Y}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{SUBTITLE_Y}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# ---------- 左面板：结构化 ----------
L.append(f'<text x="{LEFT_X+PANEL_W/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="#16a34a">结构化（StrucMemAcc）</text>')
step_h, step_gap = 56, 26
steps_left = [
    ("① memref.reinterpret_cast", "整块视图（本例 4×2=8 元素，1 条描述）"),
    ("② memref.copy", "搬整块（LoadStoreConverter.cpp:L441）"),
]
ly = TOP
for i, (label, detail) in enumerate(steps_left):
    L.append(f'<rect x="{LEFT_X}" y="{ly}" width="{PANEL_W}" height="{step_h}" rx="8" '
              'fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>')
    L.append(f'<text x="{LEFT_X+PANEL_W/2}" y="{ly+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" fill="#14532d">{esc(label)}</text>')
    L.append(f'<text x="{LEFT_X+PANEL_W/2}" y="{ly+42}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#14532d">{esc(detail)}</text>')
    if i < len(steps_left) - 1:
        L.append(f'<line x1="{LEFT_X+PANEL_W/2}" y1="{ly+step_h}" x2="{LEFT_X+PANEL_W/2}" '
                  f'y2="{ly+step_h+step_gap-4}" stroke="#16a34a" stroke-width="1.5" marker-end="url(#a)"/>')
    ly += step_h + step_gap

left_bottom = ly - step_gap + step_h
L.append(f'<rect x="{LEFT_X}" y="{left_bottom+30}" width="{PANEL_W}" height="44" rx="8" '
          'fill="#f0fdf4" stroke="#16a34a" stroke-width="1.5"/>')
L.append(f'<text x="{LEFT_X+PANEL_W/2}" y="{left_bottom+56}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
          f'fill="#14532d">共 2 条 op = O(1)（O(1) 描述 O(N)=8 个元素）</text>')

# ---------- 右面板：gather 回退 ----------
L.append(f'<text x="{RIGHT_X+PANEL_W/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="#dc2626">非结构化 gather 回退（rewriteAddPtrToUnstrucMemAcc）</text>')

TABLE_COLS = ["iv", "散乱 offset", "+base", "combinedOffset"]
ROWS = [("0", "10", "100", "110"), ("1", "3", "100", "103"),
        ("2", "7", "100", "107"), ("3", "1", "100", "101")]
tw = PANEL_W
col_w = [tw * 0.14, tw * 0.30, tw * 0.22, tw * 0.34]
col_x = [RIGHT_X]
for cw in col_w[:-1]:
    col_x.append(col_x[-1] + cw)

thead_h = 26
row_h = 24
ty = TOP
L.append(f'<rect x="{RIGHT_X}" y="{ty}" width="{tw}" height="{thead_h}" fill="#fecaca" '
          'stroke="#dc2626" stroke-width="1.5"/>')
for j, name in enumerate(TABLE_COLS):
    L.append(f'<text x="{col_x[j]+col_w[j]/2}" y="{ty+17}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
              f'fill="#7f1d1d">{esc(name)}</text>')
ty += thead_h
for r, row in enumerate(ROWS):
    ry = ty + r * row_h
    L.append(f'<rect x="{RIGHT_X}" y="{ry}" width="{tw}" height="{row_h}" fill="#fff1f2" '
              'stroke="#fca5a5" stroke-width="1"/>')
    for j, val in enumerate(row):
        L.append(f'<text x="{col_x[j]+col_w[j]/2}" y="{ry+16}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="#7f1d1d">{esc(val)}</text>')
table_bottom = ty + len(ROWS) * row_h

CAPTION_LINES = [
    "每轮：tensor.extract 散乱 offset → +base → combinedOffset",
    "→ reinterpret_cast（单元素 sizes:[1] strides:[1]）",
]
for k, line in enumerate(CAPTION_LINES):
    L.append(f'<text x="{RIGHT_X}" y="{table_bottom+18+k*15}" font-family="sans-serif" font-size="9.5" '
              f'fill="#7f1d1d">{esc(line)}</text>')

gather_box_y = table_bottom + 18 + len(CAPTION_LINES) * 15 + 10
L.append(f'<rect x="{RIGHT_X}" y="{gather_box_y}" width="{PANEL_W}" height="44" rx="8" '
          'fill="#fef2f2" stroke="#dc2626" stroke-width="1.5"/>')
L.append(f'<text x="{RIGHT_X+PANEL_W/2}" y="{gather_box_y+26}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
          f'fill="#7f1d1d">4 ×(extract+recast+load) = 12 条 op = O(N)</text>')

# 中间对比箭头
mid_y = TOP + 40
L.append(f'<line x1="{LEFT_X+PANEL_W+8}" y1="{mid_y}" x2="{RIGHT_X-8}" y2="{mid_y}" '
          'stroke="#94a3b8" stroke-width="2" marker-end="url(#a)" stroke-dasharray="6,4"/>')
L.append(f'<text x="{(LEFT_X+PANEL_W+RIGHT_X)/2}" y="{mid_y-10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10" fill="#94a3b8">同一块访存</text>')

foot_y = 500
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#0f172a">2 vs 12 条 op：正确性优先于性能——一处间接 load 就把整链从『1 条 copy 搬整块』拖成『N 次单元素循环』</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-m6-struct-vs-gather.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out}')
