#!/usr/bin/env python3
"""fig-m3-triple-to-recast: createCastOp 把 BlockData 三元组映射成
memref.reinterpret_cast 的三个参数槽（flow 模板）。offsets 先经
inferBlockOffset 塌缩成单一 offset，sizes/strides 逐维直接对位填入。
全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "createCastOp：三元组 → memref.reinterpret_cast（BlockPtrAnalysis.cpp:L322-L343）"
SUBTITLE = "offset 槽只有一个：多维偏移必须先塌缩成总和（8+3=11），这正是 reinterpret_cast『单 offset』语义的由来"

IN_BOXES = [
    ("offsets", "[8, 3]", "#93c5fd", "#1e3a8a", True),
    ("sizes", "[4, 2]", "#86efac", "#14532d", False),
    ("strides", "[2, 1]", "#f9a8d4", "#831843", False),
]

IN_W, IN_H, ROW_GAP, PAD, TOP = 200, 68, 40, 40, 116
n = len(IN_BOXES)
in_x = PAD
row_y = [TOP + i * (IN_H + ROW_GAP) for i in range(n)]

TRANS_X = in_x + IN_W + 130
TRANS_W, TRANS_H = 260, IN_H
trans_y = row_y[0]

FINAL_X = TRANS_X + TRANS_W + 130
FINAL_W = 340
FINAL_TOP = row_y[0] - 6
FINAL_H = row_y[2] + IN_H - FINAL_TOP + 6

w = FINAL_X + FINAL_W + PAD
h = row_y[-1] + IN_H + 96

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-8}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+14}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 输入三槽
for i, (name, val, fill, tf, feeds_trans) in enumerate(IN_BOXES):
    y = row_y[i]
    L.append(f'<rect x="{in_x}" y="{y}" width="{IN_W}" height="{IN_H}" rx="8" '
              f'fill="{fill}" stroke="#334155" stroke-width="1.5"/>')
    L.append(f'<text x="{in_x+IN_W/2}" y="{y+26}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="{tf}">{esc(name)}</text>')
    L.append(f'<text x="{in_x+IN_W/2}" y="{y+48}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" fill="{tf}">{esc(val)}</text>')

# offsets → 塌缩处理框
oy = row_y[0] + IN_H / 2
L.append(f'<line x1="{in_x+IN_W}" y1="{oy}" x2="{TRANS_X}" y2="{oy}" '
          'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<rect x="{TRANS_X}" y="{trans_y}" width="{TRANS_W}" height="{TRANS_H}" rx="8" '
          'fill="#fef3c7" stroke="#d97706" stroke-width="2"/>')
L.append(f'<text x="{TRANS_X+TRANS_W/2}" y="{trans_y+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" '
          f'fill="#78350f">inferBlockOffset 塌缩</text>')
L.append(f'<text x="{TRANS_X+TRANS_W/2}" y="{trans_y+42}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" fill="#78350f">0 +8 → 8 +3 → 11</text>')
L.append(f'<text x="{TRANS_X+TRANS_W/2}" y="{trans_y+60}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#92400e">L146-L151</text>')

# 塌缩结果 → 最终框
L.append(f'<line x1="{TRANS_X+TRANS_W}" y1="{oy}" x2="{FINAL_X}" y2="{FINAL_TOP+30}" '
          'stroke="#d97706" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{(TRANS_X+TRANS_W+FINAL_X)/2}" y="{oy-10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" '
          f'fill="#92400e">总 offset = 11</text>')

# sizes / strides → 最终框（直接对位，不经塌缩）
for i in (1, 2):
    y = row_y[i] + IN_H / 2
    target_y = FINAL_TOP + (30 if i == 1 else 60)
    L.append(f'<path d="M {in_x+IN_W} {y} L {TRANS_X+TRANS_W+40} {y} '
              f'L {FINAL_X} {target_y}" fill="none" stroke="#64748b" '
              'stroke-width="1.5" marker-end="url(#a)"/>')

# 最终框：发射结果
L.append(f'<rect x="{FINAL_X}" y="{FINAL_TOP}" width="{FINAL_W}" height="{FINAL_H}" rx="10" '
          'fill="#e0e7ff" stroke="#4338ca" stroke-width="2.5"/>')
L.append(f'<text x="{FINAL_X+FINAL_W/2}" y="{FINAL_TOP+26}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
          f'fill="#312e81">memref.reinterpret_cast</text>')
FIELDS = ["offset: [11]", "sizes: [4, 2]", "strides: [2, 1]"]
for i, f in enumerate(FIELDS):
    fy = FINAL_TOP + 52 + i * 22
    L.append(f'<text x="{FINAL_X+FINAL_W/2}" y="{fy}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="#3730a3">{esc(f)}</text>')
L.append(f'<text x="{FINAL_X+FINAL_W/2}" y="{FINAL_TOP+FINAL_H-14}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#4c1d95">L341-L342</text>')

foot_y = h - 20
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#0f172a">本例：2 维 block，1 条 reinterpret_cast 完整描述 4×2=8 个元素——O(1) 条指令描述 O(N) 数据</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m3-triple-to-recast.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
