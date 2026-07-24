#!/usr/bin/env python3
"""flow 模板(自定义,竖排管线):同一个真核的六层剖面,自上而下下降。
每层一个盒子:层名 + 真核锚点(代码行/文档行)+ 呼应章号,层间下箭头,末层加对位基座旁注。
全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "全链回望 — 06-fused-attention.py 一个真核串起六层下降"
SUBTITLE = "同一段 365 行代码,从语言层显式搬运一路落到 AscendC 库调用;六层不是六个例子,是同一个真核的六个剖面"

LAYERS = [  # (层名, 锚点, 章号)
    ("语言层显式搬运", "make_block_ptr 六元组 + load/store/advance · L175-L206", "ch03-08, ch12"),
    ("ttadapter 结构化下降", "q = tl.load(Q_block_ptr) 经 BlockPtrAnalysis 物化 · L226", "ch10-14"),
    ("核亲和双核分工", "两处 tl.dot 落 Cube,softmax 落 Vector · L90, L112", "ch16"),
    ("HFusion 融合", "cube→vector→cube 内循环段融成 ShallowCV/MixCV · L86-L120", "ch21-22"),
    ("HIVM 下降", "hivm.tile_mix_cube_num 提示 + 显式内存层级/同步 · best_practice.md:L896", "ch23-25"),
    ("AscendC 库调用", "整条下降链的终点(6 层剖面到此落地)", "ch25"),
]
PAIR_NOTE = "对位基座 Triton ch43:同一个 fused-attention 从 tl.* 一路到 NVIDIA PTX——这里终点是 AscendC 而非 PTX"

BOX_W, BOX_H, VGAP, PAD, TOP = 760, 78, 34, 60, 130
n = len(LAYERS)
w = PAD * 2 + BOX_W + 260
h = TOP + n * (BOX_H + VGAP) + 120

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-14}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+6}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

box_x = PAD
centers = []
for i, (name, anchor, chs) in enumerate(LAYERS):
    y = TOP + i * (BOX_H + VGAP)
    last = (i == n - 1)
    fill = "#f97316" if last else "#3b82f6"
    stroke = "#c2410c" if last else "#1d4ed8"
    L.append(f'<rect x="{box_x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" opacity="0.12" stroke="{stroke}" stroke-width="2"/>')
    # 层号圆徽
    r = 20
    L.append(f'<circle cx="{box_x+r+10}" cy="{y+BOX_H/2}" r="{r}" fill="{stroke}"/>')
    L.append(f'<text x="{box_x+r+10}" y="{y+BOX_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="15" font-weight="bold" '
              f'fill="white">{i+1}</text>')
    tx = box_x + 2 * r + 30
    L.append(f'<text x="{tx}" y="{y+26}" font-family="sans-serif" font-size="14.5" '
              f'font-weight="bold" fill="{stroke}">{esc(name)}</text>')
    L.append(f'<text x="{tx}" y="{y+46}" font-family="monospace" font-size="11" '
              f'fill="#334155">{esc(anchor)}</text>')
    # 章号胶囊贴右边
    chip_w = max(70, int(len(chs) * 6.6) + 20)
    chip_x = box_x + BOX_W - chip_w - 14
    L.append(f'<rect x="{chip_x}" y="{y+BOX_H-30}" width="{chip_w}" height="20" rx="10" '
              f'fill="white" stroke="{stroke}" stroke-width="1.3"/>')
    L.append(f'<text x="{chip_x+chip_w/2}" y="{y+BOX_H-16}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
              f'fill="{stroke}">{esc(chs)}</text>')
    centers.append((box_x + BOX_W / 2, y))

for i in range(n - 1):
    x, y = centers[i]
    y_bottom = y + BOX_H
    y_top_next = centers[i + 1][1]
    L.append(f'<line x1="{x}" y1="{y_bottom}" x2="{x}" y2="{y_top_next}" '
              f'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')

# 右侧竖排大括号 + "六层剖面,一路到底"
brace_x = box_x + BOX_W + 26
brace_top = TOP
brace_bot = TOP + n * (BOX_H + VGAP) - VGAP
L.append(f'<path d="M{brace_x},{brace_top} q14,0 14,{(brace_bot-brace_top)/2-14} '
          f'q0,14 14,14 q-14,0 -14,14 q0,{(brace_bot-brace_top)/2-14} -14,{(brace_bot-brace_top)/2-14}" '
          f'fill="none" stroke="#7c3aed" stroke-width="2"/>')
mid_y = (brace_top + brace_bot) / 2
L.append(f'<text x="{brace_x+40}" y="{mid_y-8}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#7c3aed">6 层剖面</text>')
L.append(f'<text x="{brace_x+40}" y="{mid_y+12}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#7c3aed">一路到底</text>')

# 底部对位基座旁注
note_y = TOP + n * (BOX_H + VGAP) + 10
L.append(f'<rect x="{box_x}" y="{note_y}" width="{BOX_W}" height="40" rx="8" '
          f'fill="#fff7ed" stroke="#c2410c" stroke-width="1.3"/>')
L.append(f'<text x="{box_x+16}" y="{note_y+25}" font-family="sans-serif" font-size="12" '
          f'fill="#9a3412">{esc(PAIR_NOTE)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-full-descent-chain.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
