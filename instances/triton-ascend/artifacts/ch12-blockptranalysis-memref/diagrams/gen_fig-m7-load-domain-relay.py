#!/usr/bin/env python3
"""fig-m7-load-domain-relay: LoadConverter 主干（无 mask）把 tt.load 落成
三步内存/计算域接力（tensor-flow 模板：flow 骨架 + 每条边标 shape/操作）。
字面量逐字取自 lit 夹具 legal_stride.mlir。全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "tt.load → memref.copy + to_tensor 的域接力（无 mask 主干）"
SUBTITLE = "三条 load 路径（无 mask/boundary_check/有 mask）最后都汇到 toTensorAndReplace 这条公共尾巴"

BOXES = [
    ("步①  内存域：规整窗口", "memref.reinterpret_cast\n→ memref<4x1xf32,\nstrided<[?,?],offset:?>>", "#dbeafe", "#2563eb", "#1e3a8a"),
    ("步②  内存域：本地缓冲", "%alloc = memref.alloc()\n: memref<4x1xf32>", "#fef3c7", "#d97706", "#78350f"),
    ("步③  计算域：认领为 tensor", "bufferization.to_tensor %alloc\nrestrict writable", "#dcfce7", "#16a34a", "#14532d"),
]

BOX_W, BOX_H, HGAP, PAD = 260, 100, 100, 40
TITLE_Y, SUBTITLE_Y = 26, 48
EDGE_LINES = 3
EDGE_LINE_H = 13
TOP = SUBTITLE_Y + 24 + EDGE_LINES * EDGE_LINE_H + 16
n = len(BOXES)
w = PAD * 2 + n * BOX_W + (n - 1) * HGAP

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} 340">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="340" fill="white"/>',
     f'<text x="{PAD}" y="{TITLE_Y}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{SUBTITLE_Y}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']
box_x = [PAD + i * (BOX_W + HGAP) for i in range(n)]

EDGE_LABELS = [
    "memref.copy\n(从窗口搬进本地缓冲)\nLoadStoreConverter.cpp:L441",
    "restrict+writable\n(告诉下游无别名)\nL86-L98 toTensorAndReplace",
]

for i, (label, code, fill, stroke, tf) in enumerate(BOXES):
    x = box_x[i]
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="{tf}">{esc(label)}</text>')
    lines = code.split("\n")
    n_l = len(lines)
    y0 = TOP + 46 - (n_l - 1) * 7
    for k, line in enumerate(lines):
        L.append(f'<text x="{x+BOX_W/2}" y="{y0+k*16}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="10.5" fill="{tf}">{esc(line)}</text>')
    if i < n - 1:
        ay = TOP + BOX_H / 2
        L.append(f'<line x1="{x+BOX_W}" y1="{ay}" x2="{box_x[i+1]}" y2="{ay}" '
                  'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
        elines = EDGE_LABELS[i].split("\n")
        ey0 = TOP - 12 - (len(elines) - 1) * EDGE_LINE_H
        for k, line in enumerate(elines):
            L.append(f'<text x="{(x+BOX_W+box_x[i+1])/2}" y="{ey0+k*13}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="9.5" fill="#94a3b8">{esc(line)}</text>')

foot_y = TOP + BOX_H + 40
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#0f172a">步①的 reinterpret_cast 与 fig-m3/fig-m5 是同一枚物化结果——本图接着讲它之后的内存/计算域接力</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="10.5" '
          f'fill="#64748b">达芬奇架构需要这种显式的"内存域→计算域"接力：tt.load 不是一步到位的 tensor 值</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-m7-load-domain-relay.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out}')
