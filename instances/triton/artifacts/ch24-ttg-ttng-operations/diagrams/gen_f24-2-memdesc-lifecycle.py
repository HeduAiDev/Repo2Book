#!/usr/bin/env python3
"""f24-2-memdesc-lifecycle: state-machine 模板。
主线:local_alloc -> local_store/async_copy -> local_load -> 收尾态分叉(显式 dealloc / 自动回收)。
底部信息条给 memdesc 的两个具体类型 + 5 参数 + 自动回收规则(numbers 全覆盖)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

CHAIN = ["local_alloc", "local_store /\nasync_copy_global_to_local", "local_load"]
CHAIN_LBL = ["产 memdesc 句柄", "写入共享内存"]
END = [("local_load", "local_dealloc", "显式退还"),
       ("local_load", "自动回收", "支配所有 use 的\n首个 post-dom 点")]
BOX_W, BOX_H, HGAP, PAD, TOP, SIDE_DY = 210, 56, 120, 50, 110, 130

n = len(CHAIN)
END_SPREAD = 0.62  # 分叉两态相对锚点的横向偏移系数(乘 BOX_W)
chain_w = PAD * 2 + n * BOX_W + (n - 1) * HGAP
END_X0 = PAD + (n - 1) * (BOX_W + HGAP)  # local_load 左边缘(先按 chain 布局算一次)
right_extent = END_X0 + BOX_W * END_SPREAD + BOX_W + 260  # 右侧分支框 + 标签所需宽度
w = max(chain_w, right_extent)
h = TOP + BOX_H + SIDE_DY + BOX_H + 190

X = {}
for i, s in enumerate(CHAIN):
    x = PAD + i * (BOX_W + HGAP)
    # 多行节点名居中占位:换行数影响不了坐标计算,坐标仍按单节点框走
    X[s] = (x, TOP)
END_X0 = X[CHAIN[-1]][0]
X[END[0][1]] = (END_X0 - BOX_W * END_SPREAD, TOP + BOX_H + SIDE_DY)
X[END[1][1]] = (END_X0 + BOX_W * END_SPREAD, TOP + BOX_H + SIDE_DY)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

L.append(f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" '
         f'font-size="17" font-weight="bold" fill="#0f172a">memdesc：共享内存的 SSA 句柄一生</text>')

def draw_box(name, x, y, fill="#e0f2fe", stroke="#0369a1", txtfill="#0c4a6e"):
    lines = name.split("\n")
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="20" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    if len(lines) == 1:
        L.append(f'<text x="{x+BOX_W/2}" y="{y+BOX_H/2+5}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="13" font-weight="bold" '
                 f'fill="{txtfill}">{esc(lines[0])}</text>')
    else:
        base = y + BOX_H/2 - (len(lines)-1)*8 + 5
        for li, line in enumerate(lines):
            L.append(f'<text x="{x+BOX_W/2}" y="{base+li*16}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="12" font-weight="bold" '
                     f'fill="{txtfill}">{esc(line)}</text>')

for name in CHAIN:
    x, y = X[name]
    draw_box(name, x, y)

for i in range(len(CHAIN) - 1):
    (x1, y1) = X[CHAIN[i]]
    (x2, y2) = X[CHAIN[i + 1]]
    ay = y1 + BOX_H / 2
    L.append(f'<line x1="{x1+BOX_W}" y1="{ay}" x2="{x2}" y2="{ay}" '
             'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
    L.append(f'<text x="{(x1+BOX_W+x2)/2}" y="{ay-10}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="11" fill="#334155">{esc(CHAIN_LBL[i])}</text>')

# 收尾态分叉:local_load -> {local_dealloc, 自动回收}
anchor_name = END[0][0]
(ax, ay) = X[anchor_name]
for anchor, name, lbl in END:
    fill = "#fef3c7" if name == "local_dealloc" else "#dcfce7"
    stroke = "#d97706" if name == "local_dealloc" else "#16a34a"
    txtfill = "#78350f" if name == "local_dealloc" else "#14532d"
    (sx, sy) = X[name]
    draw_box(name, sx, sy, fill=fill, stroke=stroke, txtfill=txtfill)
    fx = ax + BOX_W / 2
    fy = ay + BOX_H
    tx = sx + BOX_W / 2
    ty = sy
    L.append(f'<line x1="{fx}" y1="{fy}" x2="{tx}" y2="{ty}" '
             'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
    midx, midy = (fx + tx) / 2, (fy + ty) / 2
    dx = -70 if name == "local_dealloc" else 70
    for li, line in enumerate(lbl.split("\n")):
        L.append(f'<text x="{midx+dx}" y="{midy+li*14-4}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="11" fill="#334155">{esc(line)}</text>')

# 底部信息条(numbers 全覆盖)
info_top = TOP + BOX_H + SIDE_DY + BOX_H + 24
info_lines = [
    "local_alloc 落地的 memdesc 类型：!tt.memdesc<64x32xf16, #shared, #triton_gpu.shared_memory>",
    "local_load 读回的分布式张量：tensor<64x32xf16, #dot_op<{opIdx=0, parent=#mma, kWidth=2}>>",
    "memdesc 类型的 5 个参数：shape, elementType, encoding, memorySpace, mutable_memory",
]
info_h = 24 + len(info_lines) * 22 + 10
L.append(f'<rect x="{PAD}" y="{info_top}" width="{w-2*PAD}" height="{info_h}" rx="8" '
         'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1"/>')
for i, line in enumerate(info_lines):
    L.append(f'<text x="{PAD+18}" y="{info_top+24+i*22}" font-family="sans-serif" '
             f'font-size="11.5" fill="#334155">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("f24-2-memdesc-lifecycle.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
