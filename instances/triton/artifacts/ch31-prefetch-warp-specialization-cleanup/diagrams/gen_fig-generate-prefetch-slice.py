#!/usr/bin/env python3
"""fig-generate-prefetch-slice: tensor-flow 模板——generatePrefetch 把一个共享内存操作数
沿 K 维切出 prefetchWidth 宽的 MemDescSubview,再 LocalLoad 成带 DotOperandEncoding 的
寄存器片。opIdx=0(A)切列(kIdx=1)、opIdx=1(B)切行(kIdx=0)。
数字来源见 explainer.json mechanism prefetch-slice-generate。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "generatePrefetch —— 沿 K 维切片:MemDescSubview(视图) + LocalLoad(真搬运)"

ROWS = [
    # (标题, 源 shape, kIdx, 切法, subview shape, local_load 目标)
    ("A  opIdx=0", "shared [128, 64]", "kIdx=1(切列)", "subview [128, 16]",
     "LocalLoad → register [128, 16]\nDotOperandEncoding(opIdx=0)"),
    ("B  opIdx=1", "shared [64, 128]", "kIdx=0(切行)", "subview [16, 128]",
     "LocalLoad → register [16, 128]\nDotOperandEncoding(opIdx=1)"),
]

COL_W = [190, 170, 150, 170, 260]
GAP = 26
ROW_H, VGAP, TOP, PAD = 74, 46, 96, 40
w = PAD * 2 + sum(COL_W) + GAP * (len(COL_W) - 1)
h = TOP + len(ROWS) * (ROW_H + VGAP) + 130

def col_x(i):
    return PAD + sum(COL_W[:i]) + GAP * i

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="16.5" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>']

HEADERS = ["操作数", "MemDescSubview 源", "K 维取法", "subview(零成本视图)", "LocalLoad 后的寄存器片"]
for i, htext in enumerate(HEADERS):
    cx = col_x(i) + COL_W[i] / 2
    L.append(f'<text x="{cx}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="#64748b">{esc(htext)}</text>')

for r, (opname, src_shape, kidx, sub_shape, dst) in enumerate(ROWS):
    y = TOP + r * (ROW_H + VGAP)
    cy = y + ROW_H / 2
    # 操作数名
    x0 = col_x(0)
    L.append(f'<rect x="{x0}" y="{y}" width="{COL_W[0]}" height="{ROW_H}" rx="8" '
              f'fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>')
    L.append(f'<text x="{x0+COL_W[0]/2}" y="{cy+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" '
              f'fill="#3730a3">{esc(opname)}</text>')
    # 源 shape
    x1 = col_x(1)
    L.append(f'<rect x="{x1}" y="{y}" width="{COL_W[1]}" height="{ROW_H}" rx="8" '
              f'fill="#e2e8f0" stroke="#64748b"/>')
    L.append(f'<text x="{x1+COL_W[1]/2}" y="{cy+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="#0f172a">{esc(src_shape)}</text>')
    # k 取法
    x2 = col_x(2)
    L.append(f'<rect x="{x2}" y="{y}" width="{COL_W[2]}" height="{ROW_H}" rx="8" '
              f'fill="#fef9c3" stroke="#ca8a04"/>')
    L.append(f'<text x="{x2+COL_W[2]/2}" y="{cy+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="#78350f">{esc(kidx)}</text>')
    # subview shape
    x3 = col_x(3)
    L.append(f'<rect x="{x3}" y="{y}" width="{COL_W[3]}" height="{ROW_H}" rx="8" '
              f'fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
    L.append(f'<text x="{x3+COL_W[3]/2}" y="{cy+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#92400e">{esc(sub_shape)}</text>')
    # local_load 目标(两行)
    x4 = col_x(4)
    L.append(f'<rect x="{x4}" y="{y}" width="{COL_W[4]}" height="{ROW_H}" rx="8" '
              f'fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>')
    for li, line in enumerate(dst.split("\n")):
        L.append(f'<text x="{x4+COL_W[4]/2}" y="{cy-6+li*17}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="#14532d">{esc(line)}</text>')
    # 箭头(每两列之间)
    for i in range(4):
        xA = col_x(i) + COL_W[i]
        xB = col_x(i + 1)
        L.append(f'<line x1="{xA}" y1="{cy}" x2="{xB-2}" y2="{cy}" '
                  'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')

# 底部数字条
NUMS = [
    ("prefetchWidth", "16"),
    ("A 片 [128, 16]", "opIdx=0 · kIdx=1(列)"),
    ("B 片 [16, 128]", "opIdx=1 · kIdx=0(行)"),
    ("K 总宽 BLOCK_K", "64"),
]
ny = h - 108
L.append(f'<rect x="{PAD}" y="{ny}" width="{w-2*PAD}" height="56" rx="8" '
          'fill="#eff6ff" stroke="#93c5fd"/>')
seg_w = (w - 2*PAD) / len(NUMS)
for i, (label, val) in enumerate(NUMS):
    cx = PAD + seg_w * i + seg_w / 2
    L.append(f'<text x="{cx}" y="{ny+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" fill="#1e40af">{esc(val)}</text>')
    L.append(f'<text x="{cx}" y="{ny+40}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#334155">{esc(label)}</text>')

CAPTION_LINES = [
    "『切一片』= MemDescSubviewOp(在 shared 上开视图,零成本)+ LocalLoadOp(真搬到寄存器,附 DotOperandEncoding)。",
    "A 沿列、B 沿行各切 16 宽,恰好喂给一次 tensor-core MMA 指令的 K 步长。",
]
for i, line in enumerate(CAPTION_LINES):
    L.append(f'<text x="{w/2}" y="{h-24+i*17}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11.5" fill="#475569">{esc(line)}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-generate-prefetch-slice.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
