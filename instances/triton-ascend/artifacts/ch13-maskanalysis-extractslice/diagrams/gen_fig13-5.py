#!/usr/bin/env python3
"""fig13-5 tensor-flow 模板:解析出的矩形(offsets,dims)配全 1 strides,分两路发射——
tensor 域走 getExtractSlice→tensor.extract_slice,memref 域走 getSubview→memref.subview。
数据取自 explainer m8.figure_specs.numbers。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

SRC_BOX = ("矩形 offsets=[0,0] dims=[10,12]", "strides 全 1 [1,1]")
LEFT = ("getExtractSlice", "MaskAnalysis.cpp:L133-L143",
        "tensor.extract_slice", "源 tensor<16x16> → 结果 10x12")
RIGHT = ("getSubview", "MaskAnalysis.cpp:L180-L195",
         "memref.subview", "源 rank 不足处尾部补 offset0/dim1")

BOX_W, BOX_H = 260, 60
PAD, TOP = 60, 100
ROW_GAP = 90
COL_GAP = 80

w = PAD * 2 + BOX_W * 2 + COL_GAP
h = TOP + BOX_H * 2 + ROW_GAP + 110

src_x = PAD + (BOX_W * 2 + COL_GAP) / 2 - BOX_W / 2
src_y = TOP
left_x = PAD
right_x = PAD + BOX_W + COL_GAP
dst_y = TOP + BOX_H + ROW_GAP

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="{38}" text-anchor="middle" font-family="sans-serif" '
     f'font-size="16" font-weight="bold" fill="#0f172a">'
     f'{esc("矩形掩码的两个发射器:同一份边界,tensor / memref 两种落点")}</text>']

# source box
L.append(f'<rect x="{src_x}" y="{src_y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
          f'fill="#dbeafe" stroke="#2563eb" stroke-width="2.5"/>')
L.append(f'<text x="{src_x+BOX_W/2}" y="{src_y+24}" text-anchor="middle" '
          f'font-family="monospace" font-size="12" fill="#1e3a8a">{esc(SRC_BOX[0])}</text>')
L.append(f'<text x="{src_x+BOX_W/2}" y="{src_y+43}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#1d4ed8">{esc(SRC_BOX[1])}</text>')

def draw_branch(x, label):
    fn, anchor, opname, note = label
    L.append(f'<rect x="{x}" y="{dst_y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="#ecfdf5" stroke="#059669" stroke-width="2"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{dst_y+18}" text-anchor="middle" '
              f'font-family="monospace" font-size="12" font-weight="bold" '
              f'fill="#064e3b">{esc(fn)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{dst_y+34}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#047857">{esc(anchor)}</text>')
    L.append(f'<rect x="{x+18}" y="{dst_y+BOX_H+14}" width="{BOX_W-36}" height="26" rx="5" '
              f'fill="#134e4a" />')
    L.append(f'<text x="{x+BOX_W/2}" y="{dst_y+BOX_H+31}" text-anchor="middle" '
              f'font-family="monospace" font-size="12" fill="white">{esc(opname)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{dst_y+BOX_H+58}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#334155">{esc(note)}</text>')

draw_branch(left_x, LEFT)
draw_branch(right_x, RIGHT)

# arrows source -> each branch
src_bottom_y = src_y + BOX_H
sxl, sxr = src_x + BOX_W * 0.25, src_x + BOX_W * 0.75
lx, rx = left_x + BOX_W / 2, right_x + BOX_W / 2
L.append(f'<path d="M{sxl},{src_bottom_y} C{sxl-40},{(src_bottom_y+dst_y)/2} '
          f'{lx+20},{(src_bottom_y+dst_y)/2} {lx},{dst_y}" '
          f'fill="none" stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(sxl+lx)/2-50}" y="{(src_bottom_y+dst_y)/2}" font-family="sans-serif" '
          f'font-size="11" fill="#334155">{esc("tensor 域")}</text>')
L.append(f'<path d="M{sxr},{src_bottom_y} C{sxr+40},{(src_bottom_y+dst_y)/2} '
          f'{rx-20},{(src_bottom_y+dst_y)/2} {rx},{dst_y}" '
          f'fill="none" stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(sxr+rx)/2+10}" y="{(src_bottom_y+dst_y)/2}" font-family="sans-serif" '
          f'font-size="11" fill="#334155">{esc("memref 域")}</text>')

foot_y = h - 26
L.append(f'<text x="{w/2}" y="{foot_y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#334155">'
          f'{esc("核心洞察:结构化世界用一次「切片」表达边界,不是 GPU 逐元素的 predication。")}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig13-5.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
