#!/usr/bin/env python3
"""before-after 模板(自定义):SliceEncoding 挤掉 parent 的一维得 1D 布局,
expand_dims 把该维顶回 size-1 还原 parent——二者互逆。上下两组分别演示挤 M 方向
(dim=1)与挤 N 方向(dim=0),数据取自 matmul 里 arange->expand_dims 的真实 IR。
全坐标计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

def text_w(s, size):
    cjk = sum(1 for ch in s if ord(ch) > 0x2e7f)
    other = len(s) - cjk
    return cjk * size * 1.0 + other * size * 0.56

TITLE = "SliceEncoding 是 expand_dims 的逆:挤掉一维 <-> 顶回 size-1"
SUBTITLE = "matmul 里 tt.make_range -> tt.expand_dims 的真实 IR(Triton v3.2.0 实测)"

ROWS = [
    {
        "label": "挤 M 方向(dim=1)",
        "parent_shape": "tensor<64x1xi32, #blocked>",
        "parent_rank": "rank=2",
        "slice_shape": "tensor<64xi32, slice<dim=1,parent=#blocked>>",
        "slice_rank": "rank=1",
        "squeeze_lbl": "squeeze(dim=1)  沿 dim1 挤掉一维",
        "expand_lbl": "expand_dims(axis=1)  顶回 size-1",
    },
    {
        "label": "挤 N 方向(dim=0)",
        "parent_shape": "tensor<1x64xi32, #blocked>",
        "parent_rank": "rank=2",
        "slice_shape": "tensor<64xi32, slice<dim=0,parent=#blocked>>",
        "slice_rank": "rank=1",
        "squeeze_lbl": "squeeze(dim=0)  沿 dim0 挤掉一维",
        "expand_lbl": "expand_dims(axis=0)  顶回 size-1",
    },
]

PAD = 46
BOX_W, BOX_H = 300, 74
RBOX_W = 340
MID_GAP = 220
ROW_GAP = 160
TOP = 110

W = int(PAD * 2 + BOX_W + MID_GAP + RBOX_W)
title_w = text_w(TITLE, 17)
W = int(max(W, PAD * 2 + title_w))

CAPTION_LINES = [
    "上:parent 2D #blocked 沿 dim=1 squeeze -> 1D slice<dim=1,parent=#blocked>;expand_dims(axis=1) 逐位还原 parent。",
    "下:换挤 dim=0,同样互逆。matmul 里 arange -> expand_dims 这一对就是 slice 作 expand_dims 逆的真实现场。",
]
cap_w = max(text_w(s, 12) for s in CAPTION_LINES)
W = int(max(W, PAD + cap_w + PAD))

H = int(TOP + len(ROWS) * ROW_GAP + len(CAPTION_LINES) * 18 + 50)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="fwd" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#1d4ed8"/></marker>'
     '<marker id="rev" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="52" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

left_x = PAD
right_x = PAD + BOX_W + MID_GAP

for ri, row in enumerate(ROWS):
    y = TOP + ri * ROW_GAP
    L.append(f'<text x="{left_x}" y="{y-12}" font-family="sans-serif" font-size="13" '
              f'font-weight="bold" fill="#334155">{esc(row["label"])}</text>')
    # 左:parent(2D)
    L.append(f'<rect x="{left_x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              'fill="#dbeafe" stroke="#1d4ed8" stroke-width="2"/>')
    L.append(f'<text x="{left_x+BOX_W/2}" y="{y+28}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" fill="#1e3a5f">{esc(row["parent_shape"])}</text>')
    L.append(f'<text x="{left_x+BOX_W/2}" y="{y+50}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#1d4ed8">{esc("parent (#blocked), " + row["parent_rank"])}</text>')
    # 右:slice(1D)——文本较长,拆两行(逗号处换行)
    slice_head, slice_tail = row["slice_shape"].split(", ", 1)
    L.append(f'<rect x="{right_x}" y="{y}" width="{RBOX_W}" height="{BOX_H}" rx="8" '
              'fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
    L.append(f'<text x="{right_x+RBOX_W/2}" y="{y+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" fill="#7c4a03">{esc(slice_head + ",")}</text>')
    L.append(f'<text x="{right_x+RBOX_W/2}" y="{y+38}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" fill="#7c4a03">{esc(slice_tail)}</text>')
    L.append(f'<text x="{right_x+RBOX_W/2}" y="{y+58}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#b45309">{esc("slice, " + row["slice_rank"])}</text>')
    # 上箭头:squeeze(parent -> slice)
    fwd_y = y + BOX_H * 0.32
    L.append(f'<line x1="{left_x+BOX_W+6}" y1="{fwd_y}" x2="{right_x-6}" y2="{fwd_y}" '
              'stroke="#1d4ed8" stroke-width="2" marker-end="url(#fwd)"/>')
    L.append(f'<text x="{(left_x+BOX_W+right_x)/2}" y="{fwd_y-8}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#1d4ed8">{esc(row["squeeze_lbl"])}</text>')
    # 下箭头:expand_dims(slice -> parent)
    rev_y = y + BOX_H * 0.75
    L.append(f'<line x1="{right_x-6}" y1="{rev_y}" x2="{left_x+BOX_W+6}" y2="{rev_y}" '
              'stroke="#b45309" stroke-width="2" marker-end="url(#rev)"/>')
    L.append(f'<text x="{(left_x+BOX_W+right_x)/2}" y="{rev_y+18}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#b45309">{esc(row["expand_lbl"])}</text>')

foot_y0 = TOP + len(ROWS) * ROW_GAP + 6
for i, line in enumerate(CAPTION_LINES):
    L.append(f'<text x="{PAD}" y="{foot_y0+i*18}" font-family="sans-serif" font-size="12" '
              f'fill="#64748b">{esc(line)}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-slice-squeeze.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
