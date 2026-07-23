#!/usr/bin/env python3
"""fig-m12-interleave: tensor-flow 模板（flow 骨架 + 每条边标 shape）。
Interleave（store 侧）：两条偶/奇 materialize 经 insert_slice(stride 2, offset 0/1)
交织进 2N tensor，最后单次搬运落盘——deinterleave 的逆运算。全坐标由循环/常量计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

EVEN = [0, 1, 2, 3]
ODD = [100, 101, 102, 103]
INTERLEAVED = [0, 100, 1, 101, 2, 102, 3, 103]

W = 980
PAD, TOP = 40, 92
CX = W / 2
CELL, CGAP = 46, 4

SVG_H = 540
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {SVG_H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{SVG_H}" fill="white"/>']

L.append(f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" font-weight="bold" '
         f'fill="#0f172a">{esc("Interleave（store 侧）：偶/奇两条 materialize → insert_slice 交织 → 单次落盘")}</text>')
L.append(f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" fill="#64748b">'
         f'{esc("InterleaveStatusOptimization，InterleaveOptimization.cpp:L370-L512（deinterleave 的逆运算）")}</text>')

def small_strip(cx, y, values, fill, stroke):
    sw = len(values) * (CELL + CGAP) - CGAP
    x0 = cx - sw / 2
    for i, v in enumerate(values):
        x = x0 + i * (CELL + CGAP)
        L.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="36" rx="4" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        L.append(f'<text x="{x+CELL/2}" y="{y+24}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="12" fill="{stroke}">{esc(str(v))}</text>')

left_cx = PAD + 230
right_cx = W - PAD - 230

y1 = TOP
L.append(f'<text x="{left_cx}" y="{y1-10}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#1d4ed8">{esc("even materialize 源 [0,1,2,3]")}</text>')
L.append(f'<text x="{right_cx}" y="{y1-10}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#a21caf">{esc("odd materialize 源 [100,101,102,103]")}</text>')
small_strip(left_cx, y1, EVEN, "#dbeafe", "#1d4ed8")
small_strip(right_cx, y1, ODD, "#fce7f3", "#a21caf")

y_label1 = y1 + 36 + 22
L.append(f'<text x="{left_cx}" y="{y_label1}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" fill="#1d4ed8">{esc("① insertFirst: offset0 stride2")}</text>')
L.append(f'<text x="{right_cx}" y="{y_label1}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" fill="#a21caf">{esc("② insertSecond: offset1 stride2")}</text>')

# fork lines start BELOW the labels (clear vertical gap) so the diagonal
# never crosses the label text, then converge into the empty-tensor box.
fork_start_y = y_label1 + 14
y_merge_box = fork_start_y + 40
merge_w, merge_h = 300, 40
L.append(f'<rect x="{CX-merge_w/2}" y="{y_merge_box}" width="{merge_w}" height="{merge_h}" rx="8" '
         'fill="#f1f5f9" stroke="#64748b" stroke-width="2"/>')
L.append(f'<text x="{CX}" y="{y_merge_box+merge_h/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#334155">{esc("空 tensor<8>（emptyTensor）")}</text>')

for cx in (left_cx, right_cx):
    L.append(f'<line x1="{cx}" y1="{fork_start_y}" x2="{CX}" y2="{y_merge_box}" '
             'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')

# on-chip interleaved tensor with values (after two insert_slice)
y3 = y_merge_box + merge_h + 54
L.append(f'<text x="{CX}" y="{y3-10}" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="#0f172a">{esc("两次 tensor.insert_slice 后：tensor<8>（交错缓冲）")}</text>')
L.append(f'<line x1="{CX}" y1="{y_merge_box+merge_h}" x2="{CX}" y2="{y3-24}" '
         'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
tensor_w = len(INTERLEAVED) * (CELL + CGAP) - CGAP
tensor_x0 = CX - tensor_w / 2
for i, v in enumerate(INTERLEAVED):
    x = tensor_x0 + i * (CELL + CGAP)
    is_even = (i % 2 == 0)
    fill, stroke = ("#dbeafe", "#1d4ed8") if is_even else ("#fce7f3", "#a21caf")
    L.append(f'<rect x="{x}" y="{y3}" width="{CELL}" height="36" rx="4" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{x+CELL/2}" y="{y3+24}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="12" fill="{stroke}">{esc(str(v))}</text>')
L.append(f'<text x="{CX}" y="{y3+36+20}" text-anchor="middle" font-family="sans-serif" font-size="11.5" '
         f'fill="#64748b">{esc("[0, 100, 1, 101, 2, 102, 3, 103]")}</text>')

y4 = y3 + 36 + 46
L.append(f'<line x1="{CX}" y1="{y3+36}" x2="{CX}" y2="{y4}" '
         'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
L.append(f'<text x="{CX+14}" y="{y4-14}" font-family="sans-serif" font-size="11.5" '
         f'fill="#334155">{esc("③ MaterializeInDestination：单次落盘")}</text>')

box_w, box_h = 420, 42
L.append(f'<rect x="{CX-box_w/2}" y="{y4}" width="{box_w}" height="{box_h}" rx="8" '
         'fill="#dcfce7" stroke="#15803d" stroke-width="2"/>')
L.append(f'<text x="{CX}" y="{y4+box_h/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#15803d">{esc("dst 内存单次写回 + erase 原两条 materialize")}</text>')

foot_y = y4 + box_h + 46
L.append(f'<rect x="{PAD}" y="{foot_y-24}" width="{W-2*PAD}" height="70" rx="8" '
         'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
L.append(f'<text x="{PAD+16}" y="{foot_y-2}" font-family="sans-serif" font-size="12.5" '
         f'fill="#334155">{esc("偶半 offset0、奇半 offset1，两次 stride-2 insert_slice 交织成连续 2N，")}</text>')
L.append(f'<text x="{PAD+16}" y="{foot_y+18}" font-family="sans-serif" font-size="12.5" '
         f'fill="#334155">{esc("单次 materialize 落盘，替代两趟跨步写。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m12-interleave.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
