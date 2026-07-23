#!/usr/bin/env python3
"""fig-m11-deinterleave: tensor-flow 模板（flow 骨架 + 每条边标 shape）。
Deinterleave（load 侧）：stride=2 视图翻倍成连续 2N 搬回，再用 extract_slice
隔一取一分出偶/奇半。全坐标由循环/常量计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

INTERLEAVED = [0, 100, 1, 101, 2, 102, 3, 103]
EVEN = [0, 1, 2, 3]
ODD = [100, 101, 102, 103]

W = 980
PAD, TOP = 40, 92
CX = W / 2
CELL, CGAP = 46, 4

SVG_H = 660
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {SVG_H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{SVG_H}" fill="white"/>']

L.append(f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" font-weight="bold" '
         f'fill="#0f172a">{esc("Deinterleave（load 侧）：stride=2 跨步视图 → 连续 2N 搬运 → 片上隔一取一")}</text>')
L.append(f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" fill="#64748b">'
         f'{esc("DeinterleaveStatusOptimization，InterleaveOptimization.cpp:L169-L245（触发：stride=2 且末维偶数）")}</text>')

y = TOP
box_w, box_h = 420, 42

def box(cx, y, w, h, label, fill, stroke, bold=True, fs=13):
    L.append(f'<rect x="{cx-w/2}" y="{y}" width="{w}" height="{h}" rx="8" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    fw = 'font-weight="bold" ' if bold else ''
    L.append(f'<text x="{cx}" y="{y+h/2+5}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="{fs}" {fw}fill="{stroke}">{esc(label)}</text>')

def arrow_labeled(cx, y1, y2, label):
    L.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" '
             'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
    L.append(f'<text x="{cx+14}" y="{y1+18}" font-family="sans-serif" font-size="11.5" '
             f'fill="#334155">{esc(label)}</text>')

# Step1: origin stride-2 view
box(CX, y, box_w, box_h, "原 stride=2 跨步视图（shape 4, stride 2）", "#fee2e2", "#b91c1c")
y2 = y + box_h + 44
arrow_labeled(CX, y+box_h, y2, "① expandInterleaveMemRefType：shape 4→8，stride 2→1")

# Step2: new reinterpret_cast
box(CX, y2, box_w, box_h, "新 reinterpret_cast（连续视图，shape 8，stride 1）", "#fef3c7", "#b45309")
y3 = y2 + box_h + 78
arrow_labeled(CX, y2+box_h, y3 - 26, "② alloc + copy：一次连续搬运，8 元素")

# Step3: on-chip tensor with values
tensor_w = len(INTERLEAVED) * (CELL + CGAP) - CGAP
tensor_x0 = CX - tensor_w / 2
L.append(f'<text x="{CX}" y="{y3-10}" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="#0f172a">{esc("片上 tensor<8>（交错缓冲）")}</text>')
for i, v in enumerate(INTERLEAVED):
    x = tensor_x0 + i * (CELL + CGAP)
    is_even = (i % 2 == 0)
    fill, stroke = ("#dbeafe", "#1d4ed8") if is_even else ("#fce7f3", "#a21caf")
    L.append(f'<rect x="{x}" y="{y3}" width="{CELL}" height="36" rx="4" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{x+CELL/2}" y="{y3+24}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="12" fill="{stroke}">{esc(str(v))}</text>')
L.append(f'<text x="{CX}" y="{y3+36+20}" text-anchor="middle" font-family="sans-serif" font-size="11.5" '
         f'fill="#64748b">{esc("[0, 100, 1, 101, 2, 102, 3, 103]（bufferize to_tensor）")}</text>')

y4 = y3 + 36 + 56
L.append(f'<text x="{CX}" y="{y4-10}" text-anchor="middle" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="#334155">{esc("③ tensor.extract_slice：stride=2 隔一取一，分两支")}</text>')

# fork into even / odd
left_cx = PAD + 230
right_cx = W - PAD - 230
fork_end_y = y4 + 30
for cx in (left_cx, right_cx):
    L.append(f'<line x1="{CX}" y1="{y4}" x2="{cx}" y2="{fork_end_y}" '
             'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')

y5 = fork_end_y + 34
L.append(f'<text x="{left_cx}" y="{fork_end_y+18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="#1d4ed8">{esc("even = extract(offset0, stride2, size4)")}</text>')
L.append(f'<text x="{right_cx}" y="{fork_end_y+18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="#a21caf">{esc("odd = extract(offset1, stride2, size4)")}</text>')

def small_strip(cx, y, values, fill, stroke):
    sw = len(values) * (CELL + CGAP) - CGAP
    x0 = cx - sw / 2
    for i, v in enumerate(values):
        x = x0 + i * (CELL + CGAP)
        L.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="36" rx="4" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        L.append(f'<text x="{x+CELL/2}" y="{y+24}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="12" fill="{stroke}">{esc(str(v))}</text>')

small_strip(left_cx, y5, EVEN, "#dbeafe", "#1d4ed8")
small_strip(right_cx, y5, ODD, "#fce7f3", "#a21caf")

y6 = y5 + 36 + 20
L.append(f'<text x="{left_cx}" y="{y6}" text-anchor="middle" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="#1d4ed8">{esc("even 半")}</text>')
L.append(f'<text x="{right_cx}" y="{y6}" text-anchor="middle" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="#a21caf">{esc("odd 半")}</text>')

foot_y = y6 + 46
L.append(f'<rect x="{PAD}" y="{foot_y-24}" width="{W-2*PAD}" height="70" rx="8" '
         'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
L.append(f'<text x="{PAD+16}" y="{foot_y-2}" font-family="sans-serif" font-size="12.5" '
         f'fill="#334155">{esc("N=4：跨步 2 的偶/奇访问被还原成一次连续 2N=8 搬运替代 4 次跨步 stride-2 访问，")}</text>')
L.append(f'<text x="{PAD+16}" y="{foot_y+18}" font-family="sans-serif" font-size="12.5" '
         f'fill="#334155">{esc("跨步开销从访存挪到便宜的片上 tensor 变形。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m11-deinterleave.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
