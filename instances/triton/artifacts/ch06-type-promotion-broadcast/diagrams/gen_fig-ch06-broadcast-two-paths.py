#!/usr/bin/env python3
"""fig-ch06-broadcast-two-paths: broadcast_impl_value 两支路径对比（tensor-flow：
shape 沿箭头标注）。A 支 block+标量→splat；B 支两行 block+block（等秩扩 1 维 /
补前导维再扩）。全坐标由循环/常量计算，文本全 esc()。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def shape_str(shape):
    if shape == "()":
        return "()"
    if len(shape) == 1:
        return f"({shape[0]},)"
    return "(" + ",".join(str(d) for d in shape) + ")"


# 每行: (支路标签, 步骤说明, lhs_in, rhs_in, out_shape, 提示)
ROWS = [
    ("A：block + 标量 → splat", "标量复制填满 block 形状，无需补维",
     (128,), "()", (128,), "create_splat"),
    ("B：block + block（等秩，尺寸 1 维互扩）", "两侧秩相同：尺寸为 1 的维直接扩到对方尺寸",
     (128, 1), (1, 64), (128, 64), "create_broadcast"),
    ("B：block + block（补前导维再扩）", "秩不同：短的一方先在前面补 1 维（右对齐），再逐维扩",
     (128,), (64, 128), (64, 128), "expand_dims → create_broadcast"),
]

PAD = 44
LABEL_W = 430
BOX_W, BOX_H = 150, 60
ARROW1_GAP, ARROW2_GAP = 60, 60
ROW_H = 140
TOP = 108

lhs_x = PAD + LABEL_W + 30
op_x = lhs_x + BOX_W + ARROW1_GAP
rhs_x = op_x + 190
out_x = rhs_x + BOX_W + ARROW2_GAP

w = out_x + BOX_W + PAD
h = TOP + ROW_H * len(ROWS) + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

L.append(f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="19" font-weight="bold" '
          f'fill="#0f172a">{esc("broadcast_impl_value：广播两支——splat 与补维+扩维")}</text>')
L.append(f'<text x="{PAD}" y="58" font-family="sans-serif" font-size="13" '
          f'fill="#475569">{esc("semantic.py:L744-L794 —— Triton 3.2.0 headless 真实取证")}</text>')

for i, (label, desc, lhs_shape, rhs_shape, out_shape, op_label) in enumerate(ROWS):
    y = TOP + i * ROW_H
    cy = y + BOX_H / 2

    # left label block
    L.append(f'<rect x="{PAD}" y="{y}" width="{LABEL_W}" height="{BOX_H+30}" rx="10" '
              'fill="#f8fafc" stroke="#94a3b8" stroke-width="1"/>')
    L.append(f'<text x="{PAD+16}" y="{y+22}" font-family="sans-serif" font-size="13" '
              f'font-weight="bold" fill="#0f172a">{esc(label)}</text>')
    L.append(f'<text x="{PAD+16}" y="{y+42}" font-family="sans-serif" font-size="11" '
              f'fill="#334155">{esc(desc)}</text>')
    L.append(f'<text x="{PAD+16}" y="{y+62}" font-family="sans-serif" font-size="11" '
              f'font-weight="bold" fill="#7c3aed">{esc(f"IR 调用：{op_label}")}</text>')

    # lhs shape box
    L.append(f'<rect x="{lhs_x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="9" '
              'fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>')
    L.append(f'<text x="{lhs_x+BOX_W/2}" y="{cy-4}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#1e40af">{esc("lhs shape")}</text>')
    L.append(f'<text x="{lhs_x+BOX_W/2}" y="{cy+16}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#1e3a8a">{esc(shape_str(lhs_shape))}</text>')

    # operator chip (x)
    op_cx = op_x + 40
    L.append(f'<line x1="{lhs_x+BOX_W}" y1="{cy}" x2="{op_cx-22}" y2="{cy}" '
              'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
    L.append(f'<circle cx="{op_cx}" cy="{cy}" r="18" fill="#f1f5f9" stroke="#64748b"/>')
    L.append(f'<text x="{op_cx}" y="{cy+5}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#334155">×</text>')

    # rhs shape box
    L.append(f'<line x1="{op_cx+18}" y1="{cy}" x2="{rhs_x}" y2="{cy}" '
              'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
    L.append(f'<rect x="{rhs_x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="9" '
              'fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
    L.append(f'<text x="{rhs_x+BOX_W/2}" y="{cy-4}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#92400e">{esc("rhs shape")}</text>')
    L.append(f'<text x="{rhs_x+BOX_W/2}" y="{cy+16}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#78350f">{esc(shape_str(rhs_shape))}</text>')

    # arrow to out
    L.append(f'<line x1="{rhs_x+BOX_W}" y1="{cy}" x2="{out_x}" y2="{cy}" '
              'stroke="#16a34a" stroke-width="1.8" marker-end="url(#a)"/>')
    L.append(f'<text x="{(rhs_x+BOX_W+out_x)/2}" y="{cy-10}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#166534">{esc("对齐后")}</text>')

    # out shape box (both lhs/rhs land here with same shape)
    L.append(f'<rect x="{out_x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="9" '
              'fill="#dcfce7" stroke="#16a34a" stroke-width="1.6"/>')
    L.append(f'<text x="{out_x+BOX_W/2}" y="{cy-4}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#166534">{esc("lhs_out = rhs_out")}</text>')
    L.append(f'<text x="{out_x+BOX_W/2}" y="{cy+16}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#14532d">{esc(shape_str(out_shape))}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-ch06-broadcast-two-paths.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out}  size={w}x{h}')
