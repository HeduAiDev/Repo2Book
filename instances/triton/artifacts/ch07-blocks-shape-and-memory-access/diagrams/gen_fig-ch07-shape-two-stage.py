#!/usr/bin/env python3
"""before-after 模板改写:形状算子两段式。左列=Python 侧只算/校验 shape 元数据,
右列=builder.create_* 才落真算子;中间箭头标注具体校验规则,强调追踪期不搬数据。
三行代表 reshape/expand_dims/broadcast(数字全部来自 dossier m2-shape-metadata)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "形状算子两段式：Python 算元数据 → create_* 才落 IR"
SUBTITLE = "reshape/expand_dims/permute/broadcast/join/split/cat 共用一套机理，追踪期不搬一个字节的数据"

ROWS = [
    ("reshape(8,)→(2,4)", "校验 numel 不变:\n8 = 2×4 ✓", "create_reshape", "shape=(2,4)"),
    ("expand_dims(8,)插入", "dst_shape.insert(axis,1):\n(8,)→(1,8)", "create_expand_dims", "shape=(1,8)"),
    ("broadcast 尺寸对齐", "尺寸1可扩、非1不等\n→ ValueError", "create_broadcast", "shape 按规则扩"),
]

LEFT_W, MID_W, RIGHT_W = 280, 260, 260
BOX_H, VGAP, PAD, TOP = 84, 34, 40, 116
w = PAD * 2 + LEFT_W + MID_W + RIGHT_W
h = TOP + len(ROWS) * (BOX_H + VGAP) - VGAP + 90

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0f172a"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-6}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+18}" font-family="sans-serif" font-size="12.5" '
     f'fill="#475569">{esc(SUBTITLE)}</text>']

# 列头
col_titles = [("Python 侧调用", "#eff6ff", "#3b82f6"), ("元数据校验(纯计算)", "#fef9c3", "#ca8a04"),
              ("builder.create_*(落 IR)", "#dcfce7", "#16a34a")]
col_x = [PAD, PAD + LEFT_W + 20, PAD + LEFT_W + MID_W + 40]
col_w = [LEFT_W - 20, MID_W - 20, RIGHT_W - 20]
head_y = TOP - 34
for i, (name, fill, stroke) in enumerate(col_titles):
    L.append(f'<rect x="{col_x[i]}" y="{head_y}" width="{col_w[i]}" height="26" rx="5" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
    L.append(f'<text x="{col_x[i]+col_w[i]/2}" y="{head_y+18}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="{stroke}">{esc(name)}</text>')

for r, (call, rule, create, outshape) in enumerate(ROWS):
    y = TOP + r * (BOX_H + VGAP)
    # 左:Python 调用
    L.append(f'<rect x="{col_x[0]}" y="{y}" width="{col_w[0]}" height="{BOX_H}" rx="8" '
              'fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>')
    L.append(f'<text x="{col_x[0]+col_w[0]/2}" y="{y+BOX_H/2+5}" text-anchor="middle" '
              f'font-family="monospace" font-size="13" font-weight="bold" '
              f'fill="#1e3a8a">{esc(call)}</text>')
    # 中:校验规则(多行)
    L.append(f'<rect x="{col_x[1]}" y="{y}" width="{col_w[1]}" height="{BOX_H}" rx="8" '
              'fill="#fef9c3" stroke="#ca8a04" stroke-width="1.5"/>')
    lines = rule.split("\n")
    n = len(lines)
    y0 = y + BOX_H / 2 - (n - 1) * 9 + 4
    for k, line in enumerate(lines):
        L.append(f'<text x="{col_x[1]+col_w[1]/2}" y="{y0+k*18}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" fill="#78350f">{esc(line)}</text>')
    # 右:create_* 落地
    L.append(f'<rect x="{col_x[2]}" y="{y}" width="{col_w[2]}" height="{BOX_H}" rx="8" '
              'fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>')
    L.append(f'<text x="{col_x[2]+col_w[2]/2}" y="{y+BOX_H/2-6}" text-anchor="middle" '
              f'font-family="monospace" font-size="13" font-weight="bold" '
              f'fill="#14532d">{esc(create)}</text>')
    L.append(f'<text x="{col_x[2]+col_w[2]/2}" y="{y+BOX_H/2+16}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" fill="#166534">{esc(outshape)}</text>')
    # 箭头:左->中, 中->右
    ay = y + BOX_H / 2
    L.append(f'<line x1="{col_x[0]+col_w[0]}" y1="{ay}" x2="{col_x[1]}" y2="{ay}" '
              'stroke="#0f172a" stroke-width="1.6" marker-end="url(#a)"/>')
    L.append(f'<line x1="{col_x[1]+col_w[1]}" y1="{ay}" x2="{col_x[2]}" y2="{ay}" '
              'stroke="#0f172a" stroke-width="1.6" marker-end="url(#a)"/>')

FOOT_LINES = [
    "结论:一族形状算子只在左、中两栏做纯 Python 计算,数据是否真物化留给后端;",
    "右栏 create_reshape/create_expand_dims/create_broadcast 才是真正落地的 IR 算子。",
]
foot_y0 = h - 46
for i, line in enumerate(FOOT_LINES):
    L.append(f'<text x="{PAD}" y="{foot_y0+i*20}" font-family="sans-serif" font-size="12" '
              f'fill="#64748b">{esc(line)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch07-shape-two-stage.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
