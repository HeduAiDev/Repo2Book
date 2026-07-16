#!/usr/bin/env python3
"""flow 模板:数学函数两条路。内置路直接建 IR 节点;extern 路按 dtype 元组查符号表后
链外部 libdevice bitcode。数据来自 core.py / libdevice.py 源码常量。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

W, H, PAD, TOP = 1200, 510, 40, 110
BOX_W, BOX_H, GAP = 300, 56, 46

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="40" font-family="sans-serif" font-size="16" font-weight="bold" '
     f'fill="#1e40af">数学函数两条命运:自己建 IR 节点,或链外部 libdevice</text>',
     f'<text x="{PAD}" y="60" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'core.py:L2628-L2679;third_party/nvidia/language/cuda/libdevice.py:L29-L37</text>']

# 起点
start_x, start_y = W/2 - BOX_W/2, TOP
L.append(f'<rect x="{start_x}" y="{start_y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
          f'fill="#e2e8f0" stroke="#64748b" stroke-width="1.5"/>')
L.append(f'<text x="{W/2}" y="{start_y+BOX_H/2+5}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">调用数学函数(如 exp / mulhi)</text>')

lane_y = start_y + BOX_H + GAP
left_x = PAD + 40
right_x = W - PAD - 40 - BOX_W

# 左路:内置
L.append(f'<text x="{left_x+BOX_W/2}" y="{lane_y-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#1e40af">内置数学(如 exp/umulhi)</text>')
steps_l = ["@builtin 语言层直接调用", "_builder.create_exp(...)", "生成原生 IR 节点(1 个)"]
y_cursor_l = lane_y
for i, s in enumerate(steps_l):
    y = y_cursor_l
    L.append(f'<rect x="{left_x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="#dbeafe" stroke="#1e40af" stroke-width="1.5"/>')
    L.append(f'<text x="{left_x+BOX_W/2}" y="{y+BOX_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="#0f172a">{esc(s)}</text>')
    if i < len(steps_l)-1:
        L.append(f'<line x1="{left_x+BOX_W/2}" y1="{y+BOX_H}" x2="{left_x+BOX_W/2}" '
                  f'y2="{y+BOX_H+GAP//2-4}" stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    y_cursor_l = y + BOX_H + GAP//2

# 右路:extern
L.append(f'<text x="{right_x+BOX_W/2}" y="{lane_y-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#b45309">extern 数学(dispatch)</text>')
steps_r = ["按入参 dtype 元组查 arg_type_symbol_dict",
           "(int32,int32)->__nv_mulhi\n(uint32,uint32)->__nv_umulhi",
           "create_extern_elementwise 链 libdevice.bc"]
box_h_r = [BOX_H, BOX_H + 16, BOX_H]
y_cursor = lane_y
for i, s in enumerate(steps_r):
    y = y_cursor
    bh = box_h_r[i]
    L.append(f'<rect x="{right_x}" y="{y}" width="{BOX_W}" height="{bh}" rx="8" '
              f'fill="#fef3c7" stroke="#b45309" stroke-width="1.5"/>')
    lines = s.split("\n")
    n = len(lines)
    y0 = y + bh/2 - (n-1)*8
    for k, line in enumerate(lines):
        L.append(f'<text x="{right_x+BOX_W/2}" y="{y0+k*16+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="#0f172a">{esc(line)}</text>')
    if i < len(steps_r)-1:
        L.append(f'<line x1="{right_x+BOX_W/2}" y1="{y+bh}" x2="{right_x+BOX_W/2}" '
                  f'y2="{y+bh+GAP//2-4}" stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    y_cursor = y + bh + GAP//2

# 起点分叉到两条路
mid_x1, mid_x2 = left_x+BOX_W/2, right_x+BOX_W/2
by = start_y + BOX_H
L.append(f'<path d="M{W/2-30},{by} L{mid_x1},{lane_y-6}" stroke="#1e40af" stroke-width="1.5" '
          f'fill="none" marker-end="url(#a)"/>')
L.append(f'<path d="M{W/2+30},{by} L{mid_x2},{lane_y-6}" stroke="#b45309" stroke-width="1.5" '
          f'fill="none" marker-end="url(#a)"/>')

foot_y = max(y_cursor_l, y_cursor) + 20
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'覆盖 4 个 dtype 元组(int32/uint32/int64/uint64);不在字典的 dtype(如 float16)直接 '
          f'ValueError,不静默乱选</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch09-two-math-paths.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
