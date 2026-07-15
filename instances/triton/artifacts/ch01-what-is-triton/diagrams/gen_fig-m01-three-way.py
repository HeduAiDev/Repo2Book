#!/usr/bin/env python3
"""fig-m01-three-way: visit_Call 三岔分发决策树。
主干竖直向下(判定节点用带虚线边框的矩形,避免菱形挤压文字);每个判定"是"分支
向右到终态框、"否"继续向下;三个终态框沿右侧纵向排开,与其判定节点对齐。
底部条幅给出 demo_kernel 实测命中分布。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TRUNK_X = 200
TRUNK_W = 340
TERM_GAP = 70
TERM_W = 460
TERM_X = TRUNK_X + TRUNK_W / 2 + TERM_GAP + TERM_W / 2
W = int(TERM_X + TERM_W / 2 + 40)

def box(cx, cy, w, h, fill, stroke, lines, fs=12.5, bold_first=False, sw=1.5, dashed=False):
    x, y = cx - w / 2, cy - h / 2
    dash = ' stroke-dasharray="6,4"' if dashed else ''
    out = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="8" '
           f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>']
    n = len(lines)
    y0 = cy - (n - 1) * 8
    for k, line in enumerate(lines):
        fw = 'font-weight="bold" ' if (bold_first and k == 0) else ''
        out.append(f'<text x="{cx:.1f}" y="{y0 + k*16:.1f}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="{fs}" fill="#0f172a" {fw}>{esc(line)}</text>')
    return out

def vline(x, y1, y2, label=None):
    out = [f'<line x1="{x}" y1="{y1:.1f}" x2="{x}" y2="{y2:.1f}" '
           'stroke="#475569" stroke-width="1.5" marker-end="url(#a)"/>']
    if label:
        out.append(f'<text x="{x+10}" y="{(y1+y2)/2 - 4:.1f}" font-family="sans-serif" '
                    f'font-size="11" fill="#64748b">{esc(label)}</text>')
    return out

def hline(x1, x2, y, label=None):
    out = [f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" '
           'stroke="#475569" stroke-width="1.5" marker-end="url(#a)"/>']
    if label:
        out.append(f'<text x="{x1+14:.1f}" y="{y-6:.1f}" font-family="sans-serif" '
                    f'font-size="11" fill="#64748b">{esc(label)}</text>')
    return out

# 逐节点高度与内容
ENTRY_H = 50
DEC1_H, DEC2_H, DEC3_H = 78, 56, 56
TERM3_H = 56
VGAP = 56

TOP = 46
y_entry = TOP + ENTRY_H / 2
y_dec1 = y_entry + ENTRY_H / 2 + VGAP + DEC1_H / 2
y_dec2 = y_dec1 + DEC1_H / 2 + VGAP + DEC2_H / 2
y_dec3 = y_dec2 + DEC2_H / 2 + VGAP + DEC3_H / 2
y_term3 = y_dec3 + DEC3_H / 2 + VGAP + TERM3_H / 2

TERM1_H, TERM2_H, TERM3B_H = 72, 60, 60

BAND_Y = y_term3 + TERM3_H / 2 + 44
BAND_H = 96
h = int(BAND_Y + BAND_H + 30)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker></defs>')
L.append(f'<rect width="{W}" height="{h}" fill="white"/>')
L.append(f'<text x="30" y="26" font-family="sans-serif" font-size="16" font-weight="bold" '
          f'fill="#1e293b">visit_Call 三岔分发:恰好一岔,穷尽互斥</text>')

# 入口
L += box(TRUNK_X, y_entry, TRUNK_W, ENTRY_H, "#e2e8f0", "#475569",
          ["kernel 体内一次写成 f(...) 的调用"], bold_first=True)

# 判定 1: 前置截胡
L += vline(TRUNK_X, y_entry + ENTRY_H / 2, y_dec1 - DEC1_H / 2)
L += box(TRUNK_X, y_dec1, TRUNK_W, DEC1_H, "#fef9c3", "#a16207",
          ["命中前置截胡表(VIP 通道)？", "static_assert / static_print / int / len"], dashed=True)

# 判定 2: JITFunction
L += vline(TRUNK_X, y_dec1 + DEC1_H / 2, y_dec2 - DEC2_H / 2, "否")
L += box(TRUNK_X, y_dec2, TRUNK_W, DEC2_H, "#dbeafe", "#1d4ed8",
          ["isinstance(fn, JITFunction) ？"], dashed=True)

# 判定 3: is_builtin
L += vline(TRUNK_X, y_dec2 + DEC2_H / 2, y_dec3 - DEC3_H / 2, "否")
L += box(TRUNK_X, y_dec3, TRUNK_W, DEC3_H, "#dbeafe", "#1d4ed8",
          ["is_builtin(fn) ？"], dashed=True)

# ③ 终态(否则,主干延续向下)
L += vline(TRUNK_X, y_dec3 + DEC3_H / 2, y_term3 - TERM3_H / 2, "否")
L += box(TRUNK_X, y_term3, TRUNK_W, TERM3_H, "#fee2e2", "#b91c1c",
          ["③ 否则:return fn(*args, **kws)", "当场执行(编译期真跑,L1124-L1126)"], fs=11.5, bold_first=True)

# 终态框(向右分支,来自判定 1/2/3 的"是")
term_defs = [
    (y_dec1, TERM1_H, "#fffbeb", "#a16207",
     ["特殊通道:VIP 前置截胡", "重包结果 / 源码级报错"], "code_generator.py:L1252-L1257"),
    (y_dec2, TERM2_H, "#eff6ff", "#1d4ed8",
     ["① call_JitFunction:抄成 tt.func + tt.call", "(L1105-L1107)"], "code_generator.py:L1097-L1126"),
    (y_dec3, TERM3B_H, "#ecfdf5", "#047857",
     ["② 注入 _builder 建 IR op", "(L1108-L1114)"], "code_generator.py:L1097-L1126"),
]
for cy, th, fill, stroke, lines, prov in term_defs:
    L += hline(TRUNK_X + TRUNK_W / 2, TERM_X - TERM_W / 2, cy, "是")
    L += box(TERM_X, cy, TERM_W, th, fill, stroke, lines, bold_first=True)
    L.append(f'<text x="{TERM_X:.1f}" y="{cy+th/2+16:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="#94a3b8">{esc(prov)}</text>')

# 底部条幅:demo_kernel 实测命中
band_labels = [("①", "×1", "#1d4ed8"), ("②", "×3", "#047857"), ("③", "×0", "#b91c1c")]
L.append(f'<rect x="30" y="{BAND_Y}" width="{W-60}" height="{BAND_H}" rx="8" '
          'fill="#f8fafc" stroke="#94a3b8" stroke-width="1"/>')
L.append(f'<text x="50" y="{BAND_Y+24}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#1e293b">demo_kernel 实测命中分布(Triton v3.2.0)</text>')
seg_w = (W - 100) / 3
for i, (mark, count, color) in enumerate(band_labels):
    cx = 50 + seg_w * i + seg_w / 2
    L.append(f'<text x="{cx:.1f}" y="{BAND_Y+58}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="20" font-weight="bold" fill="{color}">{mark} {count}</text>')
L.append(f'<text x="50" y="{BAND_Y+BAND_H-14}" font-family="sans-serif" font-size="10.5" '
          f'fill="#64748b">运算符 +/* 不排此队,走 visit_BinOp(见 §5)</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m01-three-way.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} size={W}x{h}")
