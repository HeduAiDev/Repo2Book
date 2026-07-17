#!/usr/bin/env python3
"""f17-1-if-dispatch: visit_If 按『cond 是否动态 × 是否含 return』四路分派决策树。
主干竖直向下(判定节点用带虚线边框的矩形);每个判定的分支终态框沿右侧纵向排开,
与其判定节点对齐(仿 ch01 fig-m01-three-way 写法)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TRUNK_X = 210
TRUNK_W = 360
TERM_GAP = 70
TERM_W = 470
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

ENTRY_H = 50
DEC1_H, DEC2_H, DEC3_H = 56, 66, 66
TERM3_H = 66
VGAP = 58

TOP = 46
y_entry = TOP + ENTRY_H / 2
y_dec1 = y_entry + ENTRY_H / 2 + VGAP + DEC1_H / 2
y_dec2 = y_dec1 + DEC1_H / 2 + VGAP + DEC2_H / 2
y_dec3 = y_dec2 + DEC2_H / 2 + VGAP + DEC3_H / 2
y_term3 = y_dec3 + DEC3_H / 2 + VGAP + TERM3_H / 2

TERM1_H, TERM2_H, TERM3B_H = 60, 60, 66

BAND_Y = y_term3 + TERM3_H / 2 + 44
BAND_H = 60
h = int(BAND_Y + BAND_H + 30)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker></defs>')
L.append(f'<rect width="{W}" height="{h}" fill="white"/>')
L.append(f'<text x="30" y="26" font-family="sans-serif" font-size="16" font-weight="bold" '
          f'fill="#1e293b">visit_If 四路分派:cond 是否动态 x 子树是否含 return(L683-L708)</text>')

# 入口
L += box(TRUNK_X, y_entry, TRUNK_W, ENTRY_H, "#e2e8f0", "#475569",
          ["一段 if:test 先被求值成 cond"], bold_first=True)

# 判定 1: cond 是否运行时张量
L += vline(TRUNK_X, y_entry + ENTRY_H / 2, y_dec1 - DEC1_H / 2)
L += box(TRUNK_X, y_dec1, TRUNK_W, DEC1_H, "#fef9c3", "#a16207",
          ["cond 是运行时张量(tensor)吗?"], dashed=True)

# 判定 2: contains_return
L += vline(TRUNK_X, y_dec1 + DEC1_H / 2, y_dec2 - DEC2_H / 2, "是(动态)")
L += box(TRUNK_X, y_dec2, TRUNK_W, DEC2_H, "#dbeafe", "#1d4ed8",
          ["子树含 return 吗?", "(ContainsReturnChecker)"], dashed=True)

# 判定 3: scf_stack
L += vline(TRUNK_X, y_dec2 + DEC2_H / 2, y_dec3 - DEC3_H / 2, "是(含 return)")
L += box(TRUNK_X, y_dec3, TRUNK_W, DEC3_H, "#dbeafe", "#1d4ed8",
          ["scf_stack 非空吗?", "(身处 for/while 循环内)"], dashed=True)

# 终态 4(否则,主干延续向下):顶层 CFG
L += vline(TRUNK_X, y_dec3 + DEC3_H / 2, y_term3 - TERM3_H / 2, "否(不在循环内)")
L += box(TRUNK_X, y_term3, TRUNK_W, TERM3_H, "#fed7aa", "#c2410c",
          ["④ visit_if_top_level -> 顶层 CFG", "cf.cond_br 分支 + 块参数汇合(L695)"], fs=11.5, bold_first=True)

term_defs = [
    (y_dec1, TERM1_H, "#e2e8f0", "#475569",
     ["① 编译期直接择一分支运行", "visit_compound_statement(active_block)"], "code_generator.py:L707-L708"),
    (y_dec2, TERM2_H, "#dcfce7", "#15803d",
     ["② visit_if_scf -> scf.if", "(结构化,单入单出)"], "code_generator.py:L697"),
    (y_dec3, TERM3B_H, "#fecaca", "#b91c1c",
     ["③ raise:结构化循环内", "不能中途 return(L690-L694)"], "code_generator.py:L688-L694"),
]
for cy, th, fill, stroke, lines, prov in term_defs:
    L += hline(TRUNK_X + TRUNK_W / 2, TERM_X - TERM_W / 2, cy, "否" if cy == y_dec1 else ("否" if cy == y_dec2 else "是"))
    L += box(TERM_X, cy, TERM_W, th, fill, stroke, lines, bold_first=True)
    L.append(f'<text x="{TERM_X:.1f}" y="{cy+th/2+16:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="#94a3b8">{esc(prov)}</text>')

# 底部条幅
L.append(f'<rect x="30" y="{BAND_Y}" width="{W-60}" height="{BAND_H}" rx="8" '
          'fill="#f8fafc" stroke="#94a3b8" stroke-width="1"/>')
L.append(f'<text x="50" y="{BAND_Y+22}" font-family="sans-serif" font-size="12" '
          f'fill="#1e293b">四条路径互斥且穷尽:①②在编译期/结构化区域内决出,③是编译期报错,④走非结构化 CFG</text>')
L.append(f'<text x="50" y="{BAND_Y+42}" font-family="sans-serif" font-size="10.5" '
          f'fill="#64748b">visit_If 总入口:python/triton/compiler/code_generator.py L683-L708</text>')

L.append('</svg>')
out = Path(__file__).with_name("f17-1-if-dispatch.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} size={W}x{h}")
