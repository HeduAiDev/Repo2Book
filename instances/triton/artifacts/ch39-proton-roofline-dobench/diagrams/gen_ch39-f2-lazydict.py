#!/usr/bin/env python3
"""ch39-f2-lazydict: LazyDict 惰性求值(flow + 分支)。
claim: profiling 关时 metadata_fn 调用 0 次;开时钩子 enter 调 .get() 才调 1 次。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

W, H = 980, 660
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>']

def box(x, y, w, h, lines, fill, stroke, fs=13, bold=True, tc="#0f172a"):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    n = len(lines)
    y0 = y + h/2 - (n-1)*9 + 5
    for k, line in enumerate(lines):
        wt = 'font-weight="bold" ' if bold else ''
        L.append(f'<text x="{x+w/2}" y="{y0+k*18}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="{fs}" {wt}fill="{tc}">{esc(line)}</text>')

def arrow(x1, y1, x2, y2, label=None, color="#334155"):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" marker-end="url(#a)"/>')
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        L.append(f'<rect x="{mx-100}" y="{my-24}" width="200" height="19" fill="white"/>')
        L.append(f'<text x="{mx}" y="{my-10}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="11.5" font-weight="bold" fill="#1d4ed8">{esc(label)}</text>')

# --- top box ---
TOP_W, TOP_H = 340, 56
top_x, top_y = W/2 - TOP_W/2, 24
box(top_x, top_y, TOP_W, TOP_H, ["launch_metadata() 被调用", "(每次 kernel.run 前)"], "#e0f2fe", "#0369a1")

# --- decision diamond ---
dw, dh = 340, 100
dx, dy = W/2 - dw/2, 118
pts = f"{dx+dw/2},{dy} {dx+dw},{dy+dh/2} {dx+dw/2},{dy+dh} {dx},{dy+dh/2}"
L.append(f'<polygon points="{pts}" fill="#fef9c3" stroke="#ca8a04" stroke-width="1.5"/>')
L.append(f'<text x="{dx+dw/2}" y="{dy+dh/2-10}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#854d0e">launch_enter_hook is None ?</text>')
L.append(f'<text x="{dx+dw/2}" y="{dy+dh/2+10}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="#a16207">(python/triton/compiler/compiler.py:L399)</text>')

arrow(W/2, top_y+TOP_H, W/2, dy)

# --- left branch: profiling off ---
LB_X, LB_W, LB_Y, LB_H = 90, 300, 300, 60
box(LB_X, LB_Y, LB_W, LB_H, ["return None", "metadata_fn 调用次数 = 0"], "#fee2e2", "#b91c1c", tc="#7f1d1d")
arrow(dx+8, dy+dh-10, LB_X+LB_W*0.5, LB_Y, label="是(profiling 关)")

# --- right branch: profiling on (3-step chain) ---
RB_X, RB_W = 610, 320
RB_Y1, RB_H1 = 300, 54
box(RB_X, RB_Y1, RB_W, RB_H1, ["add(metadata_fn, args)", "只登记,不执行"], "#e0f2fe", "#0369a1")
arrow(dx+dw-8, dy+dh-10, RB_X+RB_W*0.5, RB_Y1, label="否(profiling 开)")

RB_Y2, RB_H2 = RB_Y1+RB_H1+50, 54
box(RB_X, RB_Y2, RB_W, RB_H2, ["钩子 enter: lazy_dict.get()", "(third_party/proton/proton/hook.py:L14)"], "#e0f2fe", "#0369a1", fs=12)
arrow(RB_X+RB_W/2, RB_Y1+RB_H1, RB_X+RB_W/2, RB_Y2)

RB_Y3, RB_H3 = RB_Y2+RB_H2+50, 60
box(RB_X, RB_Y3, RB_W, RB_H3, ["逐个执行 metadata_fn (1 次)", "用 | 合并各 extras 结果"], "#dcfce7", "#15803d", tc="#14532d")
arrow(RB_X+RB_W/2, RB_Y2+RB_H2, RB_X+RB_W/2, RB_Y3)

foot_y = RB_Y3 + RB_H3 + 40
L.append(f'<text x="30" y="{foot_y}" font-family="sans-serif" font-size="11.5" fill="#64748b">'
         f'add() 只把 (metadata_fn, args) 压进 extras 列表(python/triton/compiler/compiler.py:L326-L328)——'
         f'只登记不执行,真正调用推迟到钩子 enter。</text>')
L.append('</svg>')
out = Path(__file__).with_name("ch39-f2-lazydict.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
