#!/usr/bin/env python3
"""fig-ch01-ttadapter-passchain — flow 模板（多行 chip 链）。
ttadapter 段内 ttir_to_linalg 按序编排一长串 ascend.passes.ttir.add_*：
起手 add_triton_to_structure(L131)，收官 add_triton_to_linalg(L157)。
链首另有 add_auto_blockify 与可选自动调度（L118-L130），画作前置小行。
全部坐标由循环计算，chip 宽度按文字估算。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def text_w(s, fs):
    return fs * sum((0.98 if '一' <= c <= '鿿' else 0.58) for c in s)

PREFIX = [
    ("add_auto_blockify", "L118-L130"),
    ("可选自动调度", None),
]
MAIN_ROW1 = [
    ("add_triton_to_structure", "L131"),
    ("discrete_mask_access", None),
    ("annotation", None),
    ("unstructure", None),
    ("hivm", None),
]
MAIN_ROW2 = [
    ("hfusion", None),
    ("llvm", None),
    ("bubble_up", None),
    ("structure（再次）", None),
    ("add_triton_to_linalg", "L157"),
]
HOT = {"add_triton_to_structure", "add_triton_to_linalg"}

CHIP_H, PAD_X, GAP, ROW_GAP, PAD, TOP = 54, 16, 34, 70, 40, 96

def row_width(row):
    ws = [max(90, int(text_w(name, 12.5)) + 2 * PAD_X) for name, _ in row]
    total = sum(ws) + GAP * (len(ws) - 1)
    return ws, total

pref_ws, pref_total = row_width(PREFIX)
r1_ws, r1_total = row_width(MAIN_ROW1)
r2_ws, r2_total = row_width(MAIN_ROW2)
w = PAD * 2 + max(pref_total, r1_total, r2_total)
h = TOP + 3 * (CHIP_H + ROW_GAP) - ROW_GAP + 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc("ttadapter 段内部：ttir_to_linalg 的 pass 编排顺序")}</text>',
     f'<text x="{w/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc("链首前置 add_auto_blockify（可选自动调度）→ 主链起手/收官各标行号（compiler.py）")}</text>']

def draw_row(row, ws, y, dashed_first=False):
    total = sum(ws) + GAP * (len(ws) - 1)
    x0 = PAD + (w - 2*PAD - total) / 2
    xs_ = []
    x = x0
    for wi in ws:
        xs_.append(x)
        x += wi + GAP
    for i, ((name, loc), wi) in enumerate(zip(row, ws)):
        x = xs_[i]
        hot = name in HOT
        fill = "#fef3c7" if hot else "#e2e8f0"
        stroke = "#d97706" if hot else "#64748b"
        L.append(f'<rect x="{x}" y="{y}" width="{wi}" height="{CHIP_H}" rx="10" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{2.2 if hot else 1.3}"/>')
        ty = y + CHIP_H/2 - (5 if loc else 0) + 4
        L.append(f'<text x="{x+wi/2}" y="{ty}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="12.5" fill="#1f2937" font-weight="{"bold" if hot else "normal"}">{esc(name)}</text>')
        if loc:
            L.append(f'<text x="{x+wi/2}" y="{ty+16}" text-anchor="middle" font-family="sans-serif" '
                     f'font-size="11" fill="#b45309" font-weight="bold">{esc(loc)}</text>')
        if i < len(row) - 1:
            x1 = x + wi
            x2 = xs_[i+1]
            L.append(f'<line x1="{x1+3}" y1="{y+CHIP_H/2}" x2="{x2-3}" y2="{y+CHIP_H/2}" '
                     f'stroke="#64748b" stroke-width="1.8" marker-end="url(#a)"/>')
    return xs_, ws

y_pref = TOP
y_r1 = TOP + CHIP_H + ROW_GAP
y_r2 = y_r1 + CHIP_H + ROW_GAP

pref_xs, pref_ws2 = draw_row(PREFIX, pref_ws, y_pref)
r1_xs, r1_ws2 = draw_row(MAIN_ROW1, r1_ws, y_r1)
r2_xs, r2_ws2 = draw_row(MAIN_ROW2, r2_ws, y_r2)

# 前置行 -> 主链第一行：虚线箭头（可选路径，非严格前驱）
px = pref_xs[-1] + pref_ws2[-1] / 2
qx = r1_xs[0] + r1_ws2[0] / 2
L.append(f'<path d="M {px} {y_pref+CHIP_H} L {px} {y_pref+CHIP_H+18} L {qx} {y_r1-18} L {qx} {y_r1-3}" '
         f'fill="none" stroke="#94a3b8" stroke-width="1.6" stroke-dasharray="5,4" marker-end="url(#b)"/>')

# 主链行 1 -> 行 2 折行连接
x1 = r1_xs[-1] + r1_ws2[-1] / 2
x2 = r2_xs[0] + r2_ws2[0] / 2
mid_y = y_r1 + CHIP_H + ROW_GAP / 2
L.append(f'<path d="M {x1} {y_r1+CHIP_H} L {x1} {mid_y} L {x2} {mid_y} L {x2} {y_r2-3}" '
         f'fill="none" stroke="#64748b" stroke-width="1.8" marker-end="url(#a)"/>')

foot_y = h - 16
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
         f'fill="#64748b">{esc("这条链把 tensor-of-pointers 指针语义逐步抛弃、逆向还原成结构化 memref；逐 pass 的 C++ 实现归 P3/P5。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch01-ttadapter-passchain.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
