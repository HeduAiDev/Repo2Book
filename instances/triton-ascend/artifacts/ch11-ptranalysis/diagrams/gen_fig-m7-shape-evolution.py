#!/usr/bin/env python3
"""fig-m7-shape-evolution: 三个纯形状算子的 PtrState 演化（tensor-flow 模板）。
上行：make_range -> expand_dims -> broadcast（中间经 mul，非本机制改动，虚线旁注标出）；
下行：splat(base 指针) 独立铺 source。每条边标注该步 stateInfo 的变化。
全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "make_range→expand_dims→broadcast 与 splat：三个纯形状算子不碰 offset/source"
SUBTITLE = "expand_dims 插零 stride 维、broadcast 只加宽 size=1 维的 shape、splat 把标量（含 base 指针）沿各维铺成 stride=0"

TOP_CHAIN = [
    ("make_range\n%14=make_range(0,64)", "[(1,64,d0)]\nsizes=[64]", "normal"),
    ("expand_dims\n(axis=1) %19", "[(1,64,d0),(0,1,d1)]\nsizes=[64,1]", "normal"),
    ("broadcast\n%23", "[(%arg4,64,d0),(0,256,d1)]\nd1: shape 1→256", "final"),
]
BOT_CHAIN = [
    ("splat(%arg1)\n（!tt.ptr 标量）", "—", "normal"),
    ("%26", "[(0,64,d0),(0,256,d1)]\nsource=%arg1", "final"),
]

BOX_W, BOX_H, HGAP, VGAP = 240, 74, 210, 130
PAD, TOP = 40, 108
n_top = len(TOP_CHAIN)
n_bot = len(BOT_CHAIN)
row_w_top = n_top * BOX_W + (n_top - 1) * HGAP
row_w_bot = n_bot * BOX_W + (n_bot - 1) * HGAP
w = PAD * 2 + max(row_w_top, row_w_bot)
top_y = TOP
bot_y = top_y + BOX_H + VGAP
h = bot_y + BOX_H + 110

def x_positions(n, row_w):
    start = PAD + (max(row_w_top, row_w_bot) - row_w) / 2
    return [start + i * (BOX_W + HGAP) for i in range(n)]

xs_top = x_positions(n_top, row_w_top)
xs_bot = x_positions(n_bot, row_w_bot)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="ad" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

COLOR = {"normal": ("#e2e8f0", "#64748b", "#0f172a"),
         "final": ("#dcfce7", "#16a34a", "#14532d")}


def draw_box(x, y, label, state, kind):
    fill, stroke, text_fill = COLOR[kind]
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    lines = label.split("\n")
    y0 = y + 22 - (len(lines) - 1) * 7
    for k, line in enumerate(lines):
        L.append(f'<text x="{x+BOX_W/2}" y="{y0+k*14}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" font-weight="bold" '
                  f'fill="{text_fill}">{esc(line)}</text>')
    if state != "—":
        slines = state.split("\n")
        sy0 = y + BOX_H - 14 - (len(slines) - 1) * 12
        for k, line in enumerate(slines):
            L.append(f'<text x="{x+BOX_W/2}" y="{sy0+k*12.5}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="10" fill="{text_fill}">{esc(line)}</text>')


# 上行
for i, (label, state, kind) in enumerate(TOP_CHAIN):
    draw_box(xs_top[i], top_y, label, state, kind)
    if i < n_top - 1:
        y_mid = top_y + BOX_H / 2
        L.append(f'<line x1="{xs_top[i]+BOX_W}" y1="{y_mid}" x2="{xs_top[i+1]}" y2="{y_mid}" '
                  'stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')

# 中间 mul 旁注（expand_dims -> broadcast 之间）
mid_x = (xs_top[1] + BOX_W + xs_top[2]) / 2
L.append(f'<text x="{mid_x}" y="{top_y+BOX_H/2-10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="9.5" fill="#b45309">{esc("中间经 mul %21（见 m5）")}</text>')
L.append(f'<text x="{mid_x}" y="{top_y+BOX_H/2+18}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="9.5" fill="#b45309">{esc("d0 stride 由 1→%arg4，非本机制改动")}</text>')

# 下行
for i, (label, state, kind) in enumerate(BOT_CHAIN):
    draw_box(xs_bot[i], bot_y, label, state, kind)
    if i < n_bot - 1:
        y_mid = bot_y + BOX_H / 2
        L.append(f'<line x1="{xs_bot[i]+BOX_W}" y1="{y_mid}" x2="{xs_bot[i+1]}" y2="{y_mid}" '
                  'stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')

# 两条链的收尾用虚线汇入 addState（示意，不含因果先后，仅示意两条子状态都要喂给 m3/m4）
merge_x = w - PAD - 90
merge_y = (top_y + BOX_H + bot_y) / 2
L.append(f'<path d="M {xs_top[-1]+BOX_W} {top_y+BOX_H/2} L {merge_x} {top_y+BOX_H/2} '
          f'L {merge_x} {merge_y}" fill="none" stroke="#94a3b8" stroke-width="1.4" '
          'stroke-dasharray="5,3"/>')
L.append(f'<path d="M {xs_bot[-1]+BOX_W} {bot_y+BOX_H/2} L {merge_x} {bot_y+BOX_H/2} '
          f'L {merge_x} {merge_y}" fill="none" stroke="#94a3b8" stroke-width="1.4" '
          'stroke-dasharray="5,3" marker-end="url(#ad)"/>')
L.append(f'<rect x="{merge_x-6}" y="{merge_y-20}" width="140" height="40" rx="8" '
          'fill="#f8fafc" stroke="#94a3b8" stroke-width="1.4" stroke-dasharray="4,3"/>')
L.append(f'<text x="{merge_x+64}" y="{merge_y-4}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10" fill="#475569">{esc("交给 addState")}</text>')
L.append(f'<text x="{merge_x+64}" y="{merge_y+11}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="9.5" fill="#94a3b8">{esc("（见 m3/m4，无因果先后）")}</text>')

foot_y = h - 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("expand_dims/broadcast 全程不碰 offset/source；splat 只把已有 source（来自更早的 initStateByPointer）原样铺进整块张量，三者都不产生新 source、不改已有 offset")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m7-shape-evolution.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
