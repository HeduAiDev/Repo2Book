#!/usr/bin/env python3
"""fig-ch06-04-flip-two-paths — before-after 模板。
同一个 flip():SIMD 模式发一条 ascend.flip 就完;SIMT 模式没有这条算子,
退化成 log2(n) 轮 reshape+xor-swap。数据取自 traces/builder_calls.json(m9)。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


BLUE, ORANGE, GRAY, RED, GREEN = "#1d4ed8", "#c2410c", "#94a3b8", "#b91c1c", "#15803d"

TITLE = "flip 的两条路:同一个 API,SIMD 一条算子 vs SIMT 展开成 log2(n) 轮"

# SIMT n=4 的 8 步:bitcast→reshape(2,2)→[round1: reduce,xor]→[round2: reduce,xor]→reshape(4)→bitcast
SIMT4_STEPS = [
    ("bitcast", None),
    ("reshape → (2,2)", None),
    ("reduce(第 1 轮)", 1),
    ("xor-swap(第 1 轮)", 1),
    ("reduce(第 2 轮)", 2),
    ("xor-swap(第 2 轮)", 2),
    ("reshape → (4,)", None),
    ("bitcast", None),
]

BOX_W, BOX_H, VGAP, PAD, TOP = 250, 40, 14, 50, 130
PANEL_L_W, PANEL_R_W = 340, 340
GAP = 140
W = PAD * 2 + PANEL_L_W + GAP + PANEL_R_W + 260
H = TOP + len(SIMT4_STEPS) * (BOX_H + VGAP) + 260

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{W/2}" y="56" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc("分叉判据:builder.is_simt_mode()  ·  两条路入口是同一个 flip()")}</text>']

# ── 左面板:SIMD 模式 ───────────────────────────────────────────────────
LX = PAD
lcx = LX + PANEL_L_W / 2
L.append(f'<text x="{lcx}" y="{TOP-40}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" font-weight="bold" fill="{BLUE}">{esc("SIMD 模式")}</text>')
sy = TOP
L.append(f'<rect x="{lcx-BOX_W/2}" y="{sy}" width="{BOX_W}" height="{BOX_H+10}" rx="8" '
         f'fill="#eff6ff" stroke="{BLUE}" stroke-width="2.2"/>')
L.append(f'<text x="{lcx}" y="{sy+(BOX_H+10)/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="{BLUE}">{esc("create_flip(shape[4,8], dim=1)")}</text>')
L.append(f'<text x="{lcx}" y="{sy+BOX_H+40}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12.5" fill="#1e40af">{esc("1 次 builder 调用")}</text>')
L.append(f'<text x="{lcx}" y="{sy+BOX_H+64}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#64748b">{esc("硬件有整排翻转指令,喊一声就好")}</text>')

# ── 右面板:SIMT 模式(n=4,8 步) ─────────────────────────────────────
RX = PAD + PANEL_L_W + GAP
rcx = RX + PANEL_R_W / 2
L.append(f'<text x="{rcx}" y="{TOP-40}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" font-weight="bold" fill="{ORANGE}">{esc("SIMT 模式(n=4)")}</text>')
for i, (step, rnd) in enumerate(SIMT4_STEPS):
    y = TOP + i * (BOX_H + VGAP)
    hot = rnd is not None
    fill = "#fff7ed" if hot else "#f1f5f9"
    stroke = ORANGE if hot else GRAY
    L.append(f'<rect x="{rcx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="7" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if hot else 1.4}"/>')
    L.append(f'<text x="{rcx}" y="{y+BOX_H/2+5}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="12" fill="{"#9a3412" if hot else "#334155"}">{esc(step)}</text>')
    if i < len(SIMT4_STEPS) - 1:
        L.append(f'<line x1="{rcx}" y1="{y+BOX_H}" x2="{rcx}" y2="{y+BOX_H+VGAP-3}" '
                 f'stroke="#64748b" stroke-width="1.4" marker-end="url(#a)"/>')
# 轮次括号标注(round1: idx2-3, round2: idx4-5)
bracket_x = rcx + BOX_W / 2 + 20
for rnd, (i0, i1) in [(1, (2, 3)), (2, (4, 5))]:
    y0 = TOP + i0 * (BOX_H + VGAP)
    y1 = TOP + i1 * (BOX_H + VGAP) + BOX_H
    L.append(f'<path d="M {bracket_x} {y0} L {bracket_x+10} {y0} L {bracket_x+10} {y1} '
             f'L {bracket_x} {y1}" fill="none" stroke="{ORANGE}" stroke-width="1.6"/>')
    L.append(f'<text x="{bracket_x+18}" y="{(y0+y1)/2+4}" font-family="sans-serif" '
             f'font-size="11" font-weight="bold" fill="{ORANGE}">{esc(f"第{rnd}轮")}</text>')
ry_end = TOP + len(SIMT4_STEPS) * (BOX_H + VGAP)
L.append(f'<text x="{rcx}" y="{ry_end+16}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12.5" font-weight="bold" fill="{ORANGE}">'
         f'{esc("共 8 次 builder 调用(log2(4)=2 轮,每轮 reduce+xor 共 2 次)")}</text>')

# ── 中间:等价号 ───────────────────────────────────────────────────────
midx = (LX + PANEL_L_W + RX) / 2
midy = TOP + BOX_H / 2 + 5
L.append(f'<text x="{midx}" y="{midy}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="20" font-weight="bold" fill="#64748b">{esc("同一个")}</text>')
L.append(f'<text x="{midx}" y="{midy+26}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="16" font-weight="bold" fill="#64748b">{esc("flip()")}</text>')

# ── 底部:n=8 对照 + 前置条件 ──────────────────────────────────────────
FOOT_Y = ry_end + 60
L.append(f'<rect x="{PAD}" y="{FOOT_Y}" width="{W-2*PAD}" height="110" rx="10" '
         f'fill="#f8fafc" stroke="{GRAY}" stroke-width="1.4"/>')
FOOT = [
    "n=8 时:3 轮 xor-swap,中间 shape 是 (2,2,2),共 10 次 builder 调用(2 bitcast + 2 reshape + 3×2 轮)。",
    "SIMT 前置条件:n 必须是 2 的幂(core.static_assert(_is_power_of_two(...)));n=3 时 static assertion failed。",
    "SIMD 恒为 1 次调用,与 n 无关 —— 硬件的整排翻转指令不关心长度。",
]
for i, ln in enumerate(FOOT):
    L.append(f'<text x="{PAD+16}" y="{FOOT_Y+26+i*26}" font-family="sans-serif" font-size="12" '
             f'fill="#334155">{esc(ln)}</text>')

H_ACTUAL = FOOT_Y + 130
L.append('</svg>')
svg = '\n'.join(L)
svg = svg.replace(f'viewBox="0 0 {W} {H}"', f'viewBox="0 0 {W} {H_ACTUAL}"')
svg = svg.replace(f'<rect width="{W}" height="{H}" fill="white"/>',
                  f'<rect width="{W}" height="{H_ACTUAL}" fill="white"/>')
out = Path(__file__).with_name('fig-ch06-04-flip-two-paths.svg')
out.write_text(svg, encoding='utf-8')
print(f'wrote {out} ({W}x{H_ACTUAL})')
