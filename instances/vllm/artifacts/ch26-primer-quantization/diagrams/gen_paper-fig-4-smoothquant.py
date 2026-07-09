#!/usr/bin/env python3
"""paper-fig-4-smoothquant: 重绘自 arXiv:2211.10438 Figure 4 —— OPT-13B 某线性层里
真实的激活/权重量级实测证据（非合成小例子）。
四联伪 3D 柱状图：激活(原始，Hard，少数通道幅度 >70)→激活(SmoothQuant 后，Easy，被压平)→
权重(原始，Very easy，本就平坦)→权重(SmoothQuant 后，Harder but still easy，比原始稍陡但仍平整)。
每个 panel 用「地板 + 左墙」的斜轴测投影画网格框架，柱子沿 channel 轴排开、
高度示意该通道的相对幅度——形状取自原图的定性轮廓（少数尖峰 vs 整体平坦），
非逐通道数值复刻（原图本身也不给逐通道数表）。
"""
import math
import random
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def esc_bold(s):
    """转义并在粗体文本里把"量"字拆到 font-weight=normal 的 tspan——
    这套渲染管线(rsvg-convert)的粗体 CJK 回退字体缺"量"字形,粗体直出会变豆腐块。"""
    return '<tspan font-weight="normal">量</tspan>'.join(esc(p) for p in s.split('量'))


TITLE = "OPT-13B 真实实测：少数激活通道幅度 >70，量化难度被迁移到权重后两边都好量化"
SUBTITLE = "重绘自 arXiv:2211.10438 Figure 4（某线性层的激活/权重量级，SmoothQuant 前后）"

FLOOR_W, WALL_H = 150, 108
DEPTH_DX, DEPTH_DY = 46, -30
PANEL_W = 230
PAD = 42
N_BARS = 70

random.seed(42)


def jittered(n, base, amp, seed_add=0.0):
    vals = []
    for i in range(n):
        t = i / (n - 1)
        noise = math.sin((i * 12.9898 + seed_add) % 6.28318) * 0.5 + 0.5
        noise2 = random.random()
        vals.append(base + amp * (0.5 * noise + 0.5 * noise2))
    return vals


def profile_act_original():
    vals = [0.05] * N_BARS
    for i in range(N_BARS):
        u = i / (N_BARS - 1)
        if 0.08 <= u <= 0.30:
            vals[i] = 0.80 + 0.20 * random.random()
        elif 0.32 <= u <= 0.46:
            vals[i] = 0.35 + 0.15 * random.random()
        elif 0.55 <= u <= 0.86:
            vals[i] = 0.78 + 0.22 * random.random()
        else:
            vals[i] = 0.04 + 0.03 * random.random()
    return vals


def profile_act_smoothed():
    vals = []
    for i in range(N_BARS):
        u = i / (N_BARS - 1)
        if 0.08 <= u <= 0.30 or 0.55 <= u <= 0.86:
            vals.append(0.16 + 0.06 * random.random())
        else:
            vals.append(0.06 + 0.04 * random.random())
    return vals


def profile_weight_original():
    return [0.12 + 0.04 * random.random() for _ in range(N_BARS)]


def profile_weight_smoothed():
    return [0.14 + 0.14 * random.random() for _ in range(N_BARS)]


PANELS = [
    ("激活（原始）", profile_act_original(), "#dc2626", "#fca5a5", "难量化", "#b91c1c"),
    ("激活（SmoothQuant 后）", profile_act_smoothed(), "#2563eb", "#93c5fd", "易量化", "#15803d"),
    ("权重（原始）", profile_weight_original(), "#2563eb", "#93c5fd", "很容易量化", "#15803d"),
    ("权重（SmoothQuant 后）", profile_weight_smoothed(), "#2563eb", "#93c5fd", "较难但仍易量化", "#15803d"),
]

TOP = 112
ANN_H = 30
BOX_Y = TOP + ANN_H + 20
CAP_Y = BOX_Y + WALL_H + 30
TITLE_Y = CAP_Y + 22
w = PAD * 2 + PANEL_W * 4
h = TITLE_Y + 26


def floor_point(ox, oy, u, v):
    return (ox + u * FLOOR_W + v * DEPTH_DX, oy + v * DEPTH_DY)


def wall_point(ox, oy, v, hh):
    return (ox + v * DEPTH_DX, oy + v * DEPTH_DY - hh * WALL_H)


def box_svg(ox, oy):
    out = []
    NG = 5
    for i in range(NG + 1):
        u = i / NG
        p0 = floor_point(ox, oy, u, 0)
        p1 = floor_point(ox, oy, u, 1)
        out.append(f'<line x1="{p0[0]:.1f}" y1="{p0[1]:.1f}" x2="{p1[0]:.1f}" y2="{p1[1]:.1f}" '
                   f'stroke="#cbd5e1" stroke-width="0.8"/>')
    for j in range(NG + 1):
        v = j / NG
        p0 = floor_point(ox, oy, 0, v)
        p1 = floor_point(ox, oy, 1, v)
        out.append(f'<line x1="{p0[0]:.1f}" y1="{p0[1]:.1f}" x2="{p1[0]:.1f}" y2="{p1[1]:.1f}" '
                   f'stroke="#cbd5e1" stroke-width="0.8"/>')
    for j in range(NG + 1):
        v = j / NG
        p0 = wall_point(ox, oy, v, 0)
        p1 = wall_point(ox, oy, v, 1)
        out.append(f'<line x1="{p0[0]:.1f}" y1="{p0[1]:.1f}" x2="{p1[0]:.1f}" y2="{p1[1]:.1f}" '
                   f'stroke="#e2e8f0" stroke-width="0.8"/>')
    for i in range(NG + 1):
        hh = i / NG
        p0 = wall_point(ox, oy, 0, hh)
        p1 = wall_point(ox, oy, 1, hh)
        out.append(f'<line x1="{p0[0]:.1f}" y1="{p0[1]:.1f}" x2="{p1[0]:.1f}" y2="{p1[1]:.1f}" '
                   f'stroke="#e2e8f0" stroke-width="0.8"/>')
    # 外框
    corners = [floor_point(ox, oy, 0, 0), floor_point(ox, oy, 1, 0),
               floor_point(ox, oy, 1, 1), floor_point(ox, oy, 0, 1)]
    pts = " ".join(f'{p[0]:.1f},{p[1]:.1f}' for p in corners)
    out.append(f'<polygon points="{pts}" fill="none" stroke="#94a3b8" stroke-width="1"/>')
    wcorners = [wall_point(ox, oy, 0, 0), wall_point(ox, oy, 1, 0),
                wall_point(ox, oy, 1, 1), wall_point(ox, oy, 0, 1)]
    wpts = " ".join(f'{p[0]:.1f},{p[1]:.1f}' for p in wcorners)
    out.append(f'<polygon points="{wpts}" fill="none" stroke="#94a3b8" stroke-width="1"/>')
    return out


def bars_svg(ox, oy, values, hi_color, lo_color, hi_thresh=0.5):
    out = []
    bw = FLOOR_W / len(values)
    for i, v in enumerate(values):
        u = i / (len(values) - 1)
        base = floor_point(ox, oy, u, 0)
        top = (base[0], base[1] - v * WALL_H)
        color = hi_color if v >= hi_thresh else lo_color
        out.append(f'<line x1="{base[0]:.1f}" y1="{base[1]:.1f}" x2="{top[0]:.1f}" y2="{top[1]:.1f}" '
                   f'stroke="{color}" stroke-width="{max(1.4, bw*0.9):.1f}" opacity="0.85"/>')
    return out


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0f172a"/></marker>'
     '</defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-18}" font-family="sans-serif" font-size="15" '
     f'fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD-2}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

panel_x = [PAD + i * PANEL_W for i in range(4)]

# top annotations: "平滑" between panel0-1, "迁移量化难度" long arc panel0->panel3
smooth_x0 = panel_x[0] + FLOOR_W / 2 + 20
smooth_x1 = panel_x[1] + FLOOR_W / 2 - 20
L.append(f'<text x="{(smooth_x0+smooth_x1)/2:.1f}" y="{TOP-2}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="#0f172a">平滑</text>')
L.append(f'<path d="M {smooth_x0:.1f} {TOP+8} L {smooth_x1:.1f} {TOP+8}" fill="none" '
         f'stroke="#0f172a" stroke-width="1.2" marker-end="url(#a)"/>')

mig_x0 = panel_x[0] + FLOOR_W / 2
mig_x1 = panel_x[3] + FLOOR_W / 2
mig_y = TOP - 22
L.append(f'<text x="{(mig_x0+mig_x1)/2:.1f}" y="{mig_y-6}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="#0f172a">迁移量化难度</text>')
L.append(f'<path d="M {mig_x0:.1f} {mig_y+16} Q {(mig_x0+mig_x1)/2:.1f} {mig_y-14} '
         f'{mig_x1:.1f} {mig_y+16}" fill="none" stroke="#0f172a" stroke-width="1.2" marker-end="url(#a)"/>')

for idx, (title, values, hi_color, lo_color, cap, cap_color) in enumerate(PANELS):
    ox = panel_x[idx]
    oy = BOX_Y + WALL_H
    L.extend(box_svg(ox, oy))
    L.extend(bars_svg(ox, oy, values, hi_color, lo_color))
    L.append(f'<text x="{ox+FLOOR_W/2:.1f}" y="{CAP_Y}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="13" font-weight="bold" '
             f'fill="{cap_color}">{esc_bold(cap)}</text>')
    L.append(f'<text x="{ox+FLOOR_W/2:.1f}" y="{TITLE_Y}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="11.5" fill="#0f172a">{esc(title)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-4-smoothquant.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
