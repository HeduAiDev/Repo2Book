#!/usr/bin/env python3
"""重绘自 arXiv:2602.06036 Figure 3(Draft cost of 1/3/5-layer DFlash and 1-layer EAGLE-3)。
布局对齐原图(ref_x3.png,已下载核对:分组柱状图,x 轴 4/8/16 个 draft token,y 轴延迟 ms,
灰=EAGLE-3、蓝=DFlash(1)、橙=DFlash(3)、绿=DFlash(5))。柱高按原图像素比例视觉估读
(非精确数字化坐标提取),用于呈现关键结构:EAGLE-3 随 token 数线性上涨、DFlash 各层数
曲线几乎水平。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "起草延迟对比(重绘自 arXiv:2602.06036 Fig.3)"
SUBTITLE = "EAGLE-3 随 draft token 数线性上涨;DFlash(1/3/5 层)几乎水平——数值按原图视觉估读"

GROUPS = ["4", "8", "16"]
SERIES = [
    ("EAGLE-3", "#9ca3af", "#4b5563"),
    ("DFlash (1)", "#3b82f6", "#1d4ed8"),
    ("DFlash (3)", "#f59e0b", "#b45309"),
    ("DFlash (5)", "#22c55e", "#15803d"),
]
# 按原图视觉估读(近似值,非像素级坐标提取)
VALUES = {
    "4":  [6.3, 1.7, 3.7, 5.2],
    "8":  [11.5, 1.6, 3.3, 5.6],
    "16": [25.7, 1.6, 3.7, 5.5],
}

PAD_L, PAD_R, TOP = 104, 40, 130
PLOT_W = 760
PLOT_H = 320
BAR_W = 30
BAR_GAP = 6
GROUP_GAP = 70
MAX_Y = 28
Y_TICKS = [0, 10, 20]

w = PAD_L + PLOT_W + PAD_R
h = TOP + PLOT_H + 190

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="15.5" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

plot_x0 = PAD_L
plot_y0 = TOP
baseline_y = plot_y0 + PLOT_H

# y gridlines + labels
for ytick in Y_TICKS:
    gy = baseline_y - (ytick / MAX_Y) * PLOT_H
    L.append(f'<line x1="{plot_x0}" y1="{gy}" x2="{plot_x0+PLOT_W}" y2="{gy}" '
              'stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4,3"/>')
    L.append(f'<text x="{plot_x0-14}" y="{gy+4}" text-anchor="end" font-family="sans-serif" '
              f'font-size="11.5" fill="#64748b">{ytick}</text>')
L.append(f'<line x1="{plot_x0}" y1="{baseline_y}" x2="{plot_x0+PLOT_W}" y2="{baseline_y}" '
          'stroke="#334155" stroke-width="1.6"/>')
L.append(f'<line x1="{plot_x0}" y1="{plot_y0-10}" x2="{plot_x0}" y2="{baseline_y}" '
          'stroke="#334155" stroke-width="1.6"/>')
L.append(f'<text x="{plot_x0-58}" y="{plot_y0+PLOT_H/2}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" fill="#334155" '
          f'transform="rotate(-90 {plot_x0-58} {plot_y0+PLOT_H/2})">Latency (ms)</text>')

group_w = len(SERIES) * (BAR_W + BAR_GAP) - BAR_GAP
total_groups_w = len(GROUPS) * group_w + (len(GROUPS) - 1) * GROUP_GAP
start_x = plot_x0 + (PLOT_W - total_groups_w) / 2

for gi, g in enumerate(GROUPS):
    gx0 = start_x + gi * (group_w + GROUP_GAP)
    for si, (name, fill, stroke) in enumerate(SERIES):
        val = VALUES[g][si]
        bx = gx0 + si * (BAR_W + BAR_GAP)
        bar_h = (val / MAX_Y) * PLOT_H
        by = baseline_y - bar_h
        L.append(f'<rect x="{bx}" y="{by}" width="{BAR_W}" height="{bar_h}" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
    L.append(f'<text x="{gx0+group_w/2}" y="{baseline_y+26}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#0f172a">{esc(g)}</text>')

L.append(f'<text x="{plot_x0+PLOT_W/2}" y="{baseline_y+54}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" fill="#334155">Number of Draft Tokens</text>')

# legend
leg_y = baseline_y + 88
lx = plot_x0
for name, fill, stroke in SERIES:
    L.append(f'<rect x="{lx}" y="{leg_y}" width="16" height="16" rx="3" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
    L.append(f'<text x="{lx+22}" y="{leg_y+13}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(name)}</text>')
    lx += 22 + 10 * len(name) + 30

foot_y = leg_y + 44
L.append(f'<text x="{plot_x0}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">draft token 数从 4 涨到 16(4x):EAGLE-3 延迟约 6.3→25.7ms(约 4x,随 token 数线性);</text>')
L.append(f'<text x="{plot_x0}" y="{foot_y+20}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">DFlash(5 层)延迟约 5.2→5.5ms,几乎不变——5 层 DFlash 出 16 个 token 仍比 1 层 EAGLE-3 出 4 个 token 还快。</text>')

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-3.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
