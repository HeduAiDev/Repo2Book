#!/usr/bin/env python3
"""paper-fig-2-smoothquant: 重绘自 arXiv:2211.10438 Figure 2 —— SmoothQuant 的核心直觉图。
(a) 原始：激活 |X| 有离群值撑爆量化范围，大部分数值只剩很少有效位，难量化；
    权重 |W| 本身很平坦，很容易量化。
(b) SmoothQuant：把 scale 的方差从激活离线搬到权重上——平滑后的 |X̂| 和调整后的 |Ŵ|
    都变得容易量化。
本图是定性直觉图（论文本身也不给具体数值坐标），量级标注（10/0.1/1/1）取自原图顶部刻度。
"""
import math
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def esc_bold(s):
    """转义并在粗体文本里把"量"字拆到 font-weight=normal 的 tspan——
    这套渲染管线(rsvg-convert)的粗体 CJK 回退字体缺"量"字形,粗体直出会变豆腐块。"""
    return '<tspan font-weight="normal">量</tspan>'.join(esc(p) for p in s.split('量'))


TITLE = "SmoothQuant 的直觉：把量化难度从激活搬到权重，两边都变得好量化"
SUBTITLE = "重绘自 arXiv:2211.10438 Figure 2"

CHART_W, CHART_H = 220, 110
PANEL_W = 260
PAD = 40
COL_GAP = 60
col_x = [PAD, PAD + PANEL_W + COL_GAP]

TOP = 108
ANN_H = 46     # 顶部标注(离群值/迁移难度)预留高度
TITLE_H = 22   # |X| 等标题
AXIS_TOP_PAD = 14  # 图表内顶部给刻度数字留白
CAP_H = 24     # hard/easy to quantize 结论行
ROWCAP_GAP = 8
ROWCAP_H = 22
ROW_GAP = 26

row_a_ann_y = TOP
row_a_title_y = row_a_ann_y + ANN_H
row_a_chart_y = row_a_title_y + TITLE_H
row_a_cap_y = row_a_chart_y + CHART_H + 22
row_a_rowcap_y = row_a_cap_y + ROWCAP_GAP + ROWCAP_H

row_b_ann_y = row_a_rowcap_y + ROW_GAP
row_b_title_y = row_b_ann_y + ANN_H
row_b_chart_y = row_b_title_y + TITLE_H
row_b_cap_y = row_b_chart_y + CHART_H + 22
row_b_rowcap_y = row_b_cap_y + ROWCAP_GAP + ROWCAP_H

w = PAD * 2 + PANEL_W * 2 + COL_GAP
h = row_b_rowcap_y + 30


def wave_values(n, bumps, bump_amp, base, spike_center=None, spike_width=0.05, spike_amp=0.0):
    vals = []
    for i in range(n + 1):
        t = i / n
        v = base + bump_amp * (0.5 + 0.5 * math.sin(t * bumps * 2 * math.pi - math.pi / 2))
        if spike_center is not None:
            d = (t - spike_center) / spike_width
            v += spike_amp * math.exp(-d * d)
        vals.append(min(v, 1.0))
    return vals


def chart_svg(x0, y0, values, stroke, fill, top_label, zero_label="0", show_axis_label=False):
    out = []
    n = len(values) - 1
    pts = []
    for i, v in enumerate(values):
        px = x0 + (i / n) * CHART_W
        py = y0 + CHART_H * (1 - v)
        pts.append((px, py))
    # axis
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+CHART_H}" stroke="#94a3b8" stroke-width="1"/>')
    out.append(f'<line x1="{x0}" y1="{y0+CHART_H}" x2="{x0+CHART_W}" y2="{y0+CHART_H}" stroke="#94a3b8" stroke-width="1"/>')
    out.append(f'<text x="{x0-6}" y="{y0+4}" text-anchor="end" font-family="sans-serif" '
               f'font-size="10" fill="#64748b">{esc(top_label)}</text>')
    out.append(f'<text x="{x0-6}" y="{y0+CHART_H+3}" text-anchor="end" font-family="sans-serif" '
               f'font-size="10" fill="#64748b">{esc(zero_label)}</text>')
    for yy in (0.25, 0.5, 0.75):
        gy = y0 + CHART_H * (1 - yy)
        out.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x0+CHART_W}" y2="{gy:.1f}" '
                   f'stroke="#e2e8f0" stroke-width="0.6"/>')
    # filled area
    area = " ".join(f'{px:.1f},{py:.1f}' for px, py in pts)
    out.append(f'<polygon points="{x0},{y0+CHART_H} {area} {x0+CHART_W},{y0+CHART_H}" '
               f'fill="{fill}" opacity="0.55"/>')
    poly = " ".join(f'{px:.1f},{py:.1f}' for px, py in pts)
    out.append(f'<polyline points="{poly}" fill="none" stroke="{stroke}" stroke-width="2"/>')
    if show_axis_label:
        # 竖排两字(不用 rotate transform——几何 linter 不解算旋转矩阵,
        # 旋转文字的水平包围盒会被误判越界;竖排短字每行都是独立的正常文字框,天然规避)
        lx = x0 - 24
        out.append(f'<text x="{lx}" y="{y0+CHART_H/2-6}" text-anchor="middle" font-family="sans-serif" '
                   f'font-size="10" fill="#64748b">量</text>')
        out.append(f'<text x="{lx}" y="{y0+CHART_H/2+10}" text-anchor="middle" font-family="sans-serif" '
                   f'font-size="10" fill="#64748b">级</text>')
    return out, pts


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
     '<marker id="ab" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0f172a"/></marker>'
     '</defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-16}" font-family="sans-serif" font-size="15.5" '
     f'fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+2}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

N = 80
# --- row (a): Original ---
x_vals = wave_values(N, bumps=7, bump_amp=0.10, base=0.05, spike_center=0.20, spike_width=0.035, spike_amp=0.95)
w_vals = wave_values(N, bumps=10, bump_amp=0.06, base=0.62)

L.append(f'<text x="{col_x[0]+8}" y="{row_a_ann_y+12}" font-family="sans-serif" font-size="11" '
         f'fill="#dc2626">离群值</text>')
spike_x = col_x[0] + 0.20 * CHART_W
L.append(f'<path d="M {col_x[0]+34} {row_a_ann_y+8} Q {spike_x-10} {row_a_ann_y+14} '
         f'{spike_x} {row_a_title_y+2}" fill="none" stroke="#dc2626" stroke-width="1.2" marker-end="url(#a)"/>')
L.append(f'<text x="{col_x[0]+120}" y="{row_a_ann_y+30}" font-family="sans-serif" font-size="10.5" '
         f'fill="#dc2626">有效位太少</text>')
for fx in (0.55, 0.72):
    tx = col_x[0] + fx * CHART_W
    L.append(f'<path d="M {col_x[0]+150} {row_a_ann_y+34} L {tx:.1f} {row_a_title_y+30}" '
             f'fill="none" stroke="#dc2626" stroke-width="1" marker-end="url(#a)"/>')

L.append(f'<text x="{col_x[0]}" y="{row_a_title_y}" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#0f172a">|X|</text>')
svg, _ = chart_svg(col_x[0], row_a_chart_y, x_vals, "#dc2626", "#fca5a5", "10", "0", show_axis_label=True)
L.extend(svg)
L.append(f'<text x="{col_x[0]+CHART_W/2}" y="{row_a_cap_y}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#b91c1c">{esc_bold("难量化")}</text>')

L.append(f'<text x="{col_x[1]}" y="{row_a_title_y}" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#0f172a">|W|</text>')
svg, _ = chart_svg(col_x[1], row_a_chart_y, w_vals, "#16a34a", "#86efac", "0.1", "0")
L.extend(svg)
L.append(f'<text x="{col_x[1]+CHART_W/2}" y="{row_a_cap_y}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#15803d">{esc_bold("很容易量化")}</text>')

row_a_span_mid = (col_x[0] + col_x[1] + CHART_W) / 2
L.append(f'<text x="{row_a_span_mid}" y="{row_a_rowcap_y}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="13" font-weight="bold" fill="#0f172a">(a) 原始</text>')

# --- row (b): SmoothQuant ---
xh_vals = wave_values(N, bumps=8, bump_amp=0.28, base=0.30, spike_center=0.20, spike_width=0.06, spike_amp=0.55)
wh_vals = wave_values(N, bumps=9, bump_amp=0.34, base=0.30)

L.append(f'<text x="{col_x[0]+30}" y="{row_b_ann_y+12}" font-family="sans-serif" font-size="11" '
         f'fill="#dc2626">已抹平</text>')
L.append(f'<path d="M {col_x[0]+30} {row_b_ann_y+16} Q {spike_x-6} {row_b_ann_y+18} '
         f'{spike_x} {row_b_title_y+2}" fill="none" stroke="#dc2626" stroke-width="1.2" marker-end="url(#a)"/>')

mig_x0 = col_x[0] + CHART_W - 30
mig_x1 = col_x[1] - 8
L.append(f'<text x="{(mig_x0+mig_x1)/2:.1f}" y="{row_b_ann_y+2}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="#0f172a">难度迁移</text>')
L.append(f'<path d="M {mig_x0} {row_b_ann_y+34} Q {(mig_x0+mig_x1)/2:.1f} {row_b_ann_y+16} '
         f'{mig_x1} {row_b_ann_y+34}" fill="none" stroke="#0f172a" stroke-width="1.3" marker-end="url(#ab)"/>')

L.append(f'<text x="{col_x[0]}" y="{row_b_title_y}" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#0f172a">|X̂|</text>')
svg, _ = chart_svg(col_x[0], row_b_chart_y, xh_vals, "#16a34a", "#86efac", "1", "0", show_axis_label=True)
L.extend(svg)
L.append(f'<text x="{col_x[0]+CHART_W/2}" y="{row_b_cap_y}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#15803d">{esc_bold("容易量化")}</text>')

L.append(f'<text x="{col_x[1]}" y="{row_b_title_y}" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#0f172a">|Ŵ|</text>')
svg, _ = chart_svg(col_x[1], row_b_chart_y, wh_vals, "#16a34a", "#86efac", "1", "0")
L.extend(svg)
L.append(f'<text x="{col_x[1]+CHART_W/2}" y="{row_b_cap_y}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#15803d">{esc_bold("容易量化")}</text>')

L.append(f'<text x="{row_a_span_mid}" y="{row_b_rowcap_y}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="13" font-weight="bold" fill="#0f172a">(b) SmoothQuant</text>')

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-2-smoothquant.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
