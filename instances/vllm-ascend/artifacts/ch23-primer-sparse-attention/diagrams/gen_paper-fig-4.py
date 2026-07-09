#!/usr/bin/env python3
"""paper-fig-4: 重绘自 arXiv:2512.02556 Fig.3——V3.1-Terminus(稠密) vs V3.2(DSA)在 H800
上的实测 token 成本曲线(原图已抓到:https://arxiv.org/html/2512.02556v1/x3.png 与 x4.png,
两张子图(a)Prefilling/(b)Decoding)。信息结构对齐原图:两个折线面板,x 轴 token 位置
0K→128K,y 轴每百万 token 成本($),V3.1-Terminus(蓝,近线性陡升)vs V3.2(橙,远更平缓)。
配色套本章语言,文字译中,provenance=原论文本身(豁免 explainer figure_specs/spec.numbers
通道;曲线数值为读图近似,非逐字复刻像素坐标,仅保真定性形状与端点量级)。
全坐标由循环/常量计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK, SUB = "#0f172a", "#64748b"
V31_COLOR, V32_COLOR = "#2563eb", "#f97316"
GRID_COLOR = "#e2e8f0"

W = 1200
PAD = 50
TITLE_TOP, SUBTITLE_TOP, CONTENT_TOP = 34, 56, 100
PANEL_GAP = 60
PANEL_W = (W - PAD * 2 - PANEL_GAP) / 2
LEFT_X = PAD
RIGHT_X = PAD + PANEL_W + PANEL_GAP

CHART_H = 300
AXIS_LABEL_W = 46
LEGEND_H = 26

X_TICKS = [0, 32, 64, 96, 128]  # 单位:K token
X_MAX = 128

# ---------- (a) Prefilling:近似复原原图曲线端点与整体走势 ----------
PREFILL_TITLE = "(a) Prefilling"
PREFILL_YMAX = 0.7
PREFILL_TICKS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
PREFILL_V31 = [0.05, 0.22, 0.38, 0.53, 0.68]   # 对应 X_TICKS 五个采样点,近线性陡升
PREFILL_V32 = [0.05, 0.10, 0.13, 0.16, 0.19]   # 远更平缓(indexer 固定开销 + 稀疏主注意力)

# ---------- (b) Decoding:近似复原原图曲线端点与整体走势 ----------
DECODE_TITLE = "(b) Decoding"
DECODE_YMAX = 2.4
DECODE_TICKS = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
DECODE_V31 = [0.10, 0.65, 1.15, 1.65, 2.15]
DECODE_V32 = [0.08, 0.15, 0.20, 0.24, 0.28]


def money(v):
    return f"{v:.1f}$"


def line_chart(x0, y0, title, ymax, ticks, v31, v32):
    out = []
    chart_x = x0 + AXIS_LABEL_W
    chart_w = PANEL_W - AXIS_LABEL_W
    chart_y = y0 + LEGEND_H + 24
    chart_h = CHART_H

    out.append(f'<text x="{x0:.1f}" y="{y0+14}" font-family="sans-serif" font-size="14" '
               f'font-weight="bold" fill="{INK}">{esc(title)}</text>')

    leg_y = y0 + LEGEND_H
    out.append(f'<line x1="{chart_x:.1f}" y1="{leg_y-4}" x2="{chart_x+22:.1f}" y2="{leg_y-4}" '
               f'stroke="{V31_COLOR}" stroke-width="2.5"/>')
    out.append(f'<text x="{chart_x+28:.1f}" y="{leg_y}" font-family="sans-serif" font-size="11.5" '
               f'fill="{INK}">DeepSeek-V3.1-Terminus(稠密)</text>')
    out.append(f'<line x1="{chart_x+230:.1f}" y1="{leg_y-4}" x2="{chart_x+252:.1f}" y2="{leg_y-4}" '
               f'stroke="{V32_COLOR}" stroke-width="2.5"/>')
    out.append(f'<text x="{chart_x+258:.1f}" y="{leg_y}" font-family="sans-serif" font-size="11.5" '
               f'fill="{INK}">DeepSeek-V3.2(DSA)</text>')

    for t in ticks:
        gy = chart_y + chart_h - (t / ymax) * chart_h
        out.append(f'<line x1="{chart_x:.1f}" y1="{gy:.1f}" x2="{chart_x+chart_w:.1f}" y2="{gy:.1f}" '
                   f'stroke="{GRID_COLOR}" stroke-width="1" stroke-dasharray="3,3"/>')
        out.append(f'<text x="{chart_x-8:.1f}" y="{gy+4:.1f}" text-anchor="end" '
                   f'font-family="sans-serif" font-size="10" fill="{SUB}">{money(t)}</text>')
    out.append(f'<line x1="{chart_x:.1f}" y1="{chart_y:.1f}" x2="{chart_x:.1f}" y2="{chart_y+chart_h:.1f}" stroke="{INK}" stroke-width="1.3"/>')
    out.append(f'<line x1="{chart_x:.1f}" y1="{chart_y+chart_h:.1f}" x2="{chart_x+chart_w:.1f}" y2="{chart_y+chart_h:.1f}" stroke="{INK}" stroke-width="1.3"/>')

    def px(xk):
        return chart_x + (xk / X_MAX) * chart_w

    def py(v):
        return chart_y + chart_h - (v / ymax) * chart_h

    for xk in X_TICKS:
        out.append(f'<text x="{px(xk):.1f}" y="{chart_y+chart_h+18}" text-anchor="middle" '
                   f'font-family="sans-serif" font-size="11" fill="{INK}">{xk}K</text>')

    for values, color in [(v31, V31_COLOR), (v32, V32_COLOR)]:
        pts = " ".join(f"{px(xk):.1f},{py(v):.1f}" for xk, v in zip(X_TICKS, values))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for xk, v in zip(X_TICKS, values):
            out.append(f'<circle cx="{px(xk):.1f}" cy="{py(v):.1f}" r="3.5" fill="{color}"/>')

    # 端点数值标注(128K 处,2 位小数,与底部结论 callout 精度一致,避免看似"对不上")
    out.append(f'<text x="{px(128)-6:.1f}" y="{py(v31[-1])-8:.1f}" text-anchor="end" '
               f'font-family="sans-serif" font-size="11" font-weight="bold" fill="{V31_COLOR}">{v31[-1]:.2f}$</text>')
    out.append(f'<text x="{px(128)-6:.1f}" y="{py(v32[-1])+16:.1f}" text-anchor="end" '
               f'font-family="sans-serif" font-size="11" font-weight="bold" fill="{V32_COLOR}">{v32[-1]:.2f}$</text>')

    out.append(f'<text x="{chart_x+chart_w/2:.1f}" y="{chart_y+chart_h+38}" text-anchor="middle" '
               f'font-family="sans-serif" font-size="11" fill="{SUB}">Token Position</text>')

    return out, chart_y + chart_h + 52


left_elems, left_bottom = line_chart(LEFT_X, CONTENT_TOP, PREFILL_TITLE, PREFILL_YMAX, PREFILL_TICKS, PREFILL_V31, PREFILL_V32)
right_elems, right_bottom = line_chart(RIGHT_X, CONTENT_TOP, DECODE_TITLE, DECODE_YMAX, DECODE_TICKS, DECODE_V31, DECODE_V32)

H = int(max(left_bottom, right_bottom) + 60)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{PAD}" y="{TITLE_TOP}" font-family="sans-serif" font-size="18" '
         f'font-weight="bold" fill="{INK}">H800 实测:DSA 把每百万 token 成本曲线从近线性拉平</text>')
L.append(f'<text x="{PAD}" y="{SUBTITLE_TOP}" font-family="sans-serif" font-size="12.5" '
         f'fill="{SUB}">V3.1-Terminus(稠密,蓝)随 token 位置近线性陡升;V3.2(DSA,橙)prefill/decode 两阶段都远更平缓——理论 MAC 加速之外的真实部署证据</text>')
L.extend(left_elems)
L.extend(right_elems)

callout_y = H - 46
callout_w = W - PAD * 2
L.append(f'<rect x="{PAD}" y="{callout_y}" width="{callout_w}" height="34" rx="6" '
         f'fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+16}" y="{callout_y+22}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="#1e3a8a">128K 处 decode 成本降约 87%(2.15$→0.28$)——比 §六 MAC 理论加速账更直接的部署证据</text>')

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-4.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
