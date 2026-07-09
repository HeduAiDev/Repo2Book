#!/usr/bin/env python3
"""paper-fig-2: 重绘自 arXiv:2502.11089 Fig.1——NSA 与全量注意力的实测对比(原图已抓到:
https://arxiv.org/html/2502.11089v2/x1.png)。信息结构对齐原图两个柱状图面板:
左panel = 通用/长文/推理三类评测分数(全量注意力 vs NSA,NSA 不输甚至超过全量);
右panel = 64K 长度下解码/前向/反向三阶段实测加速比(以全量注意力=1.0x 为基线,
柱顶加速倍数直接取自原图标注:11.6x/9.0x/6.0x)。配色套本章语言,文字译中,
provenance=原论文本身(豁免 explainer figure_specs/spec.numbers 通道)。
全坐标由循环/常量计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK, SUB = "#0f172a", "#64748b"
FULL_COLOR, NSA_COLOR = "#fdba74", "#ef4444"
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

# ---------- 左 panel:整体测评分数(近似复原原图相对高度,原图无逐柱数字标注) ----------
LEFT_CATS = ["通用能力\nGeneral", "长文本\nLongBench", "推理\nReasoning"]
LEFT_FULL = [0.443, 0.437, 0.092]
LEFT_NSA = [0.456, 0.469, 0.146]
LEFT_YMAX = 0.5
LEFT_TICKS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# ---------- 右 panel:64K 长度三阶段实测加速比(倍数直接取自原图标注) ----------
RIGHT_CATS = ["解码 Decode", "前向 Forward", "反向 Backward"]
RIGHT_FULL = [1.0, 1.0, 1.0]
RIGHT_NSA = [11.6, 9.0, 6.0]
RIGHT_YMAX = 13.0
RIGHT_TICKS = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0]


def bar_chart(x0, y0, title, cats, series_full, series_nsa, y_max, ticks, value_fmt, show_bar_labels):
    """绘制一个双柱分组图,返回追加的 SVG 元素列表。"""
    out = []
    chart_x = x0 + AXIS_LABEL_W
    chart_w = PANEL_W - AXIS_LABEL_W
    chart_y = y0 + LEGEND_H + 24
    chart_h = CHART_H

    out.append(f'<text x="{x0:.1f}" y="{y0+14}" font-family="sans-serif" font-size="14" '
               f'font-weight="bold" fill="{INK}">{esc(title)}</text>')

    # 图例
    leg_y = y0 + LEGEND_H
    out.append(f'<rect x="{chart_x:.1f}" y="{leg_y-11}" width="16" height="14" fill="{FULL_COLOR}" stroke="#c2410c" stroke-width="1"/>')
    out.append(f'<text x="{chart_x+22:.1f}" y="{leg_y}" font-family="sans-serif" font-size="11.5" fill="{INK}">全量注意力 Full Attention</text>')
    out.append(f'<rect x="{chart_x+185:.1f}" y="{leg_y-11}" width="16" height="14" fill="{NSA_COLOR}" stroke="#991b1b" stroke-width="1"/>')
    out.append(f'<text x="{chart_x+207:.1f}" y="{leg_y}" font-family="sans-serif" font-size="11.5" fill="{INK}">NSA</text>')

    # 网格线 + y 轴刻度
    for t in ticks:
        gy = chart_y + chart_h - (t / y_max) * chart_h
        out.append(f'<line x1="{chart_x:.1f}" y1="{gy:.1f}" x2="{chart_x+chart_w:.1f}" y2="{gy:.1f}" '
                   f'stroke="{GRID_COLOR}" stroke-width="1" stroke-dasharray="3,3"/>')
        out.append(f'<text x="{chart_x-8:.1f}" y="{gy+4:.1f}" text-anchor="end" '
                   f'font-family="sans-serif" font-size="10.5" fill="{SUB}">{value_fmt(t)}</text>')
    # 坐标轴
    out.append(f'<line x1="{chart_x:.1f}" y1="{chart_y:.1f}" x2="{chart_x:.1f}" y2="{chart_y+chart_h:.1f}" stroke="{INK}" stroke-width="1.3"/>')
    out.append(f'<line x1="{chart_x:.1f}" y1="{chart_y+chart_h:.1f}" x2="{chart_x+chart_w:.1f}" y2="{chart_y+chart_h:.1f}" stroke="{INK}" stroke-width="1.3"/>')

    n = len(cats)
    group_w = chart_w / n
    bar_w = group_w * 0.3
    bar_gap = group_w * 0.06

    for i, cat in enumerate(cats):
        gx0 = chart_x + i * group_w
        gcx = gx0 + group_w / 2
        bx_full = gcx - bar_w - bar_gap / 2
        bx_nsa = gcx + bar_gap / 2

        h_full = (series_full[i] / y_max) * chart_h
        h_nsa = (series_nsa[i] / y_max) * chart_h
        y_full = chart_y + chart_h - h_full
        y_nsa = chart_y + chart_h - h_nsa

        out.append(f'<rect x="{bx_full:.1f}" y="{y_full:.1f}" width="{bar_w:.1f}" height="{h_full:.1f}" '
                   f'fill="{FULL_COLOR}" stroke="#c2410c" stroke-width="1"/>')
        out.append(f'<rect x="{bx_nsa:.1f}" y="{y_nsa:.1f}" width="{bar_w:.1f}" height="{h_nsa:.1f}" '
                   f'fill="{NSA_COLOR}" stroke="#991b1b" stroke-width="1"/>')

        if show_bar_labels:
            out.append(f'<text x="{bx_nsa+bar_w/2:.1f}" y="{y_nsa-8:.1f}" text-anchor="middle" '
                       f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                       f'fill="#991b1b">{value_fmt(series_nsa[i])}</text>')

        # x 轴分类标签(两行)
        lines = cat.split("\n")
        ly = chart_y + chart_h + 18
        for li, line in enumerate(lines):
            out.append(f'<text x="{gcx:.1f}" y="{ly+li*15}" text-anchor="middle" '
                       f'font-family="sans-serif" font-size="11.5" fill="{INK}">{esc(line)}</text>')

    return out, chart_y + chart_h + 40


elems = []
_, left_bottom = None, None
left_elems, left_bottom = bar_chart(
    LEFT_X, CONTENT_TOP, "整体测评:通用 / 长文本 / 推理(分数)",
    LEFT_CATS, LEFT_FULL, LEFT_NSA, LEFT_YMAX, LEFT_TICKS,
    lambda v: f"{v:.1f}", show_bar_labels=False)
right_elems, right_bottom = bar_chart(
    RIGHT_X, CONTENT_TOP, "64K 长度三阶段实测加速比(基线=1.0x)",
    RIGHT_CATS, RIGHT_FULL, RIGHT_NSA, RIGHT_YMAX, RIGHT_TICKS,
    lambda v: f"{v:.1f}", show_bar_labels=True)

H = int(max(left_bottom, right_bottom) + 70)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{PAD}" y="{TITLE_TOP}" font-family="sans-serif" font-size="18" '
         f'font-weight="bold" fill="{INK}">NSA 不输全量注意力,64K 长度下三阶段都实测加速</text>')
L.append(f'<text x="{PAD}" y="{SUBTITLE_TOP}" font-family="sans-serif" font-size="12.5" '
         f'fill="{SUB}">左:通用/长文本/推理三类评测,NSA 分数持平甚至超过全量注意力;右:64K 长度下解码 11.6x、前向 9.0x、反向 6.0x(27B 模型真实实测)</text>')
L.extend(left_elems)
L.extend(right_elems)

# 底部结论 callout
callout_y = H - 46
callout_w = W - PAD * 2
L.append(f'<rect x="{PAD}" y="{callout_y}" width="{callout_w}" height="34" rx="6" '
         f'fill="#fef2f2" stroke="#991b1b" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+16}" y="{callout_y+22}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="#991b1b">稀疏不是"精度换速度"的权衡——NSA 在这套 27B 模型实测里两头都占了</text>')

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-2.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
