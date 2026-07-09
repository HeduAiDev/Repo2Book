#!/usr/bin/env python3
"""paper-fig-1: 重绘自 arXiv:2606.19348 Fig.1(仅重绘右半:inference FLOPs / KV cache
对比曲线;原图已抓到:https://arxiv.org/html/2606.19348v1/x1.png,左半 benchmark
柱状图与本章无关,不重绘)。信息结构对齐原图右半两个折线面板:上——单 token 推理
FLOPs(T)随 token 位置(K)变化,V3.2(灰虚线)最陡、V4-Pro(深蓝)与 V4-Flash(浅蓝)
明显更平;下——累计 KV cache(GB)随序列长度(K)变化,同样三条线。标注的
「3.7x lower / 9.8x lower / 9.5x smaller / 13.7x smaller」四个倍数原样取自原图——
这是原图给出的实测结论性数字,曲线本身为示意折线(按标注倍数反推端点,保比例关系,
非逐点复刻原图像素轨迹),provenance=原论文本身。全坐标由循环/常量计算。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


INK, SUB = "#0f172a", "#64748b"
V32 = "#94a3b8"      # DeepSeek-V3.2:灰虚线(基线)
PRO = "#1d4ed8"      # DeepSeek-V4-Pro:深蓝实线
FLASH = "#60a5fa"    # DeepSeek-V4-Flash:浅蓝实线
GRID = "#e2e8f0"
ARROW = "#64748b"

W = 900
PAD_L, PAD_R = 74, 40
CHART_W = W - PAD_L - PAD_R
CHART_H = 220
GAP_BETWEEN = 90
TOP0 = 172           # 第一个面板(FLOPs)绘图区顶边(留足legend与面板标题的间距)
TOP1 = TOP0 + CHART_H + GAP_BETWEEN  # 第二个面板(KV cache)绘图区顶边
H = TOP1 + CHART_H + 70

X_TICKS = [0, 256, 512, 768, 1024]


def esc_(s):
    return esc(s)


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{ARROW}"/></marker></defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{PAD_L}" y="34" font-family="sans-serif" font-size="16" '
          f'font-weight="bold" fill="{INK}">V4 系列 vs V3.2:单 token 推理 FLOPs 与累计 KV cache 实测对比(仅重绘 Fig.1 右半)</text>')
L.append(f'<text x="{PAD_L}" y="56" font-family="sans-serif" font-size="12" '
          f'fill="{SUB}">曲线为示意折线(按原图标注倍数反推端点比例,非逐点复刻像素轨迹);四个倍数标注原样取自原图</text>')


def chart(top, y_max, y_ticks, y_unit, title, series, note):
    """series: list of (name, color, dashed, end_value, label or None)。x 固定 0..1024(K)。"""
    x0, y0 = PAD_L, top + CHART_H   # 原点(左下)
    # 标题
    L.append(f'<text x="{x0:.1f}" y="{top-14}" font-family="sans-serif" font-size="13.5" '
              f'font-weight="bold" fill="{INK}">{esc_(title)}</text>')
    # 网格 + y 轴刻度
    for yt in y_ticks:
        y = y0 - (yt / y_max) * CHART_H
        L.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+CHART_W}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        L.append(f'<text x="{x0-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" '
                  f'font-size="10.5" fill="{SUB}">{yt:g}</text>')
    # y 轴单位不用旋转文字(几何 linter 不识别 transform,且标题已含单位)——省略,靠标题里的
    # 「FLOPs(T)」「KV cache(GB)」与 y_unit 形参(仅供调用处语义留档,不渲染)传达单位
    _ = y_unit
    # x 轴
    L.append(f'<line x1="{x0}" y1="{y0}" x2="{x0+CHART_W}" y2="{y0}" stroke="{INK}" stroke-width="1.4"/>')
    for xt in X_TICKS:
        x = x0 + (xt / 1024) * CHART_W
        L.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+5}" stroke="{INK}" stroke-width="1.2"/>')
        L.append(f'<text x="{x:.1f}" y="{y0+18}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="10.5" fill="{SUB}">{xt}</text>')

    def px(xv, yv):
        return x0 + (xv / 1024) * CHART_W, y0 - (yv / y_max) * CHART_H

    for name, color, dashed, start_v, end_v, label, label_dy in series:
        x1, y1 = px(0, start_v)
        x2, y2 = px(1024, end_v)
        dash = ' stroke-dasharray="7,5"' if dashed else ''
        L.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                  f'stroke="{color}" stroke-width="2.6"{dash}/>')
        L.append(f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="3.5" fill="{color}"/>')
        if label:
            lx, ly = x2 - 8, y2 + label_dy
            box_w = 13 * len(label) * 0.62 + 14
            # 连接虚线:标签框 -> 端点(端点与标签垂直距离较大时,补一条细连线避免误读)
            L.append(f'<line x1="{lx:.1f}" y1="{ly+4:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                      f'stroke="{color}" stroke-width="1" stroke-dasharray="2,3"/>')
            L.append(f'<rect x="{lx-box_w:.1f}" y="{ly-14:.1f}" width="{box_w:.1f}" height="22" rx="5" '
                      f'fill="{color}"/>')
            L.append(f'<text x="{lx-box_w/2:.1f}" y="{ly+1:.1f}" text-anchor="middle" font-family="sans-serif" '
                      f'font-size="11.5" font-weight="bold" fill="white">{esc_(label)}</text>')

    # x 轴标题
    L.append(f'<text x="{x0+CHART_W/2:.1f}" y="{y0+40}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11.5" fill="{SUB}">{esc_(note)}</text>')


chart(TOP0, 1.2, [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], "Single-Token FLOPs (T)",
      "单 token 推理 FLOPs(T)随 token 位置(K)——V4 系列明显更平",
      [
          ("DeepSeek-V3.2", V32, True, 0.05, 1.15, None, 0),
          ("DeepSeek-V4-Pro", PRO, False, 0.05, 1.15 / 3.7, "3.7x lower", -34),
          ("DeepSeek-V4-Flash", FLASH, False, 0.02, 1.15 / 9.8, "9.8x lower", 26),
      ],
      "Token Position (K)")

chart(TOP1, 40, [0, 10, 20, 30, 40], "Accumulated KV Cache (GB)",
      "累计 KV cache(GB)随序列长度(K)——V4 系列增速大幅放缓",
      [
          ("DeepSeek-V3.2", V32, True, 0.2, 38, None, 0),
          ("DeepSeek-V4-Pro", PRO, False, 0.1, 38 / 9.5, "9.5x smaller", -34),
          ("DeepSeek-V4-Flash", FLASH, False, 0.05, 38 / 13.7, "13.7x smaller", -68),
      ],
      "Sequence Length (K)")

# 图例(共用,画在两个面板之间的空档上方一点,顶部标题区下方)
leg_y = 88
leg_items = [("DeepSeek-V3.2(基线,虚线)", V32, True), ("DeepSeek-V4-Pro", PRO, False), ("DeepSeek-V4-Flash", FLASH, False)]
lx = PAD_L
for name, color, dashed in leg_items:
    dash = ' stroke-dasharray="6,4"' if dashed else ''
    L.append(f'<line x1="{lx}" y1="{leg_y}" x2="{lx+28}" y2="{leg_y}" stroke="{color}" stroke-width="3"{dash}/>')
    L.append(f'<text x="{lx+34}" y="{leg_y+4}" font-family="sans-serif" font-size="11.5" fill="{INK}">{esc(name)}</text>')
    lx += 34 + len(name) * 11.2 + 26

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-1.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
