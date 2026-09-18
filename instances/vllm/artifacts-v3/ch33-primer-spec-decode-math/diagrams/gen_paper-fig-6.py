#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 论文精髓图 ③ · paper-fig-6（arXiv:2607.05147 Fig.6 忠实重绘）

writer figure-requests.json add：可靠性图（Alpaca）——原始置信头判别力强
（ROC-AUC 0.81–0.90）但系统性过自信（ECE 3%–8%，高置信段预测压过实测）；STS
校准后贴近对角线（平均 ECE ~1%），前缀存活概率对齐经验接受率——「调度器需要
幅度而非排序」的图证，STS 存在的理由。

原图真相源（arXiv HTML 2607.05147v1/x6.png 抓取亲读 + paper.md §4.3.3/L572）：
- 两面板并排：左=原始置信（过自信）、右=STS 校准后；
- 横轴=预测置信（前缀存活概率），纵轴=实测接受率，对角线=完美校准；
- 原始曲线低段贴线/略高、高置信段落到对角线之下（报得乐观，最大偏差在
  x≈1.0 附近）；校准后曲线全程贴住对角线（±0.02 以内）；
- 背景灰色直方=各置信桶样本量分布（高质量段样本最多，条高为原图形态示意）。

布局与信息结构对齐原图；配色/字体套本书视觉语言；文字译中。
provenance=原论文图本身+论文 §4.3.3 文字（key_figures 豁免）。
"""
import _common33 as cc
import l0_common as lc

W = 1240
lc.reset()
MX, BXR = cc.header(
    W,
    '论文原景：可靠性图——判别力一直在线，坏的是幅度；STS 修平才敢当调度真账',
    '重绘自 arXiv:2607.05147 Fig.6（Alpaca 数据集）：横轴=预测（前缀存活）置信，纵轴=实测接受率，对角线=完美校准；背景灰直方=各置信桶样本量',
    'primer · 论文精髓图重绘')

C_RAW, C_CAL = lc.C_SAM_S, lc.C_GPU_S     # 品红=原始 / 绿=校准后（好坏一眼分）

# ---------------- 图例（顶部一行，4 项）----------------
LEG = [(C_RAW, 'line', '校准前 · 原始置信'), (C_CAL, 'line', 'STS 校准后'),
       ('#cbd5e1', 'bar', '各置信桶样本量'), (lc.C_MUTE, 'dash', '完美校准对角线')]
lx = MX + 30
for col, kind, name in LEG:
    if kind == 'line':
        lc.seg(lx, 88, lx + 26, 88, col, 2.0)
        tx0 = lx + 34
    elif kind == 'bar':
        lc.rect(lx + 4, 80, 18, 15, '#e2e8f0', col, rx=2, sw=1.0)
        tx0 = lx + 34
    else:
        lc.seg(lx, 88, lx + 26, 88, col, 1.4, dash=True)
        tx0 = lx + 34
    lc.text(tx0, 91.5, name, 9.5, lc.C_TXT, 'start', maxw=170, tag='leg:' + name[:6])
    lx = tx0 + lc.tw(name, 9.5) + 46

# ---------------- 双面板 ----------------
PW, PH = 556, 418
PTY = 104
P0X, PGAP = MX, 36
AX_L, AX_R = 70, 536           # 面板内轴线左/右留白
PY0, PY1 = 150, 478

X_L, X_H = 0.0, 1.0


def xm(px, v):
    return px + AX_L + v * (AX_R - AX_L)


def ym(v):
    return PY1 - v * (PY1 - PY0)


RAW_PTS = [(0.35, 0.41), (0.45, 0.50), (0.55, 0.57), (0.65, 0.66),
           (0.75, 0.72), (0.85, 0.77), (0.95, 0.83)]
CAL_PTS = [(0.35, 0.37), (0.45, 0.46), (0.55, 0.55), (0.65, 0.64),
           (0.75, 0.75), (0.85, 0.85), (0.95, 0.94)]
HIST_L = [0.10, 0.13, 0.18, 0.28, 0.45, 0.70, 0.95]     # 原图形态：高置信桶样本最多
HIST_R = [0.07, 0.10, 0.15, 0.24, 0.42, 0.72, 1.00]

PANELS = [
    (P0X, '校准前 · 原始置信', C_RAW, RAW_PTS, HIST_L,
     [('ROC-AUC 0.81–0.90：判别力强（排序准）', lc.C_TXT, True),
      ('但 ECE 3%–8%：系统性过自信（幅度偏乐观）', '#dc2626', True)]),
    (P0X + PW + PGAP, 'STS 校准后 · 贴回对角线', C_CAL, CAL_PTS, HIST_R,
     [('平均 ECE ~1%：贴回对角线', C_CAL, True),
      ('排序不变、幅度修平——调度要的幅度可用', lc.C_MUTE, False)]),
]

for px, title, ccol, pts, hist, notes in PANELS:
    lc.rect(px, PTY, PW, PH, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
    lc.text(px + 14, 126, title, 11, ccol, 'start', True, maxw=PW - 28,
            tag='pt:' + title[:6])
    # 背景直方（先画，垫底）：0.3..1.0 七桶
    binw = (AX_R - AX_L) * 0.7 / 7
    for i, frac in enumerate(hist):
        bx = xm(px, 0.3 + i * 0.1) + 2
        bh = frac * (PY1 - PY0 - 10)
        lc.rect(bx, PY1 - bh, binw - 4, bh, '#e2e8f0', '#cbd5e1', rx=2, sw=1.0)
    # 网格 + 轴
    for gv in (0.2, 0.4, 0.6, 0.8, 1.0):
        lc.seg(px + AX_L, ym(gv), px + AX_R, ym(gv), '#eef2f7', 1.0)
    lc.seg(px + AX_L, PY0, px + AX_L, PY1, lc.C_MUTE, 1.4)
    lc.seg(px + AX_L, PY1, px + AX_R, PY1, lc.C_MUTE, 1.4)
    for gv in (0.2, 0.4, 0.6, 0.8, 1.0):
        lc.text(px + AX_L - 8, ym(gv) + 3, '%.1f' % gv, 8.5, lc.C_MUTE, 'end',
                tag='yt%d' % int(gv * 10))
        xg = xm(px, gv)
        lc.seg(xg, PY1, xg, PY1 + 5, lc.C_MUTE, 1.0)
        lc.text(xg, PY1 + 18, ('%.1f' % gv).rstrip('0').rstrip('.') or '0', 8.5,
                lc.C_MUTE, 'middle', tag='xt%d' % int(gv * 10))
    # 对角线（直方之后、曲线之前）
    lc.seg(px + AX_L, PY1, px + AX_R, PY0, lc.C_MUTE, 1.4, dash=True)
    lc.text(xm(px, 0.37), ym(0.68), '完美校准', 8.5, lc.C_FAINT, 'start',
            halo=True, tag='diag')
    # 可靠性曲线
    for i in range(len(pts) - 1):
        lc.seg(xm(px, pts[i][0]), ym(pts[i][1]), xm(px, pts[i + 1][0]),
               ym(pts[i + 1][1]), ccol, 2.2)
    for xv, yv in pts:
        lc.ELEMS.append(((xm(px, xv) - 4, ym(yv) - 4, xm(px, xv) + 4, ym(yv) + 4),
                         '<circle cx="%.1f" cy="%.1f" r="3.6" fill="#ffffff" '
                         'stroke="%s" stroke-width="2.0"/>' % (xm(px, xv), ym(yv), ccol)))
    # 面板注记（顶部两行；高置信桶直方在右、低桶矮，注记区干净）
    for j, (s, col, bold) in enumerate(notes):
        lc.text(px + AX_L + 12, PY0 + 22 + j * 18, s, 9, col, 'start', bold,
                maxw=AX_R - AX_L - 24, tag='note%d' % j)
    # 轴标题
    lc.text(px + (AX_L + AX_R) / 2, PY1 + 40, '预测置信（前缀存活概率）', 8.5,
            lc.C_MUTE, 'middle', maxw=300, tag='xlab')
    lc.text(px + AX_L - 2, 143, '实测接受率', 9, lc.C_TXT, 'end', True,
            maxw=90, tag='ylab')

# 左面板专项：高置信段落差标注（x=0.95 处，曲线点到对角线的竖直落差）
lp = P0X
lc.seg(xm(lp, 0.95), ym(0.83), xm(lp, 0.95), ym(0.95), '#dc2626', 1.4, dash=True)
lc.text(xm(lp, 0.90) - 8, ym(0.885), '高置信段：报得乐观', 8.5, '#dc2626',
        'end', True, maxw=130, tag='gap')

# ---------------- 结论 + 页脚 ----------------
CY = PTY + PH + 26
cc.conclusion(MX, BXR, CY,
              '判别力一直在线（ROC-AUC 0.81–0.90），坏的是幅度——STS 修平（ECE 3%–8% → ~1%），存活概率才从「排序信号」变成「调度器敢用的真账」。')
cc.footer(MX, BXR, CY + 22, [
    'ROC-AUC 0.81–0.90 / 原始 ECE 3%–8% / STS 后平均 ECE ~1% 引论文 §4.3.3（Alpaca）· 曲线走势（低段贴线、高置信段落到对角线下=过自信）与背景直方形态=论文 Fig.6 原图',
    '重绘自 arXiv:2607.05147 Fig.6 · 布局与信息结构对齐论文原图（arXiv HTML 原图抓取核对）· 直方条高为原图形态示意（原图无独立数值刻度）'])

H = CY + 22 + 2 * 14 + 24
cc.write('paper-fig-6', W, H)
