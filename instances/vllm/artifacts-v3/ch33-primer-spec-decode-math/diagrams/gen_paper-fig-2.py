#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 论文精髓图 ② · paper-fig-2（arXiv:2607.05147 Fig.2 忠实重绘）

writer figure-requests.json add：逐位条件接受率（分母只计前缀全接受的实例，剥离
前缀错误后的真实位质量）——自回归 Eagle3 平稳或回升（Chat 0.53→0.74）、并行
DFlash 后缀衰减（Chat 0.72→0.63、Code 0.87→0.78）、DSpark 两全（Math 起步 0.93
且整块平稳）；位 1 容量差 Math 0.88 vs 0.81、Chat 0.72 vs 0.53。

原图真相源（arXiv HTML 2607.05147v1/x2.png 抓取亲读 + paper.md §4.3.1）：
- 三面板横排：Math / Code / Chat；横轴=草稿位 1..7，纵轴=条件接受率 0.4..1.0；
- 每面板三条曲线：Eagle3 / DFlash / DSpark，图例居顶部一行；
- 端点标值=论文实测（§4.3.1 文字给定的 9 个值）；未标端点与中间位按论文定性
  形态（平稳/回升/衰减）连线，页脚整体声明。

布局与信息结构对齐原图；配色/字体套本书视觉语言（DFlash=绿=GPU 一次前向、
Eagle3=蓝、DSpark=品红=本章主角色，与两阶段图骨干绿/序列头品红同源）。
provenance=原论文图本身+论文 §4.3.1 文字（key_figures 豁免）。
"""
import _common33 as cc
import l0_common as lc

W = 1440
lc.reset()
MX, BXR = cc.header(
    W,
    '论文原景：逐位条件接受率——自回归稳或升、并行后缀衰减、DSpark 两全',
    '重绘自 arXiv:2607.05147 Fig.2：分母只计前缀全接受的实例（剥离前缀错误后的真实位质量）· Qwen3-4B target · 端点标值=论文实测，中间位按论文定性形态连线',
    'primer · 论文精髓图重绘')

# 三 drafter：色系与本章两阶段图同源（骨干绿 / 主角色品红）
C_EAGLE, C_DFLASH, C_DSPARK = lc.C_API_S, lc.C_GPU_S, lc.C_SAM_S
SERIES = [
    ('Eagle3（自回归·浅）', C_EAGLE,
     [0.81, 0.815, 0.82, 0.826, 0.832, 0.838, 0.844],
     [0.84, 0.844, 0.849, 0.855, 0.862, 0.870, 0.879],
     [0.53, 0.556, 0.585, 0.615, 0.648, 0.685, 0.74]),
    ('DFlash（并行·深）', C_DFLASH,
     [0.88, 0.871, 0.860, 0.848, 0.834, 0.818, 0.80],
     [0.87, 0.861, 0.850, 0.839, 0.826, 0.810, 0.78],
     [0.72, 0.712, 0.702, 0.690, 0.676, 0.660, 0.63]),
    ('DSpark（半自回归）', C_DSPARK,
     [0.93, 0.928, 0.926, 0.924, 0.922, 0.920, 0.918],
     [0.90, 0.898, 0.896, 0.893, 0.891, 0.889, 0.886],
     [0.71, 0.708, 0.706, 0.703, 0.701, 0.699, 0.697]),
]
# 论文实测标值（claim 的 9 个数）：(域, drafter 序号, 位, 值, 放置方式)
LABELS = [
    ('Math', 2, 1, '0.93', 'above'),
    ('Math', 1, 1, '0.88', 'left'),
    ('Math', 0, 1, '0.81', 'below'),
    ('Code', 1, 1, '0.87', 'above'),
    ('Code', 1, 7, '0.78', 'above'),
    ('Chat', 0, 1, '0.53', 'below'),
    ('Chat', 0, 7, '0.74', 'above'),
    ('Chat', 1, 1, '0.72', 'above'),
    ('Chat', 1, 7, '0.63', 'below'),
]
NOTES = {
    'Math': '位 1 容量差：并行 0.88 vs 自回归 0.81 · DSpark 0.93 起步、整块稳',
    'Code': '并行后缀衰减 0.87→0.78 · 自回归稳或回升 · DSpark 高位平稳',
    'Chat': '自回归回升 0.53→0.74 · 并行衰减 0.72→0.63 · 位 1 容量差 0.72 vs 0.53',
}
DOMAINS = ['Math', 'Code', 'Chat']

# ---------------- 图例（顶部一行）----------------
lx = 470
for name, col, *_ in SERIES:
    lc.seg(lx, 88, lx + 26, 88, col, 2.0)
    lc.ELEMS.append(((lx + 13 - 4, 88 - 4, lx + 13 + 4, 88 + 4),
                     '<circle cx="%.1f" cy="%.1f" r="3.4" fill="#ffffff" '
                     'stroke="%s" stroke-width="1.8"/>' % (lx + 13, 88, col)))
    lc.text(lx + 34, 91.5, name, 9.5, lc.C_TXT, 'start', maxw=170,
            tag='leg:' + name[:6])
    lx += 34 + lc.tw(name, 9.5) + 44
lc.text(BXR, 91.5, 'Qwen3-4B target', 8.5, lc.C_FAINT, 'end', tag='leg:model')

# ---------------- 三面板 ----------------
PW, PH = 430, 400
PTY = 110
P0X, PGAP = MX, 24
AX_L, AX_B = 50, 410          # 面板内轴线左/右留白
PY0, PY1 = 158, 440           # 纵轴 0.4→PY1、1.0→PY0
Y_LO, Y_HI = 0.4, 1.0


def ymap(v):
    return PY1 - (v - Y_LO) / (Y_HI - Y_LO) * (PY1 - PY0)


for pi, dom in enumerate(DOMAINS):
    px = P0X + pi * (PW + PGAP)
    pw_r = px + PW
    lc.rect(px, PTY, PW, PH, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
    lc.text(px + 14, 130, dom, 11, lc.C_TXT, 'start', True, maxw=120,
            tag='pt:' + dom)
    lc.text(px + 14, 148, NOTES[dom], 8.5, lc.C_MUTE, 'start',
            maxw=PW - 28, tag='pn:' + dom)
    # 网格 + 轴
    for gv in (0.6, 0.8, 1.0):
        lc.seg(px + AX_L, ymap(gv), px + AX_B, ymap(gv), '#e2e8f0', 1.0)
    lc.seg(px + AX_L, PY0, px + AX_L, PY1, lc.C_MUTE, 1.4)
    lc.seg(px + AX_L, PY1, px + AX_B, PY1, lc.C_MUTE, 1.4)
    for gv in (0.4, 0.6, 0.8, 1.0):
        lc.text(px + AX_L - 8, ymap(gv) + 3, '%.1f' % gv, 8.5, lc.C_MUTE,
                'end', tag='yt%d%s' % (int(gv * 10), dom))
    for k in range(1, 8):
        xk = px + AX_L + 30 + (k - 1) * 50
        lc.seg(xk, PY1, xk, PY1 + 5, lc.C_MUTE, 1.0)
        lc.text(xk, PY1 + 18, str(k), 8.5, lc.C_MUTE, 'middle', tag='xt%d%s' % (k, dom))
    lc.text(px + (AX_L + AX_B) / 2, PY1 + 40, '草稿位 k', 8.5, lc.C_MUTE,
            'middle', tag='xl:' + dom)
    # 三条曲线
    for si, (name, col, math_v, code_v, chat_v) in enumerate(SERIES):
        vals = (math_v, code_v, chat_v)[pi]
        pts = [(px + AX_L + 30 + (k - 1) * 50, ymap(v)) for k, v in enumerate(vals, 1)]
        for i in range(6):
            lc.seg(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], col, 1.8)
        for x, y in pts:
            lc.ELEMS.append(((x - 4, y - 4, x + 4, y + 4),
                             '<circle cx="%.1f" cy="%.1f" r="3.4" fill="#ffffff" '
                             'stroke="%s" stroke-width="1.8"/>' % (x, y, col)))
    # 论文实测标值
    col_of = [C_EAGLE, C_DFLASH, C_DSPARK]
    for ldom, si, k, txt, how in LABELS:
        if ldom != dom:
            continue
        xk = px + AX_L + 30 + (k - 1) * 50
        v = SERIES[si][1 + 1 + pi][k - 1]
        yk = ymap(v)
        if how == 'above':
            lc.text(xk, yk - 13, txt, 9, col_of[si], 'middle', True, maxw=44,
                    tag='lb%s%d' % (dom, si))
        elif how == 'below':
            lc.text(xk, yk + 17, txt, 9, col_of[si], 'middle', True, maxw=44,
                    tag='lb%s%d' % (dom, si))
        else:  # left
            lc.text(xk - 8, yk + 3, txt, 9, col_of[si], 'end', True, maxw=44,
                    tag='lb%s%d' % (dom, si))
lc.text(MX + 4, 104, '条件接受率', 9.5, lc.C_TXT, 'start', True, maxw=90,
        tag='ylab')

# ---------------- 结论 + 页脚 ----------------
CY = PTY + PH + 20
cc.conclusion(MX, BXR, CY,
              '位 1 容量差（Math 0.88 vs 0.81、Chat 0.72 vs 0.53）× 前缀生存杠杆 = 并行总 τ 反超自回归；DSpark 高位起步 + 序列头压衰减，两全。')
cc.footer(MX, BXR, CY + 22, [
    '标值端点=论文实测（位置 1 与位置 7，共 9 个值）· 未标端点与中间位按论文定性形态（平稳/回升/衰减）连线 · 指标定义：分母只计前缀 1..k-1 全接受的实例',
    '重绘自 arXiv:2607.05147 Fig.2 · 布局与信息结构对齐论文原图（三域面板 × 三 drafter 曲线 · arXiv HTML 原图抓取核对）· 数据 provenance=论文 §4.3.1 + 原图'])

H = CY + 22 + 2 * 14 + 24
cc.write('paper-fig-2', W, H)
