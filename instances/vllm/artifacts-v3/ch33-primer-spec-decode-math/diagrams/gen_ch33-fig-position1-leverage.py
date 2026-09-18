#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M2·步2 机制图 ch33-fig-position1-leverage（模板 before-after）

claim：位 1 杠杆：DFlash 衰减曲线总 τ 2.751257 反超 EAGLE3 回升曲线 2.218724——
只把首位 0.53 换成 0.72（其余不动）就 +0.436901；前缀验证下首位被拒全块作废，
首位优势被后位复利放大。
数字全部取 spec.numbers（端点=论文 Fig.2 实测，中间位插值已在 params 注明）。
"""
import _common33 as cc
import l0_common as lc

W = 1160
lc.reset()
MX, BXR = cc.header(
    W,
    '位 1 杠杆：逐位看 EAGLE3 回升该赢，但 τ 的账是连乘——DFlash 反超',
    '首位被拒整块作废：首位优势被后面每一位复利放大；换头实验把这件事隔离出来（+0.436901 > 修后三位）',
    '放大自 L0 采样出口列 · τ 在两代 drafter 上的分岔')

PANELS = [
    (MX, 'EAGLE3（浅自回归）· α 逐位回升', [0.53, 0.6, 0.67, 0.74],
     [0.53, 0.318, 0.21306, 0.157664], '2.218724'),
    (MX + (BXR - MX) / 2 + 12, 'DFlash（深并行）· α 逐位衰减', [0.72, 0.69, 0.66, 0.63],
     [0.72, 0.4968, 0.327888, 0.206569], '2.751257'),
]
PW = (BXR - MX - 12) / 2
PTY, PTH = 84, 320
A_SC, A_TOP, A_Y0 = 220.0, 0.4, 196      # α 折线：y = A_Y0-(α-A_TOP)*A_SC
B_BASE, B_SC = 344, 130.0                # a 柱基线与比例

for px, title, alphas, avals, etau in PANELS:
    lc.rect(px, PTY, PW, PTH, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
    lc.text(px + 14, PTY + 20, title, 10.5, lc.C_TXT, 'start', True, maxw=PW - 28,
            tag='t:' + title[:6])
    X0, SLOT = px + 56, (PW - 84) / 4
    pts = [(X0 + SLOT * (i + 0.5), A_Y0 - (a - A_TOP) * A_SC, a)
           for i, a in enumerate(alphas)]
    # α 折线
    for i in range(3):
        lc.seg(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], lc.C_GPU_S, 1.8)
    for i, (x, y, a) in enumerate(pts):
        lc.ELEMS.append(((x - 4, y - 4, x + 4, y + 4),
                         '<circle cx="%.1f" cy="%.1f" r="3.6" fill="#ffffff" stroke="%s" stroke-width="1.8"/>' % (x, y, lc.C_GPU_S)))
        lc.text(x, y - 10, '%.2f' % a, 8.8, lc.C_GPU_S, 'middle', True, tag='al%d' % i)
        lc.text(x, A_Y0 + 14, '位 %d' % (i + 1), 8.5, lc.C_MUTE, 'middle', tag='ax%d' % i)
    lc.text(px + PW - 12, PTY + 36, 'α 曲线', 9, lc.C_GPU_S, 'end', True, tag='lga')
    # 分隔与 a 柱（连乘阶梯）
    lc.seg(px + 14, A_Y0 + 26, px + PW - 14, A_Y0 + 26, '#e2e8f0', 1.0)
    for i, v in enumerate(avals):
        x = X0 + SLOT * (i + 0.5)
        bw = 56
        lc.rect(x - bw / 2, B_BASE - v * B_SC, bw, v * B_SC, '#fde68a', '#d97706',
                rx=3, sw=1.3)
        lc.text(x, B_BASE - v * B_SC - 8, '%g' % v, 8.2, '#d97706', 'middle', True,
                maxw=70, tag='av%d' % i)
    lc.text(px + PW - 12, A_Y0 + 42, '连乘阶梯 a_k=∏α', 9, '#d97706', 'end', True,
            tag='lgb')
    # E[τ] 徽章（柱基线之下，面板右下角）
    ex = px + PW - 14
    lc.text(ex, PTY + PTH - 34, 'E[τ] = 1 + Σa =', 9.5, lc.C_MUTE, 'end', tag='e1')
    lc.text(ex, PTY + PTH - 10, etau, 20, lc.C_SAM_S, 'end', True, tag='e2')

# ---------------- 换头实验条（底部）----------------
SY = PTY + PTH + 18
lc.rect(MX, SY, BXR - MX, 96, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, SY + 24, '换头实验：只把 EAGLE3 曲线的首位 0.53 换成 0.72、后三位不动——', 10,
        lc.C_TXT, 'start', True, maxw=470, tag='sw:t1')
lc.text(MX + 14, SY + 44, 'E[τ] 立刻 +0.436901，比修后三位赚得多。', 10,
        lc.C_TXT, 'start', True, maxw=470, tag='sw:t2')
lc.text(MX + 14, SY + 68, '直觉判 EAGLE3 赢（后段 0.53→0.74 回升）；连乘的账说 DFlash 赢。', 9,
        lc.C_MUTE, 'start', maxw=470, tag='sw:t3')
BBASE, BSC = SY + 76, 20.0
for i, (lab, v, x) in enumerate([('原 EAGLE3', 2.218724, 620), ('换头后', 2.655626, 770)]):
    h = v * BSC
    lc.rect(x, BBASE - h, 60, h, lc.C_SAM_F, lc.C_SAM_S, rx=3, sw=1.3)
    lc.text(x + 30, BBASE + 14, lab, 8.5, lc.C_MUTE, 'middle', tag='swn%d' % i)
    lc.text(x + 30, BBASE - h - 8, '%g' % v, 9.5, lc.C_SAM_S, 'middle', True,
            tag='swv%d' % i)
lc.seg(620 + 60, BBASE - 2.218724 * BSC, 770, BBASE - 2.655626 * BSC, '#d97706',
       1.8, 'std')
lc.text(850, SY + 40, '首位 0.72 > 0.53 被每一位复利放大：', 9.5, lc.C_TXT, 'start',
        maxw=260, tag='sw:n1')
lc.text(850, SY + 60, '总 τ 2.751257 反超 2.218724', 10.5, lc.C_SAM_S, 'start', True,
        maxw=260, tag='sw:n2')

# ---------------- 结论 + 页脚 ----------------
CY = SY + 96 + 22
cc.conclusion(MX, BXR, CY,
              '位 1 是杠杆最大的位置——它被拒绝，整块 draft 全作废；谱系之争的分水岭在首位，不在后段。')
cc.footer(MX, BXR, CY + 22, [
    'α 曲线端点=论文 Fig.2 实测（Chat 域：DFlash 0.72 vs EAGLE3 0.53；Math 域：DFlash 0.88 vs EAGLE3 0.81；DSpark Math 起步 0.93），中间位为示教线性插值 · 连乘账/E[τ]/换头实验=固定 seed CPU 参考实现实跑',
    '谱系与层深/块长对齐口径引 arXiv:2607.05147 §4.1 · 行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-position1-leverage', W, H)
