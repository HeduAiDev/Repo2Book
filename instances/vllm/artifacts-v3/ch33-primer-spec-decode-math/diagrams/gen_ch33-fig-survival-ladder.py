#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M2·步1 机制图 ch33-fig-survival-ladder（模板 state-table）

claim：τ 的账是前缀存活连乘：α=0.8 三级阶梯 a=[0.8, 0.64, 0.512]，
E[τ]=1+Σa=2.952=闭式 (1−0.4096)/0.2，MC 30000 周期 2.954967 对拍；
首位保底 1=恢复/bonus。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑）。
"""
import _common33 as cc
import l0_common as lc

W = 1060
lc.reset()
MX, BXR = cc.header(
    W,
    'τ 的账是前缀存活连乘：位 k 能活，要求前 k 个引线全部成功',
    'α=0.8 的三级阶梯 0.8 → 0.64 → 0.512；期望产出 = 保底的 1 + 三级阶梯之和 = 2.952（γ=3）',
    '放大自 L0 采样出口列 · 接受长度产出量 τ（品红）')

# ---------------- 阶梯柱 ----------------
SC = 240.0
BASE = 352
STEPS = [
    ('保底 1', 1.0, '1.0', '首拒恢复 / 全收 bonus'),
    ('a_1', 0.8, '0.8', 'α'),
    ('a_2', 0.64, '0.64', 'α×α'),
    ('a_3', 0.512, '0.512', 'α×α×α'),
]
BW, GAP, X0 = 110, 46, 120
for i, (name, v, vs, sub) in enumerate(STEPS):
    x = X0 + i * (BW + GAP)
    cxx = x + BW / 2
    first = (i == 0)
    lc.rect(x, BASE - v * SC, BW, v * SC, ('#eff6ff' if first else lc.C_SAM_F),
            ('#2563eb' if first else lc.C_SAM_S), rx=4, sw=1.8)
    lc.text(cxx, BASE - v * SC - 24, vs, 13,
            ('#2563eb' if first else lc.C_SAM_S), 'middle', True, tag='v%d' % i)
    lc.text(cxx, BASE - v * SC / 2, name, 11.5, lc.C_TXT, 'middle', True, tag='n%d' % i)
    lc.text(cxx, BASE - v * SC / 2 + 18, sub, 8.5,
            lc.C_MUTE, 'middle', maxw=BW, tag='s%d' % i)
    lc.text(cxx, BASE + 18, ('位 0' if first else '位 %d' % i), 9, lc.C_MUTE, 'middle',
            tag='x%d' % i)
# 连乘阶梯的下降箭头（柱顶之间）
for i in range(3):
    x1 = X0 + i * (BW + GAP) + BW
    y1 = BASE - STEPS[i][1] * SC
    x2 = X0 + (i + 1) * (BW + GAP)
    y2 = BASE - STEPS[i + 1][1] * SC
    lc.seg(x1 + 6, y1 - 8, x2 - 6, y2 - 8, lc.C_MUTE, 1.4, 'std')
# α=1 参考线（上限）
lc.seg(X0 - 24, BASE - SC, X0 + 4 * BW + 3 * GAP - 6, BASE - SC, '#2563eb', 1.6,
       dash=True)
lc.text(X0 + 4 * BW + 3 * GAP + 2, BASE - SC + 3, 'α=1（draft 完美）：阶梯不衰，上限 γ+1=4.0', 9,
        '#2563eb', 'start', maxw=BXR - (X0 + 4 * BW + 3 * GAP + 2), tag='cap')

# ---------------- 右侧对账面板 ----------------
PX = X0 + 4 * BW + 3 * GAP + 26
PW = BXR - PX
PY, PH = 130, 224
lc.rect(PX, PY, PW, PH, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(PX + 14, PY + 22, '期望接受长度', 10.5, lc.C_TXT, 'start', True, maxw=PW - 28,
        tag='acc:t')
lc.text(PX + 14, PY + 46, 'E[τ] = 1 + Σ a_k', 11, lc.C_SAM_S, 'start', True,
        maxw=PW - 28, tag='acc:f')
lc.text(PX + 14, PY + 66, '　　= 1 + 0.8 + 0.64 + 0.512', 9.5, '#334155', 'start',
        maxw=PW - 28, tag='acc:l1')
lc.text(PX + 14, PY + 90, '2.952', 22, lc.C_SAM_S, 'start', True, tag='acc:big')
lc.text(PX + 14, PY + 118, '闭式 (1−α^{γ+1})/(1−α) = (1−0.4096)/0.2 = 2.952', 9.2,
        '#334155', 'start', maxw=PW - 28, tag='acc:l2')
lc.text(PX + 14, PY + 138, 'MC 30000 周期 = 2.954967（对拍一致）', 9.2, '#334155',
        'start', maxw=PW - 28, tag='acc:l3')
lc.seg(PX + 14, PY + 154, PX + PW - 14, PY + 154, lc.C_MUTE, 1.0)
lc.text(PX + 14, PY + 174, '单调不增的 a 是下一节贪心调度的合法性来源；', 8.8,
        lc.C_MUTE, 'start', maxw=PW - 28, tag='acc:l4')
lc.text(PX + 14, PY + 190, '位 1 是杠杆最大的位置（首位被拒整块作废）', 8.8,
        lc.C_MUTE, 'start', maxw=PW - 28, tag='acc:l5')

# ---------------- 结论 + 页脚 ----------------
CY = BASE + 46
cc.conclusion(MX, BXR, CY,
              '「保底 1」与「连乘阶梯」两个组成缺一不可：位越靠后存活概率越低——但位 1 的杠杆反而最大。')
cc.footer(MX, BXR, CY + 22, [
    '数值=固定 seed 的 CPU 参考实现实跑（α=0.8、γ=3；MC 30000 周期）· E[τ]=Σ_k ∏_{i≤k} α_i 的链式几何账引 arXiv:2607.05147 §3.2.2',
    '行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-survival-ladder', W, H)
