#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M0·步2 机制图 ch33-fig-tau-break-even（模板 before-after）

claim：τ 是盈亏开关：τ=3 时 L=3.0（2.666667x 加速）、τ=1.2 时 L=7.5（1.066667x，
几乎白干）；盈亏平衡点 τ=1.125=(T_draft+T_verify)/T_verify——接受率低时投机纯亏。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑）。
"""
import _common33 as cc
import l0_common as lc

W = 1060
lc.reset()
MX, BXR = cc.header(
    W,
    'τ 是盈亏开关：同一套周期开销（1+8 ms），drafter 好坏只改分母 τ',
    'τ=3 → L=3.0 ms/token（加速 2.666667 倍）；τ=1.2 → L=7.5 ms/token（加速 1.066667 倍，几乎白干）',
    '放大自 L0 采样出口列 · spec decode 块的延迟账轴')

# ---------------- 两栏对照（左差右好）----------------
PY, PH = 84, 200
PW = (BXR - MX - 24) / 2
panels = [
    (MX, '差 drafter · τ=1.2', '大多候选被拒，每周期只多拿 1.2 个 token', 7.5,
     '7.5 ms/token', '1.066667x', '只比基线快 6%', lc.C_ABORT),
    (MX + PW + 24, '好 drafter · τ=3', '每周期稳拿 3 个 token', 3.0,
     '3.0 ms/token', '2.666667x', '对基线 8.0 加速 2.67 倍', lc.C_SAM_S),
]
BAR_MAX = 8.0
for px, title, sub, v, vlab, spd, spds, col in panels:
    lc.rect(px, PY, PW, PH, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
    lc.text(px + 14, PY + 22, title, 11, lc.C_TXT, 'start', True, maxw=PW - 28, tag='p:t')
    lc.text(px + 14, PY + 40, sub, 9, lc.C_MUTE, 'start', maxw=PW - 28, tag='p:s')
    bx, bw, by0 = px + 18, PW - 150, PY + 58
    for i, (name, val, c, f) in enumerate([
            ('基线（逐 token）', 8.0, lc.C_MUTE, '#f1f5f9'),
            ('投机 L', v, col, ('#fef2f2' if col == lc.C_ABORT else lc.C_SAM_F))]):
        y = by0 + i * 46
        lc.text(bx, y - 4, name, 9, lc.C_TXT, 'start', tag=name)
        lc.rect(bx, y, bw, 20, '#f8fafc', lc.C_MUTE, rx=3, sw=0.8)
        lc.rect(bx, y, bw * val / BAR_MAX, 20, f, c, rx=3, sw=1.3)
        lc.text(bx + bw * val / BAR_MAX + 8, y + 14, ('8.0' if i == 0 else vlab), 10,
                (lc.C_MUTE if i == 0 else col), 'start', True, tag='v%d' % i)
    lc.text(px + 14, PY + PH - 40, spd, 15, col, 'start', True, tag='spd')
    lc.text(px + 14, PY + PH - 20, spds, 9.5, lc.C_MUTE, 'start', tag='spds')

# ---------------- τ 轴（底部，含平衡点）----------------
AY = PY + PH + 46
AX0, AX1 = MX + 60, BXR - 40
TAU0, TAU1 = 1.0, 3.2
def tx_(t):
    return AX0 + (AX1 - AX0) * (t - TAU0) / (TAU1 - TAU0)

# 纯亏区底带（τ<1.125 红）与赚区（绿）
lc.rect(tx_(TAU0), AY - 26, tx_(1.125) - tx_(TAU0), 26, '#fef2f2', lc.C_ABORT,
        rx=3, sw=1.0, dash=True)
lc.text((tx_(TAU0) + tx_(1.125)) / 2, AY - 44, '纯亏区', 9, lc.C_ABORT, 'middle',
        True, tag='zone:loss')
lc.text(tx_(2.4), AY - 44, '赚区（τ 越大赚越多）', 9, lc.C_GPU_S, 'middle', True,
        tag='zone:gain')
# 轴线与刻度
lc.seg(AX0, AY, AX1, AY, lc.C_MUTE, 1.6)
for t, lab in ((1.0, 'τ=1'), (1.125, None), (1.2, None), (3.0, None), (3.2, None)):
    lc.seg(tx_(t), AY, tx_(t), AY + 6, lc.C_MUTE, 1.0)
lc.text(AX1, AY + 20, 'τ（每周期平均拿到的 token 数）', 9, lc.C_MUTE, 'end', tag='ax:l')
# 平衡点（虚线，右侧标注）
lc.seg(tx_(1.125), AY - 72, tx_(1.125), AY, lc.C_ABORT, 1.6, dash=True)
lc.text(tx_(1.125) + 8, AY - 56, '盈亏平衡 τ=1.125', 9.5, lc.C_ABORT, 'start', True,
        maxw=150, tag='be:t')
lc.text(tx_(1.125) + 8, AY - 42, '=(T_draft+T_verify)/T_verify', 8.5, lc.C_ABORT,
        'start', maxw=170, tag='be:f')
lc.text(tx_(1.125) + 8, AY - 28, '=(1.0+8.0)/8.0', 8.5, lc.C_ABORT, 'start',
        tag='be:f2')
# 两栏 τ 的落点（下方标注，错开高度防撞）
lc.seg(tx_(1.2), AY, tx_(1.2), AY + 22, lc.C_ABORT, 1.6)
lc.text(tx_(1.2) + 8, AY + 34, 'τ=1.2（左栏）', 9, lc.C_ABORT, 'start', True, tag='dp1')
lc.seg(tx_(3.0), AY, tx_(3.0), AY + 10, lc.C_SAM_S, 1.6)
lc.text(tx_(3.0), AY + 34, 'τ=3（右栏）', 9, lc.C_SAM_S, 'middle', True, tag='dp2')

# ---------------- 结论 + 预告 + 页脚 ----------------
CY = AY + 62
cc.conclusion(MX, BXR, CY,
              '低于 1.125，draft 前向的开销赚不回来——投机纯亏；接受率这枚分母，就是后面谱系与置信调度全部努力的去处。')
lc.text(MX, CY + 18, '预告 ch34：vLLM 落地侧因此允许动态调 num_speculative_tokens（按负载收缩验证长度）。',
        9, lc.C_GPU_S, 'start', tag='fwd')
cc.footer(MX, BXR, CY + 36, [
    '延迟 L 与平衡点 1.125 均为固定 seed 的 CPU 参考实现实跑（T_draft=1.0/T_verify=8.0 为示教单位，比例才有意义）· Eq.(1) 引 arXiv:2607.05147 §2.1',
    '行号基线 vLLM v0.27.1'])

H = CY + 36 + 30
cc.write('ch33-fig-tau-break-even', W, H)
