#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M1·步4 机制图 ch33-fig-lossless-two-cases（模板 before-after）

claim：无损的两情形直证在一张对账表里：每 token 接受段+残差段=p（0.2+0.3=0.5、
0.1+0.3=0.4、0.1+0=0.1），且拒绝质量与残差质量是同一个数 0.6=½‖p−q‖₁。
例：p=(0.5,0.4,0.1)（target）vs q=(0.2,0.1,0.7)（draft）。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑）。
"""
import _common33 as cc
import l0_common as lc

W = 1120
lc.reset()
MX, BXR = cc.header(
    W,
    '无损的两情形直证：接受段 + 残差段 = p，逐 token 全对上',
    '例 p_t=(0.5, 0.4, 0.1) vs p_d=(0.2, 0.1, 0.7)：p≥p_d 的坐标两段相加=p；p<p_d 的坐标残差段为 0、接受段按比值缩放后也恰好=p',
    '放大自 L0 采样出口列 ·「保分布不变」那句承诺本身（品红）')

SC = 300.0     # 概率 1.0 的像素高
BASE = 282     # 柱基线
TOKS = [
    ('token 0', 0.5, 0.2, 0.3, '0.2+0.3=0.5'),
    ('token 1', 0.4, 0.1, 0.3, '0.1+0.3=0.4'),
    ('token 2', 0.1, 0.7, 0.0, '0.1+0=0.1'),
]
PX0, SLOT, BW = MX + 30, 236, 96
# 两个情形的背景分区框（左两柱=p≥p_d，右一柱=p<p_d）
lc.rect(PX0 - 6, 88, 2 * SLOT, 252, '#ffffff', lc.C_GPU_S, rx=8, sw=1.2, dash=True)
lc.rect(PX0 + 2 * SLOT - 6, 88, SLOT, 252, '#ffffff', lc.C_ABORT, rx=8, sw=1.2, dash=True)
lc.text(PX0 - 6 + SLOT, 104, '情形一 p_t ≥ p_d：残差段补上差额', 9.5, lc.C_GPU_S,
        'middle', True, maxw=2 * SLOT - 16, tag='case1')
lc.text(PX0 + 2.5 * SLOT - 6, 104, '情形二 p_t < p_d：残差为 0', 9.5, lc.C_ABORT,
        'middle', True, maxw=SLOT - 16, tag='case2')

for i, (name, pt, pd, res, formula) in enumerate(TOKS):
    cxx = PX0 + SLOT * (i + 0.5)
    acc = min(pt, pd)
    # 接受段（品红实心，底部）
    lc.rect(cxx - BW / 2, BASE - acc * SC, BW, acc * SC, lc.C_SAM_F, lc.C_SAM_S,
            rx=3, sw=1.5)
    # 残差段（琥珀，其上）
    if res > 0:
        lc.rect(cxx - BW / 2, BASE - pt * SC, BW, res * SC, '#fde68a', '#d97706',
                rx=3, sw=1.2)
    else:
        # 情形二：残差 0（打叉示意）
        lc.text(cxx, BASE - pt * SC - 12, '残差 0', 8.5, lc.C_ABORT, 'middle', tag='z%d' % i)
    # p 的总高标线（点到 p 高度）
    lc.seg(cxx - BW / 2 - 14, BASE - pt * SC, cxx + BW / 2 + 14, BASE - pt * SC,
           lc.C_MUTE, 1.6, dash=True)
    # 段内标注
    if acc >= 0.1:
        lc.text(cxx, BASE - acc * SC / 2 + 3, '接受 %.1f' % acc, 9, lc.C_SAM_S,
                'middle', True, tag='a%d' % i)
    if res > 0:
        lc.text(cxx, BASE - acc * SC - res * SC / 2 + 3, '残差 %.1f' % res, 9,
                '#d97706', 'middle', True, tag='rr%d' % i)
    lc.text(cxx, BASE + 18, name, 9.5, lc.C_TXT, 'middle', True, tag='n%d' % i)
    lc.text(cxx, BASE + 36, formula, 10.5, lc.C_TXT, 'middle', True, tag='f%d' % i)
    lc.text(cxx + BW / 2 + 18, BASE - pt * SC + 3, 'p=%.1f' % pt, 8.5, lc.C_MUTE,
            'start', tag='p%d' % i)

# ---------------- 底部总账带：两半相等 ----------------
BY = 88 + 252 + 34
lc.rect(MX, BY, BXR - MX, 96, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 20, '全局对账：拒绝掉的总质量 = 残差总质量 = ½‖p−p_d‖₁ —— 「被拒的概率」与「从哪补回来」是同一个数', 10,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='bal:t')
BAND_Y, BAND_H, BAND_X0, BAND_W = BY + 34, 26, MX + 60, 420
acc_w, rej_w = BAND_W * 0.4, BAND_W * 0.6
lc.rect(BAND_X0, BAND_Y, acc_w, BAND_H, lc.C_SAM_F, lc.C_SAM_S, rx=3, sw=1.3)
lc.text(BAND_X0 + acc_w / 2, BAND_Y + 17, '接受质量 Σmin=0.4', 9, lc.C_SAM_S,
        'middle', True, tag='b1')
lc.rect(BAND_X0 + acc_w, BAND_Y, rej_w, BAND_H, '#fef2f2', lc.C_ABORT, rx=3, sw=1.3)
lc.text(BAND_X0 + acc_w + rej_w / 2, BAND_Y + 17, '拒绝质量 0.6', 9, lc.C_ABORT,
        'middle', True, tag='b2')
lc.text(BAND_X0 + acc_w + rej_w + 20, BAND_Y + 10, '＝', 13, lc.C_TXT, 'start', True,
        tag='eq')
lc.rect(BAND_X0 + acc_w + rej_w + 44, BAND_Y, rej_w, BAND_H, '#fef3c7', '#d97706',
        rx=3, sw=1.3)
lc.text(BAND_X0 + acc_w + rej_w + 44 + rej_w / 2, BAND_Y + 17, '残差质量 0.6（½‖p−p_d‖₁）', 9,
        '#d97706', 'middle', True, tag='b3')
lc.text(BAND_X0 + 60, BY + 84, '另一组 (A,B)：Σmin=0.8、拒绝质量 0.2、½‖·‖₁=0.2 —— 同一对账', 9,
        lc.C_MUTE, 'start', maxw=BXR - MX - 80, tag='bal2')

# ---------------- 结论 + 页脚 ----------------
CY = BY + 96 + 22
cc.conclusion(MX, BXR, CY,
              '每个 token 的产出概率 = 直接接受段 + 拒绝后残差段——这就是「加速零质量损失」的全部数学。')
cc.footer(MX, BXR, CY + 22, [
    '数值=固定 seed 的 CPU 参考实现实跑（逐 token 对账逐值断言）· 定理两情形直证引 arXiv:2607.05147 §2.1（源头 Leviathan et al. 2023）',
    '行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-lossless-two-cases', W, H)
