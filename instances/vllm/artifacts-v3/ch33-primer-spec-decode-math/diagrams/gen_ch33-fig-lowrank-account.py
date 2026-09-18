#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M3·步3 机制图 ch33-fig-lowrank-account（模板 before-after）

claim：低秩账：全表 V×V=16713318400 项（fp16 也要 33 GB 级）vs 低秩 2Vr=66191360
项（各 33095680）——省 252.5 倍；且省法是 V>2r 制度的财产（玩具反例 V=6、r=4
时 48>36 反而更贵）。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑）。
"""
import _common33 as cc
import l0_common as lc

W = 1160
lc.reset()
MX, BXR = cc.header(
    W,
    '一阶转移偏置为什么必须低秩：B=W_1W_2，V×V 全表盘不起',
    'V=129280 的词表下，全表 16713318400 项、fp16 也要 33 GB 级；两张 33095680 参数的表省 252.5 倍（论文口径 ≈253）',
    '放大自 L0 GPU 执行臂 · 序列头存储账（绿）')

# ---------------- 左：全表 ----------------
LW = 430
lc.rect(MX, 88, LW, 280, '#ffffff', lc.C_ABORT, rx=8, sw=1.2, dash=True)
lc.text(MX + 14, 110, '全表存法 · B ∈ R^{V×V}', 10.5, lc.C_ABORT, 'start', True,
        maxw=LW - 28, tag='f:t')
GX, GY, CELL, N = MX + 66, 128, 13, 13
for r in range(N):
    for c in range(N):
        v = (r * N + c) % 9 / 9.0
        lc.ELEMS.append(((GX + c * CELL - 1, GY + r * CELL - 1,
                          GX + (c + 1) * CELL - 1, GY + (r + 1) * CELL - 1),
                         '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                         'fill="rgb(254,%.0f,%.0f)" stroke="#ffffff" stroke-width="0.6"/>' % (
                             GX + c * CELL, GY + r * CELL, CELL - 1, CELL - 1,
                             226 - 90 * v, 226 - 150 * v)))
lc.text(GX + N * CELL / 2, GY + N * CELL + 18, 'V × V = 16713318400 项', 10,
        lc.C_ABORT, 'middle', True, maxw=LW - 80, tag='f:n')
lc.text(GX + N * CELL / 2, GY + N * CELL + 38, '（示意 13×13 网格；fp16 也要 33 GB 级）', 8.5,
        lc.C_MUTE, 'middle', maxw=LW - 80, tag='f:s')
lc.text(MX + 14, 354, '「每个前驱对每个后继的偏好」直接成表——盘不起', 8.5,
        lc.C_ABORT, 'start', maxw=LW - 28, tag='f:n2')

# ---------------- 右：低秩 ----------------
RX = MX + LW + 26
RW = BXR - RX
lc.rect(RX, 88, RW, 280, '#ffffff', lc.C_GPU_S, rx=8, sw=1.6)
lc.text(RX + 14, 110, '低秩存法 · B = W_1 W_2（r=256）', 10.5, lc.C_GPU_S, 'start',
        True, maxw=RW - 28, tag='l:t')
# W_1 竖长条 / r 窄缝 / W_2 横长条
W1_X, W1_Y, W1_W, W1_H = RX + 60, 132, 46, 118
W2_X, W2_Y, W2_W, W2_H = RX + 60, 274, 300, 40
lc.rect(W1_X, W1_Y, W1_W, W1_H, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.8)
lc.text(W1_X + W1_W / 2, W1_Y + W1_H / 2, 'W_1', 11, lc.C_GPU_S, 'middle', True,
        tag='l:w1')
lc.text(W1_X + W1_W + 10, W1_Y + 14, 'V × r', 8.5, lc.C_MUTE, 'start', tag='l:w1d')
lc.text(W1_X + W1_W + 10, W1_Y + 28, '嵌入查表', 8, lc.C_MUTE, 'start', tag='l:w1s')
lc.rect(W2_X, W2_Y, W2_W, W2_H, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.8)
lc.text(W2_X + W2_W / 2, W2_Y + W2_H / 2, 'W_2', 11, lc.C_GPU_S, 'middle', True,
        tag='l:w2')
lc.text(W2_X + W2_W + 8, W2_Y + 14, 'r × V', 8.5, lc.C_MUTE, 'start', tag='l:w2d')
lc.text(W2_X + W2_W + 8, W2_Y + 28, 'logit 投影', 8, lc.C_MUTE, 'start', tag='l:w2s')
# r 窄缝连接示意
lc.seg(W1_X + W1_W / 2, W1_Y + W1_H, W2_X + 24, W2_Y, lc.C_GPU_S, 1.2, dash=True)
lc.text(700, 266, '秩 r=256 的窄缝', 8.5, lc.C_GPU_S, 'middle', tag='l:r')
lc.text(RX + 14, 336, '2Vr = 66191360 项（各 33095680）', 10, lc.C_GPU_S, 'start',
        True, maxw=RW - 28, tag='l:n')
lc.text(RX + 14, 354, '省 252.5 倍 = V/(2r)（论文口径 ≈253）', 10.5, lc.C_GPU_S,
        'start', True, maxw=RW - 28, tag='l:ratio')

# ---------------- 底部：玩具反例 ----------------
BY = 88 + 280 + 16
lc.rect(MX, BY, BXR - MX, 66, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 22, '反例提醒：省法是 V>2r 制度的财产——玩具 V=6、r=4 时低秩 48 项 > 全表 36 项（V/(2r)=0.75，反而更贵）', 10,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='c:t')
lc.text(MX + 14, BY + 42, 'V=6、r=2 时 24 < 36（1.5）才划算；真实词表 V=129280 ≫ 2r=512，绰绰有余', 9,
        lc.C_MUTE, 'start', maxw=BXR - MX - 28, tag='c:l')

# ---------------- 结论 + 页脚 ----------------
CY = BY + 66 + 22
cc.conclusion(MX, BXR, CY,
              'V×V 的全表是维度灾难，W_1W_2 把它压成两张 V×r 的瘦表——前提只有一个：词表远大于秩。')
cc.footer(MX, BXR, CY + 22, [
    '数值=固定 seed 的 CPU 参考实现实跑（V=129280、r=256；比值 V/(2r)=252.5）· 低秩分解 Eq.(5) 引 arXiv:2607.05147 §3.1',
    '行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-lowrank-account', W, H)
