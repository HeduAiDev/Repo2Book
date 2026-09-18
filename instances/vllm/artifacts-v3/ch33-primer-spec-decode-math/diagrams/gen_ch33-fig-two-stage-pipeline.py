#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M3·步1 机制图 ch33-fig-two-stage-pipeline（模板 flow）

claim：两阶段的分工账：并行骨干 1 次前向出 U 形状 [3, 6]（γ=3；γ=5 时 [5, 6]
且调用数仍 1），序列头 γ 次左到右、每次 O(r) 查表+O(rV) GEMV——draft_tokens=
[1,2,2] 全部来自 U_k+B(prev) 的逐步合成。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑 + pin 源码锚点）。
"""
import _common33 as cc
import l0_common as lc

W = 1160
lc.reset()
MX, BXR = cc.header(
    W,
    'DSpark 两阶段：并行骨干 1 次前向出 U，序列头 γ 次查表加法左到右采样',
    '输入（anchor-as-first）=[2, -1, -1]（anchor+2 个 MASK）→ U 形状 [3, 6]；把块长加到 5，前向调用数还是 1——T_draft 近独立于块长',
    '放大自 L0 GPU 执行臂 · draft 段 DSpark 两阶段（绿）')

# ---------------- 左：并行骨干 ----------------
LW = 400
lc.rect(MX, 88, LW, 268, '#ffffff', lc.C_GPU_S, rx=8, sw=1.6)
lc.text(MX + 14, 110, '阶段一 · 并行骨干（干重活）', 11, lc.C_GPU_S, 'start', True,
        maxw=LW - 28, tag='b:t')
# 输入串
lc.text(MX + 14, 138, '输入串（anchor-as-first）', 8.5, lc.C_MUTE, 'start', tag='b:in')
for i, (v, is_anchor) in enumerate([('2', True), ('-1', False), ('-1', False)]):
    x = MX + 150 + i * 54
    lc.rect(x, 124, 46, 24, (lc.C_GPU_F if is_anchor else '#f1f5f9'),
            lc.C_GPU_S, rx=4, sw=1.5)
    lc.text(x + 23, 140, v, 9.5, lc.C_GPU_S, 'middle', True, tag='in%d' % i)
lc.text(MX + 150 + 3 * 54 + 6, 140, '（-1=MASK）', 7.8, lc.C_MUTE, 'start', tag='b:mask')
lc.text(MX + 150 + 23, 164, 'anchor', 7.8, lc.C_GPU_S, 'middle', tag='b:anch')
# 前向箭头 → U 矩阵
U_X, U_Y = MX + 60, 196
cell = 22
lc.seg(MX + LW / 2 - 10, 160, U_X + 3 * cell / 2, U_Y - 6, lc.C_GPU_S, 1.8, 'std')
for r in range(3):
    for c in range(6):
        v = 0.55 + 0.45 * ((r * 6 + c) % 7) / 6.0
        lc.ELEMS.append(((U_X + c * cell - 1, U_Y + r * cell - 1,
                          U_X + (c + 1) * cell - 1, U_Y + (r + 1) * cell - 1),
                         '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                         'fill="rgb(%d,%.0f,%d)" stroke="#ffffff" stroke-width="1"/>' % (
                             U_X + c * cell, U_Y + r * cell, cell - 1, cell - 1,
                             int(209 + 20 * v), 250 - 60 * v, int(235 - 40 * v))))
lc.text(U_X + 3 * cell, U_Y + 3 * cell + 16, 'U（base logits）· 形状 [3, 6]', 9.5,
        lc.C_GPU_S, 'middle', True, maxw=220, tag='b:U')
lc.rect(MX + LW - 128, 96, 114, 22, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=1.4)
lc.text(MX + LW - 71, 111, '前向调用数 = 1', 8.8, lc.C_GPU_S, 'middle', True,
        maxw=106, tag='b:call')
lc.text(MX + 14, 300, 'γ=5 复跑：输入 [2,-1,-1,-1,-1]、U 形状 [5, 6]，', 8.5,
        '#334155', 'start', maxw=LW - 28, tag='b:g5')
lc.text(MX + 14, 316, '前向调用数仍 = 1（块长白加，时间形状不变）', 8.5, lc.C_GPU_S,
        'start', True, maxw=LW - 28, tag='b:g6')

# ---------------- 右：序列头 ----------------
RX = MX + LW + 24
RW = BXR - RX
lc.rect(RX, 88, RW, 268, '#ffffff', lc.C_SAM_S, rx=8, sw=1.6)
lc.text(RX + 14, 110, '阶段二 · 序列头（串行补轻活）', 11, lc.C_SAM_S, 'start', True,
        maxw=RW - 28, tag='s:t')
STEPS = ['查 W_1[prev]', 'bias=W_2·W_1[prev]', '加到 U_k 上', '采样']
bw_ = (RW - 28 - 3 * 10) / 4
for si, s in enumerate(STEPS):
    x = RX + 14 + si * (bw_ + 10)
    lc.rect(x, 128, bw_, 40, '#fdf2f8', lc.C_SAM_S, rx=6, sw=1.4)
    lc.text(x + bw_ / 2, 152, s, 8.8, lc.C_SAM_S, 'middle', True, maxw=bw_ - 6,
            tag='s:s%d' % si)
    if si < 3:
        lc.seg(x + bw_, 148, x + bw_ + 10, 148, lc.C_SAM_S, 1.4, 'std')
# 三个步进框（位 0/1/2），prev 箭头首尾相接
for i in range(3):
    y = 190 + i * 52
    x0 = RX + 14
    lc.rect(x0, y, RW - 28, 42, '#ffffff', lc.C_MUTE, rx=6, sw=1.1)
    lc.text(x0 + 12, y + 25, '位 %d' % i, 9, lc.C_TXT, 'start', True, tag='p%d' % i)
    lc.text(x0 + 52, y + 25, 'softmax(U_%d + B(prev)) → 采出' % i, 9, lc.C_SAM_S,
            'start', True, maxw=RW - 120, tag='pf%d' % i)
    out = [1, 2, 2][i]
    lc.text(x0 + RW - 44, y + 25, '→ %d' % out, 10, lc.C_SAM_S, 'end', True,
            tag='po%d' % i)
    if i < 2:
        lc.seg(x0 + RW - 100, y + 42, x0 + RW - 100, y + 52, lc.C_SAM_S, 1.6, 'std')
        lc.text(x0 + RW - 106, y + 49, 'prev', 7.2, lc.C_SAM_S, 'end', tag='pv%d' % i)
lc.text(RX + 14, 348, 'draft_tokens = [1, 2, 2]（全部来自 U_k+B(prev) 的逐步合成）', 9,
        lc.C_TXT, 'start', True, maxw=RW - 28, tag='s:dt')

# ---------------- 底部：vLLM 形态 ----------------
BY = 88 + 268 + 16
lc.rect(MX, BY, BXR - MX, 70, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 22, 'vLLM 形态：_sample_sequential 的 for i in range(n_spec) 循环——logits_i = base_logits[:, i] + bias；循环末尾 prev = draft_sampled_i（串行依赖的全部实现就这一行）', 9.5,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='v:t')
lc.text(MX + 14, BY + 44, 'FULL CUDA graph 连这个 python 循环一起捕获（整步画进图里重放）· 序列循环延迟开销 0.2%~1.3%', 9,
        '#334155', 'start', maxw=BXR - MX - 28, tag='v:l')
lc.text(MX + 14, BY + 60, 'vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:L100-L149（L122 加法、L149 prev 赋值）与模块 docstring L22-L23', 8.5,
        lc.C_FAINT, 'start', maxw=BXR - MX - 28, tag='v:f')

# ---------------- 结论 + 页脚 ----------------
CY = BY + 70 + 22
cc.conclusion(MX, BXR, CY,
              '并行干重活、串行补轻活：骨干的 1 次前向给全块打底，序列头每步只花一次查表加一次 GEMV 把依赖补上。')
cc.footer(MX, BXR, CY + 22, [
    '输入串/U 形状/调用数/draft_tokens=固定 seed 的 CPU 参考实现实跑 · 两阶段与 Eq.(4) 引 arXiv:2607.05147 §3.1',
    '行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-two-stage-pipeline', W, H)
