#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M6·步1 机制图 ch33-fig-chain-rule-both-ways（模板 flow）

claim：链式法则双向图：正向 cumprod 把条件置信 c=[0.8,0.9,0.5] 累积成前缀存活
a=[0.8,0.72,0.36]（调度排序用）；逆向逐位相除 p_i/p_(i−1) 把 a 拆回 c（vLLM
synthetic 模式的注入方向）——往返逐位恒等，前位为 0 时守卫置 0。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑）。
"""
import _common33 as cc
import l0_common as lc

W = 1120
lc.reset()
MX, BXR = cc.header(
    W,
    '同一条链式法则的两个行进方向：论文正向连乘，vLLM 逆向逐位相除',
    '正向 [0.8, 0.9, 0.5] → [0.8, 0.72, 0.36]（调度排序用的前缀存活）；逆向原样拆回——往返逐位恒等',
    '放大自 L0 采样出口列↔调度带交界 · 置信→调度信号链')

# ---------------- 双链 ----------------
CS = [0.8, 0.9, 0.5]
AS_ = [0.8, 0.72, 0.36]
CX0, C_GAP, CW, CH = MX + 90, 190, 130, 54
CY_C, CY_A = 108, 268
for i in range(3):
    x = CX0 + i * C_GAP
    # c 节点
    lc.rect(x, CY_C, CW, CH, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.7)
    lc.text(x + CW / 2, CY_C + 24, 'c_%d' % (i + 1), 9.5, lc.C_MUTE, 'middle',
            tag='cn%d' % i)
    lc.text(x + CW / 2, CY_C + 44, '%g' % CS[i], 13, lc.C_GPU_S, 'middle', True,
            tag='cv%d' % i)
    # a 节点
    lc.rect(x, CY_A, CW, CH, lc.C_SAM_F, lc.C_SAM_S, rx=8, sw=1.7)
    lc.text(x + CW / 2, CY_A + 24, 'a_%d' % (i + 1), 9.5, lc.C_MUTE, 'middle',
            tag='an%d' % i)
    lc.text(x + CW / 2, CY_A + 44, '%g' % AS_[i], 13, lc.C_SAM_S, 'middle', True,
            tag='av%d' % i)
    # 正向箭头（×，粗）
    lc.seg(x + CW / 2 - 26, CY_C + CH, x + CW / 2 - 26, CY_A, lc.C_GPU_S, 2.4,
           'std')
    lc.text(x + CW / 2 - 34, (CY_C + CH + CY_A) / 2 + 3, '×', 12, lc.C_GPU_S,
            'end', True, tag='fx%d' % i)
    # 逆向箭头（÷，细虚）
    lc.seg(x + CW / 2 + 26, CY_A, x + CW / 2 + 26, CY_C + CH, lc.C_SAM_S, 1.4,
           'std', dash=True)
    lc.text(x + CW / 2 + 34, (CY_C + CH + CY_A) / 2 + 3, '÷', 12, lc.C_SAM_S,
            'start', True, tag='rv%d' % i)
# 方向标签
lc.text(CX0 - 60, (CY_C + CH + CY_A) / 2 - 12, '正向', 9.5, lc.C_GPU_S, 'middle',
        True, maxw=70, tag='dl:fw')
lc.text(CX0 - 60, (CY_C + CH + CY_A) / 2 + 8, '连乘 cumprod', 8, lc.C_GPU_S,
        'middle', maxw=70, tag='dl:fw2')
lc.text(CX0 + 2 * C_GAP + CW / 2 + 26, CY_C - 14, '逆向：c_i = p_i / p_{i-1}', 8.5,
        lc.C_SAM_S, 'middle', True, maxw=190, tag='dl:rv')
# a 链解释
lc.text(CX0, CY_A + CH + 20, 'a 单调不增——下一节贪心调度的合法性来源', 8.5,
        lc.C_MUTE, 'start', maxw=400, tag='a:note')
# 正向乘积注
lc.text(CX0, CY_C - 14, '0.8　→　0.8×0.9=0.72　→　0.72×0.5=0.36', 8.5,
        lc.C_GPU_S, 'start', True, maxw=430, tag='fw:calc')

# ---------------- 右侧：守卫闸门 + vLLM 徽章 ----------------
GX = CX0 + 3 * C_GAP + 44
GW = BXR - GX
lc.rect(GX, 92, GW, 118, '#ffffff', lc.C_ABORT, rx=8, sw=1.3, dash=True)
lc.text(GX + 14, 112, '除零守卫（链子的物理）', 9.5, lc.C_ABORT, 'start', True,
        maxw=GW - 28, tag='g:t')
lc.text(GX + 14, 134, '前位存活为 0 时，后面的条件率无意义', 8.2, '#334155',
        'start', maxw=GW - 28, tag='g:l1')
lc.text(GX + 14, 156, 'a = [0.5, 0.0, 0.3] → c = [0.5, 0.0, 0.0]', 9.5,
        lc.C_ABORT, 'start', True, maxw=GW - 28, tag='g:l2')
lc.text(GX + 14, 178, '（约定置 0，不除零）', 8, lc.C_MUTE, 'start', maxw=GW - 28,
        tag='g:l3')
lc.rect(GX, 226, GW, 108, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(GX + 14, 246, 'vLLM 同构（相向而行）', 9.5, lc.C_TXT, 'start', True,
        maxw=GW - 28, tag='v:t')
lc.text(GX + 14, 268, 'unconditional_to_conditional_rates', 8.5, lc.C_SAM_S,
        'start', True, maxw=GW - 28, tag='v:s')
lc.text(GX + 14, 288, 'docstring：「c_i = p_i / p_{i-1}」', 8.2, '#334155',
        'start', maxw=GW - 28, tag='v:l1')
lc.text(GX + 14, 308, 'synthetic 接受率模式的注入旋钮', 7.8,
        lc.C_MUTE, 'start', maxw=GW - 28, tag='v:l2')
lc.text(GX + 14, 322, 'vllm/v1/spec_decode/utils.py:L598-L601', 7.8,
        lc.C_MUTE, 'start', maxw=GW - 28, tag='v:l3')

# ---------------- 结论 + 页脚 ----------------
CY = CY_A + CH + 44
cc.conclusion(MX, BXR, CY,
              'cumprod 与逐位相除是同一条链式法则的两面：论文拿它算「谁更值得验」，vLLM 拿它注入测试用的接受率。')
cc.footer(MX, BXR, CY + 22, [
    '数值=固定 seed 的 CPU 参考实现实跑（正向/逆向/守卫逐位断言）· 前缀存活 a_{r,j}=∏_{i≤j}c_{r,i} 引 arXiv:2607.05147 §3.2.2',
    '行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-chain-rule-both-ways', W, H)
