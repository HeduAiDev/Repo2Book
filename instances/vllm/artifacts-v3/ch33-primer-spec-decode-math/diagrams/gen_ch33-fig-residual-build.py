#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M1·步2 机制图 ch33-fig-residual-build（模板 before-after）

claim：残差分布的构造是逐坐标裁剪：p_t−p_d 的正部 max(0,·)——(A,B) 组得
(0.2, 0.0)（恢复恒 A）；三词表组得 (0.3, 0.3, 0.0)→norm(0.5, 0.5, 0.0)；
拒绝总质量恒等于 ½‖p_t−p_d‖₁（两例 0.2 与 0.6）。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑）。
"""
import _common33 as cc
import l0_common as lc

W = 1120
lc.reset()
MX, BXR = cc.header(
    W,
    '残差分布的构造：p_t − p_d 逐坐标只留正部 max(0,·)',
    'draft 更看好的坐标被削成 0（target 没那意思，不从这里恢复）；target 多出来的质量原样保留——「主编想发但编辑没推」的稿堆',
    '放大自 L0 采样出口列 · 残差恢复支路（品红）')

SC = 210.0        # 概率 1.0 的像素高
GROUPS = [
    (MX, '例 1 · 两词表 {A, B}', 'p_t=(0.7, 0.3)　p_d=(0.5, 0.5)',
     [('A', 0.7, 0.5, 0.2), ('B', 0.3, 0.5, None)],
     'residual = (0.2, 0.0)，总质量 0.2 = ½(0.2+0.2)',
     'norm(r) = (1.0, 0.0) → 恢复恒为 A'),
    (MX + (BXR - MX) / 2 + 10, '例 2 · 三词表', 'p_t=(0.5, 0.4, 0.1)　p_d=(0.2, 0.1, 0.7)',
     [('t0', 0.5, 0.2, 0.3), ('t1', 0.4, 0.1, 0.3), ('t2', 0.1, 0.7, None)],
     'residual = (0.3, 0.3, 0.0)，总质量 0.6 = ½(0.3+0.3+0.6)',
     'norm(r) = (0.5, 0.5, 0.0) → t0/t1 五五开'),
]
BASE_Y = 318       # 柱基线
for px, title, sub, toks, res_line, norm_line in GROUPS:
    pw = (BXR - MX - 10) / 2
    lc.rect(px, 92, pw, 332, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
    lc.text(px + 14, 112, title, 10.5, lc.C_TXT, 'start', True, maxw=pw - 28, tag='t')
    lc.text(px + 14, 130, sub, 9, lc.C_MUTE, 'start', maxw=pw - 28, tag='s')
    n = len(toks)
    slot = (pw - 40) / n
    for i, (name, pt, pd, r) in enumerate(toks):
        cxx = px + 20 + slot * (i + 0.5)
        bw = 40
        # p_t 柱（实心品红）
        lc.rect(cxx - bw / 2, BASE_Y - pt * SC, bw, pt * SC, lc.C_SAM_F, lc.C_SAM_S,
                rx=3, sw=1.5)
        # p_d 水平刻度线（灰虚线，跨柱）
        lc.seg(cxx - bw / 2 - 10, BASE_Y - pd * SC, cxx + bw / 2 + 10,
               BASE_Y - pd * SC, lc.C_MUTE, 1.4, dash=True)
        # 残差正部叠层（琥珀，从 p_d 到 p_t）
        if r is not None:
            top, bot = min(pt, pd), max(pt, pd)
            lc.rect(cxx - bw / 2, BASE_Y - bot * SC, bw, (bot - top) * SC,
                    '#fde68a', '#d97706', rx=2, sw=1.2)
        # 值标注
        lc.text(cxx, BASE_Y - pt * SC - 8, 'p_t=%.1f' % pt, 8.2, lc.C_SAM_S, 'middle',
                True, tag='pt%d%s' % (i, name))
        lc.text(cxx + bw / 2 + 12, BASE_Y - pd * SC + 3, 'p_d=%.1f' % pd, 8.2,
                lc.C_MUTE, 'start', tag='pd%d%s' % (i, name))
        lc.text(cxx, BASE_Y + 16, name, 10, lc.C_TXT, 'middle', True, tag='n%s' % name)
        # 残差迷你柱（正部保留 / 负部打叉置 0）
        MY = BASE_Y + 30
        if r is not None:
            lc.rect(cxx - 9, MY - r * 60, 18, r * 60, '#fde68a', '#d97706', rx=2, sw=1.1)
            lc.text(cxx, MY + 14, 'r=%.1f' % r, 8.5, '#d97706', 'middle', True,
                    tag='r%d%s' % (i, name))
        else:
            lc.seg(cxx - 10, MY - 2, cxx + 10, MY - 22, lc.C_ABORT, 1.4)
            lc.seg(cxx + 10, MY - 2, cxx - 10, MY - 22, lc.C_ABORT, 1.4)
            lc.text(cxx, MY + 14, '置 0', 8.5, lc.C_ABORT, 'middle', tag='z%s' % name)
    lc.text(px + 14, BASE_Y + 66, res_line, 9.2, '#334155', 'start', maxw=pw - 28,
            tag='res')
    lc.text(px + 14, BASE_Y + 84, norm_line, 9.2, '#d97706', 'start', True,
            maxw=pw - 28, tag='norm')

# 图例（顶部右侧）
LY = 78
lc.text(BXR, LY, '■ p_t 柱（实心）　-- p_d 刻度线　■ 残差正部（琥珀）', 8.5,
        lc.C_MUTE, 'end', tag='legend')

# ---------------- 底部对账条 ----------------
BY = 92 + 332 + 16
lc.rect(MX, BY, BXR - MX, 56, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 22, '拒绝总质量（被拒概率）＝残差总质量＝½‖p_t−p_d‖₁：例 1 得 0.2、例 2 得 0.6', 10,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='bal:t')
lc.text(MX + 14, BY + 42, '「被拒的概率」与「从哪补回来」是同一个数——这正是下一张图无损证明对账的两半', 9,
        lc.C_MUTE, 'start', maxw=BXR - MX - 28, tag='bal:l')

# ---------------- 结论 + 页脚 ----------------
CY = BY + 56 + 22
cc.conclusion(MX, BXR, CY,
              '把 target 与 draft 逐坐标相减、只留正部：这就是拒绝后「从哪捞回一个 token」的全部来源。')
cc.footer(MX, BXR, CY + 22, [
    '数值=固定 seed 的 CPU 参考实现实跑（Gumbel-max 免归一化采样同一份残差）· 残差定义 r(x)=max(0, p_t(x)−p_d(x)) 引 arXiv:2607.05147 §1',
    'vLLM kernel 同构：prob = tl.maximum(target_prob − draft_prob, 0.0)，见 vllm/v1/sample/rejection_sampler.py · 行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-residual-build', W, H)
