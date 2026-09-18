#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M6·步4 机制图 ch33-fig-retrospective-bias（模板 state-machine）

claim：回顾式偏差的状态机：x_1=A（c_2=0.9）→Θ_2=1.134 全局最大→ℓ=2→x_1 必被
接受输出 A；x_1=B（c_2=0.0）→Θ_2=0.81<Θ_0→ℓ=0→从 p_t 重采——名单偏好改写
输出：P(Y=A)=0.85（经验 0.85039）≠0.7；早停版恒 ℓ=0，输出 0.699855≈p_t。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑）。
"""
import _common33 as cc
import l0_common as lc

W = 1160
lc.reset()
MX, BXR = cc.header(
    W,
    '回顾式调度的选择偏差：让候选 token 自己决定自己进不进考场',
    'Markov 置信头的 c₂ 要知道 x₁ 实际是什么才算得出来——回顾式全局搜索先看 x₁ 再定名单，等于把输出分布悄悄改写',
    '放大自 L0 采样出口列 · 准入边界信息流闸门（品红）')

# ---------------- 入口状态 ----------------
EX, EY, EW, EH = MX + 10, 96, 170, 44
lc.rect(EX, EY, EW, EH, lc.C_SAM_F, lc.C_SAM_S, rx=22, sw=1.8)
lc.text(EX + EW / 2, EY + 27, 'x₁ ~ p_d=(0.5, 0.5)', 9.5, lc.C_SAM_S, 'middle',
        True, maxw=EW - 12, tag='en')

def chip(x, y, w, h, t, col, fill='#ffffff', bold=True, fs=9):
    lc.rect(x, y, w, h, fill, col, rx=6, sw=1.5)
    lc.text(x + w / 2, y + h / 2 + 3.5, t, fs, col, 'middle', bold, maxw=w - 8,
            tag='c:' + t[:6])
    return x, y, w, h

# ---------------- A 臂（实线粗）----------------
AY = 106
chip(258, AY, 92, 40, 'c₂=0.9', lc.C_GPU_S)
chip(372, AY, 148, 40, 'Θ₂=1.134 全局最大', lc.C_GPU_S)
chip(542, AY, 118, 40, 'ℓ=2（收下）', lc.C_GPU_S)
lc.seg(EX + EW, EY + EH / 2, 258, AY + 20, lc.C_GPU_S, 2.6, 'std')
for x0, x1 in ((350, 372), (520, 542)):
    lc.seg(x0, AY + 20, x1, AY + 20, lc.C_GPU_S, 2.2, 'std')
# 输出 A
chip(690, AY, 96, 40, '输出 A', lc.C_GPU_S, lc.C_GPU_F, fs=10.5)
lc.seg(660, AY + 20, 690, AY + 20, lc.C_GPU_S, 2.2, 'std')
lc.text(690 + 48, AY - 10, 'x₁ 必被接受（它自己把名单抬上去了）', 8, lc.C_GPU_S,
        'middle', maxw=170, tag='a:note')

# ---------------- B 臂（红）----------------
BY_ = 216
lc.parrow([(EX + EW / 2, EY + EH), (EX + EW / 2, BY_ + 20), (258, BY_ + 20)],
          lc.C_ABORT, 2.0, 'std')
chip(258, BY_, 92, 40, 'c₂=0.0', lc.C_ABORT)
chip(372, BY_, 148, 40, 'Θ₂=0.81<Θ₀', lc.C_ABORT)
chip(542, BY_, 118, 40, 'ℓ=0（踢掉）', lc.C_ABORT)
for x0, x1 in ((350, 372), (520, 542)):
    lc.seg(x0, BY_ + 20, x1, BY_ + 20, lc.C_ABORT, 1.8, 'std')
# 从 p_t 重采
chip(690, BY_, 96, 40, '从 p_t 重采', '#2563eb', '#eff6ff', fs=9.5)
lc.seg(660, BY_ + 20, 690, BY_ + 20, lc.C_ABORT, 1.8, 'std')
lc.text(738, BY_ + 58, '0.7 → A　　0.3 → B', 8.5, '#2563eb', 'middle', True,
        maxw=170, tag='b:res')

# Θ 数组注
lc.text(EX, 296, 'Θ 数组：x₁=A → [1.0, 0.9, 1.134]；x₁=B → [1.0, 0.9, 0.81]；a₁=0.8（位 1 无条件存活）', 8.5,
        lc.C_MUTE, 'start', maxw=640, tag='th')

# ---------------- 右侧：输出分布对照 ----------------
PX = 830
PW = BXR - PX
lc.rect(PX, 88, PW, 248, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(PX + PW / 2, 110, '出口分布被改写', 10.5, lc.C_TXT, 'middle', True,
        maxw=PW - 20, tag='o:t')
lc.text(PX + PW / 2, 128, '■ 偏差版出口　□ target p_t', 8, lc.C_MUTE, 'middle',
        maxw=PW - 20, tag='o:leg')
BASE, SC = 252, 110.0
PAIRS = [('A', 0.85, 0.7), ('B', 0.15, 0.3)]
for i, (name, bias_v, tgt_v) in enumerate(PAIRS):
    cxx = PX + 74 + i * 118
    lc.rect(cxx - 24, BASE - bias_v * SC, 22, bias_v * SC, lc.C_SAM_F, lc.C_SAM_S,
            rx=3, sw=1.5)
    lc.text(cxx - 13, BASE - bias_v * SC - 7, '%g' % bias_v, 8, lc.C_SAM_S,
            'middle', True, tag='ov%d' % i)
    lc.rect(cxx + 2, BASE - tgt_v * SC, 22, tgt_v * SC, '#f0fdf4', lc.C_GPU_S,
            rx=3, sw=1.3)
    lc.text(cxx + 13, BASE - tgt_v * SC - 7, '%g' % tgt_v, 8, lc.C_GPU_S,
            'middle', True, tag='tv%d' % i)
    lc.text(cxx - 1, BASE + 14, name, 8.5, lc.C_MUTE, 'middle', tag='on%d' % i)
lc.text(PX + 14, 292, '偏差版：P(Y=A)=0.5×1+0.5×0.7=0.85', 8.2, lc.C_SAM_S,
        'start', True, maxw=PW - 28, tag='o:l2')
lc.text(PX + 14, 308, '（经验 0.85039）；target=(0.7, 0.3)', 8, '#334155',
        'start', maxw=PW - 28, tag='o:l3')
lc.text(PX + 14, 326, '早停版：0.699855 ≈ p_t（名单与实现隔离）', 8.2,
        lc.C_GPU_S, 'start', True, maxw=PW - 28, tag='o:es2')

# ---------------- 底部：对账条 ----------------
BBY = 348
lc.rect(MX, BBY, BXR - MX, 56, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BBY + 22, '两条路合起来：P(Y=A)=0.5×1 + 0.5×0.7 = 0.85 ≠ 0.7——「多看一眼再决定」破无损；经验模拟 200000 次得 0.85039 逐位应验', 10,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='bb:t')
lc.text(MX + 14, BBY + 42, '生产版因此用异步因果屏障：调度名单只准看两步前的置信（时间偏移），不看本步实采——「看的是过去」', 9,
        lc.C_MUTE, 'start', maxw=BXR - MX - 28, tag='bb:l')

# ---------------- 结论 + 页脚 ----------------
CY = BBY + 56 + 22
cc.conclusion(MX, BXR, CY,
              '非前瞻性（non-anticipating）不是学术洁癖：名单只要闻到本步实采的味道，0.5/0.5 的入口就会被掰成 0.85/0.15。')
cc.footer(MX, BXR, CY + 22, [
    '数值=固定 seed 的 CPU 参考实现实跑（MC 200000 次、seed=13：偏差版 0.85039 / 早停版 0.699855）· 非前瞻性与异步因果屏障引 arXiv:2607.05147 §5.2/§5.3',
    '行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-retrospective-bias', W, H)
