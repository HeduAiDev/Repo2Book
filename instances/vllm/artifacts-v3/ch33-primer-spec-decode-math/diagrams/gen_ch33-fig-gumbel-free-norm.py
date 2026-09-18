#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M1·步3 机制图 ch33-fig-gumbel-free-norm（模板 tensor-flow）

claim：残差采样免归一化：score=r(x)/E_x（E~Exp(1) 独立）取 argmax ≡ 从 norm(r)
抽样——两组 Exp 抽样分别选中 token 1（scores 0.176471/0.333333/0）与 token 0
（0.6/0.136364/0），分母 Z 全程不用算；MC 10000 次频率 (0.4939, 0.5061, 0.0)
对拍 norm(r)=(0.5, 0.5, 0.0)。
底部虚线框回指 V1 kernel 两行（L930/L944），并标 Z = Σr = 0.3+0.3+0.0 = 0.6
（=residual-build 图例 2 的总质量 0.6）——有值、从未计算。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑）。
"""
import _common33 as cc
import l0_common as lc

W = 1120
lc.reset()
MX, BXR = cc.header(
    W,
    '残差采样免归一化：score = r(x)/E_x 取 argmax ≡ 从 norm(r) 抽样',
    '每个候选独立掷一个指数噪声 E~Exp(1)——归一化因子 Z 在 argmax 下是常数、根本不用算（ch30 已立的 Gumbel-max 定理用在残差分布上）',
    '放大自 L0 采样出口列 · 恢复 token 采样算子（品红）')

# ---------------- 左：两次抽样表 ----------------
TX, TY = MX, 94
LBL_W, CW, RH = 118, 118, 46
TOKENS = ['token 0', 'token 1', 'token 2']
for c, t in enumerate(TOKENS):
    x = TX + LBL_W + c * CW
    lc.rect(x, TY, CW, 36, '#f8fafc', lc.C_MUTE, rx=0, sw=1.2)
    lc.text(x + CW / 2, TY + 23, t, 10, lc.C_MUTE, 'middle', True, maxw=CW - 8,
            tag='th%d' % c)
ROWS = [
    ('r = max(0, p_t−p_d)', ['0.3', '0.3', '0.0'], None, TY + 36),
    ('E（掷骰 ①）', ['1.7', '0.9', '0.3'], None, TY + 36 + 46),
    ('score = r/E ①', ['0.176471', '0.333333', '0.0'], 1, TY + 36 + 2 * 46),
    ('E（掷骰 ②）', ['0.5', '2.2', '1.1'], None, TY + 36 + 3 * 46),
    ('score = r/E ②', ['0.6', '0.136364', '0.0'], 0, TY + 36 + 4 * 46),
]
for ri, (name, vals, win, y) in enumerate(ROWS):
    lc.text(TX + LBL_W - 8, y + 28, name, 9, lc.C_MUTE, 'end', maxw=LBL_W - 10,
            tag='rl%d' % ri)
    for c, v in enumerate(vals):
        x = TX + LBL_W + c * CW
        hi = (win == c)
        lc.rect(x, y, CW, RH, ('#fdf2f8' if hi else '#ffffff'),
                (lc.C_SAM_S if hi else lc.C_MUTE), rx=0, sw=(2.0 if hi else 1.0))
        lc.text(x + CW / 2, y + 29, v, 11, (lc.C_SAM_S if hi else lc.C_TXT), 'middle',
                hi, maxw=CW - 8, tag='c%d%d' % (ri, c))
        if hi:
            lc.text(x + CW / 2, y + 14, '↓ argmax', 8.5, lc.C_SAM_S, 'middle', True,
                    tag='win%d' % c)
lc.text(TX, TY + 36 + 5 * 46 + 24,
        '抽样 ① E 小者赢 → token 1；抽样 ② → token 0：各次各选各的，频率趋于 norm(r)',
        9.5, lc.C_TXT, 'start', True, maxw=640, tag='twolines')
lc.text(TX, TY + 36 + 5 * 46 + 42, 'r=(0.3, 0.3, 0.0)（批改一节三词表组裁出的正部，即上图例 2；norm(r)=(0.5, 0.5, 0.0)）', 9,
        lc.C_MUTE, 'start', maxw=640, tag='rnote')

# ---------------- 右：MC 对拍柱图 ----------------
PX = TX + LBL_W + 3 * CW + 26
PW = BXR - PX
lc.rect(PX, TY, PW, 268, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(PX + 14, TY + 22, '蒙特卡洛 10000 次对拍', 10.5, lc.C_TXT, 'start', True,
        maxw=PW - 28, tag='mc:t')
BASE = TY + 216
MAXH = 132
for i, (freq, th) in enumerate([(0.4939, 0.5), (0.5061, 0.5), (0.0, 0.0)]):
    x = PX + 34 + i * 84
    lc.rect(x, BASE - freq * MAXH, 34, freq * MAXH, lc.C_SAM_F, lc.C_SAM_S, rx=3, sw=1.4)
    lc.text(x + 17, BASE - freq * MAXH - 8, '%.4f' % freq, 8.5, lc.C_SAM_S, 'middle',
            True, tag='mc%d' % i)
    if th > 0:
        lc.seg(x - 8, BASE - th * MAXH, x + 42, BASE - th * MAXH, '#2563eb', 1.6,
               dash=True)
    lc.text(x + 17, BASE + 16, 'token %d' % i, 9, lc.C_TXT, 'middle', tag='mct%d' % i)
lc.text(PX + PW - 14, BASE + 34, '虚线=理论 norm(r)=(0.5, 0.5, 0.0)', 8.5, '#2563eb',
        'end', tag='mcl')
lc.text(PX + 14, TY + 44, '采样频率 (0.4939, 0.5061, 0.0) 贴住理论值', 9, '#334155',
        'start', maxw=PW - 28, tag='mc:s')

# ---------------- 底部：vLLM kernel 同构（虚线框，回指正文两行；Z 有值但从未计算） ----------------
BY = TY + 36 + 5 * 46 + 58
KB_H = 84
lc.rect(MX, BY, BXR - MX, KB_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3, dash=True)
lc.text(MX + 14, BY + 20, 'vLLM V1 kernel 同构：恢复 token 的采样就是这两行', 10,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='k:t')
lc.text(MX + 14, BY + 40, 'prob = max(p_t−p_d, 0)　→　score = prob * inv_q　→　取 argmax', 10.5,
        lc.C_SAM_S, 'start', True, maxw=BXR - MX - 28, tag='k:f')
lc.text(MX + 14, BY + 58, '注释明说「不需要 prob/Σprob 归一化，因为 argmax 会选出最大值」——本例分母 Z = Σr = 0.3+0.3+0.0 = 0.6，自始至终没有算过', 9,
        '#334155', 'start', maxw=BXR - MX - 28, tag='k:n')
lc.text(MX + 14, BY + 73, 'vllm/v1/sample/rejection_sampler.py:L930 与 L944（免归一化注释 L931-L932）', 9,
        lc.C_MUTE, 'start', maxw=BXR - MX - 28, tag='k:l')

# ---------------- 结论 + 页脚 ----------------
CY = BY + KB_H + 22
cc.conclusion(MX, BXR, CY,
              '免归一化不是近似而是恒等：argmax 里除不除 Z 不改名次——省掉的是一整步词表维归约。')
cc.footer(MX, BXR, CY + 22, [
    '数值=固定 seed 的 CPU 参考实现实跑（两组 Exp 抽样与 MC 10000 次）· Gumbel-max 定理：argmax_x r(x)/E_x ≡ 从 norm(r) 抽样（E~Exp(1) 独立）',
    '行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-gumbel-free-norm', W, H)
