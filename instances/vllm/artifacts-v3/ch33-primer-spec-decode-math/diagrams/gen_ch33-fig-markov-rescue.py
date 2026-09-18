#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M2·步5 机制图 ch33-fig-markov-rescue（模板 before-after）

claim：Markov 偏置的修复：位 1 采出 'of' 后 B(of,·)=[0,0,0,3.0,−3.0] 加到边缘
logits 上——course 从 0.368 抬到 0.986667、problem 压到 0.003722；'no' 分支镜像
（problem 0.992034）；连贯率 0.4448→0.988814。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑）。
"""
import _common33 as cc
import l0_common as lc

W = 1160
lc.reset()
MX, BXR = cc.header(
    W,
    "Markov 头的修复：位 1 实采 'of' 之后，查一行偏置加到边缘 logits 上再 softmax",
    "B(of,·)=[0, 0, 0, +3.0, −3.0]——course 位 +3、problem 位 −3：两个分支都回到各自 mode 的尖峰上",
    '放大自 L0 GPU 执行臂 · 并行块之后的序列头小拍（绿↔品红交界）')

SC = 150.0
BASE = 270

# ---------------- 左：修正前 ----------------
LW = 330
lc.rect(MX, 88, LW, 264, '#ffffff', lc.C_ABORT, rx=8, sw=1.2, dash=True)
lc.text(MX + 14, 108, '修正前 · 边缘混合（并行之病）', 10, lc.C_ABORT, 'start', True,
        maxw=LW - 28, tag='b:t')
for name, v, cxx in [('course', 0.368, MX + 90), ('problem', 0.56, MX + 220)]:
    lc.rect(cxx - 24, BASE - v * SC, 48, v * SC, lc.C_SAM_F, lc.C_SAM_S, rx=3, sw=1.4)
    lc.text(cxx, BASE - v * SC - 8, '%g' % v, 9, lc.C_SAM_S, 'middle', True,
            tag='b:' + name)
    lc.text(cxx, BASE + 14, name, 8.2, lc.C_MUTE, 'middle', tag='bn:' + name)
lc.text(MX + 220, BASE - 0.56 * SC - 24, 'argmax', 8, lc.C_ABORT, 'middle', True,
        tag='b:am')
lc.rect(MX + 14, BASE + 30, LW - 28, 38, '#fef2f2', lc.C_ABORT, rx=6, sw=1.4)
lc.text(MX + LW / 2, BASE + 53, "'of problem'（跨 mode）", 10, lc.C_ABORT, 'middle',
        True, maxw=LW - 40, tag='b:r')

# ---------------- 中：偏置注入 ----------------
MID_X = MX + LW + 22
MID_W = 292
lc.rect(MID_X, 88, MID_W, 264, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MID_X + 14, 108, '偏置注入（知道前一个词）', 10, lc.C_TXT, 'start', True,
        maxw=MID_W - 28, tag='m:t')
lc.rect(MID_X + 58, 122, 176, 30, lc.C_GPU_F, lc.C_GPU_S, rx=15, sw=1.5)
lc.text(MID_X + 146, 141, "位 1 实采 'of'", 9.5, lc.C_GPU_S, 'middle', True,
        maxw=160, tag='m:prev')
lc.seg(MID_X + 146, 152, MID_X + 146, 175, lc.C_GPU_S, 1.6, 'std')
BVALS = ['0', '0', '0', '+3.0', '−3.0']
bw = 44
bx0 = MID_X + (MID_W - 5 * bw - 4 * 6) / 2
for i, v in enumerate(BVALS):
    hi = i >= 3
    lc.rect(bx0 + i * (bw + 6), 176, bw, 30, ('#fde68a' if hi else '#f8fafc'),
            ('#d97706' if hi else lc.C_MUTE), rx=4, sw=(1.6 if hi else 1.0))
    lc.text(bx0 + i * (bw + 6) + bw / 2, 195, v, 9, ('#d97706' if hi else lc.C_MUTE),
            'middle', True, maxw=bw - 4, tag='m:b%d' % i)
lc.text(MID_X + MID_W / 2, 224, 'B(of,·)：course 位 +3.0、problem 位 −3.0', 8.5,
        '#d97706', 'middle', True, maxw=MID_W - 24, tag='m:bl')
lc.seg(MID_X + MID_W / 2, 206, MID_X + MID_W / 2, 252, '#d97706', 1.6, 'std')
lc.rect(MID_X + 22, 254, MID_W - 44, 30, '#fef3c7', '#d97706', rx=6, sw=1.3)
lc.text(MID_X + MID_W / 2, 273, 'softmax(U + B)：加到该位 base logits 上', 9,
        lc.C_TXT, 'middle', True, maxw=MID_W - 48, tag='m:sl')
lc.text(MID_X + MID_W / 2, 302, '（U 是骨干给的、不动）', 8, lc.C_MUTE, 'middle',
        maxw=MID_W - 24, tag='m:sn')

# ---------------- 右：修正后 ----------------
RX = MID_X + MID_W + 22
RW = BXR - RX
lc.rect(RX, 88, RW, 264, '#ffffff', lc.C_GPU_S, rx=8, sw=1.2, dash=True)
lc.text(RX + 14, 108, "修正后 · softmax(边缘 logits + B(of,·))", 10, lc.C_GPU_S,
        'start', True, maxw=RW - 28, tag='a:t')
AFTER = [('of', 0.004005), ('no', 0.001068), ('course', 0.986667),
         ('problem', 0.003722)]
slot = (RW - 40) / 4
for i, (name, v) in enumerate(AFTER):
    cxx = RX + 20 + slot * (i + 0.5)
    hi = (name == 'course')
    lc.rect(cxx - 24, BASE - v * SC, 48, v * SC, ('#f0fdf4' if hi else lc.C_SAM_F),
            (lc.C_GPU_S if hi else lc.C_SAM_S), rx=3, sw=(1.8 if hi else 1.0))
    lc.text(cxx, BASE - v * SC - 8, '%g' % v, 8, (lc.C_GPU_S if hi else lc.C_SAM_S),
            'middle', True, maxw=76, tag='a:' + name)
    lc.text(cxx, BASE + 14, name, 8.2, lc.C_MUTE, 'middle', tag='an:' + name)
lc.text(RX + 20 + slot * 2.5, 200, 'argmax', 8.5, lc.C_GPU_S, 'middle', True,
        tag='a:am')
lc.rect(RX + 14, BASE + 30, RW - 28, 38, '#f0fdf4', lc.C_GPU_S, rx=6, sw=1.4)
lc.text(RX + RW / 2, BASE + 53, "'of course'（回到 mode 尖峰）", 10, lc.C_GPU_S,
        'middle', True, maxw=RW - 40, tag='a:r')
lc.text(RX + 14, 348, "'no' 分支镜像：course 0.001616 / problem 0.992034", 8.2,
        lc.C_MUTE, 'start', maxw=RW - 28, tag='a:mi')

# ---------------- 底部：连贯率账 ----------------
BY = 88 + 264 + 16
lc.rect(MX, BY, BXR - MX, 56, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 22, '连贯率 0.4448 → 0.988814（=0.6×0.986667 + 0.4×0.992034）——「知道前一个词是什么」换来的全部收益', 10,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='b:t')
lc.text(MX + 14, BY + 42, '代价只是每步一次 r 维查表 + 一次 GEMV（下一节算这笔账）', 9,
        lc.C_MUTE, 'start', maxw=BXR - MX - 28, tag='b:l')

# ---------------- 结论 + 页脚 ----------------
CY = BY + 56 + 22
cc.conclusion(MX, BXR, CY,
              '0.368 → 0.986667 的前后对比就是半自回归的全部主张：并行骨干干重活，一行偏置把「谁跟谁连贯」买回来。')
cc.footer(MX, BXR, CY + 22, [
    '数值=固定 seed 的 CPU 参考实现实跑（softmax(U+B) 逐坐标复算）· 半自回归与 Eq.(4) 加法注入引 arXiv:2607.05147 §3.1',
    '行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-markov-rescue', W, H)
