#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M2·步4 机制图 ch33-fig-marginal-collision（模板 flow）

claim：多模态碰撞的算术：并行位 2 学到的是边缘混合 course 0.368/problem 0.56
（0.6*0.6+0.4*0.02 与 0.6*0.3+0.4*0.95）——argmax 与位 1 的 'of' 拼出
'of problem'；按连贯率 0.4448 计约 55.5% 的块落不进任何真 mode。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑）。
"""
import _common33 as cc
import l0_common as lc

W = 1160
lc.reset()
MX, BXR = cc.header(
    W,
    '并行之病 · 多模态碰撞：位 2 学到的是「对前驱边缘化」的混合分布',
    "上下文两个都容：'of course' 与 'no problem'——并行位从没见过「前一个词实采了哪个」，只能学两个分支的加权平均",
    '放大自 L0 GPU 执行臂 · 并行 draft 块的位间关系（绿）')

SC = 170.0
BASE = 310
BW = 34

# ---------------- 左：条件分布（两分支并列柱）----------------
lc.text(MX + 4, 96, '条件分布（各自见过）', 10, lc.C_TXT, 'start', True, tag='c:t')
lc.text(MX + 4, 112, '位 2 给定「位 1 实采了什么」', 8.2, lc.C_MUTE, 'start', tag='c:s')
COND = [('course', 0.6, '0.6', 0.02, '0.02', MX + 78),
        ('problem', 0.3, '0.3', 0.95, '0.95', MX + 218)]
for name, p_of, p_of_s, p_no, p_no_s, cxx in COND:
    lc.rect(cxx - BW / 2 - 2, BASE - p_of * SC, BW, p_of * SC, lc.C_GPU_F,
            lc.C_GPU_S, rx=3, sw=1.4)
    lc.rect(cxx + BW / 2 + 2, BASE - p_no * SC, BW, p_no * SC, '#eff6ff', '#2563eb',
            rx=3, sw=1.4)
    lc.text(cxx - BW / 2 - 2 + BW / 2, BASE - p_of * SC - 8, p_of_s, 8.2,
            lc.C_GPU_S, 'middle', True, tag='c1:' + name)
    lc.text(cxx + BW / 2 + 2 + BW / 2, BASE - p_no * SC - 8, p_no_s, 8.2,
            '#2563eb', 'middle', True, tag='c2:' + name)
    lc.text(cxx, BASE + 16, name, 8.5, lc.C_TXT, 'middle', True, tag='cn:' + name)
lc.text(MX + 4, 348, 'of 分支（绿）· no 分支（蓝）', 8, lc.C_MUTE, 'start',
        tag='cleg')

# ---------------- 中：边缘混合 ----------------
MX2 = MX + 372
lc.text(MX2, 96, '边缘混合（并行位 2 实际学的）', 10, lc.C_TXT, 'start', True, tag='m:t')
lc.seg(MX + 300, 210, MX2 + 44, 244, lc.C_MUTE, 1.3, dash=True)
lc.seg(MX + 300, 260, MX2 + 166, 210, lc.C_MUTE, 1.3, dash=True)
lc.text(392, 205, '权重 0.6', 8, lc.C_GPU_S, 'middle', tag='w1')
lc.text(376, 272, '权重 0.4', 8, '#2563eb', 'middle', tag='w2')
for i, (name, v, vs, formula) in enumerate([
        ('course', 0.368, '0.368', '0.6×0.6 + 0.4×0.02'),
        ('problem', 0.56, '0.56', '0.6×0.3 + 0.4×0.95')]):
    cxx = MX2 + 44 + i * 122
    lc.rect(cxx - 26, BASE - v * SC, 52, v * SC, lc.C_SAM_F, lc.C_SAM_S, rx=3, sw=1.6)
    lc.text(cxx, BASE - v * SC - 8, vs, 9.5, lc.C_SAM_S, 'middle', True,
            tag='mv:' + name)
    lc.text(cxx, BASE + 16, name, 8.5, lc.C_TXT, 'middle', True, tag='mn:' + name)
    lc.text(cxx, BASE + 30, formula, 7.6, lc.C_FAINT, 'middle', maxw=126,
            tag='mf:' + name)

# ---------------- 右：碰撞结果 ----------------
RX = MX2 + 268
lc.text(RX, 96, '独立采样拼出谁都没想说的', 10, lc.C_TXT, 'start', True, maxw=BXR - RX,
        tag='r:t')
lc.rect(RX, 112, BXR - RX, 76, '#fef2f2', lc.C_ABORT, rx=8, sw=1.6)
lc.text(RX + 12, 132, "位 1 独立采 'of'（0.6）", 8.8, '#334155', 'start',
        maxw=BXR - RX - 24, tag='r:l1')
lc.text(RX + 12, 150, "位 2 独立采 'problem'（0.56=argmax）", 8.8, '#334155',
        'start', maxw=BXR - RX - 24, tag='r:l2')
lc.text(RX + 12, 174, "→ 'of problem'（跨 mode 废话）", 10.5, lc.C_ABORT, 'start',
        True, maxw=BXR - RX - 24, tag='r:l3')
lc.rect(RX, 202, BXR - RX, 60, '#f0fdf4', lc.C_GPU_S, rx=8, sw=1.2)
lc.text(RX + 12, 222, '真 mode 只有两个：', 8.8, lc.C_MUTE, 'start',
        maxw=BXR - RX - 24, tag='r:l4')
lc.text(RX + 12, 244, "'of course'　或　'no problem'", 10, lc.C_GPU_S, 'start', True,
        maxw=BXR - RX - 24, tag='r:l5')
lc.text(RX, 276, "平均把 'no' 分支的 problem 尖峰泄了进来", 8, lc.C_MUTE, 'start',
        maxw=BXR - RX, tag='r:l6')

# ---------------- 底部：连贯率账 ----------------
BY = BASE + 52
lc.rect(MX, BY, BXR - MX, 56, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 22, '按联合分布独立采样：连贯率 0.4448 —— 约 55.5% 的块落不进任何真 mode（draft 白算、验证白跑）', 10,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='b:t')
lc.text(MX + 14, BY + 42, '修复它就是下一张图的主角：Markov 头按「实采前驱」查一行偏置', 9,
        lc.C_MUTE, 'start', maxw=BXR - MX - 28, tag='b:l')

# ---------------- 结论 + 页脚 ----------------
CY = BY + 56 + 22
cc.conclusion(MX, BXR, CY,
              "「对前驱边缘化」的算术就在这两笔加权平均里：0.6×0.6+0.4×0.02 与 0.6×0.3+0.4×0.95——平均把两个 mode 的尖峰都磨平了。")
cc.footer(MX, BXR, CY + 22, [
    '数值=固定 seed 的 CPU 参考实现实跑（双 mode 上下文玩具）· 边缘化与后缀衰减引 arXiv:2607.05147 §2.2/§3.1',
    '行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-marginal-collision', W, H)
