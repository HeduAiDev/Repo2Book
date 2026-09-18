#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M5·步3 机制图 ch33-fig-sts-calibration（模板 before-after）

claim：STS 前后对照：过自信的累积置信 [0.941563, 0.869679, 0.786202]（ECE
0.137063/0.266679/0.359952）经逐位温度 T=[2.0, 2.25, 2.5] 校准后贴住真实存活
[0.800791, 0.60221, 0.427938] vs [0.8, 0.6, 0.42]（ECE 降至 0.010623/0.003766/
0.001795）——不校准则位 3 的存活概率虚高 1.871909 倍、τ 随之虚高约 1.28 倍
（τ̂=3.597444 对真值 2.82；Revise R1 量名修正：1.871909 是存活概率之比）。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑 + 论文 §4.3.3）。
"""
import _common33 as cc
import l0_common as lc

W = 1120
lc.reset()
MX, BXR = cc.header(
    W,
    '为什么校准不可省：调度器要的是 ∏c 的绝对幅度，不是只要排序对',
    '神经置信天然过自信——位 3 报 0.786202、真实存活只有 0.42，存活概率虚高 1.87 倍、τ 随之虚高至 3.597444/2.82≈1.28 倍，Θ=τ·SPS(B) 的调度决策直接失真',
    '放大自 L0 采样出口列上游 · 置信信号通道（品红）')

SC = 205.0
BASE = 352
TRUE_A = [0.8, 0.6, 0.42]

def panel(px, pw, title, sub, mean, ece, note, col):
    lc.rect(px, 90, pw, 340, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
    lc.text(px + 14, 112, title, 10.5, col, 'start', True, maxw=pw - 28, tag='t')
    lc.text(px + 14, 130, sub, 8.5, lc.C_MUTE, 'start', maxw=pw - 28, tag='s')
    slot = (pw - 60) / 3
    for i in range(3):
        cxx = px + 30 + slot * (i + 0.5)
        bw = 56
        # 真实存活虚线
        lc.seg(cxx - bw / 2 - 10, BASE - TRUE_A[i] * SC, cxx + bw / 2 + 10,
               BASE - TRUE_A[i] * SC, '#2563eb', 1.4, dash=True)
        lc.rect(cxx - bw / 2, BASE - mean[i] * SC, bw, mean[i] * SC,
                ('#fef2f2' if col == lc.C_ABORT else lc.C_SAM_F), col, rx=3, sw=1.5)
        lc.text(cxx, BASE - mean[i] * SC - 8, '%g' % mean[i], 8.2, col, 'middle',
                True, maxw=90, tag='m%d' % i)
        lc.text(cxx + bw / 2 + 14, BASE - TRUE_A[i] * SC + 3, '%g' % TRUE_A[i], 7.8,
                '#2563eb', 'start', tag='ta%d' % i)
        lc.text(cxx, BASE + 16, '位 %d' % (i + 1), 9, lc.C_TXT, 'middle', True,
                tag='p%d' % i)
        lc.text(cxx, BASE + 32, 'ECE %g' % ece[i], 7.6, lc.C_MUTE, 'middle',
                maxw=slot - 8, tag='e%d' % i)
    lc.text(px + 14, BASE + 56, note, 8.5, col, 'start', True, maxw=pw - 28,
            tag='note')
    lc.text(px + pw - 12, 112, '虚线=真实存活 a', 8, '#2563eb', 'end', tag='leg')

PW = (BXR - MX - 120) / 2
panel(MX, PW, '校准前 · 神经置信过自信', 'mean ∏c（三根全高过虚线）',
      [0.941563, 0.869679, 0.786202], [0.137063, 0.266679, 0.359952],
      '三根柱全飘在虚线上方——幅度系统性偏高', lc.C_ABORT)
panel(MX + PW + 120, PW, '校准后 · STS 贴住真实存活', 'mean ∏c（贴住虚线）',
      [0.800791, 0.60221, 0.427938], [0.010623, 0.003766, 0.001795],
      '温度缩放保序：谁更可信的排序一位不动', lc.C_SAM_S)

# ---------------- 中间温度计 ----------------
TX = MX + PW + 14
lc.text(TX + 46, 112, 'STS', 9.5, lc.C_MUTE, 'middle', True, maxw=92, tag='th:t')
lc.text(TX + 46, 128, '逐位温度', 8, lc.C_MUTE, 'middle', maxw=92, tag='th:s')
for i, t in enumerate(['T₁=2.0', 'T₂=2.25', 'T₃=2.5']):
    y = 152 + i * 54
    lc.rect(TX + 12, y, 68, 34, '#fef3c7', '#d97706', rx=6, sw=1.3)
    lc.text(TX + 46, y + 21, t, 8.8, '#d97706', 'middle', True, maxw=60,
            tag='t%d' % i)
lc.text(TX + 46, 330, '从左到右逐位', 7.6, lc.C_MUTE, 'middle', maxw=92, tag='tm1')
lc.text(TX + 46, 344, '1D 温度网格搜索', 7.6, lc.C_MUTE, 'middle', maxw=92,
        tag='tm2')
lc.seg(TX + 80, 220, MX + PW + 120, 220, '#d97706', 1.6, 'std')
lc.seg(TX + 12, 220, MX + PW, 220, '#d97706', 1.6, None, dash=True)

# ---------------- 底部：警示 + 论文侧 ----------------
BY = 90 + 340 + 16
lc.rect(MX, BY, BXR - MX, 62, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 22, '不校准 → 位 3 的存活概率虚高 1.871909 倍、τ 随之虚高约 1.28 倍——Θ=τ·SPS(B) 的准入与截断决策全部失真', 10,
        lc.C_ABORT, 'start', True, maxw=BXR - MX - 28, tag='b:t')
lc.text(MX + 14, BY + 44, '论文侧（Fig.6）：raw ECE 3%-8%（ROC-AUC 0.81-0.90）→ STS 后 ~1%；固定已校准的前缀、逐位找让累积乘积 ECE 最小的温度', 9,
        lc.C_MUTE, 'start', maxw=BXR - MX - 28, tag='b:l')

# ---------------- 结论 + 页脚 ----------------
CY = BY + 62 + 22
cc.conclusion(MX, BXR, CY,
              '校准解决的是「幅度」不是「排序」：三根高柱贴回虚线，调度器拿到的才是真 τ——保序意味着排序侧的信息一位不丢。')
cc.footer(MX, BXR, CY + 22, [
    '数值=固定 seed 的 CPU 参考实现实跑（STS 逐位网格搜索；保序逐位断言）· 论文侧口径引 arXiv:2607.05147 §4.3.3（Fig.6）',
    '行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-sts-calibration', W, H)
