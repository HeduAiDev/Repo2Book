#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M5·步2 机制图 ch33-fig-tv-identity-closure（模板 before-after）

claim：同一个 0.8 出现在三处：验证准则的解析期望 Σmin=0.8、Eq.(8) 标签
1−½‖·‖₁=0.8、经验接受率 0.79899（159798/200000）——训练监督与验证准则是
同一条 TV 恒等式的两个名字。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑）。
"""
import _common33 as cc
import l0_common as lc

W = 1120
lc.reset()
MX, BXR = cc.header(
    W,
    '同一个 0.8 出现在三处：验证准则、Σmin、TV 标签——同一条恒等式',
    '对 (0.7, 0.3)（target）vs (0.5, 0.5)（draft）这组分布：三条路各自算一遍，全部落在 0.8',
    '放大自 L0 采样出口列↔GPU 执行臂 · TV 恒等式回廊')

# ---------------- 顶部：分布对 ----------------
PY = 92
lc.rect(MX, PY, BXR - MX, 56, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, PY + 22, '同一组分布：p_t=(0.7, 0.3)　vs　p_d=(0.5, 0.5)（token A / B）', 10,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='d:t')
lc.text(MX + 14, PY + 42, '位级判据（§33.2 手算同一组数字）：接受概率 A=1.0 / B=0.6，残差质量 0.2', 9,
        lc.C_MUTE, 'start', maxw=BXR - MX - 28, tag='d:l')

# ---------------- 三路汇流 ----------------
PATHS = [
    (MX, '左路 · 接受准则的解析期望', '0.5×1.0 + 0.5×0.6', '（§33.2 逐位接受概率加权）', lc.C_SAM_S),
    (MX + (BXR - MX) / 3 + 6, '中路 · Σmin 形态', '0.5 + 0.3', '（逐坐标 min 求和）', lc.C_GPU_S),
    (MX + 2 * (BXR - MX) / 3 + 12, '右路 · TV 标签（Eq.8）', '1 − ½×(0.2+0.2)', '（1 − ½‖p_t−p_d‖₁）', '#d97706'),
]
PW = (BXR - MX) / 3 - 12
for px, t, f, s, col in PATHS:
    lc.rect(px, 170, PW, 96, '#ffffff', col, rx=8, sw=1.4)
    lc.text(px + PW / 2, 192, t, 9.5, col, 'middle', True, maxw=PW - 12, tag='p:t')
    lc.text(px + PW / 2, 222, f, 13, lc.C_TXT, 'middle', True, maxw=PW - 12, tag='p:f')
    lc.text(px + PW / 2, 244, s, 8, lc.C_MUTE, 'middle', maxw=PW - 12, tag='p:s')
# 汇流箭头 → 中央大徽章
BCX, BCY = (MX + BXR) / 2, 336
for px, _, _, _, col in PATHS:
    lc.parrow([(px + PW / 2, 266), (px + PW / 2, 292), (BCX, 304)], col, 1.8, 'std')
lc.rect(BCX - 60, BCY - 30, 120, 60, '#ffffff', lc.C_SAM_S, rx=30, sw=2.2)
lc.text(BCX, BCY + 7, '0.8', 17, lc.C_SAM_S, 'middle', True, tag='badge')
lc.text(BCX + 92, BCY - 4, '三路全部落在 0.8', 9.5, lc.C_TXT, 'start', True,
        maxw=180, tag='b:l1')
lc.text(BCX + 92, BCY + 14, '经验接受率 0.79899', 9.5, lc.C_SAM_S, 'start', True,
        maxw=180, tag='b:l2')
lc.text(BCX + 92, BCY + 32, '（159798 / 200000 次模拟）', 8, lc.C_MUTE, 'start',
        maxw=180, tag='b:l3')

# ---------------- 底部：第二组对账 ----------------
BY = 388
lc.rect(MX, BY, BXR - MX, 62, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 22, '换三词表组同样成立：c* = 1 − ½×(0.3+0.3+0.6) = 0.4——恒等式与词表大小无关', 10,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='b2:t')
lc.text(MX + 14, BY + 44, '换句话说：置信度头学的就是「验证会有多松」的解析值——训练它不需要跑昂贵的验证', 9,
        lc.C_MUTE, 'start', maxw=BXR - MX - 28, tag='b2:l')

# ---------------- 结论 + 页脚 ----------------
CY = BY + 62 + 22
cc.conclusion(MX, BXR, CY,
              '全章数学的两条主线在这里会师：§33.2 的接受准则与 §33.9 的置信标签，是同一条 TV 恒等式的两个名字。')
cc.footer(MX, BXR, CY + 22, [
    '三路与经验值=固定 seed 的 CPU 参考实现实跑（MC 200000 次）· TV 恒等式 Σmin(p,q)=1−TV(p,q) 引 arXiv:2607.05147 §3.2.1',
    '行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-tv-identity-closure', W, H)
