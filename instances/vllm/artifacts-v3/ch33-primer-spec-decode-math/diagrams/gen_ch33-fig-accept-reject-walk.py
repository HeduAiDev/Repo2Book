#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M1·步1 机制图 ch33-fig-accept-reject-walk（模板 state-table）

claim：三位一表的验证行走：位 0/1 比值 1.4/1.3>1 必收、位 2 比值 0.8 是唯一掷骰位
（u=0.814226≥0.8 拒绝）——首拒即停，位 2 从残差恢复 B，后面的 draft 位直接不看。
数字全部取 spec.numbers（固定 seed=2 的 CPU 参考实现实跑）。
"""
import _common33 as cc
import l0_common as lc

W = 1120
lc.reset()
MX, BXR = cc.header(
    W,
    '验证就是一张从左到右的行走表：比值≥1 必收，比值<1 才真的在抽签',
    '判据 u < p_t/p_d：比值≥1 的位 u 多大都过；首拒即停——位 2 拒绝后从残差恢复 B，其后 draft 位无论好坏全部丢弃',
    '放大自 L0 采样出口列 · RejectionSampler 逐位主循环（品红）')

# ---------------- 主表 ----------------
TX, TY = MX, 92
COLS = [('位', 52), ('draft 候选', 92), ('p_t(x)', 84), ('p_d(x)', 84),
        ('比值 p_t/p_d', 118), ('抽签 u', 104), ('判定', 178)]
HDR_H, ROW_H = 32, 46
xs = []
x = TX
for name, w in COLS:
    xs.append((x, w))
    x += w
TW = x - TX
# 表头
lc.rect(TX, TY, TW, HDR_H, '#f8fafc', lc.C_MUTE, rx=0, sw=1.2)
for (cx_, cw), (name, _) in zip(xs, COLS):
    lc.text(cx_ + cw / 2, TY + 21, name, 9.5, lc.C_MUTE, 'middle', True,
            maxw=cw - 6, tag='th:' + name)
ROWS = [
    ('位 0', 'x_1 = A', '0.7', '0.5', '1.4', '0.261612', '收（u 多大都过）', False),
    ('位 1', 'x_2 = A', '0.65', '0.5', '1.3', '0.298491', '收（u 多大都过）', False),
    ('位 2', 'x_3 = A', '0.4', '0.5', '0.8', '0.814226', '拒（u≥0.8 出界）', True),
]
for r, (pos, cand, pt, pd, ratio, u, verdict, hi) in enumerate(ROWS):
    y = TY + HDR_H + r * ROW_H
    lc.rect(TX, y, TW, ROW_H, ('#fdf2f8' if hi else '#ffffff'),
            (lc.C_SAM_S if hi else lc.C_MUTE), rx=0, sw=(1.8 if hi else 1.0))
    for c, (cx_, cw) in enumerate(xs):
        v = (pos, cand, pt, pd, ratio, u, verdict)[c]
        col = lc.C_TXT
        bold = False
        if c == 4:
            col, bold = (lc.C_SAM_S if float(ratio) < 1 else lc.C_GPU_S), True
        if c == 6:
            col = lc.C_GPU_S if v.startswith('收') else '#dc2626'
            bold = True
        lc.text(cx_ + cw / 2, y + 28, v, 10.5, col, 'middle', bold, maxw=cw - 6,
                tag='r%d%d' % (r, c))

# ---------------- 右侧：首拒之后 ----------------
PX = TX + TW + 22
PW = BXR - PX
PY, PH = TY, 176
lc.rect(PX, PY, PW, PH, '#ffffff', lc.C_SAM_S, rx=8, sw=1.6)
lc.text(PX + 14, PY + 22, '首拒之后（位 2）', 10.5, lc.C_SAM_S, 'start', True,
        maxw=PW - 28, tag='side:t')
lc.text(PX + 14, PY + 42, '位 2 全词表 p_t=(0.4, 0.6)、p_d=(0.5, 0.5)', 9.2,
        '#334155', 'start', maxw=PW - 28, tag='side:l1')
lc.text(PX + 14, PY + 60, '残差 r = max(0, p_t−p_d) = (0.0, 0.1)', 9.2, '#334155',
        'start', maxw=PW - 28, tag='side:l2')
lc.text(PX + 14, PY + 78, 'norm(r) = (0.0, 1.0) → 恢复 token B', 9.2, lc.C_SAM_S,
        'start', True, maxw=PW - 28, tag='side:l3')
lc.text(PX + 14, PY + 102, '本周期发射：[A, A, B] 共 3 个 token', 10, lc.C_TXT,
        'start', True, maxw=PW - 28, tag='side:l4')
lc.text(PX + 14, PY + 124, '位 3 及以后：draft 位直接不看（首拒即停）', 9.2,
        lc.C_MUTE, 'start', maxw=PW - 28, tag='side:l5')
# 幽灵行（位 3 被丢弃）
gy = TY + HDR_H + 3 * ROW_H
lc.rect(TX, gy, TW, ROW_H, '#ffffff', lc.C_FAINT, rx=0, sw=1.0, dash=True)
for c, (cx_, cw) in enumerate(xs):
    v = ('位 3', '（还有候选）', '—', '—', '—', '—', '整段丢弃')[c]
    lc.text(cx_ + cw / 2, gy + 28, v, 9.5, lc.C_FAINT, 'middle', maxw=cw - 6,
            tag='g%d' % c)
lc.seg(TX, gy + ROW_H / 2, TX + TW, gy + ROW_H / 2, lc.C_ABORT, 1.2)

# ---------------- 底部：全收位型徽章 ----------------
BY = gy + ROW_H + 18
lc.rect(MX, BY, TW + 22 + PW, 62, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 22, '对照 · 全收位型（另一组抽签 seed=0）', 10, lc.C_TXT,
        'start', True, tag='bonus:t')
lc.text(MX + 14, BY + 44, '位 0-2 的 u 全部落界内 → 追加一个只从 target 采的 bonus=A —— 一趟拿 4 个 token',
        9.2, '#334155', 'start', maxw=TW + PW, tag='bonus:l')

# ---------------- 结论 + 页脚 ----------------
CY = BY + 62 + 22
cc.conclusion(MX, BXR, CY,
              '四件事同表可见：min(1,·) 逐位接受、首拒即停、残差恢复、全收 bonus——验证规则的全部。')
cc.footer(MX, BXR, CY + 22, [
    '数值=固定 seed（seed=2）的 CPU 参考实现实跑；判据形态 u < p_t/p_d 见 vllm/v1/sample/rejection_sampler.py:L824-L830（min(1,·) 由 u 判据隐含，代码里从不写 min）',
    '行号基线 vLLM v0.27.1 · 论文口径 arXiv:2607.05147 §2.1'])

H = CY + 22 + 30
cc.write('ch33-fig-accept-reject-walk', W, H)
