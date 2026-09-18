#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M6·步2 机制图 ch33-fig-alg1-admission（模板 state-table）

claim：Algorithm 1 的贪心准入轨迹表：场景 B 里 init Θ=16.0 → 收 req0·位1 的
a=0.9 得 Θ=2.9*6.0=17.4（Θ_best）→ 再收 req0·位2 的 a=0.63 得 3.53*4.8=16.944
回落 → break、
ℓ*=[1,0]；场景 A（论文 App.A 数字）里第一个候选就把 Θ 从 1.0 打到 0.9——
重载侧「一个都不验」是全局最优。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑）。
"""
import _common33 as cc
import l0_common as lc

W = 1160
lc.reset()
MX, BXR = cc.header(
    W,
    'Algorithm 1 逐行长这样：每收一个查一次 Θ，一跌就刹车',
    '先把每请求的置信连乘成前缀存活 a、全局降序排成候选池，从「谁都不加」起步逐步收人——Θ=τ*·SPS(B) 涨了记账继续、跌了 break',
    '放大自 L0 调度带↔采样出口列 · 准入接口（青+品红）')

# ---------------- 主表：场景 B ----------------
TY = 92
COLS = [('动作', 214), ('收的 a', 78), ('B', 60), ('τ*', 66), ('SPS(B)', 78),
        ('Θ=τ*·SPS', 128), ('Θ 迷你条', 280)]
HDR_H, ROW_H = 30, 44
xs = []
x = MX
for name, w in COLS:
    xs.append((x, w))
    x += w
TW = x - MX
lc.text(MX, TY - 8, '场景 B（B 从 2 起步：两个请求排队）', 9.5, lc.C_TXT, 'start',
        True, maxw=400, tag='sc:t')
lc.rect(MX, TY, TW, HDR_H, '#f8fafc', lc.C_MUTE, rx=0, sw=1.2)
for (cx_, cw), (name, _) in zip(xs, COLS):
    lc.text(cx_ + cw / 2, TY + 20, name, 9, lc.C_MUTE, 'middle', True, maxw=cw - 6,
            tag='th:' + name[:4])
ROWS = [
    ('init（谁都不加）', '—', '2', '2.0', '8.0', '16.0', 16.0, False, False),
    ('admit 请求 0', '0.9', '3', '2.9', '6.0', '17.4', 17.4, True, False),
    ('admit 请求 0 · 位 2', '0.63', '4', '3.53', '4.8', '16.944', 16.944, False, True),
]
BAR_X, BAR_W = xs[6][0] + 6, xs[6][1] - 40
BAR_MAX = 18.0
for r, (act, a, B, tau, sps, th, thv, best, drop) in enumerate(ROWS):
    y = TY + HDR_H + r * ROW_H
    lc.rect(MX, y, TW, ROW_H, ('#fdf2f8' if best else '#ffffff'),
            (lc.C_SAM_S if best else lc.C_MUTE), rx=0, sw=(1.8 if best else 1.0))
    vals = (act, a, B, tau, sps, th)
    for c, (cx_, cw) in enumerate(xs[:6]):
        col = lc.C_TXT
        bold = False
        if c == 5:
            col = lc.C_SAM_S if best else (lc.C_ABORT if drop else lc.C_MUTE)
            bold = True
        if c == 0:
            col = '#334155'
        lc.text(cx_ + cw / 2, y + 27, vals[c], 9.3, col, 'middle', bold, maxw=cw - 8,
                tag='r%d%d' % (r, c))
    # Θ 迷你条
    lc.rect(BAR_X, y + 15, BAR_W, 16, '#f1f5f9', lc.C_MUTE, rx=3, sw=0.8)
    lc.rect(BAR_X, y + 15, BAR_W * thv / BAR_MAX, 16,
            (lc.C_SAM_F if best else ('#fef2f2' if drop else '#e2e8f0')),
            (lc.C_SAM_S if best else (lc.C_ABORT if drop else lc.C_MUTE)), rx=3,
            sw=1.1)
    if best:
        lc.text(BAR_X + BAR_W * thv / BAR_MAX + 8, y + 27, 'Θ_best=17.4', 8.5,
                lc.C_SAM_S, 'start', True, tag='bb')
    if drop:
        lc.text(BAR_X + BAR_W * thv / BAR_MAX + 8, y + 27, '回落 → break', 8.5,
                lc.C_ABORT, 'start', True, tag='db')
# break 行
y4 = TY + HDR_H + 3 * ROW_H
lc.rect(MX, y4, TW, ROW_H, '#ffffff', lc.C_MUTE, rx=0, sw=1.0, dash=True)
lc.text(MX + xs[0][1] / 2, y4 + 27, 'break · 返回', 9.3, lc.C_MUTE, 'middle', True,
        maxw=xs[0][1] - 8, tag='r4a')
lc.text(MX + xs[0][1] + (TW - xs[0][1]) / 2, y4 + 27,
        'ℓ*=[1, 0]：只给请求 0 验 1 位——全枚举对照 4.03→16.12、4.18→14.63，都不及 17.4（单峰下早停不亏）', 9,
        lc.C_TXT, 'middle', True, maxw=TW - xs[0][1] - 16, tag='r4b')

# ---------------- 底部：场景 A 迷你表 + 注 ----------------
AY = y4 + ROW_H + 20
lc.rect(MX, AY, BXR - MX, 74, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, AY + 20, '场景 A（论文 App.A 数字，重载：B 从 1 起步）：init Θ=1.0 → 第一个候选 a=0.8 一收，B=2、τ*=1.8、Θ=0.9 → 立即 break', 9.5,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='a:t')
lc.text(MX + 14, AY + 42, 'ℓ*=[0]、Θ_best=1.0——重载时吞吐曲线掉得凶，「一个 draft 都不验」才是全局最优', 9,
        lc.C_ABORT, 'start', True, maxw=BXR - MX - 28, tag='a:l')
lc.text(MX + 14, AY + 60, '单调不增的 a 保证候选池按贡献排序——贪心+早停的合法性来自上一张图的链式法则', 8.5,
        lc.C_MUTE, 'start', maxw=BXR - MX - 28, tag='a:n')

# ---------------- 结论 + 页脚 ----------------
CY = AY + 74 + 22
cc.conclusion(MX, BXR, CY,
              '三本账同步走：B 加一、τ* 加上它的 a、Θ 查一次吞吐表——「跌了就刹车」四个字就是 Algorithm 1 的全部控制流。')
cc.footer(MX, BXR, CY + 22, [
    '数值=固定 seed 的 CPU 参考实现实跑（场景 B/全枚举/场景 A 逐值断言；场景 A 数字=论文 App.A）· Algorithm 1 引 arXiv:2607.05147 §3.2.2',
    '行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-alg1-admission', W, H)
