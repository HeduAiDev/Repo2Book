#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M3·步2 机制图 ch33-fig-prev-chain-trace（模板 state-table）

claim：prev 链的逐步手推：位 0 查 anchor 行（bias top2 5:0.91057/0:−0.839943）
采出 1、位 1 查 token 1 行（1:0.880667）采出 2、位 2 查 token 2 行采出 2——
每步 bias 行随 prev 换、U_k 不动；probabilistic 模式还把 q 行逐位记下
（top1 0.447351/0.458878/0.49183）供验证。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑）。
"""
import _common33 as cc
import l0_common as lc

W = 1160
lc.reset()
MX, BXR = cc.header(
    W,
    'prev 链逐步手推：每步只有偏置行在换，base logits U 从头到尾不动',
    '位 0 查 anchor（token 2）那一行、位 1 换 token 1 行、位 2 又换回 token 2 行——「并行干重活」= U 是骨干一次性给定的',
    '放大自 L0 GPU 执行臂 · 序列头三次步进（绿）')

TX, TY = MX, 96
COLS = [('位', 54), ('prev（查哪行）', 128), ('U_k top2', 168), ('bias top2（B 行）', 196),
        ('采出', 90)]
HDR_H, ROW_H = 32, 50
xs = []
x = TX
for name, w in COLS:
    xs.append((x, w))
    x += w
TW = x - TX
lc.rect(TX, TY, TW, HDR_H, '#f8fafc', lc.C_MUTE, rx=0, sw=1.2)
for (cx_, cw), (name, _) in zip(xs, COLS):
    lc.text(cx_ + cw / 2, TY + 21, name, 9.5, lc.C_MUTE, 'middle', True, maxw=cw - 6,
            tag='th:' + name[:4])
ROWS = [
    ('位 0', 'prev=2（anchor）', '1: 0.9732　2: 0.413978', '5: 0.91057　0: −0.839943', '1'),
    ('位 1', 'prev=1', '2: 1.313929　1: 0.999298', '1: 0.880667　0: −0.7838', '2'),
    ('位 2', 'prev=2', '2: 0.766341　1: 0.570082', '（同 anchor 行，B 不再变）', '2'),
]
for r, vals in enumerate(ROWS):
    y = TY + HDR_H + r * ROW_H
    lc.rect(TX, y, TW, ROW_H, '#ffffff', lc.C_MUTE, rx=0, sw=1.0)
    for c, (cx_, cw) in enumerate(xs):
        col, bold = lc.C_TXT, False
        if c == 1:
            col, bold = lc.C_GPU_S, True
        if c == 4:
            col, bold = lc.C_SAM_S, True
        lc.text(cx_ + cw / 2, y + 30, vals[c], 9.5, col, 'middle', bold, maxw=cw - 8,
                tag='r%d%d' % (r, c))
# prev 链箭头（跨行）：右缘外侧 2→1→2
CHAIN_X = TX + TW + 56
lc.text(CHAIN_X, TY + 12, 'prev 链', 9, lc.C_GPU_S, 'middle', True, tag='chain:t')
prevs = [2, 1, 2]
for r in range(3):
    y_mid = TY + HDR_H + r * ROW_H + ROW_H / 2 + 3
    lc.ELEMS.append(((CHAIN_X - 13, y_mid - 13, CHAIN_X + 13, y_mid + 13),
                     '<circle cx="%d" cy="%.1f" r="13" fill="%s" stroke="%s" '
                     'stroke-width="1.4"/>' % (CHAIN_X, y_mid, lc.C_GPU_F, lc.C_GPU_S)))
    lc.text(CHAIN_X, y_mid + 3.5, str(prevs[r]), 10.5, lc.C_GPU_S, 'middle', True,
            tag='ch%d' % r)
    if r < 2:
        y0 = y_mid + 13
        y1 = y_mid + ROW_H - 13
        lc.parrow([(CHAIN_X, y0), (CHAIN_X + 26, (y0 + y1) / 2), (CHAIN_X, y1)],
                  lc.C_GPU_S, 1.5, 'std')
        lc.text(CHAIN_X + 32, (y0 + y1) / 2 + 3, '采出 %d' % (prevs[r + 1]), 7.8,
                lc.C_GPU_S, 'start', tag='chl%d' % r)

# 侧栏：U 不动 / bias 行在换
SXX = CHAIN_X + 108
SW = BXR - SXX
lc.rect(SXX, TY, SW, HDR_H + 3 * ROW_H, '#ffffff', lc.C_GPU_S, rx=8, sw=1.2)
lc.text(SXX + SW / 2, TY + 24, '每步在换的只有 bias 行', 10, lc.C_GPU_S, 'middle',
        True, maxw=SW - 16, tag='side:t')
lc.text(SXX + SW / 2, TY + 52, 'U_0/U_1/U_2：骨干一次给齐、不动', 9, '#334155',
        'middle', maxw=SW - 16, tag='side:l1')
lc.text(SXX + SW / 2, TY + 72, 'B(prev)：查表行随实采前驱逐位换', 9, '#334155',
        'middle', maxw=SW - 16, tag='side:l2')
lc.text(SXX + SW / 2, TY + 92, 'draft = [1, 2, 2]', 11, lc.C_SAM_S, 'middle', True,
        maxw=SW - 16, tag='side:dt')

# ---------------- 底部：probabilistic 变体 ----------------
BY = TY + HDR_H + 3 * ROW_H + 20
lc.rect(MX, BY, BXR - MX, 66, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 22, 'probabilistic 模式（另一组抽签 seed=9）：draft=[4, 1, 2]，每步完整分布 q(x)=softmax(U_k+B) 逐位记进 draft_logits 缓冲', 9.5,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='p:t')
lc.text(MX + 14, BY + 42, 'q 行 top1 = 0.447351 / 0.458878 / 0.49183——验证阶段概率比 p_t(x)/p_d(x) 的分母就是它', 9,
        lc.C_SAM_S, 'start', True, maxw=BXR - MX - 28, tag='p:l')

# ---------------- 结论 + 页脚 ----------------
CY = BY + 66 + 22
cc.conclusion(MX, BXR, CY,
              '「prev 链」的全部含义在这张表上：换行查表是串行的全部，U 的每一次复用都是并行的红利。')
cc.footer(MX, BXR, CY + 22, [
    '数值=固定 seed 的 CPU 参考实现实跑（top2 取自逐步打印）· Eq.(4) 的重构口径（加法注入）引 arXiv:2607.05147 §3.1',
    'vLLM 对应 _sample_sequential 逐步循环与 draft_logits 缓冲 · 行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-prev-chain-trace', W, H)
