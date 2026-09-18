#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M4·步1 机制图 ch33-fig-query-layout（模板 state-table）

claim：同一 kernel 的两种布局：DSpark 每请求恰好 N=5 个查询（anchor 当第一预测位、
sample_pos=query_pos+1、KV 槽 30/31/44/45/46），DFlash 1+N=6 个查询（anchor 是
bonus 槽、mask 位预测自身位置）；调度器 lookahead 相差 1 槽（5 vs 6），块长卫兵把
『小了』判为乱码级错误。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑 + pin 源码锚点）。
"""
import _common33 as cc
import l0_common as lc

W = 1160
lc.reset()
MX, BXR = cc.header(
    W,
    '同一 kernel 的两种布局：DSpark 恰好 N 个查询，DFlash 1+N 个',
    'DSpark 把 anchor 本身当第一道题——5 道题 5 格卷子，每格预测它的下一格；DFlash 的 anchor 只当上下文、5 个 mask 各填自己的空',
    '放大自 L0 GPU 执行臂 · draft 输入构造与调度 lookahead')

# ---------------- 上半：双栏布局表 ----------------
PW = (BXR - MX - 20) / 2
TAB_Y = 92
ROW_H = 26

def layout_table(px, title, sub, rows):
    lc.rect(px, TAB_Y, PW, 40 + len(rows) * ROW_H + 26, '#ffffff', lc.C_MUTE, rx=8,
            sw=1.2)
    lc.text(px + 14, TAB_Y + 20, title, 10.5, lc.C_TXT, 'start', True, maxw=PW - 28,
            tag='t:' + title[:6])
    lc.text(px + 14, TAB_Y + 36, sub, 8, lc.C_MUTE, 'start', maxw=PW - 28, tag='s:' + title[:6])
    hy = TAB_Y + 52
    cols = [(px + 14, 'query_off'), (px + 96, '输入'),
            (px + 226, 'target_off（预测哪位）')]
    for cx_, name in cols:
        lc.text(cx_, hy, name, 8, lc.C_MUTE, 'start', tag='h:' + name)
    for ri, (qoff, inp, toff) in enumerate(rows):
        y = hy + 14 + ri * ROW_H
        hi = toff.startswith('—')
        if ri % 2 == 0:
            lc.rect(px + 10, y - 2, PW - 20, ROW_H - 2, '#f8fafc', '#f8fafc', rx=3,
                    sw=0.5)
        lc.text(cols[0][0], y + 12, qoff, 8.5, lc.C_TXT, 'start', tag='q%d%s' % (ri, qoff))
        lc.text(cols[1][0], y + 12, inp, 8.5, lc.C_TXT, 'start', tag='i%d%s' % (ri, qoff))
        lc.text(cols[2][0], y + 12, toff, 8.5, (lc.C_MUTE if hi else lc.C_GPU_S),
                'start', not hi, maxw=PW - 240, tag='t%d%s' % (ri, qoff))

layout_table(
    MX, 'DSpark · 每请求恰好 N=5 个查询', '输入串 [7, -1, -1, -1, -1]（anchor 当第一预测位）',
    [('0', '7（anchor）', '1 = 0+1'),
     ('1', '-1（mask）', '2'),
     ('2', '-1（mask）', '3'),
     ('3', '-1（mask）', '4'),
     ('4', '-1（mask）', '5')])
layout_table(
    MX + PW + 20, 'DFlash 对照 · 1+N=6 个查询', 'anchor 只当上下文，不采样；mask 位预测自身位置',
    [('0', '7（anchor）', '— 不采样（bonus 槽）'),
     ('1', '-1（mask）', '1（预测位 1 自己）'),
     ('2', '-1（mask）', '2（预测位 2 自己）'),
     ('3', '-1（mask）', '3（预测位 3 自己）'),
     ('4', '-1（mask）', '4（预测位 4 自己）'),
     ('5', '-1（mask）', '5（预测位 5 自己）')])

# ---------------- 中部：KV 槽带 ----------------
DS_H = 40 + 6 * ROW_H + 26
KY = TAB_Y + DS_H + 26
lc.rect(MX, KY, BXR - MX, 108, '#ffffff', lc.C_KV_S, rx=8, sw=1.2)
lc.text(MX + 14, KY + 20, 'KV 寻址：query_pos 10..14 落进物理槽 30 / 31 / 44 / 45 / 46（block_size=4、block_table=[5,9,7,11]）', 10,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='k:t')
SLOTS = [28, 29, 30, 31, 44, 45, 46, 47]
CELL_W, CELL_H = 86, 30
sx = MX + 26
for bi, blk in enumerate([7, 11]):
    x0 = sx + bi * (4 * CELL_W + 22)
    lc.rect(x0 - 4, KY + 34, 4 * CELL_W + 8, CELL_H + 10, '#ecfeff', lc.C_KV_S,
            rx=6, sw=1.0, dash=True)
    lc.text(x0 - 4 + 2, KY + 34 + CELL_H + 20, '物理块 %d（块表第 %d 块）' % (blk, bi + 3),
            7.5, lc.C_KV_S, 'start', tag='kb%d' % bi)
    for ci in range(4):
        slot = SLOTS[bi * 4 + ci]
        qpos = 10 + bi * 4 + ci - 2
        used = slot in (30, 31, 44, 45, 46)
        x = x0 + ci * CELL_W
        lc.rect(x, KY + 40, CELL_W - 4, CELL_H, ('#cffafe' if used else '#ffffff'),
                lc.C_KV_S, rx=4, sw=(1.8 if used else 1.0))
        lc.text(x + (CELL_W - 4) / 2, KY + 54, '槽 %d' % slot, 8.5,
                (lc.C_KV_S if used else lc.C_FAINT), 'middle', used,
                maxw=CELL_W - 8, tag='ks%d' % slot)
        lc.text(x + (CELL_W - 4) / 2, KY + 66, ('query_pos %d' % qpos) if used else '（未用）',
                7.2, (lc.C_TXT if used else lc.C_FAINT), 'middle', maxw=CELL_W - 8,
                tag='kq%d' % slot)
lc.text(MX + 26 + 2 * (4 * CELL_W + 22) + 8, KY + 52, 'slot = 物理块×4 + 块内偏移', 8.5,
        lc.C_MUTE, 'start', maxw=BXR - MX - 26 - 2 * (4 * CELL_W + 22) - 16,
        tag='k:f')
lc.text(MX + 26 + 2 * (4 * CELL_W + 22) + 8, KY + 70, '——纯索引算术，零拷贝', 8.5,
        lc.C_MUTE, 'start', maxw=BXR - MX - 26 - 2 * (4 * CELL_W + 22) - 16,
        tag='k:f2')

# ---------------- 下部：lookahead + 卫兵 ----------------
BY = KY + 108 + 16
BW = (BXR - MX - 20) / 2
lc.rect(MX, BY, BW, 64, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 22, '调度器 lookahead：DFlash 6 vs DSpark 5', 10, lc.C_TXT,
        'start', True, maxw=BW - 28, tag='la:t')
lc.text(MX + 14, BY + 42, 'DFlash 每请求多占 1 个槽——布局差 1，调度账也差 1', 8.8,
        lc.C_MUTE, 'start', maxw=BW - 28, tag='la:l')
G2X = MX + BW + 20
lc.rect(G2X, BY, BW, 64, '#fef2f2', lc.C_ABORT, rx=8, sw=1.4)
lc.text(G2X + 14, BY + 22, '块长卫兵：num_speculative_tokens=3 < dspark_block_size=5', 9.5,
        lc.C_ABORT, 'start', True, maxw=BW - 28, tag='g:t')
lc.text(G2X + 14, BY + 42, '→ ValueError（“Smaller values produce incorrect output”）——不是多验几个白验，是乱码级错误', 8.5,
        lc.C_ABORT, 'start', maxw=BW - 28, tag='g:l')

# ---------------- 结论 + 页脚 ----------------
CY = BY + 64 + 22
cc.conclusion(MX, BXR, CY,
              '『γ 个输入出 γ 个 logits』的代码面就是这张双栏表：anchor 是题还是卷首语，差出一个查询位、一个调度槽、一道卫兵。')
cc.footer(MX, BXR, CY + 22, [
    '数值=固定 seed 的 CPU 参考实现实跑（布局/KV 槽/lookahead 逐值断言）· 卫兵行为口径 vllm/config/speculative.py:L1035-L1058',
    '行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-query-layout', W, H)
