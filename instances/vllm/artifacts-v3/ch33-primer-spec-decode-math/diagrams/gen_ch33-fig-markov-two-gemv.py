#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M3·步4 机制图 ch33-fig-markov-two-gemv（模板 flow）

claim：每步的全部计算就是两个 GEMV：embed(prev=3)=[-0.311732, 0.074316]（查表）→
bias=W_1[x]·W_2（首值 -0.061613 手工复算 ✓）；每步 256 次查表 + 33095680 次乘加
= 33095936 次操作（按乘加计），换 prev 只换查表行、W_2 不动——序列性代价被压到
0.2%~1.3% 整轮延迟。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑 + pin 源码锚点）。
"""
import _common33 as cc
import l0_common as lc

W = 1160
lc.reset()
MX, BXR = cc.header(
    W,
    '序列头每步的全部计算：两个 GEMV——查一行嵌入，乘一次投影',
    '「让每个位置知道前一个词」的全部代价：O(r) 查表 + O(rV) GEMV = 33095936 次操作（按乘加计）；换前驱只换查的行，W_2 永远不动',
    '放大自 L0 GPU 执行臂 · 序列头每步计算流（绿）')

# ---------------- 三节点流水 ----------------
NODE_Y, NODE_H = 100, 118
N1 = (MX, 180)          # prev 徽章
N2 = (MX + 220, MX + 520)  # W_1 查表
N3 = (MX + 580, MX + 980)  # W_2 GEMV
# 节点 1：prev
lc.rect(N1[0], NODE_Y, N1[1] - N1[0], NODE_H, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=1.8)
lc.text((N1[0] + N1[1]) / 2, NODE_Y + 30, 'prev', 10.5, lc.C_GPU_S, 'middle', True,
        tag='n1:t')
lc.ELEMS.append((((N1[0] + N1[1]) / 2 - 15, NODE_Y + 44, (N1[0] + N1[1]) / 2 + 15,
                  NODE_Y + 74),
                 '<circle cx="%.1f" cy="%.1f" r="14" fill="#ffffff" stroke="%s" '
                 'stroke-width="1.6"/>' % ((N1[0] + N1[1]) / 2, NODE_Y + 59, lc.C_GPU_S)))
lc.text((N1[0] + N1[1]) / 2, NODE_Y + 63, '3', 12, lc.C_GPU_S, 'middle', True,
        tag='n1:v')
lc.text((N1[0] + N1[1]) / 2, NODE_Y + 94, '上一个实采 token', 8, lc.C_MUTE, 'middle',
        maxw=N1[1] - N1[0] - 8, tag='n1:s')
# 节点 2：W_1 查表
lc.rect(N2[0], NODE_Y, N2[1] - N2[0], NODE_H, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=1.8)
lc.text((N2[0] + N2[1]) / 2, NODE_Y + 26, 'GEMV ① 查表 W_1[prev]', 10.5, lc.C_GPU_S,
        'middle', True, maxw=N2[1] - N2[0] - 12, tag='n2:t')
lc.text((N2[0] + N2[1]) / 2, NODE_Y + 50, 'r 维嵌入（r=256）', 8.5, lc.C_MUTE,
        'middle', tag='n2:s')
for i, v in enumerate([-0.311732, 0.074316]):
    y = NODE_Y + 66 + i * 20
    lc.rect(N2[0] + 34, y, 232, 16, '#ffffff', lc.C_GPU_S, rx=3, sw=1.0)
    lc.text(N2[0] + 38, y + 12, '%g' % v, 8.5, lc.C_GPU_S, 'start', True, tag='n2:v%d' % i)
lc.text(N2[0] + 34, NODE_Y + 66 + 2 * 20 + 8, '⋮（共 r 维）', 7.5, lc.C_MUTE, 'start',
        tag='n2:dots')
# 节点 3：W_2 GEMV
lc.rect(N3[0], NODE_Y, N3[1] - N3[0], NODE_H, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=1.8)
lc.text((N3[0] + N3[1]) / 2, NODE_Y + 26, 'GEMV ② 偏置 = 嵌入 · W_2', 10.5, lc.C_GPU_S,
        'middle', True, maxw=N3[1] - N3[0] - 12, tag='n3:t')
lc.text((N3[0] + N3[1]) / 2, NODE_Y + 50, 'V 维词表偏置行 B(prev,·)', 8.5, lc.C_MUTE,
        'middle', tag='n3:s')
lc.rect(N3[0] + 24, NODE_Y + 64, N3[1] - N3[0] - 48, 18, '#ffffff', lc.C_GPU_S,
        rx=3, sw=1.0)
lc.text(N3[0] + 30, NODE_Y + 77, '首值 -0.061613 = W_1[3]·W_2 列 0（手工复算 ✓）', 8.5,
        lc.C_GPU_S, 'start', True, maxw=N3[1] - N3[0] - 60, tag='n3:v')
lc.text((N3[0] + N3[1]) / 2, NODE_Y + 100, '→ 加到该位 base logits 上', 8.5,
        lc.C_MUTE, 'middle', tag='n3:o')
# 流水箭头
lc.seg(N1[1], NODE_Y + NODE_H / 2, N2[0], NODE_Y + NODE_H / 2, lc.C_GPU_S, 2.0, 'std')
lc.text((N1[1] + N2[0]) / 2, NODE_Y + NODE_H / 2 - 8, '查第 prev 行', 8, lc.C_GPU_S,
        'middle', maxw=(N2[0] - N1[1]) * 0.96, tag='a1')
lc.seg(N2[1], NODE_Y + NODE_H / 2, N3[0], NODE_Y + NODE_H / 2, lc.C_GPU_S, 2.0, 'std')
lc.text((N2[1] + N3[0]) / 2, NODE_Y + NODE_H / 2 - 8, 'r 维向量', 8, lc.C_GPU_S,
        'middle', maxw=(N3[0] - N2[1]) * 0.96, tag='a2')

# ---------------- 中部：换 prev 不换表 ----------------
SW_Y = 244
lc.rect(MX, SW_Y, BXR - MX, 74, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, SW_Y + 20, '换 prev=5（只换查表行，W_2 不动）→ bias = [-0.111318, -0.316849, -0.346378, -0.132149, 0.037032, 0.021483]', 9.5,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='sw:t')
lc.text(MX + 14, SW_Y + 40, '每步代价 = 256 次查表 + 33095680 次乘加 = 33095936 次操作', 9.5,
        lc.C_GPU_S, 'start', True, maxw=BXR - MX - 28, tag='sw:c')
lc.text(MX + 14, SW_Y + 58, '序列循环延迟开销 0.2%~1.3%（γ 4→16、batch=128，论文实测）——「轻量序列头」的全部含义', 9,
        lc.C_MUTE, 'start', maxw=BXR - MX - 28, tag='sw:l')

# ---------------- 底部：vLLM 两方法 ----------------
BY = SW_Y + 74 + 16
lc.rect(MX, BY, BXR - MX, 84, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 22, 'vLLM 把这两步做成 DSparkMarkovHead 的两个方法，给 speculator 的串行循环分两步调用：', 10,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='v:t')
bw_ = (BXR - MX - 28 - 16) / 2
for i, (sig, note) in enumerate([
        ('markov_embed(token_ids)', 'W_1 查表（O(r)）'),
        ('markov_bias(markov_embed)', 'W_2 GEMV（O(rV)）')]):
    x = MX + 14 + i * (bw_ + 16)
    lc.rect(x, BY + 34, bw_, 38, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.4)
    lc.text(x + bw_ / 2, BY + 50, sig, 9.5, lc.C_GPU_S, 'middle', True, maxw=bw_ - 10,
            tag='v:s%d' % i)
    lc.text(x + bw_ / 2, BY + 66, note, 8.2, lc.C_MUTE, 'middle', maxw=bw_ - 10,
            tag='v:n%d' % i)
lc.text(MX + 14, BY + 90 - 14, '权重不分片（replicated）：串行每步都跑、分片反而每步多一次 all-reduce · vllm/models/deepseek_v4/nvidia/dspark.py:L366-L372', 8.5,
        lc.C_FAINT, 'start', maxw=BXR - MX - 28, tag='v:f')

# ---------------- 结论 + 页脚 ----------------
CY = BY + 84 + 22
cc.conclusion(MX, BXR, CY,
              '两个 GEMV、约 3310 万次乘加，就是「知道前一个词」的全部价格——便宜到值得每次采样都付一遍。')
cc.footer(MX, BXR, CY + 22, [
    '数值=固定 seed 的 CPU 参考实现实跑（embed/bias 逐值复算，首值手工复算对拍）· GEMV 账与 0.2%~1.3% 引 arXiv:2607.05147 §3.1/§4.3.2',
    '行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-markov-two-gemv', W, H)
