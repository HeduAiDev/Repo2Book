#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M4·步4 机制图 ch33-fig-landing-topology（模板 flow）

claim：落地拓扑七件套：①权重随 target checkpoint ②自动检测 ③DSV4 复用全套
config ④parallel_drafting=True ⑤强制 V2 runner ⑥registry 双入口 ⑦块长卫兵——
诚实账：confidence_head 被显式 skip，置信调度未落地。
数字全部取 spec.numbers（pin 源码锚点）。
"""
import _common33 as cc
import l0_common as lc

W = 1160
lc.reset()
MX, BXR = cc.header(
    W,
    'DSpark 在 vLLM 的落地拓扑：从 checkpoint 到 speculator 的七件套装配线',
    '半自回归 drafter 已落地；置信调度（论文后半的数学）在 v0.27.1 未接线——底部通栏的诚实账与装配线同样重要',
    '放大自 L0 配置入口→采样出口列 · DSpark 装配路径')

# ---------------- 装配线（横向五站）----------------
AY, AH = 96, 190
NODES = [
    ('① checkpoint 包', '自带 DSpark 配件\nmodel=target、量化对齐\n“DeepSeek DSpark can\nship the weights inside\nthe target checkpoint”', lc.C_GPU_S),
    ('② SpeculativeConfig', '自动检测 architectures\n=Qwen3DSparkModel 等\n⑦ 块长卫兵：spec 长度\n< dspark_block_size\n→ 报错（乱码级拦截）', lc.C_API_S),
    ('⑤ 强制 V2 runner', '“DSpark is implemented\nonly by the V2 GPU\nmodel runner”\nV1 支路 ×：宁可报错\n不静默回落', lc.C_ENG_S),
    ('⑥ registry 双入口', 'DSparkDraftModel\n→deepseek_v4（DSV4）\nQwen3DSparkModel\n→qwen3_dspark\n③ DSV4 复用全套 config', lc.C_API_S),
    ('init_speculator 分发', '④ dflash/dspark→\nparallel_drafting=True\ndspark→DSparkSpeculator', lc.C_SAM_S),
]
NW = (BXR - MX - 4 * 24) / 5
for i, (title, body, col) in enumerate(NODES):
    x = MX + i * (NW + 24)
    lc.rect(x, AY, NW, AH, '#ffffff', col, rx=8, sw=1.6)
    lc.text(x + NW / 2, AY + 22, title, 9.5, col, 'middle', True, maxw=NW - 10,
            tag='n%d t' % i)
    lines = body.split('\n')
    for j, ln in enumerate(lines):
        lc.text(x + NW / 2, AY + 44 + j * 15, ln, 7.4, '#334155', 'middle',
                maxw=NW - 8, tag='n%db%d' % (i, j))
    if i < 4:
        lc.seg(x + NW, AY + AH / 2, x + NW + 24, AY + AH / 2, lc.C_MUTE, 1.8, 'std')
# V1 × 支路
vx = MX + 2 * (NW + 24) + NW / 2
lc.text(vx, AY + AH + 22, '×  V1：跑不了 dspark（显式报错，不静默回落）', 8.5,
        lc.C_ABORT, 'middle', True, maxw=260, tag='v1x')

# ---------------- 终点框：DSparkSpeculator ----------------
EY = AY + AH + 44
lc.rect(MX, EY, BXR - MX, 52, lc.C_SAM_F, lc.C_SAM_S, rx=8, sw=1.8)
lc.text((MX + BXR) / 2, EY + 22, 'DSparkSpeculator（GPU worker 的 draft 执行臂）', 10.5,
        lc.C_SAM_S, 'middle', True, maxw=600, tag='e:t')
lc.text((MX + BXR) / 2, EY + 40, '两阶段 draft + FULL CUDA graph——即本章「半自回归两阶段 / Markov 头」的全部机制', 8.5,
        lc.C_MUTE, 'middle', maxw=600, tag='e:s')
lc.seg((MX + BXR) / 2, EY - 24, (MX + BXR) / 2, EY, lc.C_MUTE, 1.8, 'std')

# ---------------- 底部：已落地 / 未落地双色横条 ----------------
BY = EY + 52 + 18
lc.rect(MX, BY, BXR - MX, 88, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 20, '诚实账：已落地 / 未落地', 10, lc.C_TXT, 'start', True,
        maxw=300, tag='h:t')
gw = (BXR - MX - 28 - 12) * 0.5
lc.rect(MX + 14, BY + 32, gw, 42, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.6)
lc.text(MX + 14 + gw / 2, BY + 48, '半自回归 drafter：已落地', 9.5, lc.C_GPU_S,
        'middle', True, maxw=gw - 10, tag='h:g')
lc.text(MX + 14 + gw / 2, BY + 66, '权重随包、自动检测、V2 执行、FULL CUDA graph', 7.8,
        '#334155', 'middle', maxw=gw - 10, tag='h:g2')
gx2 = MX + 14 + gw + 12
lc.rect(gx2, BY + 32, gw, 42, '#f1f5f9', lc.C_MUTE, rx=6, sw=1.2)
lc.text(gx2 + gw / 2, BY + 48, '置信调度：未落地（论文侧蓝图）', 9.5, lc.C_MUTE,
        'middle', True, maxw=gw - 10, tag='h:w')
lc.text(gx2 + gw / 2, BY + 66, '“confidence_head is not wired into inference yet; skip its weights”——权重到货、加载显式跳过', 7.8,
        lc.C_MUTE, 'middle', maxw=gw - 10, tag='h:w2')

# ---------------- 结论 + 页脚 ----------------
CY = BY + 88 + 22
cc.conclusion(MX, BXR, CY,
              '七件套装配线把「method=dspark 一个词」背后的全部检查点排成一列——而双色横条标出本章后半（置信调度）在引擎里的真实位置。')
cc.footer(MX, BXR, CY + 22, [
    '①⑦ vllm/config/speculative.py:L717-L723（随包发布）· L1035-L1058（块长卫兵）· ⑤ vllm/config/vllm.py:L583-L595（强制 V2）',
    '⑥ vllm/model_executor/models/registry.py:L617-L618 · 分发 vllm/v1/worker/gpu/spec_decode/__init__.py:L8-L40 · skip vllm/model_executor/models/qwen3_dspark.py:L184-L188 · 行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-landing-topology', W, H)
