#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M4·步3 机制图 ch33-fig-frozen-share（模板 flow）

claim：embed/lm_head 缺省别名 target（冻结）：draft 权重只多 backbone+markov_head+
confidence_head 几件——这就是 DSpark 权重能随 target checkpoint 发布、用户只配
method=dspark 的原因。
数字全部取 spec.numbers（pin 源码锚点）。
"""
import _common33 as cc
import l0_common as lc

W = 1160
lc.reset()
MX, BXR = cc.header(
    W,
    'embed / lm_head 缺省别名 target（冻结）——DSpark 权重随 target checkpoint 发布',
    '这两块最贵的矩阵（各 V×d）draft 不带自己的：加载器直接别名成 target 同名模块，且全程冻结',
    '放大自 L0 GPU 执行臂 · 权重装载环节（绿）')

# ---------------- 左：target checkpoint 包 ----------------
LW = 380
lc.rect(MX, 90, LW, 300, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.0)
lc.text(MX + 16, 114, 'target checkpoint 发布包', 11.5, lc.C_GPU_S, 'start', True,
        maxw=LW - 32, tag='p:t')
lc.text(MX + 16, 132, '（论文 §3.3：与 target 共享 embedding 与 language modeling head，且全程冻结）', 7.6,
        lc.C_MUTE, 'start', maxw=LW - 32, tag='p:s')
# embed/lm_head 大块（冻结）
for i, name in enumerate(['embed_tokens（V×d）', 'lm_head（V×d）']):
    y = 148 + i * 62
    lc.rect(MX + 24, y, LW - 48, 50, '#ffffff', lc.C_GPU_S, rx=6, sw=1.8)
    lc.text(MX + 24 + (LW - 48) / 2, y + 30, name, 9.5, lc.C_GPU_S, 'middle', True,
            maxw=LW - 60, tag='p:%d' % i)
    lc.rect(MX + LW - 96, y + 6, 66, 18, '#f1f5f9', lc.C_MUTE, rx=8, sw=1.0)
    lc.text(MX + LW - 63, y + 18.5, '冻结', 8, lc.C_MUTE, 'middle', True, tag='p:fz%d' % i)
# DSpark 附加小件
lc.text(MX + 24, 282, '随包多出的 DSpark 配件：', 8.5, lc.C_TXT, 'start', True,
        maxw=LW - 48, tag='p:x')
for i, name in enumerate(['backbone（骨干）', 'markov_head', 'confidence_head']):
    x = MX + 24 + i * 112
    lc.rect(x, 292, 104, 34, '#ffffff', lc.C_GPU_S, rx=5, sw=1.3)
    lc.text(x + 52, 312, name, 8, lc.C_GPU_S, 'middle', True, maxw=98,
            tag='p:x%d' % i)
lc.text(MX + 24, 348, '体积大头（两个 V×d 大矩阵）本来就在包里', 8, lc.C_MUTE,
        'start', maxw=LW - 48, tag='p:n')
lc.text(MX + 24, 366, '——draft 不必再带一份', 8, lc.C_MUTE, 'start', maxw=LW - 48,
        tag='p:n2')

# ---------------- 右：draft 模型框 ----------------
RX = MX + LW + 210
RW = BXR - RX
lc.rect(RX, 90, RW, 300, '#ffffff', lc.C_GPU_S, rx=10, sw=1.8)
lc.text(RX + 16, 114, 'draft 模型（装载后）', 11, lc.C_GPU_S, 'start', True,
        maxw=RW - 32, tag='d:t')
# 两个别名槽（虚框）
for i, name in enumerate(['embed_tokens', 'lm_head']):
    y = 138 + i * 62
    lc.rect(RX + 20, y, RW - 40, 50, '#ffffff', lc.C_GPU_S, rx=6, sw=1.4, dash=True)
    lc.text(RX + 34, y + 30, name, 9.5, lc.C_GPU_S, 'start', True, maxw=140,
            tag='d:%d' % i)
    lc.text(RX + RW - 34, y + 30, '← 别名 target（del 自己的）', 8.2, lc.C_MUTE,
            'end', maxw=170, tag='d:al%d' % i)
# 自有三件
for i, (name, gray) in enumerate([('backbone', False), ('markov_head', False),
                                  ('confidence_head', True)]):
    x = RX + 20 + i * ((RW - 40 - 24) / 3)
    w_ = (RW - 40 - 24) / 3
    y = 268
    lc.rect(x, y, w_, 40, ('#f1f5f9' if gray else lc.C_GPU_F),
            (lc.C_MUTE if gray else lc.C_GPU_S), rx=5, sw=1.3)
    lc.text(x + w_ / 2, y + 25, name, 8.2, (lc.C_MUTE if gray else lc.C_GPU_S),
            'middle', True, maxw=w_ - 6, tag='d:o%d' % i)
    if gray:
        lc.text(x + w_ / 2, y + 54, 'skip：未接线', 7.5, lc.C_MUTE, 'middle',
                tag='d:sk')
lc.text(RX + 20, 250, '自有件（随包到货）：', 8.5, lc.C_TXT, 'start', True,
        maxw=RW - 40, tag='d:own')

# ---------------- 中间：别名箭头 ----------------
for i in range(2):
    y = 173 + i * 62
    lc.seg(MX + LW, y, RX, y, lc.C_GPU_S, 2.0, 'std', dash=True)
    lc.text((MX + LW + RX) / 2, y - 8, 'load_dspark_model 别名', 8, lc.C_GPU_S,
            'middle', maxw=200, tag='al:l%d' % i)
lc.text((MX + LW + RX) / 2, 260, '_should_share', 8, lc.C_MUTE, 'middle',
        maxw=200, tag='al:sh')
lc.text((MX + LW + RX) / 2, 274, '按 has_own_* 标志', 7.5, lc.C_MUTE, 'middle',
        maxw=200, tag='al:sh2')
lc.text((MX + LW + RX) / 2, 288, '+ 形状判定', 7.5, lc.C_MUTE, 'middle', maxw=200,
        tag='al:sh3')

# ---------------- 底部：配置面 ----------------
BY = 90 + 300 + 16
lc.rect(MX, BY, BXR - MX, 62, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 20, '配置面：method=dspark → model=target_model_config.model、量化对齐——“DeepSeek DSpark can ship the weights inside the target checkpoint”', 9.5,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='c:t')
lc.text(MX + 14, BY + 42, 'load_weights 显式 skip：mask_embedding（未用占位）/ confidence_head（未接线）/ 缺省时的 embed_tokens、lm_head——用户不用另配 draft 模型', 9,
        '#334155', 'start', maxw=BXR - MX - 28, tag='c:l')

# ---------------- 结论 + 页脚 ----------------
CY = BY + 62 + 22
cc.conclusion(MX, BXR, CY,
              '「权重随 checkpoint 发布」的全部根据：最贵的两块矩阵是借来的且冻结，随包增量只有骨干、Markov 头、置信头几件小件。')
cc.footer(MX, BXR, CY + 22, [
    '别名装载 vllm/v1/worker/gpu/spec_decode/dspark/utils.py:L60-L77（load_dspark_model）· 配置面 vllm/config/speculative.py:L717-L723 · load_weights skip vllm/model_executor/models/qwen3_dspark.py:L184-L196',
    '论文 §3.3 引 arXiv:2607.05147 · 行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-frozen-share', W, H)
