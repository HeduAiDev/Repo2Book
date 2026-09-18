#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M2·步3 机制图 ch33-fig-drafter-lineage（模板 swimlane）

claim：谱系三代的时间形状：自回归 γ 次串行前向（T_draft∝γ、被迫浅层 1 层）；
并行单次前向（可深 5 层、γ 可 16）；DSpark=并行骨干 1 次前向 + γ 次查表加法的
序列头——T_draft 近独立于 γ 的同时拿到块内依赖。
数字全部取 spec.numbers（pin 源码锚点 + 论文 §4.1/§2.2/§4.3.2）。
"""
import _common33 as cc
import l0_common as lc

W = 1160
lc.reset()
MX, BXR = cc.header(
    W,
    'drafter 谱系三代的时间形状：串行 γ 拍 · 并行 1 拍 · 半自回归 1 拍+微串行',
    '同一时间轴：自回归被 γ 次前向逼成浅网小块；并行一次前向养得起深骨干大块；DSpark 用轻量 Markov 头把块内依赖补回来',
    '放大自 L0 GPU 执行臂 · draft 前向段（绿）')

LX, LW = MX, 168          # 泳道名区
TX0 = LX + LW + 16        # 时间轴起点
TX1 = BXR - 200           # 时间轴终点（右侧留层深徽章）
LANE_H, LANE_GAP, LY0 = 108, 16, 92

LANES = [
    ('自回归', 'EAGLE · MTP', 'γ 次串行前向', lc.C_ENG_S),
    ('并行', 'DFlash · Medusa', '单次前向出整块', lc.C_API_S),
    ('半自回归', 'DSpark', '并行骨干 + Markov 序列头', lc.C_SAM_S),
]
for li, (name, models, sub, col) in enumerate(LANES):
    ly = LY0 + li * (LANE_H + LANE_GAP)
    lc.rect(LX, ly, BXR - LX, LANE_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
    lc.text(LX + 14, ly + 26, name, 11.5, col, 'start', True, maxw=LW - 28,
            tag='ln%d' % li)
    lc.text(LX + 14, ly + 44, models, 9, lc.C_MUTE, 'start', maxw=LW - 28,
            tag='lm%d' % li)
    lc.text(LX + 14, ly + 62, sub, 8.5, lc.C_MUTE, 'start', maxw=LW - 28,
            tag='ls%d' % li)
    # 右侧层深/γ 徽章
    badges = [('层深 1 层', '被迫浅网'), ('层深 5 层', 'γ 可 16'),
              ('层深 5 层', 'T_draft 近独立于 γ')][li]
    bx = TX1 + 12
    lc.text(bx, ly + 38, badges[0], 9.5, col, 'start', True, maxw=BXR - bx - 8,
            tag='bd%d' % li)
    lc.text(bx, ly + 58, badges[1], 8.5, lc.C_MUTE, 'start', maxw=BXR - bx - 8,
            tag='bd2%d' % li)

def lane_body(li, draw):
    ly = LY0 + li * (LANE_H + LANE_GAP)
    draw(ly)

# --- 泳道 1：γ 次串行小方块 ---
def draw_ar(ly):
    n = 4
    bw = (TX1 - TX0 - 8 - (n - 1) * 8) / n
    for i in range(n):
        x = TX0 + 4 + i * (bw + 8)
        lc.rect(x, ly + 24, bw, 40, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.5)
        lc.text(x + bw / 2, ly + 48, 'draft 前向', 8.2, lc.C_GPU_S, 'middle', True,
                maxw=bw - 4, tag='ar%d' % i)
    lc.text(TX0 + 4, ly + 82, 'for step in range(1, self.num_speculative_steps)：每步一次前向 → T_draft ∝ γ',
            8.5, lc.C_GPU_S, 'start', maxw=TX1 - TX0 - 8, tag='arn')
lane_body(0, draw_ar)

# --- 泳道 2：单次宽方块 ---
def draw_par(ly):
    lc.rect(TX0 + 4, ly + 24, TX1 - TX0 - 8, 40, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.8)
    lc.text((TX0 + TX1) / 2, ly + 48, 'draft 前向（一次出整块）→ T_draft=O(1)，与块长近无关',
            9, lc.C_GPU_S, 'middle', True, maxw=TX1 - TX0 - 16, tag='par')
    lc.text(TX0 + 4, ly + 82, '代价：块内位置互相不知道对方采了什么（并行之病，下两图）', 8.5,
            lc.C_MUTE, 'start', maxw=TX1 - TX0 - 8, tag='parn')
lane_body(1, draw_par)

# --- 泳道 3：宽方块 + γ 个微小刻度 ---
def draw_semi(ly):
    w_total = TX1 - TX0 - 8
    w_back = w_total * 0.66
    lc.rect(TX0 + 4, ly + 24, w_back, 40, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.8)
    lc.text(TX0 + 4 + w_back / 2, ly + 48, '并行骨干 · 1 次前向', 9, lc.C_GPU_S,
            'middle', True, maxw=w_back - 8, tag='semib')
    sx = TX0 + 4 + w_back + 14
    n = 4
    tw_ = (TX1 - sx - 8 - (n - 1) * 6) / n
    for i in range(n):
        x = sx + i * (tw_ + 6)
        lc.rect(x, ly + 34, tw_, 20, '#fdf2f8', lc.C_SAM_S, rx=3, sw=1.3)
        lc.text(x + tw_ / 2, ly + 47, '查表', 7.5, lc.C_SAM_S, 'middle', maxw=tw_,
                tag='mk%d' % i)
    lc.text(sx, ly + 24, 'Markov 头序列循环（γ 次查表加法）', 8.2, lc.C_SAM_S, 'start',
            maxw=TX1 - sx, tag='semis')
    lc.text(TX0 + 4, ly + 82, '序列循环只给整轮延迟加 0.2%~1.3%（γ 4→16、batch=128 实测）——块内依赖补回来、时间形状仍近并行',
            8.5, lc.C_SAM_S, 'start', maxw=TX1 - TX0 - 8, tag='semin')
lane_body(2, draw_semi)

# 时间轴（三泳道共用，画在最下）
AY = LY0 + 3 * LANE_H + 2 * LANE_GAP + 12
lc.seg(TX0, AY, TX1, AY, lc.C_MUTE, 1.6)
lc.text((TX0 + TX1) / 2, AY + 16, '← 同一时间轴：draft 阶段的墙钟长度 →', 8.5,
        lc.C_MUTE, 'middle', tag='tax')

# ---------------- 底部：vLLM 方法表 12 法 ----------------
BY = AY + 30
lc.rect(MX, BY, BXR - MX, 66, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 22, 'vLLM 把谱系与零模型方法全部注册在 SpeculativeMethod 方法表（12 法）', 10,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='mt:t')
METHODS = ['ngram', 'medusa', 'mlp_speculator', 'draft_model', 'suffix', 'custom_class',
           'eagle', 'eagle3', 'mtp', 'dflash', 'ngram_gpu', 'dspark']
HI = {'eagle', 'mtp', 'dflash', 'dspark'}
cx = MX + 14
for m in METHODS:
    w = lc.tw(m, 8.5, True) + 16
    hi = m in HI
    lc.rect(cx, BY + 34, w, 20, (lc.C_GPU_F if hi else '#ffffff'),
            (lc.C_GPU_S if hi else lc.C_MUTE), rx=9, sw=(1.6 if hi else 1.0))
    lc.text(cx + w / 2, BY + 47.5, m, 8.5, (lc.C_GPU_S if hi else lc.C_MUTE),
            'middle', True, maxw=w - 4, tag='m:' + m)
    cx += w + 8

# ---------------- 结论 + 页脚 ----------------
CY = BY + 66 + 22
cc.conclusion(MX, BXR, CY,
              '三代之争全在 T_draft 的形状上：串行买得到依赖但付 γ 拍、并行省时间但丢依赖——DSpark 用 0.2%~1.3% 的串行小拍把两头都拿到。')
cc.footer(MX, BXR, CY + 22, [
    '自回归循环形态 vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py:L386-L391 · 方法表 vllm/config/speculative.py:L67-L77（eagle/mtp/dflash/dspark 高亮为本章谱系主角）',
    '层深/块长对齐（EAGLE3 1 层 vs DFlash/DSpark 5 层、γ=16 例）引 arXiv:2607.05147 §4.1/§2.2 · 序列循环开销 0.2%~1.3% 引 §4.3.2 · 行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-drafter-lineage', W, H)
