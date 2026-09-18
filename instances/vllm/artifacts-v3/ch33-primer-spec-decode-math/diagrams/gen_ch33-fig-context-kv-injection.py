#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M4·步2 机制图 ch33-fig-context-kv-injection（模板 tensor-flow）

claim：『拼接』由 KV cache 寻址实现：target 多层 aux 隐状态拼接→W_c 投影+RMSNorm
（Eq.2）→预计算每层 K/V_ctx 直接写进 draft 自己的 KV cache 槽（Eq.3 的
K_i=[W_i^K H_ctx; W_i^K H_d] 不显式 cat）——查询块注意 cache 时天然拼上上下文。
数字全部取 spec.numbers（论文 Eq.2/3 + pin 源码锚点）。
"""
import _common33 as cc
import l0_common as lc

W = 1160
lc.reset()
MX, BXR = cc.header(
    W,
    '并行骨干的上下文从哪来：『拼接』由 KV cache 寻址实现，不显式 cat',
    'target 在 prefill 把若干层隐状态借给 draft（Eq.2 投影）；context KV 被预写进 draft 自己的槽——查询块做注意力时按槽位寻址天然看到它',
    '放大自 L0 GPU 执行臂 · target↔draft 隐状态馈线（绿）')

# ---------------- 上泳道：target ----------------
TY, TH = 90, 118
lc.rect(MX, TY, BXR - MX, TH, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.2)
lc.text(MX + 14, TY + 22, 'target 模型（prefill 侧）', 10.5, lc.C_GPU_S, 'start', True,
        maxw=300, tag='t:t')
# 多层隐状态小方块
hx0 = MX + 190
for i, lab in enumerate(['H^(l1)', 'H^(l2)', 'H^(l3)']):
    x = hx0 + i * 66
    lc.rect(x, TY + 34, 56, 40, '#ffffff', lc.C_GPU_S, rx=5, sw=1.4)
    lc.text(x + 28, TY + 58, lab, 8.5, lc.C_GPU_S, 'middle', True, maxw=52,
            tag='t:h%d' % i)
lc.text(hx0 + 3 * 66 + 8, TY + 58, '…（m 层 aux 隐状态）', 8, lc.C_MUTE, 'start',
        maxw=150, tag='t:hm')
# 拼接 + 投影框
PJX = hx0 + 3 * 66 + 170
lc.rect(PJX, TY + 34, 180, 40, '#ffffff', lc.C_GPU_S, rx=5, sw=1.6)
lc.text(PJX + 90, TY + 50, 'W_c 投影 + RMSNorm', 8.8, lc.C_GPU_S, 'middle', True,
        maxw=168, tag='t:pj')
lc.text(PJX + 90, TY + 66, 'H_ctx ∈ R^{d}', 8, lc.C_MUTE, 'middle', tag='t:pj2')
for i in range(3):
    x = hx0 + i * 66 + 56
    lc.seg(x, TY + 54, PJX, TY + 54, lc.C_GPU_S, 1.0, dash=True)
# Eq.2 注
lc.text(BXR - 14, TY + 22, 'Eq.(2)  H_ctx = RMSNorm(W_c[H^(l1); …; H^(l_m)])，W_c∈R^{d×m·d}', 8.5,
        lc.C_MUTE, 'end', maxw=430, tag='t:eq2')

# ---------------- 中间馈线 ----------------
FX = MX + 300
lc.seg(FX, TY + TH, FX, TY + TH + 30, lc.C_GPU_S, 2.2, 'std')
lc.text(FX + 10, TY + TH + 20, 'H_ctx（上下文特征）', 8.5, lc.C_GPU_S, 'start',
        tag='f:l')

# ---------------- 下泳道：draft ----------------
DY = TY + TH + 30
DH = 176
lc.rect(MX, DY, BXR - MX, DH, '#ffffff', lc.C_GPU_S, rx=8, sw=1.2)
lc.text(MX + 14, DY + 22, 'draft 侧（DSpark 并行骨干）', 10.5, lc.C_GPU_S, 'start',
        True, maxw=300, tag='d:t')
# context KV 预计算框
CKX, CKW = MX + 60, 300
lc.rect(CKX, DY + 34, CKW, 74, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.6)
lc.text(CKX + CKW / 2, DY + 50, 'context KV 预计算（每层）', 9, lc.C_GPU_S, 'middle',
        True, maxw=CKW - 12, tag='d:ck')
lc.text(CKX + CKW / 2, DY + 68, '融合 KV GEMM（全层一次）→ 分组 K-norm → 融合 RoPE', 7.8,
        '#334155', 'middle', maxw=CKW - 12, tag='d:ck2')
lc.text(CKX + CKW / 2, DY + 84, '→ 逐层写进 draft 自己的 KV cache 槽', 7.8,
        '#334155', 'middle', maxw=CKW - 12, tag='d:ck3')
# 写槽箭头
lc.seg(CKX + CKW, DY + 71, CKX + CKW + 56, DY + 71, lc.C_GPU_S, 1.8, 'std')
lc.text(CKX + CKW + 28, DY + 62, '预写', 8, lc.C_GPU_S, 'middle', tag='d:wr')
# KV cache 槽带
SB_X, SB_Y, SC_W, SC_H = CKX + CKW + 60, DY + 52, 56, 44
slots = [('ctx', True)] * 4 + [('qry', False)] * 3 + [('mask', False)] * 2
labels = {'ctx': '上下文\n预写区', 'qry': '查询块', 'mask': ''}
for i, (kind, _) in enumerate(slots):
    x = SB_X + i * (SC_W + 4)
    if kind == 'ctx':
        lc.rect(x, SB_Y, SC_W, SC_H, '#cffafe', lc.C_KV_S, rx=4, sw=1.6)
    elif kind == 'qry':
        lc.rect(x, SB_Y, SC_W, SC_H, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.6)
    else:
        lc.rect(x, SB_Y, SC_W, SC_H, '#f1f5f9', lc.C_MUTE, rx=4, sw=1.0)
lc.text(SB_X + 2 * (SC_W + 4) + SC_W / 2, SB_Y - 8, 'context 预写区', 8, lc.C_KV_S,
        'middle', True, maxw=200, tag='d:cz')
lc.text(SB_X + 4.5 * (SC_W + 4) + SC_W / 2, SB_Y - 8, '查询块区（anchor+mask）', 8,
        lc.C_GPU_S, 'middle', True, maxw=210, tag='d:qz')
# 注意力框从槽带取数
lc.rect(SB_X, SB_Y + SC_H + 14, 9 * SC_W + 8 * 4, 30, '#ffffff', lc.C_GPU_S, rx=5,
        sw=1.4)
lc.text(SB_X + (9 * SC_W + 32) / 2, SB_Y + SC_H + 33, 'draft KV cache 注意力：对整条槽带寻址——序列维拼接 = KV cache 寻址（零显式 cat）', 8.8,
        lc.C_GPU_S, 'middle', True, maxw=9 * SC_W, tag='d:attn')
for i in (2, 6):
    x = SB_X + i * (SC_W + 4) + SC_W / 2
    lc.seg(x, SB_Y + SC_H, x, SB_Y + SC_H + 14, lc.C_GPU_S, 1.2, dash=True)
# Eq.3 注
lc.text(BXR - 14, DY + DH - 12, 'Eq.(3)  K_i = [W_i^K H_ctx ; W_i^K H_d]，V_i 同构；块内双向注意力', 8.5,
        lc.C_MUTE, 'end', maxw=430, tag='d:eq3')

# ---------------- 底部：落地 ----------------
BY = DY + DH + 16
lc.rect(MX, BY, BXR - MX, 62, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 20, '落地：precompute_and_store_context_kv——「context 形状每步不同、不进 CUDA graph，专门用融合 kernel 压 op 数」（vllm/model_executor/models/qwen3_dflash.py:L548-L619）', 9.5,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='b:t')
lc.text(MX + 14, BY + 42, 'DSV4 生产版同一公式换名字：combine_hidden_states = main_norm(main_proj(target 多层 aux hidden 拼接))，W_c 即 main_proj（vllm/models/deepseek_v4/nvidia/dspark.py:L143-L148）', 9,
        '#334155', 'start', maxw=BXR - MX - 28, tag='b:l')

# ---------------- 结论 + 页脚 ----------------
CY = BY + 62 + 22
cc.conclusion(MX, BXR, CY,
              '论文公式里的『沿序列维拼接』在 vLLM 里就是槽位寻址：context KV 先写进 draft 的槽，注意力自然看到——省掉一次显式 cat 与一次拷贝。')
cc.footer(MX, BXR, CY + 22, [
    'Eq.(2)(3) 引 arXiv:2607.05147 §2.2 · 块内双向注意力与 KV 注入口径同源',
    '行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-context-kv-injection', W, H)
