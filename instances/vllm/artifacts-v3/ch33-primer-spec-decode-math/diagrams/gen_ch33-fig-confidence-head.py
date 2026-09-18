#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M5·步1 机制图 ch33-fig-confidence-head（模板 flow）

claim：置信度头=一个 sigmoid 门：z=w·[h_k; W_1[x_{k-1}]]=−0.27041（12 维内积：
8 骨干隐状态+4 Markov 嵌入）→ c=0.432807；它受训逼近的标签 c*=1−½‖p_d−p_t‖₁
与 §33.2 接受准则是同一条 TV 恒等式。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑 + 论文 Eq.7/8）。
"""
import _common33 as cc
import l0_common as lc

W = 1120
lc.reset()
MX, BXR = cc.header(
    W,
    '置信度头便宜得惊人：拼 12 维、做一次内积、过一个 sigmoid',
    'c_k=σ(w·[h_k; W_1[x_{k-1}]])——「这个候选在前缀全过的前提下还能活」的预报；监督信号不是人工标注，是接受准则自己的解析期望',
    '放大自 L0 GPU 执行臂 · 序列头旁的置信小头（绿）')

# ---------------- 顶部灰徽章：vLLM 未接线 ----------------
_bw = lc.tw('vLLM v0.27.1 未接线（skip）：confidence_head 权重到货、加载显式跳过', 8.5, True) + 16
lc.rect(BXR - _bw, 72, _bw, 20, '#f1f5f9', lc.C_MUTE, rx=9, sw=1.0)
lc.text(BXR - _bw / 2, 85.5, 'vLLM v0.27.1 未接线（skip）：confidence_head 权重到货、加载显式跳过', 8.5,
        lc.C_MUTE, 'middle', True, maxw=_bw - 6, tag='skip')

# ---------------- 主流水（左→右）----------------
FY = 130          # 流水 y 中心
N1X, N1W = MX, 190      # 两条输入
N2X, N2W = MX + 220, 150   # 拼接
N3X, N3W = MX + 404, 150   # 内积
N4X, N4W = MX + 588, 120   # σ 门
N5X, N5W = MX + 742, BXR - MX - 742  # 输出徽章

def node(x, w, h, title, sub, fill, stroke, tcol=None, badge=None):
    y = FY - h / 2
    lc.rect(x, y, w, h, fill, stroke, rx=8, sw=1.6)
    lc.text(x + w / 2, y + 22, title, 9.5, (tcol or stroke), 'middle', True,
            maxw=w - 12, tag='nd:' + title[:6])
    if sub:
        lc.text(x + w / 2, y + 42, sub, 8, lc.C_MUTE, 'middle', maxw=w - 10,
                tag='nd:s:' + title[:6])
    if badge:
        lc.text(x + w / 2, y + 62, badge, 8.5, stroke, 'middle', True, maxw=w - 10,
                tag='nd:b:' + title[:6])
    return y, h

# 输入两条（h_k 8 维 / Markov 嵌入 4 维）
for i, (t, s, seg_fill) in enumerate([
        ('h_k（骨干隐状态）', '8 维', lc.C_GPU_F),
        ('W_1[x_{k-1}]（Markov 嵌入）', '4 维', lc.C_GPU_F)]):
    y = FY - 20 + i * 64 - 20
    lc.rect(N1X, y, N1W, 52, seg_fill, lc.C_GPU_S, rx=6, sw=1.4)
    lc.text(N1X + N1W / 2, y + 21, t, 8.5, lc.C_GPU_S, 'middle', True,
            maxw=N1W - 10, tag='in%d' % i)
    lc.text(N1X + N1W / 2, y + 39, s, 8, lc.C_MUTE, 'middle', tag='ind%d' % i)
    lc.seg(N1X + N1W, y + 26, N2X, FY - 12 + i * 24, lc.C_GPU_S, 1.6, 'std')
# 拼接
node(N2X, N2W, 64, '拼接', '[h_k ; W_1[x_{k-1}]] 12 维', '#ffffff', lc.C_GPU_S)
lc.seg(N2X + N2W, FY, N3X, FY, lc.C_GPU_S, 1.8, 'std')
# 内积
node(N3X, N3W, 64, '内积 w·[…]', '标量 z', '#ffffff', lc.C_GPU_S)
lc.seg(N3X + N3W, FY, N4X, FY, lc.C_GPU_S, 1.8, 'std')
lc.text((N3X + N3W + N4X) / 2, FY - 10, 'z = −0.27041', 8.5, lc.C_GPU_S, 'middle',
        True, tag='z')
# σ 门
node(N4X, N4W, 64, 'σ 门', 'c = σ(z)', lc.C_GPU_F, lc.C_GPU_S)
lc.seg(N4X + N4W, FY, N5X, FY, lc.C_GPU_S, 1.8, 'std')
# 输出徽章
node(N5X, N5W, 64, 'c = 0.432807', '前缀全过时位 k 存活的预报', '#ffffff',
      lc.C_SAM_S, lc.C_SAM_S)

# ---------------- 下支：监督标签（虚线）----------------
SY = FY + 92
lc.rect(MX, SY, BXR - MX, 88, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, SY + 20, '监督标签（Eq.8）：c_k* = 1 − ½‖p_k^d − p_k^t‖₁ ——接受率的解析期望，不是人工标注', 10,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='sup:t')
lc.text(MX + 14, SY + 42, '训练它不需要跑昂贵的验证：算一遍分布距离就够了——置信度头学的就是「验证会有多松」的解析值', 9,
        '#334155', 'start', maxw=BXR - MX - 28, tag='sup:l1')
lc.text(MX + 14, SY + 64, '与 §33.2 接受准则是同一条 TV 恒等式（下一张图三方对账）', 9,
        lc.C_SAM_S, 'start', True, maxw=BXR - MX - 28, tag='sup:l2')
lc.seg(MX + 420, SY, MX + 420, FY + 32, lc.C_MUTE, 1.4, dash=True)

# ---------------- 结论 + 页脚 ----------------
CY = SY + 88 + 22
cc.conclusion(MX, BXR, CY,
              '验证规则、训练目标、调度信号在「分布差」这一件事上会师——而这只会花掉一次 12 维内积。')
cc.footer(MX, BXR, CY + 22, [
    'z/c 数值=固定 seed 的 CPU 参考实现实跑（d=8+r=4=12 维玩具头）· Eq.(7)(8) 引 arXiv:2607.05147 §3.2.1',
    'vLLM 边界引 vllm/model_executor/models/qwen3_dspark.py:L184-L188 · 行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-confidence-head', W, H)
