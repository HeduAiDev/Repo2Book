#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 论文精髓图 ① · paper-fig-1（arXiv:2607.05147 Fig.1 忠实重绘）

writer figure-requests.json add：DSpark 架构与解码周期总览——prompt ABC 过 target
一步出 anchor D；DSpark（重并行骨干 + 轻序列 Markov 头 + 置信头）一次出草稿 EFGH
与逐位置信 c1–c4；硬件感知调度器留 EFG、剪低置信 H；target 一次前向验证收 E、F、
拒 G 并从残差补采 G*——一个周期发射 E、F、G* 三位。

原图真相源（arXiv HTML 2607.05147v1/x1.png 抓取亲读 + paper.md:L86 caption 逐字）：
- 五段横向周期动线：ABC → target 一步 → D(anchor) → DSpark 框 → 调度器 → target
  并行验证 → 发射 E F G*；
- 重活方块大（并行骨干，深网络一次前向）、轻活方块小（序列头 γ 次查表加法）——
  序列循环占整轮延迟 0.2%–1.3%（γ 4→16、batch 128，§4.3.2）；
- 置信头逐位输出 c1–c4（Eq.7 sigmoid 门），c4 低 → H 被调度器剪掉。

布局与信息结构对齐原图；配色/字体套本书视觉语言；文字译中。
provenance=原论文图本身（key_figures 豁免，不走 explainer figure_specs 通道）。
"""
import _common33 as cc
import l0_common as lc

W = 1500
lc.reset()
MX, BXR = cc.header(
    W,
    '论文原景：DSpark 架构与解码周期——重活并行干完、轻活串行补依赖、调度剪尾、验证收 E F 拒 G 补 G*',
    '重绘自 arXiv:2607.05147 Fig.1：prompt ABC 过 target 一步出 anchor D；DSpark 出草稿 EFGH 与逐位置信 c1–c4；调度器留 EFG 剪 H；target 一次前向收 E、F，拒 G 从残差补 G*——一个周期发射 E、F、G*',
    'primer · 论文精髓图重绘')

# ============ ① prompt ABC → target 一步 → D(anchor) ============
lc.text(70, 130, 'prompt', 8.5, lc.C_MUTE, 'start', tag='p:l')
for i, ch in enumerate('ABC'):
    x = 70 + i * 48
    lc.rect(x, 140, 38, 38, '#ffffff', lc.C_MUTE, rx=5, sw=1.4)
    lc.text(x + 19, 165, ch, 12, lc.C_TXT, 'middle', True, tag='tok' + ch)
lc.seg(122, 178, 122, 200, lc.C_GPU_S, 1.8, 'std')
lc.rect(46, 200, 254, 64, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.8)
lc.text(60, 224, 'target 模型 · 执行一步', 11, lc.C_GPU_S, 'start', True,
        maxw=230, tag='tg:t')
lc.text(60, 246, '大模型一次前向，产出下一个 token', 8.5, '#334155', 'start',
        maxw=230, tag='tg:s')
lc.seg(122, 264, 122, 300, lc.C_GPU_S, 1.8, 'std')
lc.rect(94, 300, 56, 48, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.8)
lc.text(122, 330, 'D', 13, lc.C_GPU_S, 'middle', True, tag='D')
lc.text(158, 318, 'anchor · 锚', 8.5, lc.C_GPU_S, 'start', True, maxw=160,
        tag='D:l1')
lc.text(158, 332, '（上轮 target 的末位产出）', 8, lc.C_MUTE, 'start', maxw=164,
        tag='D:l2')
lc.seg(150, 324, 330, 324, lc.C_GPU_S, 1.8, 'std')

# ============ ② DSpark drafter 框 ============
FRX, FRY, FRW, FRH = 330, 100, 500, 390
lc.rect(FRX, FRY, FRW, FRH, '#ffffff', lc.C_GPU_S, rx=10, sw=2.0)
lc.text(346, 124, 'DSpark drafter · 两段生成（本例 γ=4：E F G H）', 11.5,
        lc.C_GPU_S, 'start', True, maxw=460, tag='fr:t')
# —— 并行骨干（重活·大块）——
lc.rect(350, 140, 460, 120, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.6)
lc.text(364, 164, '并行骨干（重活）——深网络一次前向', 10.5, lc.C_GPU_S, 'start',
        True, maxw=300, tag='bb:t')
lc.text(364, 186, 'O(1) 次前向、与 γ 无关——省下的时延换成深网络', 8.5,
        '#334155', 'start', maxw=380, tag='bb:l1')
lc.text(364, 204, '整块一次算完基础 logits U1..Uγ', 8.5, '#334155', 'start',
        maxw=380, tag='bb:l2')
lc.rect(692, 146, 106, 22, '#ffffff', lc.C_GPU_S, rx=9, sw=1.2)
lc.text(745, 161, '1 次前向', 8.8, lc.C_GPU_S, 'middle', True, maxw=98,
        tag='bb:badge')
lc.text(796, 250, '重活 · 大块', 8.5, lc.C_GPU_S, 'end', True, maxw=90,
        tag='bb:note')
# 骨干 → 两个轻模块
lc.seg(460, 260, 460, 280, lc.C_GPU_S, 1.6, 'std')
lc.seg(700, 260, 700, 280, lc.C_GPU_S, 1.6, 'std')
# —— 序列 Markov 头（轻活·小块）——
lc.rect(350, 280, 220, 60, '#fdf2f8', lc.C_SAM_S, rx=7, sw=1.5)
lc.text(362, 302, '序列 Markov 头（轻活）', 10, lc.C_SAM_S, 'start', True,
        maxw=196, tag='mk:t')
lc.text(362, 322, 'γ 次查表加法 · 补块内依赖', 8.5, '#334155', 'start',
        maxw=196, tag='mk:s')
# —— 置信头 ——
lc.rect(590, 280, 220, 60, '#fdf2f8', lc.C_SAM_S, rx=7, sw=1.5)
lc.text(602, 302, '置信头 · sigmoid 门', 10, lc.C_SAM_S, 'start', True,
        maxw=196, tag='cf:t')
lc.text(602, 322, '逐位输出 c1–c4（Eq.7）', 8.5, '#334155', 'start', maxw=196,
        tag='cf:s')
lc.text(580, 352, '小块 = 轻活：序列循环仅占整轮延迟 0.2%–1.3%（γ 4→16 · batch 128）',
        8.5, lc.C_SAM_S, 'middle', True, maxw=460, tag='light')
# —— 草稿 token E F G H + 逐位置信 c1..c4 ——
for i, ch in enumerate('EFGH'):
    x = 432 + i * 84
    hot = ch == 'H'
    lc.rect(x, 384, 44, 44, '#f1f5f9' if hot else '#ffffff',
            '#94a3b8' if hot else lc.C_GPU_S, rx=5, sw=1.4, dash=hot)
    lc.text(x + 22, 412, ch, 12, '#94a3b8' if hot else lc.C_GPU_S, 'middle',
            True, tag='d' + ch)
CBASE, CMAX, CW_ = 466, 24, 16
for i, hfrac in enumerate([24, 20, 16, 6]):
    cx = 454 + i * 84
    lc.rect(cx - CW_ / 2, CBASE - hfrac, CW_, hfrac,
            '#f9a8d4' if i < 3 else '#f3f4f6',
            lc.C_SAM_S if i < 3 else '#94a3b8', rx=2, sw=1.1)
    lc.text(cx, 478, ('c%d' % (i + 1)) + (' · 低置信' if i == 3 else ''), 8,
            '#dc2626' if i == 3 else lc.C_MUTE, 'middle',
            i == 3, maxw=84, tag='c%d' % i)

# 草稿块 → 调度器（token 流）+ 置信 → 调度器（虚线）
lc.seg(830, 300, 860, 300, lc.C_MUTE, 1.8, 'std')
lc.parrow([(830, 420), (970, 420), (970, 330)], lc.C_SAM_S, 1.5, 'std',
          dash=True)
lc.text(900, 412, '逐位置信 c1–c4', 8, lc.C_SAM_S, 'start', maxw=110,
        tag='c2sch')

# ============ ③ 硬件感知前缀调度器 ============
lc.rect(860, 150, 220, 180, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.8)
lc.text(874, 174, '硬件感知前缀调度器', 10.5, lc.C_KV_S, 'start', True,
        maxw=196, tag='sc:t')
lc.text(874, 196, '逐位置信 × 引擎吞吐 → 定验证长度', 8.5, '#334155', 'start',
        maxw=196, tag='sc:s')
for i, ch in enumerate('EFG'):
    x = 880 + i * 36
    lc.rect(x, 210, 28, 28, '#ffffff', lc.C_KV_S, rx=4, sw=1.4)
    lc.text(x + 14, 229, ch, 9.5, lc.C_TXT, 'middle', True, tag='sc' + ch)
lc.rect(988, 210, 28, 28, '#f1f5f9', '#94a3b8', rx=4, sw=1.2, dash=True)
lc.text(1002, 229, 'H', 9.5, '#94a3b8', 'middle', True, tag='scH')
lc.seg(989, 211, 1015, 237, '#dc2626', 1.6)
lc.seg(1015, 211, 989, 237, '#dc2626', 1.6)
lc.text(970, 262, '留 E F G · 剪掉低置信 H', 9, lc.C_TXT, 'middle', True,
        maxw=196, tag='sc:cap')
lc.text(970, 284, '（H 不进验证，省 target 算力）', 8, lc.C_MUTE, 'middle',
        maxw=196, tag='sc:n1')
lc.text(970, 306, '验证长度随置信与负载动态定', 8, lc.C_MUTE, 'middle',
        maxw=196, tag='sc:n2')

# ============ ④ target 一次前向并行验证 ============
lc.seg(1080, 240, 1110, 240, lc.C_KV_S, 1.8, 'std')
lc.rect(1110, 150, 220, 180, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.8)
lc.text(1124, 174, 'target 一次前向并行验证', 10.5, lc.C_GPU_S, 'start', True,
        maxw=196, tag='vf:t')
lc.text(1124, 196, '同一批里同时验 E、F、G', 8.5, '#334155', 'start', maxw=196,
        tag='vf:s')
for i, (ch, ok) in enumerate([('E', True), ('F', True), ('G', False)]):
    x = 1130 + i * 36
    lc.rect(x, 210, 28, 28, '#ffffff', lc.C_GPU_S if ok else '#94a3b8', rx=4,
            sw=1.4, dash=not ok)
    lc.text(x + 14, 229, ch + ('✓' if ok else '✗'), 9.5,
            lc.C_GPU_S if ok else '#dc2626', 'middle', True, maxw=26,
            tag='vf' + ch)
    if not ok:
        lc.seg(x + 1, 211, x + 27, 237, '#dc2626', 1.6)
        lc.seg(x + 27, 211, x + 1, 237, '#dc2626', 1.6)
lc.rect(1240, 210, 28, 28, '#fdf2f8', lc.C_SAM_S, rx=4, sw=1.5)
lc.text(1254, 229, 'G*', 9.5, lc.C_SAM_S, 'middle', True, maxw=26, tag='vfG*')
lc.seg(1230, 224, 1240, 224, lc.C_SAM_S, 1.5, 'std')
lc.text(1124, 262, 'G 被拒 → 从残差分布补采 G*', 8.5, lc.C_TXT, 'start', True,
        maxw=196, tag='vf:n1')
lc.text(1124, 284, '（G* = 修正 token，本周期到此收口）', 8, lc.C_MUTE,
        'start', maxw=200, tag='vf:n2')

# ============ ⑤ 本周期发射 ============
lc.seg(1330, 240, 1360, 240, lc.C_GPU_S, 1.8, 'std')
lc.rect(1360, 150, 94, 180, lc.C_SAM_F, lc.C_SAM_S, rx=8, sw=1.8)
lc.text(1407, 172, '本周期发射', 9.5, lc.C_SAM_S, 'middle', True, maxw=86,
        tag='em:t')
for i, ch in enumerate(['E', 'F', 'G*']):
    lc.rect(1390, 190 + i * 36, 34, 28, '#ffffff', lc.C_SAM_S, rx=4, sw=1.4)
    lc.text(1407, 209 + i * 36, ch, 9.5, lc.C_SAM_S, 'middle', True, maxw=30,
            tag='em' + ch)
lc.text(1407, 314, '＝ 3 位', 9, lc.C_TXT, 'middle', True, maxw=60,
        tag='em:n')

# 回环：发射末位 G* → 下一轮 anchor（D 位）
lc.parrow([(1407, 330), (1407, 520), (122, 520), (122, 348)], lc.C_MUTE, 1.4,
          'std', dash=True)
lc.text(764, 510, '下一轮：末位 G* 作为新 anchor（回到 D 的位置）——周期再启',
        8.5, lc.C_MUTE, 'middle', maxw=400, tag='loop')

# ============ 结论 + 页脚 ============
CY = 566
cc.conclusion(MX, BXR, CY,
              '重活一次并行干完（1 次前向）、轻活串行只占整轮 0.2%–1.3%——大块长与剪尾验证两头一起省，这就是 DSpark 的解码周期。')
cc.footer(MX, BXR, CY + 22, [
    '周期示例（ABC→D→EFGH+c1–c4、留 EFG 剪 H、收 E F 拒 G 补 G*）= 论文 Fig.1 caption 逐字（§3）· c_k = σ(w·[h_k; W_1[x_(k-1)]]) 为置信头 Eq.7 · 序列循环 0.2%–1.3%（γ 4→16、batch 128）引论文 §4.3.2',
    '重绘自 arXiv:2607.05147 Fig.1 · 布局与信息结构对齐论文原图（arXiv HTML 原图抓取核对）· 配色/字体为本书视觉语言 · 数据 provenance=论文原图'])

H = 566 + 22 + 2 * 14 + 24
cc.write('paper-fig-1', W, H)
