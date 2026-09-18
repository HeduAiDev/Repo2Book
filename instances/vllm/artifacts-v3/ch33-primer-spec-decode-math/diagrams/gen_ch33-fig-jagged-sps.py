#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M6·步3 机制图 ch33-fig-jagged-sps（模板 before-after）

claim：早停的最优性条件=Θ 单峰：平滑 SPS 下全枚举 16.12/14.63<17.4（早停无损）；
锯齿 SPS（悬崖后回稳）下全局 4.18*4.5=18.81>17.4——早停被坑 1.41，这是论文
§5.2 生产版去早停+异步因果屏障的动机；生产实证：验证预算随并发从 4-6 平滑收缩。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑 + 论文 §5.4）。
"""
import _common33 as cc
import l0_common as lc

W = 1160
lc.reset()
MX, BXR = cc.header(
    W,
    '早停的软肋：Θ 单峰才最优，真实硬件的 SPS(B) 是离散锯齿',
    '平滑假设下早停=全局最优；锯齿曲线（悬崖后回稳）下全局最优在 18.81，早停在 17.4 就刹车——白亏 8%',
    '放大自 L0 调度带 · SPS(B) 成本表（青）')

# ---------------- 双轨迹折线 ----------------
STEPS = ['init B=2', '收 req0', '收 req1', '全枚举 4.03', '全枚举 4.18']
SMOOTH = [16.0, 17.4, 16.944, 16.12, 14.63]
JAG = [16.0, 17.4, 16.944, 18.538, 18.81]
PX0, PX1 = MX + 120, MX + 540
PY0, PY1 = 116, 296      # y 映射：Θ 14..19.5
VMIN, VMAX = 14.0, 19.5
def gx(i):
    return PX0 + (PX1 - PX0) * i / (len(STEPS) - 1)
def gy(v):
    return PY1 - (v - VMIN) * (PY1 - PY0) / (VMAX - VMIN)

# 轴
lc.seg(PX0 - 8, PY1, PX1 + 10, PY1, lc.C_MUTE, 1.4)
lc.seg(PX0 - 8, PY1, PX0 - 8, PY0 - 6, lc.C_MUTE, 1.4)
for v in (14.0, 16.0, 18.0):
    lc.seg(PX0 - 8, gy(v), PX1 + 10, gy(v), '#e2e8f0', 1.0)
    lc.text(PX0 - 14, gy(v) + 3, '%g' % v, 8, lc.C_FAINT, 'end', tag='ax%d' % v)
for i, s in enumerate(STEPS):
    lc.text(gx(i), PY1 + 16, s, 8, lc.C_MUTE, 'middle', maxw=110, tag='sx%d' % i)
lc.text(PX0 - 60, (PY0 + PY1) / 2, 'Θ', 11, lc.C_MUTE, 'middle', True, tag='axy')

def plot(vals, col, dash):
    for i in range(len(vals) - 1):
        lc.seg(gx(i), gy(vals[i]), gx(i + 1), gy(vals[i + 1]), col, 2.0,
               None, dash=dash)
    for i, v in enumerate(vals):
        lc.ELEMS.append(((gx(i) - 4, gy(v) - 4, gx(i) + 4, gy(v) + 4),
                         '<circle cx="%.1f" cy="%.1f" r="3.6" fill="#ffffff" '
                         'stroke="%s" stroke-width="1.8"/>' % (gx(i), gy(v), col)))

plot(SMOOTH, lc.C_GPU_S, False)
plot(JAG, '#d97706', True)
# 图例（画布内左上）
lc.seg(PX0 + 6, 122, PX0 + 26, 122, lc.C_GPU_S, 2.0)
lc.text(PX0 + 32, 125, '平滑 SPS（假设）', 8.5, lc.C_GPU_S, 'start', True,
        maxw=130, tag='lg1')
lc.seg(PX0 + 6, 138, PX0 + 26, 138, '#d97706', 2.0, dash=True)
lc.text(PX0 + 32, 141, '锯齿 SPS（真实硬件）', 8.5, '#d97706', 'start', True,
        maxw=140, tag='lg2')
# 早停刹车旗（两曲线共用的 17.4 点）
lc.text(gx(1), gy(17.4) - 26, '早停刹车（Θ 一跌就 break）', 8.5, lc.C_ABORT,
        'middle', True, maxw=150, tag='flag')
lc.seg(gx(1), gy(17.4) - 18, gx(1), gy(17.4) - 4, lc.C_ABORT, 1.4, dash=True)
# 全局最优星（锯齿终点 18.81）
lc.text(gx(4) - 6, gy(18.81) - 14, '★ 全局最优 18.81', 9, '#d97706', 'end', True,
        maxw=150, tag='star')
lc.text(gx(4) - 6, gy(18.81) + 18, '早停 17.4，亏 1.41（8%）', 8.5, lc.C_ABORT,
        'end', maxw=160, tag='gap')
# 平滑侧注：全枚举不值
lc.text(gx(3), gy(16.12) + 22, '16.12', 8, lc.C_GPU_S, 'middle', tag='sv3')
lc.text(gx(4), gy(14.63) + 22, '14.63（更亏）', 8, lc.C_GPU_S, 'middle', maxw=110,
        tag='sv4')
lc.text(gx(2), gy(16.944) + 24, '16.944', 8, lc.C_GPU_S, 'middle', tag='sv2')

# ---------------- 右侧：动机与出路 ----------------
RX = PX1 + 30
RW = BXR - RX
lc.rect(RX, 92, RW, 240, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(RX + 14, 114, '生产版的两难与出路', 10, lc.C_TXT, 'start', True,
        maxw=RW - 28, tag='r:t')
for i, ln in enumerate([
        '理论版：吞吐曲线平滑单峰 →',
        '　早停=全局最优',
        '',
        '生产版：真实 SPS(B) 离散锯齿',
        '　→ 去掉早停，做无约束全局搜索',
        '',
        '保无损的代价：不泄露未来 token',
        '　→ 异步调度：用两步前的置信',
        '　　定截断长度——「看的是过去」',
        '　　代替「不看后面」']):
    if ln:
        lc.text(RX + 14, 138 + i * 19, ln, 8.8,
                (lc.C_TXT if i in (0, 3, 6) else '#334155'), 'start',
                i in (0, 3, 6), maxw=RW - 28, tag='r:l%d' % i)

# ---------------- 底部：负载-预算带 ----------------
BY = PY1 + 40
lc.rect(MX, BY, BXR - MX, 96, '#ffffff', lc.C_KV_S, rx=8, sw=1.2)
lc.text(MX + 14, BY + 18, '生产实证（Fig.8）：中等并发（Flash<200、Pro<150）时验证预算从静态 2 扩到 4-6 token/请求；并发爬升平滑收缩（低置信先剪）', 9.5,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 28, tag='b:t')
BX0, BX1 = MX + 60, MX + 560
# DSpark 预算带（4-6）
lc.rect(BX0, BY + 30, 350, 20, '#ecfeff', lc.C_KV_S, rx=4, sw=1.3)
lc.text(BX0 + 175, BY + 43, 'DSpark 验证预算 4-6 token/请求（中等并发）', 8,
        lc.C_KV_S, 'middle', True, maxw=340, tag='b:band')
# 收缩趋势（手绘箭头，无 marker）
lc.seg(BX0 + 352, BY + 40, BX0 + 466, BY + 55, lc.C_KV_S, 1.6)
lc.ELEMS.append(((BX0 + 460, BY + 50, BX0 + 476, BY + 66),
                 '<polygon points="%d,%d %d,%d %d,%d" fill="%s"/>' % (
                     BX0 + 476, BY + 58, BX0 + 462, BY + 52, BX0 + 470, BY + 68,
                     lc.C_KV_S)))
lc.text(BX0 + 372, BY + 78, '并发爬升平滑收缩（低置信先剪）', 7.8, lc.C_KV_S,
        'start', maxw=200, tag='b:shrink')
# MTP-1 静态 2 对照线
lc.seg(BX0, BY + 58, BX0 + 350, BY + 58, lc.C_ABORT, 1.4, dash=True)
lc.rect(BX0 + 350, BY + 47, 104, 22, '#fef2f2', lc.C_ABORT, rx=10, sw=1.2)
lc.text(BX0 + 402, BY + 61, 'MTP-1 静态 2（对照）', 7.5, lc.C_ABORT, 'middle',
        True, maxw=98, tag='b:mtp')
lc.text(MX + 620, BY + 38, 'vLLM v0.27.1 已落地的近亲：dynamic_sd_lookup', 9,
        lc.C_KV_S, 'start', True, maxw=440, tag='b:sd')
lc.text(MX + 620, BY + 56, '按批大小查表定 K（num_speculative_tokens_per_batch_size）', 8.5,
        lc.C_MUTE, 'start', maxw=440, tag='b:sd2')

# ---------------- 结论 + 页脚 ----------------
CY = BY + 96 + 22
cc.conclusion(MX, BXR, CY,
              '「早停=最优」不是免费的：它赌的是 Θ 单峰。真实硬件的锯齿让论文生产版宁可去掉早停，用「看过去」换无损。')
cc.footer(MX, BXR, CY + 22, [
    'Θ 轨迹/全枚举/锯齿对照=固定 seed 的 CPU 参考实现实跑 · 生产实证与去早停动机引 arXiv:2607.05147 §5.2/§5.4（Fig.8）',
    'dynamic_sd_lookup 落地于 vllm/v1/core/sched/scheduler.py（配置面 num_speculative_tokens_per_batch_size）· 行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-jagged-sps', W, H)
