#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M0·步1 机制图 ch33-fig-cycle-timeline（模板 flow）

放大自 L0 采样出口列 spec decode 块：一个投机周期的三段流水 + 延迟账。
claim：一个投机周期=三段：drafter 一次出 γ 个 → target 一次前向整批验证 → 拒绝采样
发射；L=(T_draft+T_verify)/τ，本例 9.0 ms 摊 3 个 token=3.0 ms/token，加速 2.666667x。
数字全部取 spec.numbers（固定 seed CPU 参考实现实跑 / 论文 §4.2）。
"""
import _common33 as cc
import l0_common as lc

W = 1080
lc.reset()
MX, BXR = cc.header(
    W,
    '一个投机周期 = 三段流水：draft 猜 γ 个 → target 一次前向整批验证 → 拒绝采样发射',
    '延迟账 L=(T_draft+T_verify)/τ：分子是每周期两笔固定开销，分母 τ 是这趟平均真拿到几个 token',
    '放大自 L0 采样出口列 · spec decode 块（品红）')

# ---------------- 时间带 ----------------
AX0, AX1 = 62.0, 672.0          # 0ms..9ms 映射
T_MS = 9.0
BAND_Y, BAND_H = 122, 62
def tx(t):
    return AX0 + (AX1 - AX0) * t / T_MS

# 段标签（带上方）
lc.text(tx(0.5), 104, 'draft 段 · 小模型前向', 10, lc.C_GPU_S, 'middle', True,
        maxw=150, tag='seg:d')
lc.text(tx(5), 104, 'verify 段 · target 大模型一次前向验证所有位（批量摊薄）', 10,
        lc.C_GPU_S, 'middle', True, maxw=400, tag='seg:v')
# 两段
lc.rect(tx(0), BAND_Y, tx(1) - tx(0), BAND_H, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.8)
lc.rect(tx(1), BAND_Y, tx(9) - tx(1), BAND_H, '#dcfce7', lc.C_GPU_S, rx=4, sw=1.8)
lc.text(tx(0.5), BAND_Y + 34, '猜 γ=3', 9.5, lc.C_GPU_S, 'middle', True, maxw=60,
        tag='draft:t')
lc.text(tx(5), BAND_Y + 26, 'T_draft = 1.0 ms', 10, lc.C_GPU_S, 'middle', True,
        maxw=130, tag='tdraft')
lc.text(tx(5), BAND_Y + 46, 'T_verify = 8.0 ms（一次前向同时验证整块）', 10,
        lc.C_GPU_S, 'middle', True, maxw=300, tag='tverify')
# 时间轴刻度
for t, lab in ((0, '0'), (1, '1'), (9, '9 ms')):
    lc.seg(tx(t), BAND_Y + BAND_H, tx(t), BAND_Y + BAND_H + 8, lc.C_MUTE, 1.0)
    lc.text(tx(t), BAND_Y + BAND_H + 20, lab, 8.5, lc.C_MUTE, 'middle', tag='tick%s' % t)

# γ 个候选 token 小方块（draft 段下方）
GY = BAND_Y + BAND_H + 34
for i in range(3):
    x = tx(0.22) + i * 26
    lc.rect(x, GY, 22, 22, '#ffffff', lc.C_GPU_S, rx=4, sw=1.5)
    lc.text(x + 11, GY + 15, 'x%d' % (i + 1), 9, lc.C_GPU_S, 'middle', True, tag='cand%d' % i)
lc.seg(tx(0.5), BAND_Y + BAND_H + 2, tx(0.5), GY - 4, lc.C_GPU_S, 1.2, dash=True)
lc.text(tx(0.5) + 86, GY + 15, 'drafter 一次出 γ=3 个候选', 8.5, lc.C_GPU_S, 'start',
        tag='cand:l')

# 拒绝采样发射（verify 段下方的品红出口）
EY = GY
lc.text(tx(5), GY - 12, '拒绝采样发射：只放行与 target 分布一致的最长前缀，并补一个恢复/bonus token',
        9.5, lc.C_SAM_S, 'middle', maxw=430, tag='emit:l')
emit = [('收', lc.C_GPU_S, '#f0fdf4'), ('收', lc.C_GPU_S, '#f0fdf4'),
        ('恢复', '#dc2626', '#fef2f2')]
ex = tx(5) - 118
for i, (s, st, fl) in enumerate(emit):
    lc.rect(ex + i * 52, EY, 46, 24, fl, st, rx=4, sw=1.6)
    lc.text(ex + i * 52 + 23, EY + 16, s, 9.5, st, 'middle', True, tag='emit%d' % i)
# bonus 只在全收时出现（虚线蓝）
lc.rect(ex + 3 * 52, EY, 46, 24, '#ffffff', '#2563eb', rx=4, sw=1.4, dash=True)
lc.text(ex + 3 * 52 + 23, EY + 16, 'bonus', 9.5, '#2563eb', 'middle', True, tag='bonus')
lc.text(ex + 3 * 52 + 23, EY + 36, '全收才有', 8, '#2563eb', 'middle', tag='bonus:l')
# τ 花括号线
BR_Y = EY + 52
lc.seg(ex, BR_Y, ex + 2 * 52 + 46, BR_Y, lc.C_SAM_S, 1.6)
lc.seg(ex, BR_Y, ex, BR_Y - 6, lc.C_SAM_S, 1.6)
lc.seg(ex + 2 * 52 + 46, BR_Y, ex + 2 * 52 + 46, BR_Y - 6, lc.C_SAM_S, 1.6)
lc.text(tx(5), BR_Y + 14, 'τ=3 个 token / 9.0 ms 周期 → L=3.0 ms/token', 10,
        lc.C_SAM_S, 'middle', True, maxw=330, tag='tau:l')

# ---------------- 右侧延迟账面板 ----------------
PX, PW = 742, BXR - 742
PY, PH = 92, 268
lc.rect(PX, PY, PW, PH, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(PX + PW / 2, PY + 22, '延迟账（本例）', 11.5, lc.C_TXT, 'middle', True, tag='acc:t')
lc.text(PX + PW / 2, PY + 44, 'L=(T_draft+T_verify)/τ=(1.0+8.0)/3', 10.5, lc.C_SAM_S,
        'middle', True, maxw=PW - 16, tag='acc:f')
BAR_X, BAR_W = PX + 22, PW - 44
for i, (name, v, note, col) in enumerate([
        ('基线（逐 token 生成）', 8.0, '8.0 ms/token', lc.C_MUTE),
        ('投机（本例）', 3.0, '3.0 ms/token', lc.C_SAM_S)]):
    y = PY + 62 + i * 52
    lc.text(BAR_X, y, name, 9.5, lc.C_TXT, 'start', tag='bar%dn' % i)
    lc.rect(BAR_X, y + 6, BAR_W, 18, '#f1f5f9', lc.C_MUTE, rx=3, sw=1.0)
    lc.rect(BAR_X, y + 6, BAR_W * v / 8.0, 18, ('#e2e8f0' if i == 0 else lc.C_SAM_F),
            (lc.C_MUTE if i == 0 else lc.C_SAM_S), rx=3, sw=1.2)
    lc.text(BAR_X + BAR_W * v / 8.0 + 8, y + 19, note, 9.5,
            (lc.C_MUTE if i == 0 else lc.C_SAM_S), 'start', True, tag='bar%dv' % i)
lc.text(PX + PW / 2, PY + 190, '加速 2.666667 倍', 15, lc.C_SAM_S, 'middle', True,
        tag='speedup')
lc.text(PX + PW / 2, PY + 214, '（9.0 ms 摊 3 个 token）', 9, lc.C_MUTE, 'middle',
        tag='speedup:s')

# ---------------- 底部：论文量级两枚 ----------------
BY = PY + PH + 20
BW = (BXR - MX - 20) / 2
lc.rect(MX, BY, BW, 78, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 20, '论文量级（α=0.92、γ=3）', 10, lc.C_TXT, 'start', True,
        maxw=BW - 28, tag='b1:t')
lc.text(MX + 14, BY + 40, '闭式 τ=3.545088 · MC 30000 周期 3.536267', 9.3, '#334155',
        'start', maxw=BW - 28, tag='b1:l1')
lc.text(MX + 14, BY + 58, '→ L=2.538724 ms/token · 加速 3.151189 倍', 9.3,
        lc.C_SAM_S, 'start', True, maxw=BW - 28, tag='b1:l2')
B2X = MX + BW + 20
lc.rect(B2X, BY, BW, 78, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(B2X + 14, BY + 20, '论文实测 τ（Qwen3-4B、γ=7，按域平均）', 10, lc.C_TXT,
        'start', True, maxw=BW - 28, tag='b2:t')
for i, (dom, v) in enumerate((('Math', 5.57), ('Code', 5.12), ('Chat', 3.49))):
    x = B2X + 14 + i * 118
    lc.text(x, BY + 42, dom, 9, lc.C_MUTE, 'start', tag='b2:%s' % dom)
    lc.text(x, BY + 62, str(v), 13, lc.C_SAM_S, 'start', True, tag='b2:%sv' % dom)

# ---------------- 结论 + 页脚 ----------------
CY = BY + 78 + 24
cc.conclusion(MX, BXR, CY,
              '周期是全章的时间单位：谱系在优化分子（T_draft 与 τ），置信调度在优化有效 T_verify——后面每一节都在改这条时间带的某一段。')
cc.footer(MX, BXR, CY + 22, [
    '延迟账 Eq.(1) 引自 arXiv:2607.05147 §2.1 · 域平均 τ 引论文 §4.2 Table 1 · 其余数值为固定 seed 的 CPU 参考实现实跑（延迟为示教单位，比例 1:8 才有意义）',
    'vLLM 侧周期装配见 vllm/v1/worker/gpu/spec_decode/ · 拒绝采样发射器 vllm/v1/sample/rejection_sampler.py · 行号基线 vLLM v0.27.1'])

H = CY + 22 + 30
cc.write('ch33-fig-cycle-timeline', W, H)
