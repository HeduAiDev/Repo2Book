#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch33 M0·步3 机制图 ch33-fig-verify-layout（模板 layout）

claim：一次前向验证=展平 logits 里的交错排座：cu=[4,104,107,207,209]/
num_draft=[3,0,2,0,1] 时 11 个采样位=[0..3,103..106,206..208]，每请求段=
（num_draft 个 draft 验证位+1 个 bonus 位）——bonus=[3,4,7,8,10] 恰是各段末位，
draft_token_ids=[1,2,3,105,106,208] 从输入序列按 target+1 反取。
数字=spec.numbers（docstring 原装例的 numpy 复算逐值断言）。
"""
import _common33 as cc
import l0_common as lc

W = 1160
lc.reset()
MX, BXR = cc.header(
    W,
    '「一次前向验证所有位」= 展平 logits 里的交错排座',
    'cu=[4,104,107,207,209] · num_draft=[3,0,2,0,1] → 11 个采样位：每请求一段，段内 draft 验证位在前、bonus 位恒在段末',
    '放大自 L0 采样出口列 · logits→RejectionSampler 采样位切片')

LOGITS_IDX = [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
DRAFT_POS = {0, 1, 2, 5, 6, 9}          # target_logits_indices（平坦采样位）
BONUS_POS = {3, 4, 7, 8, 10}
SEGS = [(0, 3, 'req0 · draft×3+bonus'), (4, 4, 'req1 · bonus'),
        (5, 7, 'req2 · draft×2+bonus'), (8, 8, 'req3 · bonus'),
        (9, 10, 'req4 · draft×1+bonus')]

CW, CG, BX0 = 70.0, 6.0, 66.0
BAND_Y, CH = 116, 62


def cx(i):
    return BX0 + i * (CW + CG)


# ---- 请求分组括号（带上方）----
for a, b, lab in SEGS:
    x0, x1 = cx(a), cx(b) + CW
    lc.seg(x0, 108, x1, 108, lc.C_MUTE, 1.3)
    lc.seg(x0, 108, x0, 103, lc.C_MUTE, 1.3)
    lc.seg(x1, 108, x1, 103, lc.C_MUTE, 1.3)
    lc.text((x0 + x1) / 2, 96, lab, 9, lc.C_TXT, 'middle', True, maxw=x1 - x0 + 12,
            tag='seg:%s' % lab[:4])
lc.text(BXR, 96, '图例：▲ draft 验证位　★ bonus 位', 9, lc.C_MUTE, 'end', tag='legend')

# ---- 11 格长带 ----
for i, v in enumerate(LOGITS_IDX):
    is_bonus = i in BONUS_POS
    st = '#2563eb' if is_bonus else lc.C_SAM_S
    fl = '#eff6ff' if is_bonus else lc.C_SAM_F
    lc.rect(cx(i), BAND_Y, CW, CH, fl, st, rx=5, sw=1.7)
    lc.text(cx(i) + CW / 2, BAND_Y + 17, ('★' if is_bonus else '▲'), 10, st, 'middle',
            tag='mk%d' % i)
    lc.text(cx(i) + CW / 2, BAND_Y + 34, '位 %d' % i, 8, lc.C_MUTE, 'middle',
            tag='pos%d' % i)
    lc.text(cx(i) + CW / 2, BAND_Y + 53, str(v), 13, lc.C_TXT, 'middle', True,
            tag='v%d' % i)
lc.text(BX0 - 10, BAND_Y + 30, 'logits', 9.5, lc.C_SAM_S, 'end', True, tag='yl')
lc.text(BX0 - 10, BAND_Y + 46, '_indices', 9.5, lc.C_SAM_S, 'end', True, tag='yl2')
lc.text(BX0 - 10, BAND_Y + 62, '取数位', 8, lc.C_MUTE, 'end', tag='yl3')

# ---- 两组索引括号（带下方：draft 位 / bonus 位）----
def bracket(cells, y, color, label):
    runs = []
    for i in cells:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    for a, b in runs:
        x0, x1 = cx(a), cx(b) + CW
        lc.seg(x0, y, x1, y, color, 1.6)
        lc.seg(x0, y, x0, y + 5, color, 1.6)
        lc.seg(x1, y, x1, y + 5, color, 1.6)
    lc.text(BXR, y + 4, label, 8.8, color, 'end', True, maxw=BXR - cx(10) - CW - 16,
            tag='br:' + label[:10])

bracket(sorted(DRAFT_POS), BAND_Y + CH + 18, lc.C_SAM_S,
        'target_logits_indices=[0,1,2,5,6,9]')
bracket(sorted(BONUS_POS), BAND_Y + CH + 44, '#2563eb', 'bonus=[3,4,7,8,10]（各段末位）')

# ---- 底部两枚：反取 / 逐请求读法 ----
BY = BAND_Y + CH + 76
BW = (BXR - MX - 20) / 2
lc.rect(MX, BY, BW, 84, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, BY + 20, '验证谁？——从输入序列按「下一位」反取', 10, lc.C_TXT,
        'start', True, maxw=BW - 28, tag='b1:t')
lc.text(MX + 14, BY + 40, 'draft 验证平坦位 [0,1,2,5,6,9] + 1 → [1,2,3,6,7,10]', 9.2,
        '#334155', 'start', maxw=BW - 28, tag='b1:l1')
lc.text(MX + 14, BY + 57, '→ 从输入序列读出 draft_token_ids=[1,2,3,105,106,208]', 9.2,
        lc.C_SAM_S, 'start', True, maxw=BW - 28, tag='b1:l2')
lc.text(MX + 14, BY + 73, '（例：位 0 的输入是 token 0，它要验的是「下一个」=token 1）', 8.3,
        lc.C_MUTE, 'start', maxw=BW - 28, tag='b1:l3')
B2X = MX + BW + 20
lc.rect(B2X, BY, BW, 84, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(B2X + 14, BY + 20, '逐请求读法（docstring 原装例）', 10, lc.C_TXT, 'start',
        True, maxw=BW - 28, tag='b2:t')
lc.text(B2X + 14, BY + 40, 'req0：draft 位 [0,1,2] + bonus 位 [3]；req1：只有 bonus 位 [103]', 9.2,
        '#334155', 'start', maxw=BW - 28, tag='b2:l1')
lc.text(B2X + 14, BY + 57, 'req2：draft 位 [104,105] + bonus 位 [106]', 9.2, '#334155',
        'start', maxw=BW - 28, tag='b2:l2')
lc.text(B2X + 14, BY + 73, '验证的开销不在前向次数——在每请求多付的算术位（预告 ch34：排批账）', 8.3,
        lc.C_GPU_S, 'start', maxw=BW - 28, tag='b2:l3')

# ---- 结论 + 页脚 ----
CY = BY + 84 + 22
cc.conclusion(MX, BXR, CY,
              '前向照常跑完，只在取 logits 时多切几个位：bonus 恒在段末、候选按「下一位」反取——「一次前向验证」的全部代价只是索引算术。')
cc.footer(MX, BXR, CY + 22, [
    '数值例 = vllm/v1/worker/gpu_model_runner.py:L2851-L2924 _calc_spec_decode_metadata docstring 原装数字（numpy 原样复算并逐值断言相等）',
    '行号基线 vLLM v0.27.1 · bonus 位采样自 target 分布（全收时白送的第 γ+1 个 token）'])

H = CY + 22 + 30
cc.write('ch33-fig-verify-layout', W, H)
