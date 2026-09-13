#!/usr/bin/env python3
"""ch25 机制图 · 混批一刀两段:token 维切刀(figure ch25-fig-mixed-cut,模板 before-after)

放大自 L0『模型层 MLA 框』一拍前向的分流决策——L2 站 11(forward_impl 切刀);批形准备回指站 7-8。
上带=排好序的 293 token(3 decode + 2 prefill,按比例画+decode 放大镜);中缝切刀;
下带分叉两泳道(MQA 吸收 / MHA 上投影)汇回同一 output 的不相交切片。

claim:同一次前向在 token 维切一刀:前 3 个 token(Sq/Skv≈0.09-0.17 的 decode 段)走 MQA 吸收、
后 290 个(Sq/Skv≈0.75-0.82 的 prefill 段)走 MHA 上投影,两段写回同一 output 缓冲的不相交切片。

数字全部取自 figure spec 的 numbers(query_start_loc [0,1,2,3,143,293] · 切刀 3/290 与
切片 [0,3)/[3,293) · Sq/Skv 逐请求 0.091/0.167/0.111/0.824/0.750 · max diff 0.000000 ·
num_mqa_tokens=attn_metadata.num_decode_tokens L771-L772)。坐标常量/循环;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 880
MX = 52
BXR = 1448
C_MQA = lc.C_API_S          # 蓝 = MQA 吸收段(潜空间路线)
C_MQA_T = '#1e40af'
C_MHA = lc.C_GPU_S          # 绿 = MHA 上投影段(执行臂常规路线)
C_MHA_T = '#166534'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '混批一刀两段:切刀落在 token 维,不在请求维——两条数学等价的腿各走各的',
        16, lc.C_TXT, 'start', True, maxw=1080, tag='title')
lc.text(MX, 58, '批已被四区重排保证 decode 在前(L2 站 7-8),故 token 前缀恰好就是全部 decode token;切刀只读 metadata 计数',
        10.5, lc.C_MUTE, 'start', maxw=1040, tag='subtitle')
_ch = '放大自 L0『模型层 MLA 框』· L2 站 11 的机制展开(批形回指站 7-8)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 上带:293 token(比例宽) ----------------
TB_Y, TB_H = 128, 54
TB_X, TB_W = 120, 1310          # 293 token → 1310px,≈4.47px/token
reqs = [
    ('req0', 1, 11, '0.091', C_MQA, '#eff6ff'),
    ('req1', 1, 6, '0.167', C_MQA, '#eff6ff'),
    ('req2', 1, 9, '0.111', C_MQA, '#eff6ff'),
    ('req3', 140, 170, '0.824', C_MHA, '#f0fdf4'),
    ('req4', 150, 200, '0.750', C_MHA, '#f0fdf4'),
]
scale = TB_W / 293
x = TB_X
bounds = [TB_X]
for nm, q, s, skv, stk, fl in reqs:
    w_ = q * scale
    lc.rect(x, TB_Y, w_, TB_H, fl, stk, rx=2, sw=1.4)
    bounds.append(x + w_)
    if q > 100:
        lc.text(x + w_ / 2, TB_Y + 22, '%s:query %d / seq %d' % (nm, q, s), 8.5,
                C_MHA_T if stk == C_MHA else C_MQA_T, 'middle', True, maxw=w_ - 16, tag='tb:' + nm)
        lc.text(x + w_ / 2, TB_Y + 40, 'Sq/Skv = %s' % skv, 8, C_MHA_T if stk == C_MHA else C_MQA_T,
                'middle', tag='tb2:' + nm)
    x += w_
lc.text(TB_X - 8, TB_Y + 32, 'token 维', 9, lc.C_TXT, 'end', True, tag='tb:axis')
lc.text(TB_X + TB_W + 8, TB_Y + 32, '293 行', 8.5, lc.C_MUTE, 'start', tag='tb:tot')

# 切刀(在第 3 行后 = x = TB_X + 3*scale)
CUT_X = TB_X + 3 * scale
knife_y0, knife_y1 = TB_Y - 26, TB_Y + TB_H + 8
lc.seg(CUT_X, knife_y0, CUT_X, knife_y1, lc.C_ABORT, 3.0, marker=None)
lc.seg(CUT_X - 12, knife_y0 + 8, CUT_X, knife_y0, lc.C_ABORT, 2.0, marker=None)
lc.seg(CUT_X + 12, knife_y0 + 8, CUT_X, knife_y0, lc.C_ABORT, 2.0, marker=None)
lc.text(CUT_X + 14, 104, '切刀:num_mqa_tokens = num_decode_tokens = 3', 10, lc.C_ABORT, 'start', True,
        maxw=360, tag='kn:t')
lc.text(CUT_X + 14, 120, 'num_mha_tokens = 293 − 3 = 290(mla_attention.py:L771-L772)', 8, lc.C_MUTE,
        'start', maxw=380, tag='kn:s')

# decode 放大镜(左上)
ZB = (120, 208, 380, 96)
lc.circle(127, TB_Y + TB_H, 26, lc.C_MUTE, 1.6, dash=True)
lc.rect(*ZB, '#ffffff', C_MQA, rx=8, sw=1.5)
lc.text(ZB[0] + ZB[2] / 2, ZB[1] + 20, '放大:前 3 行 = 3 条 decode 请求', 9.5, C_MQA_T, 'middle', True,
        tag='zm:t')
zcells = [(ZB[0] + 30 + i * 110, ZB[1] + 34) for i in range(3)]
for i, (zx, zy) in enumerate(zcells):
    lc.rect(zx, zy, 92, 42, '#eff6ff', C_MQA, rx=4, sw=1.3)
    lc.text(zx + 46, zy + 18, 'req%d' % i, 8.5, C_MQA_T, 'middle', True, tag='zm:r%d' % i)
    lc.text(zx + 46, zy + 34, '1/%d · %s' % (reqs[i][2], reqs[i][3]), 7.5, C_MQA_T, 'middle', tag='zm:s%d' % i)
lc.text(ZB[0] + ZB[2] / 2, ZB[1] + ZB[3] - 9, 'query_start_loc = [0, 1, 2, 3, 143, 293]', 8, lc.C_MUTE,
        'middle', tag='zm:q')

# prefill 侧注(右上)
PN = (900, 208, 530, 96)
lc.rect(*PN, '#ffffff', C_MHA, rx=8, sw=1.5)
lc.text(PN[0] + PN[2] / 2, PN[1] + 20, '后 290 行 = 2 条 prefill chunk(带已算上下文)', 9.5, C_MHA_T,
        'middle', True, tag='pn:t')
lc.text(PN[0] + PN[2] / 2, PN[1] + 40, 'req3:140/170(已算 30) · req4:150/200(已算 50)', 8.5, C_MHA_T,
        'middle', tag='pn:l1')
lc.text(PN[0] + PN[2] / 2, PN[1] + 58, '同一拍内 Sq/Skv 从 0.091 到 0.824,跨两个数量级', 8.5,
        '#334155', 'middle', tag='pn:l2')
lc.text(PN[0] + PN[2] / 2, PN[1] + 78, '——这就是一刀切两段数学的动机', 8, lc.C_MUTE, 'middle', tag='pn:l3')

# ---------------- 中:分叉两泳道 ----------------
FY = 356
mq = (120, FY, 560, 130)
lc.rect(*mq, '#eff6ff', C_MQA, rx=9, sw=1.8)
lc.text(mq[0] + mq[2] / 2, FY + 22, 'MQA 吸收腿:q[:3]', 11, C_MQA_T, 'middle', True, tag='mq:t')
lc.text(mq[0] + mq[2] / 2, FY + 44, 'bmm 吸收(q_nope × W_UK_T) → 拼 rope (3,4,576)', 8.5, C_MQA_T,
        'middle', maxw=mq[2] - 20, tag='mq:l1')
lc.text(mq[0] + mq[2] / 2, FY + 62, '→ 单头 MQA kernel 直接读分页 cache', 8.5, C_MQA_T, 'middle',
        tag='mq:l2')
lc.text(mq[0] + mq[2] / 2, FY + 80, '→ _v_up_proj 乘 W_UV 回 V 维', 8.5, C_MQA_T, 'middle', tag='mq:l3')
lc.text(mq[0] + mq[2] / 2, FY + 104, '数据搬运友好:cache 每行 576 维不动', 8, lc.C_MUTE, 'middle', tag='mq:n')

mh = (820, FY, 530, 130)
lc.rect(*mh, '#f0fdf4', C_MHA, rx=9, sw=1.8)
lc.text(mh[0] + mh[2] / 2, FY + 22, 'MHA 上投影腿:q[3:]', 11, C_MHA_T, 'middle', True, tag='mh:t')
lc.text(mh[0] + mh[2] / 2, FY + 44, 'kv_b_proj 上投影潜向量 → 每头 K/V', 8.5, C_MHA_T, 'middle',
        maxw=mh[2] - 20, tag='mh:l1')
lc.text(mh[0] + mh[2] / 2, FY + 62, '→ k_pe 广播拼尾 → 标准 MHA kernel', 8.5, C_MHA_T, 'middle', tag='mh:l2')
lc.text(mh[0] + mh[2] / 2, FY + 80, '历史上下文分块算(64k workspace,下一张图)', 8.5, C_MHA_T,
        'middle', tag='mh:l3')
lc.text(mh[0] + mh[2] / 2, FY + 104, '算力友好:每份 K/V 被 Sq 个 query 摊薄', 8, lc.C_MUTE, 'middle', tag='mh:n')

# 放大框/PN → 两泳道箭头
lc.parrow([(ZB[0] + ZB[2] / 2 - 60, ZB[1] + ZB[3]), (ZB[0] + ZB[2] / 2 - 60, FY)], C_MQA, 2.0, 'std')
lc.parrow([(PN[0] + PN[2] / 2 - 60, PN[1] + PN[3]), (PN[0] + PN[2] / 2 - 60, FY)], C_MHA, 2.0, 'std')

# ---------------- 下:汇合 output 条 ----------------
OY = 560
OB_ = (120, OY, 1310, 44)
w_mqa = 3 * scale
lc.rect(OB_[0], OY, w_mqa, OB_[3], '#bfdbfe', C_MQA, rx=2, sw=1.4)
lc.rect(OB_[0] + w_mqa, OY, OB_[2] - w_mqa, OB_[3], '#bbf7d0', C_MHA, rx=2, sw=1.4)
lc.text(OB_[0] + OB_[2] / 2 - 200, OY - 10, '同一 output 缓冲的两段不相交切片', 9.5, lc.C_TXT, 'middle',
        True, maxw=320, tag='ob:t')
lc.text(720, OY + 27, 'output[3:293) ← MHA 段(290 行)', 8.5, C_MHA_T, 'middle', True, tag='ob:mh')
lc.parrow([(mq[0] + mq[2] / 2, FY + 130), (mq[0] + mq[2] / 2, 530), (127, 530), (127, OY)], C_MQA, 2.0, 'std')
lc.parrow([(mh[0] + mh[2] / 2, FY + 130), (mh[0] + mh[2] / 2, OY)], C_MHA, 2.0, 'std')
lc.text(430, 524, 'output[0:3) ← MQA 段(3 行)', 8.5, C_MQA_T, 'start', True, maxw=180, tag='ob:mq')

# ---------------- 底:等价性 + v0.27 形态注 ----------------
VB = (120, 640, 1310, 70)
lc.rect(*VB, '#f8fafc', lc.C_MUTE, rx=8, sw=1.3)
lc.text(VB[0] + VB[2] / 2, 662, '两腿合并输出与整批上投影 MHA 参照逐元素一致:max diff 0.000000——同一数学、两种算法、一次前向',
        10, lc.C_TXT, 'middle', True, maxw=1300, tag='v:t')
lc.text(VB[0] + VB[2] / 2, 684, 'v0.27.1 形态:两腿在同一次 forward 内完成(v0.21 是两条独立 forward)· 切刀处的断言哨兵:num_decode_tokens+num_prefill_tokens==num_tokens',
        8.5, lc.C_MUTE, 'middle', maxw=1300, tag='v:l1')

# ---------------- 页脚 ----------------
lc.text(MX, 760, '图例:蓝 = MQA 吸收段(token 前缀) · 绿 = MHA 上投影段(后缀) · 红 = 切刀 · 上带格宽按 token 数严格比例',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, 780, '计数/数值 = host 实测(vLLM v0.27.1 精简版实跑,float32;5 请求混批 = 3 decode + 2 prefill chunk) · 行号基线 v0.27.1(6e448d0ea)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 800, 'MINI 维度 N=4/P=16/R=64/L=512/V=16 · decode 三请求 seq 11/6/9', 8.5, lc.C_MUTE,
        'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch25-fig-mixed-cut.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
