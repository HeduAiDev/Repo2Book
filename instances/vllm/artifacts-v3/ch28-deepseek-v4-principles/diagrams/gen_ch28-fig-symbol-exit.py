#!/usr/bin/env python3
"""ch28 机制图 · 符号对照与出口：A/B/C ↔ H^pre/H^res/H^post，以及 hc_head 压回单流
（ch28-fig-symbol-exit）

claim：三份文档一套符号（V4 的 A/B/C ＝ mHC 的 H^pre/H^res/H^post ＝ 代码的 hc_* 一族），
而出口 hc_head 是 V4 自造件：先把展平残差拷给多 token 预测，再用 sigmoid 门控把 4 条流
压回 1 条。

numbers（逐字取自 explainer figure-spec）：
  符号对照三行：A_l = σ(Ã_l) ↔ H^pre = σ(H~^pre)；B_l ∈ Birkhoff ↔ H^res = Sinkhorn(H~^res)；
                C_l = 2σ(C~_l) ↔ H^post = 2σ(H~^post)
  config 口径：hc_mult = 4（流数）、hc_sinkhorn_iters = 20、hc_eps = 1e-06 与论文 t_max = 20 互证
  出口顺序：pre-hc_head 残差 → 多 token 预测缓冲 → hc_head（sigmoid 门控加权求和）
           → 最后归一化 + 算 logits

配色走 book/cartography/l0_common.py 的角色常量（GPU 绿 = 执行臂）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 800
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_SAM_S, C_SAM_F = lc.C_SAM_S, lc.C_SAM_F

EXTRA_DEFS = ('<defs>'
              '<marker id="gn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_GPU_S}"/></marker>'
              '<marker id="mg" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_SAM_S}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '一套符号三种写法，出口还有一件自造件', 17, C_TXT, 'start', True, maxw=680,
        tag='title')
lc.text(MX, 56, '同一件事在三份文档里有三个名字：V4 报告写 A/B/C、mHC 原论文写 '
                'H^pre/H^res/H^post、代码里是 hc_* 那一族；而出口的 hc_head 是 V4 自己加的',
        10, C_MUTE, 'start', maxw=1150, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』块的出口', 9, C_MUTE, 'end',
        maxw=430, tag='l0:a')
lc.text(BXR, 64, '——上游是多流残差的读写回路，下游是多 token 预测那一档', 9, C_MUTE, 'end',
        maxw=470, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_GPU_F, C_GPU_S, '主干残差（4 条流）'),
                  (C_SAM_F, C_SAM_S, '分叉给多 token 预测（草稿器）'),
                  (C_KV_F, C_KV_S, '压回单流 / 出口归一化')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=250, tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 16

# ================= 左：符号对照卡 =================
LX0, LY0, LX1, LY1 = MX, 118, 700, 470
lc.rect(LX0, LY0, LX1 - LX0, LY1 - LY0, '#ffffff', C_KV_S, rx=10, sw=1.5)
lc.text(LX0 + 16, LY0 + 26, '符号对照：同一件事的三种写法', 12, C_TXT, 'start', True, maxw=360,
        tag='sym:t')
COLS = [('V4 技术报告', LX0 + 20), ('mHC 原论文', LX0 + 240), ('实现侧', LX0 + 460)]
for lab, cx in COLS:
    lc.text(cx, LY0 + 52, lab, 9.5, C_MUTE, 'start', True, tag='sym:c%s' % lab[:2])
lc.seg(LX0 + 16, LY0 + 58, LX1 - 16, LY0 + 58, C_MUTE, 1.2)
ROWS = [('A_l = σ(Ã_l)', 'H^pre = σ(H~^pre)', '读入（哪条流被读）', 0),
        ('B_l ∈ Birkhoff', 'H^res = Sinkhorn(H~^res)', '流间混合（残差自己那一项）', 1),
        ('C_l = 2σ(C~_l)', 'H^post = 2σ(H~^post)', '写回（子层输出散回哪几条流）', 2)]
for a, b, c, i in ROWS:
    ty = LY0 + 86 + i * 66
    lc.rect(LX0 + 16, ty - 20, LX1 - LX0 - 32, 54, '#f8fafc', '#cbd5e1', rx=6, sw=1.1)
    lc.text(LX0 + 24, ty, a, 11, C_KV_DEEP, 'start', True, maxw=200, tag='sym:a%d' % i)
    lc.text(LX0 + 244, ty, b, 11, C_KV_DEEP, 'start', True, maxw=200, tag='sym:b%d' % i)
    lc.text(LX0 + 24, ty + 24, c, 8.8, C_MUTE, 'start', maxw=250, tag='sym:c%d' % i)
    lc.text(LX0 + 244, ty + 24, 'hc_scale（α，动态项）· hc_base（S，静态偏置）', 8.6, C_MUTE,
            'start', maxw=430, tag='sym:d%d' % i)
lc.text(LX0 + 16, LY1 - 64, '代码侧的另外两件：hc_mult = 4（流数）、hc_sinkhorn_iters = 20 与 '
                            'hc_eps = 1e-06（与论文 t_max = 20 互证）。', 9, C_MUTE, 'start',
        maxw=640, tag='sym:n0')
lc.text(LX0 + 16, LY1 - 44, '三张映射的角色相同、符号不同——读公式时看形状就能对上：'
                            'A 是 (2+n_hc)·n_hc 个 raw 数里的前 n_hc 个。', 9, C_MUTE, 'start',
        maxw=640, tag='sym:n1')
lc.text(LX0 + 16, LY1 - 20, '注意：实现侧的 A 还多一个 eps（σ(Ã) + eps），给门控一个严格正的下界。',
        8.8, '#155e75', 'start', True, maxw=640, tag='sym:n2')

# ================= 右：出口动线 =================
RX0, RY0, RX1, RY1 = 740, 118, BXR, 470
lc.rect(RX0, RY0, RX1 - RX0, RY1 - RY0, '#ffffff', C_GPU_S, rx=10, sw=1.5)
lc.text(RX0 + 16, RY0 + 26, '出口动线：先拷贝、再压回单流', 12, C_TXT, 'start', True, maxw=360,
        tag='ext:t')

# 4 条流
STX, STY = RX0 + 30, RY0 + 60
for i in range(4):
    lc.rect(STX, STY + i * 22, 150, 18, C_GPU_F, C_GPU_S, rx=3, sw=1.2)
    lc.text(STX + 75, STY + i * 22 + 13, '流 %d' % (i + 1), 8.6, C_TXT, 'middle', tag='ext:s%d' % i)
lc.text(STX, STY - 8, '展平前的残差（hc_mult = 4 条流）', 9, C_MUTE, 'start', maxw=250, tag='ext:st')

# ① 分叉给 MTP
MTX, MTY = RX0 + 250, RY0 + 56
lc.rect(MTX, MTY, 400, 60, C_SAM_F, C_SAM_S, rx=8, sw=1.5)
lc.text(MTX + 14, MTY + 24, '① 分叉：拷贝给多 token 预测那一档', 10.5, C_SAM_S, 'start', True,
        maxw=370, tag='ext:m0')
lc.text(MTX + 14, MTY + 44, '用的是 hc_head 之前的展平残差——多流信息在这里不丢', 8.8,
        '#334155', 'start', maxw=380, tag='ext:m1')
lc.parrow([(STX + 150, STY + 33), (MTX - 2, MTY + 22)], C_SAM_S, 2.2, marker='mg')

# ② hc_head
HHX, HHY = RX0 + 250, RY0 + 150
lc.rect(HHX, HHY, 400, 76, C_KV_F, C_KV_S, rx=8, sw=1.6)
lc.text(HHX + 14, HHY + 24, '② hc_head：sigmoid 门控加权求和', 11, C_KV_DEEP, 'start', True,
        maxw=370, tag='ext:h0')
lc.text(HHX + 14, HHY + 44, '把 4 条流压回 1 条：这一件是 V4 自造件，mHC 原论文里没有它。',
        8.8, '#334155', 'start', maxw=374, tag='ext:h1')
lc.text(HHX + 14, HHY + 62, '首层特判：入参还是 2D (T, H) 时先广播成 4 条流再进回路。', 8.6,
        C_MUTE, 'start', maxw=374, tag='ext:h2')
lc.parrow([(STX + 150, STY + 88), (STX + 190, STY + 88), (STX + 190, HHY + 38),
           (HHX - 2, HHY + 38)], C_GPU_S, 2.2, marker='gn')

# 出口
OUTY = RY0 + 268
lc.rect(HHX, OUTY, 400, 44, '#f8fafc', '#cbd5e1', rx=8, sw=1.3)
lc.text(HHX + 14, OUTY + 27, '最后归一化 + 算 logits（单流）', 10.5, C_TXT, 'start', True,
        maxw=370, tag='ext:o0')
lc.parrow([(HHX + 200, HHY + 76), (HHX + 200, OUTY - 2)], C_KV_DEEP, 2.2, marker='gn')
lc.text(RX0 + 30, RY1 - 34, '顺序别讲反：先把展平残差拷给多 token 预测（草稿器要用的是'
                            ' hc_head 之前的残差），再过 hc_head 压回单流。', 9, '#155e75',
        'start', True, maxw=660, tag='ext:n0')
lc.text(RX0 + 30, RY1 - 14, '推理期两条路：多 token 预测那一档可以整档丢掉，也可以留下当'
                            '投机解码的起草器。', 8.8, C_MUTE, 'start', maxw=660, tag='ext:n1')

# ================= 底：三处口径 =================
BY0 = 492
lc.rect(MX, BY0, BXR - MX, 168, '#f8fafc', '#cbd5e1', rx=10, sw=1.3)
lc.text(MX + 16, BY0 + 26, '读这一段要记住的三件事', 11.5, C_TXT, 'start', True, maxw=300,
        tag='bot:t')
BOT = [('① 出口顺序', 'pre-hc_head 残差 → 多 token 预测缓冲 → hc_head → 归一化 + logits',
        '多 token 预测那一档读的是 hc_head 之前的展平残差：4 条流的信息都还在'),
       ('② hc_head 的来历', 'V4 自造件（mHC 论文里没有这一件）',
        'mHC 只定义了多流的读写回路，怎么收尾回单流是 V4 自己加的'),
       ('③ 首层与其余层的差别', '首层入参是 2D (T, H) → 先广播成 4 条流',
        '其余层的入参本身就是残差（hc_mult 条流的展平态）')]
for i, (a, b, c) in enumerate(BOT):
    ty = BY0 + 58 + i * 36
    lc.text(MX + 16, ty, a, 9.8, C_TXT, 'start', True, tag='bot:a%d' % i)
    lc.text(MX + 190, ty, b, 9.8, C_KV_DEEP, 'start', True, maxw=520, tag='bot:b%d' % i)
    lc.text(MX + 760, ty, c, 8.8, C_MUTE, 'start', maxw=640, tag='bot:c%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 690, '引用出处：官方参考实现（DeepSeek-V4-Pro inference/model.py:L738-L766）的出口段；'
                 '符号对照见技术报告 §2.2 与 mHC 原论文 Eq.(3)(5)(8)。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:1')
lc.text(MX, 708, 'config 口径：hc_mult = 4、hc_sinkhorn_iters = 20、hc_eps = 1e-06。', 8.5,
        C_MUTE, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-symbol-exit.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
