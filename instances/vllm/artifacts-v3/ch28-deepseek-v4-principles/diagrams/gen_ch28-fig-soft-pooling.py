#!/usr/bin/env python3
"""ch28 机制图 · CSA 软池化压缩：2m 个元素一次 softmax 加权求和成 1 条（ch28-fig-soft-pooling）

claim：软池化＝2m 个位置一次 softmax 加权求和成 1 条：权重来自可学习投影 + 可学习
位置偏置（不是摘要生成），列方向和恒为 1，i=0 时 b 半区权重恒 0。

半区顺序（本轮定点修，figure-integration 评审阻断项）：图面左右 = 槽位顺序 = **论文拼接序
[Z^a; Z^b]**——槽 0..3 是当前窗 a（图左、蓝）、槽 4..7 是前一个窗 b（图右、灰），与
trace（run_m03_m04_csa.json B 段 slot 0..3 [a 半区(当前窗)] / slot 4..7 [b 半区(前一个窗)]）、
正文表「b 半区权重全 0（槽位 4 到 7）」与图内左下卡「slot 4..7」三方一致。官方参考实现把
前一个窗放在前半区（两半区相反、调头后逐位相同）——图右上角注记点破，正文另有专门对账。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  玩具 m=4、c=2：8 个 (Z+B) 值 [2.0, 1.0, 0.0, -1.0, -2.0, 0.0, 0.0, 0.0]
  → 权重 [0.505734, 0.186049, 0.068444, 0.025179, 0.009263, 0.068444, 0.068444, 0.068444]（和 1.000000）
  加权和 4.335023 vs 直接平均 8.000000（差 3.664977）；8 个 C 值 = [1, 3, 5, 7, 9, 11, 13, 15]
  完整链：C^Comp 形状 (3, 2)；i=0 [0.576481, 0.469171]｜i=1 [0.548559, 0.297268]｜i=2 [0.779006, 0.400221]
  i=0 的 b 半区权重全 0（exp(−inf)=0）；a 半区权重逐列和 [1.0, 1.0]
  对照 i=1 的 b 半区权重非 0：[0.054212, 0.098538] … [0.034567, 0.281588]
  config 口径放大：m=4、c=512 → 1M 个 token → 250000 条

配色走 book/cartography/l0_common.py 的角色常量（KV 青 = 内容/条目，橙 = 分数/权重）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 860
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_CUR_F, C_CUR_S = '#e0f2fe', '#0284c7'          # 当前窗（a 半区）
C_PRE_F, C_PRE_S = '#f1f5f9', '#94a3b8'          # 前一个窗（b 半区）
C_SCOR_F, C_SCOR_S = '#ffedd5', '#c2410c'        # 打分 Z / 权重 S

EXTRA_DEFS = ('<defs>'
              '<marker id="wm" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_SCOR_S}"/></marker>'
              '<marker id="cy" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV_DEEP}"/></marker>'
              '<marker id="tk" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="5.5" '
              f'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
              '</defs>')

# ---------------- 数据（全部来自实测输出） ----------------
Z = ['2.0', '1.0', '0.0', '-1.0', '-2.0', '0.0', '0.0', '0.0']
C = ['1.0', '3.0', '5.0', '7.0', '9.0', '11.0', '13.0', '15.0']
S = ['0.505734', '0.186049', '0.068444', '0.025179', '0.009263', '0.068444', '0.068444', '0.068444']
SV = [float(x) for x in S]

# ---------------- 布局常量 ----------------
CELLW, GAP = 100, 5
BX0 = 250
BW = CELLW * 8
ZB_Y, ZB_H = 142, 36          # 打分带（Z+B）
CT_Y, CT_H = 216, 36          # 内容带（C）
SM_Y0, SM_Y1 = 288, 356       # softmax 盒
WT_Y, WT_H = 392, 34          # 权重带（S）
EN_Y, EN_H = 458, 56          # 压缩条目条


def cx(i):
    return BX0 + i * CELLW


def ccx(i):
    return BX0 + (i + 0.5) * CELLW


# ---------------- 标题区 ----------------
lc.text(MX, 32, '软池化：8 个位置一次 softmax 加权求和成 1 条', 17, C_TXT, 'start', True,
        maxw=620, tag='title')
lc.text(MX, 56, 'CNN 的最大池化是『2×2 里挑一个』；这里是『按学出来的份额加权平均』——'
                '2m = 8 个位置按份额合成 1 条 c 维压缩条目', 10, C_MUTE, 'start', maxw=900, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』内压缩段的第一台机器', 9, C_MUTE,
        'end', maxw=440, tag='l0:a')
lc.text(BXR, 64, '——重叠窗、索引器、共享 KV 都是它下游或侧旁的放大', 9, C_MUTE, 'end',
        maxw=440, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_CUR_F, C_CUR_S, '当前窗（a 半区，本条目新吃进的 4 个）'),
                  (C_PRE_F, C_PRE_S, '前一个窗（b 半区，与上一条共享）'),
                  (C_SCOR_F, C_SCOR_S, '打分 Z 与权重 S（学出来的份额）'),
                  (C_KV_F, C_KV_S, '内容 C 与压缩条目（KV 青）')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=250, tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 16

# ---------------- 半区顺序注（图右上角空位：槽位序 = 论文拼接序） ----------------
NX0, NY0 = 1055, 92
lc.text(NX0, NY0, '半区顺序 = 论文拼接序 [Z^a; Z^b]', 9.2, C_TXT, 'start', True, maxw=400,
        tag='ord:t')
lc.text(NX0, NY0 + 16, '槽 0..3 = 当前窗 a（图左）｜ 槽 4..7 = 前一个窗 b（图右）', 8.8, '#334155',
        'start', maxw=400, tag='ord:1')
lc.text(NX0, NY0 + 31, '按槽位排，不是按 token 先后；', 8.4, C_MUTE, 'start', maxw=400,
        tag='ord:2')
lc.text(NX0, NY0 + 45, '官方参考实现两半区相反、调头后逐位相同。', 8.4, C_MUTE, 'start', maxw=400,
        tag='ord:3')

# ---------------- ① 两个半区（8 个位置；槽位序 = 论文拼接序 [Z^a; Z^b]） ----------------
# 左四格 = 槽 0..3 = 当前窗 a（蓝）、右四格 = 槽 4..7 = 前一个窗 b（灰）——与 trace、
# 正文表「槽位 4 到 7」、左下卡一致（原先左右标反，与三方冲突，本轮修正）。
for half, lab, sub, tcol in ((0, '当前窗 a（m=4）', '槽 0..3 · C^a / Z^a', C_CUR_S),
                             (1, '前一个窗 b（m=4）', '槽 4..7 · C^b / Z^b', '#334155')):
    c0 = half * 4
    lc.text(BX0 + (c0 + 2) * CELLW, ZB_Y - 12, lab, 10, tcol,
            'middle', True, maxw=4 * CELLW - 8, tag='tk:h%d' % half)
    lc.text(BX0 + (c0 + 2) * CELLW, ZB_Y - 28, sub, 8.5, C_MUTE, 'middle', maxw=4 * CELLW - 8,
            tag='tk:s%d' % half)

# 打分带
lc.text(BX0 - 14, ZB_Y + 16, '打分', 10.5, C_TXT, 'end', True, tag='zb:t')
lc.text(BX0 - 14, ZB_Y + 31, 'Z + B', 8.5, C_SCOR_S, 'end', tag='zb:s')
for i in range(8):
    f, s = (C_CUR_F, C_CUR_S) if i < 4 else (C_PRE_F, C_PRE_S)
    lc.rect(cx(i) + GAP / 2, ZB_Y, CELLW - GAP, ZB_H, f, s, rx=3, sw=1.2)
    lc.text(ccx(i), ZB_Y + 24, Z[i], 11.5, C_SCOR_S, 'middle', True, tag='zb:%d' % i)

# 内容带
lc.text(BX0 - 14, CT_Y + 16, '内容', 10.5, C_TXT, 'end', True, tag='ct:t')
lc.text(BX0 - 14, CT_Y + 31, 'C', 8.5, C_KV_S, 'end', tag='ct:s')
for i in range(8):
    lc.rect(cx(i) + GAP / 2, CT_Y, CELLW - GAP, CT_H, C_KV_F, C_KV_S, rx=3, sw=1.2)
    lc.text(ccx(i), CT_Y + 24, C[i], 11.5, C_KV_S, 'middle', True, tag='ct:%d' % i)
lc.text(BX0, CT_Y + CT_H + 22, '打分与内容都由同一串隐状态经两组可学习投影得到（Eq.9、Eq.10）',
        8.8, C_MUTE, 'start', maxw=BW, tag='ct:n')

# ---------------- ② softmax：跨 2m 归一 ----------------
lc.parrow([(BX0 + BW / 2, ZB_Y + ZB_H), (BX0 + BW / 2, SM_Y0)], C_SCOR_S, 2.2, marker='wm')
lc.rect(BX0, SM_Y0, BW, SM_Y1 - SM_Y0, '#ffffff', C_SCOR_S, rx=8, sw=1.8)
lc.text(BX0 + 18, SM_Y0 + 22, 'Softmax_row', 12.5, C_SCOR_S, 'start', True, tag='sm:t')
lc.text(BX0 + 150, SM_Y0 + 22, '跨 2m = 8 个元素一起归一（当前窗 4 + 前一个窗 4）', 10, C_TXT,
        'start', maxw=520, tag='sm:l0')
lc.text(BX0 + 18, SM_Y0 + 44, '权重和 = 1.000000', 11.5, C_SCOR_S, 'start', True, tag='sm:l1')
lc.text(BX0 + 178, SM_Y0 + 44, '（这是 Eq.11-12 的『凸组合』：产出落在 8 个输入的凸包内，不会放大）',
        9, C_MUTE, 'start', maxw=500, tag='sm:l2')

# ---------------- ③ 权重带（条形长度 ∝ 份额） ----------------
lc.text(BX0 - 14, WT_Y + 16, '权重', 10.5, C_TXT, 'end', True, tag='wt:t')
lc.text(BX0 - 14, WT_Y + 31, 'S', 8.5, C_SCOR_S, 'end', tag='wt:s')
for i in range(8):
    lc.rect(cx(i) + GAP / 2, WT_Y, CELLW - GAP, WT_H, '#ffffff', '#e2e8f0', rx=3, sw=1.0)
    wpx = (SV[i] / max(SV)) * (CELLW - GAP - 8)
    lc.rect(cx(i) + GAP / 2 + 4, WT_Y + 4, max(3.0, wpx), WT_H - 18, C_SCOR_S, C_SCOR_S, rx=2, sw=1)
    lc.text(ccx(i), WT_Y + 30, S[i], 8.8, '#7c2d12', 'middle', tag='wt:%d' % i)
    lc.seg(ccx(i), SM_Y1, ccx(i), WT_Y - 4, C_SCOR_S, 1.4, marker='wm')
lc.text(BX0 + BW + 16, WT_Y + 12, '最大份额 0.505734', 9.5, C_SCOR_S, 'start', True, tag='wt:m')
lc.text(BX0 + BW + 16, WT_Y + 28, '拿走 50.6% 的份额', 8.6, C_MUTE, 'start', tag='wt:m2')

# ---------------- ④ 加权求和 → 1 条条目 ----------------
lc.parrow([(BX0 + BW / 2, WT_Y + WT_H), (BX0 + BW / 2, EN_Y - 4)], C_KV_DEEP, 2.4, marker='cy')
lc.rect(BX0, EN_Y, BW, EN_H, C_KV_F, C_KV_S, rx=6, sw=1.7)
lc.text(BX0 + 18, EN_Y + 24, 'Σ_j S_j ⊙ C_j', 13.5, C_KV_DEEP, 'start', True, tag='en:t')
lc.text(BX0 + 18, EN_Y + 44, '8 份内容按 8 个权重逐元素加权求和 → 1 条 c 维压缩条目',
        9.5, '#334155', 'start', maxw=520, tag='en:s')
lc.text(BX0 + BW - 18, EN_Y + 24, '= 4.335023', 15, C_KV_DEEP, 'end', True, tag='en:v')
lc.text(BX0 + BW - 18, EN_Y + 44, '直接平均（均匀 1/8）= 8.000000，差 3.664977', 9.5, C_MUTE,
        'end', maxw=520, tag='en:c')
lc.text(BX0 + BW + 16, EN_Y + 24, '权重真的在挑人：', 9, '#334155', 'start', tag='en:n')
lc.text(BX0 + BW + 16, EN_Y + 40, '均匀权重会得到 8.0', 9, C_MUTE, 'start', tag='en:n2')

# ---------------- 底左：i=0 边界 ----------------
PY0, PY1 = 542, 740
lc.rect(MX, PY0, 700, PY1 - PY0, '#f8fafc', '#cbd5e1', rx=10, sw=1.3)
lc.text(MX + 16, PY0 + 24, 'i = 0 是唯一的特例：b 半区被 −inf / 0 填掉', 11.5, C_TXT, 'start',
        True, maxw=420, tag='p1:t')
lc.text(MX + 16, PY0 + 44, '第一个条目没有『前一个窗』可看：Z^b 填 −inf、C^b 填 0，'
                          'exp(−inf) = 0 → 那 4 个权重恰好为 0。', 8.8, C_MUTE, 'start',
        maxw=470, tag='p1:s')
LCARD = [('条目 i=0 的 b 半区权重（slot 4..7）', '= [0.0, 0.0] × 4 → 全 0', '#ffffff', C_KV_S),
         ('条目 i=1 的 b 半区权重（slot 4..7）', '= [0.054212, 0.098538] … [0.034567, 0.281588]',
          C_KV_F, C_KV_S)]
for i, (a, b, f, s) in enumerate(LCARD):
    ry = PY0 + 62 + i * 30
    lc.rect(MX + 16, ry, 640, 24, f, s, rx=4, sw=1.1)
    lc.text(MX + 26, ry + 16, a, 9.0, C_TXT, 'start', True, tag='p1:a%d' % i)
    lc.text(MX + 350, ry + 16, b, 9.0, C_KV_DEEP, 'start', maxw=290, tag='p1:b%d' % i)
lc.text(MX + 16, PY0 + 140, '条目 i=0 的 a 半区权重逐列和 = [1.0, 1.0] → 第一个窗只看自己；'
                           '归一自动落到 a 半区那一半。', 9.0, '#334155', 'start', maxw=640,
        tag='p1:l0')
lc.text(MX + 16, PY0 + 160, '归纳一步：i ≥ 1 时 b 半区取第 i−1 个窗、非退化，权重和仍然是 1。',
        9.0, '#334155', 'start', maxw=640, tag='p1:l1')
lc.text(MX + 16, PY1 - 12, '所以『第一个窗只看自己』是边界条件写出来的，不是巧合。', 8.8,
        lc.C_FAINT, 'start', maxw=640, tag='p1:l2')

# ---------------- 底右：完整链三条产出 ----------------
QX0, QX1 = 770, BXR
lc.rect(QX0, PY0, QX1 - QX0, PY1 - PY0, C_KV_F, C_KV_S, rx=10, sw=1.4)
lc.text(QX0 + 16, PY0 + 24, '完整链的三个产出（玩具 m=4、c=2、n=12）', 11.5, C_TXT, 'start', True,
        maxw=420, tag='p2:t')
lc.text(QX0 + 16, PY0 + 44, 'C^Comp 形状 (3, 2)：12 个 token 压成 3 条，每条 2 维'
                          '——条目数 = n // m，不是 n // 2m', 8.8, C_MUTE, 'start', maxw=580,
        tag='p2:s')
ROWS = [('条目 i=0', '[0.576481, 0.469171]', 'b 半区全 0 → 只看当前窗'),
        ('条目 i=1', '[0.548559, 0.297268]', '两个半区都看（权重逐列和 = [1.0, 1.0]）'),
        ('条目 i=2', '[0.779006, 0.400221]', '权重逐列和 = [1.0, 1.0]')]
for i, (a, b, c) in enumerate(ROWS):
    ty = PY0 + 72 + i * 26
    lc.text(QX0 + 16, ty, a, 9.5, C_TXT, 'start', True, tag='p2:a%d' % i)
    lc.text(QX0 + 100, ty, b, 9.5, C_KV_DEEP, 'start', True, tag='p2:b%d' % i)
    lc.text(QX0 + 300, ty, c, 8.8, C_MUTE, 'start', maxw=380, tag='p2:c%d' % i)
lc.text(QX0 + 16, PY1 - 52, 'config 口径放大：m=4、c=512 时 1M 个 token → 250000 条，'
                           '每条仍是『8 个元素的 softmax + 一次加权和』。', 9, '#155e75',
        'start', True, maxw=600, tag='p2:z0')
lc.text(QX0 + 16, PY1 - 32, '计算量随 n/m 线性；压缩核要 materialize 的权重表规模就是 (n/m, 2m, c)。',
        8.8, C_MUTE, 'start', maxw=600, tag='p2:z1')
lc.text(QX0 + 16, PY1 - 12, '没有生成器、没有语言模型：整条链只有两组投影、一次 softmax、一次加权和。',
        8.8, '#334155', 'start', maxw=600, tag='p2:z2')

# ---------------- 页脚 ----------------
lc.text(MX, 776, '数字口径：手账版取第 i=1 条条目的第 0 列（8 个分数直接摆出来，可手算）；'
                 '完整链取 m=4、c=2、n=12 的全过程输出。', 8.5, C_MUTE, 'start', maxw=BXR - MX,
        tag='ft:1')
lc.text(MX, 794, '依据 DeepSeek-V4 技术报告 §2.3.1 的 Eq.(9)-(12)（arXiv:2606.19348）；'
                 '全部数值由本章驱动脚本以 float64 实跑（权重和 1.000000 是跨 2m 归一的实证）。',
        8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:2')
lc.text(MX, 812, '注意：真实 GPU 路径是 fp8/bf16 混合精度——这里的『和恰好为 1』是 float64 实测，'
                 '其依据是式子的恒等变形而不是浮点巧合。', 8.5, C_MUTE, 'start', maxw=BXR - MX,
        tag='ft:3')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-soft-pooling.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
