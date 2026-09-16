#!/usr/bin/env python3
"""ch28 机制图 · mHC 动态参数化：一次 GEMM 出 A/B/C 的 raw 参数（ch28-fig-mhc-param）

claim：A/B/C 三张映射由展平后的无权 RMSNorm 态一次 GEMM 动态生成，再各乘可学习门控 α、
加静态偏置 S，最后分别过 σ、2σ、Sinkhorn：一次矩阵乘出全部三张映射的 raw 参数。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  展平 x = [1.0, 2.0, 0.5, −1.0] → 均方 1.5625 → x^ = [0.8, 1.599999, 0.4, −0.8]
  mixes（8 个数）= [0.8, 1.599999, 0.4, −0.8, 2.399999, 1.999999, −0.4, 0.0]
  A = σ(Ã) + eps = [0.62246, 0.71095]（值域 (0,1)）；C = 2σ(C~) = [1.197375, 0.900332]（值域 (0,2)）
  B = Sinkhorn(B~) = [[0.549833, 0.450166], [0.450166, 0.549833]]（行列和 [0.999999, 0.999999]）
  α = scale [0.5, 0.5, 0.5]；S = base [0.1, 0.1, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0]
  更新一拍：A X_0 = [0.977936, 0.53397]；X_1 = [[1.945872, 1.288864], [1.605549, 0.831248]]
  Ã = [0.5, 0.9]；C~ = [0.4, −0.2]；B~ = [[1.2, 1.0], [−0.2, 0.0]]（参考实现打包顺序 (pre, post, res)）

配色走 book/cartography/l0_common.py 的角色常量（GPU 绿 = 执行臂，KV 青 = 被投影出来的矩阵）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 760
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_SCOR_F, C_SCOR_S = '#ffedd5', '#c2410c'

EXTRA_DEFS = ('<defs>'
              '<marker id="gy" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
              '<marker id="dn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV_DEEP}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '三张映射不是三张表：一次 GEMM 当场算出来', 17, C_TXT, 'start', True, maxw=680,
        tag='title')
lc.text(MX, 56, '把 4 条流摊平、做一次没有增益的 RMSNorm，再乘一个大矩阵：8 个数一次出齐，'
                '切成三段就是 A / B / C 的 raw 参数', 10, C_MUTE, 'start', maxw=1050, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』mHC 的参数生成段', 9, C_MUTE,
        'end', maxw=430, tag='l0:a')
lc.text(BXR, 64, '——它是残差流那张图左侧的输入侧放大', 9, C_MUTE, 'end', maxw=430, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_GPU_F, C_GPU_S, '残差流 / 展平态'), (C_SCOR_F, C_SCOR_S, '门控 α 与偏置 S'),
                  (C_KV_F, C_KV_S, '三张映射（过完 σ / 2σ / Sinkhorn）')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=250, tag='lg:%s' % lab[:3])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 16

# ---------------- ① 展平与 RMSNorm ----------------
X0, Y0 = MX, 140
lc.rect(X0, Y0, 300, 108, C_GPU_F, C_GPU_S, rx=8, sw=1.5)
lc.text(X0 + 16, Y0 + 24, 'X（T = 1，n_hc = 2，d = 2）', 11, C_TXT, 'start', True, maxw=270,
        tag='x:t')
lc.text(X0 + 16, Y0 + 44, '[[1.0, 2.0], [0.5, −1.0]]', 10.5, '#334155', 'start', maxw=270,
        tag='x:v')
lc.text(X0 + 16, Y0 + 66, '展平：x = [1.0, 2.0, 0.5, −1.0]', 9.2, C_KV_DEEP, 'start', maxw=270,
        tag='x:f')
lc.text(X0 + 16, Y0 + 88, '均方 = 1.5625（4 维展平态一起算）', 9.2, C_MUTE, 'start', maxw=270,
        tag='x:m')

lc.parrow([(X0 + 300, Y0 + 54), (X0 + 360, Y0 + 54)], '#64748b', 2.0, marker='gy')
RNX = X0 + 360
lc.rect(RNX, Y0, 300, 108, C_KV_F, C_KV_S, rx=8, sw=1.5)
lc.text(RNX + 16, Y0 + 24, '无权 RMSNorm（只做缩放）', 11, C_KV_DEEP, 'start', True, maxw=270,
        tag='rn:t')
lc.text(RNX + 16, Y0 + 46, 'x^ = x · rsqrt(均方 + eps)', 10, '#334155', 'start', maxw=270,
        tag='rn:v')
lc.text(RNX + 16, Y0 + 68, 'x^ = [0.8, 1.599999, 0.4, −0.8]', 10.5, C_KV_DEEP, 'start', True,
        maxw=270, tag='rn:o')
lc.text(RNX + 16, Y0 + 90, '缩放折进后面的 GEMM，不单独占一步', 8.8, C_MUTE, 'start', maxw=270,
        tag='rn:n')

# ---------------- ② 一次 GEMM ----------------
lc.parrow([(RNX + 300, Y0 + 54), (RNX + 360, Y0 + 54)], '#64748b', 2.0, marker='gy')
GMX = RNX + 360
lc.rect(GMX, Y0, 320, 108, '#ffffff', C_SCOR_S, rx=8, sw=1.7)
lc.text(GMX + 16, Y0 + 24, '一次 GEMM：mixes = x^ · fn^T', 11, '#7c2d12', 'start', True, maxw=290,
        tag='gm:t')
lc.text(GMX + 16, Y0 + 44, 'fn 形状 (8, 4)：mix = (2 + n_hc)·n_hc = 8', 9.2, '#334155', 'start',
        maxw=290, tag='gm:s')
lc.text(GMX + 16, Y0 + 66, '[0.8, 1.599999, 0.4, −0.8,', 10, C_SCOR_S, 'start', True, maxw=290,
        tag='gm:v0')
lc.text(GMX + 16, Y0 + 84, ' 2.399999, 1.999999, −0.4, 0.0]', 10, C_SCOR_S, 'start', True,
        maxw=290, tag='gm:v1')

# ---------------- ③ 三路切片 ----------------
SX = GMX + 360
lc.rect(SX, Y0, BXR - SX, 108, '#f8fafc', '#cbd5e1', rx=8, sw=1.3)
lc.text(SX + 16, Y0 + 24, '按 (pre, post, res) 切片', 11, C_TXT, 'start', True, maxw=200,
        tag='sl:t')
SL = [('Ã = [0.5, 0.9]', 'σ(Ã) + eps → A = [0.62246, 0.71095]'),
      ('C~ = [0.4, −0.2]', '2σ(C~) → C = [1.197375, 0.900332]'),
      ('B~ 展成 2×2', 'Sinkhorn → B = [[0.549833, …]]')]
for i, (a, b) in enumerate(SL):
    ty = Y0 + 44 + i * 22
    lc.text(SX + 16, ty, a, 9.0, C_KV_DEEP, 'start', True, maxw=110, tag='sl:a%d' % i)
    lc.text(SX + 130, ty, b, 9.0, '#334155', 'start', maxw=200, tag='sl:b%d' % i)

# ---------------- ④ 三张映射卡 ----------------
MY0 = 288
MAPS = [('A_l（读入）', 'A_l = σ(Ã_l) + eps', '[0.62246, 0.71095]', '值域 (0,1)', 0),
        ('B_l（流间混合）', 'B_l = Sinkhorn(B~_l)', '[[0.549833, 0.450166],', '双随机集合内', 1),
        ('C_l（写回）', 'C_l = 2σ(C~_l)', '[1.197375, 0.900332]', '值域 (0,2)', 2)]
MW = 460
for lab, form, val, note, i in MAPS:
    cxx = MX + i * (MW + 20)
    lc.rect(cxx, MY0, MW, 108, C_KV_F, C_KV_S, rx=8, sw=1.5)
    lc.text(cxx + 16, MY0 + 26, lab, 12, C_KV_DEEP, 'start', True, maxw=240, tag='mp:t%d' % i)
    lc.text(cxx + 200, MY0 + 26, form, 10.5, '#334155', 'start', maxw=250, tag='mp:f%d' % i)
    lc.text(cxx + 16, MY0 + 52, val, 11, C_KV_DEEP, 'start', True, maxw=280, tag='mp:v%d' % i)
    if i == 1:
        lc.text(cxx + 16, MY0 + 74, ' [0.450166, 0.549833]]', 11, C_KV_DEEP, 'start', True,
                maxw=280, tag='mp:v%d2' % i)
    lc.text(cxx + 16, MY0 + 96, note, 9, C_MUTE, 'start', maxw=280, tag='mp:n%d' % i)
SLICED_CX = SX + (BXR - SX) / 2
lc.parrow([(SLICED_CX, Y0 + 108), (SLICED_CX, MY0 - 2)], C_KV_S, 2.0, marker='dn')
lc.text(SLICED_CX + 12, Y0 + 140, '切成三段，分别过 σ / 2σ / Sinkhorn', 9, C_KV_DEEP, 'start',
        maxw=300, tag='mp:flow')

# ---------------- ⑤ α 与 S ----------------
AY0 = 412
lc.rect(MX, AY0, BXR - MX, 76, C_SCOR_F, C_SCOR_S, rx=10, sw=1.4)
lc.text(MX + 16, AY0 + 26, 'α 与 S 从哪来：可学习门控 + 静态偏置（论文说的『动态项 + 静态项』）',
        11.5, '#7c2d12', 'start', True, maxw=560, tag='as:t')
lc.text(MX + 16, AY0 + 50, 'α：scale[0..2] = [0.5, 0.5, 0.5]（初始化很小，决定输入相关的动态项占多少）',
        9.2, '#334155', 'start', maxw=680, tag='as:a')
lc.text(MX + 760, AY0 + 50, 'S：base = [0.1, 0.1, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0]（输入无关的静态偏置）',
        9.2, '#334155', 'start', maxw=640, tag='as:b')
lc.text(MX + 16, AY0 + 68, 'raw 参数 = α · (动态项) + S → 门控为 0 时映射退化成静态偏置。', 8.8,
        C_MUTE, 'start', maxw=900, tag='as:c')

# ---------------- ⑥ 记号顺序陷阱 + 更新一拍 ----------------
BY0 = 508
lc.rect(MX, BY0, 726 - MX, 156, '#fef2f2', lc.C_ABORT, rx=10, sw=1.4)
lc.text(MX + 16, BY0 + 26, '记号陷阱：论文列举顺序 ≠ 参考实现打包顺序', 11.5, lc.C_ABORT,
        'start', True, maxw=460, tag='trap:t')
TR = [('论文列举 (§2.2)', 'pre, res, post', 'Ã = [0.5, 0.9]、C~ = [1.2, 1.0]、B~ = [0.4, −0.2]'),
      ('参考实现打包', 'pre, post, res', 'Ã = [0.5, 0.9]、C~ = [0.4, −0.2]、B~ = [[1.2, 1.0], [−0.2, 0.0]]')]
for i, (a, b, c) in enumerate(TR):
    ty = BY0 + 58 + i * 40
    lc.text(MX + 16, ty, a, 9.5, C_TXT, 'start', True, tag='trap:a%d' % i)
    lc.text(MX + 170, ty, b, 10, '#7c2d12', 'start', True, tag='trap:b%d' % i)
    lc.text(MX + 16, ty + 18, c, 8.8, '#334155', 'start', maxw=640, tag='trap:c%d' % i)
lc.text(MX + 16, BY0 + 142, '同一批 8 个数、换个切法——按论文的行序去读代码切片会读错。', 8.8,
        C_MUTE, 'start', maxw=640, tag='trap:n')

lc.rect(774, BY0, BXR - 774, 156, C_KV_F, C_KV_S, rx=10, sw=1.4)
lc.text(774 + 16, BY0 + 26, '更新一拍：X_1 = B X_0 + C · F(A X_0)', 11.5, C_KV_DEEP, 'start',
        True, maxw=440, tag='up:t')
lc.text(774 + 16, BY0 + 50, '（F(·) 取恒等占位：mHC 混合不关心子层内部是什么）', 8.8, C_MUTE,
        'start', maxw=620, tag='up:s')
UP = [('A X_0（4 条流聚成子层输入）', '[0.977936, 0.53397]'),
      ('C · F(A X_0)（散回 2 条流）', '[[1.197375×0.977936, 0.900332×0.53397], …]'),
      ('X_1（B 让残差在流间混合）', '[[1.945872, 1.288864], [1.605549, 0.831248]]')]
for i, (a, b) in enumerate(UP):
    ty = BY0 + 78 + i * 26
    lc.text(774 + 16, ty, a, 9.2, C_TXT, 'start', True, tag='up:a%d' % i)
    lc.text(774 + 320, ty, b, 9.2, C_KV_DEEP, 'start', maxw=340, tag='up:b%d' % i)
lc.text(774 + 16, BY0 + 146, '四个数一次算齐：2 条流的 2 维输出同时给出。', 8.8, C_MUTE,
        'start', maxw=620, tag='up:n')

# ---------------- 页脚 ----------------
lc.text(MX, 700, '数字口径：玩具 hc_mult = 2、d = 2、T = 1（4 维展平态 → 8 个数）；'
                 'config 口径 hc_mult = 4 → 展平态 4d 维、mixes 数 = 24。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:1')
lc.text(MX, 718, '依据 DeepSeek-V4 技术报告 §2.2 的 Eq.(3)-(8)（arXiv:2606.19348）；'
                 '全部数值由本章驱动脚本实跑。', 8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-mhc-param.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
