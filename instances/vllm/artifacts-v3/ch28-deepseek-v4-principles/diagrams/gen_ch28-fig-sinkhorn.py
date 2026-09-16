#!/usr/bin/env python3
"""ch28 机制图 · Sinkhorn-Knopp 迭代与 t_max = 20（ch28-fig-sinkhorn）

claim：Sinkhorn-Knopp 是交替归一：列和当轮就归 1，行/列偏差逐轮下降（本轮实测约 1/3
的收缩率），t_max = 20 是精度与开销的折中而非精确投影（20 轮后仍有 1.00e-06 的地板残差）。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  [[4.0, 1.0], [1.0, 3.0]] 前 5 轮 row_dev：0.02756985 → 0.00839717 → 0.00255908 → 0.00078041 → 0.00023847
  （第 5 轮真值 0.00023847：驱动脚本原先把第 20 轮记录错打成「第 5 轮」，2026-09-16 按
   sinkhorn_trace 复算修正；折线 5 个点 = 真实第 1–5 轮，第 20 轮另用空心点单标）
  col_dev 全程停在 1.05e-06 / 1.02e-06 / 1e-06（eps 地板 1e-06）
  第 20 轮末：row_dev = 1.00e-06、col_dev = 1.00e-06
  条件数差的矩阵 20 轮 vs 200 轮残差同为 1.00e-06
  实现顺序差异：order=row-first / col-first 的 M^(20) 数值（0.92408478、0.07580203、0.07591422…）
  与 start=exp / softmax 同结果

配色走 book/cartography/l0_common.py 的角色常量（KV 青 = 被投影的矩阵）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 820
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_SCOR_F, C_SCOR_S = '#ffedd5', '#c2410c'

EXTRA_DEFS = ('<defs>'
              '<marker id="gy" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
              '<marker id="rk" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_SCOR_S}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, 'Sinkhorn-Knopp：交替归一 20 轮，是个折中不是精确投影', 17, C_TXT, 'start',
        True, maxw=700, tag='title')
lc.text(MX, 56, '行拉到和 1、列拉到和 1、再拉行……每轮都更接近双随机，但永远差一点点'
                '（分母每次都加 1e-06 的保护项）', 10, C_MUTE, 'start', maxw=1050, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』mHC 的投影段', 9, C_MUTE, 'end',
        maxw=430, tag='l0:a')
lc.text(BXR, 64, '——它是残差流那张图里『笼子』的入口放大', 9, C_MUTE, 'end', maxw=430,
        tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_SCOR_F, C_SCOR_S, '行归一 / 列归一（交替动作）'),
                  (C_KV_F, C_KV_S, '被投影的矩阵 B~ → B'),
                  ('#f1f5f9', '#cbd5e1', 'eps 地板 1e-06')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=250, tag='lg:%s' % lab[:3])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 16

# ================= 左：交替归一的机械动作 =================
LX0, LY0, LX1, LY1 = MX, 118, 620, 440
lc.rect(LX0, LY0, LX1 - LX0, LY1 - LY0, '#ffffff', C_KV_S, rx=10, sw=1.5)
lc.text(LX0 + 16, LY0 + 26, '交替归一：两个动作成环', 12, C_TXT, 'start', True, maxw=340,
        tag='lp:t')
RAW = (LX0 + 30, LY0 + 52, 260, 54)
lc.rect(*RAW, C_KV_F, C_KV_S, rx=6, sw=1.4)
lc.text(RAW[0] + 130, LY0 + 76, 'B~ = [[4.0, 1.0], [1.0, 3.0]]', 10.5, C_KV_DEEP, 'middle',
        True, maxw=250, tag='lp:raw')
lc.text(RAW[0] + 130, LY0 + 96, '（非负，起点）', 8.8, C_MUTE, 'middle', tag='lp:raw2')

# 两个动作并排成环
B1 = (LX0 + 30, LY0 + 146, 240, 74)
B2 = (LX0 + 320, LY0 + 146, 240, 74)
lc.rect(*B1, C_SCOR_F, C_SCOR_S, rx=8, sw=1.6)
lc.text(B1[0] + 120, B1[1] + 28, '① 行归一', 12, '#7c2d12', 'middle', True, tag='lp:r0')
lc.text(B1[0] + 120, B1[1] + 50, '行和 → 1（softmax）', 8.8, '#334155', 'middle', maxw=220,
        tag='lp:r1')
lc.rect(*B2, C_SCOR_F, C_SCOR_S, rx=8, sw=1.6)
lc.text(B2[0] + 120, B2[1] + 28, '② 列归一', 12, '#7c2d12', 'middle', True, tag='lp:c0')
lc.text(B2[0] + 120, B2[1] + 50, '列和 → 1（当轮就归 1，破坏行和一点点）', 8.8, '#334155', 'middle',
        maxw=220, tag='lp:c1')
lc.parrow([(RAW[0] + 130, RAW[1] + 54), (RAW[0] + 130, B1[1] - 2)], '#64748b', 1.8, marker='gy')
lc.parrow([(B1[0] + 240, B1[1] + 22), (B2[0] - 2, B1[1] + 22)], C_SCOR_S, 2.0, marker='rk')
lc.parrow([(B2[0] + 120, B2[1] + 74), (B2[0] + 120, B2[1] + 88), (B1[0] + 120, B1[1] + 88),
           (B1[0] + 120, B1[1] + 74)], C_SCOR_S, 2.0, marker='rk')
lc.text(B2[0] + 130, B2[1] + 102, '一轮 = ① + ②，然后回到 ①', 9, C_SCOR_S, 'start', True,
        maxw=260, tag='lp:o')
lc.text(LX0 + 30, LY0 + 268, '论文写 M^(t) = T_r(T_c(·))（行算子套列算子）；两侧实现是 softmax → 列归一', 8.8,
        '#334155', 'start', maxw=550, tag='lp:n0')
lc.text(LX0 + 30, LY0 + 284, '→ (行 → 列) 重复 sinkhorn_iters − 1 次：极限相同、定点略差。', 8.8,
        '#334155', 'start', maxw=550, tag='lp:n0b')
lc.text(LX0 + 30, LY0 + 302, '起点 exp 与起点 softmax 结果相同（softmax 把第一步行归一吸收掉了）。',
        8.8, C_MUTE, 'start', maxw=550, tag='lp:n1')
lc.text(LX0 + 30, LY0 + 318, 'order = row-first 与 col-first 的 M^(20) 互为转置型'
                             '（0.92408478 / 0.07580203 / 0.07591422 …）。', 8.8, C_MUTE,
        'start', maxw=550, tag='lp:n2')

# ================= 右：收敛曲线 =================
CX0, CY0, CX1, CY1 = 660, 118, BXR, 440
lc.rect(CX0, CY0, CX1 - CX0, CY1 - CY0, '#ffffff', C_SCOR_S, rx=10, sw=1.5)
lc.text(CX0 + 16, CY0 + 26, '本轮实测：行和偏差逐轮下降，列和偏差贴着地板', 12, C_TXT, 'start',
        True, maxw=440, tag='cv:t')

AXL, AXR = CX0 + 90, CX1 - 40
AYT, AYB = CY0 + 60, CY1 - 76          # 顶部 = 大偏差，底部 = eps 地板
LO, HI = -6.4, -1.2                     # log10 范围：1e-6.4 … 1e-1.2
ROUNDS = 20


def px(r):
    return AXL + (r - 1) / (ROUNDS - 1) * (AXR - AXL)


def py(v):
    t = (math.log10(max(v, 10 ** LO)) - LO) / (HI - LO)
    return AYB - t * (AYB - AYT)


for exp in (-1, -2, -3, -4, -5, -6):
    y = py(10.0 ** exp)
    lc.seg(AXL, y, AXR, y, '#e2e8f0', 1.2, dash=True)
    lc.text(AXL - 8, y + 3, '1e%d' % exp, 8.4, C_MUTE, 'end', tag='cv:t%d' % exp)
lc.seg(AXL, AYB, AXR, AYB, '#94a3b8', 1.4)
lc.seg(AXL, AYT, AXL, AYB, '#94a3b8', 1.4)
# eps 地板
FLOOR_Y = py(1e-6)
lc.rect(AXL, FLOOR_Y - 4, AXR - AXL, 12, '#f1f5f9', '#cbd5e1', rx=3, sw=1.0)
lc.text(AXR, FLOOR_Y + 22, 'eps 地板 = 1e-06（分母每次加 eps）', 8.6, '#334155', 'end',
        maxw=300, tag='cv:floor')
# row_dev 折线：5 个实心点 = 驱动脚本实测的**真实第 1–5 轮**；第 20 轮单独一个空心点（见下）
ROW = [0.02756985, 0.00839717, 0.00255908, 0.00078041, 0.00023847]
ROW20 = 1e-6                       # 第 20 轮末 row_dev = 1.0000043e-06（印成 1.00e-06）
pts = [(px(i + 1), py(v)) for i, v in enumerate(ROW)]
r20 = (px(20), py(ROW20))
lc.seg(pts[-1][0], pts[-1][1], r20[0], r20[1], C_SCOR_S, 1.4, dash=True)   # 6–20 轮收尾示意
lc.parrow(pts, C_SCOR_S, 2.4, marker=None)
for i, (x, y) in enumerate(pts):
    lc.rect(x - 3, y - 3, 6, 6, C_SCOR_S, C_SCOR_S, rx=1.5, sw=1)
    lc.text(x + 6, y - 8, '%.8f' % ROW[i], 8.4, '#7c2d12', 'start', maxw=90, tag='cv:r%d' % i)
# 第 20 轮单点：空心方点 + 标注（halo 白描边——虚线从字形处断开）
lc.rect(r20[0] - 4, r20[1] - 4, 8, 8, '#ffffff', C_SCOR_S, rx=1.5, sw=1.8)
lc.text(r20[0] - 8, r20[1] - 10, '第 20 轮 1.00e-06（贴地板）', 8.8, C_SCOR_S, 'end', True,
        maxw=190, tag='cv:r20', halo=True)
# col_dev 折线（贴地板）
COL = [1.05e-06, 1.02e-06, 1e-06, 1e-06, 1e-06]
cpts = [(px(i + 1), py(v)) for i, v in enumerate(COL)]
lc.parrow(cpts, C_KV_S, 2.0, marker=None)
lc.text(px(5) + 8, py(1e-6) + 8, 'col_dev 全程贴在 eps 地板上', 8.8, C_KV_DEEP, 'start',
        maxw=260, tag='cv:c')
# t = 20 竖线
lc.seg(px(20), AYT, px(20), AYB, C_SCOR_S, 1.8, dash=True)
lc.text(px(20) - 6, AYT + 12, 't = 20', 9.5, C_SCOR_S, 'end', True, tag='cv:t20')
lc.text(px(20) - 10, AYT + 30, '（这里的偏差已在', 8.4, C_MUTE, 'end', maxw=140, tag='cv:t20a')
lc.text(px(20) - 10, AYT + 44, '地板上下不去了）', 8.4, C_MUTE, 'end', maxw=140, tag='cv:t20b')
lc.text(AXL, AYB + 38, '轮次 1 → 5 逐点标注（每轮 = 一次行归一 + 一次列归一；col_dev 在每轮末恰好'
                       '归 1）；虚线为第 6–20 轮收尾，右端单点标出第 20 轮',
        8.6, C_MUTE, 'start', maxw=640, tag='cv:x')
lc.text(CX0 + 16, CY1 - 22, '第 20 轮末：row_dev = 1.00e-06、col_dev = 1.00e-06'
                            '——最差的那档矩阵 20 轮仍有残差。', 9, '#7c2d12', 'start', True,
        maxw=600, tag='cv:n')

# ================= 底部：20 轮够不够 + 两处实现细节 =================
BY0 = 462
lc.rect(MX, BY0, 726 - MX, 220, '#f8fafc', '#cbd5e1', rx=10, sw=1.3)
lc.text(MX + 16, BY0 + 26, '20 轮够不够：换五档矩阵，20 轮与 200 轮的残差同档', 11.5, C_TXT,
        'start', True, maxw=500, tag='q:t')
QT = [['矩阵', '20 轮残差', '200 轮残差'],
      ['[[4, 1], [1, 3]]', '1.00e-06', '1.00e-06'],
      ['[[1, 2], [3, 4]]', '1.00e-06', '1.00e-06'],
      ['[[1, 1], [1, 1.01]]（近奇异）', '1.00e-06', '1.00e-06'],
      ['[[2, 1], [1, 1.0001]]（近奇异）', '1.00e-06', '1.00e-06'],
      ['[[1, 1e-6], [1e-6, 1]]', '1.00e-06', '1.00e-06']]
CWID = [300, 150, 150]
for r, row in enumerate(QT):
    ty = BY0 + 56 + r * 24
    cx = MX + 16
    for c, cell in enumerate(row):
        lc.text(cx, ty, cell, 9.2, C_TXT if r == 0 else '#334155', 'start', r == 0,
                maxw=CWID[c] - 8, tag='q:%d%d' % (r, c))
        cx += CWID[c]
    if r == 0:
        lc.seg(MX + 16, ty + 7, MX + 16 + sum(CWID), ty + 7, C_MUTE, 1.2)
lc.text(MX + 16, BY0 + 194, '结论：t_max = 20 是论文原话里的 as a practical value——'
                            '精度与开销的折中，不是『迭代到收敛』。', 9.5, '#7c2d12', 'start',
        True, maxw=640, tag='q:n0')
lc.text(MX + 16, BY0 + 212, '该矩阵 20 轮的结果 = [[0.77598999, 0.22400901], [0.22400901, 0.77598999]]，'
                            '行列和都是 [0.999999, 0.999999]。', 8.6, C_MUTE, 'start', maxw=640,
        tag='q:n1')

lc.rect(774, BY0, BXR - 774, 220, C_KV_F, C_KV_S, rx=10, sw=1.4)
lc.text(774 + 16, BY0 + 26, '这条曲线上的三处口径', 11.5, C_TXT, 'start', True, maxw=340,
        tag='z:t')
ZD = [('收缩率', '0.02756985 → 0.00839717 → 0.00255908 → 0.00078041 → 0.00023847（第 1–5 轮）',
       '每轮约缩到前一轮的 1/3（列和当轮归 1、下一轮行归一再破坏一点点）；第 20 轮停在地板 1.00e-06'),
      ('为什么有地板', '分母每次加 eps = 1e-06',
       '偏差不可能到 0；对残差流混合的实际影响约 0.0001% 量级的行列和不守恒'),
      ('config 口径', 'hc_sinkhorn_iters = 20、hc_eps = 1e-06',
       '与论文 t_max = 20 互证；每个半层一次 4×4 的 20 轮交替归一')]
for i, (a, b, c) in enumerate(ZD):
    ty = BY0 + 56 + i * 56
    lc.text(774 + 16, ty, a, 9.8, C_TXT, 'start', True, tag='z:a%d' % i)
    lc.text(774 + 16, ty + 18, b, 9.5, C_KV_DEEP, 'start', True, maxw=600, tag='z:b%d' % i)
    lc.text(774 + 16, ty + 36, c, 8.6, C_MUTE, 'start', maxw=620, tag='z:c%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 712, '数字口径：玩具矩阵 raw = [[4.0, 1.0], [1.0, 3.0]]（非负、起点即原矩阵，便于手算）；'
                 'eps = 1e-06、迭代 20 轮。', 8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 730, '依据 mHC 原论文 §4.2 的 Eq.(8)(9)（arXiv:2512.24880）与 DeepSeek-V4 技术报告 '
                 '§2.2 Eq.(8)（arXiv:2606.19348）；逐轮偏差由本章驱动脚本实跑。', 8.5, C_MUTE,
        'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-sinkhorn.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
