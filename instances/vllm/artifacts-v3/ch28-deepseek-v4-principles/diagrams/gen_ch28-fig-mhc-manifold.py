#!/usr/bin/env python3
"""ch28 机制图 · mHC 更新式与双随机约束（ch28-fig-mhc-manifold）

claim：多流残差的稳定性来自一个约束而不是一个技巧：流间混合 B_l 被关进
Birkhoff 多面体（双随机）→ 每次混合的谱范数恰为 1（不放大）、复合仍在内
（深堆叠稳）、并且保留了恒等映射。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  双随机 B2 = [[0.7,0.3],[0.3,0.7]]：行和/列和 [1.0, 1.0]，奇异值 [1.0, 0.4]，谱范数 1.0
  反例 M = [[0.9,0.1],[0.2,0.8]]：列和 [1.1, 0.9]，谱范数 1.009583（最坏方向上的放大上界）
  幂次谱范数（指数取 1、2、3、5、9、17、33）：M^1 1.009583 → M^2 1.018627 → M^3 1.026639
  → M^5 1.038734 → M^9 1.049989 → M^17 1.053848 → M^33 1.054092（单调增、迅速饱和；
  2026-09-16 修正：原标签整组错一位，M^k 实为 M^(k+1) 的谱范数）
  封闭性：B2@B2 = [[0.58, 0.42], [0.42, 0.58]]（谱范数 1.0）
  非扩张对照：‖X‖_F = 2.5 → 双随机 1.537856（0.615142）/ 反例 2.076656（0.830662）
  B = I 时 B X 与 X 逐位相同（True）——恒等映射仍在集合里

配色走 book/cartography/l0_common.py 的角色常量（GPU 绿 = 执行臂内的残差流）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 880
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_BAD_S, C_BAD_F = lc.C_ABORT, '#fef2f2'          # 反例 = 红
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'

EXTRA_DEFS = ('<defs>'
              '<marker id="gy" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
              '<marker id="gn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_GPU_S}"/></marker>'
              '<marker id="rd" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_BAD_S}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '四条残差流凭什么不塌：把流间混合关进 Birkhoff 多面体', 17, C_TXT, 'start',
        True, maxw=700, tag='title')
lc.text(MX, 56, '老式残差靠恒等映射这条护身符；流扩成 4 条之后护身符碎了——mHC 的答法是把'
                '流间混合矩阵限制成双随机矩阵：不放大、可复合、还留着恒等映射', 10, C_MUTE,
        'start', maxw=1150, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』的残差流形态', 9, C_MUTE, 'end',
        maxw=430, tag='l0:a')
lc.text(BXR, 64, '——从 1 条残差扩成 hc_mult = 4 条流的读写回路', 9, C_MUTE, 'end', maxw=430,
        tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_GPU_F, C_GPU_S, '残差流 / 子层（执行臂绿）'),
                  (C_KV_F, C_KV_S, '被约束的矩阵 B_l（笼子）'),
                  (C_BAD_F, C_BAD_S, '反例：只满足行和 = 1')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=250, tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 16

# ================= 左：半层回路 =================
LX0, LY0, LX1, LY1 = MX, 118, 880, 452
lc.rect(LX0, LY0, LX1 - LX0, LY1 - LY0, '#ffffff', C_GPU_S, rx=10, sw=1.5)
lc.text(LX0 + 16, LY0 + 26, '一个半层的读写回路（X_{l+1} = B_l X_l + C_l F_l(A_l X_l)）', 12,
        C_TXT, 'start', True, maxw=560, tag='loop:t')

XL = (LX0 + 20, 250, 140, 76)
lc.rect(*XL, C_GPU_F, C_GPU_S, rx=8, sw=1.5)
lc.text(XL[0] + XL[2] / 2, XL[1] + 28, 'X_l', 14, C_TXT, 'middle', True, tag='loop:xl')
lc.text(XL[0] + XL[2] / 2, XL[1] + 50, '4 条流 × d', 9, C_MUTE, 'middle', tag='loop:xl2')
lc.text(XL[0] + XL[2] / 2, XL[1] + 66, 'hc_mult = 4', 8.8, C_GPU_S, 'middle', True, tag='loop:xl3')

TOP = [('A_l 读入', 'A_l = σ(Ã_l)', 200), ('F_l 子层', '注意力或 FFN', 350), ('C_l 写回', 'C_l = 2σ(C~_l)', 500)]
for lab, sub, x in TOP:
    lc.rect(LX0 + 150 + (x - 200), 176, 120, 58, C_GPU_F, C_GPU_S, rx=8, sw=1.4)
    lc.text(LX0 + 150 + (x - 200) + 60, 198, lab, 10.5, C_TXT, 'middle', True, maxw=110,
            tag='loop:%s' % lab[:2])
    lc.text(LX0 + 150 + (x - 200) + 60, 216, sub, 8.4, C_MUTE, 'middle', maxw=112,
            tag='loop:s%s' % lab[:2])
for i in range(2):
    x0 = LX0 + 150 + (TOP[i][2] - 200) + 120
    x1 = LX0 + 150 + (TOP[i + 1][2] - 200)
    lc.seg(x0, 205, x1 - 2, 205, '#64748b', 1.8, marker='gy')
lc.parrow([(XL[0] + XL[2], XL[1] + 20), (LX0 + 148, XL[1] + 20), (LX0 + 148, 205),
           (LX0 + 150, 205)], '#64748b', 1.8, marker='gy')

# 笼子（2026-09-16 压框修复：框高 96 → 84 ⇒ 底边 416 → 404；内部四行同步收拢为
# 18 的均匀行距、首行留 20，使末行至底边净空 ≈7.7、底边至下方脚注墨迹净空 ≈5.1，
# 两处均 ≥4——只动本块坐标，文字与数字零改动）
C_PAD, C_PITCH = 20, 18
CAGE = (LX0 + 170, 320, 300, 84)
lc.rect(*CAGE, C_KV_F, C_KV_S, rx=8, sw=1.8)
lc.text(CAGE[0] + 16, CAGE[1] + C_PAD, 'B_l：流间混合（关在笼子里）', 11, C_KV_DEEP, 'start', True,
        maxw=270, tag='cage:t')
for i, s in enumerate(['非负', '行和 = 1', '列和 = 1']):
    lc.text(CAGE[0] + 16 + i * 94, CAGE[1] + C_PAD + C_PITCH, s, 9.2, C_KV_DEEP, 'start', True,
            tag='cage:c%d' % i)
lc.text(CAGE[0] + 16, CAGE[1] + C_PAD + 2 * C_PITCH, '三条一起 = 双随机矩阵 = Birkhoff 多面体的一个点', 8.8,
        '#334155', 'start', maxw=280, tag='cage:s')
lc.text(CAGE[0] + 16, CAGE[1] + C_PAD + 3 * C_PITCH, '它的顶点是置换矩阵（含恒等映射 I）', 8.8, C_MUTE,
        'start', maxw=280, tag='cage:s2')
lc.parrow([(XL[0] + XL[2] - 20, XL[1] + XL[3]), (XL[0] + XL[2] - 20, 368), (CAGE[0] - 2, 368)],
          '#64748b', 1.8, marker='gy')

# ⊕ 与 X_{l+1}
PLUS = (LX0 + 560, 205)
lc.rect(PLUS[0] - 16, PLUS[1] - 16, 32, 32, '#ffffff', '#64748b', rx=16, sw=1.6)
lc.text(PLUS[0], PLUS[1] + 6, '+', 17, '#334155', 'middle', True, tag='loop:plus')
lc.parrow([(LX0 + 150 + (TOP[2][2] - 200) + 120, 205), (PLUS[0] - 18, 205)], '#64748b', 1.8)
lc.parrow([(CAGE[0] + CAGE[2], 368), (PLUS[0], 368), (PLUS[0], PLUS[1] + 16)],
          C_KV_S, 2.0, marker=None)
lc.parrow([(PLUS[0] + 16, PLUS[1]), (LX0 + 690, PLUS[1])], '#64748b', 1.8, marker='gy')
X2 = (LX0 + 690, 176, 140, 58)
lc.rect(*X2, C_GPU_F, C_GPU_S, rx=8, sw=1.5)
lc.text(X2[0] + 70, 200, 'X_{l+1}', 12.5, C_TXT, 'middle', True, tag='loop:x2')
lc.text(X2[0] + 70, 220, '4 条流（不塌）', 8.8, C_GPU_S, 'middle', tag='loop:x2b')
lc.text(CAGE[0] + 320, 358, 'B 让残差在流间混合', 8.8, C_KV_DEEP, 'start', maxw=180, tag='loop:b')
lc.text(LX0 + 20, LY1 - 34, '读入 A 把 4 条流聚成子层的一路输入；写回 C 把子层输出散回 4 条流；'
                            'B 只负责让残差在流之间流动。', 9, '#334155', 'start', maxw=760,
        tag='loop:n0')
lc.text(LX0 + 20, LY1 - 14, '三张映射每层现算（动态项 + 静态偏置），B 现算出来还要先过 20 轮 '
                            'Sinkhorn 才进笼子。', 9, C_MUTE, 'start', maxw=760, tag='loop:n1')

# ================= 右上：笼子里的四条读数 =================
RX0 = 920
lc.rect(RX0, LY0, BXR - RX0, 234, '#ffffff', C_KV_S, rx=10, sw=1.5)
lc.text(RX0 + 16, LY0 + 26, '笼子里的四条读数（本节实测）', 12, C_TXT, 'start', True, maxw=340,
        tag='cage2:t')
RD = [('谱范数恰好 = 1', 'B2 奇异值 = [1.0, 0.4] → ‖B‖2 = 1.0',
       '1 必是特征值（行和 = 1），列和再压住上界——两头夹住只能是 1'),
      ('相乘还在笼子里', 'B2 @ B2 = [[0.58, 0.42], [0.42, 0.58]]，谱范数 1.0',
       '双随机集合对乘法封闭 → 深堆叠仍非扩张'),
      ('非扩张', '‖X‖_F 2.5 → 1.537856（比值 0.615142）',
       '第二个奇异值 0.4 才是真在收缩的那部分：它磨掉流间差异'),
      ('恒等映射还在', 'B = I 时 B X 与 X 逐位相同 = True',
       '多流退化成单流残差的那条『加回去』')]
for i, (a, b, c) in enumerate(RD):
    ty = LY0 + 52 + i * 46
    lc.text(RX0 + 16, ty, a, 10, C_TXT, 'start', True, tag='cage2:a%d' % i)
    lc.text(RX0 + 16, ty + 17, b, 9.2, C_KV_DEEP, 'start', maxw=480, tag='cage2:b%d' % i)
    lc.text(RX0 + 16, ty + 33, c, 8.5, C_MUTE, 'start', maxw=500, tag='cage2:c%d' % i)

# ================= 右下：反例卡 =================
lc.rect(RX0, 368, BXR - RX0, 84, C_BAD_F, C_BAD_S, rx=10, sw=1.5)
lc.text(RX0 + 16, 394, '反例：行和 = 1 但列和 ≠ 1 的矩阵', 11, C_BAD_S, 'start', True, maxw=340,
        tag='bad:t')
lc.text(RX0 + 16, 414, 'M = [[0.9, 0.1], [0.2, 0.8]]，列和 = [1.1, 0.9] → 谱范数 1.009583 > 1',
        9.2, '#334155', 'start', maxw=500, tag='bad:l0')
lc.text(RX0 + 16, 432, '反复自乘：单调增、迅速饱和在 1.054092——不是指数放大', 9.2, C_BAD_S,
        'start', True, maxw=500, tag='bad:l1')

# ================= 底左：反例的算术形态 =================
BY0 = 476
lc.rect(MX, BY0, 726 - MX, 150, C_BAD_F, C_BAD_S, rx=10, sw=1.4)
lc.text(MX + 16, BY0 + 24, '深堆叠为什么飘：每一次混合都在放大', 11.5, C_BAD_S, 'start', True,
        maxw=420, tag='bad2:t')
lc.text(MX + 16, BY0 + 46, '反例矩阵在最坏方向上最多放大 1.009583 倍（不是每个方向都被放大）；'
                           '同一个矩阵反复自乘，幂次谱范数单调增、迅速饱和在 1.054092。',
        9, '#334155', 'start', maxw=620, tag='bad2:s')
CP = ['M^1 1.009583', 'M^2 1.018627', 'M^3 1.026639', 'M^5 1.038734',
      'M^9 1.049989', 'M^17 1.053848', 'M^33 1.054092']
for i, s in enumerate(CP):
    cxx = MX + 16 + (i % 4) * 172
    cyy = BY0 + 78 + (i // 4) * 28
    lc.text(cxx, cyy, s, 9.5, C_BAD_S, 'start', True, tag='bad2:c%d' % i)
lc.text(MX + 16, BY0 + 138, '这就是摘要里 severe training instability 的算术形态：'
                            '把 B 关进双随机集合，这个放大源就没了。', 8.8, '#334155',
        'start', maxw=620, tag='bad2:n')

# ================= 底右：同一组流、两种混合矩阵 =================
lc.rect(774, BY0, BXR - 774, 150, '#ffffff', C_KV_S, rx=10, sw=1.4)
lc.text(774 + 16, BY0 + 24, '同一组流、两种混合矩阵（非扩张对照）', 11.5, C_TXT, 'start', True,
        maxw=420, tag='cmp:t')
lc.text(774 + 16, BY0 + 46, 'X = [[1.0, 2.0], [0.5, −1.0]]，‖X‖_F = 2.5', 9.2, '#334155',
        'start', maxw=620, tag='cmp:s')
CM = [('双随机 B2', 'B X = [[0.85, 1.1], [0.65, −0.1]]', '‖B X‖_F = 1.537856（比值 0.615142）',
       False),
      ('反例 M', 'B X = [[0.95, 1.7], [0.6, −0.4]]', '‖B X‖_F = 2.076656（比值 0.830662）', True)]
for i, (a, b, c, bad) in enumerate(CM):
    ty = BY0 + 76 + i * 32
    col = C_BAD_S if bad else C_KV_DEEP
    lc.text(774 + 16, ty, a, 9.5, C_TXT, 'start', True, tag='cmp:a%d' % i)
    lc.text(774 + 100, ty, b, 9.2, '#334155', 'start', maxw=280, tag='cmp:b%d' % i)
    lc.text(774 + 400, ty, c, 9.2, col, 'start', True, maxw=280, tag='cmp:c%d' % i)
lc.text(774 + 16, BY0 + 138, '两档都不放大到 2.5 以上，但双随机那一档压得更狠——它连流间差异一起磨。',
        8.8, C_MUTE, 'start', maxw=620, tag='cmp:n')

# ================= 底带：一句话 =================
LY = 646
lc.rect(MX, LY, BXR - MX, 44, C_KV_F, C_KV_S, rx=8, sw=1.5)
lc.text(MX + 18, LY + 27, '一句话：多流不塌靠的不是技巧而是一个约束——B_l 落在 Birkhoff 多面体里'
                          '→ 每次混合非扩张（谱范数恒 1）、复合仍在内、且恒等映射留在集合中。',
        11, C_KV_DEEP, 'start', True, maxw=BXR - MX - 40, tag='sum')

# ================= 底：双随机判定表 =================
TY = 706
lc.rect(MX, TY, BXR - MX, 108, '#f8fafc', '#cbd5e1', rx=10, sw=1.3)
lc.text(MX + 16, TY + 24, '判定表（两个玩具矩阵各跑一遍）', 11, C_TXT, 'start', True, maxw=340,
        tag='tb:t')
TB = [['矩阵', '行和', '列和', '非负', '双随机', '奇异值', '谱范数'],
      ['B2 = [[0.7, 0.3], [0.3, 0.7]]', '[1.0, 1.0]', '[1.0, 1.0]', 'True', 'True',
       '[1.0, 0.4]', '1.0'],
      ['B3 = [[0.5, 0.3, 0.2], …]（3 条流，非对称）', '[1.0, 1.0, 1.0]', '[1.0, 1.0, 1.0]',
       'True', 'True', '[1.0, 0.264575, 0.264575]', '1.0'],
      ['M = [[0.9, 0.1], [0.2, 0.8]]', '[1.0, 1.0]', '[1.1, 0.9]', 'True', 'False', '—',
       '1.009583']]
CWID = [420, 130, 130, 70, 90, 260, 120]
for r, row in enumerate(TB):
    ty = TY + 48 + r * 20
    cx = MX + 16
    for c, cell in enumerate(row):
        lc.text(cx, ty, cell, 8.8, C_TXT if r == 0 else '#334155', 'start', r == 0,
                maxw=CWID[c] - 8, tag='tb:%d%d' % (r, c))
        cx += CWID[c]
    if r == 0:
        lc.seg(MX + 16, ty + 7, MX + 16 + sum(CWID), ty + 7, C_MUTE, 1.2)

# ---------------- 页脚 ----------------
lc.text(MX, 836, '数字口径：玩具矩阵 n_hc = 2 与 3；反例矩阵与 B2 都按同一套定义算谱范数'
                 '（最大奇异值）。', 8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 854, '依据 DeepSeek-V4 技术报告 §2.2 的 Eq.(1)(2)（arXiv:2606.19348）与 mHC 原论文 '
                 '§4.1 的三性质（arXiv:2512.24880）；全部数值由本章驱动脚本实跑。', 8.5, C_MUTE,
        'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-mhc-manifold.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
