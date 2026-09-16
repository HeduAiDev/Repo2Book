#!/usr/bin/env python3
"""ch28 机制图 · MoE 两板斧：细粒度专家切分 + 共享专家无条件计算（ch28-fig-moe-axes）

claim：DeepSeekMoE 的两板斧：细粒度切分让组合数从 6 涨到 70 而算力不变（16 = 16），
共享专家无条件计算——式子里它那一项没有门控，代价是每 token 恒定多付一份算力。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  切分前 4 专家 × d_ff 8 挑 2 → 算力 16；切分后 8 专家 × 4 挑 4 → 算力 16
  组合数 C(4,2) = 6 → C(8,4) = 70（11.666667 倍）
  config：256 挑 6；moe_intermediate_size 2048 / hidden_size 4096 = 0.500000；
  对照 4×hidden 16384 → 8.000000 分之一
  共享项 [0.5, −0.5] 不随门控变化；门控翻倍后 h 从 [2.05, 0.55] 变 [2.6, 1.1]
  路由池 start=1 → 门控 [0.0, 0.1]；start=0 → 门控 [0.9, 0.0]

配色走 book/cartography/l0_common.py 的角色常量（GPU 绿 = 执行臂内的 FFN 段）。
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
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_SCOR_F, C_SCOR_S = '#ffedd5', '#c2410c'
C_SH_F, C_SH_S = '#dcfce7', '#16a34a'          # 共享专家（无条件）

EXTRA_DEFS = ('<defs>'
              '<marker id="gn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_GPU_S}"/></marker>'
              '<marker id="wm" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_SCOR_S}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '两板斧：切得更细、留一个永远算', 17, C_TXT, 'start', True, maxw=680, tag='title')
lc.text(MX, 56, '板斧一：专家数乘 m、每个的中间维砍到 1/m、激活数也乘 m——算力一分没多，'
                '可选组合从 6 种变成 70 种', 10, C_MUTE, 'start', maxw=1050, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』的 FFN 段', 9, C_MUTE, 'end',
        maxw=430, tag='l0:a')
lc.text(BXR, 64, '——下游是查表派单那几层', 9, C_MUTE, 'end', maxw=430, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_GPU_F, C_GPU_S, '路由专家（参与挑选）'),
                  (C_SH_F, C_SH_S, '共享专家（无条件计算）'),
                  (C_SCOR_F, C_SCOR_S, '门控分数 / 权重')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=250, tag='lg:%s' % lab[:3])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 16

# ================= 板斧一：前后对照 =================
PY0, PY1 = 118, 400
lc.rect(MX, PY0, 700, PY1 - PY0, '#ffffff', C_GPU_S, rx=10, sw=1.5)
lc.text(MX + 16, PY0 + 26, '切分前：4 个大专家', 12, C_TXT, 'start', True, maxw=280, tag='b1:t')
lc.text(MX + 16, PY0 + 46, 'N = 4 · d_ff = 8 · 每 token 挑 K = 2', 9.2, C_MUTE, 'start',
        maxw=400, tag='b1:s')
for i in range(4):
    cy = PY0 + 66 + i * 30
    lc.rect(MX + 20, cy, 150, 24, C_GPU_F, C_GPU_S, rx=4, sw=1.2)
    lc.text(MX + 95, cy + 16, '专家 %d（中间维 8）' % (i + 1), 9.2, C_TXT, 'middle', maxw=145,
            tag='b1:e%d' % i)
lc.text(MX + 190, PY0 + 86, '挑 2 个 → 激活算力 = 2 × 8 = 16', 9.5, '#334155', 'start', True,
        maxw=300, tag='b1:f0')
lc.text(MX + 190, PY0 + 106, '组合数 C(4, 2) = 6', 9.5, C_KV_DEEP, 'start', True, maxw=300,
        tag='b1:f1')
lc.text(MX + 190, PY0 + 126, '（每 token 只在这 6 种里选一种）', 8.8, C_MUTE, 'start', maxw=300,
        tag='b1:f2')

lc.rect(774, PY0, 686, PY1 - PY0, '#ffffff', C_GPU_S, rx=10, sw=1.5)
lc.text(774 + 16, PY0 + 26, '切分后：8 个小专家（m = 2）', 12, C_TXT, 'start', True, maxw=320,
        tag='b2:t')
lc.text(774 + 16, PY0 + 46, 'mN = 8 · d_ff/m = 4 · 每 token 挑 mK = 4', 9.2, C_MUTE, 'start',
        maxw=400, tag='b2:s')
for i in range(8):
    cx = 774 + 20 + (i % 4) * 88
    cy = PY0 + 66 + (i // 4) * 30
    lc.rect(cx, cy, 82, 24, C_GPU_F, C_GPU_S, rx=4, sw=1.2)
    lc.text(cx + 41, cy + 16, '专家 %d' % (i + 1), 9.2, C_TXT, 'middle', maxw=78, tag='b2:e%d' % i)
lc.text(774 + 400, PY0 + 86, '挑 4 个 → 激活算力 = 4 × 4 = 16', 9.5, '#334155', 'start', True,
        maxw=300, tag='b2:f0')
lc.text(774 + 400, PY0 + 106, '组合数 C(8, 4) = 70（11.666667 倍）', 9.5, C_KV_DEEP, 'start',
        True, maxw=300, tag='b2:f1')
lc.text(774 + 400, PY0 + 126, '（可选组合更多，算力却一分没多）', 8.8, C_MUTE, 'start',
        maxw=300, tag='b2:f2')

# 算力条对照
BAR = [('切分前', MX + 20, 16, C_GPU_S), ('切分后', 774 + 20, 16, C_GPU_S)]
lc.text(MX + 16, PY0 + 216, '每 token 的激活算力（按中间维计）', 9.5, C_TXT, 'start', True,
        maxw=340, tag='b3:t')
for lab, bx, val, col in BAR:
    cy = PY0 + 232 if bx < 700 else PY0 + 232
    lc.rect(bx, cy - 10, 420, 20, '#f1f5f9', '#cbd5e1', rx=3, sw=1.0)
    lc.rect(bx, cy - 10, 210, 20, col, col, rx=3, sw=1.0)
    lc.text(bx, cy + 26, '%s：2 × 8 = 16' % lab if bx < 700 else '%s：4 × 4 = 16' % lab,
            9.2, '#334155', 'start', maxw=420, tag='b3:%s' % lab)
    lc.text(bx + 218, cy + 4, '16', 11, col, 'start', True, tag='b3:v%s' % lab)
    lc.text(bx + 246, cy + 4, '（两根条一样长：算力不变）' if bx < 700 else '（与切分前相同）',
            8.8, C_MUTE, 'start', maxw=220, tag='b3:n%s' % lab)

# ================= 板斧二：共享专家 =================
SY0 = 420
lc.rect(MX, SY0, BXR - MX, 190, C_SH_F, C_SH_S, rx=10, sw=1.5)
lc.text(MX + 16, SY0 + 26, '板斧二：共享专家无条件计算（式子里它那一项没有门控）', 12.5,
        '#14532d', 'start', True, maxw=520, tag='sh:t')
lc.rect(MX + 20, SY0 + 48, 170, 44, C_SH_F, '#14532d', rx=6, sw=1.8)
lc.text(MX + 105, SY0 + 68, '共享专家（K_s = 1）', 10, '#14532d', 'middle', True, maxw=160,
        tag='sh:b0')
lc.text(MX + 105, SY0 + 84, '没有门控线连过来', 8.6, '#14532d', 'middle', maxw=160, tag='sh:b1')
lc.rect(MX + 220, SY0 + 48, 470, 44, '#ffffff', C_GPU_S, rx=6, sw=1.4)
lc.text(MX + 236, SY0 + 68, '路由池：8 个路由专家里挑 4 个（共享专家不在池内）', 9.8, C_TXT,
        'start', True, maxw=440, tag='sh:b2')
lc.text(MX + 236, SY0 + 84, '门控分数只落在它们身上', 8.6, C_MUTE, 'start', maxw=440, tag='sh:b3')
lc.text(MX + 720, SY0 + 60, 'h = 共享项 + 路由项 + u', 11, C_KV_DEEP, 'start', True, maxw=280,
        tag='sh:e0')
lc.text(MX + 720, SY0 + 80, '共享项：+ 0.5, −0.5（不动）', 9.2, '#14532d', 'start', True,
        maxw=300, tag='sh:e1')

VAL = [('门控 a = [0.4, 0.3, 0.2, 0.1]', '路由项 = [0.55, 0.55]', 'h = [2.05, 0.55]'),
       ('门控 b = 2 × a（整体翻倍）', '路由项 = [1.1, 1.1]', 'h = [2.6, 1.1]')]
for i, (a, b, c) in enumerate(VAL):
    ty = SY0 + 116 + i * 26
    lc.text(MX + 20, ty, a, 9.5, C_TXT, 'start', True, tag='sh:a%d' % i)
    lc.text(MX + 300, ty, b, 9.5, '#334155', 'start', tag='sh:b%d' % i)
    lc.text(MX + 520, ty, c, 9.5, C_KV_DEEP, 'start', True, tag='sh:c%d' % i)
    lc.text(MX + 700, ty, '共享项仍是 [0.5, −0.5]（逐位相同 = True）' if i == 1
            else '共享项 = [0.5, −0.5]', 9, '#14532d', 'start', maxw=420, tag='sh:d%d' % i)
lc.text(MX + 16, SY0 + 176, '反证方向：一旦把共享位放回路由池（start = 0），它就会像普通专家一样竞争'
                            '——挑不中就等于没算。', 9, '#14532d', 'start', True, maxw=1100,
        tag='sh:n')

# ================= 底：反证 + config 口径 =================
BY0 = 630
lc.rect(MX, BY0, 700, 150, '#f8fafc', '#cbd5e1', rx=10, sw=1.3)
lc.text(MX + 16, BY0 + 26, '反证：共享位放回池子会怎样（同一组 scores = [0.9, 0.1]、k = 1）',
        11, C_TXT, 'start', True, maxw=460, tag='ctr:t')
CT = [('start = 1（口径：共享位排除）', '门控 = [0.0, 0.1]', '分数只落在专家 1 上', False),
      ('start = 0（共享位进池子）', '门控 = [0.9, 0.0]', '分数落在专家 0 上 → 共享专家『没算』', True)]
for i, (a, b, c, bad) in enumerate(CT):
    ty = BY0 + 62 + i * 42
    col = lc.C_ABORT if bad else C_KV_DEEP
    lc.text(MX + 16, ty, a, 9.5, C_TXT, 'start', True, tag='ctr:a%d' % i)
    lc.text(MX + 16, ty + 18, b, 10.5, col, 'start', True, tag='ctr:b%d' % i)
    lc.text(MX + 220, ty + 18, c, 9, C_MUTE, 'start', maxw=440, tag='ctr:c%d' % i)
lc.text(MX + 16, BY0 + 138, '『无条件』的反面就写在第二个括号里：进池子就可能挑不中。', 8.8,
        C_MUTE, 'start', maxw=640, tag='ctr:n')

lc.rect(774, BY0, 686, 150, C_KV_F, C_KV_S, rx=10, sw=1.4)
lc.text(774 + 16, BY0 + 26, 'config 口径（V4-Flash）：切得有多细', 11.5, C_TXT, 'start', True,
        maxw=460, tag='cfg:t')
CFG = [('路由专家数 / 每 token 挑', '256 挑 6 ＋ 1 个共享专家', ''),
       ('单专家中间维', '2048 / hidden 4096 = 0.500000', '只有 hidden 的一半'),
       ('对照常规稠密 FFN', '4 × hidden = 16384 → 只有它的 8.000000 分之一', '本书自算注解')]
for i, (a, b, c) in enumerate(CFG):
    ty = BY0 + 56 + i * 28
    lc.text(774 + 16, ty, a, 9.5, C_TXT, 'start', True, tag='cfg:a%d' % i)
    lc.text(774 + 230, ty, b, 9.5, C_KV_DEEP, 'start', True, maxw=330, tag='cfg:b%d' % i)
    if c:
        lc.text(774 + 570, ty, c, 8.8, C_MUTE, 'start', maxw=180, tag='cfg:c%d' % i)
lc.text(774 + 16, BY0 + 138, '代价很直白：每个 token 都要付这个共享专家的算力，'
                             '不管它需不需要。', 8.8, '#155e75', 'start', maxw=660, tag='cfg:n')

# ---------------- 页脚 ----------------
lc.text(MX, 812, '数字口径：玩具 N = 4、K = 2、m = 2、d_ff = 8、K_s = 1（玩具 = 8 路由专家 + 1 共享专家）；'
                 'config 口径 256 挑 6 + 1 共享、moe_intermediate_size = 2048。', 8.5, C_MUTE,
        'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 830, '依据 DeepSeekMoE 论文 §3.1-§3.2 的 Eq.(3)-(11)（arXiv:2401.06066）；'
                 '算力与组合数由本章驱动脚本实跑。', 8.5, C_MUTE, 'start', maxw=BXR - MX,
        tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-moe-axes.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
