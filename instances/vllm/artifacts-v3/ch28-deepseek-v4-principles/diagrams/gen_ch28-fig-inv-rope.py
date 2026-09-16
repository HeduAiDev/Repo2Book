#!/usr/bin/env python3
"""ch28 机制图 · 输出侧的反向旋转（ch28-fig-inv-rope）

claim：正反两次旋转只差一个符号：正向偶数位 x·cos − partner·sin、反向 x·cos + partner·sin，
转回来逐位等于没转（max|差| = 0.00000000）——之所以必须转回来，是因为压缩条目带的是
**块级位置**（[0, 4, 8] 三个值），而 query 带的是自己的位置（0..11）。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  一对（位置 2、3 维）：x = [1.0, 2.0] → 正向 [-1.272233, -1.838865] → 反向 [1.0, 2.0]
  前 2 维（NoPE 段）原样不动：[9.0, 9.0] == [9.0, 9.0]（True）
  反旋后逐位回到原值：max|x'' − x| = 0.00000000
  条目的块级位置（m = 4、前 3 条）= [0, 4, 8] vs query 的位置 = [0..11]

配色走 book/cartography/l0_common.py 的角色常量（KV 青 = 压缩缓存侧）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 720
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_ACC_S = '#c2410c'
C_SOFT_F, C_SOFT_S = '#f8fafc', '#cbd5e1'

EXTRA_DEFS = ('<defs>'
              '<marker id="kv" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV_S}"/></marker>'
              '<marker id="ac" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_ACC_S}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '输出侧的反向旋转：转回来，就等于没转', 17, C_TXT, 'start', True, maxw=520,
        tag='title')
lc.text(MX, 56, '正反两次旋转只差 partner 项的一个符号；而之所以非转不可，是因为条目带的是'
                '块级位置、query 带的是自己的位置', 10, C_MUTE, 'start', maxw=1080, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』的位置处理那一格', 9, C_MUTE,
        'end', maxw=440, tag='l0:a')
lc.text(BXR, 64, '——它挂在 KV 轴合成那张图的出口侧：把条目的位置换成 query 的位置', 9,
        C_MUTE, 'end', maxw=470, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, dash, lab in ((C_KV_F, C_KV_S, False, '参与旋转的最后 64 维'),
                        ('#ffffff', C_KV_S, True, '前 448 维（NoPE 段）原样不动'),
                        ('#fff7ed', C_ACC_S, False, '转回来的核对数')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1, dash=dash)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=270, tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 18

# ================= 两格并排 =================
PY0, PY1 = 116, 452
PANELS = [
    (MX, 700, '正向旋转（query 与滑窗 KV 用）',
     [('偶数位', 'x_even · cosθ − x_odd · sinθ', 0),
      ('奇数位', 'x_odd · cosθ + x_even · sinθ', 0)],
     'x = [1.0, 2.0]', '[-1.272233, -1.838865]', C_KV_DEEP, C_KV_S),
    (760, BXR - 760, '反向旋转（输出侧按 −i 再转一次）',
     [('偶数位', 'x_even · cosθ + x_odd · sinθ', 1),
      ('奇数位', 'x_odd · cosθ − x_even · sinθ', 1)],
     '[-1.272233, -1.838865]', '[1.0, 2.0]', C_ACC_S, C_ACC_S),
]
for bx, bw, title, rules, xin, xout, col, stroke in PANELS:
    lc.rect(bx, PY0, bw, PY1 - PY0, '#ffffff', stroke, rx=10, sw=1.6)
    lc.text(bx + 20, PY0 + 30, title, 12.5, col, 'start', True, maxw=bw - 40, tag='pn:t')
    for i, (a, b, hl) in enumerate(rules):
        ty = PY0 + 62 + i * 30
        lc.text(bx + 20, ty, a, 10, C_TXT, 'start', True, maxw=90, tag='pn:a')
        lc.text(bx + 112, ty, b, 10, '#7c2d12' if hl else '#334155', 'start', maxw=bw - 132,
                tag='pn:b')
    # 数值对
    ny = PY0 + 148
    lc.text(bx + 20, ny, '一对（位置 2、3 维）：', 9.4, C_MUTE, 'start', maxw=200, tag='pn:nl')
    lc.text(bx + 20, ny + 26, xin, 11.5, '#334155', 'start', True, maxw=280, tag='pn:in')
    lc.parrow([(bx + 300, ny + 21), (bx + 356, ny + 21)], stroke, 2.2,
              marker='ac' if hl else 'kv')
    lc.text(bx + 372, ny + 26, xout, 11.5, col, 'start', True, maxw=280, tag='pn:out')
    # 面板脚注
    if hl == 0:
        lc.text(bx + 20, PY1 - 96, '只动最后 rope_dim = 64 维：', 9.2, C_KV_DEEP, 'start', True,
                maxw=300, tag='pn:f0')
        lc.text(bx + 20, PY1 - 74, '前面的 448 维是 NoPE 段，两次都不参与', 9.0, '#334155',
                'start', maxw=660, tag='pn:f1')
        lc.text(bx + 20, PY1 - 46, '前 2 维（本例的 NoPE 段）原样不动：[9.0, 9.0] == [9.0, 9.0]',
                9.0, '#334155', 'start', maxw=660, tag='pn:f2')
        lc.text(bx + 20, PY1 - 24, '旋转只发生在最后 rope_dim 维上，这是「部分 RoPE」的口径。', 8.8,
                C_MUTE, 'start', maxw=660, tag='pn:f3')
    else:
        lc.text(bx + 20, PY1 - 96, '反旋后逐位回到原值：max|x″ − x| = 0.00000000', 9.6, C_ACC_S,
                'start', True, maxw=660, tag='pn:g0')
        lc.text(bx + 20, PY1 - 72, '两个方向的公式只差 partner 项的符号——', 9.0, '#334155',
                'start', maxw=660, tag='pn:g1')
        lc.text(bx + 20, PY1 - 52, '所以「转过去再转回来」逐位复原（R(−θ)R(θ) = I）。', 9.0,
                '#334155', 'start', maxw=660, tag='pn:g2')
        lc.text(bx + 20, PY1 - 26, '一次反旋 = 把条目的位置换成 query 的位置再进输出投影。', 8.8,
                C_MUTE, 'start', maxw=660, tag='pn:g3')

# ================= 底：为什么必须转回来 =================
BY0 = 472
lc.rect(MX, BY0, BXR - MX, 176, C_SOFT_F, C_SOFT_S, rx=10, sw=1.3)
lc.text(MX + 16, BY0 + 28, '为什么必须转回来：两个位置口径对不上', 11.5, C_TXT, 'start', True,
        maxw=420, tag='b:t')
lc.text(MX + 16, BY0 + 56, '压缩条目带的是块级位置（一条条目代表整块，位置取「块起点」）：'
                           'm = 4、前 3 条 = [0, 4, 8]', 9.4, '#334155', 'start',
        maxw=680, tag='b:l1')
lc.text(MX + 16, BY0 + 80, 'query 带的是自己的位置：前 12 个 token = [0, 1, 2, 3, 4, 5, 6, 7, '
                           '8, 9, 10, 11]', 9.4, '#334155', 'start', maxw=680, tag='b:l2')
lc.text(MX + 16, BY0 + 106, '条目位置只有 3 个值、query 位置有 12 个值——不把两者换到同一个'
                            '位置口径，输出侧投影就在错位上做。', 9.4, C_ACC_S, 'start',
        maxw=680, tag='b:l3')
# 两根位置轴（同一刻度：46px / 位置）
AXY = BY0 + 136
AX0, ASTEP, AW = MX + 96, 46, 40
lc.text(MX + 16, AXY - 6, '条目位置', 9.0, C_KV_DEEP, 'start', True, tag='b:ax1')
for v in (0, 4, 8):
    lc.rect(AX0 + v * ASTEP, AXY - 20, AW, 20, C_KV_F, C_KV_S, rx=3, sw=1.1)
    lc.text(AX0 + v * ASTEP + AW / 2, AXY - 6, '%d' % v, 9.4, C_KV_DEEP, 'middle', tag='b:ep')
lc.text(MX + 16, AXY + 26, 'query 位置', 9.0, '#166534', 'start', True, tag='b:ax2')
for v in range(12):
    lc.rect(AX0 + v * ASTEP, AXY + 12, AW, 20, C_GPU_F, C_GPU_S, rx=3, sw=1.1)
    lc.text(AX0 + v * ASTEP + AW / 2, AXY + 27, '%d' % v, 8.6, '#166534', 'middle', maxw=AW,
            tag='b:qp')
lc.text(MX + 16, AXY + 48, '同一刻度下：条目位置只有 3 个值，query 位置有 12 个值。', 8.8,
        C_MUTE, 'start', maxw=760, tag='b:rng')

lc.rect(920, BY0, BXR - 920, 176, '#fff7ed', C_ACC_S, rx=10, sw=1.4)
lc.text(936, BY0 + 28, '对照表：同一对数、两个方向', 11.5, C_ACC_S, 'start', True, maxw=440,
        tag='c:t')
CP = [('x（原值）', '[1.0, 2.0]'),
      ('正向（− sin）', '[-1.272233, -1.838865]'),
      ('反向（＋ sin）', '[1.0, 2.0]'),
      ('两式差别', '只有 partner 项的符号反转'),
      ('核对', 'max|x″ − x| = 0.00000000')]
for i, (a, b) in enumerate(CP):
    ty = BY0 + 58 + i * 26
    lc.text(936, ty, a, 9.4, C_TXT, 'start', True, maxw=140, tag='c:a%d' % i)
    lc.text(1090, ty, b, 9.4, C_ACC_S if i in (1, 2, 4) else '#334155', 'start',
            maxw=340, tag='c:b%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 690, '数字口径：玩具 rope_dim = 4（真实 64）、theta = 10000、position = 3、m = 4；'
                 'config 口径 rope_head_dim = 64（只动最后 64 维）、前 448 维是 NoPE。',
        8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 708, '依据 DeepSeek-V4 技术报告 §2.3.3 的 RoPE −i 句（arXiv:2606.19348）；'
                 '正反两式与核对数值由本章算例实跑。', 8.5, C_MUTE, 'start', maxw=BXR - MX,
        tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-inv-rope.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
