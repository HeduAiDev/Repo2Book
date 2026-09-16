#!/usr/bin/env python3
"""ch28 机制图 · 低秩瓶颈一分为二（ch28-fig-latent-fork）

claim：一次降维两处用：c^Q 往左接索引器的 q（n_h^I × c^I = 64 × 128），往右接主注意力的 q
（n_h × c = n_h × 512）——索引器不是额外的投影链，它搭在主注意力本来就有的低秩瓶颈上，
所以只多一份小头缓存、不多一条降维。

numbers（逐字取自 explainer figure-spec）：
  同一个 c^Q 分叉：索引器 q → 64 × 128（config: index_n_heads / index_head_dim）；
        主注意力 q → n_h × 512（config: head_dim）
  玩具同构缩小：n_h^I = 2、c^I = 2、n_h = 2、c = 4 ⇒ q 形状 (2, 4)、kv 形状 (3, 4)、
        注意力权重每行和 = [1.0, 1.0]
  落地：升维矩阵吃的是主注意力那条低秩潜向量（先归一化、之后才分叉）

配色走 book/cartography/l0_common.py 的角色常量（KV 青 = 压缩缓存侧，GPU 绿 = 执行臂）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 780
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_SOFT_F, C_SOFT_S = '#f8fafc', '#cbd5e1'
C_ACC_S = '#c2410c'

EXTRA_DEFS = ('<defs>'
              '<marker id="kv" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV_S}"/></marker>'
              '<marker id="gn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_GPU_S}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '一次降维，两处用：索引器搭在主注意力的低秩瓶颈上', 17, C_TXT, 'start', True,
        maxw=620, tag='title')
lc.text(MX, 56, 'hidden 先压成一条低秩潜向量 c^Q，再由它分出两个 q——一个给「选谁」的索引器，'
                '一个给「算什么」的主注意力', 10, C_MUTE, 'start', maxw=1080, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』压缩段的低秩瓶颈那一格', 9,
        C_MUTE, 'end', maxw=460, tag='l0:a')
lc.text(BXR, 64, '——它是打分那张图的上游（那张讲怎么打分，这张讲打分用的 q 从哪来）', 9,
        C_MUTE, 'end', maxw=470, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_GPU_F, C_GPU_S, '这一层的隐状态'),
                  (C_KV_F, C_KV_S, '低秩潜向量 c^Q'),
                  (C_SOFT_F, C_SOFT_S, '两个下游的 q（数学符号，不是新张量）')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=280, tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 18

CX = 750  # 中轴

# ---------------- ① 隐状态 ----------------
lc.rect(CX - 380, 140, 760, 56, C_GPU_F, C_GPU_S, rx=8, sw=1.6)
lc.text(CX - 366, 166, '这一层的隐状态（n 个 token × d 维）', 11.5, '#166534', 'start', True,
        maxw=740, tag='hd:t')
lc.text(CX - 366, 186, '索引器与主注意力在这一层要看的，是同一份 hidden', 9.0, '#334155',
        'start', maxw=740, tag='hd:s')

# ---------------- ② 降维 ----------------
lc.parrow([(CX, 196), (CX, 222)], '#64748b', 2.2, marker='gn')
lc.rect(CX - 200, 224, 400, 58, '#ffffff', C_GPU_S, rx=8, sw=1.6)
lc.text(CX, 248, '一次降维：W^DQ（d → d_c）', 11.5, C_GPU_S, 'middle', True, maxw=380,
        tag='dq:t')
lc.text(CX, 270, '整个低秩瓶颈只此一条', 9.0, C_MUTE, 'middle', maxw=380, tag='dq:s')

# ---------------- ③ 低秩潜向量 ----------------
lc.parrow([(CX, 282), (CX, 308)], C_KV_S, 2.4, marker='kv')
lc.rect(CX - 300, 310, 600, 58, C_KV_F, C_KV_S, rx=8, sw=1.8)
lc.text(CX, 334, 'c^Q = hidden · W^DQ（低秩潜向量，d_c 维）', 12.5, C_KV_DEEP, 'middle', True,
        maxw=580, tag='cq:t')
lc.text(CX, 356, '一次降维、两处用：两个 q 都从它出发', 9.2, '#334155', 'middle', maxw=580,
        tag='cq:s')

# ---------------- ④ 分叉 ----------------
FX_LEFT, FX_RIGHT = 360, 1140
FORK_Y = 392
lc.parrow([(CX, 368), (CX, FORK_Y), (FX_LEFT, FORK_Y), (FX_LEFT, 426)], C_KV_S, 2.2,
          marker='kv')
lc.parrow([(CX, 368), (CX, FORK_Y), (FX_RIGHT, FORK_Y), (FX_RIGHT, 426)], C_KV_S, 2.2,
          marker='kv')

BRANCH = [
    (80, 560, '索引器 q = c^Q · W^IUQ', '选谁',
     ['形状：n_h^I × c^I = 64 × 128（config 口径）',
      '给每个压缩块打分，只挑分高的那 512 块',
      '它的头维只有 128——索引器之所以便宜，就便宜在这里'],
     C_KV_DEEP),
    (860, 560, '主注意力 q = c^Q · W^UQ', '算什么',
     ['形状：n_h × c = n_h × 512（config 口径）',
      '算出真正的注意力输出，c 维一条条目同时当 K 与 V',
      '压缩条目在头之间共享，所以头数只加在 query 侧'],
     C_KV_DEEP),
]
for bx, bw, t1, badge, rows, col in BRANCH:
    lc.rect(bx, 428, bw, 150, '#ffffff', C_KV_S, rx=10, sw=1.5)
    lc.text(bx + 16, 456, t1, 12.5, col, 'start', True, maxw=bw - 120, tag='br:t')
    lc.rect(bx + bw - 84, 438, 68, 24, '#ecfeff', C_KV_S, rx=6, sw=1.2)
    lc.text(bx + bw - 50, 455, badge, 10, C_KV_DEEP, 'middle', maxw=64, tag='br:bg')
    for i, s in enumerate(rows):
        lc.text(bx + 16, 486 + i * 26, s, 9.4, '#334155' if i < 2 else C_MUTE, 'start',
                maxw=bw - 32, tag='br:r')

# ---------------- 底：为什么这是「一条链」 ----------------
BY0 = 600
lc.rect(MX, BY0, 900 - MX, 132, C_SOFT_F, C_SOFT_S, rx=10, sw=1.3)
lc.text(MX + 16, BY0 + 28, '所以：索引器不是额外的投影链', 11.5, C_TXT, 'start', True,
        maxw=400, tag='bt:t')
B1 = [('它搭在哪', '主注意力本来就有的低秩瓶颈上——降维这一步两边共用，不新增'),
      ('它多花什么', '只多一份小头缓存（每条 128 维的小头，不是主头那 512 维）'),
      ('读法', '把两个 q 看成同一条链的两个出口，而不是两条并排的链')]
for i, (a, b) in enumerate(B1):
    ty = BY0 + 56 + i * 26
    lc.text(MX + 16, ty, a, 9.4, C_KV_DEEP, 'start', True, maxw=100, tag='bt:a%d' % i)
    lc.text(MX + 104, ty, b, 9.4, '#334155', 'start', maxw=740, tag='bt:b%d' % i)

lc.rect(920, BY0, BXR - 920, 132, '#fff7ed', C_ACC_S, rx=10, sw=1.4)
lc.text(936, BY0 + 28, '玩具同构缩小（本章算例的口径）', 11.5, C_ACC_S, 'start', True, maxw=440,
        tag='toy:t')
TOY = [('n_h^I = 2、c^I = 2、n_h = 2、c = 4',
        'q 形状 (2, 4)：2 个头 × 4 维'),
       ('kv 形状 (3, 4)',
        '3 条压缩条目 × 4 维（KV 轴上没有头）'),
       ('注意力权重每行和 = [1.0, 1.0]',
        '两个头的权重各自归一')]
for i, (a, b) in enumerate(TOY):
    ty = BY0 + 56 + i * 26
    lc.text(936, ty, a, 9.4, C_ACC_S, 'start', True, maxw=280, tag='toy:a%d' % i)
    lc.text(1230, ty, b, 9.0, '#334155', 'start', maxw=220, tag='toy:b%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 756, '数字口径：config 口径 n_h^I = 64、c^I = 128、c = 512（V4-Flash config 直抓）；'
                 '玩具口径 n_h^I = 2、c^I = 2、n_h = 2、c = 4。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:1')
lc.text(MX, 774, '依据 DeepSeek-V4 技术报告 §2.3.1 的 Eq.(13)(14)(18)（arXiv:2606.19348）；'
                 '落地形态照 vLLM v0.27.1 的注意力实现与官方参考实现（先归一化、后分叉）。',
        8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-latent-fork.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
