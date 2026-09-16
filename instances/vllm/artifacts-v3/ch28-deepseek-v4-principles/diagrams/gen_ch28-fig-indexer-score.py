#!/usr/bin/env python3
"""ch28 机制图 · 索引器打分：低秩 q + 逐头权重 + 压缩块上的 ReLU 内积（ch28-fig-indexer-score）

claim：索引器的打分是『低秩 q + 逐头标量权重 + 压缩块上的 ReLU 内积』：一次降维（c^Q）
两处用，ReLU 只让正相关推动选择，两个实现层缩放都是正标量因而改不了 top-k 次序。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  q^I = [[1.0, 0.0], [0.0, 1.0]]、K^IComp = [[2.0, 0.5], [1.0, 0.0], [−0.5, −2.0], [1.5, 0.25]]、w = [0.7, 0.3]
  逐头内积 头0 [2.0, 1.0, −0.5, 1.5] / 头1 [0.5, 0.0, −2.0, 0.25] → ReLU 后 −0.5 与 −2.0 变 0
  I = [1.55, 0.7, 0.0, 1.125]；top-2 = 块 [0, 3]
  去 ReLU 对照 = [1.55, 0.7, −0.95, 1.125]（块 2 会拿负分）
  两个缩放 0.707107（c^I^-0.5 与 n_h^I^-0.5）；缩放后 I = [0.775, 0.35, 0.0, 0.5625]，top-2 次序一致 = True
  短上下文快路径：候选 8//4 = 2、2048//4 = 512、4096//4 = 1024 对 topk 512 的三种判定
  config 口径：n_h^I = 64、c^I = 128、每 query 打分乘加数 = 250000 × 128 = 32000000（每头）

配色走 book/cartography/l0_common.py 的角色常量（GPU 绿=执行臂，KV 青=压缩块/条目，橙=分数）。
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
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_SCOR_F, C_SCOR_S = '#ffedd5', '#c2410c'

EXTRA_DEFS = ('<defs>'
              '<marker id="gu" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_GPU_S}"/></marker>'
              '<marker id="dn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV_DEEP}"/></marker>'
              '<marker id="wm" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_SCOR_S}"/></marker>'
              '<marker id="gy" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_MUTE}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '索引器：先用一支便宜的小队初筛 512 块', 17, C_TXT, 'start', True, maxw=620,
        tag='title')
lc.text(MX, 56, 'I = Σ_h w_h · ReLU(q_h · K^IComp_s)：每个索引头闻一遍所有压缩块，'
                '分数按各自的标量信誉加权，负相关一律归零', 10, C_MUTE, 'start', maxw=1050,
        tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』压缩段的选择器', 9, C_MUTE, 'end',
        maxw=430, tag='l0:a')
lc.text(BXR, 64, '——上游是压缩条目，下游是核心注意力', 9, C_MUTE, 'end', maxw=430, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_GPU_F, C_GPU_S, '主注意力 q（n_h 头）'), (C_KV_F, C_KV_S, '压缩块（KV 青）'),
                  (C_SCOR_F, C_SCOR_S, '内积分数 / 权重'),
                  ('#dcfce7', '#16a34a', '选中（top-k）')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=250, tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 16

# ---------------- ① 低秩潜向量 c^Q ----------------
CQ_X0, CQ_Y0, CQ_W, CQ_H = MX, 190, 300, 74
lc.rect(CQ_X0, CQ_Y0, CQ_W, CQ_H, '#ffffff', C_GPU_S, rx=8, sw=1.5)
lc.text(CQ_X0 + 16, CQ_Y0 + 24, 'c^Q：低秩潜向量', 12, C_TXT, 'start', True, maxw=200, tag='cq:t')
lc.text(CQ_X0 + 16, CQ_Y0 + 42, 'W^DQ：d → d_c（一次降维）', 9, C_MUTE, 'start', maxw=270,
        tag='cq:s')
lc.text(CQ_X0 + 16, CQ_Y0 + 60, '它向右接索引器 q、向上接主注意力 q', 9, C_KV_DEEP, 'start',
        maxw=270, tag='cq:n')

# 上分叉：主注意力 q
QA_X0, QA_Y0, QA_W, QA_H = MX, 118, 300, 46
lc.rect(QA_X0, QA_Y0, QA_W, QA_H, C_GPU_F, C_GPU_S, rx=8, sw=1.5)
lc.text(QA_X0 + 14, QA_Y0 + 20, '主注意力 q：n_h 头 × 512 维', 10.5, C_TXT, 'start', True,
        maxw=270, tag='qa:t')
lc.text(QA_X0 + 14, QA_Y0 + 36, 'W^UQ 升维（Eq.18）', 8.8, C_MUTE, 'start', maxw=270, tag='qa:s')
lc.parrow([(CQ_X0 + 60, CQ_Y0), (CQ_X0 + 60, QA_Y0 + QA_H)], C_GPU_S, 2.0, marker='gu')

# 右分叉：索引器 q
QI_X0, QI_Y0, QI_W, QI_H = 480, 190, 330, 74
lc.rect(QI_X0, QI_Y0, QI_W, QI_H, C_KV_F, C_KV_S, rx=8, sw=1.5)
lc.text(QI_X0 + 14, QI_Y0 + 22, '索引器 q：64 头 × 128 维', 11, C_TXT, 'start', True, maxw=300,
        tag='qi:t')
lc.text(QI_X0 + 14, QI_Y0 + 40, 'W^IUQ 升维（Eq.14）——同一份 c^Q', 8.8, C_MUTE, 'start',
        maxw=300, tag='qi:s')
lc.text(QI_X0 + 14, QI_Y0 + 58, '小头之所以便宜就便宜在这里（config 口径）', 8.8, C_KV_DEEP,
        'start', maxw=300, tag='qi:n')
lc.parrow([(CQ_X0 + CQ_W, CQ_Y0 + CQ_H / 2), (QI_X0 - 4, CQ_Y0 + CQ_H / 2)], C_KV_S, 2.2,
          marker='dn')

# ---------------- ② 逐头内积 + ReLU ----------------
TB_X0, TB_Y0 = 480, 300
lc.rect(TB_X0 - 8, TB_Y0 - 26, 520, 148, '#ffffff', C_KV_S, rx=8, sw=1.3)
lc.text(TB_X0, TB_Y0 - 8, '逐头内积 q_h · K^IComp_s（2 头 × 4 个候选块）', 9.5, C_TXT, 'start',
        True, maxw=480, tag='tb:t')
HEADS = [('头 0', '[2.0, 1.0, −0.5, 1.5]', '[2.0, 1.0, 0.0, 1.5]'),
         ('头 1', '[0.5, 0.0, −2.0, 0.25]', '[0.5, 0.0, 0.0, 0.25]')]
for i, (h, raw, rel) in enumerate(HEADS):
    ty = TB_Y0 + 16 + i * 30
    lc.text(TB_X0, ty, h, 9.5, C_TXT, 'start', True, tag='tb:h%d' % i)
    lc.text(TB_X0 + 50, ty, raw, 9.5, '#334155', 'start', tag='tb:r%d' % i)
    lc.seg(TB_X0 + 226, ty - 10, TB_X0 + 226, ty + 3, C_SCOR_S, 1.4, marker='wm')
    lc.text(TB_X0 + 236, ty, 'ReLU', 9, C_SCOR_S, 'start', True, tag='tb:k%d' % i)
    lc.text(TB_X0 + 286, ty, rel, 9.5, C_SCOR_S, 'start', True, tag='tb:v%d' % i)
lc.text(TB_X0, TB_Y0 + 90, 'w = [0.7, 0.3] → I = [1.55, 0.7, 0.0, 1.125]', 12, C_SCOR_S, 'start',
        True, maxw=480, tag='tb:i')
lc.text(TB_X0, TB_Y0 + 110, '手算核对：块 0 = 0.7×2.0 + 0.3×0.5 = 1.55；'
                            '块 2 两头都是负相关 → 拿 0 而不是 −0.95', 8.8, C_MUTE, 'start',
        maxw=490, tag='tb:n')

# ---------------- ③ 四个候选块的分数（top-2 高亮） ----------------
SC_X0, SC_Y0 = 1060, 150
lc.rect(SC_X0 - 20, SC_Y0 - 30, BXR - SC_X0 + 20, 262, '#ffffff', C_GPU_S, rx=10, sw=1.4)
lc.text(SC_X0, SC_Y0 - 10, '4 个候选块的分数（ReLU 之后）', 11, C_TXT, 'start', True, maxw=340,
        tag='sc:t')
SC = [('块 0', 1.55, '选中', True), ('块 1', 0.70, '落选', False),
      ('块 2', 0.00, '两头负相关 → 0', False), ('块 3', 1.125, '选中', True)]
BARX0, BARW = SC_X0 + 66, 230
for i, (lab, v, note, hit) in enumerate(SC):
    cy = SC_Y0 + 22 + i * 44
    f, s = ('#dcfce7', '#16a34a') if hit else (C_SCOR_F, C_SCOR_S)
    lc.text(SC_X0, cy + 5, lab, 10, C_TXT, 'start', True, tag='sc:l%d' % i)
    lc.rect(BARX0, cy - 9, max(3.0, v / 1.55 * BARW), 18, f, s, rx=3, sw=1.3)
    lc.text(BARX0 + max(v / 1.55 * BARW, 3) + 8, cy + 5, '%.3f' % v, 11, s, 'start', True,
            tag='sc:v%d' % i)
    lc.text(BARX0, cy + 24, note, 8.6, '#334155' if hit else C_MUTE, 'start', maxw=BARW + 120,
            tag='sc:n%d' % i)
lc.text(SC_X0, SC_Y0 + 202, 'top-2（Eq.17）= {块 0, 块 3}——只有这两块进核心注意力。',
        9, C_KV_DEEP, 'start', True, maxw=400, tag='sc:f')
lc.text(SC_X0, SC_Y0 + 220, '块 1 的 0.7 输给块 3 的 1.125。', 9, C_MUTE, 'start', maxw=400,
        tag='sc:f2')

# ---------------- 底左：两个实现层缩放 ----------------
PY0, PY1 = 440, 660
lc.rect(MX, PY0, 700, PY1 - PY0, '#f8fafc', '#cbd5e1', rx=10, sw=1.3)
lc.text(MX + 16, PY0 + 24, '两个实现层的缩放（论文没写）：正标量不改 top-k 次序', 11, C_TXT,
        'start', True, maxw=460, tag='pA:t')
ZP = [('softmax_scale', 'c^I^-0.5 = 0.707107', '把 q 除以正标量'),
      ('weights_scaling', 'n_h^I^-0.5 = 0.707107', '把权重乘上同一个正标量')]
for i, (a, b, c) in enumerate(ZP):
    ty = PY0 + 56 + i * 30
    lc.text(MX + 16, ty, a, 9.5, C_TXT, 'start', True, tag='pA:a%d' % i)
    lc.text(MX + 130, ty, b, 9.5, C_KV_DEEP, 'start', True, tag='pA:b%d' % i)
    lc.text(MX + 330, ty, c, 9, C_MUTE, 'start', maxw=350, tag='pA:c%d' % i)
lc.text(MX + 16, PY0 + 130, '带缩放的 I = [0.775, 0.35, 0.0, 0.5625] → 与不带缩放的 top-2 次序一致：True',
        10, '#155e75', 'start', True, maxw=640, tag='pA:l0')
lc.text(MX + 16, PY0 + 152, '一行代数：q 除以 s、权重乘上同一个 s → 每一项 w_h·ReLU(q_h·k) 逐项相等，'
                            'I 整体乘一个正常数。', 8.8, C_MUTE, 'start', maxw=660, tag='pA:l1')
lc.text(MX + 16, PY1 - 32, '所以这两个缩放是为数值/量化服务的实现细节——不改变哪几块被选中。',
        8.8, '#334155', 'start', maxw=660, tag='pA:l2')
lc.text(MX + 16, PY1 - 12, '（ReLU 那半边同理可验证：去 ReLU 时块 2 会拿到 −0.95，'
                           '规模上去后负分会累积成负分。）', 8.6, lc.C_FAINT, 'start', maxw=660,
        tag='pA:l3')

# ---------------- 底右：短上下文快路径 ----------------
QX0 = 770
lc.rect(QX0, PY0, BXR - QX0, PY1 - PY0, C_KV_F, C_KV_S, rx=10, sw=1.4)
lc.text(QX0 + 16, PY0 + 24, '短上下文里打分是纯开销：候选不够就直接全选', 11, C_TXT, 'start',
        True, maxw=460, tag='pB:t')
lc.text(QX0 + 16, PY0 + 46, '候选数 = 序列长度 // 压缩率；一旦候选 ≤ top-k，排序不改变结果。',
        8.8, C_MUTE, 'start', maxw=600, tag='pB:s')
FP = [('max_seq_len = 8', '8 // 4 = 2 ≤ 512', '→ True（全选）'),
      ('max_seq_len = 2048', '2048 // 4 = 512 ≤ 512', '→ True（刚好全选）'),
      ('max_seq_len = 4096', '4096 // 4 = 1024 > 512', '→ False（要走打分）')]
for i, (a, b, c) in enumerate(FP):
    ty = PY0 + 78 + i * 26
    lc.text(QX0 + 16, ty, a, 9.2, C_TXT, 'start', True, tag='pB:a%d' % i)
    lc.text(QX0 + 190, ty, b, 9.2, '#334155', 'start', tag='pB:b%d' % i)
    lc.text(QX0 + 400, ty, c, 9.2, C_KV_DEEP if i < 2 else C_SCOR_S, 'start', True, tag='pB:c%d' % i)
lc.text(QX0 + 16, PY0 + 172, '候选只有 2 个而 top-k = 5 时：缓冲 = [1, 0, −1, −1, −1]——'
                             '不足的位置填 −1 哨兵。', 8.8, '#334155', 'start', maxw=620,
        tag='pB:l0')
lc.text(QX0 + 16, PY1 - 32, 'config 口径：n_h^I = 64、c^I = 128、候选 250000 块 → 每 query 的'
                            '打分乘加 = 250000 × 128 = 32000000（每个索引头）。', 8.8, C_MUTE,
        'start', maxw=620, tag='pB:l1')
lc.text(QX0 + 16, PY1 - 12, '『全选』与『−1 哨兵』是同一件事的两面——短上下文里没有选择可言。',
        8.8, '#155e75', 'start', maxw=620, tag='pB:l2')

# ---------------- 页脚 ----------------
lc.text(MX, 700, '数字口径：玩具口径 n_h^I = 2、c^I = 2、4 个候选块、w = [0.7, 0.3]；'
                 'config 口径 n_h^I = 64、c^I = 128、候选 250000。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:1')
lc.text(MX, 718, '依据 DeepSeek-V4 技术报告 §2.3.1 的 Eq.(13)-(17)（arXiv:2606.19348）'
                 '与 DSA 的 ReLU 打分式（arXiv:2512.02556 Eq.(1)）；数值由本章驱动脚本实跑。',
        8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-indexer-score.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
