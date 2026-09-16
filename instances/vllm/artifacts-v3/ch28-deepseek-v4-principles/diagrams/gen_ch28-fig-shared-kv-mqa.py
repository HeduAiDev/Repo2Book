#!/usr/bin/env python3
"""ch28 机制图 · 共享 KV 的 MQA：一条压缩条目同时当 K 和 V（ch28-fig-shared-kv-mqa）

claim：一条压缩条目同时当 K 和 V（同一个张量传两次）、所有 query 头共享它，
且它来自与索引器 q 共用的那份低秩潜向量 c^Q。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  q (2, 4)、kv (3, 4)：KV 轴上没有 head 维（对比 MHA 会是 2×3×4）
  权重（2 头 × 3 条）= [[0.465836, 0.362793, 0.171371], [0.307196, 0.186324, 0.50648]]，每行和 [1.0, 1.0]
  o = [[0.647232, 1.103042, 0.828629, 0.705536], [0.400358, 1.120872, 0.49352, 1.199285]]（= weights @ kv）
  扰动 kv[0] += 1.0 → 2 个头同时变（8 个变化量：0.860147 … 0.132214）
  c^Q 分叉：索引器 q 64 × 128、主注意力 q n_h × 512

配色走 book/cartography/l0_common.py 的角色常量（KV 青 = 条目，GPU 绿 = 执行臂/query）。
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

EXTRA_DEFS = ('<defs>'
              '<marker id="cy" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV_DEEP}"/></marker>'
              '<marker id="dn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_GPU_S}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '一条条目，两个身份：它既是 K 又是 V', 17, C_TXT, 'start', True, maxw=620,
        tag='title')
lc.text(MX, 56, '压缩压到最后只剩一条 c 维向量——它想不当 K/V 都难；所有 query 头共用它，'
                '头数只能加在 query 那一侧', 10, C_MUTE, 'start', maxw=1000, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』的核心注意力段', 9, C_MUTE, 'end',
        maxw=430, tag='l0:a')
lc.text(BXR, 64, '——上游是索引器选出的 top-k 条目，下游是输出侧的反旋与分组输出投影', 9,
        C_MUTE, 'end', maxw=470, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_GPU_F, C_GPU_S, 'query（n_h = 2 个头）'), (C_KV_F, C_KV_S, '压缩条目（KV 青）'),
                  ('#f8fafc', C_MUTE, '被共享的那一份张量')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=250, tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 16

# ---------------- ① query 侧（两个头） ----------------
QX0, QY0, QW, QH = MX, 150, 340, 96
lc.rect(QX0, QY0, QW, QH, C_GPU_F, C_GPU_S, rx=8, sw=1.5)
lc.text(QX0 + 14, QY0 + 24, 'q：n_h = 2 个头 × c = 4 维', 11.5, C_TXT, 'start', True, maxw=310,
        tag='q:t')
lc.text(QX0 + 14, QY0 + 44, '来自同一份低秩潜向量 c^Q', 9, '#334155', 'start', maxw=310, tag='q:s')
lc.text(QX0 + 14, QY0 + 62, '索引器 q：64 头 × 128 维', 9, C_KV_DEEP, 'start', maxw=310,
        tag='q:l0')
lc.text(QX0 + 14, QY0 + 80, '主注意力 q：n_h 头 × 512 维（config 口径）', 9, C_KV_DEEP, 'start',
        maxw=310, tag='q:l1')

# ---------------- ② 注意力权重表（2 × 3） ----------------
TB_X0, TB_Y0 = MX, 296
lc.rect(TB_X0, TB_Y0, 340, 156, '#ffffff', C_GPU_S, rx=8, sw=1.3)
lc.text(TB_X0 + 14, TB_Y0 + 24, '注意力权重（2 头 × 3 条）', 10.5, C_TXT, 'start', True, maxw=310,
        tag='w:t')
WT = ['0.465836', '0.362793', '0.171371', '0.307196', '0.186324', '0.506480']
for i in range(6):
    r, c = divmod(i, 3)
    cxx = TB_X0 + 30 + c * 96
    cyy = TB_Y0 + 56 + r * 40
    lc.rect(cxx, cyy - 14, 86, 22, '#f8fafc', '#e2e8f0', rx=3, sw=1)
    lc.text(cxx + 43, cyy + 1, WT[i], 9.2, '#334155', 'middle', tag='w:%d' % i)
    if c == 0:
        lc.text(TB_X0 + 14, cyy + 1, '头 %d' % r, 9, C_TXT, 'start', tag='w:h%d' % r)
lc.text(TB_X0 + 14, TB_Y0 + 130, '每行和 = [1.0, 1.0]（softmax 沿 3 条条目归一）', 9, C_KV_DEEP,
        'start', maxw=310, tag='w:s')

# ---------------- ③ 共享的那份张量（K 与 V 同源） ----------------
KVX0, KVY0, KVW, KVH = 480, 296, 420, 156
lc.rect(KVX0, KVY0, KVW, KVH, C_KV_F, C_KV_S, rx=8, sw=1.7)
lc.text(KVX0 + 14, KVY0 + 26, '压缩条目 C^SprsComp：3 条 × 4 维（kv 张量）', 11, C_KV_DEEP,
        'start', True, maxw=390, tag='kv:t')
lc.text(KVX0 + 14, KVY0 + 48, '同一份张量传两次：位置 ① 当 key、位置 ② 当 value', 9,
        '#334155', 'start', maxw=390, tag='kv:s')
lc.text(KVX0 + 14, KVY0 + 66, 'KV 轴上没有 head 维（(3, 4) 而不是 2×3×4）', 9, '#334155',
        'start', maxw=390, tag='kv:s2')
KVV = ['0.647232', '1.103042', '0.828629', '0.705536']
lc.text(KVX0 + 20, KVY0 + 88, '输出 o（头 0 的 4 维）= weights @ kv', 8.6, C_KV_DEEP, 'start',
        maxw=380, tag='kv:oh')
for i, v in enumerate(KVV):
    lc.rect(KVX0 + 20 + i * 96, KVY0 + 96, 86, 24, '#ffffff', C_KV_S, rx=3, sw=1)
    lc.text(KVX0 + 63 + i * 96, KVY0 + 113, v, 9.2, C_KV_DEEP, 'middle', tag='kv:o%d' % i)
lc.text(KVX0 + 14, KVY0 + 138, 'o = weights @ kv：没有第二份 value 投影', 9.5, C_KV_DEEP, 'start',
        True, maxw=390, tag='kv:o')

# 两条箭头：同一个盒子 → K 与 V
lc.parrow([(QX0 + QW, QY0 + 40), (KVX0 + 140, QY0 + 40), (KVX0 + 140, KVY0 - 4)], C_GPU_S,
          2.2, marker='dn')
lc.text(KVX0 + 150, QY0 + 34, '① q 与条目打分得权重', 9, C_MUTE, 'start', maxw=250, tag='fl:a')
lc.parrow([(QX0 + QW, TB_Y0 + 78), (KVX0 - 4, KVY0 + 78)], C_KV_S, 2.2, marker='cy')

# ---------------- ④ 右侧读数 ----------------
RX0 = 940
lc.rect(RX0, 150, BXR - RX0, 302, '#ffffff', C_GPU_S, rx=10, sw=1.4)
lc.text(RX0 + 16, 174, '三个可观察后果（本节实测）', 11.5, C_TXT, 'start', True, maxw=320,
        tag='rd:t')
RD = [('① KV 轴没有 head 维', 'kv 形状 = (3, 4)', '而不是 (2, 3, 4)——条目在头之间共享'),
      ('② 没有第二份 value', 'o = weights @ kv', '同一个张量传两次，省掉整条 KV 轴的一倍宽度'),
      ('③ 动一条，所有头一起变', 'kv[0] += 1.0', '2 个头 × 4 维 = 8 个分量全非 0')]
for i, (a, b, c) in enumerate(RD):
    ty = 208 + i * 62
    lc.text(RX0 + 16, ty, a, 10, C_TXT, 'start', True, tag='rd:a%d' % i)
    lc.text(RX0 + 16, ty + 18, b, 10.5, C_KV_DEEP, 'start', True, tag='rd:b%d' % i)
    lc.text(RX0 + 16, ty + 35, c, 8.6, C_MUTE, 'start', maxw=470, tag='rd:c%d' % i)
lc.text(RX0 + 16, 400, '扰动后的 8 个变化量：0.860147, 1.102083, 0.779502, 0.389647,'
                       ' 0.753717, 0.850275, 0.721531, 0.132214', 8.6, '#155e75', 'start',
        maxw=490, tag='rd:p')

# ---------------- 底左：扰动实验 ----------------
PY0, PY1 = 476, 664
lc.rect(MX, PY0, 700, PY1 - PY0, '#f8fafc', '#cbd5e1', rx=10, sw=1.3)
lc.text(MX + 16, PY0 + 24, '动手试：把一条条目加 1.0，看谁受影响', 11.5, C_TXT, 'start', True,
        maxw=420, tag='pA:t')
lc.text(MX + 16, PY0 + 46, 'kv[0] += 1.0 之后，两个 query 头的输出同时改变——'
                           '没有哪一头能绕过它。', 9, '#334155', 'start', maxw=640, tag='pA:s')
DXY = [('头 0 的输出变化', '[0.860147, 1.102083, 0.779502, 0.389647]'),
       ('头 1 的输出变化', '[0.753717, 0.850275, 0.721531, 0.132214]')]
for i, (a, b) in enumerate(DXY):
    ty = PY0 + 78 + i * 26
    lc.text(MX + 16, ty, a, 9.2, C_TXT, 'start', True, tag='pA:a%d' % i)
    lc.text(MX + 160, ty, b, 9.2, C_KV_DEEP, 'start', tag='pA:b%d' % i)
lc.text(MX + 16, PY0 + 142, '对照 MHA：每个头有自己的 K/V，动一份不动另一份——'
                            '这正是『共享』与『不共享』的差别。', 9, C_MUTE, 'start', maxw=640,
        tag='pA:l0')
lc.text(MX + 16, PY1 - 14, '这一份被共享的张量就是压缩条目本身：它的宽度与头数 n_h 无关。',
        8.8, C_KV_DEEP, 'start', maxw=640, tag='pA:l1')

# ---------------- 底右：c^Q 一次降维两处用 ----------------
QX2 = 770
lc.rect(QX2, PY0, BXR - QX2, PY1 - PY0, C_KV_F, C_KV_S, rx=10, sw=1.4)
lc.text(QX2 + 16, PY0 + 24, '一次降维，两处用：c^Q 的复用人情', 11.5, C_TXT, 'start', True,
        maxw=420, tag='pB:t')
lc.text(QX2 + 16, PY0 + 46, '同一个低秩潜向量 c^Q = h · W^DQ 分叉两条路，是本章的隐藏主线。',
        9, C_MUTE, 'start', maxw=600, tag='pB:s')
BR = [('索引器 q', 'c^Q · W^IUQ → 64 × 128', '便宜的小队（索引头）'),
      ('主注意力 q', 'c^Q · W^UQ → n_h × 512', '真正上台的那一路')]
for i, (a, b, c) in enumerate(BR):
    ty = PY0 + 82 + i * 34
    lc.text(QX2 + 16, ty, a, 9.8, C_TXT, 'start', True, tag='pB:a%d' % i)
    lc.text(QX2 + 130, ty, b, 9.8, C_KV_DEEP, 'start', True, tag='pB:b%d' % i)
    lc.text(QX2 + 400, ty, c, 8.8, C_MUTE, 'start', maxw=280, tag='pB:c%d' % i)
lc.text(QX2 + 16, PY0 + 162, 'config 口径：索引器 64 头 × 128 维、主注意力 head_dim = 512。',
        8.8, C_MUTE, 'start', maxw=600, tag='pB:l0')
lc.text(QX2 + 16, PY1 - 14, '低秩降维这一次计算服务两个下游——这也是 KV 账里 584 B/条'
                            '能同时含 K 与 V 的原因。', 8.8, '#155e75', 'start', maxw=620,
        tag='pB:l1')

# ---------------- 页脚 ----------------
lc.text(MX, 700, '数字口径：玩具 q (2, 4)、kv (3, 4)，3 条压缩条目；权重与输出为单次矩阵乘实跑。',
        8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 718, '依据 DeepSeek-V4 技术报告 §2.3.1 的 Eq.(18)(19)（arXiv:2606.19348）；'
                 '扰动实验、共享性验证由本章驱动脚本实跑。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-shared-kv-mqa.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
