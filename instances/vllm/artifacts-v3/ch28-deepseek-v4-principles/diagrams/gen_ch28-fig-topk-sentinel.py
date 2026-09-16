#!/usr/bin/env python3
"""ch28 机制图 · 定长表与哨兵（ch28-fig-topk-sentinel）

claim：top-k 的输出是一张**定长表**：候选不足就填 −1 哨兵（候选 2 个而 topk=5 ⇒
[1, 0, −1, −1, −1]）；短上下文里『全选』与哨兵是同一件事的两面（8 个 token、m=4 ⇒ 候选 2 ≤ 512，
缓冲 [0, 1, −1, −1]），而 1M 下候选 250000 条里只挑 512 条（压缩比 488.28125）。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  候选 2 个而 topk=5：选中 [1, 0, −1, −1, −1]
  因果阈值 8、topk=2：选中 [7, 4]；阈值收到 6 时同一个 topk=2 选中 [4, 1]
  短上下文快路径：max_seq_len=8 ⇒ 候选 2 ≤ 512，缓冲 [0, 1, −1, −1]；
        max_seq_len=4096 ⇒ 候选 1024 > 512，走真 top-k
  1M 上下文：候选 250000 条、每 query 实看 min(512, 候选) = 512 条，压缩比 250000 / 512 = 488.28125

配色走 book/cartography/l0_common.py 的角色常量（KV 青 = 压缩缓存侧）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 740
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_ACC_S = '#c2410c'
C_SOFT_F, C_SOFT_S = '#f8fafc', '#cbd5e1'
C_SENT_F, C_SENT_S = '#f1f5f9', '#cbd5e1'
C_PICK_F = '#cffafe'

EXTRA_DEFS = ('<defs>'
              '<marker id="kv" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV_S}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '「选择」的产物是一张定长表，不是一个列表', 17, C_TXT, 'start', True, maxw=520,
        tag='title')
lc.text(MX, 56, '分数高的块号按序写进去、剩下的格子统一填 −1——所以短上下文里「全选」与哨兵'
                '是同一件事的两面，而 1M 下才显出稀疏性', 10, C_MUTE, 'start', maxw=1120, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』选择段写给核心注意力的那张表', 9,
        C_MUTE, 'end', maxw=470, tag='l0:a')
lc.text(BXR, 64, '——候选墙给出定义域，这张表给出实际写进缓冲的内容', 9, C_MUTE, 'end',
        maxw=440, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_PICK_F, C_KV_S, '被选中的块号'), (C_SENT_F, C_SENT_S, '−1 哨兵位（空槽）'),
                  (C_SOFT_F, C_SOFT_S, '候选与分数')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=250, tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 18

# ================= 左：阈值怎么进 top-k =================
AX0, AX1 = MX, 760
lc.rect(AX0, 112, AX1 - AX0, 300, '#ffffff', C_KV_S, rx=10, sw=1.5)
lc.text(AX0 + 16, 140, '① 因果约束怎么进 top-k：阈值就是候选墙', 11.5, C_KV_DEEP, 'start',
        True, maxw=560, tag='a:t')
lc.text(AX0 + 16, 160, '8 个候选块的分数 I（越界/未来位置一律置 −∞，不参与排序）', 9.0,
        C_MUTE, 'start', maxw=680, tag='a:s')

SCORES = ['0.5', '2.0', '1.0', '0.0', '3.0', '0.2', '0.1', '4.0']
BX0, BW, BGAP = AX0 + 20, 82, 4
for i, sc in enumerate(SCORES):
    x = BX0 + i * (BW + BGAP)
    lc.rect(x, 176, BW, 50, C_SOFT_F, C_SOFT_S, rx=5, sw=1.2)
    lc.text(x + BW / 2, 196, '块 %d' % i, 9.0, C_TXT, 'middle', maxw=BW + 4, tag='a:b%d' % i)
    lc.text(x + BW / 2, 216, sc, 10.5, C_KV_DEEP, 'middle', True, maxw=BW + 4, tag='a:v%d' % i)
THR = [('因果阈值 8（8 个块都够）：topk = 2', '选中 [7, 4]'),
       ('因果阈值 6（只留前 6 块）：同一个 topk = 2', '选中 [4, 1]')]
for i, (a, b) in enumerate(THR):
    ty = 256 + i * 36
    lc.rect(AX0 + 20, ty - 20, AX1 - AX0 - 40, 30, '#fff7ed' if i == 0 else '#f8fafc',
            C_ACC_S if i == 0 else C_SOFT_S, rx=5, sw=1.1)
    lc.text(AX0 + 32, ty, a, 9.4, '#334155', 'start', maxw=440, tag='a:t%d' % i)
    lc.text(AX0 + 500, ty, b, 10, C_ACC_S if i == 0 else C_KV_DEEP, 'start', True, maxw=200,
            tag='a:r%d' % i)
lc.text(AX0 + 20, 336, '阈值一收，同一份分数排出来的前二就换了人（[7, 4] → [4, 1]）——'
                       '墙在哪，排序就只在墙内进行。', 8.8, C_MUTE, 'start', maxw=680, tag='a:n1')
lc.text(AX0 + 20, 356, '两个都只是玩具口径的阈值；真实一层里阈值为「已完成的块数」。', 8.8,
        C_MUTE, 'start', maxw=680, tag='a:n2')
lc.text(AX0 + 20, 380, '越界位置置 −∞ + 候选不足填哨兵，是同一个机制的两面。', 8.8, C_KV_DEEP,
        'start', maxw=680, tag='a:n3')

# ================= 右：定长表 =================
BX0R = 800
lc.rect(BX0R, 112, BXR - BX0R, 300, C_SOFT_F, C_SOFT_S, rx=10, sw=1.3)
lc.text(BX0R + 16, 140, '② 输出：一张定长的表', 11.5, C_TXT, 'start', True, maxw=440, tag='b:t')
lc.text(BX0R + 16, 160, '选中项按分数排好填入，其余槽位一律 −1 哨兵', 9.0, C_MUTE, 'start',
        maxw=600, tag='b:s')

TABLES = [('候选只有 2 个而 topk = 5', [1, 0, -1, -1, -1], 200, 64, '后 3 个位置填哨兵'),
          ('短上下文（8 个 token、topk = 4）', [0, 1, -1, -1], 298, 44, '后 2 个位置填哨兵')]
for lab, cells, cy, cw, note in TABLES:
    lc.text(BX0R + 20, cy - 12, lab, 9.4, '#334155', 'start', True, maxw=560, tag='b:l')
    for i, v in enumerate(cells):
        x = BX0R + 20 + i * (cw + 6)
        sent = (v == -1)
        lc.rect(x, cy, cw, 38, C_SENT_F if sent else C_PICK_F, C_SENT_S if sent else C_KV_S,
                rx=5, sw=1.2)
        lc.text(x + cw / 2, cy + 25, '−1' if sent else str(v), 12,
                C_MUTE if sent else C_KV_DEEP, 'middle', True, maxw=cw + 4, tag='b:c')
    lc.text(BX0R + 20, cy + 54, note, 8.6, C_MUTE, 'start', maxw=260, tag='b:note')
lc.text(BX0R + 16, 368, '候选不足时排序无意义 → 全选与哨兵是同一件事的两面；'
                        '表的形状不随候选数变，变的是「填了几个号」。', 8.8, C_KV_DEEP, 'start',
        maxw=620, tag='b:n1')
lc.text(BX0R + 16, 392, '同一张表在 1M 下填的是 top-512 的结果——那才是真正在挑的时候。', 8.8,
        C_MUTE, 'start', maxw=620, tag='b:n2')

# ================= 下：三个规模 =================
CY0 = 428
lc.rect(MX, CY0, BXR - MX, 220, '#ffffff', C_KV_S, rx=10, sw=1.5)
lc.text(MX + 16, CY0 + 28, '③ 三个规模并排看：够不够挑', 11.5, C_KV_DEEP, 'start', True,
        maxw=400, tag='c:t')
HDR = [(MX + 20, 210, '规模'), (MX + 240, 210, '候选 = 规模 // 4'),
       (MX + 460, 300, '对 topk = 512 的判定'), (MX + 770, 640, '缓冲长什么样')]
lc.rect(MX + 16, CY0 + 42, BXR - MX - 32, 26, '#e2e8f0', '#94a3b8', rx=4, sw=1.1)
for x, w, lab in HDR:
    lc.text(x, CY0 + 60, lab, 9.2, '#334155', 'start', True, maxw=w, tag='c:h')
CROWS = [('max_seq_len = 8', '8 // 4 = 2', '2 ≤ 512 → 全选', '[0, 1, −1, −1]'),
         ('max_seq_len = 2048', '2048 // 4 = 512', '512 ≤ 512 → 全选', '[0, 1, 2, 3]'),
         ('max_seq_len = 4096', '4096 // 4 = 1024', '1024 > 512 → 走真 top-k', '排序后取前 512'),
         ('1M 上下文', '1000000 // 4 = 250000', '250000 > 512 → 挑 512 条',
          '压缩比 250000 / 512 = 488.28125')]
for i, (a, b, c, d) in enumerate(CROWS):
    ry = CY0 + 76 + i * 30
    lc.rect(MX + 16, ry, BXR - MX - 32, 27, '#f8fafc' if i % 2 else '#ffffff', C_SOFT_S,
            rx=3, sw=1.0)
    lc.text(MX + 20, ry + 19, a, 9.4, C_TXT, 'start', True, maxw=210, tag='c:a%d' % i)
    lc.text(MX + 240, ry + 19, b, 9.4, '#334155', 'start', maxw=210, tag='c:b%d' % i)
    lc.text(MX + 460, ry + 19, c, 9.4, C_ACC_S if i >= 2 else C_KV_DEEP, 'start', True,
            maxw=310, tag='c:c%d' % i)
    lc.text(MX + 770, ry + 19, d, 9.4, C_KV_DEEP if i < 3 else C_ACC_S, 'start', True,
            maxw=620, tag='c:d%d' % i)
lc.text(MX + 16, CY0 + 206, '结论：短上下文里排序是纯开销（候选全在表里），1M 下同一张表才真正'
                            '承担「从 250000 条里挑 512 条」的稀疏化。', 8.8, C_MUTE, 'start',
        maxw=BXR - MX - 32, tag='c:n')

# ---------------- 页脚 ----------------
lc.text(MX, 678, '数字口径：玩具分数 I = [0.5, 2.0, 1.0, 0.0, 3.0, 0.2, 0.1, 4.0]、m = 4、'
                 'topk 按各例标注；config 口径 topk = index_topk = 512。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:1')
lc.text(MX, 696, '依据 DeepSeek-V4 技术报告 Eq.(17)（arXiv:2606.19348）；'
                 '阈值、哨兵与快路径的逐行数值由本章算例实跑。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:2')
lc.text(MX, 714, '读法：先把 ① 的墙与 ② 的表对上（墙在哪、表就填到哪），再看 ③ 三个规模'
                 '在哪一行开始「够挑」。', 8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:3')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-topk-sentinel.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
