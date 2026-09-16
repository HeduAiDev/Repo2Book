#!/usr/bin/env python3
"""ch28 机制图 · HCA 层的「选择」就是按序全填（ch28-fig-hca-noindexer）

claim：HCA 层的『选择』就是按序全填：填进去的永远是 0..已完成块数−1，最后几格常年是 −1 哨兵
——这一层根本没有索引器对象，没有打分、没有排序。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  缓冲区（宽 8）：pos=255 → [0, 1, −1, −1, −1, −1, −1, −1]；pos=256 → 同上；
        pos=383 → [0, 1, 2, −1, −1, −1, −1, −1]；pos=1000 → [0, 1, 2, 3, 4, 5, 6, −1]
  已完成块数 = (pos+1) // 128：pos=255 与 256 都是 2 条、pos=383 是 3 条、pos=1000 是 7 条

配色走 book/cartography/l0_common.py 的角色常量（GPU 绿 = 执行臂）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 650
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_GPU_S = lc.C_GPU_S
C_ACC_S = '#c2410c'
C_SOFT_F, C_SOFT_S = '#f8fafc', '#cbd5e1'
C_FILL_F = '#dcfce7'
C_SENT_F, C_SENT_S = '#f1f5f9', '#cbd5e1'

# ---------------- 标题区 ----------------
lc.text(MX, 32, 'HCA 层的「选择」：按序全填，剩下的格子填哨兵', 17, C_TXT, 'start', True,
        maxw=580, tag='title')
lc.text(MX, 56, '填进去的永远是 0 到「已完成块数−1」——这一层没有索引器对象，没有打分、'
                '也没有排序', 10, C_MUTE, 'start', maxw=1000, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』选择段「不挑的落点」那一格', 9,
        C_MUTE, 'end', maxw=470, tag='l0:a')
lc.text(BXR, 64, '——账讲完之后的收尾：看这张表实际怎么写进缓冲', 9, C_MUTE, 'end', maxw=430,
        tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_FILL_F, C_GPU_S, '按序填进去的块号'),
                  (C_SENT_F, C_SENT_S, '−1 哨兵位（空槽）'),
                  (C_SOFT_F, C_SOFT_S, '缓冲的槽位')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=250, tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 18

# ================= 表 =================
LX0, LX1 = MX, 330
BX0, BX1 = 350, 1440
CW = (BX1 - BX0 - 7 * 8) / 8
HDR_Y = 104
lc.text(LX0, HDR_Y + 16, '位置 pos ／ 已完成块数 = (pos+1) // 128', 9.4, C_MUTE, 'start',
        maxw=290, tag='hd:l')
lc.text(BX0, HDR_Y + 16, '定长缓冲（宽 8）：填进去的块号 ／ 剩下的 −1 哨兵', 9.4, C_MUTE,
        'start', maxw=900, tag='hd:r')

ROWS = [(255, 2, [0, 1, -1, -1, -1, -1, -1, -1], '(255+1) // 128 = 2'),
        (256, 2, [0, 1, -1, -1, -1, -1, -1, -1], '(256+1) // 128 = 2'),
        (383, 3, [0, 1, 2, -1, -1, -1, -1, -1], '(383+1) // 128 = 3'),
        (1000, 7, [0, 1, 2, 3, 4, 5, 6, -1], '(1000+1) // 128 = 7')]
RY0, RH, RGAP = 134, 62, 22
for i, (pos, ncomp, cells, formula) in enumerate(ROWS):
    ry = RY0 + i * (RH + RGAP)
    lc.rect(LX0, ry, LX1 - LX0, RH, C_SOFT_F, C_SOFT_S, rx=6, sw=1.2)
    lc.text(LX0 + 12, ry + 26, 'pos = %d' % pos, 11.5, C_TXT, 'start', True, maxw=270, tag='rw:a')
    lc.text(LX0 + 12, ry + 48, formula + ' → %d 条已完成' % ncomp, 9.0, C_MUTE, 'start',
            maxw=270, tag='rw:b')
    for j, v in enumerate(cells):
        x = BX0 + j * (CW + 8)
        sent = (v == -1)
        lc.rect(x, ry, CW, RH, C_SENT_F if sent else C_FILL_F, C_SENT_S if sent else C_GPU_S,
                rx=6, sw=1.3)
        lc.text(x + CW / 2, ry + 38, '−1' if sent else str(v), 14,
                C_MUTE if sent else '#166534', 'middle', True, maxw=CW + 4, tag='cl')
    # 第 ncomp 个之后全是哨兵：在哨兵段中间那一格的格心下方标一句
    if ncomp < 8:
        mid = (ncomp + 7) // 2
        lc.text(BX0 + mid * (CW + 8) + CW / 2, ry + RH + 16,
                '这 %d 格全是哨兵' % (8 - ncomp), 8.4, C_MUTE, 'middle', maxw=CW + 4,
                tag='cl:note')

# ================= 底：三句话 =================
BY0 = 478
lc.rect(MX, BY0, BXR - MX, 118, '#ffffff', C_KV_S, rx=10, sw=1.4)
lc.text(MX + 16, BY0 + 28, '读这张表要带走的三件事', 11.5, C_KV_DEEP, 'start', True, maxw=320,
        tag='b:t')
B = [('① 填的规律', '永远是从 0 开始的连续号：已完成几块就填几个号',
      '没有打分、没有排序——「选择」这一步在这一层根本不存在'),
     ('② 哨兵是常态', 'pos = 255 与 pos = 256 都只填了 2 格，最后 6 格常年是 −1',
      '1M 上下文（pos = 999999）也不过填了 7812 格'),
     ('③ 与 CSA 层的对照', '同一张表，CSA 层填的是 top-512 的结果',
      '短上下文里两档看起来一样，正是因为候选少时 CSA 也是「顺序全填 + 哨兵」')]
for i, (a, b, c) in enumerate(B):
    ty = BY0 + 56 + i * 24
    lc.text(MX + 16, ty, a, 9.4, C_KV_DEEP, 'start', True, maxw=180, tag='b:a%d' % i)
    lc.text(MX + 180, ty, b, 9.4, '#334155', 'start', maxw=560, tag='b:b%d' % i)
    lc.text(MX + 760, ty, c, 8.8, C_MUTE, 'start', maxw=660, tag='b:c%d' % i)

# ---------------- 页脚 ----------------
lc.text(MX, 616, '数字口径：玩具缓冲区宽 8、压缩率 m′ = 128；真实一层的缓冲宽度由 '
                 'topk 上限决定。', 8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 634, '依据 DeepSeek-V4 技术报告 §2.3.2「HCA does not employ sparse attention」'
                 '（arXiv:2606.19348）；四行缓冲数值由本章算例实跑。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-hca-noindexer.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
