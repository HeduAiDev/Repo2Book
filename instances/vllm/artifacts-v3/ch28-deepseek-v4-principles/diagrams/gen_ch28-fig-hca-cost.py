#!/usr/bin/env python3
"""ch28 机制图 · 压到 1/128 之后「不挑」比「挑」划算（ch28-fig-hca-cost）

claim：压到 1/128 之后『全看不挑』更划算：HCA 每 query 看 7812 条、是 CSA 那 512 条的
15.257812 倍，但省掉整个打分器——单个索引头打分的乘加数（32000000）就是 HCA 多看那部分
（3737600）的 8.561644 倍，且还没算 64 个索引头、top-512 排序与那一份独立小头缓存
（179.000000 vs 4.562500 B/token）。

numbers（逐字取自 explainer figure-spec，provenance = 驱动脚本实测输出）：
  候选压缩块：CSA 层 1000000 // 4 = 250000 条；HCA 层 1000000 // 128 = 7812 条；
        每 query 实看 512（top-k 封顶）vs 7812（全看）
  HCA / CSA = 7812 / 512 = 15.257812 倍
  索引器打分乘加数：250000 × 128 = 32000000（每个索引头）／× 64 头 = 2048000000；
        HCA 多看的乘加数：(7812 − 512) × 512 = 3737600（每个主头）⇒ 8.561644 倍
  缓存账：CSA 层 (584 + 132) / 4 = 179.000000 B/token；HCA 层 584 / 128 = 4.562500 B/token

配色走 book/cartography/l0_common.py 的角色常量（KV 青 = 缓存账，GPU 绿 = 执行臂）。
坐标全部由常量与循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 820
MX, BXR = 40, 1460
C_MUTE, C_TXT = lc.C_MUTE, lc.C_TXT
C_KV_S, C_KV_F, C_KV_DEEP = lc.C_KV_S, lc.C_KV_F, '#155e75'
C_GPU_S, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_ACC_S = '#c2410c'
C_SOFT_F, C_SOFT_S = '#f8fafc', '#cbd5e1'
C_BAR_PICK = '#cffafe'
C_BAR_FULL = '#dcfce7'

EXTRA_DEFS = ('<defs>'
              '<marker id="gy" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '压到 1/128 之后：「不挑」比「挑」划算', 17, C_TXT, 'start', True, maxw=500,
        tag='title')
lc.text(MX, 56, 'HCA 每 query 多看 15.257812 倍条目，却省掉整个打分器——单个索引头打分的乘加数'
                '就已经是它多看那部分的 8.561644 倍', 10, C_MUTE, 'start', maxw=1080, tag='sub')
lc.text(BXR, 30, '本图的位置', 9.5, C_MUTE, 'end', True, tag='l0:t')
lc.text(BXR, 48, '全景架构图（L0）里『GPU 执行臂 · 模型层』两条注意力支路 HCA 那一档', 9,
        C_MUTE, 'end', maxw=460, tag='l0:a')
lc.text(BXR, 64, '——机制差在另一张图上，这一张只算「为什么可以不做选择」', 9, C_MUTE, 'end',
        maxw=450, tag='l0:b')

# ---------------- 图例 ----------------
lgx = MX
for f, s, lab in ((C_BAR_PICK, C_KV_S, 'CSA 层（牌子 4）：挑着看'),
                  (C_BAR_FULL, C_GPU_S, 'HCA 层（牌子 128）：一条不挑'),
                  ('#fff7ed', C_ACC_S, '两边的对账数字')):
    lc.rect(lgx, 78, 16, 11, f, s, rx=2, sw=1.1)
    lc.text(lgx + 22, 87, lab, 8.6, '#334155', 'start', maxw=280, tag='lg:%s' % lab[:4])
    lgx += 22 + 16 + lc.tw(lab, 8.6) + 18

# ================= 两栏 =================
PY0, PY1 = 118, 468
lc.rect(MX, PY0, 700 - MX, PY1 - PY0, '#ffffff', C_KV_S, rx=10, sw=1.5)
lc.text(MX + 16, PY0 + 28, 'CSA 层（牌子 4，m = 4）：先打分、再挑 512 条', 12.5, C_KV_DEEP,
        'start', True, maxw=520, tag='p1:t')
S1 = [(200, '候选压缩块', '1000000 // 4 = 250000 条'),
      (248, '① 索引器打分', '250000 × 128 = 32000000 次乘加（每个索引头）'),
      (296, '② top-512 排序', '对 250000 个分数做一次 radix top-k'),
      (344, '③ 核心注意力', '只算被挑中的 512 条（top-k 封顶）'),
      (400, '每 token 每层摊销', '(584 + 132) / 4 = 179.000000 B —— 多背一份小头缓存')]
for y, a, b in S1:
    lc.text(MX + 16, y, a, 9.8, C_KV_DEEP, 'start', True, maxw=170, tag='p1:a%d' % y)
    lc.text(MX + 190, y, b, 9.4, '#334155' if y < 400 else C_ACC_S, 'start', maxw=490,
            tag='p1:b%d' % y)

lc.rect(760, PY0, BXR - 760, PY1 - PY0, '#ffffff', C_GPU_S, rx=10, sw=1.5)
lc.text(776, PY0 + 28, 'HCA 层（牌子 128，m′ = 128）：一条不挑', 12.5, '#166534', 'start',
        True, maxw=520, tag='p2:t')
S2 = [(200, '候选压缩块', '1000000 // 128 = 7812 条'),
      (248, '① 没有打分器', '这一层根本没有索引器对象：不打分、不排序'),
      (296, '② 全部塞进表', '缓冲就是 0..已完成块数−1 的顺序全填'),
      (344, '③ 核心注意力', '老老实实算 7812 条（全看，无盲区）'),
      (400, '每 token 每层摊销', '584 / 128 = 4.562500 B —— 只有主压缩条目')]
for y, a, b in S2:
    lc.text(776, y, a, 9.8, '#166534', 'start', True, maxw=170, tag='p2:a%d' % y)
    lc.text(950, y, b, 9.4, '#334155' if y < 400 else C_ACC_S, 'start', maxw=496,
            tag='p2:b%d' % y)

# 中缝
lc.parrow([(750, PY0 + 20), (750, PY1 - 20)], '#cbd5e1', 2.0, marker=None)

# ================= 下：三笔对账 =================
CY0 = 492
lc.rect(MX, CY0, BXR - MX, 112, '#fff7ed', C_ACC_S, rx=10, sw=1.4)
lc.text(MX + 16, CY0 + 26, '三笔对账（config 口径：1M 上下文、m = 4、m′ = 128、topk = 512）',
        11.5, C_ACC_S, 'start', True, maxw=560, tag='c:t')
CP = [('① 主注意力条数', '7812 / 512 = 15.257812 倍（HCA 确实更贵）'),
      ('② 打分器的乘加数', '32000000 / 3737600 = 8.561644 倍（**单个**索引头 vs **单个**主头）'),
      ('③ 缓存', '179.000000 vs 4.562500 B/token（多一份 132 B/条的小头缓存）')]
for i, (a, b) in enumerate(CP):
    ty = CY0 + 52 + i * 22
    lc.text(MX + 16, ty, a, 9.4, C_TXT, 'start', True, tag='c:a%d' % i)
    lc.text(MX + 190, ty, b.replace('**', ''), 9.4, '#334155', 'start', maxw=690,
            tag='c:b%d' % i)
lc.text(MX + 900, CY0 + 54, '那 64 个索引头合计 2048000000 次乘加，', 9.2, C_ACC_S, 'start',
        True, maxw=500, tag='c:n1')
lc.text(MX + 900, CY0 + 78, '再加排序与那份缓存——都还没算进这个倍数里。', 9.2, C_ACC_S, 'start',
        maxw=500, tag='c:n2')

# ================= 乘加数条 =================
DY0 = 620
lc.rect(MX, DY0, BXR - MX, 118, '#ffffff', C_KV_S, rx=10, sw=1.4)
lc.text(MX + 16, DY0 + 26, '把两边的乘加数并排画（每 query 口径，同一比例尺）', 11.5,
        C_KV_DEEP, 'start', True, maxw=520, tag='d:t')
BARS = [('索引器打分（每个索引头）', 32000000, C_BAR_PICK, C_KV_S),
        ('HCA 多看的那部分（每个主头）', 3737600, C_BAR_FULL, C_GPU_S)]
BX, BMAX, BSCALE = MX + 16, 32000000, 760 / 32000000
for i, (lab, val, fill, stroke) in enumerate(BARS):
    by = DY0 + 46 + i * 34
    lc.text(BX, by + 14, lab, 9.2, '#334155', 'start', maxw=280, tag='d:l%d' % i)
    bw = val * BSCALE
    lc.rect(BX + 290, by, bw, 22, fill, stroke, rx=3, sw=1.2)
    lc.text(BX + 290 + bw + 10, by + 15, '%d' % val, 9.4, stroke, 'start', True, maxw=140,
            tag='d:v%d' % i)
lc.text(BX + 290 + 32000000 * BSCALE + 10, DY0 + 76, '≈ 8.561644 倍', 9.6, C_ACC_S, 'start',
        True, maxw=200, tag='d:r')

# ---------------- 页脚 ----------------
lc.text(MX, 758, '数字口径：config 口径 1M 上下文、m = 4、m′ = 128、topk = 512、'
                 '索引器头数 64、头维 128、主头维 512；主注意力头数本包未取到，'
                 '故一律用「每个头」并排比较。', 8.5, C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 776, '依据 DeepSeek-V4 技术报告 §2.3.2 的 Eq.(20)-(26)（arXiv:2606.19348）；'
                 '候选数、乘加数与缓存账由本章算例实跑。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:2')
lc.text(MX, 794, '读法：左栏是「便宜地看」的代价清单，右栏是「不挑」的代价——'
                 '两根条一对比，就知道哪一边的账单更长。', 8.5, C_MUTE, 'start',
        maxw=BXR - MX, tag='ft:3')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch28-fig-hca-cost.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
