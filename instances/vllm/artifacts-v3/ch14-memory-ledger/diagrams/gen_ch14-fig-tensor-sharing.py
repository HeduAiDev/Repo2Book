#!/usr/bin/env python3
"""ch14 机制图 4 · 通用张量布局：每池每组各出一层（figure_spec ch14-fig-tensor-sharing，模板 layout）

放大自 L0 KV 账本列 → GPU 列的接缝（worker 半边分配）——本章 L2 章图拍片④
「一份账喂两侧」的 worker 侧张量布局展开。架构归属回指 L0/L2（§3.3）：指北小签。

claim：通用张量布局 = group_size 个内存池、每池由每组各出一层共享：
full.0/sw.0/sw.1 共一张、full.1/sw.2 共另一张——每组的块表不同、物理上却可共享，
因为一个 block_id 同一时刻只归一个组用。

数字全部取自 figure_spec.numbers（配套精简版 host 实跑）。坐标由常量/循环计算；
文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440
FUL_S, FUL_F = lc.C_API_S, lc.C_API_F   # full attention 层 = 蓝
SWA_S, SWA_F = lc.C_KV_S, lc.C_KV_F     # SWA 层 = 青
PHY_S = '#475569'                       # 物理张量 = 石板灰

# ---------------- 标题区 ----------------
lc.text(MX, 34, '通用张量布局：账本说 10 块、物理只有 2 张张量——每池由每组各出一层合租',
        16.5, lc.C_TXT, 'start', True, maxw=990, tag='title')
lc.text(MX, 58, 'num_blocks = available // page // group_size = 1310720 // 65536 // 2 = 10——三组三张私有块表各记各的门牌，房间（物理页）合租',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · KV 账本列 → GPU 列接缝 · L2 拍片④'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左：三组 × 两池 指派矩阵 ----------------
GX, GY0 = MX, 92
RH_LBL, CW_CELL, CH_CELL, CGAP = 150, 190, 52, 10
GH = 26 + 3 * (CH_CELL + CGAP) + 34
GW_ = 16 + RH_LBL + 2 * (CW_CELL + CGAP)
lc.rect(GX, GY0, GW_, GH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(GX + 14, GY0 + 20, '三组 × 两池——第 j 池由每组的第 j 层合租', 11, lc.C_TXT,
        'start', True, maxw=GW_ - 28, tag='grid:t')
col_x = [GX + 16 + RH_LBL + i * (CW_CELL + CGAP) for i in range(2)]
for i, t in enumerate(['池1 · 张量一', '池2 · 张量二']):
    lc.rect(col_x[i], GY0 + 32, CW_CELL, 24, '#f1f5f9', PHY_S, rx=4, sw=1.0)
    lc.text(col_x[i] + CW_CELL / 2, GY0 + 48, t, 9.5, PHY_S, 'middle', True,
            maxw=CW_CELL - 8, tag='col%d' % i)
ROWS = [
    ('组1 · full 组', [('full.0', FUL_S, FUL_F), ('full.1', FUL_S, FUL_F)], '块表私有 10 块'),
    ('组2 · sw 组', [('sw.0', SWA_S, SWA_F), ('sw.2', SWA_S, SWA_F)], '块表私有 10 块'),
    ('组3 · sw 组（单层）', [('sw.1', SWA_S, SWA_F), None], '单层组只出一位'),
]
for r, (label, cells, note) in enumerate(ROWS):
    ry = GY0 + 64 + r * (CH_CELL + CGAP)
    lc.text(GX + 16, ry + 24, label, 9.5, lc.C_TXT, 'start', True, maxw=140,
            tag='row%d:l' % r)
    lc.text(GX + 16, ry + 40, note, 8, lc.C_MUTE, 'start', maxw=140, tag='row%d:n' % r)
    for c, cell in enumerate(cells):
        if cell is None:
            lc.text(col_x[c] + CW_CELL / 2, ry + 31, '—（没有第 2 层）', 8.5, lc.C_FAINT,
                    'middle', maxw=CW_CELL - 10, tag='row%d:c%d' % (r, c))
            continue
        name, s, f = cell
        lc.rect(col_x[c], ry + 6, CW_CELL, CH_CELL - 12, f, s, rx=5, sw=1.5)
        lc.text(col_x[c] + CW_CELL / 2, ry + 30, name, 10, s, 'middle', True,
                maxw=CW_CELL - 10, tag='row%d:c%d' % (r, c))
GB = GY0 + GH

# ---------------- 矩阵正下方：两张物理张量条（各对齐一列） ----------------
TY = GB + 36
STH = 118
for t_i, (tname, sharers, sub) in enumerate([
        ('张量一 · 655360 B', 'full.0 / sw.0 / sw.1', '3 层合租 · 65536 × 10 页'),
        ('张量二 · 655360 B', 'full.1 / sw.2', '2 层合租 · 65536 × 10 页')]):
    sx = col_x[t_i]
    lc.seg(sx + CW_CELL / 2, GB + 2, sx + CW_CELL / 2, TY - 3, PHY_S, 1.6, 'std')
    lc.rect(sx, TY, CW_CELL, STH, '#ffffff', PHY_S, rx=7, sw=1.4)
    lc.text(sx + CW_CELL / 2, TY + 19, tname, 9.5, PHY_S, 'middle', True,
            maxw=CW_CELL - 12, tag='ts%d:t' % t_i)
    PW_ = (CW_CELL - 16 - 9 * 2) / 10
    for p in range(10):
        lc.rect(sx + 8 + p * (PW_ + 2), TY + 30, PW_, 34, '#f8fafc', PHY_S, rx=2, sw=0.9)
        lc.text(sx + 8 + p * (PW_ + 2) + PW_ / 2, TY + 51, str(p), 8.2, PHY_S, 'middle',
                maxw=PW_, tag='pg%d%d' % (t_i, p))
    lc.text(sx + CW_CELL / 2, TY + 82, sharers, 9, lc.C_TXT, 'middle', True,
            maxw=CW_CELL - 12, tag='ts%d:s' % t_i)
    lc.text(sx + CW_CELL / 2, TY + 99, sub, 8.2, '#475569', 'middle',
            maxw=CW_CELL - 12, tag='ts%d:b' % t_i)
TB = TY + STH

# ---------------- 矩阵左下方：三张私有块表（调度器侧账本） ----------------
BT_X, BT_W = GX, RH_LBL
BT_Y = GB + 36
lc.rect(BT_X, BT_Y, BT_W, TB - BT_Y, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.3)
lc.text(BT_X + 12, BT_Y + 19, '三张私有块表', 9.5, lc.C_KV_S, 'middle', True,
        maxw=BT_W - 24, tag='bt:t')
lc.text(BT_X + 12, BT_Y + 36, '每组一张 · append-only', 8.2, '#475569', 'middle',
        maxw=BT_W - 24, tag='bt:s')
for r, (label, _, _) in enumerate(ROWS):
    ry = BT_Y + 50 + r * 24
    lc.text(BT_X + 12, ry + 10, label.split(' · ')[0], 8.4, lc.C_TXT, 'start', True,
            maxw=44, tag='bt:r%d' % r)
    for p in range(10):
        lc.rect(BT_X + 58 + p * 8.6, ry, 7.4, 14, '#ffffff', lc.C_KV_S, rx=1.5, sw=0.7)
    lc.text(BT_X + 58 + 10 * 8.6 + 4, ry + 10, '0-9', 8, lc.C_MUTE, 'start', maxw=30,
            tag='bt:e%d' % r)

# ---------------- 右：对照 · 单组异宽（不合租） ----------------
CX, CW_ = 900, BXR - 900
lc.rect(CX, GY0, CW_, GB - GY0, '#ffffff', lc.C_MUTE, rx=9, sw=1.2, dash=True)
lc.text(CX + 14, GY0 + 20, '对照 · 单组异宽（组大小 1，不合租）', 11, lc.C_TXT, 'start', True,
        maxw=CW_ - 28, tag='cp:t')
lc.text(CX + 14, GY0 + 38, '同型不同宽的两层各分一张张量、按各自页分账', 9, lc.C_MUTE,
        'start', maxw=CW_ - 28, tag='cp:s')
CT = [
    ('l0（kv_heads 8）', '页 65536 B', '张量A · 327680 B', 5, 46),
    ('l1（kv_heads 4）', '页 32768 B', '张量B · 163840 B', 5, 23),
]
cy = GY0 + 56
for name, page, tname, npages, pw in CT:
    lc.text(CX + 14, cy + 13, name, 9, lc.C_TXT, 'start', True, maxw=130, tag='ct:n')
    lc.text(CX + 14, cy + 28, page, 8.2, lc.C_MUTE, 'start', maxw=130, tag='ct:p')
    for p in range(npages):
        lc.rect(CX + 150 + p * (pw + 2), cy, pw, 34, '#f8fafc', PHY_S, rx=2, sw=0.9)
    lx2 = CX + 150 + npages * (pw + 2) + 8
    lc.text(lx2, cy + 13, tname, 8.8, PHY_S, 'start', True, maxw=CW_ - (lx2 - CX) - 12,
            tag='ct:t')
    lc.text(lx2, cy + 28, '%d 页' % npages, 8, '#475569', 'start',
            maxw=CW_ - (lx2 - CX) - 12, tag='ct:c')
    cy += 46
lc.text(CX + 14, cy + 8, '总 5 块 · 聚合页 98304 B（65536 + 32768）· 逐层各一张张量', 8.6,
        '#334155', 'start', maxw=CW_ - 28, tag='cp:sum')
lc.text(CX + 14, cy + 25, '——只有一层一种页型时不共享，页宽按层分账', 8.6, '#334155',
        'start', maxw=CW_ - 28, tag='cp:note')

# ---------------- 安全性注记（全宽） ----------------
SY2 = max(TB, cy + 45) + 22
lc.rect(MX, SY2, BXR - MX, 64, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, SY2 + 21, '共享不冲突：一个 block_id 同一时刻只归一个组使用——分配时每组经自己的 manager 从同一 BlockPool 拿块，', 9.6,
        lc.C_KV_S, 'start', True, maxw=BXR - MX - 32, tag='safe:t')
lc.text(MX + 16, SY2 + 41, '块一经分给组 g 即 ref_cnt ≥ 1、不出现在其他组的表里——共享的是『房间面积』不是『钥匙』；源码注释："As layers of different groups have different block table, they will use different parts of the shared Tensor."',
        9, '#334155', 'start', maxw=BXR - MX - 32, tag='safe:l1')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = SY2 + 88
lx = MX
for s, f, name in [(FUL_S, FUL_F, 'full attention 层'), (SWA_S, SWA_F, 'SWA 层')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, f, s, rx=3, sw=1.3)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=140, tag='lg')
    lx += 26 + lc.tw(name, 8.8) + 24
lc.rect(lx, LEG_Y - 9, 20, 13, '#f8fafc', PHY_S, rx=3, sw=1.0)
lc.text(lx + 26, LEG_Y + 1, '物理张量页（两张各 10 页）', 8.8, lc.C_TXT, 'start', maxw=190,
        tag='lg2')
lx += 26 + lc.tw('物理张量页（两张各 10 页）', 8.8) + 24
lc.text(lx, LEG_Y + 1, '页号 = block_id：一个块在它所属的每张张量里各占同号一页', 8.8,
        lc.C_MUTE, 'start', maxw=BXR - lx, tag='lg:note')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/kv_cache_utils.py:L1411-L1437（通用分支：每池由每组各出一层 · 源码注释自带三组两池 ASCII 图例）· '
        'L1386-L1408（单组异宽逐层张量）· 数字取自配套精简版 host 实跑 · 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')

# ---------------- 装配输出 ----------------
H = LEG_Y + 44
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-tensor-sharing.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
