#!/usr/bin/env python3
"""ch14 机制图 · 组×宽度桶矩阵(figure_spec ch14-fig-width-bucket-matrix,模板 state-table)

学社区 gpu_kv_planning 图的视觉语法:行 = 宽度桶(降序)、列 = 分配组;valid 格白蓝底
三行小字(spec 名/层数/Byte),灰格 = 浪费的桶级归因(条带并集,格间不可加总),
每列底部琥珀条 = 行闲置(每列唯一可加总的账)。右下角注两枚:①三对同页 = 只有 4 行
的原因;②Flash-43 同构(d=22 = 博客『组大小 22』)。

数字全部取自 explainer planning-width-bucket-matrix 的 figure_spec.numbers +
worked_example.table(61 层 Pro 主表,host 桩跑 pin 算法)。坐标由常量/循环计算;
文本全 esc()。配 ch14-fig-packed-slicing:那张讲物理形态,这张讲账面。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440
VAL_S, VAL_F = lc.C_API_S, lc.C_API_F     # valid 格 = 蓝(账上有物)
GRA_F, GRA_S = '#f1f5f9', '#cbd5e1'       # 灰格 = 桶级归因
GRA_T, GRA_T2 = '#475569', '#64748b'      # 灰格文字两级
IDLE_F, IDLE_S, IDLE_T = '#fffbeb', '#d97706', '#b45309'   # 行闲置 = 琥珀(浪费的账)

# ---------------- 标题区 ----------------
lc.text(MX, 34, '组 × 宽度桶矩阵:valid 格是每组的账,灰格只是浪费的桶级归因——每列能加总的只有最底一行',
        16.5, lc.C_TXT, 'start', True, maxw=990, tag='title')
lc.text(MX, 58, '7 个缓存形态被三对同页收进 4 个宽度桶(37440/32832/8640/1728),列 = 5 个分配组;'
               '一个块号归组时用满自己的桶,闲置尾部 [dense, block_stride) 是浪费的唯一来源',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · KV 账本列组化层 · 配 packed 切片图(物理形态)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 主面板 ----------------
PY0 = 88
LX, LW = MX, BXR - MX
PH_IN_R = LX + LW - 18                    # 面板内容右缘 1422
lc.rect(LX, PY0, LW, 588, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)

# 面板头:左上『站 8』徽标(L2 回指)+ 标题
BDW = lc.tw('站 8', 9.5, True) + 16
lc.rect(LX + 18, PY0 + 12, BDW, 20, lc.C_BADGE_F, lc.C_ENG_S, rx=9, sw=1.1)
lc.text(LX + 18 + BDW / 2, PY0 + 26, '站 8', 9.5, lc.C_ENG_S, 'middle', True, maxw=BDW - 4, tag='badge:st8')
lc.text(LX + 18 + BDW + 10, PY0 + 26,
        '主表 · 61 层 Pro 真数(社区实读权重口径)——行 = 宽度桶(降序)· 列 = 分配组 · 读法与社区 gpu_kv_planning 图一致',
        10.5, lc.C_TXT, 'start', True, maxw=900, tag='mp:t')

# 图例行(三种语义色 → 必须给图例)
LY = PY0 + 50
_lx = LX + 18
for fill, stroke, label in [
        (VAL_F, VAL_S, 'valid 格:装什么 · 几层 · 几 Byte'),
        (GRA_F, GRA_S, '灰格:他组该桶条带落在本组密排尾的并集(桶级归因,格间不可加总)'),
        (IDLE_F, IDLE_S, '行闲置:本组密排尾 [dense, block_stride)——每列唯一的账')]:
    lc.rect(_lx, LY - 9, 14, 12, fill, stroke, rx=3, sw=1.3)
    lc.text(_lx + 19, LY + 1, label, 8.6, lc.C_TXT, 'start', maxw=420, tag='leg:' + label[:10])
    _lx += 19 + lc.tw(label, 8.6) + 24

# ---------------- 列头(5 组) ----------------
LBX, LBW = LX + 18, 228                   # 行标签列
GX0, CW, CGAP = 320, 212, 10              # 组列起点/列宽/列距
CHY = PY0 + 64
GROUPS = [
    ('G0 主账', '91 层 · 块 256', 'main×61 + indexer×30'),
    ('G1 swa 前半', '31 层 · 块 64', ''),
    ('G2 swa 后半', '30 层 · 块 64', ''),
    ('G3 c4 状态', '60 层 · 块 4', 'attn_state_c4 + indexer_state'),
    ('G4 c128 状态', '31 层 · 块 8', 'attn_state_c128'),
]
lc.text(LBX + 2, CHY + 13, '宽度桶(页宽 B)', 9.5, lc.C_TXT, 'start', True, maxw=LBW, tag='ch:lb')
lc.text(LBX + 2, CHY + 27, '同一页宽 = 一行', 7.8, lc.C_MUTE, 'start', maxw=LBW, tag='ch:lb2')
for j, (name, meta, extra) in enumerate(GROUPS):
    cx = GX0 + j * (CW + CGAP)
    lc.text(cx + CW / 2, CHY + 13, name, 10, lc.C_TXT, 'middle', True, maxw=CW - 8, tag='ch:g%d' % j)
    lc.text(cx + CW / 2, CHY + 26, meta, 8.6, lc.C_MUTE, 'middle', maxw=CW - 8, tag='ch:g%dm' % j)
    if extra:
        lc.text(cx + CW / 2, CHY + 38, extra, 7.6, lc.C_FAINT, 'middle', maxw=CW - 6, tag='ch:g%de' % j)
lc.seg(LBX, CHY + 46, PH_IN_R, CHY + 46, '#e2e8f0', 1.0)

# ---------------- 矩阵本体:4 行 × 5 列 ----------------
# 每格:('v', spec名, 层数, Byte) 或 ('g', Byte, 条带注, 是否琥珀虚线框[G4 三灰格])
ROWS = [
    (37440, 'swa ≡ main_c4 同页对', '跨 G0/G1/G2 三组 valid', [
        ('v', 'main_c4', '×30 层', '1123200 B'),
        ('v', 'swa', '×31 层', '1160640 B'),
        ('v', 'swa', '×30 层', '1123200 B'),
        ('g', '112320 B', '3 条并集', False),
        ('g', '290880 B', '12 条并集', True),
    ]),
    (32832, 'attn_state 两族同页对', 'c4 族落 G3 · c128 族落 G4', [
        ('g', '0 B', '最宽租客 · 无行尾', False),
        ('g', '65664 B', '2 条', False),
        ('g', '98496 B', '3 条', False),
        ('v', 'attn_state_c4', '×30 层', '984960 B'),
        ('v', 'attn_state_c128', '×31 层', '1017792 B'),
    ]),
    (8640, 'indexer ≡ indexer_state 同页对', 'G0 与 G3 各一半', [
        ('v', 'indexer', '×30 层', '259200 B'),
        ('g', '60480 B', '7 条', False),
        ('g', '69120 B', '8 条', False),
        ('v', 'indexer_state', '×30 层', '259200 B'),
        ('g', '112320 B', '13 条', True),
    ]),
    (1728, 'main_c128 独占', '独门独户一档', [
        ('v', 'main_c128', '×31 层', '53568 B'),
        ('g', '53568 B', '31 条全中', False),
        ('g', '53568 B', '31 条', False),
        ('g', '53568 B', '31 条', False),
        ('g', '53568 B', '31 条', True),
    ]),
]
R0, RH, VG = CHY + 56, 84, 8
for i, (page, pair1, pair2, cells) in enumerate(ROWS):
    ry = R0 + i * (RH + VG)
    # 行标签:页宽徽标 + 同页对注记
    lc.rect(LBX, ry + 8, 110, 30, '#ffffff', lc.C_MUTE, rx=6, sw=1.4)
    lc.text(LBX + 55, ry + 28, f'{page} B', 12.5, lc.C_TXT, 'middle', True, maxw=102, tag='pg:%d' % page)
    lc.text(LBX + 2, ry + 56, pair1, 8.4, lc.C_TXT, 'start', True, maxw=LBW - 4, tag='pr:%d1' % i)
    lc.text(LBX + 2, ry + 70, pair2, 8.0, lc.C_MUTE, 'start', maxw=LBW - 4, tag='pr:%d2' % i)
    # 格子
    for j, cell in enumerate(cells):
        cx = GX0 + j * (CW + CGAP)
        if cell[0] == 'v':
            _, name, layers, byts = cell
            lc.rect(cx, ry, CW, RH, VAL_F, VAL_S, rx=5, sw=1.4)
            lc.text(cx + CW / 2, ry + 26, name, 10, lc.C_TXT, 'middle', True, maxw=CW - 10, tag='c%d%d:n' % (i, j))
            lc.text(cx + CW / 2, ry + 45, layers, 8.8, '#334155', 'middle', maxw=CW - 10, tag='c%d%d:l' % (i, j))
            lc.text(cx + CW / 2, ry + 66, byts, 10.5, VAL_S, 'middle', True, maxw=CW - 10, tag='c%d%d:b' % (i, j))
        else:
            _, byts, note, hot = cell
            if hot:   # G4 三灰格:琥珀虚线框(与底行『灰格不可加总』实证呼应)
                lc.rect(cx, ry, CW, RH, GRA_F, IDLE_S, rx=5, sw=1.3, dash=True)
            else:
                lc.rect(cx, ry, CW, RH, GRA_F, GRA_S, rx=5, sw=1.1)
            lc.text(cx + CW / 2, ry + 38, byts, 10.5, GRA_T, 'middle', True, maxw=CW - 10, tag='c%d%d:b' % (i, j))
            lc.text(cx + CW / 2, ry + 56, note, 8.2, GRA_T2, 'middle', maxw=CW - 10, tag='c%d%d:t' % (i, j))

# ---------------- 行闲置条(每列唯一的账) ----------------
IY = R0 + 4 * (RH + VG) + 2
lc.rect(LBX, IY, LBW, 40, '#ffffff', 'none', rx=0, sw=0)
lc.text(LBX + 2, IY + 13, '行闲置 = stride − dense', 9.5, lc.C_TXT, 'start', True, maxw=LBW - 4, tag='id:lb')
lc.text(LBX + 2, IY + 27, 'block_stride 1435968 = G0 密排', 7.8, lc.C_MUTE, 'start', maxw=LBW - 4, tag='id:lb2')
lc.text(LBX + 2, IY + 38, '= G0 三 valid 格之和 1123200+259200+53568', 7.8, lc.C_MUTE, 'start', maxw=LBW - 4, tag='id:lb3')
IDLE = ['0 B（0%）', '275328 B（19.174%）', '312768 B（21.781%）', '191808 B（13.357%）', '418176 B（29.122%）']
for j, s in enumerate(IDLE):
    cx = GX0 + j * (CW + CGAP)
    lc.rect(cx, IY, CW, 40, IDLE_F, IDLE_S, rx=5, sw=1.3)
    lc.text(cx + CW / 2, IY + 25, s, 10.5, IDLE_T, 'middle', True, maxw=CW - 10, tag='id:g%d' % j)

# ---------------- 灰格不可加总实证条 ----------------
AY = IY + 40 + 10
lc.rect(LBX, AY, PH_IN_R - LBX, 34, '#ffffff', IDLE_S, rx=6, sw=1.1, dash=True)
lc.text(LBX + 12, AY + 21,
        '灰格不可加总(G4 列实证,琥珀虚线框三格):290880 + 112320 + 53568 = 456768 > 行闲置 418176——'
        '跨桶条带可在行尾同 offset 互相别名,灰格只是桶级归因;20 格 = 8 valid + 12 灰格,每列可加总的只有行闲置一行',
        9, IDLE_T, 'start', maxw=PH_IN_R - LBX - 24, tag='alias')

# ---------------- 右下角注两枚 ----------------
NY = PY0 + 588 + 14
NH = 62
B1X, B1W = MX, 668
lc.rect(B1X, NY, B1W, NH, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(B1X + 14, NY + 18, '① 为什么只有 4 行——三对同页', 9.5, lc.C_TXT, 'start', True, maxw=B1W - 28, tag='cn1:t')
lc.text(B1X + 14, NY + 35, '7 形态 − 3 对同页 = 4 页宽(降序四行):37440 = swa ≡ main_c4 · 8640 = indexer ≡ indexer_state',
        8.6, '#334155', 'start', maxw=B1W - 28, tag='cn1:l1')
lc.text(B1X + 14, NY + 51, '32832 = attn_state 两族 · 1728 = main_c128 独占;同页 = 两形态共用同一张物理张量',
        8.6, '#334155', 'start', maxw=B1W - 28, tag='cn1:l2')
B2X = B1X + B1W + 20
B2W = BXR - B2X
lc.rect(B2X, NY, B2W, NH, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(B2X + 14, NY + 18, '② Flash-43 同构:同一条算法、同一层构', 9.5, lc.C_TXT, 'start', True, maxw=B2W - 28, tag='cn2:t')
lc.text(B2X + 14, NY + 35, '4 桶元组 [21, 43, 21, 20] → 近似 GCD 选 d=22(垫 5)= 博客『组大小 22』· stride 1002240',
        8.6, '#334155', 'start', maxw=B2W - 28, tag='cn2:l1')
lc.text(B2X + 14, NY + 51, '博客三桶每块足迹 47808 B = 1728+8640+37440(推导口径,恰为 pin 一个层元组宽);'
                           'pin 的块 = stride 整行 1435968 B(max 不是 sum)',
        8.6, '#334155', 'start', maxw=B2W - 28, tag='cn2:l2')

# ---------------- 出处脚注 ----------------
FY = NY + NH + 12
lc.text(MX, FY, '数字全部出自 pin 算法 host 桩跑的规划矩阵(61 层 Pro 社区实读权重口径 · Flash-43 行为本轮 pin 真跑)· '
                'block_stride = max(各组 dense):vllm/v1/core/kv_cache_utils.py:L1303 · 行号基线 vLLM v0.27.1',
        8, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
H = int(FY + 14)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-width-bucket-matrix.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
