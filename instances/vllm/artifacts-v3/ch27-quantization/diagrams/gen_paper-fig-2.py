#!/usr/bin/env python3
"""paper-fig-2 · AWQ 论文精髓图重绘(arXiv:2306.00978 Fig.2,writer 图集变更 add)

AWQ 核心三联图:左 RTN 量化(PPL 43.2)→ 中按激活分布挑 1% 显著权重保留 FP16 的理想解
(PPL 13.0,但混合精度硬件不友好)→ 右 AWQ 逐通道缩放(全程 INT、接近理想解)。
与本章放大镜手算图([0.9,9.9] 误差比)互补:那张是逐通道数值账、这张是论文一图流证据。

忠实重绘:三联横排,中联按激活分布高亮显著通道、右联逐通道缩放;PPL 数字与 Table 1
对照全部来自 spec.numbers(原论文 Fig.2 caption + §3.1 Table 1);左联 FP16→INT3
示例格值取自原图(a)图面。配色/字体套本书视觉语言,文字译中,非像素复制。
key_figures provenance 豁免:数据 provenance = 原论文本身。
坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 720
MX = 60
BXR = 1440
GRID = '#e2e8f0'

lc.text(MX, 34, 'AWQ 一图流:显著与否,看激活不看权重',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, 'RTN 一刀切 → 按激活分布挑 1% 显著权重留 FP16(理想解,但混合精度硬件不友好)→ AWQ 逐通道缩放(全程 INT,接近理想解)· 测量设置:OPT-6.7B · INT3-g128 · WikiText PPL',
        10.5, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
_ch = '论文精髓图重绘 · arXiv:2306.00978 Fig.2'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

C_FP16_F, C_FP16_S = '#f8fafc', '#94a3b8'      # FP16 格
C_INT_F, C_INT_S = '#e2e8f0', '#94a3b8'        # INT3 格
C_SAL_F, C_SAL_S = '#fecaca', lc.C_ABORT       # 显著通道(按激活)
C_KEEP_F, C_KEEP_S = '#ffffff', lc.C_ABORT     # 留 FP16 的格

# ================= (a) RTN 量化 =================
PAX, PAY, PAW, PAH = MX, 96, 430, 350
lc.rect(PAX, PAY, PAW, PAH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(PAX + 16, PAY + 26, '(a) RTN 量化', 12.5, lc.C_TXT, 'start', True,
        maxw=PAW - 32, tag='pa:t')
lc.text(PAX + 16, PAY + 46, '全部权重四舍五入到 INT3 网格', 9.5, lc.C_MUTE, 'start',
        maxw=PAW - 32, tag='pa:s')
FP16 = [['+2.4', '-3.5', '-2.8', '-3.9'], ['+0.1', '-3.8', '+2.4', '+3.4'],
        ['+0.9', '+3.3', '-1.9', '-2.3']]
INT3 = [['+2', '-4', '-3', '-4'], ['+0', '-4', '+2', '+3'], ['+1', '+3', '-2', '-2']]
CW_, CH_ = 58, 24


def val_grid(x0, y0, rows, fill, stroke, hot_col=None, keep=False):
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            f_, s_ = fill, stroke
            if hot_col is not None and c == hot_col:
                f_, s_ = (C_KEEP_F, C_KEEP_S) if keep else (C_SAL_F, C_SAL_S)
            lc.rect(x0 + c * CW_, y0 + r * CH_, CW_ - 3, CH_ - 4, f_, s_, rx=2.5, sw=1.1)
            if v:
                col = lc.C_ABORT if (hot_col is not None and c == hot_col) else '#334155'
                lc.text(x0 + c * CW_ + (CW_ - 3) / 2, y0 + r * CH_ + CH_ / 2 - 1, v, 8.5,
                        col, 'middle', maxw=CW_ - 6, tag='vg' + v + str(r) + str(c))


GAX, GAY = PAX + 78, PAY + 66
lc.text(GAX - 10, GAY + 1.5 * CH_ - 2, 'W(FP16)', 9, lc.C_MUTE, 'end', maxw=64, tag='pa:wf')
val_grid(GAX, GAY, FP16, C_FP16_F, C_FP16_S)
_ax = GAX + (4 * CW_) / 2 - 1.5
_ay = GAY + 3 * CH_
lc.seg(_ax, _ay + 8, _ax, _ay + 34, lc.C_MUTE, 2.0, 'std')
lc.text(_ax + 8, _ay + 26, '除以 Δ、取整', 8.5, lc.C_MUTE, 'start', maxw=90, tag='pa:q')
lc.text(GAX - 10, _ay + 42 + 1.5 * CH_ - 2, 'W(INT3)', 9, lc.C_MUTE, 'end', maxw=64, tag='pa:wi')
val_grid(GAX, _ay + 42, INT3, C_INT_F, C_INT_S)
# PPL 芯片
lc.text(PAX + PAW / 2, PAY + PAH - 44, 'PPL 43.2', 13, lc.C_ABORT, 'middle', True,
        maxw=140, tag='pa:ppl')
lc.text(PAX + PAW / 2, PAY + PAH - 24, '基线:全部 INT3', 8.5, lc.C_MUTE, 'middle',
        maxw=140, tag='pa:ppl2')

# ================= (b) 按激活分布挑 1% 显著权重留 FP16 =================
PBX, PBY, PBW, PBH = 520, 96, 460, 350
lc.rect(PBX, PBY, PBW, PBH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(PBX + 16, PBY + 26, '(b) 按激活分布挑 1% 显著权重留 FP16', 12.5, lc.C_TXT, 'start',
        True, maxw=PBW - 32, tag='pb:t')
lc.text(PBX + 16, PBY + 46, '显著通道 = 激活幅度大的通道(不是权重大)', 9.5, lc.C_MUTE,
        'start', maxw=PBW - 32, tag='pb:s')
# 左:X 矩阵,最右列 = 显著通道
lc.text(PBX + 96, PBY + 78, '激活 X', 9, lc.C_TXT, 'middle', True, maxw=80, tag='pb:x')
XG_X, XG_Y = PBX + 30, PBY + 90
val_grid(XG_X, XG_Y, [['', '', '', ''], ['', '', '', ''], ['', '', '', '']], '#f1f5f9',
         '#cbd5e1', hot_col=3)
lc.text(XG_X + 4 * CW_ / 2, XG_Y + 3 * CH_ + 16, '最右列:激活幅度大', 8.5, lc.C_ABORT,
        'middle', maxw=190, tag='pb:xh')
# 挑选箭头:X 显著列 → W 对应列
_ay2 = XG_Y + 1.5 * CH_
lc.seg(XG_X + 4 * CW_, _ay2, PBX + 250, _ay2, lc.C_ABORT, 2.2, 'std')
lc.text((XG_X + 4 * CW_ + PBX + 250) / 2, _ay2 - 8, '按列挑', 8.5, lc.C_ABORT, 'middle',
        maxw=70, tag='pb:sel')
# 右:W 网格,最右列留 FP16
WG_X, WG_Y = PBX + 260, PBY + 90
val_grid(WG_X, WG_Y, [['', '', '', ''], ['', '', '', ''], ['', '', '', '']], C_INT_F,
         C_INT_S, hot_col=3, keep=True)
lc.text(WG_X + 4 * CW_ / 2, WG_Y + 3 * CH_ + 16, '最右列留 FP16,其余 INT3', 8.5, lc.C_ABORT,
        'middle', maxw=220, tag='pb:wh')
# PPL 芯片 + 混合精度徽标
lc.text(PBX + 150, PBY + PBH - 44, 'PPL 13.0', 13, lc.C_GPU_S, 'middle', True, maxw=140,
        tag='pb:ppl')
_bw = lc.tw('混合精度:硬件不友好', 9, True) + 16
lc.rect(PBX + PBW - _bw - 14, PBY + PBH - 56, _bw, 20, '#fff7ed', lc.C_ENG_S, rx=9, sw=1.2)
lc.text(PBX + PBW - _bw / 2 - 14, PBY + PBH - 42, '混合精度:硬件不友好', 9, lc.C_ENG_S,
        'middle', True, maxw=_bw - 4, tag='pb:mix')
lc.text(PBX + PBW / 2, PBY + PBH - 22, '理想解:0.1%-1% 的通道高精度即可大幅救回', 8.5,
        lc.C_MUTE, 'middle', maxw=PBW - 28, tag='pb:ideal')

# ================= (c) AWQ:逐通道缩放 =================
PCX, PCY, PCW, PCH = 1010, 96, 430, 350
lc.rect(PCX, PCY, PCW, PCH, '#f0fdf4', lc.C_GPU_S, rx=8, sw=1.4)
lc.text(PCX + 16, PCY + 26, '(c) AWQ:逐通道缩放后再量化', 12.5, lc.C_TXT, 'start', True,
        maxw=PCW - 32, tag='pc:t')
lc.text(PCX + 16, PCY + 46, '显著通道 ×s、激活 ÷s——量化在放大的格上进行,全程 INT', 9.5,
        lc.C_MUTE, 'start', maxw=PCW - 32, tag='pc:s')
# 左:X 显著通道 ÷s
lc.text(PCX + 96, PCY + 78, '激活 X', 9, lc.C_TXT, 'middle', True, maxw=80, tag='pc:x')
X2X, X2Y = PCX + 30, PCY + 90
val_grid(X2X, X2Y, [['', '', '', ''], ['', '', '', ''], ['', '', '', '']], '#f1f5f9',
         '#cbd5e1', hot_col=3)
lc.text(X2X + 4 * CW_ / 2, X2Y + 3 * CH_ + 16, '显著通道 ÷s', 8.5, lc.C_ENG_S, 'middle',
        maxw=190, tag='pc:xh')
# 右:W 显著通道 ×s → 量化 → ÷s 还原
_ay3 = X2Y + 1.5 * CH_
lc.seg(X2X + 4 * CW_, _ay3, PCX + 250, _ay3, lc.C_GPU_S, 2.2, 'std')
lc.text((X2X + 4 * CW_ + PCX + 250) / 2, _ay3 - 8, '逐通道', 8.5, lc.C_GPU_S, 'middle',
        maxw=70, tag='pc:sel')
W2X, W2Y = PCX + 260, PCY + 90
val_grid(W2X, W2Y, [['', '', '', ''], ['', '', '', ''], ['', '', '', '']], C_INT_F,
         C_INT_S, hot_col=3)
lc.text(W2X + 4 * CW_ / 2, W2Y + 3 * CH_ + 16, '显著通道 ×s 后取整,读回时 ÷s', 8.5,
        lc.C_GPU_S, 'middle', maxw=230, tag='pc:wh')
lc.text(PCX + 150, PCY + PCH - 44, '全程 INT', 13, lc.C_GPU_S, 'middle', True, maxw=140,
        tag='pc:int')
_bw2 = lc.tw('硬件友好 · 接近理想解', 9, True) + 16
lc.rect(PCX + PCW - _bw2 - 14, PCY + PCH - 56, _bw2, 20, '#f0fdf4', lc.C_GPU_S, rx=9, sw=1.2)
lc.text(PCX + PCW - _bw2 / 2 - 14, PCY + PCH - 42, '硬件友好 · 接近理想解', 9, lc.C_GPU_S,
        'middle', True, maxw=_bw2 - 4, tag='pc:hw')
lc.text(PCX + PCW / 2, PCY + PCH - 22, '不用留任何 FP16:缩放折进相邻算子,存档仍是一副 INT 网格',
        8.5, lc.C_MUTE, 'middle', maxw=PCW - 28, tag='pc:note')

# ================= 底部:Table 1 对照(按权重挑 ≈ 随机挑) =================
BY, BH = 470, 178
lc.rect(MX, BY, 1380, BH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(MX + 16, BY + 24, '证据:同一张 INT3 网格,「按什么挑」决定成败(Table 1,OPT-6.7B · INT3-g128 · WikiText PPL ↓)',
        11, lc.C_TXT, 'start', True, maxw=1100, tag='b:h')
BARS = [
    ('RTN(不挑)', 23.54, lc.C_ABORT, '#fef2f2'),
    ('按激活挑 1%', 11.39, lc.C_GPU_S, '#f0fdf4'),
    ('按权重挑 1%', 22.37, lc.C_ENG_S, '#fff7ed'),
    ('随机挑 1%', 24.23, lc.C_MUTE, '#f1f5f9'),
]
BX0, BW_, BGAP = MX + 150, 220, 40
VMAX = 26.0
BAR_Y0, BAR_Y1 = BY + 44, BY + 132
for i, (name, v, col, f_) in enumerate(BARS):
    x = BX0 + i * (BW_ + BGAP)
    hgt = v / VMAX * (BAR_Y1 - BAR_Y0)
    lc.rect(x, BAR_Y1 - hgt, BW_, hgt, f_, col, rx=4, sw=1.6)
    lc.text(x + BW_ / 2, BAR_Y1 - hgt - 9, f'{v:.2f}', 10, col, 'middle', True, maxw=80,
            tag='bar' + str(i))
    lc.text(x + BW_ / 2, BAR_Y1 + 16, name, 9, '#334155', 'middle', maxw=BW_, tag='barn' + str(i))
lc.rect(BX0 - 12, BAR_Y1, 4 * BW_ + 3 * BGAP + 24, 2, '#334155', '#334155', rx=1, sw=0)
lc.text(MX + 16, BAR_Y0 + 6, 'PPL ↓', 9, lc.C_MUTE, 'start', maxw=60, tag='b:ax')
lc.text(BX0 + 4 * BW_ + 3 * BGAP + 24, BAR_Y1 - 8, '按权重挑 ≈ 随机挑;', 9.5,
        lc.C_TXT, 'start', True, maxw=195, tag='b:c1')
lc.text(BX0 + 4 * BW_ + 3 * BGAP + 24, BAR_Y1 + 8, '显著与否看激活,不看权重', 9.5, lc.C_TXT,
        'start', True, maxw=195, tag='b:c2')

# ================= 页脚 =================
lc.text(MX, 694, '重绘自 arXiv:2306.00978 Fig.2 · Table 1(§3.1):PPL 43.2→13.0 与四条对照数字均出自原论文 · 左联 FP16→INT3 示例格值取自原图 (a) · 与本章放大镜手算图互补',
        9, lc.C_FAINT, 'start', maxw=1380, tag='foot')

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'paper-fig-2.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
