#!/usr/bin/env python3
"""paper-fig-3 · SmoothQuant 论文精髓图重绘(arXiv:2211.10438 Fig.2,writer 图集变更 add)

SmoothQuant 的搬家直觉图:难量化的激活 X(少数离群通道把量化范围撑爆、多数值只剩极少
有效位)与好量化的平坦权重 W,经离线的逐通道 ÷s / ×s 迁移后,两边都变得好量化。
与本章 6×4 数值推演图互补:这张是论文作者的直觉原图、那张是手算账。

忠实重绘:两行结构(原始 / 平滑后),每行 X 热图 + W 热图,红=难量化、绿边=好量化,
中间 ÷s(橙)/×s(紫)对偶箭头(与本章搬家图同色约定)。配色/字体套本书视觉语言,
文字译中,非像素复制。key_figures provenance 豁免:数据 provenance = 原论文本身。
坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 660
MX = 60
BXR = 1440
GRID = '#e2e8f0'

lc.text(MX, 34, 'SmoothQuant 直觉:量化难度可以搬家',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '激活 X 难量化(离群把量化范围撑爆,多数值只剩极少有效位)· 权重 W 平坦好量化 —— 逐通道 ÷s / ×s 把 scale 方差从激活搬进权重(离线),X̂ 与 Ŵ 都变得好量化',
        10.5, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
_ch = '论文精髓图重绘 · arXiv:2211.10438 Fig.2'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

C_HOT_F, C_HOT_S = '#fca5a5', lc.C_ABORT        # 离群通道(红)
C_COLD_F, C_COLD_S = '#f1f5f9', '#cbd5e1'       # 普通激活格
C_W_F, C_W_S = '#e2e8f0', '#cbd5e1'             # 权重格(平坦)
C_W_ADJ_F = '#c7d7f0'                           # 调整后权重格(稍陡仍平)

NROW, NCOL = 5, 9
CW_, CH_ = 40, 27
HM_W = NCOL * CW_


def heatmap(x0, y0, hot_cols, fill, stroke, border, border_sw, adj=False):
    """热图:NROW×NCOL;hot_cols 里的列画红柱(离群),其余浅色。"""
    for r in range(NROW):
        for c in range(NCOL):
            hot = c in hot_cols
            f_, s_ = (C_HOT_F, C_HOT_S) if hot else (fill, stroke)
            lc.rect(x0 + c * CW_, y0 + r * CH_, CW_ - 2, CH_ - 3, f_, s_, rx=2, sw=1.0)
            if hot:
                # 列内画高柱,示意「该通道整列幅度大、逐 token 持续」
                lc.rect(x0 + c * CW_ + 7, y0 + r * CH_ + 3, CW_ - 16, CH_ - 9,
                        C_HOT_S, C_HOT_S, rx=1.5, sw=0)
    lc.rect(x0 - 4, y0 - 4, HM_W + 8, NROW * CH_ + 7, 'none', border, rx=5, sw=border_sw)
    if adj:
        lc.text(x0 + HM_W + 10, y0 + NROW * CH_ / 2, '稍陡,仍平坦', 8.5, lc.C_MUTE,
                'start', maxw=90, tag='adj' + str(int(x0)))


def ruler(x0, y0, dense, label, col):
    """量化格点密度尺:密 = 有效位多。"""
    n = 13 if dense else 5
    lc.rect(x0, y0, HM_W, 3, '#334155', '#334155', rx=1.5, sw=0)
    for i in range(n):
        tx = x0 + i * (HM_W - 3) / (n - 1)
        lc.rect(tx - 0.5, y0 - 3, 1.2, 9, col, col, rx=0.5, sw=0)
    lc.text(x0 + HM_W + 10, y0 + 5, label, 8.5, col, 'start', maxw=130, tag='ru' + str(int(x0)))


# ================= 第一行:原始(搬家前) =================
R1Y = 118
X1X, W1X = MX + 60, 800
lc.text(X1X + HM_W / 2, R1Y, '激活 X —— 难量化', 11, lc.C_ABORT, 'middle', True, maxw=200,
        tag='r1x')
heatmap(X1X, R1Y + 14, {1, 6}, C_COLD_F, C_COLD_S, lc.C_ABORT, 2.6)
lc.text(X1X + HM_W / 2, R1Y + 14 + NROW * CH_ + 18, '少数离群通道撑爆量化范围', 8.5,
        lc.C_ABORT, 'middle', maxw=HM_W + 40, tag='r1xn')
ruler(X1X, R1Y + 14 + NROW * CH_ + 32, False, '格点稀:有效位少', lc.C_ABORT)

lc.text(W1X + HM_W / 2, R1Y, '权重 W —— 好量化', 11, lc.C_GPU_S, 'middle', True, maxw=200,
        tag='r1w')
heatmap(W1X, R1Y + 14, set(), C_W_F, C_W_S, lc.C_GPU_S, 2.6)
lc.text(W1X + HM_W / 2, R1Y + 14 + NROW * CH_ + 18, '分布平坦、范围集中', 8.5, lc.C_GPU_S,
        'middle', maxw=HM_W + 40, tag='r1wn')
ruler(W1X, R1Y + 14 + NROW * CH_ + 32, True, '格点密:有效位足', lc.C_GPU_S)

# ================= 中间:离线搬家(对偶箭头) =================
MIDY = 330
_ax = X1X + HM_W / 2
_wx = W1X + HM_W / 2
_y0 = 306
_y1 = 442
lc.seg(_ax, _y0, _ax, _y1, lc.C_ENG_S, 2.6, 'up')
lc.text(_ax + 10, (_y0 + _y1) / 2 - 4, 'X ÷ s(逐通道)', 9.5, lc.C_ENG_S, 'start', True,
        maxw=130, tag='mid:x')
lc.text(_ax + 10, (_y0 + _y1) / 2 + 12, '离线', 8.5, lc.C_MUTE, 'start', maxw=60, tag='mid:x2')
lc.seg(_wx, _y0, _wx, _y1, lc.C_ZMQ_S, 2.6, 'dn')
lc.text(_wx + 10, (_y0 + _y1) / 2 - 4, 'W × s', 9.5, lc.C_ZMQ_S, 'start', True, maxw=90,
        tag='mid:w')
lc.text(_wx + 10, (_y0 + _y1) / 2 + 12, '折进前一层', 8.5, lc.C_MUTE, 'start', maxw=90,
        tag='mid:w2')
_bw = lc.tw('离线迁移 · 乘法结果不变:Y = X·W = X̂·Ŵ', 10, True) + 22
lc.rect((_ax + _wx) / 2 - _bw / 2, (_y0 + _y1) / 2 - 15, _bw, 26, '#ffffff', lc.C_MUTE,
        rx=12, sw=1.2, dash=True)
lc.text((_ax + _wx) / 2, (_y0 + _y1) / 2 + 3, '离线迁移 · 乘法结果不变:Y = X·W = X̂·Ŵ', 10,
        lc.C_MUTE, 'middle', True, maxw=_bw - 8, tag='mid:eq')

# ================= 第二行:平滑后(搬家后) =================
R2Y = 436
X2X, W2X = X1X, W1X
lc.text(X2X + HM_W / 2, R2Y, 'X̂ —— 好量化', 11, lc.C_GPU_S, 'middle', True, maxw=200,
        tag='r2x')
heatmap(X2X, R2Y + 14, set(), C_COLD_F, C_COLD_S, lc.C_GPU_S, 2.6)
lc.text(X2X + HM_W / 2, R2Y + 14 + NROW * CH_ + 18, '离群被压平:通道间幅度接近', 8.5,
        lc.C_GPU_S, 'middle', maxw=HM_W + 40, tag='r2xn')
ruler(X2X, R2Y + 14 + NROW * CH_ + 32, True, '格点变密', lc.C_GPU_S)

lc.text(W2X + HM_W / 2, R2Y, 'Ŵ —— 仍好量化', 11, lc.C_GPU_S, 'middle', True, maxw=200,
        tag='r2w')
heatmap(W2X, R2Y + 14, set(), C_W_ADJ_F, C_W_S, lc.C_GPU_S, 2.6, adj=True)
lc.text(W2X + HM_W / 2, R2Y + 14 + NROW * CH_ + 18, '接住搬来的离群,仍平坦', 8.5, lc.C_GPU_S,
        'middle', maxw=HM_W + 40, tag='r2wn')
ruler(W2X, R2Y + 14 + NROW * CH_ + 32, True, '格点仍密', lc.C_GPU_S)

# ================= 页脚 =================
lc.text(MX, 636, '重绘自 arXiv:2211.10438 Fig.2(直觉图):布局与信息结构对齐论文原图 · 与本章数值推演图(6×4 手算账 + α 扫描)互补:本图 = 作者的直觉原图',
        9, lc.C_FAINT, 'start', maxw=1380, tag='foot')

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'paper-fig-3.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
