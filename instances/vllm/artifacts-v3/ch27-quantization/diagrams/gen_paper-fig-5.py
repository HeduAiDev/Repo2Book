#!/usr/bin/env python3
"""paper-fig-5 · SmoothQuant 论文精髓图重绘(arXiv:2211.10438 Fig.4,writer 图集变更 add)

真实模型的离群证据:OPT-13B 某线性层输入激活的少数通道幅度 >70、且逐 token 持续;
权重分布平坦。量化后激活被压平、权重变稍陡但仍然平。打消「合成小例子是不是编的」
疑虑的实测证据。与本章合成 4 通道坍缩账图互补:这张是真实模型的实测证据。

忠实重绘:四联 3D 柱状(原始 X / 原始 W / 平滑后 X̂ / 调整后 Ŵ),横轴=通道、纵深=token、
竖轴=幅度。只标可溯源数字(>70 量级、OPT-13B);逐柱值论文 md 不含,按论文图趋势示意并
整体标注「柱值按原图趋势示意」。配色/字体套本书视觉语言,文字译中,非像素复制。
key_figures provenance 豁免:数据 provenance = 原论文本身。
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

lc.text(MX, 34, '真实模型的长相:OPT-13B 某线性层的激活与权重',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '原始激活:少数通道幅度 >70、逐 token 持续(同一通道方差小)· 原始权重:平坦均匀 —— 平滑后激活被压平、权重稍陡仍平坦 · 柱值按原图趋势示意',
        10.5, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
_ch = '论文精髓图重绘 · arXiv:2211.10438 Fig.4'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 3D 柱(等距投影:front 面 + top 面 + side 面) =================
NCH, NTOK = 8, 6
DX_CH, DX_D, DY_D = 30, 9, 8          # 通道步进 / 纵深 x 步 / 纵深 y 步
BAR_W = 22
HMAX = 170                            # 最高柱(像素)


def bars(px, floor_y, heights, front, side, top, stroke):
    """heights[c][t] = 0..1 归一化高度;纵深 token t 越大越远(先画远的)。"""
    for t in range(NTOK - 1, -1, -1):
        for c in range(NCH):
            h = max(4.0, heights[c][t] * HMAX)
            x = px + c * DX_CH + t * DX_D
            yb = floor_y + t * DY_D          # 该柱地面(前面底边)
            # 右侧面
            lc.ELEMS.append(((x + BAR_W - 1, yb - h, x + BAR_W + DX_D + 1, yb + DY_D),
                             (f'<polygon points="{x + BAR_W:.1f},{yb:.1f} '
                              f'{x + BAR_W + DX_D:.1f},{yb - DY_D:.1f} '
                              f'{x + BAR_W + DX_D:.1f},{yb - DY_D - h:.1f} '
                              f'{x + BAR_W:.1f},{yb - h:.1f}" fill="{side}" '
                              f'stroke="none"/>')))
            # 顶面
            lc.ELEMS.append(((x - 1, yb - h - DY_D - 1, x + BAR_W + DX_D + 1, yb - h + 1),
                             (f'<polygon points="{x:.1f},{yb - h:.1f} '
                              f'{x + DX_D:.1f},{yb - h - DY_D:.1f} '
                              f'{x + DX_D + BAR_W:.1f},{yb - h - DY_D:.1f} '
                              f'{x + BAR_W:.1f},{yb - h:.1f}" fill="{top}" '
                              f'stroke="none"/>')))
            # 前面
            lc.rect(x, yb - h, BAR_W, h, front, stroke, rx=1.5, sw=0.7)


def flat(v):
    return [[v] * NTOK for _ in range(NCH)]


# 高度(0..1,按原图趋势示意):X 通道 1/5 离群 ~0.42(>70 量级),其余 ~0.01
X_H = [[0.012] * NTOK for _ in range(NCH)]
for c in (1, 5):
    for t in range(NTOK):
        X_H[c][t] = 0.42 + 0.02 * ((c + t) % 2)
W_H = flat(0.035)
XS_H = [[0.018 + 0.004 * ((c + t) % 2) for t in range(NTOK)] for c in range(NCH)]
WA_H = [[0.14 + 0.02 * ((c + t) % 2) for t in range(NTOK)] for c in range(NCH)]

PANEL_W, PGAP = 330, 20
PANELS_X = [MX + p * (PANEL_W + PGAP) for p in range(4)]
PY, PH = 100, 330
FLOOR = PY + 258

PANELS = [
    ('(a) 原始激活 X', '少数通道一柱擎天', '难量化', lc.C_ABORT, '#fef2f2',
     X_H, '#f97316', '#c2410c', '#fdba74', '#ea580c'),
    ('(b) 原始权重 W', '全通道平坦均匀', '好量化', lc.C_GPU_S, '#f0fdf4',
     W_H, '#93c5fd', '#3b82f6', '#bfdbfe', '#60a5fa'),
    ('(c) 平滑后 X̂ = X ÷ s', '离群被逐通道压平', '好量化', lc.C_GPU_S, '#f0fdf4',
     XS_H, '#fdba74', '#c2410c', '#fed7aa', '#ea580c'),
    ('(d) 调整后 Ŵ = W × s', '全部抬升,仍平坦', '仍好量化', lc.C_GPU_S, '#f0fdf4',
     WA_H, '#bfdbfe', '#3b82f6', '#dbeafe', '#60a5fa'),
]
for p, (title, sub, verdict, vc, vbg, hs, f_front, f_side, f_top, f_stroke) in enumerate(PANELS):
    px = PANELS_X[p]
    lc.rect(px, PY, PANEL_W, PH, '#ffffff', GRID, rx=8, sw=1.2)
    lc.text(px + 14, PY + 24, title, 11.5, lc.C_TXT, 'start', True, maxw=PANEL_W - 28,
            tag='pt' + str(p))
    lc.text(px + 14, PY + 42, sub, 8.5, lc.C_MUTE, 'start', maxw=PANEL_W - 28, tag='ps' + str(p))
    bars(px + 26, FLOOR, hs, f_front, f_side, f_top, f_stroke)
    # 地面参考线(最前排柱底)
    lc.rect(px + 20, FLOOR, NCH * DX_CH + 10, 1.5, '#94a3b8', '#94a3b8', rx=0.5, sw=0)
    # 轴标(合并一行,放地面下方——避开柱体)
    lc.text(px + 26, FLOOR + 16, '通道 → · 纵深 = token(同通道逐 token 持续)', 8, lc.C_MUTE,
            'start', maxw=PANEL_W - 44, tag='axc' + str(p))
    # >70 标注(仅面板 a)
    if p == 0:
        _tx = px + 26 + 1 * DX_CH + BAR_W / 2
        _ty = FLOOR - X_H[1][0] * HMAX - 14
        lc.text(_tx + 26, _ty, '>70', 10, lc.C_ABORT, 'middle', True, maxw=60, tag='gt70')
        lc.seg(_tx + 12, _ty + 3, _tx + 4, _ty + 10, lc.C_ABORT, 1.2)
    vw = lc.tw(verdict, 9.5, True) + 16
    lc.rect(px + (PANEL_W - vw) / 2, PY + PH - 30, vw, 19, vbg, vc, rx=9, sw=1.2)
    lc.text(px + PANEL_W / 2, PY + PH - 17, verdict, 9.5, vc, 'middle', True, maxw=vw - 6,
            tag='vd' + str(p))

# ================= 底部:三条观察(原 caption 逐字口径) =================
BY, BH = 448, 160
lc.rect(MX, BY, 1380, BH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(MX + 16, BY + 24, '论文的三条观察(原图 caption,OPT-13B 某线性层)', 11, lc.C_TXT,
        'start', True, maxw=800, tag='b:h')
OBS = [
    ('①', '原始激活里,少数通道的幅度非常大(>70)——把 per-tensor 的尺子撑爆,普通通道只剩极少有效位'),
    ('②', '同一激活通道内的方差小——离群不是偶发:哪个通道离群,就逐 token 持续离群(红柱成墙)'),
    ('③', '原始权重分布平坦均匀——天然好量化;SmoothQuant 后:激活离群大幅平滑,权重仍相当平坦'),
]
for i, (n, txt) in enumerate(OBS):
    lc.text(MX + 20, BY + 52 + i * 26, n + ' ' + txt, 9.5, '#334155', 'start', maxw=1340,
            tag='obs' + str(i))
lc.text(MX + 20, BY + 136, '「合成 4 通道小例子是不是编的?」——不是:真实生产级模型里,离群通道确实长这样。',
        9, lc.C_MUTE, 'start', maxw=1340, tag='b:note')

# ================= 页脚 =================
lc.text(MX, 636, '重绘自 arXiv:2211.10438 Fig.4(§2 · OPT-13B 某线性层实测)· 图上唯一实测数字 = 幅度 >70 与模型名 OPT-13B;逐柱值论文未载,按原图趋势示意',
        9, lc.C_FAINT, 'start', maxw=1380, tag='foot')

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'paper-fig-5.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
