#!/usr/bin/env python3
"""paper-fig-1 · GPTQ 论文精髓图重绘(arXiv:2210.17323 Fig.2,writer 图集变更 add)

论文作者自画的算法全景图:权重矩阵按连续列分块,块内白色列正在量化、蓝色剩余列在块末
统一更新,逆 Hessian 信息存在 Cholesky 分解里。与本章手算例(1×4 lazy batch)互补:
那张是逐轮数值账、这张是全矩阵结构。

忠实重绘:布局与信息结构对齐论文原图(e-print gptq-new-2.pdf 矢量级核对)——
左=上三角 Cholesky 阶梯(带 B×B 粗框),右=权重矩阵五条带(浅橙|橙|白|蓝|浅蓝)+块粗框,
两板之间端帽连线,底部成对色块图例。配色/字体套本书视觉语言,文字译中,非像素复制。
key_figures provenance 豁免:数据 provenance = 原论文本身。
坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 650
MX = 60
BXR = 1440
GRID = '#e2e8f0'

lc.text(MX, 34, 'GPTQ 全景:一整片权重矩阵的逐块量化',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '加粗的连续列块逐步量化——逆 Hessian 信息存在 Cholesky 分解里;蓝色剩余列在块末统一更新;块内递归进行,白色中列 = 当前正在量化的列',
        10.5, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
_ch = '论文精髓图重绘 · arXiv:2210.17323 Fig.2'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 配色(语义沿原论文:橙=已量化,蓝=未量化,白=当前列) =================
F_Q_PREV, S_Q_PREV = '#ffedd5', '#fdba74'      # 前面块已量化(浅橙)
F_Q_BLK, S_Q_BLK = '#fdba74', '#ea580c'        # 本块已量化列(橙)
F_CUR, S_CUR = '#ffffff', '#0f172a'            # 当前列(白,黑描边)
F_U_BLK, S_U_BLK = '#93c5fd', '#2563eb'        # 本块未量化列(蓝)
F_U_OUT, S_U_OUT = '#dbeafe', '#93c5fd'        # 块外剩余(浅蓝)
F_HES, S_HES = '#dbeafe', '#93c5fd'            # Hessian 阶梯

# ================= 左板:逆层 Hessian(Cholesky 形) =================
LPX, LPY, LPW, LPH = MX, 96, 480, 400
lc.rect(LPX, LPY, LPW, LPH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(LPX + 16, LPY + 26, '逆层 Hessian(Cholesky 形)', 12.5, lc.C_TXT, 'start', True,
        maxw=LPW - 32, tag='lp:t')
lc.text(LPX + 16, LPY + 46, 'H⁻¹ ← Cholesky(H⁻¹)ᵀ:进场一次算好,全程只读', 9.5, lc.C_MUTE,
        'start', maxw=LPW - 32, tag='lp:s')

# 上三角阶梯:16 行,行 r 自对角线铺到右缘
NROW, RH, DX, HX0, HY0 = 16, 15, 12.6, LPX + 60, LPY + 78
HXR = HX0 + 218                                  # 阶梯右缘
for r in range(NROW):
    x0 = HX0 + r * DX
    lc.rect(x0, HY0 + r * RH, HXR - x0, RH, F_HES, S_HES, rx=1.5, sw=0.9)
# 对角阶梯边(加粗)
_diag = []
for r in range(NROW):
    _diag.append((HX0 + r * DX, HY0 + (r + 1) * RH))
_path = ' '.join(f'{"M" if i == 0 else "L"}{p[0]:.1f},{p[1]:.1f}' for i, p in enumerate(_diag))
lc.ELEMS.append(((HX0 - 4, HY0 - 4, HXR + 4, HY0 + NROW * RH + 4),
                 f'<path d="{_path}" fill="none" stroke="#2563eb" stroke-width="2.0"/>'))
# B×B 子块粗框(对角线上的一段,行 9..12 区域贴右缘)
BBX0, BBY0 = HX0 + 9 * DX, HY0 + 9 * RH
lc.rect(BBX0, BBY0, HXR - BBX0, 4 * RH, 'none', lc.C_TXT, rx=3, sw=2.6)
lc.text(HXR + 14, BBY0 + 24, '块 i 的 B×B 子块', 9.5, lc.C_TXT, 'start', True,
        maxw=LPW - (HXR - LPX) - 24, tag='lp:bb')
lc.text(HXR + 14, BBY0 + 40, '(更新限制在块内列与', 8.5, lc.C_MUTE, 'start',
        maxw=LPW - (HXR - LPX) - 24, tag='lp:bb2')
lc.text(HXR + 14, BBY0 + 54, '这块 B×B 子块里)', 8.5, lc.C_MUTE, 'start',
        maxw=LPW - (HXR - LPX) - 24, tag='lp:bb3')
lc.text(LPX + 16, HY0 + NROW * RH + 26, '量化列 j 只读第 j 行(自对角线起)', 9, lc.C_MUTE,
        'start', maxw=LPW - 32, tag='lp:row')
lc.text(LPX + 16, HY0 + NROW * RH + 44, '所有行共用同一列序与同一套补偿', 9, lc.C_MUTE,
        'start', maxw=LPW - 32, tag='lp:row2')

# ================= 右板:权重矩阵 W(按连续列分块) =================
RPX, RPY, RPW, RPH = 660, 96, 780, 400
lc.rect(RPX, RPY, RPW, RPH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(RPX + 16, RPY + 26, '权重矩阵 W(按连续列分块)', 12.5, lc.C_TXT, 'start', True,
        maxw=RPW - 32, tag='rp:t')
lc.text(RPX + 16, RPY + 46, '每列 = 一个输入通道;行 = 输出通道——所有行按同一列序一起量化', 9.5,
        lc.C_MUTE, 'start', maxw=RPW - 32, tag='rp:s')

# 五条带(x 坐标链式推进)
MY0, MH = RPY + 92, 216
BANDS = [
    ('A', 132, F_Q_PREV, S_Q_PREV),   # 前面块已量化
    ('B', 96, F_Q_BLK, S_Q_BLK),      # 本块已量化列
    ('cur', 38, F_CUR, S_CUR),        # 当前列
    ('C', 62, F_U_BLK, S_U_BLK),      # 本块未量化列
]
_bx = RPX + 40
band_x = {}
for name, bw, f_, s_ in BANDS:
    band_x[name] = (_bx, bw)
    lc.rect(_bx, MY0, bw, MH, f_, s_, rx=2, sw=1.4)
    _bx += bw
BLK_X0, BLK_X1 = band_x['B'][0], _bx          # 块 i = B+cur+C
GAP = 46
DX0 = _bx + GAP                                # 块外剩余带(隔开,给块末箭头留位)
DX1 = DX0 + 150
lc.rect(DX0, MY0, DX1 - DX0, MH, F_U_OUT, S_U_OUT, rx=2, sw=1.4)
# 块 i 粗框
lc.rect(BLK_X0, MY0 - 6, BLK_X1 - BLK_X0, MH + 12, 'none', lc.C_TXT, rx=4, sw=3.0)
lc.text((BLK_X0 + BLK_X1) / 2, MY0 - 18, '块 i(B 个连续列 × 全部行)', 10.5, lc.C_TXT,
        'middle', True, maxw=BLK_X1 - BLK_X0 + 60, tag='rp:blk')
# 当前列小标
lc.text(band_x['cur'][0] + band_x['cur'][1] / 2, MY0 + MH + 16, '当前列', 8.5, lc.C_TXT,
        'middle', True, maxw=70, tag='rp:cur')
# 块末一次性更新箭头:块右缘 → 块外剩余带左缘(两端都贴框边)
_ay = MY0 + MH / 2
lc.seg(BLK_X1, _ay, DX0, _ay, lc.C_ABORT, 3.6, 'ab')
lc.text((BLK_X1 + DX0) / 2, MY0 - 18, '块末', 9.5, lc.C_ABORT, 'middle', True, maxw=60,
        tag='rp:be1')
lc.text((BLK_X1 + DX0) / 2, _ay + 18, '一次性更新', 8.5, lc.C_ABORT, 'middle', True, maxw=70,
        tag='rp:be2')
# 块内递归标注
lc.text((BLK_X0 + BLK_X1) / 2, MY0 + MH + 40, '块内逐列递归:量化 → 记误差 → 即时补偿块内',
        9, lc.C_TXT, 'middle', True, maxw=RPW - 80, tag='rp:rec')
# 列方向轴标
lc.text(DX1 - 20, MY0 + MH + 62, '列(输入通道)→', 8.5, lc.C_MUTE, 'end', maxw=140, tag='rp:ax')

# ================= 两板之间的信息流(端帽线,沿原图) =================
_cy = LPY + 200
lc.seg(LPX + LPW, _cy, RPX, _cy, lc.C_MUTE, 2.2)
for _ex in (LPX + LPW, RPX):
    lc.rect(_ex - 4, _cy - 4, 8, 8, lc.C_MUTE, lc.C_MUTE, rx=1, sw=0)
lc.text((LPX + LPW + RPX) / 2, _cy - 12, '逆 Hessian 信息', 9, lc.C_MUTE, 'middle', True,
        maxw=110, tag='mid:1')
lc.text((LPX + LPW + RPX) / 2, _cy + 16, '存在 Cholesky 分解里', 8.5, lc.C_MUTE, 'middle',
        maxw=110, tag='mid:2')

# ================= 图例(成对色块,沿原图) =================
LGY = 516
LX = MX
LEG = [
    ([(F_Q_PREV, S_Q_PREV), (F_Q_BLK, S_Q_BLK)], '已量化权重(前面块 · 本块已量化列)'),
    ([(F_CUR, S_CUR)], '当前列:正在量化'),
    ([(F_U_BLK, S_U_BLK), (F_U_OUT, S_U_OUT)], '未量化权重(块末统一更新)'),
]
for swatches, name in LEG:
    x = LX
    for f_, s_ in swatches:
        lc.rect(x, LGY, 17, 13, f_, s_, rx=2.5, sw=1.1)
        x += 19
    lc.text(x + 4, LGY + 11, name, 9.5, '#334155', 'start', maxw=400, tag='leg:' + name[:10])
    LX = x + 4 + lc.tw(name, 9.5) + 34

# ================= 底部:读图注 =================
lc.text(MX, 566, '读图:块 i 内逐列「量化 → 记误差 → 补偿块内」;整块完成后,误差账(Cholesky 行)一次性更新全部蓝色剩余列。',
        9.5, '#334155', 'start', maxw=1380, tag='note1')
lc.text(MX, 586, '正文的手算例(1×4 lazy batch)就是把『块 i』放大成逐列数值账;本图补上它看不到的全矩阵结构。',
        9.5, '#334155', 'start', maxw=1380, tag='note2')

# ================= 页脚 =================
lc.text(MX, 626, '重绘自 arXiv:2210.17323 Fig.2(§4):布局与信息结构对齐论文原图,配色/文字套本书视觉语言 · 机制释义 = 原 caption 逐字译写',
        9, lc.C_FAINT, 'start', maxw=1380, tag='foot')

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'paper-fig-1.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
