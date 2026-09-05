#!/usr/bin/env python3
"""paper-fig-4 · SmoothQuant 论文精髓图重绘(arXiv:2211.10438 Fig.3,writer 图集变更 add)

三种量化粒度的定义图:per-tensor(整阵一把尺)/ per-token(每行一把,挂 token 维 T)/
per-channel(每列一把,挂输出通道维 C_o)——并标出 GEMM 只认外维(T、C_o)缩放、
内维(输入通道维 C_i)不可行。与本章粒度数值图互补:那张是数值坍缩对照+GroupShape 词汇,
这张是论文原版定义坐标(T/C_i/C_o 轴标)。

忠实重绘:四张矩阵横排(X per-tensor / X per-token / W per-channel / W per-token 灰色不可行),
绿框 = 一把 Δ 的共享范围;底部 GEMM 缩放位置示意。配色/字体套本书视觉语言,文字译中,
非像素复制。key_figures provenance 豁免:数据 provenance = 原论文本身。
坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 680
MX = 60
BXR = 1440
GRID = '#e2e8f0'

lc.text(MX, 34, '三种量化粒度 = scale 沿哪根轴共享',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, 'per-tensor 整片一把 · per-token 每行一把(挂外维 T)· per-channel 每列一把(挂外维 C_o)——INT8 GEMM 只认外维缩放;挂在内维 C_i 的尺精度最好却不可行',
        10.5, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
_ch = '论文精髓图重绘 · arXiv:2211.10438 Fig.3'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

NROW, NCOL = 5, 6
CW_, CH_ = 34, 25
MAT_W, MAT_H = NCOL * CW_, NROW * CH_
PANEL_W = 330
PANELS_X = [MX, MX + PANEL_W + 20, MX + 2 * (PANEL_W + 20), MX + 3 * (PANEL_W + 20)]
PY, PH = 100, 330
MAT_Y = PY + 78

FR = {'tensor': 'whole', 'token': 'rows', 'channel': 'cols', 'bad': 'rows'}
PANELS = [
    ('per-tensor', '整片矩阵一把 Δ(最省)', 'X', 'T', 'C_i', 'tensor', False,
     '一把尺量所有元素'),
    ('per-token', '每 token 一把(挂外维 T)', 'X', 'T', 'C_i', 'token', False,
     'GEMM 出口可逐行乘回'),
    ('per-channel', '每输出通道一把(挂外维 C_o)', 'W', 'C_i', 'C_o', 'channel', False,
     'GEMM 出口可逐列乘回'),
    ('per-token', '每输入通道一把(挂内维 C_i)', 'W', 'C_i', 'C_o', 'bad', True,
     '尺挂内维 → INT8 GEMM 不收'),
]

for p, (title, sub, mat, ldim, bdim, frame, bad, verdict) in enumerate(PANELS):
    px = PANELS_X[p]
    lc.rect(px, PY, PANEL_W, PH, '#f8fafc' if bad else '#ffffff',
            lc.C_FAINT if bad else GRID, rx=8, sw=1.2, dash=bad)
    if bad:
        bw = lc.tw('内维 ✗', 9, True) + 14
        lc.rect(px + PANEL_W - bw - 8, PY + 8, bw, 18, '#fff7ed', lc.C_ENG_S, rx=8, sw=1.1)
        lc.text(px + PANEL_W - bw / 2 - 8, PY + 21, '内维 ✗', 9, lc.C_ENG_S, 'middle', True,
                maxw=bw - 4, tag='badge' + str(p))
    lc.text(px + 14, PY + 26, title, 12.5, lc.C_TXT if not bad else lc.C_MUTE, 'start', True,
            maxw=PANEL_W - 60, tag='pt' + str(p))
    lc.text(px + 14, PY + 46, sub, 9, lc.C_MUTE, 'start', maxw=PANEL_W - 28, tag='ps' + str(p))
    mx0 = px + (PANEL_W - MAT_W) / 2
    fc = lc.C_FAINT if bad else lc.C_GPU_S
    # 矩阵格
    for r in range(NROW):
        for c in range(NCOL):
            lc.rect(mx0 + c * CW_, MAT_Y + r * CH_, CW_ - 2, CH_ - 3,
                    '#f1f5f9' if bad else '#eef2f7', '#cbd5e1', rx=2, sw=0.9)
    # 共享范围框
    if frame == 'whole':
        lc.rect(mx0 - 4, MAT_Y - 4, MAT_W + 8, MAT_H + 7, 'none', fc, rx=5, sw=2.4)
    elif frame == 'rows':
        for r in range(NROW):
            lc.rect(mx0 - 4, MAT_Y + r * CH_ - 4, MAT_W + 8, CH_ + 5, 'none', fc, rx=5,
                    sw=1.8, dash=bad)
    else:
        for c in range(NCOL):
            lc.rect(mx0 + c * CW_ - 4, MAT_Y - 4, CW_ + 6, MAT_H + 7, 'none', fc, rx=5, sw=1.8)
    # 轴标
    lc.text(mx0 - 12, MAT_Y + MAT_H / 2, ldim, 9, lc.C_MUTE, 'end', maxw=40,
            tag='axl' + str(p))
    lc.text(mx0 + MAT_W / 2, MAT_Y + MAT_H + 16, bdim, 9, lc.C_MUTE, 'middle', maxw=40,
            tag='axb' + str(p))
    lc.text(mx0 + MAT_W + 12, MAT_Y + 14, mat, 11, lc.C_TXT, 'start', True, maxw=30,
            tag='axm' + str(p))
    # 判词
    vy = MAT_Y + MAT_H + 44
    vw = lc.tw(verdict, 9.5, True) + 18
    lc.rect(px + (PANEL_W - vw) / 2, vy - 14, vw, 20,
            '#f8fafc' if bad else '#f0fdf4', fc, rx=9, sw=1.2, dash=bad)
    lc.text(px + PANEL_W / 2, vy, verdict, 9.5, fc, 'middle', True, maxw=vw - 6,
            tag='vd' + str(p))

# ================= 底部:GEMM 缩放位置示意 =================
BY, BH = 452, 168
lc.rect(MX, BY, 1380, BH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(MX + 16, BY + 24, '为什么内维不行:GEMM 的 scale 只能挂在「乘法出口」上', 11, lc.C_TXT,
        'start', True, maxw=800, tag='b:h')
# 三只小矩阵 + 乘号
gm_y = BY + 44
gm_h = 64
mm_w = 44


def mini_mat(x, label, dims, notes):
    """notes = [(text, color), ...] 逐行叠在矩阵下方。"""
    lc.rect(x, gm_y, mm_w, gm_h, '#eef2f7', '#94a3b8', rx=3, sw=1.2)
    lc.text(x + mm_w / 2, gm_y + gm_h / 2 + 4, label, 10, lc.C_TXT, 'middle', True, maxw=36,
            tag='mm' + label)
    lc.text(x + mm_w / 2, gm_y - 8, dims, 8.5, lc.C_MUTE, 'middle', maxw=80,
            tag='mmd' + label)
    for i, (txt, col) in enumerate(notes):
        lc.text(x + mm_w / 2, gm_y + gm_h + 14 + i * 14, txt, 8.5, col, 'middle', True,
                maxw=150, tag='mmn' + label + str(i))


gx = MX + 60
mini_mat(gx, 'X', 'T × C_i', [('✓ T(外维)', lc.C_GPU_S), ('✗ C_i(内维)', lc.C_ABORT)])
lc.text(gx + mm_w + 16, gm_y + gm_h / 2 + 4, '·', 14, lc.C_MUTE, 'middle', True, maxw=20,
        tag='dot1')
gx2 = gx + mm_w + 42
mini_mat(gx2, 'W', 'C_i × C_o', [('✗ C_i(内维)', lc.C_ABORT), ('✓ C_o(外维)', lc.C_GPU_S)])
lc.text(gx2 + mm_w + 16, gm_y + gm_h / 2 + 4, '=', 14, lc.C_MUTE, 'middle', True, maxw=20,
        tag='eq1')
gx3 = gx2 + mm_w + 44
mini_mat(gx3, 'Y', 'T × C_o', [('出口只剩 T、C_o', lc.C_GPU_S)])
# 说明
lc.text(gx3 + mm_w + 40, gm_y + 12, 'INT8 Tensor Core 一次算完 X̄·W̄,scale 只能在出口逐个乘回:', 9.5,
        lc.C_TXT, 'start', maxw=560, tag='b:e1')
lc.text(gx3 + mm_w + 40, gm_y + 32, 'Y = diag(Δ_X) · ( X̄ · W̄ ) · diag(Δ_W)', 9.5, lc.C_API_S,
        'start', True, maxw=560, tag='b:e2')
lc.text(gx3 + mm_w + 40, gm_y + 52, 'T 维逐行、C_o 维逐列——恰是 per-token / per-channel;', 9,
        lc.C_MUTE, 'start', maxw=560, tag='b:e3')
lc.text(gx3 + mm_w + 40, gm_y + 70, 'C_i 在乘法内部被累加消掉,出口没有它——尺挂 C_i 插不进出口', 9,
        lc.C_MUTE, 'start', maxw=560, tag='b:e4')

# ================= 页脚 =================
lc.text(MX, 656, '重绘自 arXiv:2211.10438 Fig.3(§2 定义图)· 出口缩放公式 = 原文 §3 Eq.2 · 与本章粒度数值图互补:本图 = 论文原版定义坐标(T/C_i/C_o 轴)',
        9, lc.C_FAINT, 'start', maxw=1380, tag='foot')

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'paper-fig-4.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
