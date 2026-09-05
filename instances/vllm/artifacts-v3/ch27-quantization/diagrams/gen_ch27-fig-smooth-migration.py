#!/usr/bin/env python3
"""ch27 机制图 6 · SmoothQuant 搬家:难度从激活搬进权重(figure_spec ch27-fig-smooth-migration,模板 before-after)

放大自 L0 GPU 执行臂(绿)第三块『模型层 forward + 编译』里 Linear 层的输入侧——
SmoothQuant 之后进入这块的激活已是平滑过的 X̂(运行期零开销,图上无任何运行期算子)。
推导链第 6 环,不画架构元素。

claim:α=0.5 精确配平:逐通道 s_j=√(max|X_j|/max|W_j|) 使 max|X̂_j|==max|Ŵ_j|
(本例 4 通道两两相等),激活通道间 96.2 倍差距压到 6.9 倍,W8A8 误差 0.4404→0.2326;
严格等价(max|X̂Ŵ−XW|=0.0),s 离线折进前一层、运行期零开销。

数字全部取自本章参考实现实跑(与 explainer 素材同源);坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 782
MX = 60
BXR = 1440
GRID = '#e2e8f0'

lc.text(MX, 34, 'SmoothQuant 搬家:难度守恒,但可以从激活搬进权重',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '逐通道一对儿 ÷s_j / ×s_j:乘法结果浮点级不变(max|X̂·Ŵ−X·W| = 0.0);α=0.5 精确配平 max|X̂_j| == max|Ŵ_j|,激活通道间 96.2 倍 → 6.9 倍,W8A8 误差 0.4404 → 0.2326',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L0 GPU 执行臂 · 模型层 forward + 编译 · Linear 层输入侧'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 数据(trace 原样) =================
CH0 = [70, 63, 77, 66, 59, 72]                     # 6 token × 通道 0
MAXX = ['77.0', '0.9', '1.2', '0.8']
MAXW = ['0.5', '2.0', '1.0', '1.0']
S_J = ['12.4097', '0.6708', '1.0954', '0.8944']
EQ = ['6.2048', '1.3416', '1.0954', '0.8944']      # max|X̂_j| == max|Ŵ_j|
ALPHAS = [0.0, 0.12, 0.25, 0.38, 0.5, 0.62, 0.75, 0.88, 1.0]
ERRS = [0.44, 0.42, 0.30, 0.29, 0.23, 0.28, 0.39, 0.37, 0.27]

CWD, CHT = 54, 28            # X 热图单元格
XHM_X, XHM_Y = 90, 168       # X 热图左上


def x_heat(x0, y0, equalized):
    """X/X̂ 热图:6 token × 4 通道;通道 0 热(搬迁后减弱),其余浅。"""
    for r in range(6):
        for c in range(4):
            x, y = x0 + c * CWD, y0 + r * CHT
            if c == 0:
                f_, k_ = ('#fed7aa' if equalized else '#fdba74'), lc.C_ENG_S
            else:
                f_, k_ = '#e8edf3', '#cbd5e1'
            lc.rect(x, y, CWD - 2, CHT - 2, f_, k_, rx=3, sw=0.9)
            if not equalized:
                lc.text(x + CWD / 2 - 1, y + CHT / 2 + 2.5, str(CH0[r]), 8, '#7c2d12',
                        'middle', True, tag='xh' + str(r))


def w_heat(x0, y0, rowmax):
    """W/Ŵ 热图:4 输入通道 × 2 输出通道,平坦。"""
    for r in range(4):
        for c in range(2):
            x, y = x0 + c * CWD, y0 + r * 24
            lc.rect(x, y, CWD - 2, 22, '#e8edf3', '#cbd5e1', rx=3, sw=0.9)
        lc.text(x0 - 8, y0 + r * 24 + 14, rowmax[r], 8.2, lc.C_MUTE, 'end',
                tag='wh' + str(r) + rowmax[r])


# ================= 左:X 与 W(搬家前) =================
lc.text(XHM_X + 2 * CWD, 138, 'X(6 token × 4 通道):通道 0 一柱擎天', 10, lc.C_TXT, 'middle',
        True, maxw=300, tag='x:t')
for c in range(4):
    lc.text(XHM_X + c * CWD + CWD / 2, 158, f'max {MAXX[c]}', 7.8,
            lc.C_ENG_S if c == 0 else lc.C_MUTE, 'middle', tag='xm' + str(c))
x_heat(XHM_X, XHM_Y, equalized=False)
WHM_X, WHM_Y = XHM_X, XHM_Y + 6 * CHT + 46
lc.text(XHM_X + CWD, WHM_Y - 10, 'W(4×2):分布平坦,行 max →', 9.5, lc.C_TXT, 'middle', True,
        maxw=280, tag='w:t')
w_heat(WHM_X, WHM_Y, MAXW)

# ================= 中:逐通道 s 表 =================
MTX, MTY, MTW, MTH = 380, 150, 350, 210
lc.rect(MTX, MTY, MTW, MTH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(MTX + 14, MTY + 20, '逐通道搬家系数 α=0.5', 10.5, lc.C_TXT, 'start', True,
        maxw=MTW - 28, tag='mt:h')
lc.text(MTX + 14, MTY + 38, 's_j = √(max|X_j| / max|W_j|)', 9, lc.C_MUTE, 'start',
        maxw=MTW - 28, tag='mt:f')
MHDR = [('j', 'start', 16), ('max|X_j|', 'end', 126), ('max|W_j|', 'end', 192),
        ('s_j', 'end', 252), ('max|X̂_j|=max|Ŵ_j|', 'end', 338)]
lc.rect(MTX + 8, MTY + 48, MTW - 16, 20, '#f1f5f9', GRID, rx=4, sw=0.8)
for name, anc, dx in MHDR:
    lc.text(MTX + dx, MTY + 62, name, 7.8, lc.C_MUTE, anc, maxw=100,
            tag='mth:' + name[:4])
for i in range(4):
    yy = MTY + 84 + i * 24
    vals = [str(i), MAXX[i], MAXW[i], S_J[i], EQ[i]]
    for j, (name, anc, dx) in enumerate(MHDR):
        col = '#334155'
        bold = False
        if j == 4:
            col, bold = lc.C_GPU_S, True
        if j == 0:
            col, bold = (lc.C_ENG_S if i == 0 else lc.C_MUTE), True
        lc.text(MTX + dx, yy + 2, vals[j], 8.8, col, anc, bold=bold,
                tag='mtv' + str(i) + str(j))
lc.text(MTX + 14, MTY + MTH - 12, '配平恒等式:max|X̂_j| = max|Ŵ_j| = √(max|X_j|·max|W_j|)',
        8.2, lc.C_GPU_S, 'start', True, maxw=MTW - 28, tag='mt:eq')

# 搬家箭头:X →(÷s_j)→ 表 →(结果)→ X̂ ;W →(×s_j)→ Ŵ
ay1 = XHM_Y + 3 * CHT
lc.seg(XHM_X + 4 * CWD, ay1, MTX, ay1, lc.C_ENG_S, 2.2, 'std')
lc.text((XHM_X + 4 * CWD + MTX) / 2, ay1 - 10, '激活 ÷ s_j', 8.5, lc.C_ENG_S, 'middle',
        True, tag='ar:x')
aw1 = WHM_Y + 2 * 24
lc.seg(WHM_X + 2 * CWD, aw1, MTX, aw1, lc.C_ZMQ_S, 2.2, 'std')
lc.text((WHM_X + 2 * CWD + MTX) / 2, aw1 + 14, '权重 × s_j', 8.5, lc.C_ZMQ_S, 'middle',
        True, tag='ar:w')

# ================= 右:X̂ 与 Ŵ(搬家后) =================
XH_X = 820
lc.text(XH_X + 2 * CWD, 138, 'X̂:通道间只剩 6.9 倍(96.2 → 6.9)', 10, lc.C_GPU_S, 'middle',
        True, maxw=320, tag='xh:t')
for c in range(4):
    lc.text(XH_X + c * CWD + CWD / 2, 158, f'max {EQ[c]}', 7.8, lc.C_GPU_S, 'middle',
            tag='xhm' + str(c))
x_heat(XH_X, XHM_Y, equalized=True)
WH_X, WH_Y = XH_X, WHM_Y
lc.text(WH_X + CWD, WH_Y - 10, 'Ŵ:行 max 同步抬到 →', 9.5, lc.C_GPU_S, 'middle', True,
        maxw=280, tag='wh:t')
w_heat(WH_X, WH_Y, EQ)
# 表 → 右侧结果箭头
lc.seg(MTX + MTW, MTY + MTH / 2, XH_X - 6, MTY + MTH / 2, lc.C_GPU_S, 2.2, 'std')
lc.text((MTX + MTW + XH_X) / 2, MTY + MTH / 2 - 10, '两边都落进', 8.5, lc.C_GPU_S, 'middle',
        tag='ar:r1')
lc.text((MTX + MTW + XH_X) / 2, MTY + MTH / 2 + 4, 'W8A8 能力圈', 8.5, lc.C_GPU_S, 'middle',
        tag='ar:r2')

# ================= 右侧:等价性 + 真实收益 =================
EQX, EQY, EQW, EQH = 1100, 150, 340, 210
lc.rect(EQX, EQY, EQW, EQH, '#f0fdf4', lc.C_GPU_S, rx=8, sw=1.3)
lc.text(EQX + 14, EQY + 22, '严格无损 + 零开销', 10.5, lc.C_GPU_S, 'start', True,
        maxw=EQW - 28, tag='eq:h')
EQ_LINES = [
    'max|X̂·Ŵ − X·W| = 0.0(12 位小数)',
    '每个求和项 s_j 与 1/s_j 严格相消——',
    '对任意正 s_j 成立,迁移是免费的',
    '',
    's 离线折进前一层(LayerNorm/Linear',
    '参数):运行期零开销,在线多算一次',
    '乘法都没有',
    '',
    'W8A8 per-tensor 误差:0.4404(α=0)',
    '→ 0.2326(α=0.5,1.9 倍)vs 0.2742(α=1)',
]
for i, ln in enumerate(EQ_LINES):
    if ln:
        lc.text(EQX + 14, EQY + 44 + i * 16.5, ln, 8.5, '#334155', 'start', maxw=EQW - 28,
                tag='eql' + str(i))

# ================= 底部:α 扫描曲线 + 论文甜点 =================
APX, APY, APW, APH = MX, 500, 930, 210
lc.rect(APX, APY, APW, APH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(APX + 16, APY + 22, '搬家力度 α:s_j = max|X_j|^α / max|W_j|^(1−α) —— 9 点扫描,U 形甜点 0.5',
        10.5, lc.C_TXT, 'start', True, maxw=APW - 28, tag='ap:h')
lc.text(APX + 16, APY + 40, 'α=0 不搬(离群全留激活)· α=1 全搬(激活全平、权重爆:W 行 max 变 [38.5, 1.8, 1.2, 0.8])',
        8.5, lc.C_MUTE, 'start', maxw=APW - 28, tag='ap:sub')
GX0, GX1 = APX + 60, APX + APW - 40
GY0, GY1 = APY + 66, APY + APH - 58
EMIN, EMAX = 0.18, 0.50


def gxa(a):
    return GX0 + a / 1.0 * (GX1 - GX0)


def gya(e):
    return GY0 + (e - EMIN) / (EMAX - EMIN) * (GY1 - GY0)


lc.rect(GX0, GY1, GX1 - GX0, 2, '#334155', '#334155', rx=1, sw=0)
lc.rect(GX0, GY0, 2, GY1 - GY0, '#334155', '#334155', rx=1, sw=0)
pts = [(gxa(a), gya(e)) for a, e in zip(ALPHAS, ERRS)]
path = ' '.join(f'{"M" if i == 0 else "L"}{p[0]:.1f},{p[1]:.1f}' for i, p in enumerate(pts))
lc.ELEMS.append(((GX0 - 4, GY0 - 4, GX1 + 4, GY1 + 4),
                 f'<path d="{path}" fill="none" stroke="{lc.C_API_S}" stroke-width="2.2"/>'))
for a in (0.0, 0.25, 0.5, 0.75, 1.0):
    lc.text(gxa(a), GY1 + 16, f'{a:g}', 8.5, lc.C_MUTE, 'middle', tag='ap:x' + str(a))
lc.text(GX1 + 10, GY1 + 16, 'α', 8.5, lc.C_MUTE, 'start', tag='ap:xax')
lc.text(GX0 - 10, GY0 + 8, 'W8A8 误差', 8.5, lc.C_MUTE, 'end', tag='ap:yax')
ANCH = [
    (0.0, 0.44, '0.4404', lc.C_ABORT, 'start', 10),
    (0.5, 0.23, '0.2326(甜点,9 点 argmin=0.5)', lc.C_GPU_S, 'start', -12),
    (1.0, 0.27, '0.2742', lc.C_ABORT, 'end', -12),
]
for a, e, lab, col, anc, dy in ANCH:
    lc.rect(gxa(a) - 4, gya(e) - 4, 8, 8, col, col, rx=4, sw=1)
    lc.text(gxa(a) + (8 if anc == 'start' else -8), gya(e) + dy, lab, 8.5, col, anc, True,
            maxw=280, tag='ap:p' + str(a))
# GLM-130B 注
lc.text(GX0 + 20, GY1 - 18, '论文:OPT/BLOOM 通用 0.5;GLM-130B(离群 ~30% 通道)用 0.75',
        8.2, lc.C_MUTE, 'start', maxw=560, tag='ap:glm')

# 右下:真实收益
RPX, RPY, RPW, RPH = 1020, 500, 420, 210
lc.rect(RPX, RPY, RPW, RPH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(RPX + 14, RPY + 22, '真实收益口径(论文 Table 3,OPT-175B)', 10.5, lc.C_TXT, 'start',
        True, maxw=RPW - 28, tag='rp:h')
RP = [
    ('FP16', 'PPL 10.99', lc.C_API_S),
    ('W8A8 per-tensor(不搬)', 'PPL 93080(崩溃)', lc.C_ABORT),
    ('SmoothQuant O3(α=0.5)', 'PPL 11.17(追平 FP16)', lc.C_GPU_S),
]
for i, (n, v, col) in enumerate(RP):
    yy = RPY + 50 + i * 44
    lc.rect(RPX + 14, yy, RPW - 28, 36, '#ffffff', col, rx=6, sw=1.4)
    lc.text(RPX + 26, yy + 15, n, 9, lc.C_TXT, 'start', True, maxw=RPW - 60, tag='rpn' + str(i))
    lc.text(RPX + 26, yy + 29, v, 8.8, col, 'start', True, maxw=RPW - 60, tag='rpv' + str(i))
lc.text(RPX + 14, RPY + RPH - 14, '线性层 Frobenius 口径 ~2 倍;PPL 口径收益远大于此', 8.2,
        lc.C_MUTE, 'start', maxw=RPW - 28, tag='rp:note')

# ================= 页脚 =================
lc.text(MX, RPY + RPH + 26, '论文口径 arXiv:2211.10438 §4 Eq.3-Eq.4 · Figure 2(搬家直觉)/Figure 5(α=0.5 主例)· §5.1 Table 3 · 数值:本章 NumPy 参考实现实跑 · 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=1380, tag='foot')

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch27-fig-smooth-migration.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
