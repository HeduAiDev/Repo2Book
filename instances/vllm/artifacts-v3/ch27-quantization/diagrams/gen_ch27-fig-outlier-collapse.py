#!/usr/bin/env python3
"""ch27 机制图 3 · RTN 之死:离群通道撑爆全场共用的 Δ(figure_spec ch27-fig-outlier-collapse,模板 before-after)

放大自 L0 GPU 执行臂(绿)第三块『模型层 forward + 编译』里 Linear 层的输入激活/权重——
为什么进 kernel 的整数码会失真,病根在量化之前。推导链第 3 环,不画架构元素。

claim:一个 ~70 的离群通道把 per-tensor 的尺子撑爆:普通通道(值 ~1)只剩 5.0-8.12 个
有效台阶,8 个原始值坍缩到 ±0.61 的 3 个倍数上;per-channel 一把尺/通道则全部保住
(误差 52 倍差距)——『聪明量化』的全部动机。

数字全部取自本章参考实现实跑(与 explainer 素材同源);坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 640
MX = 60
BXR = 1440
GRID = '#e2e8f0'

lc.text(MX, 34, 'RTN 之死:一颗老鼠屎坏一锅粥——离群通道不改别人的值,只改大家共用的 Δ',
        16.5, lc.C_TXT, 'start', True, maxw=1060, tag='title')
lc.text(MX, 58, '有效级数 = 2^N·m_i/m:通道 0 逐 token 持续 ~70,把全场尺子定在 m=77;普通通道(值 ~1)被压到 5-8 个台阶,8 个值坍缩到 ±0.61 的倍数上;per-channel 每 channel 一把尺全部保住',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L0 GPU 执行臂 · 模型层 forward + 编译 · Linear 层输入/权重'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 数据(trace 原样) =================
C0 = [70, 63, 77, 66, 59, 72, 68, 74]                      # 通道 0(离群)
CH1_ORIG = [-0.52, 1.14, -0.55, -0.79, -0.61, 0.19, 0.78, -1.5]
CH1_TENSOR = [-0.61, 1.21, -0.61, -0.61, -0.61, 0.0, 0.61, -1.21]
CH1_CHANNEL = [-0.52, 1.15, -0.56, -0.79, -0.6, 0.19, 0.78, -1.5]
ACCOUNT = [
    ('0(离群)', '77.0', '256.0', '0.1102', '0.1102'),
    ('1(普通)', '1.5', '5.0', '0.131', '0.0021'),
    ('2(普通)', '2.06', '6.84', '0.217', '0.0034'),
    ('3(普通)', '2.44', '8.12', '0.1712', '0.0045'),
]


def heat_c1(v):
    """通道 1 值的语义着色:正=暖、负=冷,两档强度(值来自 trace,无杜撰)。"""
    if v >= 1.0:
        return '#fed7aa', '#f59e0b'
    if v > 0:
        return '#ffedd5', '#fdba74'
    if v > -1.0:
        return '#e0f2fe', '#7dd3fc'
    return '#bae6fd', '#38bdf8'


# ================= 三条热图条 =================
CELL_W, CELL_H = 56, 27
STRIPS = [
    ('① 原始(8 token × 4 通道)', C0, CH1_ORIG, '#ffffff', GRID, False),
    ('② per-tensor 反量化(Δ=0.6063 全场一把)', C0, CH1_TENSOR, '#fff7ed', lc.C_ENG_S, True),
    ('③ per-channel 反量化(Δ_ch1=0.0118)', C0, CH1_CHANNEL, '#f0fdf4', lc.C_GPU_S, False),
]
SX0, SY0 = MX, 122
STRIP_GAP = 76
STRIP_W = 4 * CELL_W + 8
strips_x = []
for s, (title, c0, c1, fill, stroke, hot) in enumerate(STRIPS):
    sx = SX0 + s * (STRIP_W + STRIP_GAP)
    strips_x.append(sx)
    lc.rect(sx, SY0, STRIP_W, 8 * CELL_H + 56, fill, stroke, rx=8,
            sw=2.0 if hot else 1.2)
    lc.text(sx + STRIP_W / 2, SY0 + 20, title, 9.5, lc.C_TXT, 'middle', True,
            maxw=STRIP_W + 70, tag='st' + str(s))
    for c in range(4):
        lc.text(sx + 4 + c * CELL_W + CELL_W / 2, SY0 + 38, f'c{c}', 8.5, lc.C_MUTE, 'middle',
                tag='sc' + str(s) + str(c))
    for r in range(8):
        for c in range(4):
            x = sx + 4 + c * CELL_W
            y = SY0 + 46 + r * CELL_H
            if c == 0:
                f_, k_ = '#fdba74', lc.C_ENG_S
            elif c == 1:
                f_, k_ = heat_c1(c1[r])
            else:
                f_, k_ = '#eef2f7', '#cbd5e1'
            lc.rect(x, y, CELL_W - 2, CELL_H - 2, f_, k_, rx=3, sw=0.9)
            if c == 0 and s == 0:
                lc.text(x + CELL_W / 2 - 1, y + CELL_H / 2 + 2.5, str(c0[r]), 8,
                        '#7c2d12', 'middle', True, tag='c0v' + str(r))
            if c == 1:
                lc.text(x + CELL_W / 2 - 1, y + CELL_H / 2 + 2.5, f'{c1[r]:g}', 7.8, '#334155',
                        'middle', tag='c1v' + str(s) + str(r))
# 条间箭头
for s in range(2):
    ax0 = strips_x[s] + STRIP_W
    ax1 = strips_x[s + 1]
    ay = SY0 + 8 * CELL_H / 2 + 30
    lc.seg(ax0, ay, ax1, ay, lc.C_MUTE, 1.8, 'std')
    lc.text((ax0 + ax1) / 2, ay - 10, ['量化-反量化', '换粒度'][s], 8.5, lc.C_MUTE, 'middle',
            tag='al' + str(s))
lc.text(SX0, SY0 + 8 * CELL_H + 76,
        'c0 离群:逐 token 持续 ~70(59-77)· c2/c3 同为普通通道(值 ~1)· 离群通道两种粒度逐码相同——它就是 per-tensor 尺子的定尺者',
        8.5, lc.C_MUTE, 'start', maxw=860, tag='cols:note')

# ================= 右上:逐通道账 =================
TAX, TAY, TAW, TAH = 950, 122, 490, 252
lc.rect(TAX, TAY, TAW, TAH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(TAX + 14, TAY + 22, '逐通道账(m = 77.0,INT8)', 11, lc.C_TXT, 'start', True,
        maxw=TAW - 28, tag='acc:h')
HDRS = [('通道', 'start', 16), ('max m_j', 'start', 96), ('有效级数 256·m_j/m', 'start', 170),
        ('per-tensor 误差', 'end', 392), ('per-channel 误差', 'end', 478)]
lc.rect(TAX + 8, TAY + 32, TAW - 16, 22, '#f1f5f9', GRID, rx=4, sw=0.8)
for name, anchor, dx in HDRS:
    lc.text(TAX + dx, TAY + 47, name, 8, lc.C_MUTE, anchor,
            maxw=150 if dx == 170 else 118, tag='th:' + name[:6])
for i, row in enumerate(ACCOUNT):
    yy = TAY + 76 + i * 30
    outlier = (i == 0)
    if outlier:
        lc.rect(TAX + 8, yy - 12, TAW - 16, 24, '#fff7ed', lc.C_ENG_S, rx=4, sw=1.0)
    for j, (name, anchor, dx) in enumerate(HDRS):
        v = row[j]
        col, bold = '#334155', False
        if not outlier:
            if j == 2:
                col, bold = lc.C_ABORT, True
            elif j == 3:
                col = lc.C_ABORT
            elif j == 4:
                col, bold = lc.C_GPU_S, True
        lc.text(TAX + dx, yy + 4, v, 8.8, col, anchor, bold=bold, tag='tv' + str(i) + str(j))
lc.text(TAX + TAW / 2, TAY + 76 + 4 * 30 + 20,
        '普通通道:per-channel 全部保住,与 per-tensor 差 52 倍', 9.5,
        lc.C_GPU_S, 'middle', True, maxw=TAW - 28, tag='acc:v')

# ================= 右下:公式自检 =================
FY2 = 396
lc.rect(TAX, FY2, TAW, 122, '#ffffff', lc.C_MUTE, rx=8, sw=1.1, dash=True)
lc.text(TAX + 14, FY2 + 20, '公式自检:有效级数 = 2^N · m_i / m', 10, lc.C_TXT, 'start', True,
        maxw=TAW - 28, tag='f:h')
CHECKS = [
    ('m_i = 77(离群自身)', '2^8·(77/77) = 256'),
    ('m_i = 38.5(减半)', '2^8·(38.5/77) = 128'),
    ('离群 70 倍的普通值', '2^8·(1/70) = 3.66 · 2^4·(1/70) = 0.229'),
]
for i, (a, b) in enumerate(CHECKS):
    yy = FY2 + 42 + i * 22
    lc.text(TAX + 14, yy, a, 8.8, '#334155', 'start', tag='fa' + str(i))
    lc.text(TAX + 196, yy, b, 8.8, lc.C_TXT, 'start', True, tag='fb' + str(i))
lc.text(TAX + 14, FY2 + 42 + 3 * 22 + 8, '级数随 m_i 线性缩水、随离群 m 反比恶化——对一切 m_i 成立',
        8.5, lc.C_MUTE, 'start', maxw=TAW - 28, tag='f:note')

# ================= 左下:论文口径 =================
BY = 430
lc.rect(MX, BY, 860, 132, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(MX + 16, BY + 22, '真实口径:同 bit、同一副网格,round 方式是唯一变量(GPTQ §5 Table 3,OPT-175B Wiki2 PPL)',
        10, lc.C_TXT, 'start', True, maxw=830, tag='pp:h')
PPL = [
    ('FP16', '8.34', lc.C_API_S),
    ('RTN 4-bit', '10.54(还认字但变傻)', lc.C_ABORT),
    ('GPTQ 4-bit', '8.37', lc.C_GPU_S),
    ('RTN 3-bit', '7.3e3(痴呆)', lc.C_ABORT),
    ('GPTQ 3-bit', '8.68', lc.C_GPU_S),
]
bx = MX + 24
for name, val, col in PPL:
    wch = max(lc.tw(name, 9.5, True), lc.tw(val, 9.5)) + 22
    lc.rect(bx, BY + 36, wch, 42, '#ffffff', col, rx=6, sw=1.4)
    lc.text(bx + wch / 2, BY + 51, name, 9.5, lc.C_TXT, 'middle', True, tag='ppn:' + name)
    lc.text(bx + wch / 2, BY + 69, val, 9, col, 'middle', True, maxw=wch - 6,
            tag='ppv:' + name)
    bx += wch + 14
lc.text(MX + 16, BY + 102, '离群不是合成数据的特产:OPT-13B 线性层输入若干通道幅度 >70、逐 token 持续',
        8.8, lc.C_MUTE, 'start', maxw=560, tag='pp:n1')
lc.text(MX + 590, BY + 102, '(SmoothQuant §3 Figure 4)——与上图同量级', 8.8, lc.C_MUTE,
        'start', maxw=260, tag='pp:n2')

# ================= 页脚 =================
lc.text(MX, BY + 152, '数字:本章 NumPy 参考实现实跑(8×4,通道 0 显式值、其余固定种子) · 论文口径 arXiv:2210.17323 §5 Table 3 · arXiv:2211.10438 §3 obs.2/Figure 4 · 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=1380, tag='foot')

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch27-fig-outlier-collapse.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
