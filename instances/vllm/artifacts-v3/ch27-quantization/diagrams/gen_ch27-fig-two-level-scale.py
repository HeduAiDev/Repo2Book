#!/usr/bin/env python3
"""ch27 机制图 8 · 两级 scale:FP4 的一把秤称不动整张权重(figure_spec ch27-fig-two-level-scale,模板 before-after)

放大自 L0 GPU 执行臂(绿)第三块『模型层 forward + 编译』里量化 Linear 的权重格式——
modelopt.py 装载的 NVFP4 三件套就是这两级秤。推导链第 8 环,直供 ch28,不画架构元素。

claim:e2m1 只有 16 个格点(块内动态范围 6/0.5=12):单级全局 scale 下,16 值全 ~0.1 的
小块在张量 amax=6 的尺子上全部坍缩到 0(平均误差 0.0983);两级(块 scale e4m3 × 全局
fp32)给小块自带 0.019531 的小尺子,值全落在 4/6 格点(平均误差 0.011,改善 8.9 倍)——
e8m0 块 scale 只能取 2 的幂(exp2∘ceil∘log2,平均多花 1.4426/1.4427 倍 scale 换不溢出)。

数字全部取自本章参考实现实跑(与 explainer 素材同源);坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 852
MX = 60
BXR = 1440
GRID = '#e2e8f0'
E2M1 = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

lc.text(MX, 34, '两级秤:FP4 只有一个指针对——每 16 个值配小秤,再配一台总秤',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, 'e2m1 全部格点 ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}(块内动态范围 6/0.5 = 12):单级全局 scale 下,16 值全 ~0.1 的小块全部坍缩到 0;两级(块 scale e4m3 × 全局 fp32)误差 0.0983 → 0.011,好 8.9 倍',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L0 GPU 执行臂 · 模型层 forward + 编译 · 量化 Linear 的权重格式'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')


def e2m1_axis(x0, x1, y, v_lo, v_hi, tcol=lc.C_MUTE):
    lc.rect(x0, y, x1 - x0, 3, '#334155', '#334155', rx=1.5, sw=0)
    f = lambda v: x0 + (v - v_lo) / (v_hi - v_lo) * (x1 - x0)
    for v in [0.0] + E2M1:
        lc.seg(f(v), y - 6, f(v), y, tcol, 1.2)
        lc.text(f(v), y + 18, f'{v:g}', 8, tcol, 'middle', tag='tk' + f'{v:g}')
    return f


# ================= 上:单级(一把大秤) =================
U1Y, U1H = 122, 224
lc.rect(MX, U1Y, 940, U1H, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(MX + 16, U1Y + 22, '① 单级:全局一把秤(scale = 6/amax = 6/6 = 1.0)', 11,
        lc.C_TXT, 'start', True, maxw=880, tag='u1:h')
lc.text(MX + 16, U1Y + 40, '张量 amax=6 定尺(块 B 定的)——块 A 的值 0.083-0.118 离最近格点 0.5 还差 4-6 倍',
        9, lc.C_MUTE, 'start', maxw=880, tag='u1:sub')
AY1 = U1Y + 138
X1 = e2m1_axis(MX + 40, MX + 880, AY1, -0.3, 6.6)
# 块 A:16 值挤在 0.083-0.118(两排 8 点)
for i in range(8):
    vx = 0.083 + i * 0.005
    lc.rect(X1(vx) - 3, AY1 - 46 - (0 if i % 2 == 0 else 0) - (12 if i >= 4 else 0), 6, 6,
            lc.C_API_S, lc.C_API_S, rx=3, sw=0.8)
lc.text(X1(0.10) + 14, AY1 - 52, '块 A:16 值全在 0.083-0.118', 8.5, lc.C_API_S, 'start', True,
        tag='u1:a')
# 坍缩大箭头:簇 → 0
lc.seg(X1(0.10) - 4, AY1 - 38, X1(0.0), AY1 - 4, lc.C_ABORT, 2.4, 'std')
lc.text(X1(0.0) + 12, AY1 - 24, '全部坍缩到 0 · 平均|误差| 0.0983', 9, lc.C_ABORT, 'start',
        True, tag='u1:col')
# 块 B
lc.rect(X1(5.9) - 4, AY1 - 52, 8, 8, lc.C_MUTE, lc.C_MUTE, rx=4, sw=1)
lc.text(X1(5.9), AY1 - 60, '块 B(absmax=6)', 8.5, lc.C_MUTE, 'end', tag='u1:b')
lc.text(MX + 16, U1Y + U1H - 12, 'e2m1 的 16 个编码撑不起一张真实权重的动态范围——小块挤不进任何一个非零格点', 8.5,
        lc.C_MUTE, 'start', maxw=880, tag='u1:foot')

# ================= 中:两级换装箭头 =================
MIDY = U1Y + U1H
lc.seg(500, MIDY + 4, 500, MIDY + 44, lc.C_GPU_S, 2.4, 'std')
lc.text(516, MIDY + 22, '给块 A 配一把 e4m3 小秤:块 scale = 0.019531(fp32 全局 × 块 scale 两级)', 9.5,
        lc.C_GPU_S, 'start', True, maxw=700, tag='mid')

# ================= 下:两级(小秤 + 总秤) =================
U2Y, U2H = MIDY + 52, 250
lc.rect(MX, U2Y, 940, U2H, '#f0fdf4', lc.C_GPU_S, rx=8, sw=1.4)
lc.text(MX + 16, U2Y + 22, '② 两级:块 A 自己的秤——值 ×(1/0.019531) = 4.25-6.04,落进 4/6 格点', 11,
        lc.C_GPU_S, 'start', True, maxw=880, tag='u2:h')
lc.text(MX + 16, U2Y + 40, '块 scale e4m3 存 0.019531(raw 0.01967 → e4m3 格点);反量化 4×0.019531=0.0781、6×0.019531=0.1172',
        9, lc.C_MUTE, 'start', maxw=880, tag='u2:sub')
AY2 = U2Y + 140
X2 = e2m1_axis(MX + 40, MX + 880, AY2, -0.3, 6.6, tcol=lc.C_GPU_S)
# 8 值落 4、8 值落 6:两摞点
for i in range(8):
    dy = -14 - (i % 4) * 11
    lc.rect(X2(4.0) - 3, AY2 + dy, 6, 6, lc.C_API_S, lc.C_API_S, rx=3, sw=0.8)
    lc.rect(X2(6.0) - 3, AY2 + dy, 6, 6, lc.C_API_S, lc.C_API_S, rx=3, sw=0.8)
lc.seg(X2(4.0), AY2 - 4, X2(4.0), AY2 - 50, lc.C_GPU_S, 1.4)
lc.seg(X2(6.0), AY2 - 4, X2(6.0), AY2 - 50, lc.C_GPU_S, 1.4)
lc.text(X2(4.0), AY2 - 58, '8 个值落 4', 8.5, lc.C_GPU_S, 'middle', True, tag='u2:c4')
lc.text(X2(6.0), AY2 - 58, '8 个值落 6', 8.5, lc.C_GPU_S, 'middle', True, tag='u2:c6')
# 改善 chip
vw = lc.tw('平均|误差| 0.011 —— 改善 8.9 倍', 9.5, True) + 16
lc.rect(MX + 940 - vw - 14, U2Y + U2H - 34, vw, 22, '#f0fdf4', lc.C_GPU_S, rx=8, sw=1.3)
lc.text(MX + 940 - vw / 2 - 14, U2Y + U2H - 19, '平均|误差| 0.011 —— 改善 8.9 倍', 9.5,
        lc.C_GPU_S, 'middle', True, tag='u2:chip')
lc.text(MX + 16, U2Y + U2H - 12, '装载期预计算 alpha = 两级乘积(input 全局 × weight 全局 × 块 scale)', 8.5,
        lc.C_MUTE, 'start', maxw=640, tag='u2:alpha')

# ================= 右上:e8m0 挡位秤 =================
EX, EY, EW, EH = 1030, 122, 410, 252
lc.rect(EX, EY, EW, EH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(EX + 14, EY + 20, 'e8m0:只会跳挡的秤(块 scale 另一种 dtype)', 10, lc.C_TXT, 'start',
        True, maxw=EW - 28, tag='e8:h')
lc.text(EX + 14, EY + 38, '8 位纯指数、只能是 2 的幂:exp2(ceil(log2(·))) 向上取整保不溢出', 8.5,
        lc.C_MUTE, 'start', maxw=EW - 28, tag='e8:sub')
GEARS = ['0.0039', '0.0156', '0.0312', '0.0625', '1', '2', '4']
GX0 = EX + 30
GXW = (EW - 60) / len(GEARS)
for i, g in enumerate(GEARS):
    x = GX0 + i * GXW + GXW / 2
    lc.seg(x, EY + 66, x, EY + 78, lc.C_ENG_S, 1.4)
    lc.text(x, EY + 92, g, 7.8, lc.C_ENG_S, 'middle', tag='e8:g' + g)
lc.rect(GX0, EY + 78, EW - 60, 2, '#334155', '#334155', rx=1, sw=0)
EXS = [('0.013', '→ 0.0156'), ('0.02', '→ 0.0312'), ('0.0037', '→ 0.0039'),
       ('1.2', '→ 2.0'), ('3.0', '→ 4.0')]
for i, (a, b) in enumerate(EXS):
    yy = EY + 112 + i * 17
    lc.text(EX + 40, yy, a, 8.2, '#334155', 'end', tag='e8a' + str(i))
    lc.text(EX + 56, yy, b, 8.2, lc.C_ENG_S, 'start', True, tag='e8b' + str(i))
lc.text(EX + 14, EY + EH - 12, '平均开销 ×1.4426(理论 1/ln2 = 1.4427,100 万样本)——多花 ~44% 换不溢出',
        8.2, lc.C_MUTE, 'start', maxw=EW - 28, tag='e8:foot')

# ================= 右下:NVFP4 装载三件套 =================
NX, NY, NW, NH = 1030, 388, 410, 262
lc.rect(NX, NY, NW, NH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(NX + 14, NY + 20, 'NVFP4 装载三件套(vLLM modelopt.py)', 10, lc.C_TXT, 'start', True,
        maxw=NW - 28, tag='n:h')
NLINES = [
    '① 两个 e2m1 打包一个 uint8(input 维 //2)',
    '    —— L1151-L1162',
    '② 每 16 个输入元素一个 e4m3 块 scale',
    '    (group_size=16 · L1013,L1178-L1190)',
    '③ input_scale / weight_scale_2 两个 fp32',
    '    全局标量 · 装载期预计算 alpha(L1216-L1219)',
    '',
    'QuantKey 词汇:ScaleDesc(1,16) 的 scale',
    '+ kStaticTensorScale 的 scale2 双层描述',
    '—— quant_utils.py:L148-L156',
]
for i, ln in enumerate(NLINES):
    if ln:
        lc.text(NX + 14, NY + 42 + i * 19, ln, 8.3,
                lc.C_GPU_S if ln[0] in '①②③' else '#334155', 'start', maxw=NW - 28,
                tag='nl' + str(i))

# ================= 底部:分组思想的论文账 =================
BY = 668
lc.rect(MX, BY, 1380, 96, '#ffffff', lc.C_MUTE, rx=8, sw=1.1, dash=True)
lc.text(MX + 16, BY + 20, '分组思想的论文账(GPTQ §5):粒度是买来的,用存储开销换 PPL', 9.5,
        lc.C_TXT, 'start', True, maxw=1340, tag='nt:h')
lc.text(MX + 16, BY + 42, 'group-size 1024 ≈ 每 权重 0.02 额外 bit · g128 ≈ 0.15 额外 bit · Table 7:2-bit 下 g128 9.58 → g32 8.94(Wiki2 PPL,更低更好)',
        8.5, '#334155', 'start', maxw=1340, tag='nt:l1')
lc.text(MX + 16, BY + 62, '两级 scale 的这套词汇(QuantKey 的 scale + scale2)直通 ch28:DSV4-Flash 的 FP4 MoE capstone(预告)',
        8.5, '#334155', 'start', maxw=1340, tag='nt:l2')

lc.text(MX, BY + 116, '格式出处:OCP MX / NVIDIA Blackwell(不在论文包) · 格点为按规范位级枚举 · 论文账 arXiv:2210.17323 §5 · 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=1380, tag='foot')

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch27-fig-two-level-scale.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
