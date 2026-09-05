#!/usr/bin/env python3
"""ch27 机制图 7 · INT 等距 vs FP8 e4m3 指数分段(figure_spec ch27-fig-fp8-grid-vs-int,模板 layout)

放大自 L0 GPU 执行臂(绿)第三块『模型层 forward + 编译』里量化 Linear 的权重/激活存取格式——
fp8.py 装载的 e4m3 weight + weight_scale 就是这副格点。推导链第 7 环,不画架构元素。

claim:同样 8 bit:INT 等距格点一刀切(铺满 ±448 步长恒 3.5137),FP8 e4m3 按指数分段
(段内 8 格、段间倍增,步长 0.001953125→32)——126 个正格点里 55 个在 (0,1)、23 个 ≥64:
动态范围换段内精度,且无需 zero-point(amax 对称 scale=448/amax 一路到格点)。

格点值按 OCP FP8 规范位级枚举(与 explainer 素材同源);坐标由常量/循环计算;文本全 esc()。
"""
import math
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

lc.text(MX, 34, 'INT 等距 vs FP 指数分段:尺子自己会变疏密',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, 'INT8 是一把刚性直尺(±448 内步长恒 3.5137,小值区一个格点都没有);e4m3 是倍增尺——每段 8 格等距、段间步长翻倍:离群天生有大格点可落,普通值保住相对精度',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L0 GPU 执行臂 · 模型层 forward + 编译 · 量化 Linear 的存取格式'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 主面板:对数数轴双排刻度 =================
PX, PY, PW, PH = MX, 122, 1380, 486
lc.rect(PX, PY, PW, PH, '#ffffff', GRID, rx=8, sw=1.2)
AX0, AX1 = 150, 1350
LOG_LO, LOG_HI = -9.0, 9.0


def X(v):
    return AX0 + (math.log2(v) - LOG_LO) / (LOG_HI - LOG_LO) * (AX1 - AX0)


# ---- 上排:INT8 等距 ----
IY = 196
lc.text(PX + 16, IY - 22, '上排 INT8:等距,正半轴 128 格、步长 3.5137(896/255)——线性均匀',
        10, lc.C_TXT, 'start', True, maxw=900, tag='i:lbl')
STEP = 3.5137
int8_xs = [X(k * STEP) for k in range(1, 128)]
# 挤成一团的高亮带(x>=64)
lc.rect(X(64.0) - 3, IY - 4, X(448.0) - X(64.0) + 6, 18, '#fef2f2', 'none', rx=3, sw=0)
for x in int8_xs:
    lc.seg(x, IY, x, IY + 10, lc.C_ABORT, 0.7)
lc.text(X(STEP), IY + 26, '3.5137', 8, lc.C_ABORT, 'middle', tag='i:first')
lc.text(X(448.0), IY + 26, '448', 8, lc.C_ABORT, 'middle', tag='i:last')
# 小值区空带标注
lc.text((AX0 + X(STEP)) / 2, IY + 6, '0.00195 – 3.51:一个格点都没有(小值区全空)', 8.5,
        lc.C_MUTE, 'middle', tag='i:empty')
# 基线
lc.rect(AX0, IY + 10, AX1 - AX0, 2, '#334155', '#334155', rx=1, sw=0)
lc.text(X(64.0) - 10, IY - 4, 'x ≥ 64:110 个刻度挤成一团', 8.2, lc.C_ABORT, 'end', True,
        maxw=220, tag='i:bunch')

# ---- 下排:e4m3 分段 ----
FY = 330
lc.text(PX + 16, FY - 44, '下排 FP8 e4m3:指数分段——每段 8 格等距、段间步长翻倍', 10,
        lc.C_TXT, 'start', True, maxw=900, tag='f:lbl')
# 正格点全集:次正规 7 个 + 14 个八度×8 + [256,448] 7 个 = 126
fp_vals = [2 ** -9 * m for m in range(1, 8)]                       # subnormal 0.00195..0.0137
for k in range(-6, 8):                                             # 0.015625..240
    fp_vals += [2.0 ** k * (1 + j / 8.0) for j in range(8)]
fp_vals += [256 * (1 + j / 8.0) for j in range(7)]                 # 256..448(480 让位 NaN)
for x in fp_vals:
    lc.seg(X(x), FY - 10, X(x), FY, lc.C_GPU_S, 0.8)
lc.rect(AX0, FY, AX1 - AX0, 2, '#334155', '#334155', rx=1, sw=0)
# 段界标签(每 3 个八度一个 + 首尾)
for k, lab in ((-9, '0.00195'), (-6, '0.0156'), (-3, '0.125'), (0, '1'), (3, '8'),
               (6, '64'), (8, '256'), (None, '448')):
    xv = 0.001953125 if k == -9 else (2.0 ** k if k is not None else 448.0)
    lc.text(X(xv), FY + 18, lab, 7.5, lc.C_MUTE, 'middle', tag='f:b' + lab)
# 高亮两段:[0.5,1) 与 [256,448]
lc.rect(X(0.5), FY - 14, X(1.0) - X(0.5), 26, 'none', lc.C_ENG_S, rx=4, sw=2.0)
lc.rect(X(256.0) - 4, FY - 14, X(448.0) - X(256.0) + 8, 26, 'none', lc.C_ENG_S, rx=4, sw=2.0)
lc.text(X(0.75), FY + 36, '段 [0.5,1)', 8, lc.C_ENG_S, 'middle', True, tag='f:s1')
lc.text(X(352.0) - 30, FY + 36, '段 [256,448]', 8, lc.C_ENG_S, 'end', True, tag='f:s2')
# 密度 chip
lc.rect(PX + PW - 320, FY - 78, 304, 24, '#f0fdf4', lc.C_GPU_S, rx=8, sw=1.1)
lc.text(PX + PW - 168, FY - 61, '126 个正格点:55 个在 (0,1) · 23 个 ≥ 64', 8.5,
        lc.C_GPU_S, 'middle', True, maxw=290, tag='f:density')

# ---- 放大镜条:两段对比 ----
ZY = 420
lc.text(PX + 16, ZY, '两段放大:同样是 8 个格点——小数区步长 0.0625,大数区步长 32(480 让位给 NaN,末段仅 7 格)',
        9.5, lc.C_TXT, 'start', True, maxw=1340, tag='z:lbl')


def zoom_strip(zx0, zx1, zy, vals, step_lab, color):
    lc.rect(zx0, zy - 8, zx1 - zx0, 2, '#334155', '#334155', rx=1, sw=0)
    n = len(vals)
    for i, v in enumerate(vals):
        x = zx0 + (i + 0.5) * (zx1 - zx0) / n
        lc.seg(x, zy - 8, x, zy - 18, color, 1.2)
        lab = f'{v:g}'
        lc.text(x, zy + 16, lab, 7.5, '#334155', 'middle', tag='z:v' + lab)
    lc.text((zx0 + zx1) / 2, zy + 34, step_lab, 8.5, color, 'middle', True, maxw=(zx1 - zx0),
            tag='z:s' + step_lab[:6])
    # 连接虚线(无箭头,装饰连通)
    return


z1 = [0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375]
z2 = [256, 288, 320, 352, 384, 416, 448]
zoom_strip(200, 700, ZY + 60, z1, '步长 0.0625(段内 8 格)', lc.C_GPU_S)
zoom_strip(800, 1300, ZY + 60, z2, '步长 32(段内 7 格 + NaN 让位)', lc.C_GPU_S)
lc.seg(X(0.75), FY + 44, 450, ZY + 34, lc.C_ENG_S, 1.0, dash=True)
lc.seg(X(352.0) - 30, FY + 44, 1050, ZY + 34, lc.C_ENG_S, 1.0, dash=True)

# ================= 底部左:amax 缩放示例 =================
BX, BY, BW, BH = MX, 630, 700, 160
lc.rect(BX, BY, BW, BH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(BX + 14, BY + 20, 'amax 对称缩放:无 zero-point,scale = 448/amax = 4.48', 10.5,
        lc.C_TXT, 'start', True, maxw=660, tag='b:h')
lc.text(BX + 14, BY + 38, 'x=[1.0, 0.55, -0.3, 0.9, 100.0] · 100.0 精确映到 448', 8.5,
        lc.C_MUTE, 'start', maxw=660, tag='b:sub')
BH2 = [('x', 'end', 90), ('x×4.48', 'end', 220), ('最近 e4m3 格点', 'end', 390), ('误差', 'end', 480)]
AMAX = [
    ('1.0', '4.48', '4.5', '-0.0045'),
    ('0.55', '2.464', '2.5', '-0.008'),
    ('-0.3', '-1.344', '-1.375', '0.0069'),
    ('0.9', '4.032', '4.0', '0.0071'),
    ('100.0', '448.0', '448.0', '0.0 ✓'),
]
lc.rect(BX + 100, BY + 46, 396, 20, '#f1f5f9', GRID, rx=4, sw=0.8)
for name, anc, dx in BH2:
    lc.text(BX + dx, BY + 60, name, 7.8, lc.C_MUTE, anc, maxw=140, tag='bth:' + name[:4])
for i, row in enumerate(AMAX):
    yy = BY + 82 + i * 14.5
    for j, (name, anc, dx) in enumerate(BH2):
        col = lc.C_GPU_S if (i == 4 and j == 3) else '#334155'
        lc.text(BX + dx, yy, row[j], 8, col, anc, tag='bv' + str(i) + str(j))
lc.text(BX + 510, BY + 82 + 1 * 14.5, '最大 |误差| = 0.008', 9, lc.C_ABORT, 'start', True,
        tag='b:max')
lc.text(BX + 510, BY + 82 + 2.6 * 14.5, '落格后 ÷scale 还原', 8, lc.C_MUTE, 'start',
        maxw=180, tag='b:note')

# ================= 底部右:位型 + vLLM 装载 =================
CX, CW_ = 790, 650
lc.rect(CX, BY, CW_, BH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(CX + 14, BY + 20, 'e4m3 位型(1 符号 + 4 指数 bias 7 + 3 尾数)', 10.5, lc.C_TXT,
        'start', True, maxw=620, tag='c:h')
CLINES = [
    'max = ±448 · min normal = 0.015625 · min subnormal = 0.001953125',
    '仅 S.1111.111 一个模式为 NaN(fn 格式)· 有限值共 253 个(±126 + 0)',
    'vLLM 装载:weight e4m3 + per-tensor / 128×128 块 scale',
    '(weight_scale_inv 命名沿用 DeepSeek;块 scale dtype 可 float8_e8m0fnu)',
    'vllm/model_executor/layers/quantization/fp8.py:L342-L369',
    '参考实现 scaled_quantize:quant_utils.py:L359-L411(±448 常量:L27-L35)',
]
for i, ln in enumerate(CLINES):
    lc.text(CX + 14, BY + 42 + i * 18, ln, 8.3, '#334155' if i < 4 else lc.C_FAINT,
            'start', maxw=620, tag='cl' + str(i))

# ================= 页脚 =================
lc.text(MX, BY + BH + 26, '格式出处:OCP FP8 / NVIDIA《FP8 Formats for Deep Learning》(不在论文包,正文一句带过) · 格点为按规范位级枚举(与 vLLM ±448 常量同语义) · 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=1380, tag='foot')

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch27-fig-fp8-grid-vs-int.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
