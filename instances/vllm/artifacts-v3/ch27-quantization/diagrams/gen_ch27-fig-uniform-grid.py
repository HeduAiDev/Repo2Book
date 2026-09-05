#!/usr/bin/env python3
"""ch27 机制图 1 · 均匀量化底座:一把尺子的两个自由度(figure_spec ch27-fig-uniform-grid,模板 before-after)

放大自 L0 GPU 执行臂(绿)第三块『模型层 forward + 编译』里的 Linear 层权重——
qweight/scales 检查点里存的整数码与步长就是这副网格。推导链第 1 环,不画架构元素。

claim:均匀量化 = 除以 Δ、取整、乘回。对称式 Δ=max|x|/qmax 一把居中尺子,单点误差 ≤ Δ/2;
非对称式加 zp=[qmin−round(xmin/scale)] 平移网格原点,xmin/xmax 两端精确落格——
INT4 全部可用精度就是这 16 个码。

数字全部取自本章参考实现实跑(与 explainer 素材同源);坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 812
MX = 60
BXR = 1440
GRID = '#e2e8f0'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '均匀量化的第一课:一把尺子有 Δ 和原点两个自由度',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '除以步长 → 取整 → 乘回:对称式用 absmax 定 Δ、原点钉在 0;非对称式(min-max 网格)再用 zp 把原点搬到分布的实际位置——无论哪种,单点误差 ≤ 半格 Δ/2',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L0 GPU 执行臂 · 模型层 forward + 编译 · Linear 层权重'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')


def axis(x0, x1, y, v_lo, v_hi):
    """一根数轴:细 rect + 映射函数。返回 v→px 的映射。"""
    lc.rect(x0, y, x1 - x0, 3, '#334155', '#334155', rx=1.5, sw=0)
    return lambda v: x0 + (v - v_lo) / (v_hi - v_lo) * (x1 - x0)


def tick(X, v, y, h, color=lc.C_MUTE, sw=1.2):
    lc.seg(X(v), y, X(v), y + h, color, sw)


def dot(X, v, y, color):
    lc.rect(X(v) - 5, y - 5, 10, 10, color, color, rx=5, sw=1.0)


# ================= 面板一:对称式 =================
P1Y = 92
lc.rect(MX, P1Y, 1180, 332, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(MX + 16, P1Y + 24, '① 对称式:Δ = max|x|/qmax —— 一把居中尺子(INT4:q ∈ [-7,7],本例 Δ = 1.0/7 = 0.1429)',
        11.5, lc.C_TXT, 'start', True, maxw=1100, tag='p1:h')
lc.text(MX + 16, P1Y + 42, 'x = [0.9, -0.4, 0.15, -1.0, 0.62, -0.77, 0.3, 0.05],逐点投影到最近的格点(×8 值无一原生落格;-1.0 恰在格点上,误差 0)',
        9, lc.C_MUTE, 'start', maxw=1100, tag='p1:sub')

# 数据(traces 对称 INT4,逐字)
SYM = [
    # x, x/Δ, q, x̂, 误差
    (0.9, 6.3, 6, 0.8571, 0.0429),
    (-0.4, -2.8, -3, -0.4286, 0.0286),
    (0.15, 1.05, 1, 0.1429, 0.0071),
    (-1.0, -7.0, -7, -1.0, 0.0),
    (0.62, 4.34, 4, 0.5714, 0.0486),
    (-0.77, -5.39, -5, -0.7143, -0.0557),
    (0.3, 2.1, 2, 0.2857, 0.0143),
    (0.05, 0.35, 0, 0.0, 0.05),
]
DELTA = 0.1429
AXY = P1Y + 238            # 轴 y
X1 = axis(110, 880, AXY, -1.2, 1.2)
# 15 个码的格点 + 码标签
for q in range(-7, 8):
    tick(X1, q * DELTA, AXY - 6, 6, lc.C_MUTE, 1.2)
    lab = str(q)
    lc.text(X1(q * DELTA), AXY + 22, lab, 8.5, lc.C_MUTE, 'middle', tag='p1:q' + lab)
lc.text(888, AXY + 22, 'q 码', 8.5, lc.C_MUTE, 'start', tag='p1:qax')
# 数值刻度(0 与两端)
for v, lab in ((0, '0'), (1.0, '+1.0'), (-1.0, '-1.0')):
    lc.text(X1(v), AXY + 36, lab, 8, lc.C_FAINT, 'middle', tag='p1:v' + lab)
# Δ 括注:-7 到 -6 格(该区间无投影箭头穿越)
b0x, b1x = X1(-7 * DELTA), X1(-6 * DELTA)
by = AXY - 40
lc.seg(b0x, by, b0x, by + 10, lc.C_MUTE, 1.0)
lc.seg(b1x, by, b1x, by + 10, lc.C_MUTE, 1.0)
lc.seg(b0x, by + 5, b1x, by + 5, lc.C_MUTE, 1.0)
lc.text((b0x + b1x) / 2, by - 4, 'Δ=0.1429', 9, lc.C_TXT, 'middle', True, tag='p1:delta')
# 8 个值:两级交错标记 + 投影箭头(完整误差账在右侧逐项表)
ROW_A, ROW_B = AXY - 78, AXY - 122
for i, (x, xo, q, xh, err) in enumerate(SYM):
    ry = ROW_A if i % 2 == 0 else ROW_B
    dot(X1, x, ry, lc.C_API_S)
    lc.text(X1(x), ry - 12, f'x={x:g}', 8.5, lc.C_API_S, 'middle', tag='p1:x' + str(i))
    # 箭头:标记框底 → 轴上格点(端点都贴元素边)
    lx = X1(q * DELTA)
    lc.seg(X1(x), ry + 5, lx, AXY - 4, lc.C_API_S, 1.1, 'std')
    # 轴上只标两处:精确落格(-1.0,误差 0)与最坏值(-0.77,误差 -0.0557)
    if err == 0.0:
        lc.text(lx - 10, AXY - 12, '误差 0', 8.5, lc.C_MUTE, 'end', tag='p1:e0')
    if err == -0.0557:
        lc.text(lx - 12, AXY - 12, f'{err:+.4f}', 8.5, lc.C_ABORT, 'end', bold=True,
                tag='p1:ehot')
# 半格界示意:最坏值 -0.77 的落点 q=-5 两侧 ±Δ/2 两段短竖线(证据带)
gx = X1(-5 * DELTA)
half_l, half_r = X1(-5 * DELTA - DELTA / 2), X1(-5 * DELTA + DELTA / 2)
lc.seg(half_l, AXY + 30, half_l, AXY + 44, lc.C_ABORT, 1.2, dash=True)
lc.seg(half_r, AXY + 30, half_r, AXY + 44, lc.C_ABORT, 1.2, dash=True)
lc.text((half_l + half_r) / 2, AXY + 56, '最坏 |误差| 0.0557 ≤ Δ/2 = 0.0714', 8.5, lc.C_ABORT,
        'middle', True, tag='p1:worst')
# 右侧逐项账(整表收在面板一内,与数轴同高并列)
TBX, TBY, TBW = 904, P1Y + 60, 328
lc.rect(TBX, TBY, TBW, 246, '#ffffff', GRID, rx=6, sw=1.0)
lc.text(TBX + 12, TBY + 18, '8 值逐项账(码 / 反量化 / 误差)', 9.5, lc.C_TXT, 'start', True,
        maxw=TBW - 20, tag='p1:t:h')
for i, (x, xo, q, xh, err) in enumerate(SYM):
    yy = TBY + 36 + i * 24
    hot = (err == -0.0557)
    if hot:
        lc.rect(TBX + 8, yy - 12, TBW - 16, 20, '#fef2f2', lc.C_ABORT, rx=4, sw=1.0)
    lc.text(TBX + 12, yy, f'x={x:<6g}x/Δ={xo:<6g}', 8, '#334155', 'start', tag='p1:t:a' + str(i))
    lc.text(TBX + 118, yy, f'q={q:<3d}', 8, '#334155', 'start', tag='p1:t:q' + str(i))
    lc.text(TBX + 158, yy, f'x̂={xh:<8g}', 8, '#334155', 'start', tag='p1:t:x' + str(i))
    lc.text(TBX + TBW - 12, yy, f'{err:+.4f}', 8, lc.C_ABORT if hot else lc.C_MUTE, 'end',
            bold=hot, tag='p1:t:e' + str(i))
lc.text(MX + 16, P1Y + 316, '对称式码域永不越界:Δ=max|x|/qmax ⇒ |x|/Δ ≤ qmax 恒成立(clip 只是浮点护栏,本例未触发)',
        9, lc.C_MUTE, 'start', maxw=1140, tag='p1:foot')

# ================= 面板二:非对称式(ReLU 后) =================
P2Y = 442
lc.rect(MX, P2Y, 700, 258, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(MX + 16, P2Y + 24, '② 非对称式:scale=(xmax−xmin)/15,zp=qmin−round(xmin/scale) —— 原点搬家',
        11.5, lc.C_TXT, 'start', True, maxw=660, tag='p2:h')
lc.text(MX + 16, P2Y + 42, 'ReLU 后全非负:x=[0.0, 0.53, 1.0, 0.73],scale=1/15=0.0667,zp=−8(实数 0 的码)——xmin/xmax 两端精确落格',
        9, lc.C_MUTE, 'start', maxw=670, tag='p2:sub')
ASY = P2Y + 176
X2 = axis(120, 660, ASY, -0.08, 1.18)
SC = 1 / 15
for q in range(-8, 8):
    v = (q - (-8)) * SC
    tick(X2, v, ASY - 6, 6, lc.C_MUTE, 1.2)
    lc.text(X2(v), ASY + 22, str(q), 8.5, lc.C_MUTE, 'middle', tag='p2:q' + str(q))
lc.text(668, ASY + 22, 'q 码', 8.5, lc.C_MUTE, 'start', tag='p2:qax')
ASYM = [
    # x, q, x̂, 误差, 精确?
    (0.0, -8, 0.0, 0.0, True),
    (0.53, 0, 0.5333, -0.0033, False),
    (1.0, 7, 1.0, 0.0, True),
    (0.73, 3, 0.7333, -0.0033, False),
]
for i, (x, q, xh, err, exact) in enumerate(ASYM):
    ry = ASY - 62 if i % 2 == 0 else ASY - 104
    dot(X2, x, ry, lc.C_ZMQ_S)
    lc.text(X2(x), ry - 12, f'x={x:g}', 8.5, lc.C_ZMQ_S, 'middle', tag='p2:x' + str(i))
    lx = X2((q + 8) * SC)
    lc.seg(X2(x), ry + 5, lx, ASY - 4, lc.C_ZMQ_S, 1.1, 'std')
    mark = ' ✓' if exact else f' 误 {err:+.4f}'
    lc.text(lx + (10 if q >= 0 else -10), ASY - 14, mark.strip(), 8.5,
            lc.C_GPU_S if exact else lc.C_MUTE, 'start' if q >= 0 else 'end',
            bold=exact, tag='p2:m' + str(i))
lc.text(MX + 16, P2Y + 216, '两端 xmin=0→qmin=−8、xmax=1.0→qmax=7 精确落格;中间值误差 ≤ scale/2=0.0333',
        9, lc.C_MUTE, 'start', maxw=660, tag='p2:foot')
lc.text(MX + 16, P2Y + 234, 'zp 不改间距:网格平移,Δ 仍是那把尺子',
        9, lc.C_MUTE, 'start', maxw=660, tag='p2:foot2')

# ---- 面板二b:负偏置对照 ----
P2BX, P2BY, P2BW, P2BH = 780, 442, 660, 258
lc.rect(P2BX, P2BY, P2BW, P2BH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(P2BX + 16, P2BY + 24, '②b 分布整体偏负:x=[-2.0, -1.0, 0.45] → scale=2.45/15=0.1633,zp=+4',
        11.5, lc.C_TXT, 'start', True, maxw=620, tag='p2b:h')
lc.text(P2BX + 16, P2BY + 42, 'zp 为正:网格原点(q=0 处)搬到 −0.6533——分布偏哪儿,原点搬哪儿,两端仍精确落格',
        9, lc.C_MUTE, 'start', maxw=630, tag='p2b:sub')
NB = [
    (-2.0, -8, -1.96, -0.04, True),
    (-1.0, -2, -0.98, -0.02, False),
    (0.45, 7, 0.49, -0.04, True),
]
NBY = P2BY + 168
X3 = axis(P2BX + 60, P2BX + 560, NBY, -2.25, 0.7)
SC3 = 2.45 / 15
ZP3 = 4
for q in range(-8, 8):
    v = (q - ZP3) * SC3
    tick(X3, v, NBY - 6, 6, lc.C_MUTE, 1.2)
    if q in (-8, -4, 0, 4, 7):
        lc.text(X3(v), NBY + 22, str(q), 8.5, lc.C_MUTE, 'middle', tag='p2b:q' + str(q))
lc.text(P2BX + 568, NBY + 22, 'q 码', 8.5, lc.C_MUTE, 'start', tag='p2b:qax')
# 原点标注(q=0 ↔ x=-0.6533):q 码下一行、短句
ox = X3((0 - ZP3) * SC3)
lc.seg(ox, NBY + 30, ox, NBY + 40, lc.C_ENG_S, 1.4)
lc.text(ox, NBY + 52, 'q=0 ↔ x=−0.6533(zp 搬来的原点)', 8.5, lc.C_ENG_S, 'middle', True,
        maxw=260, tag='p2b:origin')
for i, (x, q, xh, err, exact) in enumerate(NB):
    ry = NBY - 54 if i % 2 == 0 else NBY - 90
    dot(X3, x, ry, lc.C_ZMQ_S)
    lc.text(X3(x), ry - 12, f'x={x:g}', 8.5, lc.C_ZMQ_S, 'middle', tag='p2b:x' + str(i))
    lx = X3((q - ZP3) * SC3)
    lc.seg(X3(x), ry + 5, lx, NBY - 4, lc.C_ZMQ_S, 1.1, 'std')
    mark = '✓' if exact else f'误 {err:+.2f}'
    lc.text(lx + (10 if q >= 0 else -10), NBY - 14, mark, 8.5, lc.C_GPU_S if exact else lc.C_MUTE,
            'start' if q >= 0 else 'end', bold=exact, tag='p2b:m' + str(i))
lc.text(P2BX + 16, P2BY + 240, '误差 [−0.04, −0.02, −0.04],|max|=0.04 ≤ scale/2=0.0817;0.45 精确映 qmax=7',
        9, lc.C_MUTE, 'start', maxw=620, tag='p2b:foot')

# ================= 页脚 =================
FY = 726
lc.rect(MX, FY, 1380, 62, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(MX + 14, FY + 18, '统计口径:512 随机值两组——N=4 最大误差 0.1947 ≤ 半格 0.195;N=8 最大误差 0.0081 ≤ 0.0081(位每加 1,步长近似减半)',
        9, '#334155', 'start', maxw=1340, tag='ft:1')
lc.text(MX + 14, FY + 36, '代码对照:vllm/model_executor/layers/quantization/utils/quant_utils.py:L672-L734(quantize_weights 对称/非对称两分支:scale=(max−min)/qmax + zp=round(|min|/scale))',
        9, '#334155', 'start', maxw=1340, tag='ft:2')
lc.text(MX + 14, FY + 54, '论文口径:arXiv:2211.10438 §2 Eq.1(Δ=max|x|/(2^{N-1}-1));AWQ arXiv:2306.00978 §3.2 Eq.1 分母 2^{N-1}=8——两文各自约定,本章并存示教 · 数值取自本章 NumPy 参考实现实跑',
        9, lc.C_FAINT, 'start', maxw=1340, tag='ft:3')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch27-fig-uniform-grid.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
