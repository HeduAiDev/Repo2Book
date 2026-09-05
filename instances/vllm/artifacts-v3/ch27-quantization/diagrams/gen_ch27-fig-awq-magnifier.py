#!/usr/bin/env python3
"""ch27 机制图 5 · AWQ 放大镜:显著权重 ×s、激活 ÷s(figure_spec ch27-fig-awq-magnifier,模板 before-after)

放大自 L0 GPU 执行臂(绿)第三块『模型层 forward + 编译』里 Linear 层权重的离线加工——
AWQ 的 s 早已折进检查点 scales,此图画的是『折进去之前』的数学。推导链第 5 环,不画架构元素。

claim:给显著权重戴放大镜:同一组 [0.9, 9.9],w=0.9 乘 s=2、激活除 2 后,量化发生在放大
后的网格上——误差 0.3375→-0.28125(平均口径减半到 0.5),组 max 不变(Δ′=Δ),乘积严格等价;
『显著』按激活幅度选、不按权重范数选。

数字全部取自本章参考实现实跑(与 explainer 素材同源);坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 878
MX = 60
BXR = 1440
GRID = '#e2e8f0'

lc.text(MX, 34, 'AWQ 的放大镜:重要的字先用放大镜看——误差平均减半,乘积一分不变',
        16.5, lc.C_TXT, 'start', True, maxw=1040, tag='title')
lc.text(MX, 58, '同一组 [0.9, 9.9]:显著权重 w=0.9 乘 s=2、激活除 2,量化发生在放大后的网格上;RoundErr 期望恒 0.25 而 Δ′≈Δ ⇒ 相对误差自动除以 s(平均口径 0.5);甜点 s≈2,再大轮到非显著通道被 Δ′/Δ>1 反噬',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L0 GPU 执行臂 · 模型层 forward + 编译 · Linear 层权重(离线加工)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')


def axis(x0, x1, y, v_lo, v_hi):
    lc.rect(x0, y, x1 - x0, 3, '#334155', '#334155', rx=1.5, sw=0)
    return lambda v: x0 + (v - v_lo) / (v_hi - v_lo) * (x1 - x0)


def dot(X, v, y, color):
    lc.rect(X(v) - 5, y - 5, 10, 10, color, color, rx=5, sw=1.0)


DELTA = 1.2375
LW_X0, LW_X1 = 110, 860

# ================= 上:不缩放(RTN) =================
U1Y, U1H = 122, 200
lc.rect(MX, U1Y, 900, U1H, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(MX + 16, U1Y + 22, '① 不缩放(RTN):Δ = 9.9/8 = 1.2375(AWQ Eq.1,分母 2^(N-1))', 11,
        lc.C_TXT, 'start', True, maxw=860, tag='u1:h')
lc.text(MX + 16, U1Y + 40, '组 [0.9, 9.9] · x=1.0(单元素贡献 y=w·x,便于手算)', 9,
        lc.C_MUTE, 'start', maxw=860, tag='u1:sub')
AY1 = U1Y + 130
XU = axis(LW_X0, LW_X1, AY1, -0.4, 10.6)
for k in range(9):
    v = k * DELTA
    lc.seg(XU(v), AY1 - 6, XU(v), AY1, lc.C_MUTE, 1.2)
    lc.text(XU(v), AY1 + 20, str(k), 8.5, lc.C_MUTE, 'middle', tag='u1:k' + str(k))
lc.text(LW_X1 + 8, AY1 + 20, 'q 码', 8.5, lc.C_MUTE, 'start', tag='u1:kax')
# w=0.9
dot(XU, 0.9, AY1 - 62, lc.C_API_S)
lc.text(XU(0.9), AY1 - 74, 'w=0.9', 9, lc.C_API_S, 'middle', True, tag='u1:w')
lc.seg(XU(0.9), AY1 - 57, XU(DELTA), AY1 - 4, lc.C_API_S, 1.2, 'std')
lc.text(XU(DELTA) + 10, AY1 - 16, 'Q=1.2375 · 误差 +0.3375(≈ 27% 步长)', 9, lc.C_ABORT,
        'start', True, tag='u1:err')
# 9.9 定尺者
dot(XU, 9.9, AY1 - 62, lc.C_MUTE)
lc.text(XU(9.9), AY1 - 74, '9.9 定尺者 ✓', 8.5, lc.C_MUTE, 'middle', tag='u1:99')
lc.text(MX + 16, U1Y + U1H - 12, 'y = Q(0.9)·x = 1.2375 vs 真值 0.9 —— 误差 +0.3375', 9,
        '#334155', 'start', maxw=860, tag='u1:foot')

# ================= 中:×s / ÷s 对偶箭头 =================
MIDY = U1Y + U1H
lc.seg(300, MIDY + 4, 300, MIDY + 48, lc.C_ENG_S, 2.2, 'std')
lc.text(312, MIDY + 24, '权重 ×s=2(显著通道)', 9.5, lc.C_ENG_S, 'start', True, tag='mid:dn')
lc.seg(850, MIDY + 48, 850, MIDY + 4, lc.C_ZMQ_S, 2.2, 'std')
lc.text(838, MIDY + 24, '激活 ÷s=2', 9.5, lc.C_ZMQ_S, 'end', True, tag='mid:up')
lc.text(575, MIDY + 26, 'Q(w·s)·(x/s) ≡ 同一乘法的另一条量化路径(严格等价,无近似)', 9,
        lc.C_MUTE, 'middle', maxw=620, tag='mid:eq')

# ================= 下:×s 后 =================
U2Y, U2H = MIDY + 52, 200
lc.rect(MX, U2Y, 900, U2H, '#f0fdf4', lc.C_GPU_S, rx=8, sw=1.4)
lc.text(MX + 16, U2Y + 22, '② ×s=2 后:w·s = 1.8,Δ′ = Δ = 1.2375(组 max 不变:1.8 < 9.9)', 11,
        lc.C_GPU_S, 'start', True, maxw=860, tag='u2:h')
lc.text(MX + 16, U2Y + 40, 'Round(1.8/1.2375) = Round(1.4545) = 1 → 同一格点;回来时 ÷s,格点相对变密', 9,
        lc.C_MUTE, 'start', maxw=860, tag='u2:sub')
AY2 = U2Y + 130
XD = axis(LW_X0, LW_X1, AY2, -0.4, 10.6)
for k in range(9):
    v = k * DELTA
    lc.seg(XD(v), AY2 - 6, XD(v), AY2, lc.C_MUTE, 1.2)
    lc.text(XD(v), AY2 + 20, str(k), 8.5, lc.C_MUTE, 'middle', tag='u2:k' + str(k))
lc.text(LW_X1 + 8, AY2 + 20, 'q 码', 8.5, lc.C_MUTE, 'start', tag='u2:kax')
dot(XD, 1.8, AY2 - 62, lc.C_API_S)
lc.text(XD(1.8), AY2 - 74, 'w·s=1.8', 9, lc.C_API_S, 'middle', True, tag='u2:ws')
lc.seg(XD(1.8), AY2 - 57, XD(DELTA), AY2 - 4, lc.C_API_S, 1.2, 'std')
lc.text(XD(DELTA) + 10, AY2 - 16, '÷s 回来:0.61875 · 误差 −0.28125', 9, lc.C_GPU_S,
        'start', True, tag='u2:err')
lc.text(MX + 16, U2Y + U2H - 12, 'y = Q(1.8)·(x/s) = 1.2375×0.5 = 0.61875 vs 真值 0.9 —— 误差 −0.28125', 9,
        '#334155', 'start', maxw=860, tag='u2:foot')

# ================= 右上:Table 2 =================
TX, TY, TW_, TH_ = 990, 122, 450, 240
lc.rect(TX, TY, TW_, TH_, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(TX + 14, TY + 20, 'Table 2(OPT-6.7B,INT3-g128):甜点 s=2', 10.5, lc.C_TXT, 'start',
        True, maxw=TW_ - 28, tag='t2:h')
lc.text(TX + 14, TY + 38, 's 太大反噬:s=4 时 21.2% 通道 Δ′/Δ > 1,PPL 反弹', 8.5,
        lc.C_MUTE, 'start', maxw=TW_ - 28, tag='t2:sub')
HD = [('s', 'start', 16), ('Δ′≠Δ 比例', 'end', 214), ('平均误差比', 'end', 312), ('Wiki2 PPL', 'end', 424)]
lc.rect(TX + 8, TY + 48, TW_ - 16, 20, '#f1f5f9', GRID, rx=4, sw=0.8)
for name, anc, dx in HD:
    lc.text(TX + dx, TY + 62, name, 8, lc.C_MUTE, anc, maxw=92, tag='t2h:' + name[:4])
T2 = [
    ('1', '0.0', '1.0', '23.54', False, False),
    ('1.25', '0.028', '0.804', '12.87', False, False),
    ('1.5', '0.044', '0.676', '12.48', False, False),
    ('2', '0.082', '0.519', '11.92', True, False),
    ('4', '0.212', '0.303', '12.36', False, True),
]
for i, (s_, d1, d2, ppl, sweet, back) in enumerate(T2):
    yy = TY + 84 + i * 24
    if sweet or back:
        col = lc.C_GPU_S if sweet else lc.C_ABORT
        lc.rect(TX + 8, yy - 12, TW_ - 16, 20, '#f0fdf4' if sweet else '#fef2f2', col,
                rx=4, sw=1.1)
    vals = [s_, d1, d2, ppl]
    for j, (name, anc, dx) in enumerate(HD):
        col = '#334155'
        bold = False
        if sweet:
            col, bold = lc.C_GPU_S, True
        if back and j == 3:
            col, bold = lc.C_ABORT, True
        lc.text(TX + dx, yy + 2, vals[j], 8.8, col, anc, bold=bold, tag='t2v' + str(i) + str(j))
lc.text(TX + 14, TY + TH_ - 12, '合成层同协议:Δ′≠Δ 仅 0.0547 · 平均 Δ′/Δ=1.0202 · 平均误差比 0.5101', 8.2,
        lc.C_MUTE, 'start', maxw=TW_ - 28, tag='t2:synth')

# ================= 右下:α 搜索 U 形 =================
CXP, CYP, CWP, CHP = 990, 378, 450, 224
lc.rect(CXP, CYP, CWP, CHP, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(CXP + 14, CYP + 20, '搜索空间 s = s_X^α:U 形曲线,20 点网格', 10.5, lc.C_TXT, 'start',
        True, maxw=CWP - 28, tag='sw:h')
lc.text(CXP + 14, CYP + 38, 'α=0 即 RTN;内点 α*=0.32 打败 RTN 1.84 倍;α=1 反弹——甜点在中间', 8.5,
        lc.C_MUTE, 'start', maxw=CWP - 28, tag='sw:sub')
ALPHA = [0.0, 0.05, 0.11, 0.16, 0.21, 0.26, 0.32, 0.37, 0.42, 0.47, 0.53, 0.58, 0.63, 0.68, 0.74, 0.79, 0.84, 0.89, 0.95, 1.0]
LOSS = [13.2416, 11.4855, 10.2312, 9.1963, 8.4996, 7.524, 7.197, 7.2522, 7.5205, 7.7736,
        8.4777, 9.2501, 9.9259, 10.6854, 11.2445, 12.2877, 13.2613, 14.1347, 14.7408, 15.402]
GX0, GX1 = CXP + 48, CXP + CWP - 24
GY0, GY1 = CYP + 60, CYP + CHP - 44          # y 轴上/下(值域倒置:loss 大在上)
LMIN, LMAX = 6.5, 15.5


def gx(a):
    return GX0 + (a - 0.0) / 1.0 * (GX1 - GX0)


def gy(l_):
    return GY0 + (l_ - LMIN) / (LMAX - LMIN) * (GY1 - GY0)


lc.rect(GX0, GY1, GX1 - GX0, 2, '#334155', '#334155', rx=1, sw=0)
lc.rect(GX0, GY0, 2, GY1 - GY0, '#334155', '#334155', rx=1, sw=0)
pts = [(gx(a), gy(l_)) for a, l_ in zip(ALPHA, LOSS)]
path = ' '.join(f'{"M" if i == 0 else "L"}{p[0]:.1f},{p[1]:.1f}' for i, p in enumerate(pts))
lc.ELEMS.append(((GX0 - 4, GY0 - 4, GX1 + 4, GY1 + 4),
                 f'<path d="{path}" fill="none" stroke="{lc.C_API_S}" stroke-width="2.2"/>'))
for a, lab in ((0.0, '0'), (0.5, '0.5'), (1.0, '1')):
    lc.text(gx(a), GY1 + 16, lab, 8.5, lc.C_MUTE, 'middle', tag='sw:x' + lab)
lc.text(GX1 + 8, GY1 + 16, 'α', 8.5, lc.C_MUTE, 'start', tag='sw:xax')
lc.text(CXP + 40, GY0 + 8, 'L(s)', 8.5, lc.C_MUTE, 'end', tag='sw:yax')
# 关键点
for a, l_, name, col, dy in ((0.0, 13.2416, 'α=0(RTN) L=13.2416', lc.C_ABORT, -10),
                             (0.32, 7.197, 'α*=0.32 L=7.197', lc.C_GPU_S, -12),
                             (1.0, 15.402, 'α=1 L=15.402', lc.C_ABORT, -10)):
    lc.rect(gx(a) - 4, gy(l_) - 4, 8, 8, col, col, rx=4, sw=1)
    lc.text(gx(a), gy(l_) + dy, name, 8.2, col, 'middle', True, tag='sw:p' + str(a))
lc.text(CXP + 14, CYP + CHP - 12, '「显著」按激活幅度选(s_X 底数),不按权重范数选(按 W ≈ 随机)', 8.2,
        lc.C_MUTE, 'start', maxw=CWP - 28, tag='sw:foot')

# ================= 底部三注 =================
BY = 630
lc.rect(MX, BY, 1380, 128, '#ffffff', lc.C_MUTE, rx=8, sw=1.1, dash=True)
NOTES = [
    ('单点 vs 平均', [
        '单点误差比 0.28125/0.3375 = 0.83333(5/6)',
        '理论比(Δ′/Δ)·(1/s) = 0.5 是期望口径:',
        'RoundErr ~ 期望 0.25(20 万样本实测 0.2502)']),
    ('两板斧(AWQ §3.1)', [
        '理想解:1% 显著通道保留 FP16,',
        'PPL 43.2 → 13.0——但混合精度硬件不友好',
        '等价缩放接近其效果,且全程 INT']),
    ('落地打包(§4.2)', [
        '8 个 4-bit 码 interleave [0,2,4,6,1,3,5,7]',
        '压进一个 32-bit 字:0x75316420',
        'vllm quant_utils.py:L880-L899 同一 interleave']),
]
NW = 448
for i, (t, lines) in enumerate(NOTES):
    x = MX + 12 + i * (NW + 12)
    lc.text(x, BY + 20, t, 9.5, lc.C_TXT, 'start', True, tag='nt' + str(i))
    for j, ln in enumerate(lines):
        lc.text(x, BY + 40 + j * 18, ln, 8.3, '#334155', 'start', maxw=NW - 8,
                tag='ntl' + str(i) + str(j))

lc.text(MX, BY + 144, '论文口径 arXiv:2306.00978 §3.1-§3.2(Eq.1-Eq.5 · Table 1/Table 2 · Figure 2) · 数值:本章 NumPy 参考实现实跑 · AWQ Eq.1 分母 2^(N-1)=8(与 SmoothQuant Eq.1 的 2^(N-1)−1 各自约定) · 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=1380, tag='foot')

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch27-fig-awq-magnifier.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
