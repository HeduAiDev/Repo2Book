#!/usr/bin/env python3
"""ch20 论文精髓图 ② · paper-fig-2(arXiv:2205.14135 Fig.2 忠实重绘)

writer figure-requests.json add:FA 论文三联实测图——左:GPT-2 medium 前向+反向账单,证明
HBM 访问量(而非 FLOP 数)决定 runtime(FA 反向重算 FLOP 更多反而更快);中:分块 Bc 对前向
耗时,块越大越快、超过 256 后收益封顶;右:block-sparse 变体加速比随稀疏度按比例上升。

原图真相源(arXiv e-print 2205.14135 源码):
- 左 = src/theory.tex 里 Fig.2 的 minipage 表格(逐字):GFLOPs 66.6 vs 75.2 ·
  HBM R/W (GB) 40.3 vs 4.4 · Runtime (ms) 41.7 vs 7.3;配置 GPT-2 medium
  (seq 1024, head dim 64, 16 heads, batch 64) A100。
- 中 = figs/flashattn_micros.pdf "Effect of Block Size":x = Block Size 64/128/256/512,
  左轴 HBM Accesses (GB) 刻度 0/2/4/6、右轴 Runtime(ms);绿曲线 HBM 访问随 Bc 增大
  递减(矢量提取 ≈6.6→0.9 GB),蓝曲线前向耗时递减且 256 后走平(≈8.6→3.2 ms)——
  只画趋势与刻度,不标逐点数值(writer 指示:论文 md 文本不含,以示意呈现)。
- 右 = 同 PDF "Sparsity Speedup":x = % Non-Zero Blocks(刻度 20/60),y 刻度 2/4/6,
  图例 Dense / Block-Sparse FlashAttention;曲线随非零占比下降而抬升,基准线平;
  seq 4K(caption:faster by a factor proportional to the sparsity)。

布局与信息结构对齐原图;配色/字体套本书视觉语言;文字译中。provenance=原论文图本身。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 660
MX = 42
BXR = 1458
C_RED = '#dc2626'
C_ACC = '#16a34a'      # HBM 访问曲线 = 绿
C_RUN = '#2563eb'      # runtime 曲线 = 蓝
C_SPARSE = '#7c3aed'   # block-sparse 曲线 = 紫
EXTRA_DEFS = (
    '<marker id="grn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
    '<marker id="blu" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '论文实测:HBM 访问量(而非 FLOP 数)决定 runtime——FLOP 更多,反而快 5.7×',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 56, '重绘自 arXiv:2205.14135 Fig.2:左 = GPT-2 medium 前向+反向账单 · 中 = 分块 Bc 的影响(超过 256 收益封顶)· 右 = block-sparse 提速与稀疏度成比例 · 论文引言口径:HBM 访问最多省 9×',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = 'primer · 论文精髓图重绘'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

PY = 100                     # 三联顶部基线
PH = 380                     # 面板高(下缘 480)

# ================= 左:账单表(原图左联即此表) =================
LX, LW = MX, 430
lc.rect(LX, PY, LW, PH, '#ffffff', lc.C_MUTE, rx=10, sw=1.4)
lc.text(LX + 16, PY + 24, '(左)前向+反向账单:谁的 FLOP 多?谁快?', 11.5, lc.C_TXT,
        'start', True, maxw=LW - 32, tag='L:t')
lc.text(LX + 16, PY + 42, 'GPT-2 medium:seq 1024 · head dim 64 · 16 heads · batch 64 · A100',
        8.5, lc.C_MUTE, 'start', maxw=LW - 28, tag='L:cfg')
# 表格:3 行指标 × 2 实现
TX, TY = LX + 30, PY + 66
COL1, COL2, COL3 = 150, 130, 110          # 指标列/标准列/FA 列宽
ROWH = 52
lc.text(TX + COL1 / 2, TY + 4, 'Attention', 9.5, lc.C_MUTE, 'middle', True, maxw=COL1,
        tag='tb:h0')
lc.text(TX + COL1 + COL2 / 2, TY + 4, '标准实现', 9.5, lc.C_MUTE, 'middle', True,
        maxw=COL2, tag='tb:h1')
lc.text(TX + COL1 + COL2 + COL3 / 2, TY + 4, 'FlashAttention', 9.5, lc.C_GPU_S, 'middle',
        True, maxw=COL3, tag='tb:h2')
lc.seg(TX, TY + 14, TX + COL1 + COL2 + COL3, TY + 14, lc.C_MUTE, 1.2)
ROWS = [                                   # (指标, 标准, FA, 结论, 结论色)
    ('GFLOPs', '66.6', '75.2', 'FLOP 反而更多(反向重算)', '#b45309'),
    ('HBM R/W (GB)', '40.3', '4.4', '读写量 40.3 → 4.4:省 9.2×', lc.C_GPU_S),
    ('Runtime (ms)', '41.7', '7.3', '耗时 41.7 → 7.3:快 5.7×', lc.C_GPU_S),
]
for i, (m, a, b, note, nc) in enumerate(ROWS):
    ry = TY + 30 + i * ROWH
    lc.text(TX, ry + 12, m, 10, lc.C_TXT, 'start', True, maxw=COL1 - 8, tag='tb:m%d' % i)
    lc.text(TX, ry + 26, note, 8, nc, 'start', maxw=COL1 - 4, tag='tb:n%d' % i)
    lc.text(TX + COL1 + COL2 / 2, ry + 18, a, 13, lc.C_MUTE, 'middle', True, maxw=COL2 - 10,
            tag='tb:a%d' % i)
    lc.text(TX + COL1 + COL2 + COL3 / 2, ry + 18, b, 13, nc, 'middle', True, maxw=COL3 - 10,
            tag='tb:b%d' % i)
    if i < 2:
        lc.seg(TX, ry + ROWH - 6, TX + COL1 + COL2 + COL3, ry + ROWH - 6, '#e2e8f0', 1.0)
# 底部结论条
CY_ = PY + PH - 52
lc.rect(LX + 16, CY_, LW - 32, 38, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.4)
lc.text(LX + LW / 2, CY_ + 17, '同一个反直觉:HBM 访问才是主因——FLOP 多 13%,却快 5.7×',
        9.5, '#166534', 'middle', True, maxw=LW - 48, tag='L:concl1')
lc.text(LX + LW / 2, CY_ + 31, '(论文引言:HBM accesses up to 9× fewer, as shown in Fig. 2)',
        8, '#166534', 'middle', maxw=LW - 48, tag='L:concl2')

# ================= 中:Effect of Block Size =================
MXX, MW = 500, 450
lc.rect(MXX, PY, MW, PH, '#ffffff', lc.C_MUTE, rx=10, sw=1.4)
lc.text(MXX + 16, PY + 24, '(中)分块大小 Bc 的影响(前向)', 11.5, lc.C_TXT, 'start', True,
        maxw=MW - 32, tag='M:t')
lc.text(MXX + 16, PY + 42, '同一配置;实测曲线示意——趋势与刻度按原图', 8.5, lc.C_MUTE,
        'start', maxw=MW - 28, tag='M:cfg')
# 绘图区:左轴 HBM 访问(GB) 0-7,右轴 runtime;Bc 64/128/256/512
PX0, PX1 = MXX + 74, MXX + MW - 74
PB0, PB1 = PY + PH - 66, PY + 92               # x 轴 y / 绘图顶
GY0, GY7 = PB0, PB1                             # 左轴 0-7 GB 线性映射


def gy(v):
    return GY0 - (GY0 - GY7) * v / 7.0


BCS = [64, 128, 256, 512]


def bx(bc):
    return PX0 + (PX1 - PX0) * (bc - 64) / (512 - 64)


# 网格 + 左轴刻度 0/2/4/6(原图刻度)
for tv in (0, 2, 4, 6):
    y = gy(tv)
    lc.seg(PX0, y, PX1, y, '#e2e8f0' if tv else lc.C_MUTE, 1.0 if tv else 1.4)
    lc.text(PX0 - 8, y + 4, str(tv), 8.5, lc.C_MUTE, 'end', tag='M:lt%d' % tv)
lc.text(PX0 - 36, (PB1 + PB0) / 2, 'HBM 访问', 9.5, lc.C_TXT, 'middle', True, maxw=64,
        tag='M:yl')
lc.text(PX0 - 36, (PB1 + PB0) / 2 + 14, '(GB)', 8.5, lc.C_MUTE, 'middle', maxw=64,
        tag='M:yl2')
# x 轴
lc.seg(PX0, PB0, PX1, PB0, lc.C_MUTE, 1.4)
for bc in BCS:
    x = bx(bc)
    lc.seg(x, PB0, x, PB0 + 5, lc.C_MUTE, 1.2)
    lc.text(x, PB0 + 18, str(bc), 8.5, lc.C_MUTE, 'middle', tag='M:x%d' % bc)
lc.text((PX0 + PX1) / 2, PB0 + 34, 'Block Size Bc', 9, lc.C_TXT, 'middle', True, maxw=140,
        tag='M:xl')
# 绿:HBM 访问(矢量提取的形状:6.6→3.4→1.7→0.9,只画趋势)
pts_g = [(bx(64), gy(6.6)), (bx(128), gy(3.4)), (bx(256), gy(1.7)), (bx(512), gy(0.9))]
lc.parrow(pts_g, C_ACC, 2.4, marker=None)
for x, y in pts_g:
    lc.circle(x, y, 4, C_ACC, sw=1.8)
# 蓝:前向耗时(另一标尺,归一到左轴 0-7 区间画趋势:起点高、256 后走平)
pts_b = [(bx(64), gy(6.15)), (bx(128), gy(3.2)), (bx(256), gy(2.2)), (bx(512), gy(2.1))]
lc.parrow(pts_b, C_RUN, 2.4, marker=None, dash=True)
for x, y in pts_b:
    lc.circle(x, y, 4, C_RUN, sw=1.8)
# 256 封顶注记
x256 = bx(256)
lc.seg(x256, PB0, x256, PB1 - 6, C_RED, 1.2, dash=True)
lc.text(x256 + 8, PB1 + 4, 'Bc > 256:收益封顶', 9, C_RED, 'start', True, maxw=120,
        tag='M:cap1')
lc.text(x256 + 8, PB1 + 18, '瓶颈转为算术运算等其他因素;', 8, C_RED, 'start', maxw=150,
        tag='M:cap2')
lc.text(x256 + 8, PB1 + 31, '再大 SRAM 也装不下', 8, C_RED, 'start', maxw=150, tag='M:cap3')
# 图例
lc.text(MXX + 16, PY + PH - 20, '绿实线 = HBM 访问量(左轴,越少趟数越少)· 蓝虚线 = 前向耗时(趋势,Bc 越大越快、256 后走平)',
        8.5, lc.C_MUTE, 'start', maxw=MW - 28, tag='M:leg')

# ================= 右:Sparsity Speedup =================
RX, RW = 970, 488
lc.rect(RX, PY, RW, PH, '#ffffff', lc.C_MUTE, rx=10, sw=1.4)
lc.text(RX + 16, PY + 24, '(右)block-sparse:提速与稀疏度成比例', 11.5, lc.C_TXT, 'start',
        True, maxw=RW - 32, tag='R:t')
lc.text(RX + 16, PY + 42, 'seq 4K;实测曲线示意——趋势与刻度按原图', 8.5, lc.C_MUTE,
        'start', maxw=RW - 28, tag='R:cfg')
# 绘图区:x = 非零块占比 10-90%(刻度 20/60),y = 提速 0-7(刻度 2/4/6)
QX0, QX1 = RX + 60, RX + RW - 40
QB0, QB1 = PY + PH - 66, PY + 92


def qy(v):
    return QB0 - (QB0 - QB1) * v / 7.0


def qx(p):
    return QX0 + (QX1 - QX0) * (p - 10) / 80.0


for tv in (2, 4, 6):
    y = qy(tv)
    lc.seg(QX0, y, QX1, y, '#e2e8f0', 1.0)
    lc.text(QX0 - 8, y + 4, str(tv), 8.5, lc.C_MUTE, 'end', tag='R:lt%d' % tv)
lc.seg(QX0, QB0, QX1, QB0, lc.C_MUTE, 1.4)
for p in (20, 60):
    x = qx(p)
    lc.seg(x, QB0, x, QB0 + 5, lc.C_MUTE, 1.2)
    lc.text(x, QB0 + 18, '%d%%' % p, 8.5, lc.C_MUTE, 'middle', tag='R:x%d' % p)
lc.text((QX0 + QX1) / 2, QB0 + 34, '非零块占比(越少 = 越稀疏)', 9, lc.C_TXT, 'middle', True,
        maxw=200, tag='R:xl')
lc.text(QX0 - 34, (QB1 + QB0) / 2, '相对提速', 9.5, lc.C_TXT, 'middle', True, maxw=64,
        tag='R:yl')
# Dense 基准(红平线,提速=1)与 block-sparse 曲线(矢量提取形状:10%→~7、90%→~1)
lc.seg(QX0, qy(1.0), QX1, qy(1.0), C_RED, 2.0, dash=True)
pts_s = [(qx(10), qy(6.9)), (qx(20), qy(6.2)), (qx(30), qy(5.4)), (qx(40), qy(4.6)),
         (qx(50), qy(3.9)), (qx(60), qy(3.2)), (qx(70), qy(2.5)), (qx(80), qy(1.8)),
         (qx(90), qy(1.1))]
lc.parrow(pts_s, C_SPARSE, 2.4, marker=None)
for x, y in pts_s:
    lc.circle(x, y, 4, C_SPARSE, sw=1.8)
lc.text(qx(12), qy(6.9) - 12, '非零块越少(越稀疏),提速越高', 8.5, C_SPARSE, 'start', True,
        maxw=150, tag='R:hi')
lc.text(1350, 402, '非零块 90% ≈ Dense', 8.5, C_SPARSE, 'middle', maxw=120, tag='R:lo')
lc.text(1210, 372, 'Dense FlashAttention(基准 1×)', 8.5, C_RED, 'end', maxw=190,
        tag='R:base')
lc.text(RX + 16, PY + PH - 20, '紫线 = Block-Sparse FlashAttention · 红虚线 = Dense 基准 · 提速 ∝ 稀疏度(非零块越少,跳过越多)',
        8.5, lc.C_MUTE, 'start', maxw=RW - 28, tag='R:leg')

# ---------------- 页脚:出处 ----------------
LY = 500
lc.text(MX, LY, '左表数字为论文原图逐字(GFLOPs 66.6/75.2 · HBM R/W 40.3/4.4 GB · Runtime 41.7/7.3 ms);中/右为实测曲线示意(趋势与坐标刻度按原图,不标逐点数值)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, LY + 18, '重绘自 arXiv:2205.14135 Fig.2:HBM access is the primary factor affecting runtime · 超过 256 被 arithmetic 等因素卡住、再大装不进 SRAM(§3.2)· block-sparse 提速与稀疏度成比例(Fig.2 caption)· 数据 provenance = 论文原图与源码',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'paper-fig-2.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
