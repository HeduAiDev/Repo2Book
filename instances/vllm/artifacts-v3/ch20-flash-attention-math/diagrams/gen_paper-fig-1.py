#!/usr/bin/env python3
"""ch20 论文精髓图 ① · paper-fig-1(arXiv:2205.14135 Fig.1 忠实重绘)

writer figure-requests.json add:FlashAttention 标志性配图——左:外层循环沿 K/V 列块(红箭头)
搬进 SRAM、内层沿 Q 行块(蓝箭头),N×N 注意力矩阵(虚线框)从不物化到 HBM;右:GPT-2 上
注意力本体对 PyTorch 标准实现 7.6× 加速。

原图真相源(arXiv e-print 2205.14135 源码 figs/banner_pdf.pdf 文本层 + 矢量几何逐项提取):
- 左:三层带宽金字塔 GPU SRAM 19 TB/s(20 MB) / GPU HBM 1.5 TB/s(40 GB) / 主存 CPU DRAM
  12.8 GB/s(>1 TB);HBM 里 Q(N×d) 左、K^T(d×N) 上、V(N×d) 下、O(N×d) 右四条矩阵带,
  N×N(QK^T)以 dotted box 标注、从不读/写 HBM;GPU SRAM 盒内做块计算,红箭头=外层 K/V 列块、
  蓝箭头=内层 Q 行块、输出写回 HBM。
- 右:两根堆叠柱,y 轴 Time (ms) 刻度 0/5/10/15;PyTorch 柱 5 段(Matmul/Mask/Softmax/Dropout/
  Matmul,按原图矢量比例 2.2/4.5/3.7/4.6/1.8 ms,合计 16.8 ms),FlashAttention 柱单段
  Fused Kernel(2.2 ms),比值 7.6×(图上只标 7.6× 与刻度,逐段 ms 不上图)。

布局与信息结构对齐原图;配色/字体套本书视觉语言;文字译中。provenance=原论文图本身
(key_figures 豁免,不走 explainer figure_specs 通道)。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 700
MX = 42
BXR = 1458
C_RED = '#dc2626'
C_BLUE = '#2563eb'
C_SRAM, C_SRAM_F = lc.C_ENG_S, lc.C_ENG_F        # SRAM = 橙(快而小)
C_HBM, C_HBM_F = lc.C_GPU_S, lc.C_GPU_F          # HBM = 绿
C_DRAM, C_DRAM_F = '#64748b', '#f1f5f9'          # CPU DRAM = 灰
EXTRA_DEFS = (
    '<marker id="red" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
    '<marker id="blu" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker>'
    '<marker id="grn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>')

# ---------------- 标题区 ----------------
lc.text(MX, 32, '论文原景:外层搬 K/V 列块、内层搬 Q 行块,N×N 从不落 HBM——GPT-2 实测 7.6×',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 56, '重绘自 arXiv:2205.14135 Fig.1:左 = tiling 机制(存储层级 + 双循环搬运 + 虚线 N×N)· 右 = GPT-2 注意力耗时对比(PyTorch 五算子分开跑 vs FlashAttention 一个融合 kernel)',
        10.5, lc.C_MUTE, 'start', maxw=1060, tag='subtitle')
_ch = 'primer · 论文精髓图重绘'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 左:tiling 机制 =================
lc.text(MX, 92, '(左)机制:两层数据搬运循环', 12, lc.C_TXT, 'start', True, maxw=380, tag='L:t')
lc.text(560, 92, '红 = 外层循环(沿 K/V 列块)· 蓝 = 内层循环(沿 Q 行块)· 绿 = 输出写回',
        9, lc.C_MUTE, 'start', maxw=380, tag='L:leg')

# ---- 存储层级金字塔(带宽/容量逐字取自原图) ----
PYX, PYY, PW = 50, 110, 150
LAYERS = [                               # (宽比, 高, 名, 带宽·容量, 色)
    (0.44, 44, 'GPU SRAM', '19 TB/s · 20 MB', C_SRAM, C_SRAM_F),
    (0.70, 52, 'GPU HBM', '1.5 TB/s · 40 GB', C_HBM, C_HBM_F),
    (1.00, 60, '主存 CPU DRAM', '12.8 GB/s · >1 TB', C_DRAM, C_DRAM_F),
]
py_y = PYY
for frac, lh, name, bwc, cst, cfl in LAYERS:
    lw = PW * frac
    lx = PYX + (PW - lw) / 2
    lc.rect(lx, py_y, lw, lh, cfl, cst, rx=4, sw=1.5)
    lc.text(PYX + PW / 2, py_y + lh / 2 + 3.5, name, 9.5, cst, 'middle', True,
            maxw=lw - 6, tag='py:' + name)
    lc.seg(lx + lw + 2, py_y + lh / 2, PYX + PW + 14, py_y + lh / 2, cst, 1.0)
    lc.text(PYX + PW + 18, py_y + lh / 2 + 3, bwc, 8.5, cst, 'start', maxw=118,
            tag='py:r' + name)
    py_y += lh
lc.text(PYX + PW / 2, py_y + 16, '存储层级:越往上越快、越小', 8.5, lc.C_MUTE, 'middle',
        maxw=PW + 40, tag='py:cap')

# ---- GPU SRAM 计算盒(左中) ----
SX, SY, SW_, SH_ = 330, 330, 190, 182
lc.rect(SX, SY, SW_, SH_, C_SRAM_F, C_SRAM, rx=10, sw=2.0)
lc.text(SX + SW_ / 2, SY + 20, 'GPU SRAM(片上)', 10.5, C_SRAM, 'middle', True,
        maxw=SW_ - 16, tag='sr:t')
lc.text(SX + SW_ / 2, SY + 38, '快而小——只装得下小块', 8, lc.C_MUTE, 'middle',
        maxw=SW_ - 14, tag='sr:sub')
lc.rect(SX + 22, SY + 52, 34, 20, '#fde68a', '#d97706', rx=3, sw=1.4)
lc.rect(SX + 22, SY + 78, 34, 20, '#fde68a', '#d97706', rx=3, sw=1.4)
lc.text(SX + 64, SY + 66, 'K_j、V_j(外层逐块)', 8.5, '#b45309', 'start', maxw=112,
        tag='sr:kv')
lc.rect(SX + 22, SY + 104, 34, 20, '#bfdbfe', C_BLUE, rx=3, sw=1.4)
lc.text(SX + 64, SY + 118, 'Q_i(内层,每块遍历)', 8.5, C_BLUE, 'start', maxw=108, tag='sr:q')
lc.text(SX + SW_ / 2, SY + 146, '片上算完 S_ij = Q_i K_j^T', 8.5, lc.C_TXT, 'middle',
        maxw=SW_ - 12, tag='sr:c1')
lc.text(SX + SW_ / 2, SY + 162, '→ softmax → ·V_j → 折算累进 O_i', 8.5, lc.C_TXT, 'middle',
        maxw=SW_ - 12, tag='sr:c2')

# ---- HBM 大框:四条矩阵带 + 上中虚线 N×N ----
HX, HY, HW, HH = 560, 100, 350, 480      # x 560-910, y 100-580
lc.rect(HX, HY, HW, HH, C_HBM_F, C_HBM, rx=10, sw=2.0)
lc.text(HX + 14, HY + 18, 'GPU HBM(慢而大)', 9.5, C_HBM, 'start', True, maxw=130,
        tag='hbm:t')

NCOL, QROW = 9, 7
CW, CH_, CG = 26, 32, 3                 # 横带格宽 / 竖带格高 / 格距
# K^T 带(上,d×N)
KT_X, KT_Y = 640, 130
for c in range(NCOL):
    hot = c == 0
    lc.rect(KT_X + c * (CW + CG), KT_Y, CW, 26,
            '#fde68a' if hot else '#f1f5f9', '#d97706' if hot else '#cbd5e1',
            rx=3, sw=1.6 if hot else 1.0)
lc.text(KT_X + 13, KT_Y + 17, 'K_j', 8, '#b45309', 'middle', True, maxw=24, tag='kt:hot')
lc.text(KT_X - 10, KT_Y + 17, 'K^T: d×N', 9.5, lc.C_TXT, 'end', True, maxw=80, tag='kt:lab')
# V 带(下,N×d)
V_X, V_Y = 640, 510
for c in range(NCOL):
    hot = c == 0
    lc.rect(V_X + c * (CW + CG), V_Y, CW, 26,
            '#fde68a' if hot else '#f1f5f9', '#d97706' if hot else '#cbd5e1',
            rx=3, sw=1.6 if hot else 1.0)
lc.text(V_X + 13, V_Y + 17, 'V_j', 8, '#b45309', 'middle', True, maxw=24, tag='v:hot')
lc.text(V_X - 10, V_Y + 17, 'V: N×d', 9.5, lc.C_TXT, 'end', True, maxw=70, tag='v:lab')
# Q 带(左,N×d)竖条:y 200-460
Q_X, Q_Y = 584, 200
for r in range(QROW):
    hot = r == 3
    lc.rect(Q_X, Q_Y + r * (CH_ + CG), 26, CH_,
            '#bfdbfe' if hot else '#f1f5f9', C_BLUE if hot else '#cbd5e1',
            rx=3, sw=1.6 if hot else 1.0)
lc.text(Q_X + 13, Q_Y + 3 * (CH_ + CG) + CH_ / 2 + 3, 'Q_i', 8, C_BLUE, 'middle', True,
        maxw=24, tag='q:hot')
lc.text(Q_X + 13, Q_Y - 10, 'Q: N×d', 9.5, lc.C_TXT, 'middle', True, maxw=80, tag='q:lab')
# O 带(右,N×d)竖条:y 200-460
O_X, O_Y = 864, 200
for r in range(QROW):
    hot = r == 3
    lc.rect(O_X, O_Y + r * (CH_ + CG), 26, CH_,
            '#bbf7d0' if hot else '#f1f5f9', lc.C_GPU_S if hot else '#cbd5e1',
            rx=3, sw=1.6 if hot else 1.0)
lc.text(O_X + 13, O_Y + 3 * (CH_ + CG) + CH_ / 2 + 3, 'O_i', 8, lc.C_GPU_S, 'middle', True,
        maxw=24, tag='o:hot')
lc.text(O_X + 13, O_Y - 10, 'O: N×d', 9.5, lc.C_TXT, 'middle', True, maxw=80, tag='o:lab')

# 虚线 N×N 框(上中):从不物化
DB_X, DB_Y, DB_W, DB_H = 660, 186, 190, 152
lc.rect(DB_X, DB_Y, DB_W, DB_H, '#fef2f2', C_RED, rx=8, sw=2.0, dash=True)
lc.text(DB_X + DB_W / 2, DB_Y + 26, 'QK^T: N×N', 13, C_RED, 'middle', True, maxw=DB_W,
        tag='db:main')
lc.text(DB_X + DB_W / 2, DB_Y + 44, 'softmax(QK^T) 同样不落 HBM', 8.5, lc.C_MUTE,
        'middle', maxw=DB_W - 14, tag='db:sm')
cx, cy, cr = DB_X + DB_W / 2, DB_Y + 96, 22
lc.seg(cx - cr, cy - cr, cx + cr, cy + cr, C_RED, 3.0)
lc.seg(cx + cr, cy - cr, cx - cr, cy + cr, C_RED, 3.0)
lc.text(DB_X + DB_W / 2, DB_Y + DB_H - 12, '从不读 / 写 HBM(dotted box)', 8.5, C_RED,
        'middle', True, maxw=DB_W - 8, tag='db:never')

# ---- 四条箭头(全部沿无障碍廊道,端点贴框边) ----
# 外层红:K_j 出 K^T 带 → 上廊道(y=173)→ 左缘竖廊(x=548)→ SRAM 盒右缘
lc.parrow([(653, 158), (653, 173), (548, 173), (548, 340), (524, 340)], C_RED, 2.2, 'red')
# 内层蓝:Q_i 出 Q 带左缘 → 横穿 HBM 边界 → SRAM 盒右缘
qi_cy = Q_Y + 3 * (CH_ + CG) + CH_ / 2
lc.parrow([(Q_X - 2, qi_cy), (524, qi_cy)], C_BLUE, 2.2, 'blu')
# 外层红:V_j 出 V 带 → 下廊道(y=493)→ 左缘竖廊 → SRAM 盒右缘
lc.parrow([(653, 508), (653, 493), (548, 493), (548, 452), (524, 452)], C_RED, 2.2, 'red')
# 输出绿:SRAM 盒底 → 画布底廊(y=600)→ 右缘竖廊(x=900)→ O 带右缘
oi_cy = O_Y + 3 * (CH_ + CG) + CH_ / 2
lc.parrow([(SX + SW_ / 2, SY + SH_ + 2), (SX + SW_ / 2, 600), (900, 600), (900, oi_cy),
           (894, oi_cy)], lc.C_GPU_S, 2.2, 'grn')
lc.text(660, 614, '输出写回 HBM', 9, lc.C_GPU_S, 'middle', True, maxw=120, tag='ar:out3')

# ================= 右:GPT-2 耗时对比柱 =================
RPX = 942
lc.text(RPX, 92, '(右)GPT-2 注意力耗时', 12, lc.C_TXT, 'start', True, maxw=320, tag='R:t')
AX_X, AX_B, AX_T = RPX + 60, 586, 128          # y 轴 x / 基线 / 顶
AX_H = AX_B - AX_T                              # 375px
MS = 17.5                                       # 轴顶对应 17.5 ms


def ms_y(v):
    return AX_B - AX_H * v / MS


lc.seg(AX_X, AX_T, AX_X, AX_B, lc.C_MUTE, 1.6)
lc.seg(AX_X, AX_B, RPX + 492, AX_B, lc.C_MUTE, 1.6)
for tv in (0, 5, 10, 15):                       # 原图刻度 0/5/10/15
    y = ms_y(tv)
    lc.seg(AX_X - 5, y, AX_X, y, lc.C_MUTE, 1.2)
    lc.text(AX_X - 10, y + 4, str(tv), 9, lc.C_MUTE, 'end', tag='ax:tick%d' % tv)
lc.text(AX_X - 36, (AX_T + AX_B) / 2, '耗时 (ms)', 10, lc.C_TXT, 'middle', True, maxw=70,
        tag='ax:ylab')
# PyTorch 柱:5 段堆叠(段高按原图矢量比例 2.2/4.5/3.7/4.6/1.8 ms;只标算子名)
BW = 92
B1X = AX_X + 66
SEGS = [('Matmul', 2.2, '#fca5a5'), ('Mask', 4.5, '#f87171'), ('Softmax', 3.7, '#ef4444'),
        ('Dropout', 4.6, '#dc2626'), ('Matmul', 1.8, '#b91c1c')]
sy = AX_B
for name, v, col in SEGS:
    h = AX_H * v / MS
    sy -= h
    lc.rect(B1X, sy, BW, h, col, '#ffffff', rx=2, sw=1.0)
    lc.text(B1X + BW / 2, sy + h / 2 + 3.5, name, 9, '#ffffff', 'middle', True,
            maxw=BW - 6, tag='seg:' + name)
lc.text(B1X + BW / 2, AX_B + 18, 'PyTorch 标准实现', 9.5, lc.C_TXT, 'middle', True,
        maxw=BW + 60, tag='b1:lab')
lc.text(B1X + BW / 2, AX_B + 34, '(五个算子分开跑)', 8, lc.C_MUTE, 'middle', maxw=BW + 40,
        tag='b1:sub')
# FlashAttention 柱:单段融合 kernel(2.2 ms)
B2X = AX_X + 258
h2 = AX_H * 2.2 / MS
lc.rect(B2X, AX_B - h2, BW, h2, lc.C_GPU_S, '#15803d', rx=2, sw=1.2)
lc.text(B2X + BW / 2, AX_B - h2 / 2 + 3.5, '融合 kernel', 9, '#ffffff', 'middle', True,
        maxw=BW - 6, tag='b2:seg')
lc.text(B2X + BW / 2, AX_B + 18, 'FlashAttention', 9.5, lc.C_TXT, 'middle', True,
        maxw=BW + 40, tag='b2:lab')
lc.text(B2X + BW / 2, AX_B + 34, '(一个 kernel 做完全部)', 8, lc.C_MUTE, 'middle',
        maxw=BW + 50, tag='b2:sub')
# 7.6× 注记:两柱顶之间
ty1, ty2 = AX_B - AX_H * 16.8 / MS, AX_B - h2
lc.parrow([(B1X + BW + 8, ty1 + 30), (B2X - 10, ty1 + 30)], C_RED, 1.8, 'red')
lc.parrow([(B2X - 10, ty2 - 14), (B1X + BW + 8, ty2 - 14)], C_RED, 1.8, 'red')
lc.text((B1X + BW + B2X) / 2, ty1 + 58, '7.6×', 22, C_RED, 'middle', True, maxw=110,
        tag='x76')
lc.text((B1X + BW + B2X) / 2, ty1 + 78, '提速(注意力计算本体)', 9, C_RED, 'middle',
        maxw=170, tag='x76:sub')
lc.text(RPX + 6, AX_T - 14, '不读、不写大 N×N 矩阵,把 HBM 往返全省掉', 9.5, lc.C_BEAT_T,
        'start', True, maxw=470, tag='r:claim')

# ---------------- 页脚:图例 + 出处 ----------------
LY = 648
lc.text(MX, LY, '图例:橙 = SRAM(片上)· 绿 = HBM · 灰 = CPU DRAM · 黄块 = K_j/V_j 列块 · 蓝块 = Q_i 行块 · 绿块 = O_i · 红虚线框 = N×N(从不物化)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, LY + 18, '带宽/容量、算子分解、刻度均取自论文原图(GPU SRAM 19 TB/s · 20 MB / GPU HBM 1.5 TB/s · 40 GB / CPU DRAM 12.8 GB/s · >1 TB;Time (ms) 0/5/10/15;PyTorch = Matmul+Mask+Softmax+Dropout+Matmul vs Fused Kernel)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, LY + 36, '重绘自 arXiv:2205.14135 Fig.1:外层沿 K/V 列块、内层沿 Q 行块搬进 SRAM,N×N(虚线框)从不物化——GPT-2 上注意力本体 7.6× 加速 · 数据 provenance = 论文原图(arXiv e-print 源码矢量提取)',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
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
