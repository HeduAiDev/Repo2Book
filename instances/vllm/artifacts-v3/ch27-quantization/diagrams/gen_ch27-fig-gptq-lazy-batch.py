#!/usr/bin/env python3
"""ch27 机制图 4 · GPTQ 二阶补偿与 lazy batch(figure_spec ch27-fig-gptq-lazy-batch,模板 tiling)

放大自 L0 GPU 执行臂(绿)第三块『模型层 forward + 编译』里 Linear 层权重矩阵的离线加工车间——
GPTQ 的全部产出(qweight/scales/g_idx)在此定型,之后 vLLM 只是消费。推导链第 4 环、全章最重一图。

claim:lazy batch 的结构(block_size=2 的 1×4 实例):块内逐列『量化→记误差 E→即时补偿块内』,
块末 E@U 一次性总账块外——col 0 的误差只即时改写 col 1(取整从 -8 翻成 -7),col 2/3 的改写
全部推迟到块末;同一副网格,层输出误差 0.001667 vs RTN 0.003978。

数字全部取自本章参考实现实跑(与 explainer 素材同源);坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 975
MX = 60
BXR = 1440
GRID = '#e2e8f0'
F_HOT, K_HOT = '#fff7ed', lc.C_ENG_S      # 被补偿改写
F_FRZ, K_FRZ = '#f1f5f9', lc.C_FAINT      # 已定格
F_LED, K_LED = '#f0fdf4', lc.C_GPU_S      # E 账本 / 保住

lc.text(MX, 34, 'GPTQ 主循环一图流:块内即时找补,块末一次性总账(lazy batch)',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '1×4 权重行、block_size=2:块内逐列「取整 → 记误差 E → 补偿块内还没取整的列」,块末把攒的账 E@U 一次摊给块外全部剩余列——bit 没变、网格没变,变的只是 round 的顺序与找补',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L0 GPU 执行臂 · 模型层 forward + 编译 · Linear 层权重(离线加工)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 主面板:tiling 条 + 状态演化表 =================
PX, PY, PW = MX, 122, 990
PH = 560
lc.rect(PX, PY, PW, PH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(PX + 16, PY + 22, '工作权重 W(1×4)的逐轮状态:每个方块一列,改写=橙,定格=灰 ✓',
        11, lc.C_TXT, 'start', True, maxw=800, tag='mp:h')

# ---- tiling 条:4 块 + 两个 block 外框 + 块内细箭头 / 块末粗箭头 ----
TW, TH = 96, 46
T_GAP_IN, T_GAP_BT = 14, 64            # 块内列距 / 块间距离
TX0 = PX + 150
TY = PY + 44
tiles_x = [TX0, TX0 + TW + T_GAP_IN,
           TX0 + 2 * TW + T_GAP_IN + T_GAP_BT, TX0 + 3 * TW + 2 * T_GAP_IN + T_GAP_BT]
W_INIT = ['0.45', '-0.2', '-0.05', '0.15']
for i, x in enumerate(tiles_x):
    lc.rect(x, TY, TW, TH, '#ffffff', lc.C_MUTE, rx=6, sw=1.2)
    lc.text(x + TW / 2, TY + 18, f'col {i}', 8.5, lc.C_MUTE, 'middle', tag='tile' + str(i))
    lc.text(x + TW / 2, TY + 36, f'w={W_INIT[i]}', 9.5, lc.C_TXT, 'middle', True, tag='tilew' + str(i))
# block 外框
BX_A = (tiles_x[0] - 8, tiles_x[1] + TW + 8)
BX_B = (tiles_x[2] - 8, tiles_x[3] + TW + 8)
BY0, BY1 = TY - 26, TY + TH + 8
lc.rect(BX_A[0], BY0, BX_A[1] - BX_A[0], BY1 - BY0, 'none', lc.C_MUTE, rx=8, sw=1.6, dash=True)
lc.rect(BX_B[0], BY0, BX_B[1] - BX_B[0], BY1 - BY0, 'none', lc.C_MUTE, rx=8, sw=1.6, dash=True)
lc.text((BX_A[0] + BX_A[1]) / 2, BY0 - 8, '块 A(B=2)', 9, lc.C_MUTE, 'middle', True, tag='blkA')
lc.text((BX_B[0] + BX_B[1]) / 2, BY0 - 8, '块 B(B=2)', 9, lc.C_MUTE, 'middle', True, tag='blkB')
# 块内即时补偿:细箭头 col0→col1、col2→col3
ay = TY + TH / 2
lc.seg(tiles_x[0] + TW, ay, tiles_x[1], ay, K_HOT, 1.2, 'std')
lc.text((tiles_x[0] + TW + tiles_x[1]) / 2, ay + 16, '即时', 8, K_HOT, 'middle', tag='imm0')
lc.seg(tiles_x[2] + TW, ay, tiles_x[3], ay, K_HOT, 1.2, 'std')
lc.text((tiles_x[2] + TW + tiles_x[3]) / 2, ay + 16, '即时', 8, K_HOT, 'middle', tag='imm1')
# 块末总账:粗箭头 A 外框右缘 → B 外框左缘(上/下两支)
lc.seg(BX_A[1], TY + 8, BX_B[0], TY + 8, lc.C_GPU_S, 2.6, 'std')
lc.seg(BX_A[1], TY + TH - 8, BX_B[0], TY + TH - 8, lc.C_GPU_S, 2.6, 'std')
lc.text((BX_A[1] + BX_B[0]) / 2, TY + TH + 22, 'E@U 一次性总账(块末)', 8.5, lc.C_GPU_S,
        'middle', True, tag='lazy')

# ---- 状态演化表 ----
TBLY = TY + TH + 44
ROW_H = 56
LBL_X = PX + 14
CELL_W = TW
CELL_X0 = tiles_x[0]
ANN_X = tiles_x[3] + TW + 16
ROWS = [
    ('初始', ['0.45', '-0.2', '-0.05', '0.15'], [], [],
     '网格提前固定:scale=0.65/15=0.04333,zp=−3(GPTQ 与 RTN 同一副网格)'),
    ('① 量化 col 0', ['0.43333', '-0.19445', '-0.05', '0.15'], [0], [1],
     'q=7,ŵ=0.43333 · 误差账 +0.02749 · 块内补偿改写 col 1:−0.2 → −0.19445'),
    ('② 量化 col 1', ['0.43333', '-0.17333', '-0.05', '0.15'], [1], [],
     'q=−7(RTN 是 −8!)——补偿把 −0.19445 推过取整边界 · 账 −0.03693'),
    ('A 块末 · 总账', ['0.43333', '-0.17333', '-0.04519', '0.1451'], [], [2, 3],
     'E=[+0.02749, −0.03693] 一次 E@U:col 2/3 同时改写——块外补偿全部推迟到此刻'),
    ('③ 量化 col 2', ['0.43333', '-0.17333', '-0.04333', '0.14448'], [2], [3],
     'q=−4,ŵ=−0.04333 · 账 −0.00218 · 块内补偿改写 col 3'),
    ('④ 量化 col 3', ['0.43333', '-0.17333', '-0.04333', '0.13'], [3], [],
     'q=0,ŵ=0.13 · 账 +0.03559(暂存)'),
    ('B 块末 · 总账', ['0.43333', '-0.17333', '-0.04333', '0.13'], [], [],
     '无剩余列 · E@U 空转——账本清零,算法终止'),
]
# 表头
lc.text(LBL_X, TBLY - 8, '轮次', 8.5, lc.C_MUTE, 'start', tag='th:lbl')
for i in range(4):
    lc.text(tiles_x[i] + CELL_W / 2, TBLY - 8, f'col {i}', 8.5, lc.C_MUTE,
            'middle', tag='th:c' + str(i))
lc.text(ANN_X, TBLY - 8, '发生的事(误差账 = (w−ŵ)/U_jj)', 8.5, lc.C_MUTE, 'start', tag='th:ann')
for r, (lbl, vals, frozen, changed, ann) in enumerate(ROWS):
    yy = TBLY + r * ROW_H
    block_end = lbl.startswith(('A', 'B'))
    lc.text(LBL_X, yy + 16, lbl, 9, lc.C_TXT if not block_end else lc.C_GPU_S, 'start', True,
            maxw=134, tag='rl' + str(r))
    for c in range(4):
        x = tiles_x[c]
        if c in frozen:
            f_, k_ = F_FRZ, K_FRZ
        elif c in changed:
            f_, k_ = F_HOT, K_HOT
        else:
            f_, k_ = '#ffffff', GRID
        sw = 1.8 if c in changed and block_end else 1.1
        lc.rect(x, yy, CELL_W, 34, f_, k_, rx=5, sw=sw)
        mark = ' ✓' if c in frozen else ''
        col_ = K_FRZ if c in frozen else (K_HOT if c in changed else '#334155')
        lc.text(x + CELL_W / 2, yy + 22, vals[c] + mark, 9, col_, 'middle',
                bold=(c in changed), tag='cv' + str(r) + str(c))
    # 注释两行
    lc.text(ANN_X, yy + 13, ann, 8, '#334155', 'start', maxw=PW - (ANN_X - PX) - 14,
            tag='ra' + str(r))
    if r < len(ROWS) - 1:
        lc.seg(CELL_X0 - 40, yy + 40, CELL_X0 - 40, yy + ROW_H - 4, lc.C_FAINT, 1.0)
# 底注
lc.text(PX + 16, PY + PH - 26, '终止性:列指针每轮 +1,恰量化一次;已定格列此后无人读(lazy 合法:推迟对后续列的补偿不改变任何已做的取整)',
        8.5, lc.C_MUTE, 'start', maxw=940, tag='mp:foot')

# ================= 右列三面板 =================
RX, RW = 1080, 360

# ---- U 上三角 + E 账本 ----
UY, UH = 122, 226
lc.rect(RX, UY, RW, UH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(RX + 14, UY + 20, 'U = Cholesky(H⁻¹)ᵀ 上三角(E@U 的 U)', 10, lc.C_TXT, 'start', True,
        maxw=RW - 28, tag='u:h')
UM = [
    ['0.60637', '-0.20205', '0.19435', '-0.19824'],
    ['', '0.57172', '0.27481', '-0.28031'],
    ['', '', '0.85195', '-0.2821'],
    ['', '', '', '0.40689'],
]
UCW = 82
ux0, uy0 = RX + 20, UY + 34
for r in range(4):
    for c in range(4):
        v = UM[r][c]
        if not v:
            continue
        diag = (r == c)
        x = ux0 + c * UCW
        y = uy0 + r * 24
        if diag:
            lc.rect(x - 2, y - 14, UCW - 6, 20, F_LED, K_LED, rx=4, sw=1.2)
        lc.text(x + UCW / 2 - 3, y, v, 7.8, K_LED if diag else '#334155', 'middle', bold=diag,
                tag='um' + str(r) + str(c))
lc.text(RX + 14, UY + 140, '对角 U_jj:0.60637 / 0.57172 / 0.85195 / 0.40689', 8.2, lc.C_MUTE,
        'start', maxw=RW - 28, tag='u:diag')
lc.text(RX + 14, UY + 158, 'E 账本:块 A [+0.02749, −0.03693]', 8.5, K_LED, 'start', True,
        maxw=RW - 28, tag='u:e1')
lc.text(RX + 14, UY + 174, '块 B [−0.00218, +0.03559](块末清零)', 8.5, K_LED, 'start',
        maxw=RW - 28, tag='u:e2')
lc.text(RX + 14, UY + 196, 'Cholesky 替掉反复求逆的数值累积误差(§4 Step 3)', 8, lc.C_MUTE,
        'start', maxw=RW - 28, tag='u:note')

# ---- H 面板 ----
HY, HH = 360, 178
lc.rect(RX, HY, RW, HH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(RX + 14, HY + 20, 'H = 2XᵀX(4 条 0/1 校准样本,可手算)', 10, lc.C_TXT, 'start', True,
        maxw=RW - 28, tag='h:h')
HM = [[4, 2, 0, 2], [2, 4, 0, 2], [0, 0, 2, 2], [2, 2, 2, 6]]
for r in range(4):
    row = '  '.join(str(v) for v in HM[r])
    lc.text(RX + 24, HY + 42 + r * 18, '[ ' + row + ' ]', 8.5, '#334155', 'start', tag='hm' + str(r))
lc.text(RX + 14, HY + 122, '· 只依赖层输入、与权重无关 → 全行共享同一列序(§4 Step 1)', 8.2,
        '#334155', 'start', maxw=RW - 28, tag='h:l1')
lc.text(RX + 14, HY + 138, '· λ=0.04 dampening(1% 平均对角):样本数<特征数时 H 奇异,防反复求逆推成不定阵', 8.2,
        '#334155', 'start', maxw=RW - 28, tag='h:l2')
lc.text(RX + 14, HY + 160, '· OBQ 贪心序 [2, 0, 1, 3] vs 固定列序 0→3:最终误差相近(1.48 倍内)', 8.2,
        '#334155', 'start', maxw=RW - 28, tag='h:l3')

# ---- 记分板 ----
SY, SH = 550, 132
lc.rect(RX, SY, RW, SH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(RX + 14, SY + 20, '记分板:同一副网格,只差 round 方式', 10, lc.C_TXT, 'start', True,
        maxw=RW - 28, tag='s:h')
lc.text(RX + 14, SY + 44, 'GPTQ 码 [7, −7, −4, 0]  vs  RTN 码 [7, −8, −4, 0]', 9,
        lc.C_TXT, 'start', True, maxw=RW - 28, tag='s:codes')
lc.text(RX + 14, SY + 64, '层输出误差(Eq.1):GPTQ 0.001667 vs RTN 0.003978 = 2.4 倍', 9,
        K_LED, 'start', True, maxw=RW - 28, tag='s:err')
lc.text(RX + 14, SY + 84, 'col 1 单点误差反而更大:0.02667 vs 0.01667——', 8.2, lc.C_MUTE,
        'start', maxw=RW - 28, tag='s:note1')
lc.text(RX + 14, SY + 100, '二阶补偿优化的是「层输出」,不是「单点」', 8.2, lc.C_MUTE,
        'start', maxw=RW - 28, tag='s:note2')

# ================= 底部三注 =================
BY = 698
lc.rect(MX, BY, 1380, 118, '#ffffff', lc.C_MUTE, rx=8, sw=1.1, dash=True)
NOTES = [
    ('复杂度(§4 Step 1)', [
        'OBQ O(d_row·d_col³) → GPTQ O(max{d_row·d_col², d_col³})',
        '4096×4096 层:2.8e14 → 6.9e10 FLOPs = 4096 倍',
        'H⁻¹ 更新从每权重一次降到每列一次 · 175B 单卡 A100 ≈ 4 GPU 小时']),
    ('等价性(逐位对账)', [
        'lazy 不改结果:B=8 与 B=3 码逐位相同',
        'Cholesky+lazy 与朴素逐列 Eq.2/Eq.3 码逐位相同',
        'w_hat 最大|diff| = 0.0——只改执行方式,不改数学']),
    ('规模记分板', [
        '8×12 合成层:3-bit 177.5999 → 30.6096(5.8 倍)',
        '4-bit 17.8034 → 4.1593(4.28 倍)',
        '论文口径 OPT-175B 3-bit:RTN 7.3e3 → GPTQ 8.68']),
]
NW = 448
for i, (t, lines) in enumerate(NOTES):
    x = MX + 12 + i * (NW + 12)
    lc.text(x, BY + 20, t, 9.5, lc.C_TXT, 'start', True, tag='nt' + str(i))
    for j, ln in enumerate(lines):
        lc.text(x, BY + 40 + j * 18, ln, 8.3, '#334155', 'start', maxw=NW - 8, tag='ntl' + str(i) + str(j))

lc.text(MX, BY + 134, '论文口径 arXiv:2210.17323 §3 Eq.2 / §4 Algorithm 1(结构对照其 Figure 2:白色列正在量化、蓝色为待更新——本图按本章 1×4 实例重画) · 数值:本章 NumPy 参考实现实跑 · 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=1380, tag='foot')

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch27-fig-gptq-lazy-batch.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
