#!/usr/bin/env python3
"""ch20 机制图 ③ · FA tiling 免物化(figure_spec ch20-fig-fa-tiling,模板 tiling)

放大自 L0 中列『GPU 执行臂』(绿色列)第三块『模型层 forward + 编译』内 attention kernel
的循环内部。primer 推导链第 ④ 环:把 online-softmax 的单遍递推装进双循环——SRAM 里的
2×2 小表与永不落地的 N×N 大表同框对照。架构归属回指 L0(FIGURE-SYSTEM §3.3)。

claim:FA tiling——外层搬 KV 块、内层过 Q 块,片上只存在 B_r×B_c=2×2 的 S_ij 与局部 softmax,
running (m,ℓ,O) 逐块折算-累加;4×4 的 S/P 整表从头到尾不存在,输出却与朴素实现逐位相同。

数字全部取自 figure_spec.numbers(片上最大块形状 (2,2) vs 整表 (4,4) 实跑断言;步 (1,0)/(1,1)
折算事件 0.3679 与 ℓ 变化;每步『至今为止的正确答案』差 0.0;终值逐行相同——host NumPy
参考实现实跑)。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 670
MX = 60
BXR = 1440
C_RED = '#dc2626'
C_GRAY_CELL = '#f8fafc'
EXTRA_DEFS = ('<marker id="grn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'FA tiling:桌面上永远只有 2×2 的草稿——4×4 的整张 S/P 从头到尾不存在',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'N=4、d=2、B_r=B_c=2 的心算例:外层搬 KV 列块、内层过 Q 行块,每个 (j,i) 块在片上算局部 softmax,running (m,ℓ,O) 折算-累加(arXiv:2205.14135 §3.1 Alg.1)',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = '推导链 ④ · 放大自 L0 GPU 执行臂内 attention kernel 的双循环'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左:整张表的虚影(从未存在) ----------------
lc.text(96, 94, '标准做法要的整张 S(和 P)', 11, lc.C_TXT, 'start', True, maxw=260, tag='lp:t')
GX0, GY0, GC, GS = 96, 118, 42, 2
GW = 4 * GC + 3 * GS
for r in range(4):
    for c in range(4):
        lc.rect(GX0 + c * (GC + GS), GY0 + r * (GC + GS), GC, GC, C_GRAY_CELL,
                '#cbd5e1', rx=2, sw=1.0, dash=True)
lc.seg(GX0 + 2, GY0 + 2, GX0 + GW - 2, GY0 + GW - 2, C_RED, 4.0)
lc.seg(GX0 + GW - 2, GY0 + 2, GX0 + 2, GY0 + GW - 2, C_RED, 4.0)
lc.text(GX0 + GW / 2, GY0 + GW + 22, '从未被创建', 11, C_RED, 'middle', True, maxw=200, tag='ghost:t')
lc.text(GX0 + GW / 2, GY0 + GW + 40, '实跑断言:S/P 只以 2×2 块存在于片上', 8.5, '#334155',
        'middle', maxw=260, tag='ghost:l1')
lc.text(GX0 + GW / 2, GY0 + GW + 56, '局部变量(代表 SRAM)——最大块形状', 8.5, '#334155',
        'middle', maxw=260, tag='ghost:l2')
lc.text(GX0 + GW / 2, GY0 + GW + 72, '(2,2),整表 (4,4) 一次都没建过', 8.5, '#334155',
        'middle', maxw=260, tag='ghost:l3')

# 对比箭头
lc.text(322, 190, '同一份数学', 9, lc.C_MUTE, 'middle', maxw=120, tag='vs')
lc.seg(288, 205, 352, 205, lc.C_MUTE, 2.0, 'std')

# ---------------- 中:FA 双循环 2×2 分块格盘 ----------------
lc.text(360, 94, 'FA 的走法:双循环,格盘上只有 2×2', 11, lc.C_TXT, 'start', True, maxw=330,
        tag='rp:t')
QX0, QY0, QC, QI, QG = 360, 122, 44, 2, 14          # 起点/格边长/格内距/象限距
QW = 2 * (2 * QC + QI) + QG                          # 194
QH = QW
quads = [  # (j, i, x, y, 状态)
    (0, 0, QX0, QY0, 'done'),
    (1, 0, QX0 + 2 * QC + QI + QG, QY0, 'cur'),
    (0, 1, QX0, QY0 + 2 * QC + QI + QG, 'done'),
    (1, 1, QX0 + 2 * QC + QI + QG, QY0 + 2 * QC + QI + QG, 'todo'),
]
QFILL = {'done': ('#dcfcc7', lc.C_GPU_S, True), 'cur': ('#86efac', lc.C_GPU_S, False),
         'todo': (C_GRAY_CELL, '#94a3b8', True)}
for j, i, qx, qy, st in quads:
    fill, stroke, dash = QFILL[st]
    for r in range(2):
        for c in range(2):
            lc.rect(qx + c * (QC + QI), qy + r * (QC + QI), QC, QC, fill, stroke,
                    rx=3, sw=1.0, dash=dash)
# 步序徽标 ①-④(挂在各象限左上角)
BADGE = {0: (373, 135, '①', 'done'), 1: (373, 245, '②', 'done'),
         2: (477, 135, '③', 'cur'), 3: (477, 245, '④', 'todo')}
for n, (bx, by, ch_, st) in BADGE.items():
    fill, txt = (lc.C_GPU_S, '#ffffff') if st == 'cur' else (
        '#dcfcc7' if st == 'done' else '#e2e8f0', lc.C_TXT)
    lc.rect(bx - 11, by - 11, 22, 22, fill, lc.C_GPU_S if st != 'todo' else '#94a3b8',
            rx=11, sw=1.4)
    lc.text(bx, by + 4.5, ch_, 11, txt, 'middle', True, maxw=20, tag='bdg' + ch_)
# 步序箭头 ①→②→③→④(竖段走象限间隙)
lc.seg(373, 148, 373, 232, lc.C_GPU_S, 1.8, 'grn')
lc.parrow([(386, 245), (457, 245), (457, 135), (464, 135)], lc.C_GPU_S, 1.8, 'grn')
lc.seg(477, 148, 477, 232, lc.C_GPU_S, 1.8, 'grn')
# 轴标
lc.text(405, 116, 'K_0,V_0(列 0-1)', 8.5, lc.C_KV_S, 'middle', True, maxw=120, tag='ax:k0')
lc.text(509, 116, 'K_1,V_1(列 2-3)', 8.5, lc.C_KV_S, 'middle', True, maxw=120, tag='ax:k1')
lc.text(352, 163, 'Q_0', 9, lc.C_GPU_S, 'end', True, maxw=60, tag='ax:q0')
lc.text(352, 175, '行 0-1', 8, lc.C_MUTE, 'end', maxw=60, tag='ax:q0s')
lc.text(352, 273, 'Q_1', 9, lc.C_GPU_S, 'end', True, maxw=60, tag='ax:q1')
lc.text(352, 285, '行 2-3', 8, lc.C_MUTE, 'end', maxw=60, tag='ax:q1s')
# 当前块 → 显微镜连线
lc.seg(554 + 2, 167, 620 - 2, 167, lc.C_GPU_S, 2.2, 'grn')
# 步序说明
lc.text(QX0 + QW / 2, QY0 + QH + 22,
        '步序(Alg.1 line 5/7):外层 j = KV 列块、内层 i = Q 行块', 8.5, lc.C_MUTE, 'middle',
        maxw=280, tag='rp:ord1')
lc.text(QX0 + QW / 2, QY0 + QH + 38, '①→②→③→④,当前步 = ③', 8.5, lc.C_MUTE, 'middle',
        maxw=280, tag='rp:ord2')

# ---------------- 右:显微镜(当前块片上三件套) ----------------
MP_X, MP_Y, MP_W, MP_H = 620, 94, 820, 200
lc.rect(MP_X, MP_Y, MP_W, MP_H, '#ffffff', lc.C_GPU_S, rx=8, sw=1.6)
lc.text(MP_X + 14, MP_Y + 20, '显微镜:步 ③(j=1, i=0)此刻片上——Q_0 行块 × K_1 列块,SRAM 里只有这些',
        10.5, lc.C_GPU_S, 'start', True, maxw=MP_W - 28, tag='mp:t')
# S_ij 2×2 值格
SC, SS = 40, 3
SX0, SY0 = MP_X + 22, MP_Y + 36
svals = [['1', '0'], ['1', '2']]
for r in range(2):
    for c in range(2):
        lc.rect(SX0 + c * (SC + SS), SY0 + r * (SC + SS), SC, SC, '#f0fdf4', lc.C_GPU_S,
                rx=3, sw=1.2)
        lc.text(SX0 + c * (SC + SS) + SC / 2, SY0 + r * (SC + SS) + SC / 2 + 4, svals[r][c],
                11, lc.C_TXT, 'middle', True, maxw=SC - 6, tag=f'sv{r}{c}')
lc.text(SX0 + SC + 1.5, SY0 + 2 * SC + SS + 16, 'S_ij = Q_0·K_1^T', 9, lc.C_MUTE, 'middle',
        maxw=140, tag='mp:s')
lc.text(SX0 + SC + 1.5, SY0 + 2 * SC + SS + 32, '(至多 2×2)', 8.5, lc.C_MUTE, 'middle',
        maxw=140, tag='mp:s2')
# 右侧步骤文字
TX = MP_X + 160
lines = [
    ('局部(片上一步算完,全是小账):m̃ = 块内行 max(行0 1.0 / 行1 2.0),P̃ = e^(S_ij−m̃),ℓ̃ = 行和 1.3679 / 1.3679', False),
    ('并入 running · 行0:max 未变 → 白折算 1.0:m 1.0→1.0,ℓ 1.3679→2.7358,O ← [3.5379, 4.5379]', False),
    ('并入 running · 行1:出新最大 → 旧账缩水:m 1.0→2.0,旧 O × 0.3679,ℓ 1.3679→1.8711,O ← [5.3864, 6.3864]', True),
    ('写回:折算-累加完的 O 除以 ℓ_new(归一化)——片上从头到尾只有 2×2 的草稿,从不铺开整表', False),
]
ly = MP_Y + 44
for txt, hot in lines:
    lc.text(TX, ly, txt, 9.5 if hot else 9, lc.C_BEAT_T if hot else '#334155', 'start', hot,
            maxw=MP_W - 160 - 20, tag='mp:l' + str(ly))
    ly += 36

# 图例行(显微镜下方)
lc.text(MP_X, MP_Y + MP_H + 18,
        '图例:绿实心块+白字徽标 = 当前步 ③ · 浅绿 = 已完成 ①② · 灰虚线 = 未到 ④ · 橙框(下图)= 发生非平凡折算的步',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MP_X, tag='leg')

# ---------------- 底部:4 步时间线 ----------------
lc.text(MX, 400, '4 步时间线(外层 j × 内层 i 各 2 步):每步片上打分块至多 2×2=4 元素;整表 4×4=16 元素从未存在,输出却步步是『至今为止的正确答案』',
        10.5, lc.C_TXT, 'start', True, maxw=BXR - MX, tag='tl:t')
STEPS = [
    ('步 ① (j=0, i=0)', False, [
        '行0/行1:首块初始化',
        'm:−∞→1.0 / −∞→1.0',
        '折算:— / —(旧账为 0)',
        'ℓ:1.3679 / 1.3679',
        'O:[1.5379,2.5379] / [2.4621,3.4621]']),
    ('步 ② (j=0, i=1)', False, [
        '行2/行3:首块初始化',
        'm:−∞→1.0 / −∞→2.0',
        '折算:— / —(旧账为 0)',
        'ℓ:2.0 / 1.1353',
        'O:[2.0,3.0] / [1.2384,2.2384]']),
    ('步 ③ (j=1, i=0)', True, [
        '行0 白折算;行1 出新最大',
        'm:1.0→1.0 / 1.0→2.0',
        '折算:1.0 / 0.3679(旧账缩水)',
        'ℓ:2.7358 / 1.8711',
        'O:[3.5379,4.5379] / [5.3864,6.3864]']),
    ('步 ④ (j=1, i=1)', True, [
        '行2 出新最大;行3 白折算',
        'm:1.0→2.0 / 2.0→2.0',
        '折算:0.3679(旧账缩水) / 1.0',
        'ℓ:2.7358 / 2.2707',
        'O:[4.9242,5.9242] / [3.2384,4.2384]']),
]
CH_Y, CH_H, CH_W = 416, 108, 330
for si, (title, hot, lines) in enumerate(STEPS):
    x0 = MX + si * (CH_W + 20)
    stroke = lc.C_ENG_S if hot else lc.C_MUTE
    lc.rect(x0, CH_Y, CH_W, CH_H, '#ffffff', stroke, rx=7, sw=hot and 1.8 or 1.1)
    lc.text(x0 + 12, CH_Y + 19, title, 10, stroke, 'start', True, maxw=CH_W - 24, tag='st' + title)
    for li, ln in enumerate(lines):
        lc.text(x0 + 12, CH_Y + 36 + li * 16, ln, 8.3, '#334155', 'start',
                maxw=CH_W - 20, tag=f'stl{si}{li}')
    if si < 3:
        lc.seg(x0 + CH_W + 2, CH_Y + CH_H / 2, x0 + CH_W + 18, CH_Y + CH_H / 2, lc.C_MUTE,
               1.6, 'std')

# 核验条
VY = CH_Y + CH_H + 16
lc.rect(MX, VY, BXR - MX, 58, '#ffffff', lc.C_GPU_S, rx=8, sw=1.3, dash=True)
lc.text(MX + 14, VY + 20, '逐块核验(实跑断言):每步之后,该行块的 O 恰为『只看已见 KV(前 kv_end 个)的朴素注意力输出』——4 步 max|差| 全部 0.0',
        9.5, lc.C_GPU_S, 'start', True, maxw=BXR - MX - 28, tag='v1')
lc.text(MX + 14, VY + 40, '终值:FA_O = standard_O 逐行相同(max|差|=0.0;float64 下仅剩 8.88e-16 级求和顺序差)——Theorem 1:任意合法分块,输出精确等于 softmax(QK^T)V',
        9, '#334155', 'start', maxw=BXR - MX - 28, tag='v2')

# ---------------- 页脚 ----------------
lc.text(MX, VY + 84, '算法出处 arXiv:2205.14135 §3.1 Algorithm 1:line 5/7 循环序(外层 KV 列块、内层 Q 行块)· line 9-13 块内五步(S_ij → 局部(m̃,P̃,ℓ̃) → running 更新 → O 折算累加)· line 1 块尺寸 Bc=⌈M/4d⌉',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, VY + 102, '数值取自论文忠实 NumPy 参考实现实跑(host,float64;softmax_scale 显式传 1.0 便于心算——vLLM 默认 1/√d,vllm/vllm_flash_attn/flash_attn_interface.py:L285-L286)· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch20-fig-fa-tiling.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
