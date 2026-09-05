#!/usr/bin/env python3
"""ch20 机制图 ② · online-softmax 单遍递推(figure_spec ch20-fig-online-softmax-recurrence,模板 state-table)

放大自 L0 中列『GPU 执行臂』(绿色列)第三块『模型层 forward + 编译』内 attention kernel 的
数学内部——kernel 每次只看得见一小块分数,却要算整行 softmax 的分母:地基就是这张单遍递推表。
primer 推导链第 ② 环;架构归属回指 L0(FIGURE-SYSTEM §3.3)。

claim:running (m_j, d_j) 单遍递推——每来一个新元素,旧账先按 e^(旧max−新max) 折算到新基准
再累加新项;末值与三遍 safe softmax 恒等(Theorem 1),5 元素实例可整表心算核验。

数字全部取自 figure_spec.numbers(递推全表/三版末值/溢出对照/遍数账:host NumPy 参考实现实跑;
vLLM 照应 triton_merge_attn_states.py:L278-L284)。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 838
MX = 60
BXR = 1440
C_RED = '#dc2626'

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'online softmax:一本账单遍扫完——每来一个新最大,旧账先折算、再记新账',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'x = [1, 3, 2, 5, 4](5 个一位数,可整表心算):naive 两遍会溢出(e^1000=inf)、safe 减 max 要三遍;online 把『找最大』与『求和』融成一遍(arXiv:1805.02867 §3 Alg.3)',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = '推导链 ② · 放大自 L0 GPU 执行臂内 attention kernel 的数学内部'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 主表:5 轮递推 ----------------
COLS = [
    ('轮 j', 52),
    ('来访 x_j', 74),
    ('旧 max m_(j−1)', 128),
    ('新 max m_j', 104),
    ('折算 e^(m旧−m新)', 216),
    ('旧账折算后', 132),
    ('新项 e^(x_j−m_j)', 128),
    ('新账 d_j', 96),
    ('界 1≤d_j≤j', 164),
]
TX0 = (W - sum(w for _, w in COLS)) / 2
HDR_Y, HDR_H, ROW_H = 100, 40, 62
TAB_H = HDR_H + 5 * ROW_H

xs = []
cx = TX0
for _, w in COLS:
    xs.append((cx, w))
    cx += w
TAB_W = cx - TX0

# 表底框
lc.rect(TX0, HDR_Y, TAB_W, TAB_H, '#ffffff', lc.C_MUTE, rx=6, sw=1.4)
lc.rect(TX0, HDR_Y, TAB_W, HDR_H, '#f1f5f9', lc.C_MUTE, rx=6, sw=1.2)
for i in range(1, len(COLS)):
    x = xs[i][0]
    lc.seg(x, HDR_Y, x, HDR_Y + TAB_H, '#e2e8f0', 1.0)
# 列头
for (x, w), (name, _) in zip(xs, COLS):
    lc.text(x + w / 2, HDR_Y + 24, name, 9.5, lc.C_MUTE, 'middle', True, maxw=w - 8,
            tag='hd:' + name)

ROWS = [
    ('1', '1', '−∞(初始 m_0)', '1.0', ('—(首步无旧账)', 'd_0 = 0,旧账为空'), '0', '1.0', '1.0',
     '✓ 1≤1.0≤1', False),
    ('2', '3', '1.0', '3.0', ('0.1353', '旧账缩水到 13.53%'), '0.1353', '1.0', '1.1353',
     '✓ 1≤1.1353≤2', True),
    ('3', '2', '3.0', '3.0', ('1.0', '白折算:max 未变'), '1.1353', '0.3679', '1.5032',
     '✓ 1≤1.5032≤3', False),
    ('4', '5', '3.0', '5.0', ('0.1353', '又一次旧账缩水(同为 e^(−2))'), '0.2034', '1.0',
     '1.2034', '✓ 1≤1.2034≤4', True),
    ('5', '4', '5.0', '5.0', ('1.0', '白折算'), '1.2034', '0.3679', '1.5713',
     '✓ 1≤1.5713≤5', False),
]
for ri, row in enumerate(ROWS):
    ry = HDR_Y + HDR_H + ri * ROW_H
    (j, xj, mold, mnew, (rescale, rnote), oldacc, newterm, dj, bound, hot) = row
    if hot:
        lc.rect(TX0 + 1, ry + 1, TAB_W - 2, ROW_H - 2, lc.C_BEAT_F, 'none', rx=0, sw=0)
    if ri > 0:
        lc.seg(TX0, ry, TX0 + TAB_W, ry, '#e2e8f0', 1.0)
    cy = ry + ROW_H / 2
    vals = [j, xj, mold, mnew, None, oldacc, newterm, dj, bound]
    for ci, v in enumerate(vals):
        x, w = xs[ci]
        if v is None:
            top_fs, top_bold = (11 if hot else 10), True
            lc.text(x + w / 2, cy - 4, rescale, top_fs, lc.C_BEAT_T if hot else lc.C_TXT,
                    'middle', top_bold, maxw=w - 8, tag=f'r{ri}c{ci}')
            lc.text(x + w / 2, cy + 14, rnote, 8, lc.C_BEAT_T if hot else lc.C_MUTE,
                    'middle', maxw=w - 8, tag=f'r{ri}c{ci}n')
        else:
            fs = 10.5 if ci in (7,) else 10
            bold = ci in (1, 7)
            fill = lc.C_GPU_S if ci == 8 else lc.C_TXT
            lc.text(x + w / 2, cy + 3, v, fs, fill, 'middle', bold, maxw=w - 8,
                    tag=f'r{ri}c{ci}')

# 递推式条
FY = HDR_Y + TAB_H + 24
lc.text(W / 2, FY, '递推:d_j ← d_(j−1) · e^(m_(j−1)−m_j) + e^(x_j−m_j) —— 旧账每一项恰好折算到新基准,m_j 单调不减、循环有限步终止',
        10.5, lc.C_TXT, 'middle', True, maxw=BXR - MX, tag='formula')

# ---------------- 三条对照带 ----------------
SB_Y, SB_H = FY + 18, 148
SBW = (BXR - MX - 2 * 20) / 3
strips = [
    ('① naive 两遍:直接 e^x 会溢出', C_RED, [
        'x = [1000, 1001] 时 e^1000 = inf(float64)',
        'inf / inf → naive 得 [nan, nan]',
        'safe / online 先减 max = 1001:',
        '最大项 e^0 = 1,不再上溢',
        '→ [0.2689, 0.7311]',
    ]),
    ('② 三版末值恒等(Theorem 1)', lc.C_GPU_S, [
        '末值 (m_V, d_V) = (5.0, 1.5713)',
        'naive = safe = online',
        '  = [0.0117, 0.0861, 0.0317, 0.6364, 0.2341]',
        'max|差| = 0.0(逐位相同)',
        'online 末值恒等于三遍 safe softmax',
    ]),
    ('③ 访存账:一遍的价值', lc.C_ENG_S, [
        'safe:统计量 3 遍 / 每元素 4 次访存',
        'online:统计量 1 遍 / 每元素 3 次',
        '(论文原话:4 down to 3)',
        'decode 一行 8192 个分数:',
        '32768 → 24576 次元素访问(省 25%)',
    ]),
]
for si, (title, color, lines) in enumerate(strips):
    x0 = MX + si * (SBW + 20)
    lc.rect(x0, SB_Y, SBW, SB_H, '#ffffff', color, rx=8, sw=1.4)
    lc.text(x0 + 12, SB_Y + 20, title, 10.5, color, 'start', True, maxw=SBW - 24,
            tag=f'strip{si}:t')
    for li, ln in enumerate(lines):
        emph = ln.startswith('→') or ln.startswith('  =') or 'max|差|' in ln
        lc.text(x0 + 12, SB_Y + 40 + li * 19, ln, 8.7,
                color if emph else '#334155', 'start', emph, maxw=SBW - 20,
                tag=f'strip{si}:l{li}')

# ---------------- vLLM 照应注 ----------------
VY = SB_Y + SB_H + 16
lc.rect(MX, VY, BXR - MX, 54, '#ffffff', lc.C_GPU_S, rx=8, sw=1.3, dash=True)
lc.text(MX + 14, VY + 20, 'vLLM 落地照应:merge kernel 的 max_lse 稳定化(vllm/v1/attention/ops/triton_merge_attn_states.py:L278-L284)是同一数学在 log 域的化身',
        9.5, lc.C_GPU_S, 'start', True, maxw=BXR - MX - 28, tag='vllm:t')
lc.text(MX + 14, VY + 38, '『记账时不知道全局最大,每来一个新最大就把旧账全部折算』——这一页数学就是 FlashAttention 分块 softmax 的地基',
        9, '#334155', 'start', maxw=BXR - MX - 28, tag='vllm:l')

# ---------------- 页脚:图例 + 出处 ----------------
LY = VY + 76
lc.text(MX, LY, '图例:橙底行 = 非平凡折算事件(出现新最大,0.1353 = e^(−2)) · 绿字 = 核验通过 · 红字 = 溢出对照',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, LY + 20, '递推与定理出处 arXiv:1805.02867 §2-§3(Alg.3 + Theorem 1)· 界 1≤d_j≤j:每项 e^(x_k−m_j)≤1 且最大项=1,保证 32 位浮点能处理 1.7×10^37 个元素不溢出(论文 §3 数值界)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, LY + 38, '数值取自论文忠实 NumPy 参考实现实跑(host,float64)· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch20-fig-online-softmax-recurrence.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
