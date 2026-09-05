#!/usr/bin/env python3
"""ch27 机制图 2 · 粒度谱:谁跟谁共享一把尺子(figure_spec ch27-fig-granularity-rulers,模板 layout)

放大自 L0 GPU 执行臂(绿)第三块『模型层 forward + 编译』里 Linear 层的输入/权重——
GroupShape 的粒度词汇就是给这三把尺子起的正式名字。推导链第 2 环,不画架构元素。

claim:粒度=共享尺子的范围。本例 token 0 比其他 token 大 ~100 倍,per-tensor 一把尺下
小 token 的码全坍缩(独立码 3、误差 0.3358);per-token 各拿各的尺(独立码 11、误差
0.0013,好 254 倍);但激活 per-channel 挂在 GEMM 缩减维,INT8 Tensor Core 不认
(SmoothQuant Table 1:71.6→32.3→31.7→71.4,仅 per-channel 保住精度)。

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

lc.text(MX, 34, '粒度 = 尺子挂在哪根轴上:谁跟谁共享一把 Δ',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, 'token 0 比其他 token 大 ~100 倍:per-tensor 一把尺 → 小 token 的码全坍缩;per-token 各拿各的尺 → 好 254 倍;per-channel 精度更好却挂在 GEMM 缩减维,Tensor Core 不收',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L0 GPU 执行臂 · 模型层 forward + 编译 · Linear 层输入/权重'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 图例条(三矩阵共用) =================
LGX = MX
LEGY = 98
LEG = [
    ('#fed7aa', lc.C_ENG_S, 'token 0(absmax 130.3,离群)'),
    ('#e8edf3', '#cbd5e1', '其余 token(absmax ~0.9)'),
    (None, lc.C_GPU_S, '绿框 = 一把 Δ 的共享范围'),
]
for fill, stroke, name in LEG:
    if fill:
        lc.rect(LGX, LEGY - 8, 18, 12, fill, stroke, rx=3, sw=1.0)
    else:
        lc.rect(LGX, LEGY - 8, 18, 12, 'none', stroke, rx=4, sw=2.0)
    lc.text(LGX + 24, LEGY + 1, name, 9, '#334155', 'start', tag='leg:' + name[:8])
    LGX += 24 + lc.tw(name, 9) + 26

# ================= 三张同构矩阵 =================
CW_, CH_ = 56, 42          # 单元格
GAP = 3
MAT_W = 4 * CW_ + 3 * GAP
PANEL_W = 420
PANELS_X = [MX, MX + PANEL_W + 30, MX + 2 * (PANEL_W + 30)]
PY, PH = 120, 360
TITLES = [
    ('per-tensor', '全场一把尺(最省、最粗)'),
    ('per-token', '每 token 一把(尺挂外维 T)'),
    ('per-channel', '每输入通道一把(尺挂内维 C_i)'),
]
DELTAS = [
    None,
    ['1.0261', '0.0071', '0.0043', '0.0058'],
    ['0.2721', '0.6469', '0.2602', '1.0261'],
]
RESULTS = [
    ('小 token 平均|误差| 0.3358 · 独立码 3', '码全坍缩', lc.C_ABORT),
    ('小 token 平均|误差| 0.0013 · 独立码 11', '好 254.2 倍', lc.C_GPU_S),
    ('平均|误差| 0.1962 · 独立码 5', 'INT8 Tensor Core 不认', lc.C_ENG_S),
]
MAT_Y = PY + 104           # 矩阵左上角

for p in range(3):
    px = PANELS_X[p]
    gray = (p == 2)
    lc.rect(px, PY, PANEL_W, PH, '#f8fafc' if gray else '#ffffff',
            lc.C_FAINT if gray else GRID, rx=8, sw=1.2, dash=gray)
    if gray:
        bw = lc.tw('GEMM 内维 ✗', 9, True) + 12
        lc.rect(px + PANEL_W - bw - 8, PY + 6, bw, 18, '#fff7ed', lc.C_ENG_S, rx=8, sw=1.1)
        lc.text(px + PANEL_W - bw / 2 - 8, PY + 19, 'GEMM 内维 ✗', 9, lc.C_ENG_S, 'middle',
                True, tag='pbadge' + str(p))
    lc.text(px + 14, PY + 24, TITLES[p][0], 12, lc.C_TXT, 'start', True, tag='pt' + str(p))
    lc.text(px + 14, PY + 42, TITLES[p][1], 9, lc.C_MUTE, 'start', maxw=PANEL_W - 28,
            tag='ps' + str(p))
    # 轴标注 + 通道/ token 标签(仅第一张画,三张同构)
    if p == 0:
        lc.text(px + 14, MAT_Y - 44, 'token T = 外维(可行 ↓)', 8.5, lc.C_MUTE, 'start',
                maxw=MAT_W, tag='ax:t')
        lc.text(px + 14, MAT_Y - 28, '输入通道 C_i = GEMM 缩减维(内维 →)', 8.5, lc.C_MUTE,
                'start', maxw=MAT_W + 20, tag='ax:ci')
        for c in range(4):
            cx = px + 14 + c * (CW_ + GAP) + CW_ / 2
            lc.text(cx, MAT_Y - 8, f'c{c}', 8.5, lc.C_MUTE, 'middle', tag='ch' + str(c))
        for r in range(4):
            ry = MAT_Y + r * (CH_ + GAP)
            lc.text(px + 6, ry + CH_ / 2 + 3, f't{r}', 8.5, lc.C_MUTE, 'end', tag='tk' + str(r))
    # 16 格:token 0 离群(橙),其余普通(浅)
    for r in range(4):
        for c in range(4):
            x = px + 14 + c * (CW_ + GAP)
            y = MAT_Y + r * (CH_ + GAP)
            hot = (r == 0)
            fill = '#fed7aa' if hot else '#e8edf3'
            stroke = lc.C_ENG_S if hot else '#cbd5e1'
            lc.rect(x, y, CW_, CH_, fill, stroke, rx=3, sw=1.0)
    # 「谁共享一把 Δ」的绿框:per-tensor 一个大框 / per-token 每行 / per-channel 每列
    fx0 = px + 14 - 4
    if p == 0:
        lc.rect(fx0, MAT_Y - 4, MAT_W + 8, 4 * CH_ + 3 * GAP + 8, 'none', lc.C_GPU_S, rx=6, sw=2.2)
    elif p == 1:
        for r in range(4):
            lc.rect(fx0, MAT_Y + r * (CH_ + GAP) - 4, MAT_W + 8, CH_ + 8, 'none', lc.C_GPU_S,
                    rx=6, sw=2.2)
    else:
        for c in range(4):
            lc.rect(px + 14 + c * (CW_ + GAP) - 4, MAT_Y - 4, CW_ + 8,
                    4 * CH_ + 3 * GAP + 8, 'none', lc.C_GPU_S, rx=6, sw=2.2)
    # Δ 列表
    if DELTAS[p]:
        lc.text(px + PANEL_W / 2, MAT_Y + 4 * (CH_ + GAP) + 18,
                'Δ: ' + ' | '.join(DELTAS[p]), 8.5, lc.C_GPU_S, 'middle',
                tag='dl' + str(p))
    else:
        lc.text(px + PANEL_W / 2, MAT_Y + 4 * (CH_ + GAP) + 18, 'Δ = 1.0261(全矩阵 absmax/127)',
                8.5, lc.C_GPU_S, 'middle', tag='dl' + str(p))
    # 结果两行
    ry_ = MAT_Y + 4 * (CH_ + GAP) + 44
    res, verdict, vc = RESULTS[p]
    lc.text(px + PANEL_W / 2, ry_, res, 9.5, '#334155', 'middle', tag='res' + str(p))
    vw = lc.tw(verdict, 9.5, True) + 14
    lc.rect(px + (PANEL_W - vw) / 2, ry_ + 8, vw, 18,
            '#fef2f2' if vc == lc.C_ABORT else ('#f0fdf4' if vc == lc.C_GPU_S else '#fff7ed'),
            vc, rx=8, sw=1.2)
    lc.text(px + PANEL_W / 2, ry_ + 21, verdict, 9.5, vc, 'middle', True, tag='vd' + str(p))

# ================= 底部左:Table 1 条形 =================
BY, BH = 496, 274
lc.rect(MX, BY, 700, BH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(MX + 16, BY + 24, '真实口径:OPT-175B 平均准确率(SmoothQuant §3 Table 1,INT8 静态量化)',
        11, lc.C_TXT, 'start', True, maxw=660, tag='b1:h')
BARS = [
    ('FP16', 71.6, lc.C_API_S, False),
    ('per-tensor', 32.3, lc.C_ABORT, False),
    ('per-token', 31.7, lc.C_ABORT, False),
    ('per-channel', 71.4, lc.C_FAINT, True),
]
BASE_Y = BY + BH - 64
BAR_W, BAR_GAP = 96, 46
bx0 = MX + 60
MAXV = 80
for i, (name, v, color, grayed) in enumerate(BARS):
    x = bx0 + i * (BAR_W + BAR_GAP)
    hgt = (v / MAXV) * 150
    lc.rect(x, BASE_Y - hgt, BAR_W, hgt, '#eff6ff' if not grayed else '#f1f5f9', color,
            rx=4, sw=1.4, dash=grayed)
    lc.text(x + BAR_W / 2, BASE_Y - hgt - 8, f'{v}', 10, color, 'middle', True, tag='bar' + str(i))
    lc.text(x + BAR_W / 2, BASE_Y + 16, name, 8.5, lc.C_MUTE, 'middle', tag='barn' + str(i))
# 基线
lc.rect(bx0 - 10, BASE_Y, 4 * BAR_W + 3 * BAR_GAP + 20, 2, '#334155', '#334155', rx=1, sw=0)
lc.text(MX + 16, BY + 60, '%', 9, lc.C_MUTE, 'start', tag='pct')
lc.text(MX + 16, BASE_Y + 44, 'per-token/per-tensor 崩(31.7/32.3);per-channel 保精度(71.4)但挂内维、INT8 GEMM 不可行 → 可行集 = per-tensor / per-token / 分组',
        8.5, lc.C_MUTE, 'start', maxw=660, tag='b1:note')

# ================= 底部右:vLLM 词汇 + 存储账 =================
lc.rect(790, BY, 650, BH, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(806, BY + 24, 'vLLM 的一等词汇:GroupShape(quant_utils.py:L69-L71)', 11, lc.C_TXT,
        'start', True, maxw=600, tag='b2:h')
VOCAB = [
    ('GroupShape.PER_TENSOR', '整张矩阵一把(本图左)'),
    ('GroupShape.PER_TOKEN', '外维 token 一把 → QuantFP8 dynamic per-token 即此档'),
    ('GroupShape.PER_CHANNEL', '外维输出通道一把 → 权重侧标准做法'),
    ('(128,128) 块 / (1,128) 逐 token 逐组', 'DeepSeek 式分组量化(L356-L357 注释)'),
]
vy = BY + 46
for name, desc in VOCAB:
    lc.text(806, vy, name, 9.5, lc.C_GPU_S, 'start', True, maxw=280, tag='v' + name[:10])
    lc.text(1090, vy, desc, 9, '#334155', 'start', maxw=340, tag='vd' + name[:10])
    vy += 26
lc.text(806, vy + 6, '分组的存储代价(group-size 128 ≈ 每 权重 0.15 额外 bit;g1024≈0.02,GPTQ §5)', 9,
        lc.C_MUTE, 'start', maxw=620, tag='b2:store')
lc.text(806, vy + 26, '把「想要 per-channel 精度」的欲望引向了 SmoothQuant 的搬家(→本章 §6)', 9,
        lc.C_MUTE, 'start', maxw=620, tag='b2:lead')

# ================= 页脚 =================
FY = 790
lc.text(MX, FY, '数字:本章 NumPy 参考实现实跑(4×4,固定种子) · 论文口径 arXiv:2211.10438 §3 Table 1 · 词汇 vllm/model_executor/layers/quantization/utils/quant_utils.py:L69-L71 · L356-L357 · 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=1380, tag='foot')

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch27-fig-granularity-rulers.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
