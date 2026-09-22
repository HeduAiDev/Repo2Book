#!/usr/bin/env python3
"""ch36 机制图 · 每 token 稳态 KV 摊销三级阶梯（figure_spec fig_m17_amortized_ladder，模板 before-after）

放大自 L2 章图 south『why · 五本账一个池』注的量化展开——L0 kv_column 的
『每 token 多贵』一格。

claim：每 token 稳态 KV 摊销三级阶梯：Llama-70B GQA 327,680B → DSV4 只做 MLA 不压缩
25,696B → DSV4 压缩+fp8+fp4 indexer 3,584.25B（C4A 146.25×21 + C128 6.75×20 +
indexer 18×21）——每级砍一个量级、合计 91.4×。

数字全部取自 figure_spec.numbers（amortized_per_token 段 + 模型卡口径）。
坐标由常量/循环计算；文本全 esc()。
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1440, 756
MX, BXR = 56, 1384
DEFS = lc.DEFS

C_GQA_S, C_GQA_F = lc.C_ABORT, '#fee2e2'            # GQA 世代 = 红
C_MLA_S, C_MLA_F = lc.C_API_S, lc.C_API_F           # 不压缩 = 蓝
C_DSV4_S, C_DSV4_F = lc.C_GPU_S, lc.C_GPU_F         # 压缩档 = 绿

# ---------------- 标题区 ----------------
lc.text(MX, 34, '三级行李费：每 token 稳态 KV 从 320KiB 砍到 3.5KiB——1M 能活的物质基础', 16.5,
        lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '每 token 稳态摊销 = Σ(页 ÷ 256 × 层数)，与 max_model_len 无关 · 窗口/状态账另有界（窗口+在途 chunk 封顶），全长账单里只有这一项线性涨',
        10.5, lc.C_MUTE, 'start', maxw=1300, tag='subtitle')
_ch = '放大自 L2 south『why · 五本账一个池』· L0 显存账本列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 左：对数阶梯 =================
LP_W = 768
lc.rect(MX, 100, LP_W, 420, '#ffffff', lc.C_MUTE, rx=10, sw=1.6)
lc.text(MX + 18, 126, '每 token 稳态 KV（对数刻度）', 12.5, lc.C_TXT, 'start', True, maxw=400, tag='lp:t')

BARS = [
    (C_GQA_S, C_GQA_F, 'Llama-70B GQA', '2(K,V) × 8 kv 头 × 128 维 × 2B × 80 层', 327680, '327,680B = 320KiB'),
    (C_MLA_S, C_MLA_F, 'DSV4 只做 MLA、不压缩', '584B × 44 本账', 25696, '25,696B ≈ 25.1KiB'),
    (C_DSV4_S, C_DSV4_F, 'DSV4 压缩 + fp8 + fp4 检索', 'C4A + C128 + indexer 稳态摊销', 3584.25, '3,584.25B ≈ 3.5KiB'),
]
BAR_X0 = MX + 28
LOG_K = 272            # px / 10 倍
LOG_OFF = 3.30         # log10 下限


def blog_px(v):
    return (math.log10(v) - LOG_OFF) * LOG_K


# 刻度网格（10^4 / 10^5）
for gv, gl in ((10 ** 4, '10⁴ B'), (10 ** 5, '10⁵ B')):
    gx = BAR_X0 + blog_px(gv)
    lc.seg(gx, 150, gx, 490, '#eef2f7', 1.0)
    lc.text(gx, 502, gl, 8, lc.C_FAINT, 'middle', maxw=60, tag='grid' + gl[:3])

RY0, RBAR_H, RPITCH = 172, 40, 118
for i, (s, f, name, sub, val, vlab) in enumerate(BARS):
    ry = RY0 + i * RPITCH
    w_ = blog_px(val)
    lc.text(BAR_X0, ry - 12, name, 10.5, s, 'start', True, maxw=330, tag=f'b{i}n')
    lc.text(BAR_X0 + 336, ry - 12, sub, 8.4, '#475569', 'start', maxw=330, tag=f'b{i}s')
    lc.rect(BAR_X0, ry, w_, RBAR_H, f, s, rx=4, sw=1.5)
    if w_ >= 150:
        lc.text(BAR_X0 + w_ / 2, ry + 25, vlab, 9.6, s, 'middle', True, maxw=w_ - 12, tag=f'b{i}v')
    else:
        lc.text(BAR_X0 + w_ + 10, ry + 25, vlab, 9.6, s, 'start', True, maxw=170, tag=f'b{i}v')

# 倍率标注（右侧双折线；括号线不带箭头——端点不落框边，带 marker 会被几何门禁判悬空）
MRK_X = MX + LP_W - 100
b_ends = [RY0 + i * RPITCH + RBAR_H / 2 for i in range(3)]
lc.seg(MRK_X, b_ends[0], MRK_X, b_ends[2], C_GQA_S, 1.6, dash=True)
lc.text(MRK_X + 8, (b_ends[0] + b_ends[2]) / 2 - 12, '91.4×', 11, C_GQA_S, 'start', True, maxw=60, tag='mr1')
lc.seg(MRK_X + 64, b_ends[1], MRK_X + 64, b_ends[2], C_MLA_S, 1.6, dash=True)
lc.text(MRK_X + 58, (b_ends[1] + b_ends[2]) / 2 + 4, '7.2×', 10.5, C_MLA_S, 'end', True, maxw=60, tag='mr2')
lc.text(MX + 18, 490, '每级各砍约一个量级；两级合计 91.4×（对照 GQA 世代）', 9, lc.C_MUTE, 'start',
        maxw=LP_W - 130, tag='lp:n')

# ================= 右：3.5KiB 的构成 =================
RP_X = MX + LP_W + 22
RP_W = BXR - RP_X
lc.rect(RP_X, 100, RP_W, 420, '#ffffff', lc.C_MUTE, rx=10, sw=1.6)
lc.text(RP_X + 18, 126, '3,584.25B 的构成（压缩档稳态斜率 × 层数）', 12, lc.C_TXT, 'start', True,
        maxw=RP_W - 36, tag='rp:t')

COMP = [
    (lc.C_ZMQ_S, lc.C_ZMQ_F, 'C4A 主账', '146.25B × 21 层', 3071.25),
    (lc.C_ENG_S, lc.C_ENG_F, 'indexer fp4', '18B × 21 层', 378),
    (lc.C_KV_S, lc.C_KV_F, 'C128 主账', '6.75B × 20 层', 135),
]
TOT = 3584.25
SB_X0, SB_W, SB_Y, SB_H = RP_X + 18, RP_W - 36 - 96, 168, 46
bx = SB_X0
for s, f, name, sub, v in COMP:
    w_ = v / TOT * SB_W
    lc.rect(bx, SB_Y, w_, SB_H, f, s, rx=3, sw=1.4)
    if w_ >= 80:
        lc.text(bx + w_ / 2, SB_Y + 21, name, 9.4, s, 'middle', True, maxw=w_ - 8, tag='cp:' + name[:4])
        lc.text(bx + w_ / 2, SB_Y + 38, f'{v:,.2f}B'.replace('.00B', 'B'), 8.2, s, 'middle', maxw=w_ - 8, tag='cpv:' + name[:4])
    bx += w_
lc.text(SB_X0 + SB_W + 10, SB_Y + 21, '合计', 9.4, lc.C_TXT, 'start', True, maxw=60, tag='sb:t')
lc.text(SB_X0 + SB_W + 10, SB_Y + 38, '3,584.25B', 8.6, lc.C_TXT, 'start', True, maxw=86, tag='sb:v')

# 构成明细行（窄段 indexer/C128 的数值列出）
cy = SB_Y + 74
for s, f, name, sub, v in COMP:
    lc.rect(RP_X + 18, cy, 14, 11, f, s, rx=3, sw=1.2)
    lc.text(RP_X + 38, cy + 10, f'{name}：{sub} = {v:,.2f}B'.replace('.00B', 'B'), 9,
            '#334155', 'start', maxw=RP_W - 56, tag='cd:' + name[:4])
    cy += 22

# 换算注记
NOTE_Y = cy + 8
lc.rect(RP_X + 18, NOTE_Y, RP_W - 36, 108, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.3)
lc.text(RP_X + 32, NOTE_Y + 22, '这就是『1M 能活』的物质基础', 10.5, lc.C_GPU_S, 'start', True,
        maxw=RP_W - 60, tag='nt:t')
for i, ln in enumerate(['· 1M token 单请求：主账 + indexer ≈ 3.50GiB',
                        '· fp8 indexer 档则合计 3,915B（每 token 摊）',
                        '· 官方口径：1M 上下文 KV 占用 ≈ V3.2 的 10%（HF 模型卡）']):
    lc.text(RP_X + 32, NOTE_Y + 44 + i * 19, ln, 9, '#334155', 'start', maxw=RP_W - 60, tag=f'nt:{i}')

# ---------------- 底部结论带 ----------------
BB_Y = 544
lc.rect(MX, BB_Y, BXR - MX, 88, '#f8fafc', lc.C_MUTE, rx=10, sw=1.5)
lc.text(MX + 18, BB_Y + 24, '窗口账 / 状态账不进这张表：按请求有界（窗口 + 在途 chunk 封顶），不随全长线性涨', 11,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 36, tag='bb:t')
lc.text(MX + 18, BB_Y + 48, '· 全长线性项只有主账与 indexer 账（斜率 146.25 / 6.75 / 18 B/token）——1M 的账单里，这一项 ≈ 3.5GiB',
        9.2, '#334155', 'start', maxw=BXR - MX - 36, tag='bb:l1')
lc.text(MX + 18, BB_Y + 68, '· 对照公式为通识算术：GQA 每层 2(K,V) × kv_heads × head_dim × 2B', 9.2,
        '#334155', 'start', maxw=BXR - MX - 36, tag='bb:l2')

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 44, '斜率 ＝ 页 ÷ 256 token（kv_cache_interface.py:L403-L426 的页/token 公式）· 三级数值 / 倍率 ＝ 本章账本算术脚本实算（pin spec 类）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 29, '官方 10% 口径 ＝ HF 模型卡 deepseek-ai/DeepSeek-V4-Flash（2026-09-22 取回）· 对照公式（Llama-70B GQA）为通识算术',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')
lc.text(MX, H - 14, '行号基线 vLLM v0.27.1（6e448d0ea）', 8.5, lc.C_FAINT, 'start', maxw=400, tag='foot3')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m17_amortized_ladder.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
