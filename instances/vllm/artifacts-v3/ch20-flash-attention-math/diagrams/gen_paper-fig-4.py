#!/usr/bin/env python3
"""ch20 论文精髓图 ④ · paper-fig-4(arXiv:2307.08691 Fig.4 忠实重绘)

writer figure-requests.json add:A100、head_dim=64、含因果掩码,前向+反向吞吐(TFLOPs/s)
随序列长度 512→16k:FA-2 相对 FA 约 1.7-3.0×、相对标准 PyTorch 实现最高 10×,绝对值最高
230 TFLOPs/s(73% 理论峰值)——「综合约 2×」的实测证据。

原图真相源(arXiv e-print 2307.08691 源码 figs/flash2_causal_True_hdim_64_fwd_bwd_speed.pdf,
即论文 Fig.4 的 (c) 子面板——含因果掩码、head dim 64;PDF 矢量+文字层逐柱提取,柱值与
原图印在柱顶的数字标签 30/30 全部对上):
- 标题 Attention forward + backward speed (A100 80GB SXM4);y 轴 Speed (TFLOPs/s)
  刻度 50/100/150/200;x 轴 Sequence length 512/1k/2k/4k/8k/16k;
- 组内柱序:PyTorch(蓝)/ FlashAttention(橙)/ xformers(绿)/ FlashAttention Triton(红)/
  FlashAttention-2(紫);
- 柱值:PyTorch 15/16/17/18/18/OOM · FlashAttention 58/70/77/87/92/97 · xformers
  51/60/66/68/69/80 · FA Triton 59/75/79/76/79/67 · FA-2 88/119/140/156/165/171;
- PyTorch@16k = OOM(标准实现在 16k 物化 N×N 直接爆显存);
- §4 口径:FA-2 比 FA 快 1.7-3.0×、比标准实现快 3-10×、最高 230 TFLOPs/s = 73% 理论峰值。

布局与信息结构对齐原图;配色/字体套本书视觉语言;文字译中。provenance=原论文图本身。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 720
MX = 42
BXR = 1458
SERIES = [
    ('PyTorch 标准实现', '#2563eb', '#dbeafe', [15, 16, 17, 18, 18, None]),
    ('FlashAttention', '#ea580c', '#ffedd5', [58, 70, 77, 87, 92, 97]),
    ('xformers(cutlass)', '#16a34a', '#dcfcc7', [51, 60, 66, 68, 69, 80]),
    ('FlashAttention Triton', '#dc2626', '#fee2e2', [59, 75, 79, 76, 79, 67]),
    ('FlashAttention-2', '#7c3aed', '#ede9fe', [88, 119, 140, 156, 165, 171]),
]
XLAB = ['512', '1k', '2k', '4k', '8k', '16k']

# ---------------- 标题区 ----------------
lc.text(MX, 32, '论文实测:FA-2 比 FA 快约 2×、比标准实现快 3-10×(A100,前向+反向吞吐)',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 56, '重绘自 arXiv:2307.08691 Fig.4(取含因果掩码、head dim 64 子面板,即原图 Fig.4c):柱值为原图逐柱数字(30 根全部核对);序列越长,标准实现越撑不住——16k 直接 OOM',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = 'primer · 论文精髓图重绘'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 图例(顶部一行) ----------------
LGY = 86
lx = MX
for name, cst, cfl, _ in SERIES:
    lc.rect(lx, LGY - 9, 16, 12, cfl, cst, rx=2, sw=1.2)
    lc.text(lx + 22, LGY, name, 9, lc.C_TXT, 'start', True, maxw=170, tag='lg:' + name[:8])
    lx += 22 + lc.tw(name, 9, True) + 26

# ---------------- 柱状区 ----------------
AX_X, AX_B = 100, 500                  # y 轴 x / 基线 y
AX_T = 130                             # 轴顶
AX_H = AX_B - AX_T                     # 370px ↔ 200 TFLOPs/s
YMAX = 200.0
GX0, GX1 = AX_X + 20, 1420             # 6 组分布区
GW = (GX1 - GX0) / 6                   # 组宽 216.7
BW = 34                                # 柱宽
BSTEP = 38                             # 柱距


def vy(v):
    return AX_B - AX_H * v / YMAX


lc.seg(AX_X, AX_T, AX_X, AX_B, lc.C_MUTE, 1.6)
lc.seg(AX_X, AX_B, GX1 + 16, AX_B, lc.C_MUTE, 1.6)
for tv in (50, 100, 150, 200):
    y = vy(tv)
    lc.seg(GX0 - 20, y, GX1 + 16, y, '#e2e8f0', 1.0)
    lc.seg(AX_X - 5, y, AX_X, y, lc.C_MUTE, 1.2)
    lc.text(AX_X - 10, y + 4, str(tv), 9, lc.C_MUTE, 'end', tag='ax:t%d' % tv)
lc.text(AX_X - 40, (AX_T + AX_B) / 2 - 6, '吞吐', 10.5, lc.C_TXT, 'middle', True, maxw=60,
        tag='ax:yl')
lc.text(AX_X - 40, (AX_T + AX_B) / 2 + 10, '(TFLOPs/s)', 8.5, lc.C_MUTE, 'middle', maxw=60,
        tag='ax:yl2')

oom_x = None
for gi in range(6):
    gx = GX0 + gi * GW + (GW - (5 * BSTEP - (BSTEP - BW))) / 2   # 组内 5 柱居中
    for si, (name, cst, cfl, vals) in enumerate(SERIES):
        v = vals[gi]
        bx = gx + si * BSTEP
        if v is None:                       # PyTorch@16k OOM
            lc.rect(bx, AX_B - 30, BW, 30, '#f8fafc', '#94a3b8', rx=2, sw=1.0, dash=True)
            lc.seg(bx + 5, AX_B - 25, bx + BW - 5, AX_B - 5, '#94a3b8', 1.0)
            lc.seg(bx + BW - 5, AX_B - 25, bx + 5, AX_B - 5, '#94a3b8', 1.0)
            lc.text(bx + BW / 2, AX_B - 38, 'OOM', 8.5, C_RED := '#dc2626', 'middle', True,
                    maxw=44, tag='bar:oom')
            oom_x = bx + BW / 2
            continue
        h = AX_H * v / YMAX
        lc.rect(bx, AX_B - h, BW, h, cfl, cst, rx=2, sw=1.2)
        lc.text(bx + BW / 2, AX_B - h - 9, str(v), 8.5, cst, 'middle', True, maxw=36,
                tag='bar:%d%d' % (si, gi))
    lc.text(gx + (5 * BSTEP - 4) / 2, AX_B + 18, XLAB[gi], 10, lc.C_TXT, 'middle', True,
            maxw=60, tag='xl:%d' % gi)
lc.text((GX0 + GX1) / 2, AX_B + 38, '序列长度(Sequence length)', 9.5, lc.C_MUTE, 'middle',
        maxw=200, tag='ax:xl')

# FA-2 vs FA 提速括注(16k 组上空)
g16 = GX0 + 5 * GW
fa2_16, fa_16 = 171, 97
bx_fa2 = g16 + (GW - (5 * BSTEP - 4)) / 2 + 4 * BSTEP + BW / 2
bx_fa = g16 + (GW - (5 * BSTEP - 4)) / 2 + 1 * BSTEP + BW / 2
yb = vy(fa2_16) - 26
lc.seg(bx_fa, yb + 10, bx_fa, yb, lc.C_MUTE, 1.2)
lc.seg(bx_fa2, yb + 10, bx_fa2, yb, lc.C_MUTE, 1.2)
lc.seg(bx_fa, yb, bx_fa2, yb, lc.C_MUTE, 1.2)
lc.text((bx_fa + bx_fa2) / 2, yb - 8, '16k:171 / 97 ≈ 1.8×', 9.5, lc.C_TXT, 'middle', True,
        maxw=170, tag='an:16k')

# ---------------- 底部三签 ----------------
SB_Y, SB_H = 548, 110
SBW = (BXR - MX - 2 * 20) / 3
chips = [
    ('① FA-2 vs FA:约 2×(论文口径 1.7-3.0×)', '#7c3aed', [
        '本面板逐组:88/58=1.5× → 171/97≈1.8×',
        '序列越长越接近 2×;',
        '论文 §4 综合(四种配置):1.7-3.0×',
    ]),
    ('② FA-2 vs 标准实现:3-10×', '#2563eb', [
        '本面板逐组:88/15=5.9× → 165/18≈9.2×',
        '16k 更直接:标准实现 OOM(物化 N×N)',
        '论文 §4:3-10× faster than standard',
    ]),
    ('③ 绝对吞吐:最高 230 TFLOPs/s', lc.C_GPU_S, [
        '230 = A100 理论峰值 312 的 73%(§4)',
        '端到端训练 225 TFLOPs/s(72% MFU)',
        '本面板最高柱:FA-2@16k = 171',
    ]),
]
for si, (title, color, lines) in enumerate(chips):
    x0 = MX + si * (SBW + 20)
    lc.rect(x0, SB_Y, SBW, SB_H, '#ffffff', color, rx=8, sw=1.4)
    lc.text(x0 + 12, SB_Y + 20, title, 10, color, 'start', True, maxw=SBW - 24,
            tag=f'chip{si}:t')
    for li, ln in enumerate(lines):
        lc.text(x0 + 12, SB_Y + 40 + li * 17, ln, 8.3, '#334155', 'start', maxw=SBW - 20,
                tag=f'chip{si}:l{li}')

# ---------------- 页脚:出处 ----------------
LY = SB_Y + SB_H + 22
lc.text(MX, LY, '重绘自 arXiv:2307.08691 Fig.4:Attention forward + backward speed on A100 GPU(80GB SXM4)· 本图取含因果掩码、head dim 64 的子面板(原图 Fig.4c)· 基准设置:hidden 2048、batch 按 16k token 补齐',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, LY + 18, '30 根柱值与 OOM 均逐字取自原图柱顶数字(PDF 矢量+文字层提取,30/30 核对)· 提速区间与 230=73% 为论文 §4 口径 · provenance = 论文原图',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'paper-fig-4.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
