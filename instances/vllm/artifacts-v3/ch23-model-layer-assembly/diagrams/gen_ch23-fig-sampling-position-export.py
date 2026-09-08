#!/usr/bin/env python3
"""ch23 机制图 4 · 采样位出口（figure_spec ch23-fig-sampling-position-export，模板 tensor-flow）

放大自本章 L2 章图拍片 ⑧ compute_logits（站 11/12）· L0：GPU 执行臂 × 模型层框。

claim: 8-token 批（r0=5 全 + r1=3 部分请求）只在 2 个采样位物化 logits——
hidden_states [8,64] 经 logits_indices=[4,7] 切成 [2,64]，lm_head GEMM 到
[2,128]（pad 后词表），裁 padding 成 [2,100] 交付；HF 式全位置物化是 1024 值 vs 200 值。

数字全部取自 figure_spec.numbers（8/64/5/3/[0,5,8]/[4,7]/2/128/100/1024/200/75，
均出自本章精简版实跑 trace）。坐标由常量/循环计算；文本全 esc()。
行号基线 vLLM v0.27.1。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1600, 780
MX = 56
BXR = 1544
R0_F, R0_S = '#dcfcc7', '#15803d'      # r0 = 完整 prompt 请求（绿系）
R1_F, R1_S = '#fef3c7', '#b45309'      # r1 = chunked prefill 部分请求（琥珀）

# ---------------- 标题区 ----------------
lc.text(MX, 34, '出口只在采样位物化：8 行 hidden 压成 2 行过 lm_head——1024 值的活只花 200 值的账',
        15.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '一拍批 = 8 token 两请求（r0 完整 prompt 5 token + r1 chunked prefill 部分请求本拍排进 3 token）：哪些位置判分写在 runner 的监考守则里，模型只提供 compute_logits 这个判分能力',
        10, lc.C_MUTE, 'start', maxw=1060, tag='subtitle')
_ch = '放大自 L2 拍片 ⑧ compute_logits（站 11/12）· L0：GPU 执行臂 × 模型层框'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# 迷你图例：请求色
LY = 86
lx = MX
for fill, stroke, lab in ((R0_F, R0_S, 'r0：完整 prompt，本拍 5 token（第 0-4 行）'),
                          (R1_F, R1_S, 'r1：chunked prefill 部分请求，本拍排进 3 token（第 5-7 行）')):
    lc.rect(lx, LY - 9, 18, 12, fill, stroke, rx=3, sw=1.4)
    lc.text(lx + 24, LY + 1, lab, 9, '#334155', 'start', tag='leg:' + lab[:4])
    lx += 24 + lc.tw(lab, 9) + 26

# ---------------- 共享几何：8 行矩阵字形 ----------------
ROW_H, G8_W = 17, 130


def matrix8(x0, y0):
    """8 行矩阵字形：前 5 行 r0 绿、后 3 行 r1 琥珀。返回各行 y 区间。"""
    rows = []
    for r in range(8):
        fill, stroke = (R0_F, R0_S) if r < 5 else (R1_F, R1_S)
        y = y0 + r * ROW_H
        lc.rect(x0, y, G8_W, ROW_H - 2, fill, stroke, rx=2, sw=1.0)
        rows.append((y, y + ROW_H - 2))
    return rows


# ---------------- 站 ① 前向产物 ----------------
GX1, GY1 = 150, 160
lc.text(120, 112, 'query_start_loc = [0, 5, 8]', 9.5, lc.C_TXT, 'start', True, tag='qsl')
lc.text(120, 130, '（批内请求前缀和）', 8.2, lc.C_MUTE, 'start', tag='qsl:sub')
lc.text(GX1 + G8_W / 2, 152, '① 前向产物', 10.5, lc.C_TXT, 'middle', True, tag='s1:t')
rows1 = matrix8(GX1, GY1)
lc.text(GX1 + G8_W / 2, GY1 + 8 * ROW_H + 20, 'hidden_states [8, 64]', 9.5, lc.C_TXT,
        'middle', True, tag='s1:s')
lc.text(GX1 + G8_W / 2, GY1 + 8 * ROW_H + 38, '全 8 个位置都算', 8.5, lc.C_MUTE, 'middle',
        tag='s1:n')
# 左侧请求括标
lc.seg(GX1 - 8, GY1, GX1 - 8, GY1 + 5 * ROW_H - 2, R0_S, 2.4)
lc.seg(GX1 - 4, GY1, GX1 - 8, GY1, R0_S, 1.4)
lc.seg(GX1 - 4, GY1 + 5 * ROW_H - 2, GX1 - 8, GY1 + 5 * ROW_H - 2, R0_S, 1.4)
lc.text(GX1 - 14, GY1 + 2.5 * ROW_H, 'r0', 9.5, R0_S, 'end', True, tag='b:r0')
lc.seg(GX1 - 8, GY1 + 5 * ROW_H, GX1 - 8, GY1 + 8 * ROW_H - 2, R1_S, 2.4)
lc.seg(GX1 - 4, GY1 + 5 * ROW_H, GX1 - 8, GY1 + 5 * ROW_H, R1_S, 1.4)
lc.text(GX1 - 14, GY1 + 6.5 * ROW_H, 'r1', 9.5, R1_S, 'end', True, tag='b:r1')
# 行号：由 ② 的「第 4 行 / 第 7 行」标注与 [4,7] 公式承载（① 侧不重复刻度，避让箭头标签）

# ---------------- 站 ② 采样位切片 ----------------
GX2 = 360
lc.text(GX2 + 170, 112, 'logits_indices = query_start_loc[1:] − 1 = [4, 7]', 9.5, lc.C_TXT,
        'middle', True, tag='s2:f')
lc.text(GX2 + G8_W / 2, 152, '② 采样位切片（策略归 runner）', 10.5, lc.C_TXT, 'middle', True,
        maxw=260, tag='s2:t')
rows2 = matrix8(GX2, GY1)
for r in (4, 7):
    y0_, y1_ = rows2[r]
    lc.rect(GX2 - 4, y0_ - 2, G8_W + 8, ROW_H + 2, 'none', lc.C_GPU_S, rx=3, sw=1.8, dash=True)
    lc.text(GX2 + G8_W + 14, y0_ + ROW_H / 2, f'第 {r} 行', 7.8, lc.C_GPU_S, 'start', True,
            tag='pick' + str(r))
# ② 结果 [2,64]
R2Y = 340
for i, (fill, stroke, lab) in enumerate(((R0_F, R0_S, '第 4 行（r0）'), (R1_F, R1_S, '第 7 行（r1）'))):
    y = R2Y + i * 19
    lc.rect(GX2, y, G8_W, 17, fill, stroke, rx=2, sw=1.4)
    lc.text(GX2 + G8_W / 2, y + 12, lab, 8.2, stroke, 'middle', True, tag='r2row' + str(i))
lc.seg(GX2 + G8_W / 2 - 30, GY1 + 8 * ROW_H + 4, GX2 + G8_W / 2 - 30, R2Y - 6, lc.C_GPU_S,
       1.8, 'std')
lc.text(GX2 + G8_W / 2 - 22, GY1 + 8 * ROW_H + 16, '只取 2 行', 8.2, lc.C_GPU_S, 'start', tag='s2:cut')
lc.text(GX2 + G8_W / 2, R2Y + 48, 'sample_hidden_states [2, 64]', 9.5, lc.C_TXT, 'middle', True,
        tag='s2:s')
lc.text(GX2 + G8_W / 2, R2Y + 66, '采样位数 = 请求数；r1 是部分请求也照采、', 8.2, lc.C_MUTE,
        'middle', maxw=280, tag='s2:n1')
lc.text(GX2 + G8_W / 2, R2Y + 82, '结果被忽略（woosuk NOTE）——不靠「不采」靠「采了丢弃」', 8.2,
        lc.C_MUTE, 'middle', maxw=300, tag='s2:n2')
# ① → ②
lc.seg(GX1 + G8_W + 34, GY1 + 4 * ROW_H, GX2 - 6, GY1 + 4 * ROW_H, lc.C_GPU_S, 2.0, 'std')
lc.text((GX1 + G8_W + GX2) / 2, GY1 + 4 * ROW_H - 8, 'hidden_states', 8.5, lc.C_GPU_S,
        'middle', tag='a12')

# ---------------- 站 ③ lm_head GEMM ----------------
WG_X, WG_Y, WG_W, WG_H = 580, 168, 36, 122     # 权重字形 [128,64]
lc.rect(WG_X, WG_Y, WG_W, WG_H, '#ffffff', lc.C_MUTE, rx=3, sw=1.4)
for i in range(16):
    lc.seg(WG_X, WG_Y + i * (WG_H / 16), WG_X + WG_W, WG_Y + i * (WG_H / 16), '#e2e8f0', 0.8)
lc.text(WG_X + WG_W + 10, WG_Y + 34, 'lm_head.weight', 9, lc.C_TXT, 'start', True, tag='wg:t')
lc.text(WG_X + WG_W + 10, WG_Y + 50, '[128, 64]', 8.5, '#334155', 'start', tag='wg:s')
lc.text(WG_X + WG_W + 10, WG_Y + 68, '词表 pad 到 64 的', 8, lc.C_MUTE, 'start', maxw=140,
        tag='wg:n1')
lc.text(WG_X + WG_W + 10, WG_Y + 82, '倍数 = 128', 8, lc.C_MUTE, 'start', maxw=140, tag='wg:n2')
GMX, GMY, GMW, GMH = 560, 312, 220, 84
lc.seg(WG_X + WG_W / 2, WG_Y + WG_H, GMX + GMW / 2, GMY, lc.C_MUTE, 1.6, 'std')
lc.rect(GMX, GMY, GMW, GMH, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.8)
lc.text(GMX + GMW / 2, GMY + 22, '③ lm_head GEMM', 10.5, lc.C_GPU_S, 'middle', True, tag='gm:t')
lc.text(GMX + GMW / 2, GMY + 42, 'h @ lm_head.weight.T', 9, '#334155', 'middle', tag='gm:l1')
lc.text(GMX + GMW / 2, GMY + 60, '一次 GEMM + 一次切片裁剪，', 8.2, lc.C_MUTE, 'middle',
        tag='gm:l2')
lc.text(GMX + GMW / 2, GMY + 74, '无第二遍全词表遍历', 8.2, lc.C_MUTE, 'middle', tag='gm:l3')
# ② → ③
lc.seg(GX2 + G8_W + 10, R2Y + 18, GMX - 6, R2Y + 18, lc.C_GPU_S, 2.0, 'std')
# ③ 结果 [2,128]（尾部 28 列 padding 灰）
R3X, R3Y, R3W = 830, 330, 260
for i, (fill, stroke) in enumerate(((R0_F, R0_S), (R1_F, R1_S))):
    y = R3Y + i * 19
    lc.rect(R3X, y, R3W * 100 / 128, 17, fill, stroke, rx=2, sw=1.4)
    lc.rect(R3X + R3W * 100 / 128, y, R3W * 28 / 128, 17, '#e2e8f0', '#cbd5e1', rx=2, sw=1.0)
lc.seg(GMX + GMW + 4, GMY + GMH / 2, R3X - 6, GMY + GMH / 2, lc.C_GPU_S, 2.0, 'std')
lc.text(R3X + R3W / 2, R3Y - 10, '[2, 128]', 9.5, lc.C_TXT, 'middle', True, tag='r3:s')
lc.text(R3X + R3W + 10, R3Y + 12, '尾部 28 列', 8, lc.C_MUTE, 'start', tag='r3:tail1')
lc.text(R3X + R3W + 10, R3Y + 26, '= 词表 padding', 8, lc.C_MUTE, 'start', tag='r3:tail2')

# ---------------- 站 ④ 裁 padding + argmax ----------------
lc.text(R3X + R3W / 2, R3Y + 56, '④ logits[..., :100] 裁掉 padding', 10.5, lc.C_TXT, 'middle',
        True, maxw=280, tag='s4:t')
R4X, R4Y, R4W = 1180, 330, 200
for i, (fill, stroke) in enumerate(((R0_F, R0_S), (R1_F, R1_S))):
    y = R4Y + i * 19
    lc.rect(R4X, y, R4W, 17, fill, stroke, rx=2, sw=1.4)
lc.seg(R3X + R3W + 14, R3Y + 18, R4X - 6, R4Y + 18, lc.C_GPU_S, 2.0, 'std', dash=True)
lc.text(R4X + R4W / 2, R4Y - 10, '交付 logits [2, 100]', 9.5, lc.C_TXT, 'middle', True,
        tag='r4:s')
# argmax
lc.seg(R4X + R4W / 2, R4Y + 38, R4X + R4W / 2, R4Y + 64, lc.C_GPU_S, 1.8, 'std')
AGX, AGY = R4X + 30, R4Y + 66
lc.rect(AGX, AGY, 140, 32, '#ffffff', lc.C_GPU_S, rx=6, sw=1.4)
lc.text(AGX + 70, AGY + 21, 'argmax', 9.5, lc.C_GPU_S, 'middle', True, tag='ag')
lc.text(R4X + R4W / 2, AGY + 54, 'r0 → 75（下一 token id）', 10, R0_S, 'middle', True,
        tag='out0')
lc.text(R4X + R4W / 2, AGY + 74, 'r1 行同样物化——', 8.2, R1_S, 'middle', tag='out1a')
lc.text(R4X + R4W / 2, AGY + 88, '属部分请求，被丢弃', 8.2, R1_S, 'middle', tag='out1b')
# ④ 后续：交给采样管线（预告 ch29）
lc.seg(R4X + R4W + 6, R4Y + 9, BXR - 60, R4Y + 9, lc.C_SAM_S, 2.0, 'std')
lc.text(R4X + R4W + 14, R4Y + 2, '→ ch29 采样管线（预告）', 8.5, lc.C_SAM_S, 'start', tag='next')

# ---------------- 底部：反事实条带 ----------------
CFY, CF_H = 560, 156
lc.rect(MX + 40, CFY, BXR - MX - 80, CF_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3, dash=True)
lc.text(MX + 60, CFY + 24, '反事实：若按 HF 式全位置物化（整卷都判分）', 10.5, lc.C_TXT,
        'start', True, maxw=420, tag='cf:t')
BAR_X, BAR_AH = MX + 60, 24
wA, wB = 620, 121        # 1024 : 200 等比（×0.605）
lc.rect(BAR_X, CFY + 44, wA, BAR_AH, '#fecaca', '#b91c1c', rx=4, sw=1.2)
lc.text(BAR_X + wA / 2, CFY + 60, 'HF 式全位置 [8,128] = 1024 值', 9, '#7f1d1d', 'middle',
        True, tag='cf:a')
lc.text(BAR_X + wA + 12, CFY + 60, 'org 口径 800 值', 8.5, lc.C_MUTE, 'start', tag='cf:a2')
lc.rect(BAR_X, CFY + 80, wB, BAR_AH, R0_F, R0_S, rx=4, sw=1.2)
lc.text(BAR_X + wB + 12, CFY + 96, '采样位 [2,100] = 200 值', 9, R0_S, 'start', True, tag='cf:b')
lc.text(BAR_X + wB + 150, CFY + 96, '——同 org 口径 4 倍差距，vocab 越大等比放大', 8.5,
        lc.C_MUTE, 'start', tag='cf:b2')
RSX = 900
lc.text(RSX, CFY + 44, '实尺放大（vocab=129280、fp32、4096-token prefill chunk）：', 9,
        lc.C_TXT, 'start', True, maxw=BXR - RSX - 40, tag='cf:r0')
lc.text(RSX, CFY + 66, '· 全位置物化 4096 × 129280 × 4B ≈ 2.0 GB 纯浪费（每 token 都算分布）',
        8.5, '#334155', 'start', maxw=BXR - RSX - 40, tag='cf:r1')
lc.text(RSX, CFY + 84, '· 采样位口径每请求仅 129280 × 4B ≈ 0.5 MB', 8.5, '#334155', 'start',
        maxw=BXR - RSX - 40, tag='cf:r2')
lc.text(RSX, CFY + 102, '· TP 下全位置物化还把 lm_head 词表分片的 gather 通信同倍放大', 8.5,
        '#334155', 'start', maxw=BXR - RSX - 40, tag='cf:r3')
lc.text(RSX, CFY + 124, '（本章 trace 以 tp=1 旁路经过 gather 分支；语义引源码注释）', 8,
        lc.C_MUTE, 'start', maxw=BXR - RSX - 40, tag='cf:r4')

# ---------------- 页脚 ----------------
lc.text(MX, 748, '读图：① 前向把 8 个位置全算成 hidden → ② runner 按前缀和只勾每请求最后一行（[4,7]）→ ③ 2 行过 lm_head GEMM（词表 pad 到 128）→ ④ 裁回 100 交 argmax——判分只看每人最后一道题。',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, 766, '行号基线 vLLM v0.27.1（gpu_model_runner.py:L2239/L4484-L4485 · logits_processor.py:L137-L153）· 数值取自本章精简版实跑核验（host CPU）',
        8, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch23-fig-sampling-position-export.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
