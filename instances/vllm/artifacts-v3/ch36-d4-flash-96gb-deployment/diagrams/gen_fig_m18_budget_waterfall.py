#!/usr/bin/env python3
"""ch36 机制图 · 96GiB 预算瀑布与四个阀门（figure_spec fig_m18_budget_waterfall，模板 layout）

放大自 L2 章图 north『出 · 启动日志（部署者读的三行）』+ 拍片⑥⑧（测量三件套 / packed
定账）——L0 启动视角显存账整格的实战兑现图。

claim：96GiB 预算一条瀑布：96GiB×0.92=88.32GiB → −权重（TP 档定：TP2≈78.7 est）→
−激活峰（profile 实测）→ −图池（估计）→ 池（≈5GiB）→ ÷917,568 → 5,851 块 →
×每请求峰值预留 → 并发行；逃生门按杠杆排序：max_model_len/chunk（分母 −69.4%）→
indexer fp4（+9.23% 块）→ TP 档（池 ×9）→ override（手动门）。

数字全部取自 figure_spec.numbers（measurement / scenarios / indexer_lever 段）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1520, 800
MX, BXR = 56, 1464
DEFS = lc.DEFS

C_POOL_S, C_POOL_F = lc.C_KV_S, lc.C_KV_F      # 池/进水 = 青
C_SUB_S, C_SUB_F = lc.C_ABORT, '#fee2e2'       # 扣减项 = 红
C_KEEP_S = lc.C_MUTE

# ---------------- 标题区 ----------------
lc.text(MX, 34, '96GiB 预算一条瀑布：池小了不是认输，按杠杆大小依次拧四个阀门', 16.5,
        lc.C_TXT, 'start', True, maxw=1040, tag='title')
lc.text(MX, 58, '进水口 96GiB × 0.92 = 88.32GiB · 权重是最大闸门（TP 档定开度）· 暗渠是 profile 实测激活峰与图池估计 · 出口读数 = 启动日志最后三行',
        10.5, lc.C_MUTE, 'start', maxw=1340, tag='subtitle')
_ch = '放大自 L2 拍片⑥⑧ + north 出口 · L0 启动视角'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 左：瀑布 =================
WF_X0, WF_X1 = 56, 920
BAR_X0 = 336
PXB = 560 / 96.0                      # px / GiB（96GiB → 560px）


def wf_bar(y, name, g0, g1, kind, note, val_in):
    """一行瀑布：g0→g1 为从左起的字节区间（GiB）；kind: start|keep|sub|pool"""
    lc.text(MX + 8, y + 25, name, 9.6, lc.C_TXT, 'start', True, maxw=270, tag='wf:' + name[:8])
    x0 = BAR_X0 + g0 * PXB
    w_ = abs(g1 - g0) * PXB
    if kind == 'sub':
        lc.rect(x0, y + 8, w_, 34, C_SUB_F, C_SUB_S, rx=3, sw=1.2, dash=True)
        lc.text(x0 + w_ / 2, y + 30, note, 9, C_SUB_S, 'middle', True, maxw=max(w_ - 8, 60), tag='wfn:' + name[:6])
    else:
        fill, stroke = (C_POOL_F, C_POOL_S) if kind in ('start', 'pool') else ('#ffffff', C_KEEP_S)
        lc.rect(x0, y + 8, w_, 34, fill, stroke, rx=3, sw=1.4)
        if kind == 'pool':
            lc.text(x0 + w_ / 2, y + 30, note, 9.6, C_POOL_S, 'middle', True, maxw=270, tag='wfn:' + name[:6])
        elif w_ >= 60:
            lc.text(x0 + w_ / 2, y + 30, note, 9, stroke, 'middle', True, maxw=w_ - 8, tag='wfn:' + name[:6])
        else:
            lc.text(x0 + w_ + 8, y + 30, note, 9, stroke, 'start', True, maxw=150, tag='wfn:' + name[:6])


RY0, RPITCH = 96, 52
wf_bar(RY0, '总显存（假设口径：96GiB 整）', 0, 96, 'start', '96GiB', True)
wf_bar(RY0 + RPITCH, '×0.92 → requested', 0, 88.32, 'start', '88.32GiB（余量 7.68GiB 留给看不见的峰）', True)
wf_bar(RY0 + 2 * RPITCH, '− 权重（TP2 · non_kv）', 88.32 - 78.7, 88.32, 'sub', '−78.7 est', True)
wf_bar(RY0 + 3 * RPITCH, '− 激活峰（profile 实测）', 88.32 - 78.7 - 3.6, 88.32 - 78.7, 'sub', '−3.6 est', True)
wf_bar(RY0 + 4 * RPITCH, '− CUDA 图池（估计）', 88.32 - 78.7 - 3.6 - 1.0, 88.32 - 78.7 - 3.6, 'sub', '−1.0 est', True)
wf_bar(RY0 + 5 * RPITCH, '= 池（available）', 0, 5.00, 'pool', '5.00GiB →『Available KV cache memory』', True)

# 各行右端剩余值（贴在扣减段右缘内侧之外）
for i, rem in [(2, '剩 9.62'), (3, '剩 6.02'), (4, '剩 5.02')]:
    lc.text(BAR_X0 + 88.32 * PXB + 8, RY0 + i * RPITCH + 30, rem, 8.2, lc.C_FAINT, 'start', maxw=64, tag=f'rem{i}')

# 池 → 块数 → 并发（链式两步）
CH_Y = RY0 + 6 * RPITCH + 4
lc.seg(BAR_X0 + 5.00 * PXB / 2, RY0 + 5 * RPITCH + 42, BAR_X0 + 5.00 * PXB / 2, CH_Y - 2, C_POOL_S, 2.0, 'std')
lc.rect(BAR_X0, CH_Y, 560, 40, '#ffffff', C_POOL_S, rx=7, sw=1.4)
lc.text(BAR_X0 + 14, CH_Y + 25, '÷ 917,568B/块 → 5,851 块（num_blocks = available // stride，一次整除）', 9.8,
        C_POOL_S, 'start', True, maxw=540, tag='chain1')
lc.text(MX + 8, CH_Y + 25, '→ 三行日志的出口读数', 9.6, lc.C_TXT, 'start', True, maxw=210, tag='chain0')

# ================= 右：护栏门 =================
GB_X = WF_X1 + 16
GB_W = BXR - GB_X
lc.rect(GB_X, RY0, GB_W, 178, C_SUB_F, C_SUB_S, rx=10, sw=1.8)
lc.text(GB_X + 16, RY0 + 26, '门口量身高（护栏 check_enough）', 12, C_SUB_S, 'start', True,
        maxw=GB_W - 32, tag='gb:t')
for i, ln in enumerate(['128K 上下文 · chunk 8192 → 在途 16,384',
                        '每请求峰值预留 7,194 块 → 护栏需要 6.453GiB > 池 5GiB',
                        '→ ValueError：附二分估计 max_len = 13,572',
                        '（护栏亲口告诉你的可行值——第一刀就砍这里）']):
    lc.text(GB_X + 16, RY0 + 50 + i * 19, ln, 9.2, '#334155' if i < 3 else '#7c2d12', 'start',
            maxw=GB_W - 30, tag=f'gb:{i}')
lc.text(GB_X + 16, RY0 + 50 + 4 * 19 + 6, '过护栏 ⇒ 并发 ≥ 1：『Maximum concurrency: N.NNx』必能打印', 8.6,
        lc.C_MUTE, 'start', maxw=GB_W - 30, tag='gb:n')

# ================= 底部：四个阀门（按杠杆排序） =================
VB_Y = 486
lc.rect(MX, VB_Y, BXR - MX, 200, '#f8fafc', lc.C_MUTE, rx=10, sw=1.5)
lc.text(MX + 18, VB_Y + 24, '四个阀门（同一方程链 available → num_blocks → 并发 → 护栏，收益全可换算成 Δ块数比较）', 12,
        lc.C_TXT, 'start', True, maxw=1100, tag='vb:t')
VALVES = [
    (lc.C_ENG_S, lc.C_ENG_F, '① max_model_len / chunk', '分母杠杆 −69.4%',
     ['护栏给 13,572：贴边活（1.0497×）', '或砍 chunk 8192→2048：在途 16,384→4,096', '分母 7,194→2,202 · 128K 并发 2.6571×']),
    (lc.C_ZMQ_S, lc.C_ZMQ_F, '② indexer fp8 → fp4', 'stride 杠杆 +9.23%',
     ['stride 1,002,240 → 917,568', '同池块数 +9.23%（8GiB：8,570→9,361）', '不换卡、不改并行，白拿']),
    (lc.C_GPU_S, lc.C_GPU_F, '③ TP 2 → 4', '池杠杆 ×9（est）',
     ['权重每卡 ≈78.7→39.4GiB(est)', '池 ≈44GiB(est) → 51,488 块', '128K 并发 7.1571×（offloading 配方即此形态）']),
    (lc.C_SAM_S, lc.C_SAM_F, '④ override / 手动档', '最后的手动门',
     ['num_gpu_blocks_override：配错 = OOM', 'kv_cache_memory_bytes：跳过 profile 直接定池', '手动档不受 utilization 约束']),
]
VGAP = 14
VW = (BXR - MX - 36 - 3 * VGAP) / 4
for i, (s, f, name, lever, lines) in enumerate(VALVES):
    vx = MX + 18 + i * (VW + VGAP)
    lc.rect(vx, VB_Y + 38, VW, 148, f, s, rx=7, sw=1.4)
    lc.text(vx + 12, VB_Y + 58, name, 9.8, s, 'start', True, maxw=VW - 24, tag=f'v{i}n')
    lc.text(vx + 12, VB_Y + 74, lever, 8.4, s, 'start', maxw=VW - 24, tag=f'v{i}l')
    for j, ln in enumerate(lines):
        lc.text(vx + 12, VB_Y + 94 + j * 17, ln, 8.2, '#334155', 'start', maxw=VW - 22, tag=f'v{i}j{j}')
    if i < 3:
        lc.seg(vx + VW, VB_Y + 112, vx + VW + VGAP - 2, VB_Y + 112, lc.C_MUTE, 1.6, 'std')

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 46, 'vllm/v1/worker/gpu_worker.py:L544-L548（方程 = requested − non_kv − cudagraph）· L563-L566（『Available KV cache memory』）· vllm/v1/worker/utils.py:L414-L416（ceil(总×util)）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 31, 'vllm/config/cache.py:L68（util 默认 0.92）· vllm/v1/core/kv_cache_utils.py:L2235-L2240（容量 / 并发两行日志）· 权重 / 激活 / 图池 = est（量级估计）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')
lc.text(MX, H - 16, '块数 / 护栏 / 并发 / 倍率 ＝ 本章账本算术脚本实算（pin 公式逐字复刻）· 行号基线 vLLM v0.27.1（6e448d0ea）', 8.5,
        lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot3')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m18_budget_waterfall.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
