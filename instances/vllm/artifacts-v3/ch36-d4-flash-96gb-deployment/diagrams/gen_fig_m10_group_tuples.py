#!/usr/bin/env python3
"""ch36 机制图 · DSV4 专属分组四步（figure_spec fig_m10_group_tuples，模板 flow）

放大自 L2 章图拍片⑦（DSV4 专属分组）的机制小图——L0 启动视角『分组』格。

claim：分组四步：168 份 spec 按 (block,window) 分 4 桶（full_mla 62 / (64,128) 44 /
(4,8) 42 / (8,128) 20）→ 每桶元组数 [21,44,21,20] → _approximate_gcd 暴力选 22
（总垫 4 最小、d=21 垫 20）→ SWA 桶拆两半、(4,8) 桶 zip 成 21 对双生元组，
final 5 组、含 MTP 的组标 eagle。

数字全部取自 figure_spec.numbers（approximate_gcd_walk / final_groups / eagle_group_index）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 660
MX, BXR = 56, 1444
DEFS = lc.DEFS

C_MLA_S, C_MLA_F = lc.C_ZMQ_S, lc.C_ZMQ_F      # full_mla = 紫
C_SWA_S, C_SWA_F = lc.C_API_S, lc.C_API_F      # 窗口 = 蓝
C_ST4_S, C_ST4_F = lc.C_ENG_S, lc.C_ENG_F      # C4 状态 = 橙
C_ST128_S, C_ST128_F = lc.C_SAM_S, lc.C_SAM_F  # C128 状态 = 品红

# ---------------- 标题区 ----------------
lc.text(MX, 34, '分组的数学：四个桶层数互质，暴力搜出总垫最小的粒度 22（只垫 4 层）', 16.5,
        lc.C_TXT, 'start', True, maxw=1020, tag='title')
lc.text(MX, 58, 'SlidingWindowMLASpec 按 (block, window) 分桶、全部 MLAAttentionSpec 合一 full_mla —— 『目前只有 DeepseekV4 用』（kv_cache_utils.py:L1592-L1598）',
        10.5, lc.C_MUTE, 'start', maxw=1300, tag='subtitle')
_ch = '放大自 L2 拍片⑦ · L0 启动视角'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 四阶段面板 ----------------
PY, PH = 100, 360
S1W, S2W, S3W, S4W = 316, 226, 424, 360
S1X, S2X = MX, MX + S1W + 18
S3X, S4X = S2X + S2W + 18, BXR - S4W          # 634 / 1084（S3 右缘 1058，与 S4 净距 26）

PANELS = [
    (S1X, S1W, '① (block, window) 分桶'),
    (S2X, S2W, '② 每桶元组数'),
    (S3X, S3W, '③ _approximate_gcd 暴力选粒度'),
    (S4X, S4W, '④ 拆分 · 配对 · eagle'),
]
for px, pw, title in PANELS:
    lc.rect(px, PY, pw, PH, '#ffffff', lc.C_MUTE, rx=10, sw=1.6)
    lc.text(px + 14, PY + 24, title, 11.5, lc.C_TXT, 'start', True, maxw=pw - 28, tag='sp:' + title[:4])

# ① 四个桶
BUCKETS = [
    (C_MLA_S, C_MLA_F, 'full_mla（全部 MLAAttentionSpec 合一）', '62'),
    (C_SWA_S, C_SWA_F, '(64, 128) 窗口', '44'),
    (C_ST4_S, C_ST4_F, '(4, 8) C4 状态', '42'),
    (C_ST128_S, C_ST128_F, '(8, 128) C128 状态', '20'),
]
by = PY + 44
for s, f, name, n in BUCKETS:
    lc.rect(MX + 14, by, S1W - 28, 46, f, s, rx=7, sw=1.4)
    lc.text(MX + 26, by + 28, name, 9.2, s, 'start', True, maxw=S1W - 92, tag='bk:' + name[:6])
    bw = 30
    lc.rect(MX + S1W - 14 - bw, by + 10, bw, 26, '#ffffff', s, rx=12, sw=1.2)
    lc.text(MX + S1W - 14 - bw / 2, by + 28, n, 11, s, 'middle', True, maxw=bw - 4, tag='bkn:' + n)
    by += 56
lc.text(MX + 14, PY + PH - 14, '168 份 spec → 4 个 Uniform 桶', 8.8, lc.C_MUTE, 'start', maxw=S1W - 28, tag='b1n')

# ② 元组数
lc.text(S2X + S2W / 2, PY + 92, '[21, 44, 21, 20]', 17, lc.C_TXT, 'middle', True, maxw=S2W - 20, tag='tup')
lc.text(S2X + S2W / 2, PY + 118, 'full_mla · SWA · C4态 · C128态', 8.8, lc.C_MUTE, 'middle', maxw=S2W - 16, tag='tup:o')
for i, ln in enumerate(['每桶元组数 = 最常见页的层数',
                        '桶内各页大小的层数必相等',
                        '（源码 assert）',
                        '→ zip(*layers) 恰好配平不剩单']):
    lc.text(S2X + 14, PY + 152 + i * 19, ln, 9.2, '#334155', 'start', maxw=S2W - 26, tag=f't2:{i}')

# ③ gcd 暴力走表（d=21..44 的总垫层数）
PADW = {21: 20, 22: 4, 23: 9, 24: 14, 25: 19, 26: 24, 27: 29, 28: 34, 29: 39, 30: 44,
        31: 49, 32: 54, 33: 59, 34: 64, 35: 69, 36: 74, 37: 79, 38: 84, 39: 89, 40: 94,
        41: 99, 42: 104, 43: 109, 44: 70}
CH_X0, CH_Y0, CH_Y1 = S3X + 44, PY + 280, PY + 150      # 基线 / 顶
max_pad = max(PADW.values())
bw_, gap_ = 12.5, 2.5
lc.seg(CH_X0 - 8, CH_Y0, CH_X0 + 24 * (bw_ + gap_) + 4, CH_Y0, '#cbd5e1', 1.2)
lc.seg(CH_X0 - 8, CH_Y0, CH_X0 - 8, CH_Y1 - 6, '#cbd5e1', 1.2)
lc.text(S3X + 14, CH_Y1 + 2, '总垫层数', 8.2, lc.C_FAINT, 'start', maxw=50, tag='ax:pad')
for i, d in enumerate(range(21, 45)):
    h_ = PADW[d] / max_pad * (CH_Y0 - CH_Y1 - 10)
    x = CH_X0 + i * (bw_ + gap_)
    hit = (d == 22)
    lc.rect(x, CH_Y0 - h_, bw_, h_, lc.C_GPU_F if hit else '#e2e8f0', lc.C_GPU_S if hit else '#94a3b8',
            rx=1.5, sw=1.0)
    if d in (21, 22, 23, 44):
        lc.text(x + bw_ / 2, CH_Y0 + 12, str(d), 8, lc.C_GPU_S if hit else lc.C_MUTE, 'middle', True,
                maxw=30, tag=f'dl{d}')
    if hit:
        lc.text(x + bw_ / 2, CH_Y0 - h_ - 6, '垫 4', 8.6, lc.C_GPU_S, 'middle', True, maxw=40, tag='dhit')
lc.text(S3X + 14, PY + PH - 42, 'd∈[21,44] 逐个试：pad(d)=Σ(⌈x/d⌉·d−x)，总垫最小者胜、并列取大 d', 8.6,
        '#334155', 'start', maxw=S3W - 28, tag='gcd:n1')
lc.text(S3X + 14, PY + PH - 24, 'd=21 垫 20 · d=22 垫 4（最小 → 选定）· d=23 垫 9 · d=44 垫 70', 8.6,
        lc.C_GPU_S, 'start', True, maxw=S3W - 28, tag='gcd:n2')

# ④ 收尾
S4X = BXR - S4W
FIN = [
    ('SWA 桶拆分', 'cdiv(44, 22) = 2 → 两个 22 层子组'),
    ('(4,8) 桶双生配对', 'zip(*两页类) → 21 对 [attn 32,832B + indexer 8,640B]'),
    ('final 5 组', '62 / 22 / 22 / 42 / 20 层'),
    ('eagle 标注', '含 MTP 末层（SWA-2）→ 整组标 eagle（spec decode 档期组）'),
]
fy = PY + 44
for name, note in FIN:
    lc.rect(S4X + 14, fy, S4W - 28, 62, '#f8fafc', lc.C_MUTE, rx=7, sw=1.1)
    lc.text(S4X + 26, fy + 20, name, 9.6, lc.C_TXT, 'start', True, maxw=S4W - 52, tag='fn:' + name[:4])
    lc.text(S4X + 26, fy + 40, note, 8.6, '#475569', 'start', maxw=S4W - 46, tag='fo:' + name[:4])
    fy += 72
lc.text(S4X + 14, PY + PH - 12, '喂给 packed 定账（下一张图）', 8.8, lc.C_MUTE, 'start', maxw=S4W - 28, tag='f4n')

# 阶段间箭头（面板边缘 → 面板边缘）
mid_y = PY + PH / 2
for x0, x1 in ((S1X + S1W, S2X), (S2X + S2W, S3X), (S3X + S3W, S4X)):
    lc.seg(x0 + 1, mid_y, x1 - 2, mid_y, lc.C_MUTE, 2.0, 'std')

# ---------------- 底部结论带 ----------------
BB_Y = PY + PH + 20
lc.rect(MX, BB_Y, BXR - MX, 76, '#f8fafc', lc.C_MUTE, rx=10, sw=1.5)
lc.text(MX + 18, BB_Y + 24, '为什么必须垫整：packed 布局要求各组元组数对齐——22 是唯一让总垫最小的粒度（垫 4 层 vs d=21 的 20 层）', 11,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 36, tag='bb:t')
lc.text(MX + 18, BB_Y + 48, 'MTP 落在第二个 SWA 子组、整组标 eagle——spec decode 的档期组（草稿层的窗口账与主账同池共管，verify 链回指 ch34）',
        9.2, '#475569', 'start', maxw=BXR - MX - 36, tag='bb:n')

# ---------------- 页脚锚点 ----------------
lc.text(MX, H - 44, 'vllm/v1/core/kv_cache_utils.py:L1592-L1632（DSV4 专属分组 ·『目前只有 DeepseekV4 用』）· L1635-L1667（_approximate_gcd：暴力 d、并列取大 d）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, H - 29, 'vllm/v1/core/kv_cache_utils.py:L1726-L1729（桶内等数 assert）· L1757-L1778（eagle 标注）· 垫整走表 / final 5 组 ＝ 本章账本算术脚本实算（pin 函数逐字复刻）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')
lc.text(MX, H - 14, '行号基线 vLLM v0.27.1（6e448d0ea）', 8.5, lc.C_FAINT, 'start', maxw=400, tag='foot3')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'fig_m10_group_tuples.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
