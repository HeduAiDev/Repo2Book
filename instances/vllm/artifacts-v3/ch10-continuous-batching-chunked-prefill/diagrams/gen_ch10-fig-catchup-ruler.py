#!/usr/bin/env python3
"""ch10 机制图 2 · 追赶公式的进度带（figure_spec ch10-fig-catchup-ruler，模板 tiling）

放大自 L0『调度 · 显存账本』（kv_column 青色列）上半 Scheduler 框的「token 预算 ·
RUNNING 先行」格——即本章 L2 章图 center ① 拍片『RUNNING 先行 · 追赶公式』内公式的
机制展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：一条追赶公式三种形状：8192-token 的进度带被 2048 预算切成 4 块瓷砖（新 prompt
一块 + 续 chunk 三块），追平后每拍只铺 1 格——进度带总长 = num_tokens_with_spec，
铺过的高亮 = num_computed_tokens，每拍新铺的长度 = 公式结果被预算钳制后的 n。

数字全部取自 figure_spec.numbers（精简版 companion host 实测 trace：chunk_sizes
[2048,2048,2048,2048,1,1,1]，computed_progression [2048..8195]，raw_gap
8192→6144→4096→2048→1→1，占位项恒 0）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 780
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '追赶公式：没有 prefill 相位、没有 decode 相位，只有「已算追片长」一件事',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'num_new_tokens = num_tokens_with_spec + num_output_placeholders − num_computed_tokens（scheduler.py:L516-L520）——'
        '8192-token prompt 在 2048 预算下切 4 拍追平，之后每拍恰 1',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ① RUNNING 先行 · 追赶公式 · L0：调度账本列上半'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 进度带几何 ----------------
BAND_X0, BAND_X1 = 176.0, 936.0          # 进度带横向范围（760px = 8195 token）
PROMPT = 8192
PPX = (BAND_X1 - BAND_X0) / (PROMPT + 3)  # px per token


def bx(tok):
    return BAND_X0 + tok * PPX


# 七拍数据（trace m2：chunk / computed_after / raw_gap / with_spec）
BEATS = [
    (1, 2048, 2048, 8192, '新 prompt（WAITING 侧切块）'),
    (2, 2048, 4096, 8192, '续 chunk（RUNNING 侧同公式）'),
    (3, 2048, 6144, 8192, '续 chunk（差再减 2048）'),
    (4, 2048, 8192, 8192, '末 chunk——差恰等于预算'),
    (5, 1, 8193, 8193, 'decode'),
    (6, 1, 8194, 8194, 'decode'),
    (7, 1, 8195, 8195, 'decode'),
]

# 刻度带（进度带上方）
SCALE_Y = 108
lc.seg(BAND_X0 - 6, SCALE_Y, BAND_X1 + 6, SCALE_Y, lc.C_MUTE, 1.2)
for tok, lab in [(0, '0'), (2048, '2048'), (4096, '4096'), (6144, '6144'), (8192, '8192')]:
    x = bx(tok)
    lc.seg(x, SCALE_Y - 5, x, SCALE_Y + 5, lc.C_MUTE, 1.2)
    lc.text(x, SCALE_Y - 10, lab, 8.5, lc.C_MUTE, 'middle', maxw=60, tag='sc' + lab)
lc.text(MX + 44, SCALE_Y, 'token 进度', 8.5, lc.C_MUTE, 'end', maxw=90, tag='sc:unit')

# ---------------- 七拍行 ----------------
ROW_Y0, ROW_H = 128, 58
BAR_Y0, BAR_H = 14, 26
C_OLD, C_NEW, C_TODO = lc.C_KV_S, '#0e7490', lc.C_KV_F   # 已铺(中青) / 本拍新铺(深青) / 未铺(浅)
row_mid = {}
for bi, (beat, chunk, computed, with_spec, tagtxt) in enumerate(BEATS):
    ry = ROW_Y0 + bi * ROW_H
    mid = ry + BAR_Y0 + BAR_H / 2
    row_mid[beat] = mid
    # 拍号徽标
    lc.rect(MX, mid - 13, 42, 26, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.2)
    lc.text(MX + 21, mid + 3.5, f'拍 {beat}', 9.5, lc.C_ENG_S, 'middle', True, maxw=38,
            tag=f'bdg{beat}')
    # 进度带：已铺(旧) / 本拍新铺 / 未铺——长度严格 ∝ token 数
    old_end = computed - chunk
    if old_end > 0:
        lc.rect(bx(0), ry + BAR_Y0, bx(old_end) - bx(0), BAR_H, C_OLD, C_OLD, rx=2, sw=0)
    lc.rect(bx(old_end), ry + BAR_Y0, max(bx(computed) - bx(old_end), 2.2), BAR_H, C_NEW,
            C_NEW, rx=2, sw=0)
    todo0 = computed
    todo_w = bx(with_spec) - bx(computed)
    if todo_w > 0.6:
        lc.rect(bx(todo0), ry + BAR_Y0, todo_w, BAR_H, C_TODO, lc.C_KV_S, rx=2, sw=1.0)
    # decode 拍：右端 1 格白边可见性极小 → 由右侧放大镜承担（见下），行标签只画拍 1-4
    if beat <= 4:
        lc.text(bx(with_spec) + 10, mid - 3, tagtxt, 8.5, '#334155', 'start', maxw=140,
                tag=f'tg{beat}')
        lc.text(bx(with_spec) + 10, mid + 13, f'领到 {chunk}', 8.5, lc.C_KV_S, 'start', True,
                maxw=90, tag=f'got{beat}')

# ---------------- 公式代入列（右侧） ----------------
FX = 1120
lc.text(FX, 100, '公式代入（占位项恒 0）', 9, lc.C_MUTE, 'start', True, maxw=180, tag='fx:hd')
FORMULAS = [
    (1, '8192+0−0 = 8192 → 钳 2048'),
    (2, '8192+0−2048 = 6144 → 钳 2048'),
    (3, '8192+0−4096 = 4096 → 钳 2048'),
    (4, '8192+0−6144 = 2048 → 不钳，2048'),
    (5, 'decode：8193+0−8192 = 1 → 领 1'),
    (6, 'decode：8194+0−8193 = 1 → 领 1'),
    (7, 'decode：8195+0−8194 = 1 → 领 1'),
]
for beat, f in FORMULAS:
    lc.text(FX, row_mid[beat] + 3, f, 9, lc.C_TXT, 'start', maxw=BXR - FX, tag=f'fx{beat}')

# ---------------- 拍 4/5 之间：is_prefill_chunk 翻转分隔线 ----------------
sep_y = (row_mid[4] + row_mid[5]) / 2 + 6
lc.seg(MX, sep_y, 1084, sep_y, lc.C_BEAT_S, 1.6, dash=True)
lc.text(MX + 356, sep_y - 7, '拍 4 追平：is_prefill_chunk True → False（此后走 decode 闭环，差距恒回 1）',
        8.8, lc.C_BEAT_T, 'middle', True, maxw=560, tag='sep')

# ---------------- decode 拍放大镜（视觉锤：每拍恰 1 格） ----------------
LENS_CX, LENS_CY, LENS_R = 1000, row_mid[6], 52
lc.circle(LENS_CX, LENS_CY, LENS_R, lc.C_MUTE, 1.4, dash=True)
lc.seg(bx(8192) + 1, row_mid[5] + 4, LENS_CX - LENS_R * 0.72, LENS_CY - LENS_R * 0.72,
       lc.C_FAINT, 1.1, dash=True)
lc.seg(bx(8193) + 1, row_mid[6] - 2, LENS_CX - LENS_R * 0.72, LENS_CY - 2,
       lc.C_FAINT, 1.1, dash=True)
lc.seg(bx(8194) + 1, row_mid[7] - 4, LENS_CX - LENS_R * 0.72, LENS_CY + LENS_R * 0.55,
       lc.C_FAINT, 1.1, dash=True)
# 镜内：末三格放大（拍 5/6/7 各自新铺的那 1 格）
zy0 = LENS_CY - 8
for i in range(3):
    cx0 = LENS_CX - 42 + i * (24 + 6)
    beat_no = 5 + i
    lc.rect(cx0, zy0, 24, 22, '#ffffff', C_NEW, rx=3, sw=1.3)
    lc.text(cx0 + 12, zy0 + 14, f'拍{beat_no}', 7.5, '#334155', 'middle', True,
            maxw=24, tag=f'zc{beat_no}')
    lc.text(cx0 + 12, zy0 + 34, str(8192 + i), 7.5, lc.C_MUTE, 'middle', maxw=36,
            tag=f'zcl{beat_no}')
lc.text(LENS_CX, zy0 - 9, '每拍恰新铺 1 格', 8.8, lc.C_BEAT_T, 'middle', True, maxw=96, tag='lens:t')

# ---------------- why 注（虚线框） ----------------
WHY_Y = row_mid[7] + 52
lc.rect(MX, WHY_Y, 860, 62, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(MX + 14, WHY_Y + 18, '为什么同一公式吃得下三种形状', 9.5, lc.C_TXT, 'start', True,
        maxw=830, tag='why:t')
lc.text(MX + 14, WHY_Y + 36, '· 新 prompt：差 = 整段片长，被预算切成 ⌈8192/2048⌉ = 4 块瓷砖；首块走 WAITING 侧，'
        '后三块走 RUNNING 侧同一公式', 8.5, '#334155', 'start', maxw=830, tag='why:l1')
lc.text(MX + 14, WHY_Y + 53, '· decode：⑤ 拍回填 +1 与上拍乐观计入 +1 相抵，差距恢复型不变量恒 1——稳态即目的，不再收敛',
        8.5, '#334155', 'start', maxw=830, tag='why:l2')

# 量化小注（右下）
Q_X = 956
lc.text(Q_X, WHY_Y + 18, '三个刻度（同一公式）', 9.5, lc.C_TXT, 'start', True, maxw=200, tag='qt')
lc.text(Q_X, WHY_Y + 36, '8192 prompt：预算 8192/16384 → 1 拍；2048 → 4 拍', 8.5, '#334155',
        'start', maxw=BXR - Q_X, tag='q1')
lc.text(Q_X, WHY_Y + 53, '32k prompt 在 8192 预算 = 4 拍；公式 O(1) 每请求每拍', 8.5,
        '#334155', 'start', maxw=BXR - Q_X, tag='q2')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = WHY_Y + 88
lx = MX
items = [
    ('old', '已铺（此前拍累计 = num_computed_tokens）'),
    ('new', '本拍新铺 = n'),
    ('todo', '未铺（剩余差距）'),
]
for kind, name in items:
    if kind == 'old':
        lc.rect(lx, LEG_Y - 8, 20, 12, C_OLD, C_OLD, rx=3, sw=0)
    elif kind == 'new':
        lc.rect(lx, LEG_Y - 8, 20, 12, C_NEW, C_NEW, rx=3, sw=0)
    else:
        lc.rect(lx, LEG_Y - 8, 20, 12, C_TODO, lc.C_KV_S, rx=3, sw=1.2)
    lc.text(lx + 26, LEG_Y + 2, name, 8.5, lc.C_TXT, 'start', maxw=330, tag='leg' + kind)
    lx += 26 + lc.tw(name, 8.5) + 24
lc.seg(lx, LEG_Y - 2, lx + 22, LEG_Y - 2, lc.C_BEAT_S, 1.3, dash=True)
lc.text(lx + 28, LEG_Y + 2, '相位翻转线', 8.5, lc.C_TXT, 'start', maxw=90, tag='leg:flip')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/sched/scheduler.py:L516-L520（追赶公式）· L521-L532（三重钳制）· '
        '字段定义 vllm/v1/core/sched/request.py:L271-L277 · 占位项恒 0：同步版语义（async 才灌值 → ch12）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '七拍读数取自精简版 companion host 实测 · max_model_len 保险钳本例不触发（headroom ≥10239）——它是 spec/边界的保险 '
        '· 行号基线 vLLM v0.27.1', 8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch10-fig-catchup-ruler.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
