#!/usr/bin/env python3
"""ch15 机制图 11 · 混合命中调和的不动点（figure_spec ch15-fig-hybrid-fixed-point，模板 state-table）

放大自 L0 KV 账本列（kv_column）缓存区·命中主循环——「多组不动点」一拍的展开。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：混合命中调和是不动点——每个注意力类型对候选长度要么接受要么缩短，任一缩短就重启
全类型校验；长度单调递减有下界必收敛，full 组向下封闭只在首轮查一次。

数字全部取自 figure_spec.numbers（配套精简版 host 实跑）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440
GREEN = '#16a34a'
RED = '#dc2626'
ORANGE = '#ea580c'
GRAY = '#94a3b8'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '会签合同的不动点：谁砍短了就得全体重审一轮——长度只减不增、必收敛',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, 'full(16) + SWA(窗48) + SWA(窗32) 三组、96 token：full 给 80，SWA48 砍到 48；48 < 95 重启第二轮——5 次 finder 调用收敛',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · KV 账本列缓存区 · 命中主循环「多组不动点」'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

LY = 96

# ---------------- 左：finder 调用账本（逐行） ----------------
LX, LW = MX, 850
lc.rect(LX, LY, LW, 452, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(LX + 16, LY + 22, 'finder 调用账本（候选长度入 → 命中出）· 96 token · 稀疏驻留摘掉 SWA48 hash[3]、SWA32 hash[4]',
        11, lc.C_TXT, 'start', True, maxw=LW - 32, tag='lp:t')
HEADS = [('轮', 44), ('finder', 100), ('max_length 入', 106), ('hit 出', 74), ('动作 / 要点', 470)]
ROWS = [
    ('1', 'full', '95', '80', '左到右扫 5 块——full 排首，先给最紧初始上界', lc.C_ENG_S, False),
    ('1', 'SWA(48)', '80', '48', '右到左找 3 连续块；hash[3] 被摘 → 窗口连续段断成 3 块', ORANGE, True),
    ('1', 'SWA(32)', '48', '48', '拿到的候选已是 48——[NULL,b1,b2] 也只值 48（轮内传递）', GRAY, False),
    ('—', '', '', '', '48 < 95 ⇒ 重启全类型校验（第二轮）', lc.C_TXT, False),
    ('2', 'full', '（缺席）', '—', '向下封闭：首轮查过、后续轮只 min 裁剪——不再查', GRAY, False),
    ('2', 'SWA(48)', '48', '48', '以 48 复验通过 ✓', GREEN, False),
    ('2', 'SWA(32)', '48', '48', '以 48 复验通过 ✓ → 收敛（候选不再变短）', GREEN, False),
]
hy = LY + 46
cx0 = LX + 18
for (name, cwid) in HEADS:
    lc.text(cx0, hy, name, 8.8, lc.C_MUTE, 'start', True, maxw=cwid + 60, tag='th:' + name)
    cx0 += cwid
for i, (rnd, fd, m_in, h_out, note, col, shrink) in enumerate(ROWS):
    yy = hy + 18 + i * 46
    if i == 3:
        lc.rect(LX + 12, yy - 6, LW - 24, 34, '#f8fafc', GRAY, rx=5, sw=0.9, dash=True)
    if shrink:
        lc.rect(LX + 12, yy - 6, LW - 24, 46, '#fff7ed', ORANGE, rx=5, sw=0.0)
    cx0 = LX + 18
    lc.text(cx0, yy + 12, rnd, 9.4, lc.C_TXT, 'start', True, maxw=36, tag='r%d:n' % i)
    cx0 += HEADS[0][1]
    lc.text(cx0, yy + 12, fd, 9.4, col if fd else GRAY, 'start', True, maxw=96, tag='r%d:f' % i)
    cx0 += HEADS[1][1]
    lc.text(cx0, yy + 12, m_in, 9.4, GRAY if m_in in ('（缺席）', '') else lc.C_TXT, 'start',
            True, maxw=100, tag='r%d:i' % i)
    cx0 += HEADS[2][1]
    lc.text(cx0, yy + 12, h_out, 9.4, GREEN if h_out == '48' else (lc.C_TXT if h_out else GRAY),
            'start', True, maxw=68, tag='r%d:o' % i)
    cx0 += HEADS[3][1]
    lc.text(cx0, yy + 12, note, 8.6, '#334155', 'start', maxw=HEADS[4][1], tag='r%d:t' % i)

# ---------------- 右：候选长度阶梯 + 收据 ----------------
RX, RW = LX + LW + 24, BXR - (LX + LW + 24)
RH = 452
lc.rect(RX, LY, RW, RH, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(RX + 16, LY + 22, '候选长度阶梯（只降不升）', 11, lc.C_TXT, 'start', True, maxw=RW - 32,
        tag='rp:t')
STEPS = [(95, 'max_cache_hit_length = 96−1', GRAY), (80, 'full 的链上命中', lc.C_ENG_S),
         (48, 'SWA48 砍定（后续全 48）', ORANGE)]
SY0, SLINE = LY + 56, 44
for i, (val, label, col) in enumerate(STEPS):
    yy = SY0 + i * SLINE
    bx = RX + 30
    bw = (RW - 60) * val / 100
    lc.text(bx + 4, yy - 6, '%d · %s' % (val, label), 8.2, col, 'start', maxw=RW - 64,
            tag='st%d' % i)
    lc.rect(bx, yy, bw, 22, '#ffffff', col, rx=4, sw=1.4)
    if i > 0:
        prev_bw = (RW - 60) * STEPS[i - 1][0] / 100
        lc.seg(RX + 30 + prev_bw / 2, yy - SLINE + 22, RX + 30 + bw / 2, yy - 2, col, 1.2,
               'std')
lc.text(RX + 16, SY0 + 3 * SLINE + 12, '对齐格点（16 的倍数）上的严格递减非负整数列——有限步必停',
        8.4, '#475569', 'start', maxw=RW - 32, tag='rp:st')
# 收据
ry = SY0 + 3 * SLINE + 34
lc.rect(RX + 16, ry, RW - 32, 108, lc.C_KV_F, lc.C_KV_S, rx=6, sw=1.3)
lc.text(RX + 30, ry + 21, '收敛收据：2 轮 / 5 次 finder 调用', 9.8, lc.C_KV_S, 'start', True,
        maxw=RW - 60, tag='rp:r1')
lc.text(RX + 30, ry + 41, 'reconciled 48 · longest 80', 9, '#334155', 'start', True,
        maxw=RW - 60, tag='rp:r2')
lc.text(RX + 30, ry + 59, 'uncached = longest 80 − reconciled 48 = 32', 9, GREEN, 'start', True,
        maxw=RW - 60, tag='rp:r3')
lc.text(RX + 30, ry + 79, '——各组都认、稀疏组还没缓的共享前缀，', 8.4, '#334155', 'start',
        maxw=RW - 60, tag='rp:r4')
lc.text(RX + 30, ry + 95, '正是下一站 junction 的原料', 8.4, '#334155', 'start',
        maxw=RW - 60, tag='rp:r5')
# 上界
uy = ry + 128
lc.text(RX + 16, uy, '上界轮数 95/16 = 5 轮（同块尺寸 lcm=16）、实测 2 轮；', 8.4, '#475569',
        'start', maxw=RW - 32, tag='rp:u1')
lc.text(RX + 16, uy + 17, 'simple hybrid（1 full + 1 other）首轮后直接 break——', 8.4,
        '#475569', 'start', maxw=RW - 32, tag='rp:u2')
lc.text(RX + 16, uy + 34, '场景 A（两组）就是 1 轮 / 2 次调用收敛到 48', 8.4, '#475569',
        'start', maxw=RW - 32, tag='rp:u3')

# ---------------- 底部不变量条（全宽） ----------------
BY = LY + 452 + 16
lc.rect(MX, BY, BXR - MX, 58, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, BY + 22, '不动点必收敛：每轮每类型要么接受候选、要么换成自己 finder 的返回值（≤ 当前候选）——轮末仅当 curr < hit 才重开',
        9.6, lc.C_KV_S, 'start', True, maxw=BXR - MX - 32, tag='inv:t')
lc.text(MX + 16, BY + 42, '加速器三层：full 向下封闭（查一次后只 min 裁剪）· simple hybrid 首轮即停 · full 排首让左到右扫描先给最紧上界、后面组少做功',
        9, '#334155', 'start', maxw=BXR - MX - 32, tag='inv:l1')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BY + 80
lx = MX
for fill, stroke, tcol, name in [
        ('#fff7ed', ORANGE, ORANGE, '把候选砍短的 finder 调用'),
        ('#ffffff', GREEN, GREEN, '复验通过（接受候选）'),
        ('#ffffff', lc.C_ENG_S, lc.C_ENG_S, 'full 组（排首 · 向下封闭）')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, fill, stroke, rx=3, sw=1.2)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=220, tag='lg')
    lx += 26 + lc.tw(name, 8.8) + 24
lc.text(lx, LEG_Y + 1, 'finder = 各注意力类型自己的最长命中查找；NULL 占位 = 窗外块换位（ch14 已立）',
        8.8, lc.C_MUTE, 'start', maxw=BXR - lx, tag='lg:note')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/kv_cache_coordinator.py:L685-L817（不动点循环：接受或缩短 · 重启 · '
        'num_uncached_common_prefix_tokens = longest − reconciled）· SWA finder 右到左找窗口连续块（ch14 remove_skipped 的对偶）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '数字取自配套精简版 host 实跑（96 token · 三组 · 摘 hash[3]/hash[4] 模拟稀疏驻留）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=700, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 60
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch15-fig-hybrid-fixed-point.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
