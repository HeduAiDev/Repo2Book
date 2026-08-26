#!/usr/bin/env python3
"""ch10 机制图 3 · 两阶段刷一张卡（figure_spec ch10-fig-one-budget-two-phases，模板 state-table）

放大自 L0『调度 · 显存账本』（kv_column 青色列）上半 Scheduler 框的「token 预算」格——
即本章 L2 章图 center ①-⑥ 拍片共用的那条预算水位线（跨拍片贯穿线）的机制展开。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：一个 token_budget 变量跨 RUNNING/WAITING 两阶段分账：预算 4 的心算例里，
拍 1 全家刷 4（r1 全量 2 + r2 首 chunk 2）、拍 2/3 又各刷 4、拍 4 只刷 2——
每拍水位降到 0 或停在中途，拍拍守恒。

数字全部取自 figure_spec.numbers（精简版 companion host 实测 trace：场景 B 四拍
花销 4/4/4/2、余额 0/0/0/2；r2 chunk [2,3,3] 后转 decode 领 1；场景 A 拍 2 花销
32 恰打满、拍 4 只花 9 余 23；扣减点 L459/L523+L637/L913+L1073/L1108-L1111）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 792
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'RUNNING 与 WAITING 刷的是同一张卡——token_budget 单变量跨两阶段分账',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '阶段一在途请求先刷，阶段二新请求只刷余额；卡里的数字一路单调下降、永远不为负——拍末守恒断言把这条账律写成机器自检',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ①-⑥ 贯穿的 token_budget 水位线 · L0：调度账本列上半'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 场景 B（主图，预算 4 的心算刻度） ----------------
SB_Y = 92
lc.text(MX, SB_Y, '场景 B · 预算 4 的心算刻度：r1 2-token prompt + r2 8-token prompt（r2 被余额切成 2+3+3 三段）',
        10.5, lc.C_TXT, 'start', True, maxw=1000, tag='sb:t')
HDR_Y = SB_Y + 22
ROW_Y0, ROW_H = HDR_Y + 8, 84

# 列头
lc.text(MX + 23, HDR_Y, '拍', 9.5, lc.C_MUTE, 'middle', True, maxw=40, tag='hd:beat')
lc.text(170, HDR_Y, '本拍账单（票据宽 ∝ 领取额，左侧标刷卡阶段）', 9.5, lc.C_MUTE, 'start',
        True, maxw=420, tag='hd:bill')
G_X, G_W = 700, 26
lc.text(G_X + G_W / 2, HDR_Y, '水位条 0..4', 9.5, lc.C_MUTE, 'middle', True, maxw=110, tag='hd:g')
lc.text(792, HDR_Y, '花销 / 余额', 9.5, lc.C_MUTE, 'middle', True, maxw=110, tag='hd:amt')
lc.text(920, HDR_Y, '守恒断言', 9.5, lc.C_MUTE, 'middle', True, maxw=110, tag='hd:asrt')
lc.text(1010, HDR_Y, '解读', 9.5, lc.C_MUTE, 'start', True, maxw=100, tag='hd:obs')

B_BEATS = [
    (1, 'WAITING 收新', [('r1', 2, 'wait'), ('r2', 2, 'wait')], 4, 0,
     ['r1 全量 2；r2 首 chunk 2', '（需求 8 被余 2 截）']),
    (2, 'RUNNING 先行', [('r1', 1, 'run'), ('r2', 3, 'run')], 4, 0,
     ['r1 decode 1；r2 续 chunk 3', '（差 6 被余 3 截）']),
    (3, 'RUNNING 先行', [('r1', 1, 'run'), ('r2', 3, 'run')], 4, 0,
     ['r2 续 chunk 3', '（差 3 恰等于余 3）']),
    (4, 'RUNNING 先行', [('r1', 1, 'run'), ('r2', 1, 'run')], 2, 2,
     ['r2 转 decode 恰 1', 'chunk 收官 · 批不满']),
]
TKPX = 30.0   # px per token
for bi, (beat, phase, tickets, spent, left, obs) in enumerate(B_BEATS):
    ry = ROW_Y0 + bi * ROW_H
    if bi > 0:
        lc.seg(MX, ry - 2, 990, ry - 2, '#e2e8f0', 1.0)
    # 拍号徽标
    lc.rect(MX, ry + 24, 46, 34, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.2)
    lc.text(MX + 23, ry + 45, f'拍 {beat}', 10.5, lc.C_ENG_S, 'middle', True, maxw=42, tag=f'bdg{beat}')
    # 阶段标签
    lc.text(124, ry + 34, phase, 8.5, lc.C_MUTE if phase.startswith('R') else lc.C_BEAT_T,
            'end', True, maxw=86, tag=f'ph{beat}')
    # 票据
    tx = 140
    for rid, n, side in tickets:
        w = max(12.0, n * TKPX)
        ty = ry + 22
        if side == 'run':
            lc.rect(tx, ty, w, 24, lc.C_KV_S, lc.C_KV_S, rx=3, sw=0)
        else:
            lc.rect(tx, ty, w, 24, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.4)
        lc.text(tx + w / 2, ry + 70, f'{rid}:{n}', 9, lc.C_TXT, 'middle', True, maxw=70,
                tag=f'tk{beat}{rid}')
        tx += w + 26
    # 水位条（0..4，自底向上染到花销）
    gy0, gh = ry + 8, 66
    lc.rect(G_X, gy0, G_W, gh, '#ffffff', lc.C_MUTE, rx=3, sw=1.1)
    fh = gh * spent / 4
    if fh > 0.5:
        lc.rect(G_X, gy0 + gh - fh, G_W, fh, lc.C_BEAT_S, lc.C_BEAT_S, rx=2, sw=0)
    for k in range(1, 4):
        yy = gy0 + gh - gh * k / 4
        lc.seg(G_X, yy, G_X + 5, yy, lc.C_MUTE, 0.9)
    # 花销/余额
    lc.text(792, ry + 38, f'花 {spent}', 11, lc.C_BEAT_T, 'middle', True, maxw=80, tag=f'sp{beat}')
    lc.text(792, ry + 58, f'余 {left}', 9.5, lc.C_MUTE, 'middle', maxw=80, tag=f'lf{beat}')
    # 守恒断言
    lc.text(920, ry + 48, f'Σ={spent} ≤ 4 ✓', 10, lc.C_GPU_S, 'middle', True, maxw=110,
            tag=f'as{beat}')
    # 解读
    for li, ln in enumerate(obs):
        lc.text(1010, ry + 38 + li * 19, ln, 8.7, '#334155', 'start', maxw=BXR - 1010,
                tag=f'obs{beat}:{li}')

B_END = ROW_Y0 + 4 * ROW_H

# ---------------- 场景 A（对照，32 刻度） ----------------
SA_Y = B_END + 30
lc.text(MX, SA_Y, '场景 A 对照 · 同一画法 32 刻度（复现 m1 的混合批）：拍 2 三个 decode 各 1（RUNNING）+ r4 首 chunk 29（WAITING）恰打满；拍 4 只花 9 余 23',
        10.5, lc.C_TXT, 'start', True, maxw=1080, tag='sa:t')
A_ROW_Y, A_ROW_H = SA_Y + 14, 72
A_BEATS = [
    (2, [('r1', 1, 'run'), ('r2', 1, 'run'), ('r3', 1, 'run'), ('r4', 29, 'wait')], 32, 0,
     '阶段一先付 3，阶段二 r4 只领余 29——32 恰打满'),
    (4, [('r1', 1, 'run'), ('r2', 1, 'run'), ('r3', 1, 'run'), ('r4', 6, 'run')], 9, 23,
     '尾 chunk 6 收官：只花 9，余 23'),
]
ATKPX = 8.0
for bi, (beat, tickets, spent, left, note) in enumerate(A_BEATS):
    ry = A_ROW_Y + bi * A_ROW_H
    lc.rect(MX, ry + 16, 46, 30, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.1)
    lc.text(MX + 23, ry + 36, f'拍 {beat}', 9.5, lc.C_ENG_S, 'middle', True, maxw=42, tag=f'abd{beat}')
    tx = 140
    for rid, n, side in tickets:
        w = max(10.0, n * ATKPX)
        ty = ry + 14
        if side == 'run':
            lc.rect(tx, ty, w, 22, lc.C_KV_S, lc.C_KV_S, rx=3, sw=0)
        else:
            lc.rect(tx, ty, w, 22, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.4)
        if n > 1:
            lc.text(tx + w / 2, ry + 29, f'{rid}:{n}', 8.5, '#ffffff', 'middle', True, maxw=w - 6,
                    tag=f'atk{beat}{rid}')
        else:
            lc.text(tx + w / 2, ry + 52, f'{rid}:{n}', 8, lc.C_MUTE, 'middle', maxw=56,
                    tag=f'atk{beat}{rid}')
        tx += w + 18
    # 32 刻度横条（花销/32）
    bar_x, bar_w, bar_y, bar_h = 700, 240, ry + 20, 18
    lc.rect(bar_x, bar_y, bar_w, bar_h, '#ffffff', lc.C_MUTE, rx=3, sw=1.1)
    fw = bar_w * spent / 32
    if fw > 0.5:
        lc.rect(bar_x, bar_y, fw, bar_h, lc.C_BEAT_S, lc.C_BEAT_S, rx=2, sw=0)
    lc.text(bar_x + bar_w + 10, ry + 33, f'花 {spent} / 余 {left}', 8.5, lc.C_MUTE, 'start',
            maxw=110, tag=f'abar{beat}')
    lc.text(1010, ry + 33, note, 8.7, '#334155', 'start', maxw=BXR - 1010, tag=f'aobs{beat}')

# ---------------- 归纳链小注 + 图例 ----------------
NOTE_Y = A_ROW_Y + 2 * A_ROW_H + 24
lc.rect(MX, NOTE_Y, 1080, 58, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(MX + 14, NOTE_Y + 18, '守恒不是事后校验，是构造出来的（归纳链）', 9.5, lc.C_TXT,
        'start', True, maxw=1050, tag='nt:t')
lc.text(MX + 14, NOTE_Y + 36, '· 拍首 budget = max（基例）；每次领取都被 min(num_new_tokens, token_budget) 钳到 ≤ 余额再扣减 → 扣后仍 ≥ 0（归纳步）',
        8.5, '#334155', 'start', maxw=1050, tag='nt:l1')
lc.text(MX + 14, NOTE_Y + 52, '· 两个循环的入环条件都含 token_budget > 0——预算耗尽即停止入账，断言（L1108-L1111）只是最后一道保险丝',
        8.5, '#334155', 'start', maxw=1050, tag='nt:l2')
# 右侧量级注
lc.text(1170, NOTE_Y + 18, '记账量级', 9.5, lc.C_TXT, 'start', True, maxw=140, tag='q:t')
lc.text(1170, NOTE_Y + 36, '每拍 O(running+waiting) 次', 8.5, '#334155', 'start',
        maxw=BXR - 1170, tag='q:l1')
lc.text(1170, NOTE_Y + 52, '每次 O(1)，纯 Python 循环', 8.5, '#334155', 'start',
        maxw=BXR - 1170, tag='q:l2')

LEG_Y = NOTE_Y + 82
lx = MX
items = [
    ('run', 'RUNNING 侧领取（阶段一，深青）'),
    ('wait', 'WAITING 侧领取（阶段二，浅青）'),
    ('gag', '预算水位（本拍花销）'),
]
for kind, name in items:
    if kind == 'run':
        lc.rect(lx, LEG_Y - 8, 20, 12, lc.C_KV_S, lc.C_KV_S, rx=3, sw=0)
    elif kind == 'wait':
        lc.rect(lx, LEG_Y - 8, 20, 12, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.4)
    else:
        lc.rect(lx, LEG_Y - 8, 14, 14, lc.C_BEAT_S, lc.C_BEAT_S, rx=3, sw=0)
    lc.text(lx + 26, LEG_Y + 2, name, 8.5, lc.C_TXT, 'start', maxw=300, tag='leg' + kind)
    lx += 26 + lc.tw(name, 8.5) + 22
lc.text(lx, LEG_Y + 2, '✓ = 守恒断言通过', 8.5, lc.C_GPU_S, 'start', True, maxw=160, tag='leg:ok')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/sched/scheduler.py:L459（token_budget = max_num_scheduled_tokens 初始化）· L523+L637（RUNNING 侧钳制+扣减）· '
        'L913+L1073（WAITING 侧钳制+扣减）· L1108-L1111（守恒断言）', 8.5, lc.C_FAINT, 'start',
        maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '两场景读数取自精简版 companion host 实测 · 4/32 为示教预算——真实默认 2048（config/scheduler.py:L42）/ 服务端 8192/16384（arg_utils.py:L2541-L2563）'
        '· 行号基线 vLLM v0.27.1', 8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch10-fig-one-budget-two-phases.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
