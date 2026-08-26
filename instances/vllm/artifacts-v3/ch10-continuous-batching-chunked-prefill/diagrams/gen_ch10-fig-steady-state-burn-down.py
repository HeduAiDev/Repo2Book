#!/usr/bin/env python3
"""ch10 机制图 9 · 稳态批的燃尽表（figure_spec ch10-fig-steady-state-burn-down，模板 state-table）

放大自 L0『调度 · 显存账本』（kv_column 青色列）上半 Scheduler 框整体（token 预算 +
RUNNING 先行 + chunked 切块三格合观的稳态视图）——即本章 L2 章图 center ①-⑦ 拍片带 +
loop 回环箭头『下一拍：chunk 未完的请求已住 running…decode 请求恰 1 token』的机制展开。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：连续批处理稳态批组成表：五拍里 [1,1,30] → [1,1,30] → [1,1,4] → [1,1,1] → [1,1,1]
——decode 的 1 恒定、chunk 按预算燃尽（30/30/4）后并入 decode，批合计恒 ≤ 32 且两拍
恰好打满。

数字全部取自 figure_spec.numbers（精简版 companion host 实测 trace：批合计轨迹
32/32/32/6/3/3/3（拍 1-7；表从拍 2 起的混合视图）；r3 chunk 序列 [30,30,4]（差 34 截到
30、追平 64/64）；拍 5-7 纯 decode 三人各 1；预算 32）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 762
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '稳态批的燃尽表：批一直在换血，token 合计从来没破 32',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '预算 32 下拍 2/3 的批都是 [1,1,30]——同一个 r3 在烧自己的 64-token prompt（差 34 被余额 30 截）；拍 4 尾 chunk 4 收官、拍 5 起三人各恰 1',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ①-⑦ + 下一拍回环 · L0：调度账本列上半'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 六行状态表（拍 2-7） ----------------
HDR_Y = 96
ROW_Y0, ROW_H = 116, 72
TOKPX = 11.0   # r3 票据 px/token（30 → 330px）

lc.text(MX + 23, HDR_Y, '拍', 9.5, lc.C_MUTE, 'middle', True, maxw=40, tag='hd:beat')
lc.text(150, HDR_Y, 'r1', 9.5, lc.C_MUTE, 'middle', True, maxw=60, tag='hd:r1')
lc.text(268, HDR_Y, 'r2', 9.5, lc.C_MUTE, 'middle', True, maxw=60, tag='hd:r2')
lc.text(486, HDR_Y, 'r3（chunk 燃尽轨迹）', 9.5, lc.C_MUTE, 'middle', True, maxw=200, tag='hd:r3')
TOT_X = 940
G_X, G_W = 1040, 24
lc.text(TOT_X, HDR_Y, '批合计', 9.5, lc.C_MUTE, 'middle', True, maxw=80, tag='hd:tot')
lc.text(G_X + G_W / 2, HDR_Y, '预算 32 水位', 9.5, lc.C_MUTE, 'middle', True, maxw=110, tag='hd:gage')
lc.text(1140, HDR_Y, '关键观察', 9.5, lc.C_MUTE, 'start', True, maxw=100, tag='hd:obs')

BEATS = [
    (2, 1, 1, 30, 32, 0, ['RUNNING 先吃 2，新请求 r3 领余 30']),
    (3, 1, 1, 30, 32, 0, ['混合批再次打满——同一形状复现', '（差 34 被余额 30 截）']),
    (4, 1, 1, 4, 6, 26, ['r3 已算 64/64', 'is_prefill_chunk 翻 False']),
    (5, 1, 1, 1, 3, 29, ['纯 decode 稳态：三人各恰 1']),
    (6, 1, 1, 1, 3, 29, ['稳态延续——差恒 1 的闭环自持']),
    (7, 1, 1, 1, 3, 29, ['稳态不变——形状不再变化']),
]

r3_right_edges = []
for bi, (beat, v1, v2, v3, total, left, obs) in enumerate(BEATS):
    ry = ROW_Y0 + bi * ROW_H
    mid = ry + 28
    if bi > 0:
        lc.seg(MX, ry - 4, 1110, ry - 4, '#e2e8f0', 1.0)
    # 拍号徽标
    lc.rect(MX, mid - 13, 46, 26, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.2)
    lc.text(MX + 23, mid + 3.5, f'拍 {beat}', 9.5, lc.C_ENG_S, 'middle', True, maxw=42,
            tag=f'bdg{beat}')
    # r1 / r2：恒窄条 1（decode）
    for cx, val in ((150, v1), (268, v2)):
        lc.rect(cx - 8, mid - 9, 16, 18, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.3)
        lc.text(cx, mid + 32, str(val), 8, lc.C_MUTE, 'middle', maxw=30, tag=f'd{bi}{cx}')
    # r3：宽条 ∝ token（prefill chunk 深青 → decode 浅青）
    x0 = 386
    w = max(16.0, v3 * TOKPX)
    if v3 > 1:
        lc.rect(x0, mid - 9, w, 18, lc.C_KV_S, lc.C_KV_S, rx=3, sw=0)
    else:
        lc.rect(x0, mid - 9, 16, 18, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.3)
    lc.text(x0 + w / 2, mid + 3.5, f'{v3}', 9, '#ffffff' if v3 > 1 else lc.C_KV_S, 'middle',
            True, maxw=60, tag=f'v3{beat}')
    lc.text(x0 + w + 10, mid + 3.5, 'chunk' if v3 > 1 else 'decode', 8, lc.C_MUTE, 'start',
            maxw=60, tag=f'v3k{beat}')
    r3_right_edges.append((x0 + w, mid))
    # 批合计徽章
    hot = (total == 32)
    bw = 54
    lc.rect(TOT_X - bw / 2, mid - 14, bw, 28, lc.C_BEAT_F if hot else '#ffffff',
            lc.C_BEAT_S if hot else lc.C_MUTE, rx=13, sw=1.4 if hot else 1.2)
    lc.text(TOT_X, mid + 3.5, str(total), 11, lc.C_BEAT_T if hot else lc.C_TXT, 'middle',
            True, maxw=44, tag=f'tot{beat}')
    # 预算水位竖条
    gy0, gh = ry + 2, 56
    lc.rect(G_X, gy0, G_W, gh, '#ffffff', lc.C_MUTE, rx=3, sw=1.1)
    fh = gh * total / 32
    if fh > 0.5:
        lc.rect(G_X, gy0 + gh - fh, G_W, fh, lc.C_BEAT_S, lc.C_BEAT_S, rx=2, sw=0)
    lc.text(G_X + G_W + 8, mid + 3.5, f'余 {left}', 8, lc.C_MUTE, 'start', maxw=48,
            tag=f'lft{beat}')
    # 关键观察
    for li, ln in enumerate(obs):
        lc.text(1140, mid - 6 + li * 17 + 6, ln, 8.5, '#334155', 'start', maxw=BXR - 1140,
                tag=f'obs{beat}:{li}')

# r3 燃尽对角虚线（连三拍宽条右缘）
pts = [(r3_right_edges[0][0], r3_right_edges[0][1]),
       (r3_right_edges[1][0], r3_right_edges[1][1]),
       (r3_right_edges[2][0], r3_right_edges[2][1])]
for i in range(len(pts) - 1):
    lc.seg(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], lc.C_BEAT_T, 1.6, dash=True)
lc.text(pts[2][0] + 14, pts[2][1] + 20, 'is_prefill_chunk=False 翻转（拍 4 追平 64/64）', 8.5,
        lc.C_BEAT_T, 'start', True, maxw=250, tag='flip')

# ---------------- 底部：通式 + 图例 + 页脚 ----------------
FRM_Y = ROW_Y0 + 6 * ROW_H + 24
lc.rect(MX, FRM_Y, 1080, 46, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(MX + 14, FRM_Y + 19, '批组成通式：K 个 decode × 1 + Σ chunks ≤ B（预算）——本例 2×1+30 = 32 连续两拍、2×1+4 = 6、'
        '随后 3×1 = 3 永续', 8.7, '#334155', 'start', maxw=1050, tag='frm:l1')
lc.text(MX + 14, FRM_Y + 36, '「连续」二字的机器含义：没有人离场时需要等整批，也没有人进场时需要等空位——每拍批都装着当时正好有活干的每个人',
        8.7, '#334155', 'start', maxw=1050, tag='frm:l2')
lc.text(1160, FRM_Y + 19, 'r3 的 prefill 占 3 拍', 8.5, lc.C_MUTE, 'start', True,
        maxw=BXR - 1160, tag='frm:r1')
lc.text(1160, FRM_Y + 36, '（64 token 按逐拍余额 30/30/4）', 8.5, lc.C_MUTE, 'start',
        maxw=BXR - 1160, tag='frm:r2')

LEG_Y = FRM_Y + 74
lx = MX
items = [
    ('dec', 'decode 份（恰 1）'),
    ('chk', 'prefill chunk 份（宽 ∝ token）'),
    ('gag', '预算水位（本拍合计 / 32）'),
]
for kind, name in items:
    if kind == 'dec':
        lc.rect(lx, LEG_Y - 8, 16, 12, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.3)
    elif kind == 'chk':
        lc.rect(lx, LEG_Y - 8, 20, 12, lc.C_KV_S, lc.C_KV_S, rx=3, sw=0)
    else:
        lc.rect(lx, LEG_Y - 8, 14, 14, lc.C_BEAT_S, lc.C_BEAT_S, rx=3, sw=0)
    lc.text(lx + 26, LEG_Y + 2, name, 8.5, lc.C_TXT, 'start', maxw=280, tag='leg' + kind)
    lx += 26 + lc.tw(name, 8.5) + 22
lc.seg(lx, LEG_Y - 2, lx + 22, LEG_Y - 2, lc.C_BEAT_T, 1.4, dash=True)
lc.text(lx + 28, LEG_Y + 2, '燃尽轨迹', 8.5, lc.C_TXT, 'start', maxw=80, tag='leg:burn')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/sched/scheduler.py:L516-L520（RUNNING 侧续 chunk 同公式）· L1335-L1337（is_prefill_chunk 判定与翻转）· '
        '守恒断言 L1108-L1111', 8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '七拍读数取自精简版 companion host 实测（表取拍 2-7 混合视图；拍 1 为 r1/r2 各 16 全量、合计恰 32）· '
        '32 为示教预算，服务端 8192/16384（arg_utils.py:L2541-L2563）· 行号基线 vLLM v0.27.1', 8.5,
        lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch10-fig-steady-state-burn-down.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
