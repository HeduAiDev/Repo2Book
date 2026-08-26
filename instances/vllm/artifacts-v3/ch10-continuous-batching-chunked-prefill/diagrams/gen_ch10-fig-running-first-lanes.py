#!/usr/bin/env python3
"""ch10 机制图 4 · RUNNING 先行双泳道（figure_spec ch10-fig-running-first-lanes，模板 swimlane）

放大自 L0『调度 · 显存账本』（kv_column 青色列）上半 Scheduler 框的「RUNNING 先行」格——
即本章 L2 章图 center ① 拍片『First, schedule the RUNNING requests』与 ③ 拍片之间
顺序关系的机制展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：allocate 实测顺序：拍 2 里 r1→r2→r3 三个 decode 先各领 1（预算 16 掉到 13），
新到的 r4 只能领 13——RUNNING 先行使在途请求的 1 token 永远先入账。

数字全部取自 figure_spec.numbers（精简版 companion host 实测 trace：拍 2 分配顺序
[r1:1, r2:1, r3:1, r4:13]；r4 需求 20 被截到 13（65%）；r4 chunk 序列 [13,7,1,1]；
预算 16，拍 2 合计 16 恰打满、拍 3 合计 10）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 622
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'RUNNING 先行：在途 decode 的 1 永远先入账，新请求的 chunk 用余额分期付',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '「First, schedule the RUNNING requests.」（scheduler.py:L483-L485）——拍 2 三个 decode 先各领 1（预算 16→13），'
        '20-token 新 prompt 首拍只领到 13（65%）',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ①→③ 两阶段顺序 · L0：调度账本列上半'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 泳道几何 ----------------
LANE_X0, LANE_X1 = 210.0, 1420.0
RUN_Y, RUN_H = 130, 150          # 上泳道 RUNNING
WAIT_Y, WAIT_H = 330, 118        # 下泳道 WAITING
RULER_Y = RUN_Y + RUN_H + 20     # 泳道间共享预算标尺
N_BEATS = 4
COL_PAD = 26
colw = (LANE_X1 - LANE_X0 - 2 * COL_PAD) / N_BEATS
col_x = [LANE_X0 + COL_PAD + i * colw for i in range(N_BEATS)]   # 每拍列左缘

def beat_hdr(i):
    cx = col_x[i] + colw / 2
    lc.rect(cx - 26, 100, 52, 24, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.2)
    lc.text(cx, 116, f'拍 {i + 1}', 10, lc.C_ENG_S, 'middle', True, maxw=46, tag=f'bdg{i}')

for i in range(N_BEATS):
    beat_hdr(i)
    if i < N_BEATS - 1:
        lc.seg(col_x[i] + colw - 4, 104, col_x[i] + colw + 4, 104, '#e2e8f0', 1.0)

# 泳道底板与泳道头
lc.rect(MX, RUN_Y, 130, RUN_H, '#ffffff', lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 65, RUN_Y + 62, '上泳道', 10.5, lc.C_KV_S, 'middle', True, maxw=110, tag='lane:r1')
lc.text(MX + 65, RUN_Y + 82, 'RUNNING', 10.5, lc.C_KV_S, 'middle', True, maxw=110, tag='lane:r2')
lc.text(MX + 65, RUN_Y + 102, '（在途请求）', 8.5, lc.C_MUTE, 'middle', maxw=110, tag='lane:r3')
lc.rect(LANE_X0, RUN_Y, LANE_X1 - LANE_X0, RUN_H, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.2)
lc.rect(MX, WAIT_Y, 130, WAIT_H, '#ffffff', lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 65, WAIT_Y + 42, '下泳道', 10.5, lc.C_KV_S, 'middle', True, maxw=110, tag='lane:w1')
lc.text(MX + 65, WAIT_Y + 62, 'WAITING', 10.5, lc.C_KV_S, 'middle', True, maxw=110, tag='lane:w2')
lc.text(MX + 65, WAIT_Y + 82, '（新请求）', 8.5, lc.C_MUTE, 'middle', maxw=110, tag='lane:w3')
lc.rect(LANE_X0, WAIT_Y, LANE_X1 - LANE_X0, WAIT_H, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.2)

TOKPX_RUN = 13.0   # 上泳道票据 px/token（4-token 票 = 52px）
TOKPX_WAIT = 15.0  # 下泳道票据 px/token（13-token 票 = 195px）

def run_ticket(i, rid, n, y, note=''):
    """上泳道票据：宽 ∝ n，深青。x 从列左缘起。窄票据（n=1）标签放右侧。"""
    x0 = col_x[i] + 6
    w = max(14.0, n * TOKPX_RUN)
    lc.rect(x0, y, w, 22, lc.C_KV_S, lc.C_KV_S, rx=3, sw=0)
    lab = f'{rid}:{n}'
    if lc.tw(lab, 8.5, True) + 8 <= w:
        lc.text(x0 + w / 2, y + 15, lab, 8.5, '#ffffff', 'middle', True, maxw=w - 4,
                tag=f'rt{i}{rid}')
    else:
        lc.text(x0 + w + 5, y + 15, lab, 8.5, lc.C_KV_S, 'start', True, maxw=70,
                tag=f'rt{i}{rid}')
    if note:
        lc.text(x0 + w / 2, y + 38, note, 8, lc.C_MUTE, 'middle', maxw=colw - 12, tag=f'rtn{i}{rid}')
    return x0 + w

def wait_ticket(i, rid, n, y, note=''):
    x0 = col_x[i] + 6
    w = max(14.0, n * TOKPX_WAIT)
    lc.rect(x0, y, w, 22, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.4)
    lab = f'{rid}:{n}'
    if lc.tw(lab, 8.5, True) + 8 <= w:
        lc.text(x0 + w / 2, y + 15, lab, 8.5, lc.C_KV_S, 'middle', True, maxw=w - 4,
                tag=f'wt{i}{rid}')
    else:
        lc.text(x0 + w + 5, y + 15, lab, 8.5, lc.C_KV_S, 'start', True, maxw=70,
                tag=f'wt{i}{rid}')
    if note:
        lc.text(x0 + w / 2, y + 38, note, 8, lc.C_MUTE, 'middle', maxw=colw - 12, tag=f'wtn{i}{rid}')
    return x0 + w

# ---- 拍 1：running 空，三个新 prompt 全量 4 走 WAITING ----
lc.text(col_x[0] + colw / 2, RUN_Y + 34, 'running 空——', 8.5, lc.C_FAINT, 'middle', maxw=colw - 14,
        tag='e1')
lc.text(col_x[0] + colw / 2, RUN_Y + 52, '本拍没有在途请求', 8.5, lc.C_FAINT, 'middle', maxw=colw - 14,
        tag='e2')
for k, rid in enumerate(['r1', 'r2', 'r3']):
    wait_ticket(0, rid, 4, WAIT_Y + 16 + k * 30)

# ---- 拍 2：RUNNING 三个 decode 各 1 → WAITING r4 首 chunk 13 ----
for k, rid in enumerate(['r1', 'r2', 'r3']):
    run_ticket(1, rid, 1, RUN_Y + 16 + k * 30)
lc.text(col_x[1] + 6 + 60, RUN_Y + 118, '先付 3（每拍每人恰 1）', 8, lc.C_MUTE, 'start',
        maxw=150, tag='p2r')
wait_ticket(1, 'r4', 13, WAIT_Y + 26, '首 chunk：需求 20 截到余 13（65%）')

# ---- 拍 3：RUNNING 三个 decode 各 1 → WAITING r4 尾 chunk 7 ----
for k, rid in enumerate(['r1', 'r2', 'r3']):
    run_ticket(2, rid, 1, RUN_Y + 16 + k * 30)
lc.text(col_x[2] + 6 + 60, RUN_Y + 118, '又先付 3', 8, lc.C_MUTE, 'start', maxw=110, tag='p3r')
wait_ticket(2, 'r4', 7, WAIT_Y + 26, '尾 chunk 7——prefill 收官')

# ---- 拍 4：r4 转 decode，四人同拍各 1（上楼） ----
for k, rid in enumerate(['r1', 'r2', 'r3', 'r4']):
    run_ticket(3, rid, 1, RUN_Y + 16 + k * 30)
lc.text(col_x[3] + 6, RUN_Y + 144, 'r4 转 decode，加入在途（此后每拍恰 1）', 8, lc.C_MUTE,
        'start', maxw=190, tag='p4r')
lc.text(col_x[3] + colw / 2, WAIT_Y + 40, '——（waiting 空）', 8.5, lc.C_FAINT, 'middle',
        maxw=colw - 14, tag='e3')

# r4 上楼箭头：拍 3 下泳道 r4 票右缘 → 拍 3/4 列缝（穿标尺间隙）→ 拍 4 上泳道 r4 票左缘
_src = (col_x[2] + 6 + 7 * TOKPX_WAIT, WAIT_Y + 26 + 11)
_gx = col_x[2] + colw - 5               # 列缝 x（标尺条间 10px 空隙）
_dy = RUN_Y + 16 + 3 * 30 + 11          # 拍 4 r4 票纵中
_dst = (col_x[3] + 6, _dy)
lc.parrow([_src, (_gx, _src[1]), (_gx, _dy), _dst], lc.C_ENG_S, 1.6, 'up')
lc.text(_gx - 6, RULER_Y - 22, '转 decode 上楼', 8, lc.C_ENG_S, 'end', True, maxw=80, tag='up:lbl')

# ---------------- 泳道间共享预算标尺（0..16，每拍一格） ----------------
BAR_H = 20
lc.text(MX + 65, RULER_Y - 12, '共享预算 16', 9.5, lc.C_BEAT_T, 'middle', True, maxw=110, tag='rl:hd')
lc.seg(LANE_X0, RULER_Y - 4, LANE_X1, RULER_Y - 4, '#e2e8f0', 1.0)
RUN_FIRST = [False, True, True, True]
TOTALS = [12, 16, 10, 4]
RUNPAY = [0, 3, 3, 3]
for i in range(N_BEATS):
    bx0 = col_x[i]
    bw = colw - 10
    lc.rect(bx0, RULER_Y, bw, BAR_H, '#ffffff', lc.C_MUTE, rx=3, sw=1.1)
    fw = bw * TOTALS[i] / 16
    if fw > 0.5:
        lc.rect(bx0, RULER_Y, fw, BAR_H, lc.C_BEAT_S, lc.C_BEAT_S, rx=2, sw=0)
    if RUN_FIRST[i]:
        cut = bw * RUNPAY[i] / 16
        lc.seg(bx0 + cut, RULER_Y - 6, bx0 + cut, RULER_Y + BAR_H + 6, lc.C_ABORT, 1.6)
        lc.text(bx0 + cut, RULER_Y + BAR_H + 18, '先付 3', 8, lc.C_ABORT, 'middle', True,
                maxw=60, tag=f'cut{i}')
    lc.text(bx0 + bw / 2, RULER_Y - 8, f'合计 {TOTALS[i]}' + ('（打满）' if TOTALS[i] == 16 else ''),
            8.5, lc.C_BEAT_T if TOTALS[i] == 16 else lc.C_MUTE, 'middle', maxw=110, tag=f'tot{i}')

# ---------------- why 注（虚线框） ----------------
WHY_Y = WAIT_Y + WAIT_H + 24
lc.rect(MX, WHY_Y, 880, 62, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(MX + 14, WHY_Y + 18, '顺序 = TPOT 优先于 TTFT（L0 图『why · TTFT↔TPOT 交易』的机制面）', 9.5,
        lc.C_TXT, 'start', True, maxw=850, tag='why:t')
lc.text(MX + 14, WHY_Y + 36, '· 只要预算 ≥ 在场 decode 数且差距 ≥ 1，每个在途 decode 每拍必领 1——新请求只能压缩自己的 chunk，'
        '不能压缩 decode 的 1', 8.5, '#334155', 'start', maxw=850, tag='why:l1')
lc.text(MX + 14, WHY_Y + 53, '· 代价：老请求绝对优先 → waiting 无 admission control 无上界增长，TTFT 无上界；抢占拍连收新都暂停（下一图）',
        8.5, '#334155', 'start', maxw=850, tag='why:l2')

# 反事实算术（右下）
CF_X = 980
lc.text(CF_X, WHY_Y + 18, '反事实（若顺序倒置）', 9.5, lc.C_TXT, 'start', True, maxw=180, tag='cf:t')
lc.text(CF_X, WHY_Y + 36, 'r4 先领 min(20,16)=16 → 余额 0', 8.5, '#334155', 'start',
        maxw=BXR - CF_X, tag='cf:l1')
lc.text(CF_X, WHY_Y + 53, '三个 decode 落空一拍，TPOT 出尖峰（算术推演，非 trace）', 8.5,
        '#334155', 'start', maxw=BXR - CF_X, tag='cf:l2')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = WHY_Y + 84
lx = MX
items = [
    ('run', 'RUNNING 泳道票据（decode 恰 1 / 续 chunk）'),
    ('wait', 'WAITING 泳道票据（新请求 chunk）'),
    ('gag', '共享预算标尺（0..16，本拍合计）'),
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
lc.seg(lx, LEG_Y - 2, lx + 20, LEG_Y - 2, lc.C_ABORT, 1.5)
lc.text(lx + 26, LEG_Y + 2, '「先付」分割线', 8.5, lc.C_TXT, 'start', maxw=110, tag='leg:cut')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/sched/scheduler.py:L483-L485（RUNNING 先行注释）· L523+L636-L637（decode 领 1 入账）· '
        'L1069-L1075（WAITING 侧 r4 首 chunk 被余额截断）', 8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '分配顺序与票据读数取自精简版 companion host 实测（allocate_slots 调用代理记录：拍 2 [r1:1, r2:1, r3:1, r4:13]；'
        '65% = 13/20 为驱动脚本记录字段）· 行号基线 vLLM v0.27.1', 8.5, lc.C_FAINT, 'start',
        maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch10-fig-running-first-lanes.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
