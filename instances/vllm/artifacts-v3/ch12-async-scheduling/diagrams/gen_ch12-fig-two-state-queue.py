#!/usr/bin/env python3
"""ch12 机制图 2 · 两态心跳的逐拍状态表（figure_spec ch12-fig-two-state-queue，模板 state-table）

放大自 L0 循环框（loop_box）的心跳节拍——即本章 L2 章图 center 两态拍片（①-④ 填管道优先 ·
⑤-⑥ pop 收结果）在『时间 × 队列水位』维度的展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：
图右上角指北小签。

claim：深度 2 的批队列让每次心跳调用『调度一个新批 + 收掉一个旧批』——填管道优先于取输出：
5 次调用走出 2 次真前向 + 剪枝/flush/排空 3 次空拍，批 A 的 D2H 未完成时批 B 已在调度。

数字全部取自 figure_spec.numbers（队列水位逐拍 [] → [A] → [B,A]→pop A → [C,B]→pop B →
[D,C]→pop C → pop D → []；返回值序列 (None, True) / outputs / outputs / outputs（空）/
({}, False)；盲调度实证 拍2 ph=1、输出空、D2H pending=True；交货拍2 [7]、拍3 [9] LENGTH、
early-stop 4≥4 剪出空批C；deque maxlen=2 稳态一进一出）。deque 每行画 appendleft 后、
pop 前的瞬态（被 pop 的槽虚线淡出）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 862
MX, BXR = 60, 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '两态心跳的逐拍状态表：填管道优先，队满（或无活可调）才 pop 收结果',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '深度恰为 2 时每次调用排一个新批的同时收掉一个旧批——拍 2 调度批 B 时批 A 的 D2H 事件还 pending=True：'
        'CPU 调度时间被藏进 GPU 计算时间里',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ①-⑥ 两态 · L0：循环框'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 列几何 ----------------
X_SCHED = 150          # 排入 start
SCHED_W = 245
DEQ_CX = 555           # deque 中心
X_EVID = 700           # 盲调度证据 start
EVID_W = 235
X_POP = 965            # pop 动作 start
POP_W = 250
X_RET = 1300           # 返回 middle
X_FW = 1395            # 前向 middle

# ---------------- 两态分组条 ----------------
GRP_Y = 92
G_UP0, G_UP1 = 140, 945
G_DN0, G_DN1 = 955, 1440
lc.rect(G_UP0, GRP_Y, G_UP1 - G_UP0, 22, lc.C_BEAT_F, lc.C_BEAT_S, rx=5, sw=1.2)
lc.text((G_UP0 + G_UP1) / 2, GRP_Y + 15, '上半段 · 填管道优先（core.py:L652-L687：排入 + 队列水位 + 盲调度证据）',
        9, lc.C_BEAT_T, 'middle', True, maxw=G_UP1 - G_UP0 - 10, tag='grp:up')
lc.rect(G_DN0, GRP_Y, G_DN1 - G_DN0, 22, lc.C_ENG_F, lc.C_ENG_S, rx=5, sw=1.2)
lc.text((G_DN0 + G_DN1) / 2, GRP_Y + 15, '下半段 · pop 收结果（core.py:L689-L739）', 9,
        lc.C_ENG_S, 'middle', True, maxw=G_DN1 - G_DN0 - 10, tag='grp:dn')

HDR_Y = 132
for x, t, anc, mw in [(95, '拍', 'middle', 40), (X_SCHED, '排入（schedule）', 'start', 180),
                      (DEQ_CX, '队列水位（appendleft 后·pop 前）', 'middle', 230),
                      (X_EVID, '盲调度证据', 'start', 120), (X_POP, 'pop 收结果', 'start', 120),
                      (X_RET, '返回', 'middle', 60), (X_FW, '前向', 'middle', 60)]:
    lc.text(x, HDR_Y, t, 9, lc.C_MUTE, anc, True, maxw=mw, tag='hd:' + t[:6])

# ---------------- 逐拍行 ----------------
# 行数据全部取自 m4 e2e trace beats[] 与 schedule_log：
# (拍号, 排入两行, 瞬态槽[index0,index1](None=空), 被pop槽idx, pop字母, 证据两行, pop动作两行, 返回, executed)
BEATS = [
    (1, ['批A {req-0: 2}', '（全量 prefill）'], ['A', None], None, None,
     ['—（首拍无在飞）', ''], ['队未满 → 不进下半段', '（return None 不等结果）'],
     '(None, True)', True),
    (2, ['批B {req-0: 1}', '（追赶公式 2+1−2=1）'], ['B', 'A'], 1, 'A',
     ['ph=1、输出空', '批A D2H 事件 pending=True'], ['pop A → update_from_output', '交货 [7]（t7 到账）'],
     'outputs', True),
    (3, ['批C {}（空批）', '（early-stop 4≥4 剪枝）'], ['C', 'B'], 1, 'B',
     ['ph=1 但确信到顶', '（max_tokens=2 用尽）'], ['pop B 交货 [9]', '→ LENGTH 终态'],
     'outputs', False),
    (4, ['批D {}（空批）', '（flush finished_req_ids）'], ['D', 'C'], 1, 'C',
     ['—（无真调度）', ''], ['pop C（plain future）', '（通知 worker 清缓存）'],
     'outputs（空）', False),
    (5, ['无（has_requests=False）', ''], ['D', None], 0, 'D',
     ['—', ''], ['pop D 排空', '→ has_work()=False'],
     '({}, False)', False),
]
ROW_Y0, ROW_H = 148, 104
DEQ_DY = 26            # deque 顶在行内的 y 偏移
SLOT_W, SLOT_H, SLOT_GAP = 30, 26, 4
DEQ_X = DEQ_CX - (2 * SLOT_W + SLOT_GAP) / 2   # deque 左缘

for bi, (beat, sched, slots, pop_idx, pop_l, evid, act, ret, executed) in enumerate(BEATS):
    ry = ROW_Y0 + bi * ROW_H
    mid = ry + 14
    if bi > 0:
        lc.seg(MX, ry - 6, 1440, ry - 6, '#e2e8f0', 1.0)
    # 拍徽标
    lc.rect(80, mid - 10, 30, 20, lc.C_BADGE_F, lc.C_ENG_S, rx=9, sw=1.1)
    lc.text(95, mid + 3.5, str(beat), 9.5, lc.C_ENG_S, 'middle', True, tag='bdg' + str(beat))
    # 排入
    for j, ln in enumerate(sched):
        if ln:
            lc.text(X_SCHED, ry + 26 + j * 15, ln, 8.6, '#334155', 'start', maxw=SCHED_W,
                    tag='sc' + str(beat) + str(j))
    # 队列水位 deque（瞬态：实槽=留在队里的批；虚线槽=本拍被 pop 走的批）
    dy = ry + DEQ_DY
    for si in range(2):
        sx = DEQ_X + si * (SLOT_W + SLOT_GAP)
        v = slots[si]
        if v is None:
            lc.rect(sx, dy, SLOT_W, SLOT_H, '#ffffff', '#cbd5e1', rx=3, sw=1.0, dash=True)
        elif si == pop_idx:
            lc.rect(sx, dy, SLOT_W, SLOT_H, '#ffffff', lc.C_ENG_S, rx=3, sw=1.1, dash=True)
            lc.text(sx + SLOT_W / 2, dy + 17, v, 10, lc.C_ENG_S, 'middle', True,
                    tag='qp' + str(beat) + str(si))
        else:
            lc.rect(sx, dy, SLOT_W, SLOT_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=3, sw=1.2)
            lc.text(sx + SLOT_W / 2, dy + 17, v, 10, lc.C_BEAT_T, 'middle', True,
                    tag='qs' + str(beat) + str(si))
    # appendleft 标签（左进）与 pop 标签（右出）
    new_l = slots[0] if (bi < 4 and sched and not sched[0].startswith('无')) else None
    if beat == 1:
        lc.text(DEQ_X - 8, dy + SLOT_H + 14, 'appendleft A', 8.2, lc.C_BEAT_T, 'end', maxw=110,
                tag='pushlbl' + str(beat))
    elif new_l:
        lc.text(DEQ_X - 8, dy + SLOT_H + 14, 'appendleft ' + new_l, 8.2, lc.C_BEAT_T, 'end',
                maxw=110, tag='pushlbl' + str(beat))
    if pop_l:
        px = DEQ_X + pop_idx * (SLOT_W + SLOT_GAP) + SLOT_W / 2
        lc.seg(px, dy + SLOT_H + 2, px, dy + SLOT_H + 10, lc.C_ENG_S, 1.4)
        lc.text(px, dy + SLOT_H + 20, 'pop ' + pop_l, 8.2, lc.C_ENG_S, 'middle', maxw=70,
                tag='poplbl' + str(beat))
    # 盲调度证据
    for j, ln in enumerate(evid):
        if ln:
            col = lc.C_ENG_S if (beat == 2 and j == 1) else '#334155'
            lc.text(X_EVID, ry + 26 + j * 15, ln, 8.6, col, 'start', maxw=EVID_W,
                    tag='ev' + str(beat) + str(j))
    # pop 动作
    for j, ln in enumerate(act):
        if ln:
            lc.text(X_POP, ry + 26 + j * 15, ln, 8.6, '#334155', 'start', maxw=POP_W,
                    tag='ac' + str(beat) + str(j))
    # 返回
    lc.text(X_RET, ry + 34, ret, 8.8, lc.C_TXT, 'middle', True, maxw=100, tag='ret' + str(beat))
    # executed 徽标
    if executed:
        lc.rect(X_FW - 19, ry + 26, 38, 16, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.1)
        lc.text(X_FW, ry + 37.5, '前向', 8, lc.C_GPU_S, 'middle', True, maxw=32, tag='fw' + str(beat))
    else:
        lc.text(X_FW, ry + 37.5, '空拍', 8, lc.C_MUTE, 'middle', maxw=44, tag='nfw' + str(beat))

lc.text(DEQ_CX, ROW_Y0 + 5 * ROW_H + 4, 'deque(maxlen=2)：appendleft 进 index0（左端）· pop 出末尾（右端，最老批）——虚线槽 = 本拍被 pop 走的批',
        8.4, lc.C_MUTE, 'middle', maxw=560, tag='deqnote')

# ---------------- 双缓冲结论横幅 ----------------
BN_Y = 700
lc.rect(MX, BN_Y, 1380, 38, lc.C_BEAT_F, lc.C_BEAT_S, rx=7, sw=1.4)
lc.text(MX + 690, BN_Y + 16.5, '深度 2 = 双缓冲：稳态每拍一进一出（appendleft 恰一次 + pop 恰一次）——返回值序列 '
        '(None, True) / outputs / outputs / outputs（空） / ({}, False)',
        10, lc.C_BEAT_T, 'middle', True, maxw=1360, tag='banner1')
lc.text(MX + 690, BN_Y + 30.5, '批 B 在 CPU 排队入场的那段时间，批 A 正在 GPU 上算——5 次调用 = 2 次真前向（批A/批B）+ 剪枝空拍 + flush 空拍 + 排空拍（同步版串行脊柱见 ch9 图）',
        9, lc.C_BEAT_T, 'middle', True, maxw=1360, tag='banner2')

# ---------------- 终态注记 ----------------
TN_Y = 760
lc.rect(MX, TN_Y, 1380, 34, '#ffffff', lc.C_MUTE, rx=7, sw=1.2, dash=True)
lc.text(MX + 16, TN_Y + 15, '终止性：每请求 num_output_tokens 每交货一拍 +1 且上界 max_tokens → FINISHED → 下一次 schedule 以空批 flush（拍 4）→ 队列 FIFO 排空（拍 5）→ has_work()=False 引擎静止',
        8.6, '#334155', 'start', maxw=1340, tag='term1')
lc.text(MX + 16, TN_Y + 28, '每拍账：computed/ph = 2/1 → 3/1 → 3/0 → 3/0 → 3/0（真实已算 computed−ph = 1→2→3→3→3，占位账本见下页图）',
        8.6, lc.C_MUTE, 'start', maxw=1340, tag='term2')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = 826
lx = MX
for kind, name in [('beat', '批槽（appendleft 后仍在队的批）'), ('pop', '本拍被 pop 走的批（虚线）'),
                   ('empty', '空槽'), ('fw', '真前向拍'), ('updn', '上半段/下半段分组')]:
    if kind == 'beat':
        lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_BEAT_F, lc.C_BEAT_S, rx=3, sw=1.2)
    elif kind == 'pop':
        lc.rect(lx, LEG_Y - 9, 20, 12, '#ffffff', lc.C_ENG_S, rx=3, sw=1.1, dash=True)
    elif kind == 'empty':
        lc.rect(lx, LEG_Y - 9, 20, 12, '#ffffff', '#cbd5e1', rx=3, sw=1.0, dash=True)
    elif kind == 'fw':
        lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.1)
    else:
        lc.seg(lx, LEG_Y - 3, lx + 12, LEG_Y - 3, lc.C_BEAT_S, 2.2)
        lc.seg(lx + 12, LEG_Y - 3, lx + 24, LEG_Y - 3, lc.C_ENG_S, 2.2)
    lc.text(lx + 26, LEG_Y + 1, name, 8.5, lc.C_TXT, 'start', maxw=280, tag='leg' + kind)
    lx += 26 + lc.tw(name, 8.5) + 18

lc.text(MX, 850, '逐字锚 vllm/v1/engine/core.py:L625-L739（step_with_batch_queue）/ L681（appendleft 三元组）/ '
        'L696（pop 最老批）· AsyncScheduler.schedule（async_scheduler.py）· 水位/返回值/盲调度证据取自配套精简版 host 实跑'
        '（prompt=2、max_tokens=2、无 spec）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch12-fig-two-state-queue.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
