#!/usr/bin/env python3
"""ch11 机制图 2 · FCFS 抢占的受害者选择（figure_spec ch11-fig-victim-youngest，模板 before-after）

放大自 L0 右列『调度 · 显存账本』（kv_column 青色列）上半 Scheduler 框——即本章 L2 章图
center ③ 抢谁 拍片的机制展开；非新架构画法，架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：FCFS 抢占的受害者恒是 running 队尾（最年轻者）：本拍最大占用者 r1（3 块）与触发
分配失败的 r2 都不被选，被弹走的是同尺寸但最晚入列的 r3。

数字全部取自 figure_spec.numbers（before running=[r1,r2,r3] 持块 2/1/1 / after [r1,r2]
持块 3/2/0 被抢 0/0/1 / arrival 1002.0 / PRIORITY 对照源码引文），源出配套精简版 host 实跑 trace
（5 块池 × block_size 16；r1=32-token、r2/r3=16-token）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 788
MX, BXR = 60, 1440

EXTRA_DEFS = ('<defs>'
              '<marker id="kvm" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
              f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_KV_S}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, '抢谁不看占桌多少、也不看是谁触发的——FCFS 恒弹队尾：最晚入列的 r3 让位',
        16.5, lc.C_TXT, 'start', True, maxw=1030, tag='title')
lc.text(MX, 58, 'self.running.pop()（scheduler.py:L615）只看列表位置：对 FCFS 公平序破坏最小、已投入的重算成本通常也最少——选序即策略',
        10.5, lc.C_MUTE, 'start', maxw=1040, tag='subtitle')
_ch = '放大自 L2 拍片 ③ 抢谁 · L0：调度·显存账本列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_KV_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 面板几何 ----------------
PANEL_W = 650
LP_X, RP_X = MX, 790                  # 左右面板左缘
STRIP_Y, CELL_H = 176, 54
UNIT, GAP = 100, 8                    # 1 块 = 100px

def panel(x, head, sub):
    lc.rect(x, 96, PANEL_W, 400, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
    lc.rect(x, 96, PANEL_W, 26, lc.C_KV_F, lc.C_KV_S, rx=9, sw=1.2)
    lc.text(x + 14, 114, head, 11, lc.C_KV_S, 'start', True, maxw=PANEL_W - 28, tag='p:' + head[:6])
    lc.text(x + PANEL_W - 14, 114, sub, 8.8, lc.C_MUTE, 'end', maxw=330, tag='ps:' + head[:6])

panel(LP_X, '拍 1 · 抢占前（首拍全准入）', 'running 序 = 入列序 = FCFS 序')
panel(RP_X, '拍 2 · 抢占后', '触发者 r2 需第 2 块 → None → 弹队尾')

# running 序方向标（左面板顶部内）
lc.text(LP_X + 16, 156, 'running 序：队头（等得最久）→ 队尾（最年轻）', 9, lc.C_MUTE, 'start', tag='dir1')

# ---------------- 左面板：拍 1 条 ----------------
BEFORE = [('r1', 2, '最大占用者', lc.C_KV_S), ('r2', 1, '触发者（拍 2）', lc.C_KV_S), ('r3', 1, '最年轻 · 被弹走', lc.C_ABORT)]
bx = LP_X + 46
for name, nblk, tagtxt, col in BEFORE:
    w = nblk * UNIT
    lc.rect(bx, STRIP_Y, w, CELL_H, lc.C_KV_F if col == lc.C_KV_S else '#fff7ed', col, rx=6, sw=1.8)
    lc.text(bx + w / 2, STRIP_Y + 23, name, 13, lc.C_TXT, 'middle', True, tag='b:' + name)
    lc.text(bx + w / 2, STRIP_Y + 42, f'{nblk} 块', 9.5, col, 'middle', tag='bn:' + name)
    # 角标
    bw = lc.tw(tagtxt, 8.4, True) + 12
    lc.rect(bx + w / 2 - bw / 2, STRIP_Y + CELL_H + 10, bw, 18, lc.C_BADGE_F, col, rx=9, sw=1.0)
    lc.text(bx + w / 2, STRIP_Y + CELL_H + 22.5, tagtxt, 8.4, col, 'middle', True, maxw=bw - 4,
            tag='bt:' + name)
    bx += w + GAP
# arrival 行
lc.text(LP_X + 16, STRIP_Y + CELL_H + 56, 'arrival_time（入列时刻）', 9, lc.C_MUTE, 'start', tag='arr:t')
bx = LP_X + 46
for name, at in [('r1', '1000.0'), ('r2', '1001.0'), ('r3', '1002.0')]:
    w = dict((('r1', 2), ('r2', 1), ('r3', 1)))[name] * UNIT
    col = lc.C_ABORT if name == 'r3' else '#334155'
    lc.text(bx + w / 2, STRIP_Y + CELL_H + 78, at, 10, col, 'middle', True, tag='at:' + name)
    bx += w + GAP
lc.text(LP_X + 16, STRIP_Y + CELL_H + 104, '三者中 r3 最晚到达（1002.0）——选择器只看这个序，不看谁占块多',
        9, '#334155', 'start', maxw=PANEL_W - 32, tag='l-note1')
lc.text(LP_X + 16, STRIP_Y + CELL_H + 122, '持块 r1/r2/r3 = 2/1/1；被抢次数 = 0/0/0；池 5 块用 4 空闲 1',
        9, lc.C_MUTE, 'start', maxw=PANEL_W - 32, tag='l-note2')

# ---------------- 中间 pop 箭头：左 r3 格 → 右 r3 虚影格 ----------------
POP_Y = STRIP_Y + CELL_H / 2
L_R3_X1 = LP_X + 46 + 2 * UNIT + 3 * GAP + UNIT          # r3 格右缘 = 522
lc.parrow([(L_R3_X1 + 4, POP_Y), (620, POP_Y), (620, 330), (RP_X + 330, 330)], lc.C_ABORT, 2.2, 'ab')
lc.text(870, 316, 'self.running.pop() 恒取队尾（L615）', 10, lc.C_ABORT, 'middle', True, maxw=300, tag='pop1')
lc.text(870, 350, '与谁触发、谁占块多均无关', 9, lc.C_MUTE, 'middle', tag='pop2')

# ---------------- 右面板：拍 2 条 ----------------
AFTER = [('r1', 3, '最大占用者毫发无损', lc.C_KV_S), ('r2', 2, '触发者也不被选', lc.C_KV_S)]
bx = RP_X + 40
for name, nblk, tagtxt, col in AFTER:
    w = nblk * UNIT
    lc.rect(bx, STRIP_Y, w, CELL_H, lc.C_KV_F, col, rx=6, sw=1.8)
    lc.text(bx + w / 2, STRIP_Y + 23, name, 13, lc.C_TXT, 'middle', True, tag='a:' + name)
    lc.text(bx + w / 2, STRIP_Y + 42, f'{nblk} 块', 9.5, col, 'middle', tag='an:' + name)
    lc.text(bx + 12, STRIP_Y + 70, tagtxt, 9, col, 'start', maxw=w - 20, tag='at2:' + name)
    bx += w + GAP
# r3 虚影格（被弹走）
GH_X, GH_Y = RP_X + 330, 306
lc.rect(GH_X, GH_Y, UNIT, CELL_H, '#ffffff', lc.C_ABORT, rx=6, sw=1.6, dash=True)
lc.text(GH_X + UNIT / 2, GH_Y + 23, 'r3', 13, lc.C_ABORT, 'middle', True, tag='g:r3')
lc.text(GH_X + UNIT / 2, GH_Y + 42, '0 块', 9.5, lc.C_ABORT, 'middle', tag='gn:r3')
lc.text(GH_X + UNIT / 2, GH_Y + CELL_H + 16, '被弹走 · PREEMPTED', 9, lc.C_ABORT, 'middle', True,
        maxw=200, tag='gt:r3')
lc.text(GH_X + UNIT / 2, GH_Y + CELL_H + 32, '被抢次数 0/0/1', 9, lc.C_MUTE, 'middle', tag='gc:r3')
# r3 → waiting 队头（ghost 右缘 → 队列框左缘，同一水平）
WQ_X, WQ_Y, WQ_W, WQ_H = RP_X + 470, 306, 150, 54
lc.seg(GH_X + UNIT, GH_Y + CELL_H / 2, WQ_X, WQ_Y + CELL_H / 2, lc.C_ENG_S, 2.0, 'up')
lc.rect(WQ_X, WQ_Y, WQ_W, WQ_H, '#ffffff', lc.C_ENG_S, rx=7, sw=1.4)
lc.text(WQ_X + WQ_W / 2, WQ_Y + 22, 'waiting 队列', 9.5, lc.C_TXT, 'middle', True, tag='wq:t')
lc.text(WQ_X + WQ_W / 2, WQ_Y + 41, '[r3]（prepend 回队头）', 9, lc.C_ENG_S, 'middle', maxw=WQ_W - 12,
        tag='wq:l')
# 右面板底部注记
lc.text(RP_X + 16, 470, '持块 r1/r2/r3 = 3/2/0：r1 从 2 块涨到 3 块、r2 涨到 2 块——', 9, '#334155',
        'start', maxw=PANEL_W - 32, tag='r-note1')
lc.text(RP_X + 16, 488, '触发者 r2 抢完队尾后原样重试成功 (r2,1)→OK；池空闲 1→0', 9, lc.C_MUTE,
        'start', maxw=PANEL_W - 32, tag='r-note2')

# ---------------- 底部：判据 + PRIORITY 对照 ----------------
BB_Y, BB_H = 520, 116
BB1_W = 640
lc.rect(MX, BB_Y, BB1_W, BB_H, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.4)
lc.text(MX + 16, BB_Y + 24, '选择判据 = 列表位置（scheduler.py:L615）', 11, lc.C_KV_S, 'start', True,
        maxw=BB1_W - 32, tag='bb1:t')
for i, ln in enumerate(['· 选择器是 self.running.pop()（deque 尾）：输入只有列表位置，',
                        '  列表位置由入列顺序 append 决定且环内无 append',
                        '· 与『谁触发分配失败』无关、与『谁持块多』无关',
                        '· 等得最久的队头永不被此环选中（除非全场只剩它自己）']):
    lc.text(MX + 16, BB_Y + 46 + i * 17, ln, 9, '#334155', 'start', maxw=BB1_W - 30, tag='bb1:l' + str(i))

BB2_X = MX + BB1_W + 24
BB2_W = BXR - BB2_X
lc.rect(BB2_X, BB_Y, BB2_W, BB_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.4, dash=True)
lc.text(BB2_X + 16, BB_Y + 24, '对照 · PRIORITY 策略（L588-L613 源码引文，精简版删不可运行）', 11, lc.C_TXT,
        'start', True, maxw=BB2_W - 32, tag='bb2:t')
for i, ln in enumerate(['victim = max(running, key=(priority, arrival_time))，且要把被抢者',
                        '本拍已领的 token/块/预算全部回滚——同一 while True 环的',
                        '另一种『最不应保留』的定义（随精简版删除、不可运行）']):
    lc.text(BB2_X + 16, BB_Y + 46 + i * 17, ln, 9, '#334155', 'start', maxw=BB2_W - 30, tag='bb2:l' + str(i))

# ---------------- 图例 + 页脚 ----------------
LEG_Y = 678
lx = MX
for kind, name in [('kv', 'running 成员（青=安全留下）'), ('bad', '受害者（红）'), ('ghost', '被弹走（虚线）'),
                   ('eng', '回 waiting 队头')]:
    if kind == 'kv':
        lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.4)
    elif kind == 'bad':
        lc.rect(lx, LEG_Y - 9, 20, 12, '#fff7ed', lc.C_ABORT, rx=3, sw=1.4)
    elif kind == 'ghost':
        lc.rect(lx, LEG_Y - 9, 20, 12, '#ffffff', lc.C_ABORT, rx=3, sw=1.2, dash=True)
    else:
        lc.seg(lx, LEG_Y - 3, lx + 18, LEG_Y - 3, lc.C_ENG_S, 2.0)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=220, tag='leg' + kind)
    lx += 26 + lc.tw(name, 8.8) + 20
lc.text(BXR, LEG_Y + 1, '格宽 ∝ 持块数（1 块 = 100px）', 8.8, lc.C_MUTE, 'end', tag='leg-scale')

lc.text(MX, 706, '逐字锚 vllm/v1/core/sched/scheduler.py:L588-L615（FCFS pop 队尾 / PRIORITY max+回滚）· '
        '持块/被抢次数/arrival 取自配套精简版 host 实跑（5 块池 × block_size 16，r1=32-token、r2/r3=16-token，r3 arrival=1002.0）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')
lc.text(MX, 728, '读图：左（拍 1）三请求同台；右（拍 2）r2 触发分配失败，弹走的却是队尾 r3——不是 r1（占块最多）也不是 r2（触发者本人）',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch11-fig-victim-youngest.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
