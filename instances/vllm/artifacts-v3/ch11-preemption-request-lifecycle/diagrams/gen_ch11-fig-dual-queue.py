#!/usr/bin/env python3
"""ch11 机制图 5 · 双队列防队头阻塞（figure_spec ch11-fig-dual-queue，模板 layout）

放大自 L0 右列『调度 · 显存账本』（kv_column 青色列）上半 Scheduler 框——即本章 L2 章图
north 行三容器（running/waiting/skipped_waiting）+ center ⑥ 双队列遍历拍片的机制展开；
非新架构画法，架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：双队列把『等外部事件』与『马上能调度』隔离开：阻塞队头每拍只花一次 peek 就被跳过，
绝不堵死 ready 请求；本拍跳过者步末按跳过序插回 skipped 队头，下轮最先重试。

数字全部取自 figure_spec.numbers（初始 skipped=[older]、waiting=[newer,ready]；拍 1 ready 准入
{ready:16}、步末 skipped=[older,newer]；拍 2 older 再跳过、newer resumed=[newer]），
源出配套精简版 host 实跑 trace（older=阻塞态、newer=PREEMPTED+stale=1、ready=普通 WAITING）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 908
MX, BXR = 60, 1440

EXTRA_DEFS = ('<defs>'
              '<marker id="okg" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
              f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_GPU_S}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, '双队列：把『等外部事件』与『马上能调度』隔离开——队头阻塞只花一次 peek，绝不堵死 ready',
        16.5, lc.C_TXT, 'start', True, maxw=1030, tag='title')
lc.text(MX, 58, '每拍先看阻塞队头好了没（O(1) peek，没好就跳过）；本拍跳过者步末按跳过序插回 skipped 队头——被跳过的反而是下轮最先重试的，不饿死',
        10.5, lc.C_MUTE, 'start', maxw=1040, tag='subtitle')
_ch = '放大自 L2 拍片 ⑥ + north 三容器 · L0：调度·显存账本列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_KV_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 上方：两条队列 ----------------
Q_Y, Q_H = 96, 150
SK_X, SK_W = 100, 560                       # skipped_waiting
WT_X, WT_W = 720, 560                       # waiting
CELL_W, CELL_H = 118, 62

def queue_frame(x, w, name, sub):
    lc.rect(x, Q_Y, w, Q_H, '#ffffff', lc.C_MUTE, rx=9, sw=1.3)
    lc.text(x + 14, Q_Y + 22, name, 11, lc.C_TXT, 'start', True, maxw=w - 260, tag='q:' + name[:6])
    lc.text(x + w - 14, Q_Y + 22, sub, 8.8, lc.C_MUTE, 'end', maxw=360, tag='qs:' + name[:6])
    lc.text(x + 14, Q_Y + Q_H - 10, '队头（peek 从这里）→ 队尾', 8.4, lc.C_FAINT, 'start', tag='qdir:' + name[:6])

queue_frame(SK_X, SK_W, 'skipped_waiting（等外部事件 · 隔离区）', '阻塞子态被路由进这里')
queue_frame(WT_X, WT_W, 'waiting（马上能调度）', '普通 WAITING / PREEMPTED 恢复者')

def qcell(x, y, rid, l1, l2, fill, stroke):
    lc.rect(x, y, CELL_W, CELL_H, fill, stroke, rx=6, sw=1.6)
    lc.text(x + CELL_W / 2, y + 22, rid, 12, lc.C_TXT, 'middle', True, tag='qc:' + rid)
    lc.text(x + CELL_W / 2, y + 40, l1, 8, '#334155', 'middle', maxw=CELL_W - 8, tag='qc1:' + rid)
    lc.text(x + CELL_W / 2, y + 53, l2, 8, '#334155', 'middle', maxw=CELL_W - 8, tag='qc2:' + rid)

CY = Q_Y + 44
OLDER_CX = SK_X + 24 + CELL_W / 2                       # 183
NEWER_CX = WT_X + 24 + CELL_W / 2                       # 803
qcell(SK_X + 24, CY, 'older', 'WAITING_FOR_STRUCTURED', 'OUTPUT_GRAMMAR · 阻塞', '#f1f5f9', lc.C_MUTE)
lc.text(SK_X + 24 + CELL_W + 18, CY + CELL_H / 2, '（每拍恰 peek 一次）', 8.4, lc.C_MUTE, 'start', tag='peek1')
qcell(WT_X + 24, CY, 'newer', 'PREEMPTED · stale=1', '在途未排干（async 模拟）', '#fff7ed', lc.C_ENG_S)
qcell(WT_X + 24 + CELL_W + 12, CY, 'ready', '普通 WAITING', '可立即调度', lc.C_KV_F, lc.C_KV_S)

# ---------------- 中部：遍历框 ----------------
D_Y, D_H = 300, 96
D_X, D_W = 350, 500
lc.rect(D_X, D_Y, D_W, D_H, '#ffffff', lc.C_KV_S, rx=8, sw=1.8)
lc.text(D_X + D_W / 2, D_Y + 24, '每拍遍历：逐队头 peek（O(1) 判定）', 11.5, lc.C_TXT, 'middle', True,
        maxw=D_W - 20, tag='d:t')
lc.text(D_X + D_W / 2, D_Y + 46, 'FCFS 择队：skipped 优先（L2065-L2066）· PRIORITY 换两队队头比较（L2068-L2073）', 9,
        lc.C_MUTE, 'middle', maxw=D_W - 20, tag='d:l1')
lc.text(D_X + D_W / 2, D_Y + 68, '跳过者 pop 出队、收进 step_skipped_waiting（不卡队头）', 9, '#334155',
        'middle', maxw=D_W - 20, tag='d:l2')

# peek 箭头：older 底 → D 左缘（肘形）；newer 底 → D 顶（直落）
lc.parrow([(OLDER_CX, CY + CELL_H), (OLDER_CX, D_Y + 44), (D_X, D_Y + 44)], lc.C_MUTE, 1.8, 'std')
lc.seg(NEWER_CX, CY + CELL_H, NEWER_CX, D_Y, lc.C_MUTE, 1.8, 'std')
lc.text(NEWER_CX + 8, D_Y - 8, 'peek', 8.4, lc.C_MUTE, 'start', tag='peeklbl2')

# ---------------- 三个出口 ----------------
OUT_Y, OUT_H = 440, 60
# ① older：阻塞 → 跳过（灰）
S1_X, S1_W = 100, 400
lc.parrow([(D_X, D_Y + 24), (S1_X + S1_W / 2, D_Y + 24), (S1_X + S1_W / 2, OUT_Y)], lc.C_MUTE, 1.8, 'std')
lc.rect(S1_X, OUT_Y, S1_W, OUT_H, '#f1f5f9', lc.C_MUTE, rx=7, sw=1.3)
lc.text(S1_X + 14, OUT_Y + 22, '① older：仍在等 → 跳过', 10, lc.C_TXT, 'start', True, maxw=S1_W - 28, tag='o1t')
lc.text(S1_X + 14, OUT_Y + 44, 'promote 未好（O(1) 尝试），弹出收进 step_skipped', 8.6, '#334155', 'start',
        maxw=S1_W - 26, tag='o1l')
# ② newer：stale>0 且非 drop → 推迟恢复一拍（橙）
S2_X, S2_W = 550, 400
lc.parrow([(D_X + D_W / 2, D_Y + D_H), (S2_X + S2_W / 2, D_Y + D_H), (S2_X + S2_W / 2, OUT_Y)], lc.C_ENG_S, 1.8, 'up')
lc.rect(S2_X, OUT_Y, S2_W, OUT_H, '#fff7ed', lc.C_ENG_S, rx=7, sw=1.4)
lc.text(S2_X + 14, OUT_Y + 22, '② newer：stale=1>0 → 推迟恢复一拍', 10, lc.C_TXT, 'start', True,
        maxw=S2_W - 28, tag='o2t')
lc.text(S2_X + 14, OUT_Y + 44, '现在恢复会重采输出稍后要送的位置（L713-L722）', 8.6, '#334155', 'start',
        maxw=S2_W - 26, tag='o2l')
# ③ ready：可调度 → 准入（绿）
S3_X, S3_W = 1000, 400
lc.parrow([(D_X + D_W, D_Y + 24), (S3_X + S3_W / 2, D_Y + 24), (S3_X + S3_W / 2, OUT_Y)], lc.C_GPU_S, 1.8, 'okg')
lc.rect(S3_X, OUT_Y, S3_W, OUT_H, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.4)
lc.text(S3_X + 14, OUT_Y + 22, '③ ready：可调度 → 准入 {ready:16}', 10, lc.C_TXT, 'start', True,
        maxw=S3_W - 28, tag='o3t')
lc.text(S3_X + 14, OUT_Y + 44, 'waiting 弹出 → running（本拍就进批）', 8.6, '#334155', 'start',
        maxw=S3_W - 26, tag='o3l')

# ---------------- 收集框 + 步末回插 ----------------
CO_Y, CO_H = 560, 66
CO_X, CO_W = 150, 660
lc.rect(CO_X, CO_Y, CO_W, CO_H, '#ffffff', lc.C_ENG_S, rx=8, sw=1.5)
lc.text(CO_X + 14, CO_Y + 24, 'step_skipped_waiting = [newer, older]（收集序：prepend，后跳者在前）', 10.5,
        lc.C_TXT, 'start', True, maxw=CO_W - 28, tag='co:t')
lc.text(CO_X + 14, CO_Y + 47, '本拍跳过的都收在这里，步末整体回插——队列里绝不留阻塞者占道', 9, lc.C_MUTE,
        'start', maxw=CO_W - 26, tag='co:l')
# ①/② 出口框 → 收集框
lc.seg(S1_X + S1_W / 2, OUT_Y + OUT_H, S1_X + S1_W / 2, CO_Y, lc.C_MUTE, 1.6, 'std')
lc.seg(S2_X + S2_W / 2, OUT_Y + OUT_H, S2_X + S2_W / 2, CO_Y, lc.C_ENG_S, 1.6, 'up')

# 步末粗箭头：收集框左缘 → 左缘走廊 → 上方走廊 → skipped 队头顶部进入
lc.parrow([(CO_X, CO_Y + CO_H / 2), (84, CO_Y + CO_H / 2), (84, 76), (OLDER_CX, 76), (OLDER_CX, Q_Y)],
          lc.C_ENG_S, 2.6, 'up')
lc.text(200, 88, '步末：skipped.prepend_requests(step_skipped)（L1099-L1101）——extendleft 反转 × prepend 反转 = 跳过序 [older, newer]',
        9.5, lc.C_ENG_S, 'start', True, maxw=820, tag='reins')

# ---------------- 右下：两拍时间轴快照 ----------------
TL_X, TL_Y, TL_W, TL_H = 860, 560, 580, 210
lc.rect(TL_X, TL_Y, TL_W, TL_H, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(TL_X + 14, TL_Y + 22, '两拍快照（实跑）', 11, lc.C_TXT, 'start', True, maxw=TL_W - 28, tag='tl:t')
ROWS = [
    ('拍 1', ['skipped=[older,newer]（=本拍跳过序）· waiting=[]', 'running ← ready 准入 {ready:16}',
              'newer 推迟、older 阻塞——三个都没丢']),
    ('拍 2', ['stale 已排干：newer 以 resumed 恢复（resumed=[newer]）', 'older 仍阻塞再跳过（每拍恰 peek 一次）',
              'skipped=[older]，waiting=[]']),
]
ry = TL_Y + 44
for bid, lines in ROWS:
    bw = 16 + 11 * len(bid)
    lc.rect(TL_X + 16, ry, bw, 20, lc.C_BADGE_F, lc.C_ENG_S, rx=9, sw=1.1)
    lc.text(TL_X + 16 + bw / 2, ry + 13.5, bid, 9.5, lc.C_ENG_S, 'middle', True, tag='tlb' + bid)
    for j, ln in enumerate(lines):
        lc.text(TL_X + 16 + bw + 14, ry + 7 + j * 17, ln, 8.8, '#334155', 'start',
                maxw=TL_W - 44 - bw - 14, tag='tl' + bid + str(j))
    ry += 80

# ---------------- 底部：单队列反事实 + 图例 + 页脚 ----------------
FC_Y = TL_Y + TL_H + 22
lc.rect(MX, FC_Y, BXR - MX, 44, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(MX + 16, FC_Y + 18, '单队列反事实：older 卡在队头则 newer/ready 全体饿死——双队列把阻塞态的代价从『堵住整条队』降为『每拍一次 peek』', 9.5,
        '#334155', 'start', maxw=BXR - MX - 32, tag='fc1')
lc.text(MX + 16, FC_Y + 34, '三个运行时来源（grammar 编译 / 远程 KV / 流式输入）分别归 ch30 / ch16 / ch12 语境；PRIORITY 择队 = 两队队头比较（L2068-L2073，随精简版删）', 8.6,
        lc.C_MUTE, 'start', maxw=BXR - MX - 32, tag='fc2')

LEG_Y = FC_Y + 66
lx = MX
for kind, name in [('grey', '阻塞（skipped 隔离）'), ('orange', '推迟恢复（stale>0）'), ('green', '当拍准入'),
                   ('thick', '步末回插（粗）')]:
    if kind == 'grey':
        lc.rect(lx, LEG_Y - 9, 20, 12, '#f1f5f9', lc.C_MUTE, rx=3, sw=1.2)
    elif kind == 'orange':
        lc.rect(lx, LEG_Y - 9, 20, 12, '#fff7ed', lc.C_ENG_S, rx=3, sw=1.2)
    elif kind == 'green':
        lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.2)
    else:
        lc.seg(lx, LEG_Y - 3, lx + 18, LEG_Y - 3, lc.C_ENG_S, 3.0)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=200, tag='leg' + kind)
    lx += 26 + lc.tw(name, 8.8) + 20

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/sched/scheduler.py:L687-L722（双队列遍历）/ L2050-L2062（阻塞态路由）'
        '/ L2065-L2073（择队：FCFS skipped 优先）/ L1099-L1101（步末重排 prepend_requests）· 队列快照取自配套精简版 host 实跑 · 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch11-fig-dual-queue.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
