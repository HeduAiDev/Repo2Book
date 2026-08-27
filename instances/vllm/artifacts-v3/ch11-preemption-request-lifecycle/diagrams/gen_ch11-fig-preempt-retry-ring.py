#!/usr/bin/env python3
"""ch11 机制图 1 · RUNNING 侧抢占重试环（figure_spec ch11-fig-preempt-retry-ring，模板 flow）

放大自 L0 右列『调度 · 显存账本』（kv_column 青色列）上半 Scheduler 框——即本章 L2 章图
（L2-ch11.png）center ② allocate_slots→None → ③ 抢谁 → ⑤ 环终止·守卫关闸 拍片的机制展开；
非新架构画法，架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：RUNNING 侧 allocate_slots 一返回 None 就进 while True：抢占 FCFS 队尾腾块后原样
重试，抢到自己仍分不到才整拍放弃——同一个 None 在 WAITING 侧只 break。

数字全部取自 figure_spec.numbers（A-2 调用序列 / B-2 抢到自己 / B-3 经 WAITING 恢复 /
WAITING 侧反差 A-3），源出配套精简版 host 实跑 trace（场景 A：4 块池 r1/r2/r3；场景 B：1 块池单请求）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 962
MX, BXR = 60, 1440

EXTRA_DEFS = ('<defs>'
              '<marker id="okg" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
              f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_GPU_S}"/></marker>'
              '<marker id="kvm" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
              f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_KV_S}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'RUNNING 侧的抢占重试环：None → 抢 FCFS 队尾腾块 → 原样重试，抢到自己仍分不到才整拍放弃',
        16.5, lc.C_TXT, 'start', True, maxw=1030, tag='title')
lc.text(MX, 58, 'while True（scheduler.py:L575-L630）：同一个 allocate_slots 返回 None，在 WAITING 侧只 break——新请求绝不赶走老请求',
        10.5, lc.C_MUTE, 'start', maxw=1040, tag='subtitle')
_ch = '放大自 L2 拍片 ②③⑤ · L0：调度·显存账本列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_KV_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 环几何（中线 x=660） ----------------
CX = 660
BOX_W = 460
BX = CX - BOX_W / 2                      # 430

# A: allocate_slots
A_Y, A_H = 100, 56
lc.rect(BX, A_Y, BOX_W, A_H, '#ffffff', lc.C_KV_S, rx=8, sw=1.8)
lc.text(CX, A_Y + 21, 'RUNNING 增长：allocate_slots(request, num_new_tokens)', 11.5, lc.C_TXT, 'middle', True,
        maxw=BOX_W - 20, tag='a:t')
lc.text(CX, A_Y + 42, 'scheduler.py:L576-L585', 8.5, lc.C_FAINT, 'middle', tag='a:f')

lc.seg(CX, A_Y + A_H, CX, A_Y + A_H + 26, lc.C_KV_S, 2.0, 'kvm')

# B: 判定 pill
B_Y, B_H = A_Y + A_H + 26, 40            # 182
PILL_W = 300
lc.rect(CX - PILL_W / 2, B_Y, PILL_W, B_H, lc.C_KV_F, lc.C_KV_S, rx=19, sw=1.8)
lc.text(CX, B_Y + 25, 'new_blocks ?', 11.5, lc.C_TXT, 'middle', True, tag='b:t')

# OK 出口（右）
C_X, C_Y, C_W, C_H = 950, B_Y - 6, 470, 52
lc.rect(C_X, C_Y, C_W, C_H, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.6)
lc.text(C_X + C_W / 2, C_Y + 21, 'OK：调度成功——领到 new_blocks，break 出环（L584-L586）', 10.5, lc.C_TXT,
        'middle', True, maxw=C_W - 16, tag='c:t')
lc.text(C_X + C_W / 2, C_Y + 40, '本请求照常进本拍批次', 9, lc.C_MUTE, 'middle', tag='c:s')
lc.seg(CX + PILL_W / 2, B_Y + B_H / 2, C_X, B_Y + B_H / 2, lc.C_GPU_S, 2.0, 'okg')

# None 出口（下，红）
lc.seg(CX, B_Y + B_H, CX, B_Y + B_H + 34, lc.C_ABORT, 2.0, 'ab')
lc.text(CX + 8, B_Y + B_H + 22, 'None · 内存见底（L590）', 9, lc.C_ABORT, 'start', tag='none-lbl')

# D: 抢谁
D_Y, D_H = B_Y + B_H + 34, 84            # 330
lc.rect(BX, D_Y, BOX_W, D_H, '#ffffff', lc.C_KV_S, rx=8, sw=1.6)
lc.text(CX, D_Y + 20, '抢谁：preempted_req = self.running.pop()（L615）', 10.5, lc.C_TXT, 'middle', True,
        maxw=BOX_W - 16, tag='d:t')
lc.text(CX, D_Y + 40, 'FCFS 恒取队尾 = 最年轻者——谁触发分配失败、谁占块多都不看', 9, '#334155', 'middle',
        maxw=BOX_W - 16, tag='d:l1')
lc.text(CX, D_Y + 57, 'PRIORITY 半边：max((priority, arrival_time)) + 回滚其本拍已领（L588-L613）', 9, C_MUTE_S := lc.C_MUTE,
        'middle', maxw=BOX_W - 16, tag='d:l2')
lc.text(CX, D_Y + 74, '精简版删去不可运行——语义以源码引文呈现', 8.2, lc.C_FAINT, 'middle', tag='d:l3')

lc.seg(CX, D_Y + D_H, CX, D_Y + D_H + 24, lc.C_KV_S, 2.0, 'kvm')

# E: _preempt_request
E_Y, E_H = D_Y + D_H + 24, 58            # 472
lc.rect(BX, E_Y, BOX_W, E_H, '#ffffff', lc.C_KV_S, rx=8, sw=1.6)
lc.text(CX, E_Y + 21, '_preempt_request(preempted_req)（L617-L621）', 10.5, lc.C_TXT, 'middle', True,
        maxw=BOX_W - 16, tag='e:t')
lc.text(CX, E_Y + 43, '块归池（哈希留表）· 六件事 → 回 waiting 队头', 9, '#334155', 'middle',
        maxw=BOX_W - 16, tag='e:l1')

lc.seg(CX, E_Y + E_H, CX, E_Y + E_H + 24, lc.C_KV_S, 2.0, 'kvm')

# F: 自我判定 pill
F_Y, F_H = E_Y + E_H + 24, 40            # 594
lc.rect(CX - PILL_W / 2, F_Y, PILL_W, F_H, lc.C_KV_F, lc.C_KV_S, rx=19, sw=1.8)
lc.text(CX, F_Y + 25, 'preempted_req == request ?', 11, lc.C_TXT, 'middle', True, tag='f:t')

# 否：回边原样重试（左肘形）
LOOP_X = 350
lc.parrow([(CX - PILL_W / 2, F_Y + F_H / 2), (LOOP_X, F_Y + F_H / 2), (LOOP_X, A_Y + A_H / 2),
           (BX, A_Y + A_H / 2)], lc.C_KV_S, 2.0, 'kvm')
lc.text(LOOP_X - 10, A_Y + A_H + 64, '否：腾出块后原样重试', 9.5, lc.C_KV_S, 'end', True, tag='loop1')
lc.text(LOOP_X - 10, A_Y + A_H + 80, '（while True 下一轮，L577）', 8.5, lc.C_MUTE, 'end', tag='loop2')

# 是：整拍放弃（下）
lc.seg(CX, F_Y + F_H, CX, F_Y + F_H + 30, lc.C_ABORT, 2.0, 'ab')
lc.text(CX + 8, F_Y + F_H + 20, '是：把自己都抢了仍分不到', 9, lc.C_ABORT, 'start', tag='yes-lbl')

G_Y, G_H = F_Y + F_H + 30, 64            # 724
lc.rect(BX, G_Y, BOX_W, G_H, '#fff7ed', lc.C_ABORT, rx=8, sw=1.6)
lc.text(CX, G_Y + 22, 'break：整拍放弃——『No more request to preempt』（L624）', 10.5, lc.C_TXT, 'middle', True,
        maxw=BOX_W - 16, tag='g:t')
lc.text(CX, G_Y + 46, 'new_blocks 仍 None → break 出整个 RUNNING 循环（L627-L630）：本拍不再调度新请求', 9,
        '#334155', 'middle', maxw=BOX_W - 16, tag='g:l1')

lc.seg(CX, G_Y + G_H, CX, G_Y + G_H + 26, lc.C_ENG_S, 2.0, 'up')

H_Y, H_H = G_Y + G_H + 26, 56
lc.rect(BX, H_Y, BOX_W, H_H, '#ffffff', lc.C_ENG_S, rx=8, sw=1.6)
lc.text(CX, H_Y + 21, '被抢者回 waiting 队头 → 下一拍经 WAITING 准入恢复', 10.5, lc.C_TXT, 'middle', True,
        maxw=BOX_W - 16, tag='h:t')
lc.text(CX, H_Y + 42, '整序列门与水位都在那条路上（放弃 ≠ 死亡，见下方 B-3）', 9, lc.C_MUTE, 'middle',
        maxw=BOX_W - 16, tag='h:l1')

# ---------------- 右侧：WAITING 侧对照（虚线） ----------------
WQ_X, WQ_Y, WQ_W, WQ_H = 950, 320, 470, 190
lc.rect(WQ_X, WQ_Y, WQ_W, WQ_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.4, dash=True)
lc.text(WQ_X + 16, WQ_Y + 24, '对照 · WAITING 侧：同一个 None 只 break（L987-L994）', 11, lc.C_TXT, 'start', True,
        maxw=WQ_W - 32, tag='wq:t')
_wl = ['· 新请求 allocate_slots 返回 None → break 本请求，',
       '  环根本不在那条路上——在场请求绝不被抢',
       '· 门口分不到位子的新客只会继续排队，',
       '  绝没有『为给新客腾位赶走堂食客』的事',
       '· A-3：r3 恢复被拒即此出口——重命中 16 只差',
       '  1 新块 > 空闲 0 → None 只 break，本拍照旧']
for i, ln in enumerate(_wl):
    lc.text(WQ_X + 16, WQ_Y + 48 + i * 18, ln, 9, '#334155', 'start', maxw=WQ_W - 28, tag='wq:l' + str(i))
lc.text(WQ_X + 16, WQ_Y + WQ_H - 12, 'vllm/v1/core/sched/scheduler.py:L965-L994', 8.5, lc.C_FAINT, 'start',
        tag='wq:f')

# ---------------- 底部：三个数字标注（钉在环上的实跑账） ----------------
NB_Y, NB_H = 824, 74
NB_W = 412
NBS = [
    ('A-2', lc.C_KV_S, ['调用序列 (r1,1)→OK · (r2,1)→None · (r2,1)→OK',
                        'r1 领走末块；r2 差 1 → 抢队尾 r3 腾块 → 原样重试成功',
                        '（池空闲 1→0，本拍被抢 [r3]，调度 {r1:1, r2:1}）']),
    ('B-2', lc.C_ABORT, ['场景 B（1 块池）：唯一调用 (r1,1)→None',
                         '抢到自己（唯一在场者）→ 整拍放弃：sched={}、',
                         'preempted=[r1]、池空闲 0→1']),
    ('B-3', lc.C_ENG_S, ['下一拍经 WAITING 准入恢复：重命中 16 + 补 1',
                         '（num_scheduled={r1:1}，resumed）',
                         '——放弃 ≠ 死亡，同一请求带着前缀回来']),
]
for i, (bid, col, lines) in enumerate(NBS):
    x = MX + i * (NB_W + 22)
    lc.rect(x, NB_Y, NB_W, NB_H, '#ffffff', col, rx=8, sw=1.3, dash=True)
    bw = 16 + 11 * len(bid)
    lc.rect(x + 12, NB_Y + 10, bw, 20, lc.C_BADGE_F, col, rx=9, sw=1.1)
    lc.text(x + 12 + bw / 2, NB_Y + 23.5, bid, 9.5, col, 'middle', True, tag='nb' + bid)
    for j, ln in enumerate(lines):
        lc.text(x + 16, NB_Y + 44 + j * 15, ln, 8.6, '#334155', 'start', maxw=NB_W - 26,
                tag='nb' + bid + ':' + str(j))

# ---------------- 图例 + 页脚 ----------------
LEG_Y = 922
lx = MX
for kind, name in [('ok', 'OK / 放行路径'), ('none', 'None / 抢占触发'), ('kv', '重试环（调度账本）'),
                   ('eng', '恢复去向'), ('dash', '另一侧对照')]:
    if kind == 'ok':
        lc.seg(lx + 2, LEG_Y - 3, lx + 32, LEG_Y - 3, lc.C_GPU_S, 2.0)
    elif kind == 'none':
        lc.seg(lx + 2, LEG_Y - 3, lx + 32, LEG_Y - 3, lc.C_ABORT, 2.0)
    elif kind == 'kv':
        lc.seg(lx + 2, LEG_Y - 3, lx + 32, LEG_Y - 3, lc.C_KV_S, 2.0)
    elif kind == 'eng':
        lc.seg(lx + 2, LEG_Y - 3, lx + 32, LEG_Y - 3, lc.C_ENG_S, 2.0)
    else:
        lc.seg(lx + 2, LEG_Y - 3, lx + 32, LEG_Y - 3, lc.C_MUTE, 1.6, dash=True)
    lc.text(lx + 38, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=200, tag='leg' + kind)
    lx += 38 + lc.tw(name, 8.8) + 20

lc.text(MX, 946, '逐字锚 vllm/v1/core/sched/scheduler.py:L575-L630（重试环）/ L615（running.pop 队尾）/ L624（No more request to preempt）'
        '/ L987-L994（WAITING 侧 None→break）· 调用序列与拍账取自配套精简版 host 实跑（场景 A：4 块池；场景 B：1 块池）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch11-fig-preempt-retry-ring.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
