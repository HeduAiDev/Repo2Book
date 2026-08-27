#!/usr/bin/env python3
"""ch11 机制图 3 · _preempt_request 六件事状态表（figure_spec ch11-fig-preempt-six-things，模板 state-table）

放大自 L0 右列『调度 · 显存账本』（kv_column 青色列）上半 Scheduler 框——即本章 L2 章图
center ④ _preempt_request·六件事 拍片的机制展开；非新架构画法，架构归属回指 L0/L2
（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：_preempt_request 一口气做六件事，把请求带回与首调度同构的初态：free 块（哈希留表）/
PREEMPTED / computed=0 / 清 spec / stale←in_flight / preemptions+1 → 回 waiting 队头。

数字全部取自 figure_spec.numbers（computed 16→0 / spec [9,9]→[] / stale 0→2 assign 不累加 /
preemptions 0→1 / 持块 1→0 池 0→1→0 / cached 哈希 2→2 / waiting [w1]→[r2,w1]），
源出配套精简版 host 实跑 trace（2 块池；r2 被抢前后快照）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 700
MX, BXR = 60, 1440

EXTRA_DEFS = ('<defs>'
              '<marker id="kvm" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
              f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_KV_S}"/></marker>'
              '</defs>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, '_preempt_request 六件事：被抢占不是被删除，是退房重排——回到与首调度同构的初态',
        16.5, lc.C_TXT, 'start', True, maxw=1030, tag='title')
lc.text(MX, 58, '六件事在同一个函数内顺序完成（scheduler.py:L1274-L1315），唯一 assert 是入口守卫 RUNNING——中途不可观测，恢复因此复用首调度通道',
        10.5, lc.C_MUTE, 'start', maxw=1040, tag='subtitle')
_ch = '放大自 L2 拍片 ④ 六件事 · L0：调度·显存账本列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_KV_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 列几何 ----------------
EV_X, EV_W = 76, 360            # 事件列（序号+名称+源码行）
BF_X, BF_W = 456, 250           # 抢占前
AF_X, AF_W = 856, 250           # 后
NT_X, NT_W = 1130, 300          # 右侧注记

HDR_Y = 104
lc.text(EV_X + 14, HDR_Y, '六件事（发生序）', 10, lc.C_MUTE, 'start', True, tag='hd1')
lc.text(BF_X + BF_W / 2, HDR_Y, '抢占前（r2 实跑快照）', 10, lc.C_MUTE, 'middle', True, tag='hd2')
lc.text(AF_X + AF_W / 2, HDR_Y, '_preempt_request 后', 10, lc.C_MUTE, 'middle', True, tag='hd3')
lc.text(NT_X, HDR_Y, '注记', 10, lc.C_MUTE, 'start', True, tag='hd4')

# 行数据：(①号, 名称, 源码行, before, after, 注记两行, after强调色)
ROWS = [
    ('①', 'free 全部块 · 哈希留表', 'L1290  _free_request_blocks',
     '持块 1 · 池空闲 0', '持块 0 · 池空闲 1', ['cached 哈希 2 → 2（不清）', '——恢复时重命中的伏笔（→ch15）'], lc.C_KV_S),
    ('②', 'status = PREEMPTED', 'L1293  request.status = …',
     'RUNNING', 'PREEMPTED', ['与首调度同构的', '可调度初态的一半'], lc.C_KV_S),
    ('③', 'num_computed_tokens = 0', 'L1294  request.num_computed…',
     '16', '0', ['recompute-only 的语义本体', '——清零即『重算』的意思'], lc.C_ABORT),
    ('④', '清 spec_token_ids', 'L1295-L1296  spec_token…= []',
     '[9, 9]', '[]', ['作废未用赠券：', '恢复后由追赶公式重排'], lc.C_KV_S),
    ('⑤', 'stale ← in_flight（assign）', 'L1307  stale = in_flight',
     'stale 0（in_flight 2）', 'stale 2（in_flight 2）', ['赋值而非累加——在途输出照常', '送达，但不得改已清零的计数器'], lc.C_KV_S),
    ('⑥', 'preemptions + 1 → 回队头', 'L1309/L1314  +=1 · prepend',
     '0 · waiting [w1]', '1 · waiting [r2, w1]', ['一生的累计伤疤，从不回零；', 'prepend 到队头而非队尾'], lc.C_ENG_S),
]
ROW_Y0, ROW_H = 122, 66
for i, (num, name, srcline, before, after, notes, col) in enumerate(ROWS):
    ry = ROW_Y0 + i * ROW_H
    if i > 0:
        lc.seg(MX, ry - 4, BXR, ry - 4, '#e2e8f0', 1.0)
    # 事件列：序号徽标 + 名称 + 源码行
    lc.rect(EV_X, ry + 14, 26, 22, lc.C_BADGE_F, col, rx=11, sw=1.1)
    lc.text(EV_X + 13, ry + 29, num, 11, col, 'middle', True, tag='num' + num)
    lc.text(EV_X + 40, ry + 24, name, 10.5, lc.C_TXT, 'start', True, maxw=EV_W - 44, tag='nm' + num)
    lc.text(EV_X + 40, ry + 44, srcline, 8.4, lc.C_FAINT, 'start', maxw=EV_W - 44, tag='src' + num)
    # before 值
    lc.rect(BF_X, ry + 8, BF_W, 40, '#ffffff', lc.C_MUTE, rx=6, sw=1.1)
    lc.text(BF_X + BF_W / 2, ry + 33, before, 10.5, '#334155', 'middle', maxw=BF_W - 14, tag='bf' + num)
    # 箭头
    lc.seg(BF_X + BF_W + 8, ry + 28, AF_X - 10, ry + 28, lc.C_KV_S, 1.8, 'kvm')
    # after 值
    lc.rect(AF_X, ry + 8, AF_W, 40, lc.C_KV_F if col != lc.C_ABORT else '#fff7ed', col, rx=6, sw=1.5)
    lc.text(AF_X + AF_W / 2, ry + 33, after, 10.5, col, 'middle', True, maxw=AF_W - 14, tag='af' + num)
    # 注记
    for j, ln in enumerate(notes):
        lc.text(NT_X, ry + 26 + j * 16, ln, 8.8, lc.C_MUTE if j else '#334155', 'start', maxw=NT_W,
                tag='nt' + num + str(j))

# ---------------- 底部窄条：waiting 队列 prepend ----------------
BQ_Y = ROW_Y0 + 6 * ROW_H + 14
lc.rect(MX, BQ_Y, 700, 92, '#ffffff', lc.C_ENG_S, rx=8, sw=1.4)
lc.text(MX + 16, BQ_Y + 22, '第⑥件的另一半：waiting 队列 [w1] → [r2, w1]（prepend 到队头，不是队尾）', 10.5,
        lc.C_TXT, 'start', True, maxw=670, tag='bq:t')
# 队列小格：前
_qx = MX + 40
lc.text(_qx + 34, BQ_Y + 48, '前', 9, lc.C_MUTE, 'middle', tag='bq:pre')
lc.rect(_qx + 60, BQ_Y + 36, 46, 26, '#ffffff', lc.C_MUTE, rx=4, sw=1.2)
lc.text(_qx + 83, BQ_Y + 53, 'w1', 10, '#334155', 'middle', True, tag='bq:w1')
lc.seg(_qx + 118, BQ_Y + 49, _qx + 168, BQ_Y + 49, lc.C_ENG_S, 2.0, 'up')
lc.text(_qx + 143, BQ_Y + 40, 'prepend', 8.4, lc.C_ENG_S, 'middle', tag='bq:pre-lbl')
# 后：r2 在前
_qx2 = _qx + 186
lc.rect(_qx2, BQ_Y + 36, 46, 26, lc.C_ENG_F, lc.C_ENG_S, rx=4, sw=1.6)
lc.text(_qx2 + 23, BQ_Y + 53, 'r2', 10, lc.C_ENG_S, 'middle', True, tag='bq:r2')
lc.rect(_qx2 + 54, BQ_Y + 36, 46, 26, '#ffffff', lc.C_MUTE, rx=4, sw=1.2)
lc.text(_qx2 + 77, BQ_Y + 53, 'w1', 10, '#334155', 'middle', True, tag='bq:w1b')
lc.text(_qx2 + 128, BQ_Y + 48, '后（队头=r2）', 9, lc.C_MUTE, 'start', tag='bq:post')

# 右下：swap 注记（虚线）
SW_X = MX + 724
lc.rect(SW_X, BQ_Y, BXR - SW_X, 92, '#ffffff', lc.C_MUTE, rx=8, sw=1.3, dash=True)
lc.text(SW_X + 16, BQ_Y + 22, 'v1 从未有过 v0 的 swap', 10.5, lc.C_TXT, 'start', True, maxw=400, tag='sw:t')
for j, ln in enumerate(['首提交 6c5af09b3 即 num_computed_tokens=0 的 recompute-only（git 证据）；',
                        'v0 swap 的死法『Aborted due to the lack of CPU swap space』全引擎崩——',
                        '现行源码无此代码，丢弃重算是自首提交起的唯一路径']):
    lc.text(SW_X + 16, BQ_Y + 42 + j * 16, ln, 8.8, '#334155' if j == 0 else lc.C_MUTE, 'start',
            maxw=BXR - SW_X - 30, tag='sw:l' + str(j))

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BQ_Y + 118
lx = MX
for kind, name in [('kv', '值区（青=六件事产出）'), ('abort', '语义本行（红=清零本体）'), ('arrow', '同一函数内顺序完成'),
                   ('dash', '史实注记')]:
    if kind == 'kv':
        lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.4)
    elif kind == 'abort':
        lc.rect(lx, LEG_Y - 9, 20, 12, '#fff7ed', lc.C_ABORT, rx=3, sw=1.4)
    elif kind == 'arrow':
        lc.seg(lx, LEG_Y - 3, lx + 18, LEG_Y - 3, lc.C_KV_S, 1.8)
    else:
        lc.seg(lx, LEG_Y - 3, lx + 18, LEG_Y - 3, lc.C_MUTE, 1.4, dash=True)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=240, tag='leg' + kind)
    lx += 26 + lc.tw(name, 8.8) + 20

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/sched/scheduler.py:L1274-L1315（_preempt_request）/ L1290 free / L1293 status / '
        'L1294 computed / L1295-L1296 spec / L1307 stale←in_flight / L1309 +1 / L1314 prepend · 快照数字取自配套精简版 host 实跑'
        '（2 块池，r2 被抢前后；r1 重试后池空闲回 0）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch11-fig-preempt-six-things.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
