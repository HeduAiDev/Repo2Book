#!/usr/bin/env python3
"""ch18 机制图 1 · 差量协议五拍载荷（figure_spec ch18-fig-diff-protocol，模板 flow）

放大自 L0『GPU 执行臂』（gpu_column 绿色列）『执行臂中层』GPUModelRunner 框的入线——
即本章 L2 章图 north『入 · EngineCore.step ②』→ center ②『_update_states · 差量调和』
之间标着「SchedulerOutput 差量过线」那根箭头的载荷展开。架构归属回指 L0/L2
（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：同一根 execute_model 入线上，五拍发出五种形状完全不同的载荷——新请求拍背全量
（prompt+块+采样参数）、稳态拍只背 2 条 diff（1 个块号 + 2 个 int）、finished 拍多带
1 个 id、resumed 拍的块号从「追加」变「整体替换」——协议把通信量从
「请求数×prompt 长度」压缩到「变更数」。

数字全部取自 figure_spec.numbers（traces/ch18_m02_reconcile.json 五拍 wire 字段 +
output.py:L194-L200 / L118-L121 注释原文）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 812
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '首拍背全量，此后只发 diff——同一根入线，五拍五种载荷',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'execute_model 每拍收到的 SchedulerOutput：新请求拍两份全套建档、稳态拍只有 1 个块号 + 2 个计数、finished 拍多 1 个 id、resumed 拍同名块号字段从「追加」翻成「整体替换」',
        10.5, lc.C_MUTE, 'start', maxw=1020, tag='subtitle')
_ch = '放大自 L2 入线 SchedulerOutput 差量过线 · L0：GPU 执行臂中层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左右两端：Scheduler / worker·InputBatch（配色回指 L0：调度=青、执行臂=绿） ----------------
LANE_Y0, LANE_H, N_LANE = 110, 100, 5
BOT_Y = LANE_Y0 + N_LANE * LANE_H              # 610
SCH_X, SCH_W = MX, 168
WRK_X, WRK_W = BXR - 168, 168

lc.rect(SCH_X, LANE_Y0, SCH_W, BOT_Y - LANE_Y0, lc.C_KV_F, lc.C_KV_S, rx=9, sw=2.0)
lc.text(SCH_X + SCH_W / 2, LANE_Y0 + 130, 'Scheduler', 13, lc.C_KV_S, 'middle', True, tag='sch:t')
lc.text(SCH_X + SCH_W / 2, LANE_Y0 + 150, '调度账本列（L0）', 8.5, lc.C_MUTE, 'middle', tag='sch:s')
lc.text(SCH_X + SCH_W / 2, LANE_Y0 + 176, '每拍只发', 9, lc.C_MUTE, 'middle', tag='sch:l1')
lc.text(SCH_X + SCH_W / 2, LANE_Y0 + 192, '「变更的部分」', 9, lc.C_MUTE, 'middle', tag='sch:l2')
lc.text(SCH_X + SCH_W / 2, BOT_Y - 44, '上游为何只发 diff', 8.2, lc.C_FAINT, 'middle', tag='sch:f1')
lc.text(SCH_X + SCH_W / 2, BOT_Y - 30, 'ch10/ch12 已立', 8.2, lc.C_FAINT, 'middle', tag='sch:f2')

lc.rect(WRK_X, LANE_Y0, WRK_W, BOT_Y - LANE_Y0, lc.C_GPU_F, lc.C_GPU_S, rx=9, sw=2.0)
lc.text(WRK_X + WRK_W / 2, LANE_Y0 + 130, 'worker', 13, lc.C_GPU_S, 'middle', True, tag='wrk:t')
lc.text(WRK_X + WRK_W / 2, LANE_Y0 + 150, 'GPU 执行臂中层', 8.5, lc.C_MUTE, 'middle', tag='wrk:s')
lc.text(WRK_X + WRK_W / 2, LANE_Y0 + 176, 'requests 缓存全量', 9, lc.C_MUTE, 'middle', tag='wrk:l1')
lc.text(WRK_X + WRK_W / 2, LANE_Y0 + 192, '+ InputBatch 持久批次', 9, lc.C_MUTE, 'middle', tag='wrk:l2')
lc.text(WRK_X + WRK_W / 2, BOT_Y - 44, '差量调和消费点', 8.2, lc.C_FAINT, 'middle', tag='wrk:f1')
lc.text(WRK_X + WRK_W / 2, BOT_Y - 30, '_update_states（L2 拍片 ②）', 8.2, lc.C_FAINT, 'middle', tag='wrk:f2')

# ---------------- 票据绘制原语 ----------------
def full_ticket(x, y, w, h, rid, tok, blk):
    """NewRequestData 全量票：橙框（首拍建档）。"""
    lc.rect(x, y, w, h, lc.C_ENG_F, lc.C_ENG_S, rx=6, sw=1.5)
    lc.text(x + w / 2, y + 17, f'{rid} · prompt {tok} token（全量）', 9.3, lc.C_TXT, 'middle', True,
            maxw=w - 8, tag=f'ft:{rid}:1')
    lc.text(x + w / 2, y + 34, f'块号 {blk} · 采样参数全套', 8.7, '#334155', 'middle',
            maxw=w - 8, tag=f'ft:{rid}:2')

def diff_ticket(x, y, w, h, rid, l1, l2):
    """CachedRequestData 差量票：青底小票。"""
    lc.rect(x, y, w, h, lc.C_KV_F, lc.C_KV_S, rx=6, sw=1.4)
    lc.text(x + w / 2, y + 17, f'{rid} · {l1}', 9.3, lc.C_TXT, 'middle', True, maxw=w - 8,
            tag=f'dt:{rid}:1')
    lc.text(x + w / 2, y + 34, l2, 8.7, '#334155', 'middle', maxw=w - 8, tag=f'dt:{rid}:2')

def fin_flag(x, y, rid):
    """finished_req_ids 小旗：红虚线。"""
    w, h = 108, 42
    lc.rect(x, y, w, h, '#ffffff', lc.C_ABORT, rx=6, sw=1.3, dash=True)
    lc.text(x + w / 2, y + 17, f'finished [{rid}]', 9.3, lc.C_ABORT, 'middle', True, maxw=w - 8,
            tag=f'ff:{rid}:1')
    lc.text(x + w / 2, y + 34, '完结通知', 8.7, lc.C_MUTE, 'middle', maxw=w - 8, tag=f'ff:{rid}:2')
    return w

# ---------------- 五拍泳道 ----------------
AX0, AX1 = SCH_X + SCH_W + 4, WRK_X - 4       # 箭头横跨区间
TK_Y, TK_H = 22, 44                            # 票据在泳道内的相对位置
TKW = 186                                      # 票宽
LANES = [
    # (拍名, 票据绘制回调列表, 底注)
    ('拍 1 · 新请求拍', [
        lambda x, y: full_ticket(x, y, TKW, TK_H, 'r1', 2, '[1]'),
        lambda x, y: full_ticket(x + TKW + 10, y, TKW, TK_H, 'r2', 3, '[2]'),
    ], 'cached 0 条 · total 5 token——两份全套只此一次'),
    ('拍 2 · 稳态拍', [
        lambda x, y: diff_ticket(x, y, TKW, TK_H, 'r1', '+块号 [3]（追加 1 个）', 'num_computed ← 2'),
        lambda x, y: diff_ticket(x + TKW + 10, y, TKW, TK_H, 'r2', '新块 0 个（None）', 'num_computed ← 3'),
    ], '全量 0 份 · total 2——通信量与请求数×prompt 长度脱钩'),
    ('拍 3 · mixed 拍', [
        lambda x, y: fin_flag(x, y, 'r2'),
        lambda x, y: full_ticket(x + 118 + 8, y, TKW, TK_H, 'r3', 2, '[4]'),
        lambda x, y: diff_ticket(x + 118 + 8 + TKW + 10, y, TKW - 24, TK_H, 'r1', '新块 0 个', 'num_computed ← 3'),
    ], 'finished 1 + 全量 1 + diff 1 同拍并存 · total 3'),
    ('拍 4 · 被抢占拍', [
        lambda x, y: diff_ticket(x, y, TKW - 24, TK_H, 'r1', '新块 0 个', 'num_computed ← 4'),
    ], 'r3 未排——差量里根本没有它 · total 1（worker 靠缓存留住快照）'),
    ('拍 5 · resumed 拍', [
        lambda x, y: diff_ticket(x, y, TKW - 24, TK_H, 'r1', '新块 0 个', 'num_computed ← 5'),
        lambda x, y: diff_ticket(x + TKW - 14, y, TKW + 4, TK_H, 'r3', '块号 [5] 整体替换（原 [4]）', 'resumed ∈ resumed_req_ids'),
    ], '同一个 new_block_ids 字段，两种语义 · total 4'),
]

for i, (name, drawers, note) in enumerate(LANES):
    ly = LANE_Y0 + i * LANE_H
    if i > 0:
        lc.seg(AX0, ly - 1, AX1, ly - 1, '#e2e8f0', 1.0)
    # 拍名（泳道左端，start 对齐）
    lc.text(AX0 + 6, ly + 16, name, 9.5, lc.C_TXT, 'start', True, maxw=170, tag=f'lane{i}:name')
    # 票据簇（居中偏右排布）
    cluster_w = 186 * 2 + 10 if i != 2 else (118 + 8 + TKW + 10 + TKW - 24)
    x0 = AX0 + 190 + (AX1 - AX0 - 190 - cluster_w) / 2
    for d in drawers:
        d(x0, ly + TK_Y)
    # 差量箭头：Scheduler 右缘 → worker 左缘（贴框边）
    ay = ly + TK_Y + TK_H + 14
    lc.seg(AX0, ay, AX1, ay, lc.C_ENG_S, 2.0, 'std')
    # 底注
    lc.text((AX0 + AX1) / 2, ay + 17, note, 8.6, lc.C_MUTE, 'middle', maxw=AX1 - AX0 - 40,
            tag=f'lane{i}:note')

# 拍5 resumed 徽标（紫：语义分叉标记）
ry5 = LANE_Y0 + 4 * LANE_H + TK_Y
r3x = AX0 + 190 + (AX1 - AX0 - 190 - (TKW - 24 + 10 + TKW + 4)) / 2 + TKW - 14
lb = '语义分叉'
lc.rect(r3x + TKW + 4, ry5 - 12, 52, 15, lc.C_ZMQ_F, lc.C_ZMQ_S, rx=7, sw=1.0)
lc.text(r3x + TKW + 4 + 26, ry5 - 1.5, lb, 7.8, lc.C_ZMQ_S, 'middle', True, maxw=48, tag='rsm:badge')

# ---------------- 底部：对照条 + 协议注释引文 ----------------
CMP_Y = BOT_Y + 26
lc.text(MX, CMP_Y, '对照：若每拍全量重发', 10, lc.C_TXT, 'start', True, maxw=170, tag='cmp:t1')
for k in range(5):
    lc.rect(MX + 178 + k * 62, CMP_Y - 11, 54, 22, '#ffffff', lc.C_MUTE, rx=4, sw=1.1, dash=True)
lc.text(MX + 178 + 5 * 62 + 10, CMP_Y, '5 拍 × 每请求全套 × prompt 长度——随批规模线性涨', 8.6,
        lc.C_MUTE, 'start', maxw=430, tag='cmp:r')
lc.text(MX + 990, CMP_Y, '差量协议：全量只在首拍，此后 ∝ 本拍变更数（稳态拍=1 个块号+2 个 int）', 8.6,
        lc.C_KV_S, 'start', True, maxw=BXR - MX - 990, tag='cmp:d')

QU_Y = CMP_Y + 30
lc.rect(MX, QU_Y, 700, 78, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(MX + 14, QU_Y + 17, '协议注释自述（vllm/v1/core/sched/output.py:L194-L200）', 9.3, lc.C_TXT,
        'start', True, maxw=660, tag='q1:t')
lc.text(MX + 14, QU_Y + 36, '「We cache the request\'s data in each worker process, so that we don\'t need to re-send it every scheduling step.',
        8.2, '#334155', 'start', maxw=672, tag='q1:l1')
lc.text(MX + 14, QU_Y + 52, '…we only send the diff to minimize the communication cost.」', 8.2, '#334155',
        'start', maxw=672, tag='q1:l2')
lc.text(MX + 14, QU_Y + 69, 'resumed 语义分叉（output.py:L118-L121，消费点 gpu_model_runner.py:L1441-L1452）：', 8.2,
        lc.C_MUTE, 'start', maxw=672, tag='q2:t')

QR_X = MX + 716
lc.rect(QR_X, QU_Y, BXR - QR_X, 78, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(QR_X + 14, QU_Y + 17, '「For those in the set, new_block_ids will be used as the request block IDs', 8.2,
        '#334155', 'start', maxw=BXR - QR_X - 26, tag='q2:l1')
lc.text(QR_X + 14, QU_Y + 33, 'instead of appending to the existing block IDs.」——同名块号字段，', 8.2,
        '#334155', 'start', maxw=BXR - QR_X - 26, tag='q2:l2')
lc.text(QR_X + 14, QU_Y + 49, '普通请求追加 / resumed 整体替换——拍 5 的 [5]↖换掉 [4] 由此消费。', 8.2,
        '#334155', 'start', maxw=BXR - QR_X - 26, tag='q2:l3')

# 图例
LEG_Y = QU_Y + 96
lx = MX
items = [
    ('full', lc.C_ENG_S, lc.C_ENG_F, 'NewRequestData 全量（首拍建档）'),
    ('diff', lc.C_KV_S, lc.C_KV_F, 'CachedRequestData 差量'),
    ('fin', lc.C_ABORT, '#ffffff', 'finished_req_ids（完结通知）'),
    ('rsm', lc.C_ZMQ_S, lc.C_ZMQ_F, 'resumed：块号整体替换'),
]
for kind, s, f, name in items:
    if kind == 'fin':
        lc.rect(lx, LEG_Y - 9, 20, 12, f, s, rx=3, sw=1.2, dash=True)
    else:
        lc.rect(lx, LEG_Y - 9, 20, 12, f, s, rx=3, sw=1.3)
    lc.text(lx + 25, LEG_Y + 1, name, 8.5, lc.C_TXT, 'start', maxw=260, tag='leg:' + kind)
    lx += 25 + lc.tw(name, 8.5) + 20
lc.text(lx, LEG_Y + 1, '票据宽窄 ∝ 载荷量（示意）', 8.5, lc.C_MUTE, 'start', maxw=210, tag='leg:note')

# 页脚
lc.text(MX, LEG_Y + 24, '逐字锚 vllm/v1/core/sched/output.py:L118-L121（resumed 语义）· L193-L205（new/cached 注释）· vllm/v1/worker/gpu_model_runner.py:L1441-L1452（resumed 消费点）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 40, '五拍载荷读数取自精简版 companion host 实测的五拍下发载荷记录（新/差量/finished/resumed 逐字段）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch18-fig-diff-protocol.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
