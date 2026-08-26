#!/usr/bin/env python3
"""ch10 机制图 7 · 首件全量、补件只发 diff（figure_spec ch10-fig-new-full-cached-diff，模板 before-after）

放大自 L0『调度 · 显存账本』（kv_column 青色列）上半 Scheduler 框的「增量下发
new/cached」格——即本章 L2 章图 center ⑦ 拍片『SchedulerOutput：new 全量 / cached 增量』
的机制展开（产出面 → ch18 的调度器侧半边）。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：
图右上角指北小签。

claim：同一拍里两种包裹：首次调度的 r4 发全量（64 个 prompt token + 2 个块表项 + 采样
参数），三个在途请求只发增量（各 1 个 token 记账、0 个新块、连 token 表都不带）——
worker 缓存全量、调度器只发 diff。

数字全部取自 figure_spec.numbers（精简版 companion host 实测 trace 的 wire_payload：
新 r4 prompt_token_ids 64 + block_ids 2 + 采样参数 1 份；三老请求 num_scheduled 各 1、
new_block_ids 各 0、all_token_ids 表空；注释原文 output.py:L194-L200；resumed 整体替换
output.py:L118-L121）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 596
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '首件发整箱，补件只发 diff——worker 持仓，IPC 每拍只寄真正变化的那一点',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '拍 2 的 SchedulerOutput（scheduler.py:L1131-L1163）：新到的 r4 拿到全量装备，三个 decode 老请求每人一张 1-token 记账条——块表和 token 表一概不发',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ⑦ SchedulerOutput 组装 · L0：调度账本列上半'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左栏：NewRequestData 全量货箱 ----------------
LX, LY, LW, LH = MX, 96, 660, 352
lc.rect(LX, LY, LW, LH, '#ffffff', lc.C_ENG_S, rx=9, sw=2.0)
lc.text(LX + 16, LY + 24, '首次调度 · NewRequestData.from_request(r4)', 11.5, lc.C_ENG_S,
        'start', True, maxw=LW - 32, tag='new:t')
lc.text(LX + 16, LY + 42, '拍 2 唯一的新请求——整箱装备只发这一次', 8.5, lc.C_MUTE, 'start',
        maxw=LW - 32, tag='new:s')

# 第 1 层：prompt_token_ids 条（64 格全宽）
CELL_PITCH = (LW - 32) / 64          # 9.8125
CELL_W = CELL_PITCH - 1.2
cells_x0, cells_y, CELL_H = LX + 16, LY + 76, 26
for i in range(64):
    lc.rect(cells_x0 + i * CELL_PITCH, cells_y, CELL_W, CELL_H, lc.C_KV_S, lc.C_KV_S, rx=1, sw=0)
for k in range(1, 4):                # 16-token 分块白线
    xx = cells_x0 + k * 16 * CELL_PITCH
    lc.seg(xx - 0.6, cells_y - 4, xx - 0.6, cells_y + CELL_H + 4, '#ffffff', 2.2)
lc.text(LX + 16, cells_y - 8, '① prompt_token_ids', 9.5, lc.C_TXT, 'start', True, maxw=240,
        tag='new:l1')
lc.text(cells_x0 + 32 * CELL_PITCH, cells_y + CELL_H + 16, '64 个 prompt token（全量；白线 = 16-token 块界）',
        8.5, lc.C_KV_S, 'middle', True, maxw=420, tag='new:c1n')

# 第 2 层：block_ids（2 项）
S2_Y = cells_y + CELL_H + 34
lc.text(LX + 16, S2_Y - 8, '② block_ids · 完整块表', 9.5, lc.C_TXT, 'start', True, maxw=240,
        tag='new:l2')
bx0 = LX + 16
for i in range(2):
    lc.rect(bx0 + i * 68, S2_Y, 60, 32, lc.C_KV_F, lc.C_KV_S, rx=4, sw=1.5)
    lc.text(bx0 + i * 68 + 30, S2_Y + 20, f'块 {i}', 9, lc.C_KV_S, 'middle', True, maxw=56,
            tag=f'new:blk{i}')
lc.text(bx0 + 2 * 68 + 16, S2_Y + 20, '2 项（64 token 本拍排 29：首 chunk 落进前 2 块）', 8.5,
        lc.C_MUTE, 'start', maxw=380, tag='new:blkn')

# 第 3 层：sampling_params
S3_Y = S2_Y + 50
lc.text(LX + 16, S3_Y - 8, '③ sampling_params · 采样参数 1 份', 9.5, lc.C_TXT, 'start', True,
        maxw=280, tag='new:l3')
lc.rect(bx0, S3_Y, 320, 32, lc.C_KV_F, lc.C_KV_S, rx=4, sw=1.5)
lc.text(bx0 + 160, S3_Y + 20, 'temperature / top_p / max_tokens …', 8.5, lc.C_KV_S,
        'middle', maxw=310, tag='new:sp')

# worker 仓库存档注
WY = S3_Y + 50
lc.rect(LX + 16, WY, LW - 32, 66, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(LX + 28, WY + 16, 'worker 仓库存档（从此每拍只收 diff）', 9.5, lc.C_TXT, 'start', True,
        maxw=LW - 56, tag='new:wh')
lc.text(LX + 28, WY + 34, '「We cache the request data in each worker process, so that we don\'t need to re-send it every',
        8.2, '#334155', 'start', maxw=LW - 52, tag='new:wq1')
lc.text(LX + 28, WY + 50, 'scheduling step.」——output.py:L194-L200 注释原文', 8.2, '#334155',
        'start', maxw=LW - 52, tag='new:wq2')

# ---------------- 右栏：CachedRequestData 增量票据 ----------------
RX, RY, RW, RH = 862, 96, 578, 352
lc.rect(RX, RY, RW, RH, lc.C_KV_F, lc.C_KV_S, rx=9, sw=2.0)
lc.text(RX + 16, RY + 24, '已调度 · CachedRequestData（r1 / r2 / r3）', 11.5, lc.C_KV_S, 'start',
        True, maxw=RW - 32, tag='csh:t')
lc.text(RX + 16, RY + 42, '三个 decode 老请求——worker 仓库里已有它们的全部档案', 8.5,
        lc.C_MUTE, 'start', maxw=RW - 32, tag='csh:s')

slip_y = [RY + 58, RY + 132, RY + 206]
for k, rid in enumerate(['r1', 'r2', 'r3']):
    sy = slip_y[k]
    lc.rect(RX + 16, sy, RW - 32, 62, '#ffffff', lc.C_KV_S, rx=7, sw=1.4)
    lc.rect(RX + 34, sy + 19, 14, 24, lc.C_KV_S, lc.C_KV_S, rx=2, sw=0)
    lc.text(RX + 41, sy + 13, '1 token', 7.5, lc.C_KV_S, 'middle', True, maxw=60, tag=f's:{rid}:n')
    lc.text(RX + 62, sy + 26, f'{rid} · num_scheduled = 1', 9.5, lc.C_TXT, 'start', True,
            maxw=220, tag=f's:{rid}:t')
    lc.text(RX + 62, sy + 44, 'new_block_ids = 0 项 · all_token_ids 表空', 8.5, lc.C_MUTE,
            'start', maxw=330, tag=f's:{rid}:l')
lc.text(RX + 16, RY + 280, '「Since the request\'s data is already cached in the worker processes,',
        8.2, '#334155', 'start', maxw=RW - 32, tag='csh:q1')
lc.text(RX + 16, RY + 296, 'we only send the diff to minimize the communication cost.」——output.py:L198-L200',
        8.2, '#334155', 'start', maxw=RW - 32, tag='csh:q2')
lc.text(RX + 16, RY + 314, '上拍调度过的连 all_token_ids 都省（prev_step_scheduled_req_ids 每拍刷新，L1160-L1163）',
        8.2, lc.C_MUTE, 'start', maxw=RW - 32, tag='csh:f1')
lc.text(RX + 16, RY + 330, 'resumed 特例：new_block_ids 整体替换而非追加（output.py:L118-L121）——深挖 → ch18',
        8.2, lc.C_MUTE, 'start', maxw=RW - 32, tag='csh:f2')
lc.text(RX + 16, RY + 348, '合计：3 枚 token 格 + 3 个记账数，再无别物', 9, lc.C_KV_S, 'start',
        True, maxw=RW - 32, tag='csh:sum')

# ---------------- 中缝箭头 ----------------
MID_CX = (LX + LW + RX) / 2
my = LY + LH / 2
lc.parrow([(LX + LW + 2, my - 26), (RX - 3, my - 26)], lc.C_ENG_S, 2.2, 'std')
lc.parrow([(LX + LW + 2, my + 26), (RX - 3, my + 26)], lc.C_ENG_S, 2.2, 'std')
lc.text(MID_CX, my - 42, '只发 diff', 10, lc.C_ENG_S, 'middle', True, maxw=110, tag='mid:lbl')

# ---------------- 底部：量级对比 + 图例 + 页脚 ----------------
CMP_Y = LY + LH + 28
lc.text(MX, CMP_Y, '同一拍并排可数：', 10, lc.C_TXT, 'start', True, maxw=140, tag='cmp:t')
lc.text(MX + 150, CMP_Y, '新 r4 首发全量 = 64 token + 2 块表项 + 1 份采样参数', 9.5, lc.C_ENG_S,
        'start', True, maxw=470, tag='cmp:l')
lc.text(MX + 660, CMP_Y, '三个老请求合计 = 3 ×「1 token 记账」+ 0 新块 + 空 token 表', 9.5,
        lc.C_KV_S, 'start', True, maxw=540, tag='cmp:r')

LEG_Y = CMP_Y + 30
lx = MX
items = [
    ('new', '全量件（只发一次：token / 块表 / 采样参数）'),
    ('diff', '增量件（每拍 diff：1 枚 token 格 + 记账数）'),
]
for kind, name in items:
    if kind == 'new':
        lc.rect(lx, LEG_Y - 8, 20, 12, lc.C_KV_S, lc.C_KV_S, rx=3, sw=0)
    else:
        lc.rect(lx, LEG_Y - 8, 20, 12, '#ffffff', lc.C_KV_S, rx=3, sw=1.4)
    lc.text(lx + 26, LEG_Y + 2, name, 8.5, lc.C_TXT, 'start', maxw=330, tag='leg' + kind)
    lx += 26 + lc.tw(name, 8.5) + 22
lc.text(lx, LEG_Y + 2, '虚线框 = 源码注释原文', 8.5, lc.C_MUTE, 'start', maxw=200, tag='leg:note')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/core/sched/scheduler.py:L1131-L1163（二分组装）· vllm/v1/core/sched/output.py:L193-L205（new/cached 注释）· L118-L121（resumed 整体替换）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '包裹读数取自精简版 companion host 实测的拍 2 下发载荷记录（同一拍两种包裹并排可数）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch10-fig-new-full-cached-diff.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
