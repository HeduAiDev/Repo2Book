#!/usr/bin/env python3
"""ch12 机制图 5 · 一个采样 token 的两条消费路径（figure_spec ch12-fig-token-two-paths，模板 tensor-flow）

放大自 L0 执行臂列（gpu_column）与循环框的接缝——即本章 L2 章图 south
『worker 影子 · token 不落 CPU』框的数据流展开：一个采样 token 从采样器出发的两条消费路径。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：一个采样 token 有两个消费者走两条路——留 GPU 的 prev_sampled_token_ids
（与采样张量 is 同一对象、零拷贝）喂下一拍前向，copy stream 异步 D2H 出门给用户；
CPU 侧账本行只写 −1 占位、行长照走。

数字全部取自 figure_spec.numbers（快路 is 同一对象=True、token_ids_cpu 行 [1,2,3,-1]、
num_tokens_no_spec 3→4；慢路 e2e 拍2 pop 批A 时 D2H pending=True、交货 [7] 后才能判停；
闭环 input_ids.gpu 首位=7、token_ids_cpu 同期 [1,2,-1,-1]；同步对照 [1,2,3,9]、prev=None；
discard 槽位表 {req-0:0, req-2:2}、invalid 行不写占位）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 810
MX, BXR = 60, 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '一个采样 token 的两条路：留 GPU 的喂下一拍前向，异步 D2H 的出门给用户',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, '『采样 token 不落 CPU』不是比喻——prev_sampled_token_ids 与采样张量 is 同一对象'
        '（源码注释 avoid CPU sync），CPU 侧账本只写 −1 占位、行长照走',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 worker 影子框 · L0：执行臂列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 源头：采样输出张量（GPU） ----------------
SRC_X, SRC_Y, SRC_W, SRC_H = 545, 92, 410, 78
lc.rect(SRC_X, SRC_Y, SRC_W, SRC_H, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.8)
lc.text(SRC_X + SRC_W / 2, SRC_Y + 21, '采样器输出张量 sampled_token_ids [[ 9 ]]（GPU 张量）',
        10.5, lc.C_GPU_S, 'middle', True, maxw=SRC_W - 20, tag='src:t')
lc.text(SRC_X + SRC_W / 2, SRC_Y + 40, '_bookkeeping_sync 的 async 分支（gpu_model_runner.py:L3797-L3842）',
        8.4, '#334155', 'middle', maxw=SRC_W - 20, tag='src:sub')
lc.text(SRC_X + SRC_W / 2, SRC_Y + 59, '同一托盘不改嫁、零搬运——两个消费者各取所需',
        8.4, lc.C_MUTE, 'middle', maxw=SRC_W - 20, tag='src:sub2')

# ---------------- 两条泳道 ----------------
LANE_Y0, LANE_H = 210, 210
LANE_W, LANE_GAP = 672, 36
FX = MX            # 快路左缘
SX = MX + LANE_W + LANE_GAP   # 慢路左缘

# ---- 快路（GPU→GPU，绿） ----
lc.rect(FX, LANE_Y0, LANE_W, LANE_H, '#ffffff', lc.C_GPU_S, rx=8, sw=1.6)
lc.text(FX + 16, LANE_Y0 + 22, '快路 · 留 GPU（零拷贝）——喂下一拍前向', 11, lc.C_GPU_S, 'start',
        True, maxw=LANE_W - 30, tag='f:t')
FAST_BOXES = [
    ('prev_sampled_token_ids', ['与采样张量 is 同一对象 = True', '整张张量原样缓存在 GPU', '（L3802 注释：avoid CPU sync）']),
    ('下一拍 input_ids.gpu', ['首位 = 7（批A 采出的 token', '直接变成批B 的输入）', 'CPU 侧同期行 = [1,2,-1,-1]']),
]
fbx = FX + 18
fbw = (LANE_W - 36 - 20) / 2
for i, (t, lines) in enumerate(FAST_BOXES):
    bx = fbx + i * (fbw + 20)
    lc.rect(bx, LANE_Y0 + 38, fbw, 118, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.2)
    lc.text(bx + fbw / 2, LANE_Y0 + 58, t, 9.8, lc.C_GPU_S, 'middle', True, maxw=fbw - 12, tag='fb:' + t[:8])
    for j, ln in enumerate(lines):
        lc.text(bx + fbw / 2, LANE_Y0 + 78 + j * 16, ln, 8.5, '#334155', 'middle',
                maxw=fbw - 12, tag='fb:%s%d' % (t[:8], j))
    if i == 0:
        lc.seg(bx + fbw + 2, LANE_Y0 + 97, bx + fbw + 18, LANE_Y0 + 97, lc.C_GPU_S, 2.0, 'std')
lc.text(FX + LANE_W / 2, LANE_Y0 + 180, '闭环：t7 从 GPU 直达 GPU（回填三岔口见下一图）——没有一次 D2H/H2D 为它发生',
        8.6, lc.C_MUTE, 'middle', maxw=LANE_W - 30, tag='f:note')
# 源头 → 快路 箭头
lc.seg(FX + LANE_W / 2, SRC_Y + SRC_H + 2, FX + LANE_W / 2, LANE_Y0 - 4, lc.C_GPU_S, 2.2, 'std')
lc.text(FX + LANE_W / 2 + 8, SRC_Y + SRC_H + 16, 'is 同一对象', 8.6, lc.C_GPU_S, 'start', maxw=110, tag='f:lbl')

# ---- 慢路（异步 D2H，橙） ----
lc.rect(SX, LANE_Y0, LANE_W, LANE_H, '#ffffff', lc.C_ENG_S, rx=8, sw=1.6)
lc.text(SX + 16, LANE_Y0 + 22, '慢路 · copy stream 异步 D2H——出门给用户（判停/回扣）', 11, lc.C_ENG_S,
        'start', True, maxw=LANE_W - 30, tag='s:t')
SLOW_BOXES = [
    ('AsyncGPUModelRunnerOutput', ['构造即发起异步 D2H', '+ 记录拷贝事件 event', '（不等、不阻塞）']),
    ('EngineCore future.result()', ['只等这个拷贝事件', 'e2e 拍2 pop 批A 时', 'D2H 事件 pending=True']),
    ('update_from_output', ['交货 [7] 到账', '才能推进状态机 / 判停', '（LENGTH 终态在此判）']),
]
sbx = SX + 18
sbw = (LANE_W - 36 - 2 * 16) / 3
for i, (t, lines) in enumerate(SLOW_BOXES):
    bx = sbx + i * (sbw + 16)
    lc.rect(bx, LANE_Y0 + 38, sbw, 118, lc.C_ENG_F, lc.C_ENG_S, rx=6, sw=1.2)
    lc.text(bx + sbw / 2, LANE_Y0 + 58, t, 9.2, lc.C_ENG_S, 'middle', True, maxw=sbw - 10, tag='sb:' + t[:8])
    for j, ln in enumerate(lines):
        lc.text(bx + sbw / 2, LANE_Y0 + 78 + j * 16, ln, 8.4, '#334155', 'middle',
                maxw=sbw - 10, tag='sb:%s%d' % (t[:8], j))
    if i < 2:
        lc.seg(bx + sbw + 2, LANE_Y0 + 97, bx + sbw + 14, LANE_Y0 + 97, lc.C_ENG_S, 2.0, 'std')
lc.text(SX + LANE_W / 2, LANE_Y0 + 180, 'AsyncOutputFuture 只等拷贝事件、不等计算——CPU 世界慢半拍，占位数（上图账本）就是这半拍的记账',
        8.6, lc.C_MUTE, 'middle', maxw=LANE_W - 30, tag='s:note')
# 源头 → 慢路 箭头
lc.seg(SX + LANE_W / 2, SRC_Y + SRC_H + 2, SX + LANE_W / 2, LANE_Y0 - 4, lc.C_ENG_S, 2.2, 'std')
lc.text(SX + LANE_W / 2 + 8, SRC_Y + SRC_H + 16, 'copy stream D2H', 8.6, lc.C_ENG_S, 'start', maxw=130,
        tag='s:lbl')

# ---------------- CPU 侧账本条（两对照） ----------------
CB_Y, CB_H = LANE_Y0 + LANE_H + 18, 150
lc.rect(MX, CB_Y, 1380, CB_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.3)
lc.text(MX + 16, CB_Y + 20, 'CPU 侧账本 token_ids_cpu 的行——async 写 −1 占位、行长照走；同步分支才写真 token',
        9.8, lc.C_TXT, 'start', True, maxw=900, tag='cb:t')
CW2, CH2, CG2 = 58, 30, 5
ROW_Y = CB_Y + 40
# async 行
lc.text(MX + 22, ROW_Y + 20, 'async 分支', 9, lc.C_GPU_S, 'start', True, maxw=80, tag='r:a')
async_row = [('1', 'real'), ('2', 'real'), ('3', 'real'), ('−1', 'ph'), ('0', 'empty')]
ax0 = MX + 110
for k, (v, kind) in enumerate(async_row):
    cx = ax0 + k * (CW2 + CG2)
    if kind == 'real':
        lc.rect(cx, ROW_Y, CW2, CH2, '#ffffff', lc.C_GPU_S, rx=4, sw=1.2)
        lc.text(cx + CW2 / 2, ROW_Y + 20, v, 10, lc.C_GPU_S, 'middle', True, tag='ar' + str(k))
    elif kind == 'ph':
        lc.rect(cx, ROW_Y, CW2, CH2, '#fff7ed', lc.C_BEAT_S, rx=4, sw=1.3)
        lc.seg(cx + 5, ROW_Y + CH2 - 5, cx + CW2 - 5, ROW_Y + 5, lc.C_BEAT_S, 1.4)
        lc.text(cx + CW2 / 2, ROW_Y + 20, v, 10, lc.C_BEAT_T, 'middle', True, tag='ar' + str(k))
    else:
        lc.rect(cx, ROW_Y, CW2, CH2, '#ffffff', '#cbd5e1', rx=4, sw=1.0, dash=True)
        lc.text(cx + CW2 / 2, ROW_Y + 20, v, 10, '#94a3b8', 'middle', tag='ar' + str(k))
lc.text(ax0 + 5 * (CW2 + CG2) + 12, ROW_Y + 20, '位置 3 是占位（真 token 9 在 GPU 张量里）· is_token_ids=True · num_tokens_no_spec 3→4 行长照走',
        8.6, '#334155', 'start', maxw=560, tag='r:anote')
# 同步行
ROW_Y2 = ROW_Y + 46
lc.text(MX + 22, ROW_Y2 + 20, '同步对照', 9, lc.C_API_S, 'start', True, maxw=80, tag='r:s')
sync_row = [('1', 'real'), ('2', 'real'), ('3', 'real'), ('9', 'real'), ('0', 'empty')]
sx0 = MX + 110
for k, (v, kind) in enumerate(sync_row):
    cx = sx0 + k * (CW2 + CG2)
    if kind == 'real':
        hot = (v == '9')
        lc.rect(cx, ROW_Y2, CW2, CH2, '#eff6ff' if hot else '#ffffff', lc.C_API_S, rx=4, sw=1.3 if hot else 1.2)
        lc.text(cx + CW2 / 2, ROW_Y2 + 20, v, 10, lc.C_API_S, 'middle', True, tag='sr' + str(k))
    else:
        lc.rect(cx, ROW_Y2, CW2, CH2, '#ffffff', '#cbd5e1', rx=4, sw=1.0, dash=True)
        lc.text(cx + CW2 / 2, ROW_Y2 + 20, v, 10, '#94a3b8', 'middle', tag='sr' + str(k))
lc.text(sx0 + 5 * (CW2 + CG2) + 12, ROW_Y2 + 20, '真 token 9 落 CPU（_to_list——那个 9 就是走过 PCIe 的货号）· prev_sampled_token_ids = None',
        8.6, '#334155', 'start', maxw=560, tag='r:snote')
lc.text(MX + 690, ROW_Y2 + 44, '分叉点 = use_async_scheduling 判定（gpu_model_runner.py:L3796 / L3822）——同一输入、两种账本',
        8.4, lc.C_MUTE, 'middle', maxw=1000, tag='r:fork')

# ---------------- discard 场景小条 ----------------
DC_Y = CB_Y + CB_H + 14
DC_H = 118
lc.rect(MX, DC_Y, 1380, DC_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.2, dash=True)
lc.text(MX + 16, DC_Y + 19, '乐观纠错下游 · discard 行不进槽位表：3 请求各 1 prompt、req-1 行被 discard（optimistic_seq_lens < num_tokens 的行不采样）',
        9.2, lc.C_TXT, 'start', True, maxw=1200, tag='dc:t')
DCW, DCH, DCG = 96, 24, 8
DY = DC_Y + 32
dc_rows = [('req-0', [('1', 'real'), ('−1', 'ph')], '槽位 0', True),
           ('req-1', [('2', 'keep'), ('0', 'empty')], '不进表（未动）', False),
           ('req-2', [('3', 'real'), ('−1', 'ph')], '槽位 2', True)]
dx0 = MX + 30
for i, (rid, cells, tagtxt, valid) in enumerate(dc_rows):
    qx = dx0 + i * (DCW * 2 + DCG + 150)
    lc.text(qx, DY - 4, rid, 8.6, lc.C_GPU_S if valid else '#94a3b8', 'start', True, maxw=60,
            tag='dc:' + rid)
    for k, (v, kind) in enumerate(cells):
        cx = qx + 52 + k * (DCW - 30 + DCG)
        if kind == 'ph':
            lc.rect(cx, DY, DCW - 30, DCH, '#fff7ed', lc.C_BEAT_S, rx=3, sw=1.2)
            lc.seg(cx + 4, DY + DCH - 4, cx + DCW - 34, DY + 4, lc.C_BEAT_S, 1.2)
            lc.text(cx + (DCW - 30) / 2, DY + 16, v, 8.6, lc.C_BEAT_T, 'middle', True, tag='dc:%s%d' % (rid, k))
        elif kind == 'empty':
            lc.rect(cx, DY, DCW - 30, DCH, '#ffffff', '#cbd5e1', rx=3, sw=1.0, dash=True)
            lc.text(cx + (DCW - 30) / 2, DY + 16, v, 8.6, '#94a3b8', 'middle', tag='dc:%s%d' % (rid, k))
        else:
            lc.rect(cx, DY, DCW - 30, DCH, '#ffffff', lc.C_GPU_S if kind == 'real' else '#94a3b8',
                    rx=3, sw=1.1)
            lc.text(cx + (DCW - 30) / 2, DY + 16, v, 8.6, lc.C_GPU_S if kind == 'real' else '#94a3b8',
                    'middle', tag='dc:%s%d' % (rid, k))
    lc.text(qx + 52 + 2 * (DCW - 30 + DCG) + 6, DY + 16, tagtxt, 8.4,
            lc.C_GPU_S if valid else '#94a3b8', 'start', maxw=140, tag='dc:tag' + rid)
lc.text(MX + 22, DC_Y + 92, '槽位表（prev_req_id_to_index）= {req-0: 0, req-2: 2}——row1 不进表、invalid 行不写占位（sampled_ids=None → continue），行长 [2,1,2]：有效行 1→2、invalid 行不动',
        8.4, '#334155', 'start', maxw=1336, tag='dc:tbl')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = DC_Y + DC_H + 24
lx = MX
lc.rect(lx, LEG_Y - 9, 22, 13, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.2)
lc.text(lx + 28, LEG_Y + 1, 'GPU 侧 / 留 GPU', 8.5, lc.C_TXT, 'start', maxw=160, tag='leg:gpu')
lx += 28 + lc.tw('GPU 侧 / 留 GPU', 8.5) + 16
lc.rect(lx, LEG_Y - 9, 22, 13, lc.C_ENG_F, lc.C_ENG_S, rx=3, sw=1.2)
lc.text(lx + 28, LEG_Y + 1, 'D2H / 引擎侧', 8.5, lc.C_TXT, 'start', maxw=160, tag='leg:eng')
lx += 28 + lc.tw('D2H / 引擎侧', 8.5) + 16
lc.rect(lx, LEG_Y - 9, 22, 13, '#fff7ed', lc.C_BEAT_S, rx=3, sw=1.2)
lc.seg(lx + 3, LEG_Y + 1, lx + 19, LEG_Y - 7, lc.C_BEAT_S, 1.2)
lc.text(lx + 28, LEG_Y + 1, '占位 −1（行长照走）', 8.5, lc.C_TXT, 'start', maxw=180, tag='leg:ph')
lx += 28 + lc.tw('占位 −1（行长照走）', 8.5) + 16
lc.rect(lx, LEG_Y - 9, 22, 13, '#ffffff', '#cbd5e1', rx=3, sw=1.0, dash=True)
lc.text(lx + 28, LEG_Y + 1, '未记（0）', 8.5, lc.C_TXT, 'start', maxw=120, tag='leg:empty')
lx += 28 + lc.tw('未记（0）', 8.5) + 16
lc.rect(lx, LEG_Y - 9, 22, 13, '#eff6ff', lc.C_API_S, rx=3, sw=1.2)
lc.text(lx + 28, LEG_Y + 1, '同步分支真 token 落 CPU', 8.5, lc.C_TXT, 'start', maxw=220, tag='leg:sync')

lc.text(MX, LEG_Y + 28, '逐字锚 vllm/v1/worker/gpu_model_runner.py:L3797-L3842（_bookkeeping_sync 的 async 分支 / L3802 注释 / L3821-L3842 写回循环）· '
        'vllm/v1/worker/gpu_input_batch.py:L309-L316（prev 槽位表影子字段）· 数字取自配套精简版 host 实跑（单请求影子 + discard + 同步对照 + e2e 闭环）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch12-fig-token-two-paths.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
