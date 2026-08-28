#!/usr/bin/env python3
"""ch18 机制图 2 · 五拍调和状态表（figure_spec ch18-fig-reconcile-five-beats，模板 state-table）

放大自 L0『GPU 执行臂』（gpu_column 绿色列）『执行臂中层』GPUModelRunner 框的 center ②
拍片『_update_states · 差量调和』（站 2-5）——即本章 L2 章图 center ② 拍片的机制展开。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：五拍里同一个持久批次经历 全量建档→稳态不动→finished 打洞+新请求填洞→被抢移出
（缓存保留）→resumed 整体替换回填，批容器与 requests 缓存始终不清空重建——每拍只对
差量动手。

数字全部取自 figure_spec.numbers（traces/ch18_m02_reconcile.json 五拍 after_update /
gathered_input_ids / positions / sampled + gpu_model_runner.py:L1248-L1251 NOTE 原文）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 892
MX = 60
BXR = 1440

# 请求身份色（非架构色——图例注明）
RC = {'r1': lc.C_KV_S, 'r2': lc.C_API_S, 'r3': lc.C_ZMQ_S}

# ---------------- 标题区 ----------------
lc.text(MX, 34, '批次从不清空重建，每拍只对差量记账——五拍同一块 slot 板，三种事件全靠差量字段驱动',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '_update_states（gpu_model_runner.py:L1192-L1566）五拍实录：全量建档 → 稳态不动（只覆盖 2 个计数）→ finished 打洞+新请求填洞 → 被抢移出（快照留缓存）→ resumed 块号整体替换回填',
        10.5, lc.C_MUTE, 'start', maxw=1020, tag='subtitle')
_ch = '放大自 L2 拍片 ② _update_states 差量调和 · L0：GPU 执行臂中层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 列布局 ----------------
HDR_Y = 92
ROW_Y0, RH, N_ROW = 106, 122, 5
BOT_Y = ROW_Y0 + N_ROW * RH                     # 716

BDG_X, BDG_W = MX, 46                           # 拍徽标
PAY_X, PAY_W = 118, 286                         # 差量载荷小票
SLOT_X, CELL_W, CELL_H, CELL_GAP = 420, 72, 46, 6   # slot 板 4 格
INP_X, INP_W = 756, 372                         # 收出 input 条
CACHE_X, CACHE_W = 1150, 290                    # requests 缓存栏（右侧常驻）

for cx, cw, lab in [
    (BDG_X, BDG_W, '拍'),
    (PAY_X, PAY_W, 'SchedulerOutput 差量载荷'),
    (SLOT_X, 4 * CELL_W + 3 * CELL_GAP, '调和后批内 slot 板（4 格）'),
    (INP_X, INP_W, '本拍收出 input_ids（positions）'),
]:
    lc.text(cx + cw / 2 if cx != SLOT_X else cx + (4 * CELL_W + 3 * CELL_GAP) / 2,
            HDR_Y, lab, 9.5, lc.C_MUTE, 'middle', True, maxw=cw + 40, tag='hd:' + lab[:6])

# ---------------- requests 缓存栏（右侧常驻，五拍连续） ----------------
lc.rect(CACHE_X, ROW_Y0 - 6, CACHE_W, BOT_Y - ROW_Y0 + 12, '#ffffff', lc.C_MUTE, rx=9, sw=1.4)
lc.text(CACHE_X + CACHE_W / 2, ROW_Y0 + 10, 'requests 缓存 · worker 全量档案', 9.5, lc.C_TXT,
        'middle', True, maxw=CACHE_W - 16, tag='cache:t')
lc.text(CACHE_X + CACHE_W / 2, ROW_Y0 + 24, '（CachedRequestState，移出批次≠删快照）', 7.8, lc.C_MUTE,
        'middle', maxw=CACHE_W - 16, tag='cache:s')
CACHE_KEYS = [
    (['r1', 'r2'], '建档 ×2'),
    (['r1', 'r2'], ''),
    (['r1', 'r3'], 'r2 出缓存出批次'),
    (['r1', 'r3'], 'r3 快照仍在'),
    (['r1', 'r3'], ''),
]
for i, (keys, note) in enumerate(CACHE_KEYS):
    cy = ROW_Y0 + 34 + i * RH
    if i > 0:
        lc.seg(CACHE_X + 10, cy - 12, CACHE_X + CACHE_W - 10, cy - 12, '#e2e8f0', 1.0)
    kx = CACHE_X + 14
    for rid in ['r1', 'r2', 'r3']:
        present = rid in keys
        if present:
            lc.rect(kx, cy - 2, 44, 22, '#ffffff', RC[rid], rx=5, sw=1.5)
            lc.text(kx + 22, cy + 13, rid, 9.5, RC[rid], 'middle', True, maxw=40, tag=f'ck:{i}:{rid}')
        else:
            # 出缓存：灰 + 删除线记号
            lc.rect(kx, cy - 2, 44, 22, '#f8fafc', '#cbd5e1', rx=5, sw=1.0, dash=True)
            lc.text(kx + 22, cy + 13, rid, 9, '#cbd5e1', 'middle', maxw=40, tag=f'ck:{i}:{rid}:x')
        kx += 52
    if note:
        hot = 'r3' in note
        lc.text(kx + 8, cy + 13, note, 8.4, RC['r3'] if hot else lc.C_MUTE, 'start', True,
                maxw=CACHE_W - (kx - CACHE_X) - 20, tag=f'ck:{i}:n')

# ---------------- 五拍数据 ----------------
BEATS = [
    # (拍号, 事件名, 载荷两行, slot 内容 [(state, rid, 角标)], 收出 tokens[(rid, tok, pos)], positions 注, 采样注)
    (1, '全量建档',
     ['new 全量×2：r1（2 token+块[1]）· r2（3 token+块[2]）', 'cached 0 条 · total 5'],
     [('r1', 'r1@0 块[1]'), ('r2', 'r2@1 块[2]'), None, None],
     [('r1', '101', 0), ('r1', '102', 1), ('r2', '201', 0), ('r2', '202', 1), ('r2', '203', 2)],
     '采样 {r1:11, r2:21} → 写回行尾：[101,102,11] / [201,202,203,21]'),
    (2, '稳态不动',
     ['diff×2：r1 +块号[3] · r2 新块 0 个', 'computed ← [2,3] · total 2'],
     [('r1', 'r1@0 块[1,3]'), ('r2', 'r2@1 块[2]'), None, None],
     [('r1', '11', 2), ('r2', '21', 3)],
     '采样 {r1:12, r2:22}——positions 恰是拍 1 写回的两个 token（写回→收集闭环）'),
    (3, 'finished 打洞 + 新请求填洞',
     ['finished {r2} + new r3（2 token+块[4]）', '+ diff r1（computed←3）· total 3'],
     [('r1', 'r1@0'), ('r3', 'r3@1 块[4]'), None, None],
     [('r1', '12', 3), ('r3', '301', 0), ('r3', '302', 1)],
     '采样 {r1:13, r3:31}；洞行陈旧尾巴 [203,21,22] 不搬不动'),
    (4, '被抢移出（缓存保留）',
     ['diff×1：r1（computed←4）· total 1', 'r3 不在差量里——unscheduled 出批次'],
     [('r1', 'r1@0'), None, None, None],
     [('r1', '13', 4)],
     '采样 {r1:14}——批内只剩 r1，缓存仍 {r1,r3}'),
    (5, 'resumed 整体替换回填',
     ['diff×2：r1（computed←5）', 'r3 resumed：块[5] 替换 [4] · computed←0 · total 4'],
     [('r1', 'r1@0'), ('r3', 'r3@1 块[4]→[5]'), None, None],
     [('r1', '14', 5), ('r3', '301', 0), ('r3', '302', 1), ('r3', '31', 2)],
     '采样 {r1:15, r3:30}；r3 连自己的 output 31 一起重算'),
]

# slot 三态绘制
def slot_cell(x, y, spec):
    if spec is None:
        return
    rid, lab = spec
    parts = lab.split(' ', 1)
    lc.rect(x, y, CELL_W, CELL_H, '#ffffff', RC[rid], rx=6, sw=1.8)
    lc.text(x + CELL_W / 2, y + 19, parts[0], 10.5, RC[rid], 'middle', True, maxw=CELL_W - 6,
            tag='sc:' + rid)
    if len(parts) > 1:
        lc.text(x + CELL_W / 2, y + 37, parts[1], 7.8, '#334155', 'middle',
                maxw=CELL_W - 6, tag='sc:l:' + rid)

def _wrap_event(ev):
    return [ev] if len(ev) <= 12 else [ev[:6], ev[6:]]

def _slot_note(i):
    return ['num_reqs=2 · 两份全量建档入批',
            'num_reqs=2 · 结构零变动，r1 块表 append [3]',
            'num_reqs=2 · r3 落进 r2 让出的洞 1（pop_removed 最小空位）',
            'num_reqs=1 · r3 出批次（unscheduled）',
            'num_reqs=2 · 洞 1 回填；r3 块表行 [4]→[5] 整体替换'][i]

def _pos_note(i):
    return ['positions [0,1,0,1,2]',
            'positions [2,3]',
            'positions [3,0,1]',
            'positions [4]',
            'positions [5,0,1,2]'][i]

# ---------------- 主表逐行 ----------------
for i, (beat, ev, pay, slots, toks, smp) in enumerate(BEATS):
    ry = ROW_Y0 + i * RH
    if i > 0:
        lc.seg(MX, ry - 2, CACHE_X - 12, ry - 2, '#e2e8f0', 1.0)
    # 拍徽标 + 事件名
    lc.rect(BDG_X, ry + 16, BDG_W, 32, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.2)
    lc.text(BDG_X + BDG_W / 2, ry + 36, f'拍 {beat}', 10.5, lc.C_ENG_S, 'middle', True, maxw=44,
            tag=f'bdg{beat}')
    for li, ln in enumerate(_wrap_event(ev)):
        lc.text(BDG_X + BDG_W / 2, ry + 60 + li * 13, ln, 8.2, lc.C_MUTE, 'middle', maxw=120,
                tag=f'ev{beat}:{li}')
    # 载荷小票（青底=diff 系、橙框=含全量）
    has_full = any('new' in p for p in pay)
    lc.rect(PAY_X, ry + 12, PAY_W, 50, lc.C_ENG_F if has_full else lc.C_KV_F,
            lc.C_ENG_S if has_full else lc.C_KV_S, rx=6, sw=1.4)
    for li, ln in enumerate(pay):
        lc.text(PAY_X + PAY_W / 2, ry + 31 + li * 19, ln, 8.8, '#334155', 'middle',
                maxw=PAY_W - 12, tag=f'pay{beat}:{li}')
    # slot 板
    for k, spec in enumerate(slots):
        slot_cell(SLOT_X + k * (CELL_W + CELL_GAP), ry + 24, spec)
    lc.text(SLOT_X + (4 * CELL_W + 3 * CELL_GAP) / 2, ry + 92, _slot_note(i), 8.2, lc.C_MUTE,
            'middle', maxw=4 * CELL_W + 3 * CELL_GAP, tag=f'sn{i}')
    # 收出 token 条 + positions
    cw_, chh = 34, 26
    for k, (rid, tok, pos) in enumerate(toks):
        tx = INP_X + k * (cw_ + 6)
        lc.rect(tx, ry + 20, cw_, chh, '#ffffff', RC[rid], rx=4, sw=1.3)
        lc.text(tx + cw_ / 2, ry + 37, tok, 9.3, RC[rid], 'middle', True, maxw=cw_ - 4,
                tag=f'tk{beat}:{k}')
        lc.text(tx + cw_ / 2, ry + 58, str(pos), 8, lc.C_MUTE, 'middle', maxw=cw_, tag=f'ps{beat}:{k}')
    lc.text(INP_X + len(toks) * (cw_ + 6) + 10, ry + 37, _pos_note(i), 8.4, lc.C_MUTE, 'start',
            maxw=INP_X + INP_W - (INP_X + len(toks) * (cw_ + 6)) - 12, tag=f'pn{i}')
    # 采样→写回注
    lc.text(INP_X, ry + 86, smp, 8.2, '#334155', 'start', maxw=INP_W, tag=f'smp{beat}')

# ---------------- 底部：NOTE 赌注 + 图例 ----------------
NT_Y = BOT_Y + 18
lc.rect(MX, NT_Y, 880, 74, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(MX + 14, NT_Y + 17, '核心假设自白 · NOTE(woosuk)（vllm/v1/worker/gpu_model_runner.py:L1248-L1251）', 9.3,
        lc.C_TXT, 'start', True, maxw=840, tag='note:t')
lc.text(MX + 14, NT_Y + 36, '「The persistent batch optimization assumes that consecutive batches contain mostly the same requests. If batches have low request', 8.2, '#334155',
        'start', maxw=850, tag='note:l1')
lc.text(MX + 14, NT_Y + 52, 'overlap (e.g., alternating between two distinct sets of requests), this optimization becomes very inefficient.」', 8.2,
        '#334155', 'start', maxw=850, tag='note:l2')
lc.text(MX + 14, NT_Y + 68, '消费点：_update_states L1192-L1566——差量字段驱动全部五类批结构变动。', 8.2, lc.C_MUTE,
        'start', maxw=850, tag='note:l3')

SUM_X = MX + 900
lc.text(SUM_X, NT_Y + 12, '五拍批布局：[r1,r2] → [r1,r2] → [r1,r3]（填洞）→ [r1] → [r1,r3]', 9.5,
        lc.C_TXT, 'start', True, maxw=BXR - SUM_X, tag='sum:1')
lc.text(SUM_X, NT_Y + 32, '批容器与 requests 缓存五拍从不清空重建——每拍只对差量动手。', 9,
        lc.C_GPU_S, 'start', True, maxw=BXR - SUM_X, tag='sum:2')

LEG_Y = NT_Y + 92
lx = MX
for rid in ['r1', 'r2', 'r3']:
    lc.rect(lx, LEG_Y - 9, 20, 12, '#ffffff', RC[rid], rx=3, sw=1.4)
    lc.text(lx + 25, LEG_Y + 1, rid, 8.5, RC[rid], 'start', True, maxw=30, tag='leg:' + rid)
    lx += 25 + lc.tw(rid, 8.5, True) + 16
lc.text(lx, LEG_Y + 1, '= 请求身份色（非架构色）· 票据底色 橙=含全量 / 青=纯差量', 8.5, lc.C_MUTE,
        'start', maxw=430, tag='leg:note')
lc.text(MX, LEG_Y + 20, '逐字锚 vllm/v1/worker/gpu_model_runner.py:L1192-L1566（_update_states）· L1203-L1253（移除）· L1295-L1309（建快照）· L1441-L1474（老请求调和/resumed）· L1509-L1520（收尾落位）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 36, '五拍读数取自精简版 companion host 实测（调和后批次布局 / 收出 input 与 positions / 采样写回逐拍记录）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch18-fig-reconcile-five-beats.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
