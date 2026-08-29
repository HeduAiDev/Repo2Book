#!/usr/bin/env python3
"""ch16 机制图 1 · role-split（figure_spec ch16-fig-role-split，模板 swimlane）

放大自 L0「KV 账本列 × GPU 列之间的那条进程边界」（本章 l0_zoom「KV 账本+边界」）、
L2 站 1-2（装配·两个进程各建一份）。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：同一个 KVConnectorBase_V1 类按 KVConnectorRole 分居调度器进程与 worker 进程、
分开构建零共享——调度器侧实例只产『搬运计划』（不透明 KVConnectorMetadata 随
SchedulerOutput 过线、不许改 scheduler_output），worker 侧实例只认 block_ids+注册的
池张量搬运，回程只有 KVConnectorOutput。

数字全部取自 figure_spec.numbers：SCHEDULER=0/WORKER=1（base.py:L124-L130）、
调度器侧原语 5 条 / worker 侧原语 7 条（base.py:L3-L41 docstring 两半清单）、
NOTE 原话『We build separately to enforce strict separation』（factory.py:L74）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX = 54
BXR = 1446

# ---------------- 标题区 ----------------
lc.text(MX, 36, '同一个类，两份实例：决策与搬运分居两个进程', 16.5, lc.C_TXT, 'start', True,
        maxw=900, tag='title')
lc.text(MX, 60, 'KVConnectorRole.SCHEDULER=0 / WORKER=1（base.py:L124-L130）· 工厂按角色分别构建：'
                '『We build separately to enforce strict separation』（factory.py:L74）',
        10.5, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
_ch = '放大自 L2 站 1-2 装配 · L0：KV 账本列 × GPU 列之间的进程边界'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 工厂带 ----------------
FACT_X, FACT_Y, FACT_W, FACT_H = 470, 84, 560, 88
lc.rect(FACT_X, FACT_Y, FACT_W, FACT_H, '#ffffff', lc.C_TXT, rx=8, sw=1.8)
lc.text(FACT_X + FACT_W / 2, FACT_Y + 24, 'factory.create_connector(config, role, kv_cache_config)',
        12.5, lc.C_TXT, 'middle', True, maxw=FACT_W - 24, tag='fact:t')
lc.text(FACT_X + FACT_W / 2, FACT_Y + 46, '同一个类 · 按角色各建一份：调度器侧 scheduler.py:L125-L130，worker 侧 gpu_worker.py:L662',
        9, '#334155', 'middle', maxw=FACT_W - 24, tag='fact:l1')
lc.text(FACT_X + FACT_W / 2, FACT_Y + 66, 'NOTE(Kuntai)：v1 connector 显式拆成两个角色——『We build separately to enforce strict separation』',
        9, lc.C_MUTE, 'middle', maxw=FACT_W - 24, tag='fact:l2')

# ---------------- 双泳道 ----------------
LANE_Y, LANE_H = 226, 560
LX, LW = MX, 560            # 左泳道（调度器进程）
RX, RW = 886, 560           # 右泳道（worker 进程）
GAP_X0, GAP_X1 = LX + LW, RX          # 中缝 614..886
MIDX = (GAP_X0 + GAP_X1) / 2

lc.rect(LX, LANE_Y, LW, LANE_H, lc.C_KV_F, lc.C_KV_S, rx=10, sw=2.0)
lc.text(LX + 16, LANE_Y + 26, '调度器进程 · 决策侧', 13, lc.C_KV_S, 'start', True,
        maxw=300, tag='lane:l:t')
lc.text(LX + 16, LANE_Y + 46, '查账 · 记账 · 下搬运单——绝不碰 GPU 张量', 9.5, lc.C_MUTE, 'start',
        maxw=340, tag='lane:l:s')
lc.rect(LX + LW - 158, LANE_Y + 10, 148, 22, '#ffffff', lc.C_KV_S, rx=9, sw=1.1)
lc.text(LX + LW - 84, LANE_Y + 25, 'role=SCHEDULER=0', 9.5, lc.C_KV_S, 'middle', True,
        maxw=138, tag='lane:l:role')

lc.rect(RX, LANE_Y, RW, LANE_H, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.0)
lc.text(RX + 16, LANE_Y + 26, 'worker 进程 · 搬运侧（GPU 执行臂）', 13, lc.C_GPU_S, 'start', True,
        maxw=330, tag='lane:r:t')
lc.text(RX + 16, LANE_Y + 46, '只认单子上的门牌号：block_ids + 注册的池张量', 9.5, lc.C_MUTE,
        'start', maxw=340, tag='lane:r:s')
lc.rect(RX + RW - 138, LANE_Y + 10, 128, 22, '#ffffff', lc.C_GPU_S, rx=9, sw=1.1)
lc.text(RX + RW - 74, LANE_Y + 25, 'role=WORKER=1', 9.5, lc.C_GPU_S, 'middle', True,
        maxw=118, tag='lane:r:role')

# 实例框
INST_Y, INST_H = LANE_Y + 62, 54
lc.rect(LX + 20, INST_Y, LW - 40, INST_H, '#ffffff', lc.C_KV_S, rx=7, sw=1.4)
lc.text(LX + 20 + (LW - 40) / 2, INST_Y + 22, 'KVConnectorBase_V1 实例（调度器侧）', 11.5,
        lc.C_TXT, 'middle', True, maxw=LW - 80, tag='inst:l')
lc.text(LX + 20 + (LW - 40) / 2, INST_Y + 41, 'bind_gpu_block_pool 直读块池元数据（base.py:L455-L464）',
        8.5, lc.C_MUTE, 'middle', maxw=LW - 80, tag='inst:l:s')
lc.rect(RX + 20, INST_Y, RW - 40, INST_H, '#ffffff', lc.C_GPU_S, rx=7, sw=1.4)
lc.text(RX + 20 + (RW - 40) / 2, INST_Y + 22, 'KVConnectorBase_V1 实例（worker 侧）', 11.5,
        lc.C_TXT, 'middle', True, maxw=RW - 80, tag='inst:r')
lc.text(RX + 20 + (RW - 40) / 2, INST_Y + 41, 'register_kv_caches 注册池张量（base.py:L263-L272）',
        8.5, lc.C_MUTE, 'middle', maxw=RW - 80, tag='inst:r:s')

# 工厂 → 两实例的装配箭头（贴边：起点=工厂框底边，终点=实例框顶边）
FORK_Y = 206
lc.parrow([(FACT_X + 90, FACT_Y + FACT_H), (FACT_X + 90, FORK_Y), (LX + LW / 2, FORK_Y),
           (LX + LW / 2, INST_Y)], lc.C_KV_S, 1.8, 'std')
lc.parrow([(FACT_X + FACT_W - 90, FACT_Y + FACT_H), (FACT_X + FACT_W - 90, FORK_Y),
           (RX + RW / 2, FORK_Y), (RX + RW / 2, INST_Y)], lc.C_GPU_S, 1.8, 'std')

# 原语清单框
PRIM_Y = INST_Y + INST_H + 18
ROW_H = 30
L_PRIMS = [
    ('get_num_new_matched_tokens()', '查外部缓存还能给多少 token（可答 None=稍后再问）'),
    ('update_state_after_alloc()', '分配后对账：告知本拍外部加载量'),
    ('update_connector_output()', '收 worker 回执，更新状态'),
    ('request_finished()', '终局交接：块现在放、还是归 connector 管'),
    ('take_events()', '取走新收集的 KV 事件'),
]
R_PRIMS = [
    ('handle_preemptions()', '抢占 / 驱逐前抢救将被覆写的块'),
    ('start_load_kv()', '前向开始前：异步发起全部层加载'),
    ('wait_for_layer_load()', '第 i 层注意力前：只等本层 KV 到位'),
    ('save_kv_layer()', '第 i 层注意力后：异步存出本层'),
    ('wait_for_save()', '栅栏：等全部存完（防 paged buffer 覆写）'),
    ('get_finished()', '上报收 / 发完成的请求与失败块'),
    ('build_connector_worker_meta()', '组回程 worker 元数据'),
]


def prim_box(x, w, y0, title, prims, stroke):
    h = 30 + len(prims) * ROW_H + 6
    lc.rect(x, y0, w, h, '#ffffff', stroke, rx=7, sw=1.3)
    lc.text(x + 14, y0 + 20, title, 10, stroke, 'start', True, maxw=w - 28, tag='pb:t')
    for i, (name, duty) in enumerate(prims):
        ry = y0 + 30 + i * ROW_H
        if i:
            lc.seg(x + 12, ry, x + w - 12, ry, '#e2e8f0', 1.0)
        lc.text(x + 14, ry + 19, name, 9.5, lc.C_TXT, 'start', True, maxw=250, tag='pb:n')
        lc.text(x + w - 14, ry + 19, duty, 8.5, '#334155', 'end', maxw=w - 285, tag='pb:d')
    return y0 + h


prim_end_l = prim_box(LX + 20, LW - 40, PRIM_Y, '原语 5 条（docstring 上半 · base.py:L8-L24）',
                      L_PRIMS, lc.C_KV_S)
prim_end_r = prim_box(RX + 20, RW - 40, PRIM_Y, '原语 7 条（docstring 下半 · base.py:L26-L41）',
                      R_PRIMS, lc.C_GPU_S)

# 职责注（泳道底部）
NOTE_YL = prim_end_l + 16
lc.rect(LX + 20, NOTE_YL, LW - 40, 52, 'none', lc.C_FAINT, rx=7, sw=1.1, dash=True)
lc.text(LX + 34, NOTE_YL + 20, '「同一间总部办公室」：查目录（外部缓存=第二个前缀缓存）、', 8.5,
        lc.C_MUTE, 'start', maxw=LW - 60, tag='note:l1')
lc.text(LX + 34, NOTE_YL + 38, '记账（块池元数据）、终局交接——两间办公室零通话零共享', 8.5,
        lc.C_MUTE, 'start', maxw=LW - 60, tag='note:l2')
NOTE_YR = prim_end_r + 16
lc.rect(RX + 20, NOTE_YR, RW - 40, 52, 'none', lc.C_FAINT, rx=7, sw=1.1, dash=True)
lc.text(RX + 34, NOTE_YR + 20, '「同一间分部办公室」：拿到门牌号（block_ids）与楼层平面图', 8.5,
        lc.C_MUTE, 'start', maxw=RW - 60, tag='note:r1')
lc.text(RX + 34, NOTE_YR + 38, '（注册的池张量）直接进屋搬运——不过前台、不走中间层', 8.5,
        lc.C_MUTE, 'start', maxw=RW - 60, tag='note:r2')

# ---------------- 中缝：进程边界 + 两封信 ----------------
lc.seg(MIDX, LANE_Y + 6, MIDX, LANE_Y + LANE_H - 6, lc.C_MUTE, 1.4, dash=True)
lc.text(MIDX, LANE_Y + LANE_H + 18, '进程边界', 9.5, lc.C_MUTE, 'middle', maxw=100, tag='pb-line')

# 下行信：KVConnectorMetadata（调度器 → worker，随 SchedulerOutput 过线）
A1_Y = 380
lc.text(MIDX, A1_Y - 66, 'KVConnectorMetadata', 10, lc.C_KV_S, 'middle', True, maxw=GAP_X1 - GAP_X0, tag='ltr1')
lc.text(MIDX, A1_Y - 50, '不透明搬运计划：', 8.5, '#334155', 'middle', maxw=GAP_X1 - GAP_X0, tag='ltr1s')
lc.seg(GAP_X0, A1_Y, GAP_X1 - 4, A1_Y, lc.C_KV_S, 2.2, 'std')
lc.text(MIDX, A1_Y + 16, '随 SchedulerOutput 过线', 8.5, lc.C_KV_S, 'middle', maxw=GAP_X1 - GAP_X0, tag='ltr1a')
lc.text(MIDX, A1_Y + 30, '不许改 scheduler_output', 8.5, lc.C_KV_S, 'middle', maxw=GAP_X1 - GAP_X0, tag='ltr1b')

# 上行信：KVConnectorOutput（worker → 调度器）
A2_Y = 520
lc.text(MIDX, A2_Y - 66, 'KVConnectorOutput', 10, lc.C_GPU_S, 'middle', True, maxw=GAP_X1 - GAP_X0, tag='ltr2')
lc.text(MIDX, A2_Y - 50, '完成 / 失败回执（唯一回信）', 8.5, '#334155', 'middle', maxw=GAP_X1 - GAP_X0, tag='ltr2s')
lc.seg(GAP_X1, A2_Y, GAP_X0 + 4, A2_Y, lc.C_GPU_S, 2.2, 'std')
lc.text(MIDX, A2_Y + 16, 'finished_recving / finished_sending', 8, lc.C_GPU_S, 'middle',
        maxw=GAP_X1 - GAP_X0, tag='ltr2a')
lc.text(MIDX, A2_Y + 30, 'invalid_block_ids（失败块）', 8, lc.C_GPU_S, 'middle',
        maxw=GAP_X1 - GAP_X0, tag='ltr2b')

# 零共享注记（中缝下半）
ZS_Y = 596
lc.rect(GAP_X0 + 8, ZS_Y, GAP_X1 - GAP_X0 - 16, 92, 'none', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(MIDX, ZS_Y + 20, '零共享状态', 9.5, lc.C_TXT, 'middle', True, maxw=GAP_X1 - GAP_X0 - 28, tag='zs:t')
lc.text(MIDX, ZS_Y + 38, '两份实例互不见面，', 8.5, lc.C_MUTE, 'middle', maxw=GAP_X1 - GAP_X0 - 28, tag='zs:l1')
lc.text(MIDX, ZS_Y + 52, '跨线的只有这两封信。', 8.5, lc.C_MUTE, 'middle', maxw=GAP_X1 - GAP_X0 - 28, tag='zs:l2')
lc.text(MIDX, ZS_Y + 70, '决策与搬运分离——', 8.5, lc.C_MUTE, 'middle', maxw=GAP_X1 - GAP_X0 - 28, tag='zs:l3')
lc.text(MIDX, ZS_Y + 84, '一份契约伺候多个后端', 8.5, lc.C_MUTE, 'middle', maxw=GAP_X1 - GAP_X0 - 28, tag='zs:l4')

# ---------------- 页脚 ----------------
FY = LANE_Y + LANE_H + 40
lc.text(MX, FY, '逐字锚 vllm/distributed/kv_transfer/kv_connector/v1/base.py:L3-L41（docstring 原语两半清单）· L124-L130（KVConnectorRole）· '
                'factory.py:L66-L74（NOTE）· scheduler.py:L125-L130 / gpu_worker.py:L662（两侧装配）· vllm/v1/outputs.py:L223-L234（KVConnectorOutput 字段）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 16, '角色色即身份：青=决策（调度器侧）/ 绿=搬运（worker 侧），与全书 L0/L2 同源 · P/D、offload 等后端本体 → 第 36-37 章（预告）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FY + 34

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch16-fig-role-split.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
