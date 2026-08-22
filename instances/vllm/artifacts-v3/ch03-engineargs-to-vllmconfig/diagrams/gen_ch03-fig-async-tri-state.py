#!/usr/bin/env python3
"""ch03 机制图 1 · async_scheduling 三态决策（figure_spec ch03-fig-async-tri-state，模板 state-machine）

放大自 L0 启动视角（boot）第 11 站——即本章 L2 章图 center 拍片 ⑤ 『async 三态决策』的机制展开。
架构归属回指 L2/L0（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：async_scheduling 进场为 None(默认)时五类不兼容条件全不命中则落到 True，显式 True 对同类
不兼容直接 raise，显式 False 整段跳过——同一棵决策树、三种进场值、三种纪律。

数字全部取自 figure_spec.numbers（五场景 host 实测 trace + pin 锚点）；坐标由常量/循环计算；
文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

# 追加两个语义 marker（绿=能力反问 / 暖=推导主线）——沿用 l0_common 配色常量，不另造色值
DEFS = lc.DEFS.replace(
    '</defs>',
    f'<marker id="gp" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_GPU_S}"/></marker>'
    f'<marker id="bt" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_BEAT_T}"/></marker>'
    '</defs>')

W, H = 1500, 884
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '三态开关：None 逐条排除、True 硬校验即 raise、False 不查不问',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, 'async_scheduling 在 VllmConfig.__post_init__ 里定值——v0.27.1 服务默认心跳（AsyncScheduler + 批队列重叠）的出生地',
        10.5, lc.C_MUTE, 'start', maxw=1020, tag='subtitle')
_ch = '放大自 L2 拍片 ⑤ async 三态决策 · L0：启动视角（boot）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 顶行：上下文注 / 根节点 / 执行器能力反问 ----------------
TOP_Y, TOP_H = 80, 96

# 上下文 + 不变量注（虚线）
CTX_X, CTX_W = 60, 410
lc.rect(CTX_X, TOP_Y, CTX_W, TOP_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.2, dash=True)
lc.text(CTX_X + 14, TOP_Y + 20, 'VllmConfig.__post_init__ 的第 11 站', 10, lc.C_TXT, 'start',
        True, maxw=CTX_W - 28, tag='ctx:t')
lc.text(CTX_X + 14, TOP_Y + 38, 'vllm/config/vllm.py:L1052-L1143 · 装配线里的一个推导步骤：', 8.5,
        '#334155', 'start', maxw=CTX_W - 28, tag='ctx:l1')
lc.text(CTX_X + 14, TOP_Y + 54, '对用户值的赋值只发生在 None 分支——显式 True 只校验', 8.5,
        '#334155', 'start', maxw=CTX_W - 28, tag='ctx:l2')
lc.text(CTX_X + 14, TOP_Y + 70, '（不过即 raise）、显式 False 整段跳过：显式值永不被改写，', 8.5,
        '#334155', 'start', maxw=CTX_W - 28, tag='ctx:l3')
lc.text(CTX_X + 14, TOP_Y + 86, '由控制流结构保证，不靠约定。', 8.5,
        '#334155', 'start', maxw=CTX_W - 28, tag='ctx:l4')

# 根节点（进场值三岔）
ROOT_X, ROOT_W = 500, 500
lc.rect(ROOT_X, TOP_Y, ROOT_W, TOP_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=7, sw=1.9)
lc.text(ROOT_X + 16, TOP_Y + 22, '进场值 async_scheduling ∈ {True, False, None}', 12, lc.C_TXT,
        'start', True, maxw=ROOT_W - 32, tag='root:t')
lc.text(ROOT_X + 16, TOP_Y + 44, '三态字段，默认 None（bool | None = None——vllm/config/scheduler.py:L148-L151）',
        9.5, '#334155', 'start', maxw=ROOT_W - 32, tag='root:l1')
lc.text(ROOT_X + 16, TOP_Y + 62, 'None：__post_init__ 替你推导 · True：硬校验不过即 raise · False：不查不问',
        9.5, '#334155', 'start', maxw=ROOT_W - 32, tag='root:l2')
lc.text(ROOT_X + 16, TOP_Y + 82, '同一个值，决定第 16 站工厂② 选 AsyncScheduler 还是 Scheduler', 9,
        lc.C_BEAT_T, 'start', True, maxw=ROOT_W - 32, tag='root:l3')

# 执行器能力反问（绿——与 L2 章图工厂① 同色）
EX_X, EX_W = 1060, 380
lc.rect(EX_X, TOP_Y, EX_W, TOP_H, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.6)
lc.text(EX_X + 14, TOP_Y + 20, '反问执行器 · Executor.get_class', 10.5, lc.C_TXT, 'start', True,
        maxw=EX_W - 28, tag='ex:t')
lc.text(EX_X + 14, TOP_Y + 38, 'executor_class.supports_async_scheduling()（vllm.py:L1056-L1057）', 8.5,
        '#334155', 'start', maxw=EX_W - 28, tag='ex:l1')
lc.text(EX_X + 14, TOP_Y + 55, '配置层反查实现层的罕见回路——工厂① 的第一次调用，比', 8.5,
        '#334155', 'start', maxw=EX_W - 28, tag='ex:l2')
lc.text(EX_X + 14, TOP_Y + 71, '正式选定（from_engine_args）早一个阶段。取值：', 8.5,
        '#334155', 'start', maxw=EX_W - 28, tag='ex:l3')
lc.text(EX_X + 14, TOP_Y + 88, 'uni=True · mp=True · ray=False（继承基类 abstract.py:L364）', 8.5,
        lc.C_GPU_S, 'start', True, maxw=EX_W - 28, tag='ex:l4')

# 反问箭头：根 → 执行器（先于三岔发生）
lc.seg(ROOT_X + ROOT_W + 2, 128, EX_X - 3, 128, lc.C_GPU_S, 1.8, 'gp', dash=True)
lc.text((ROOT_X + ROOT_W + EX_X) / 2, 118, '反问能力', 8.5, lc.C_GPU_S, 'middle', True,
        maxw=56, tag='a:ask')

# ---------------- 三岔臂 ----------------
HDR_Y, HDR_H = 216, 38
A_CX, B_CX, C_CX = 280, 720, 1160            # None / False / True 三列中心

lc.parrow([(620, TOP_Y + TOP_H), (620, 198), (A_CX, 198), (A_CX, HDR_Y)],
          lc.C_BEAT_T, 2.0, 'bt')
lc.text(450, 192, 'None（默认）', 8.5, lc.C_BEAT_T, 'middle', True, maxw=90, tag='arm:none')
lc.seg(B_CX, TOP_Y + TOP_H, B_CX, HDR_Y, lc.C_BEAT_T, 2.0, 'bt')
lc.text(B_CX + 8, 194, 'False（显式关）', 8.5, lc.C_BEAT_T, 'start', True, maxw=100, tag='arm:false')
lc.parrow([(880, TOP_Y + TOP_H), (880, 196), (C_CX, 196), (C_CX, HDR_Y)],
          lc.C_BEAT_T, 2.0, 'bt')
lc.text(1020, 190, 'True（显式开）', 8.5, lc.C_BEAT_T, 'middle', True, maxw=90, tag='arm:true')

# ---------------- 列头 ----------------
def col_header(x, w, title, anchor):
    lc.rect(x, HDR_Y, w, HDR_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=6, sw=1.4)
    lc.text(x + 12, HDR_Y + 16, title, 10.5, lc.C_BEAT_T, 'start', True, maxw=w - 24, tag='hd:' + title[:8])
    lc.text(x + 12, HDR_Y + 31, anchor, 8, lc.C_FAINT, 'start', maxw=w - 24, tag='ha:' + anchor[:10])

col_header(MX, 440, '臂 None（默认）——五条排除，命中即静默置 False', 'elif None 分支 · vllm.py:L1095-L1143')
col_header(540, 360, '臂 False（显式关）——整段跳过', '既非 True 也非 None → 两分支都不进')
col_header(940, 500, '臂 True（显式开）——四条硬校验，任一命中当场 raise', 'if True 分支 · vllm.py:L1064-L1094')

# ---------------- 列 A：None 五条排除链 + 左侧 False 汇流轨 ----------------
CHK_X, CHK_W, CHK_H, CHK_STEP = 140, 360, 38, 58
CHK_Y0 = 278
RAIL_X = 90

EXCL = [
    ('排除 ① · 模型是 pooling？', 'async 对 pooling 反而负收益（runner_type 判定）', '场景 2'),
    ('排除 ② · 投机方法不在白名单？', '白名单：EAGLE / MTP / NGram GPU / DSpark', '场景 3 medusa'),
    ('排除 ③ · disable_padded_drafter_batch=True？', '投机批填充与 async 排程互斥', None),
    ('排除 ④ · 执行器不支持 async？', '左上「反问能力」返回 False 时（如 ray）', None),
    ('排除 ⑤ · ROCm DeepEP 高吞吐 DBO？', '该组合会污染 DP+EP 生成精度', None),
]
mids = []
for i, (main, sub, tag) in enumerate(EXCL):
    y = CHK_Y0 + i * CHK_STEP
    mid = y + CHK_H / 2
    mids.append(mid)
    lc.rect(CHK_X, y, CHK_W, CHK_H, '#ffffff', lc.C_BEAT_S, rx=6, sw=1.3)
    tag_w = lc.tw(tag, 8, True) if tag else 0
    lc.text(CHK_X + 12, y + 16, main, 9.5, lc.C_TXT, 'start', True,
            maxw=CHK_W - 24 - (tag_w + 14 if tag else 0), tag='ex' + str(i))
    lc.text(CHK_X + 12, y + 31, sub, 8, lc.C_MUTE, 'start', maxw=CHK_W - 24, tag='exs' + str(i))
    if tag:
        lc.text(CHK_X + CHK_W - 10, y + 16, tag, 8, lc.C_MUTE, 'end', True,
                maxw=tag_w + 6, tag='ext' + str(i))
    # 命中 → 左侧汇流轨
    lc.seg(CHK_X - 2, mid, RAIL_X + 4, mid, lc.C_MUTE, 1.5, 'std')

# 链间「不中 → 下一条」箭头
for i in range(len(EXCL) - 1):
    y1 = CHK_Y0 + i * CHK_STEP + CHK_H
    y2 = CHK_Y0 + (i + 1) * CHK_STEP
    lc.seg(A_CX, y1 + 1, A_CX, y2 - 2, lc.C_BEAT_T, 1.6, 'bt')
lc.text(A_CX + 8, CHK_Y0 + CHK_H + 14, '不中 → 问下一条', 8, lc.C_MUTE, 'start', maxw=110,
        tag='a:miss')

# 汇流轨：五条命中共用一根 → 终值 False
lc.seg(RAIL_X, mids[0], RAIL_X, OUTCOME_Y := 676, lc.C_MUTE, 1.5, 'std')
lc.text(RAIL_X, 268, '命中 → 静默置 False', 8, lc.C_MUTE, 'middle', maxw=104, tag='rail:lbl')

# else 收尾：五条全不中 → True
PILL_Y = CHK_Y0 + 5 * CHK_STEP + 20
lc.seg(A_CX, CHK_Y0 + 4 * CHK_STEP + CHK_H + 1, A_CX, PILL_Y - 2, lc.C_BEAT_T, 1.6, 'bt')
lc.text(A_CX + 8, CHK_Y0 + 4 * CHK_STEP + CHK_H + 16, '全不中', 8, lc.C_MUTE, 'start', maxw=50,
        tag='a:allmiss')
lc.rect(CHK_X, PILL_Y, CHK_W, 34, lc.C_BEAT_F, lc.C_BEAT_S, rx=6, sw=1.4)
lc.text(CHK_X + 12, PILL_Y + 21, '五条全不中 → else 置 True', 9.5, lc.C_BEAT_T, 'start', True,
        maxw=CHK_W - 90, tag='pillA')
lc.text(CHK_X + CHK_W - 10, PILL_Y + 21, '场景 1', 8, lc.C_MUTE, 'end', True, maxw=50, tag='pillA:tag')

# ---------------- 列 B（False）：整段跳过 ----------------
SK_X, SK_W = 540, 360
SK_Y, SK_H = 274, 80
lc.seg(B_CX, HDR_Y + HDR_H, B_CX, SK_Y - 2, lc.C_BEAT_T, 2.0, 'bt')
lc.rect(SK_X, SK_Y, SK_W, SK_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.6)
lc.text(SK_X + 14, SK_Y + 20, '不查 pooling、不查投机、不问执行器', 9.5, lc.C_TXT, 'start', True,
        maxw=SK_W - 28, tag='sk:t')
lc.text(SK_X + 14, SK_Y + 38, '零判定——if/elif 两分支都不进，', 8.5, '#334155', 'start',
        maxw=SK_W - 28, tag='sk:l1')
lc.text(SK_X + 14, SK_Y + 54, '进场的 False 原样通过（本臂无任何赋值）', 8.5, '#334155', 'start',
        maxw=SK_W - 28, tag='sk:l2')
lc.text(SK_X + 14, SK_Y + 71, '场景 5：显式 False · 兼容性检查全部跳过', 8, lc.C_MUTE, 'start',
        maxw=SK_W - 28, tag='sk:l3')

# 判定量小注（虚线）
QN_Y = 384
lc.rect(SK_X, QN_Y, SK_W, 78, '#ffffff', lc.C_MUTE, rx=7, sw=1.2, dash=True)
lc.text(SK_X + 14, QN_Y + 18, '判定量（决策本身 O(1) 纯函数）', 9.5, lc.C_TXT, 'start', True,
        maxw=SK_W - 28, tag='qn:t')
lc.text(SK_X + 14, QN_Y + 36, 'None 最多 5 条排除 · True 4 条硬校验 ·', 8.5, '#334155', 'start',
        maxw=SK_W - 28, tag='qn:l1')
lc.text(SK_X + 14, QN_Y + 52, 'False 0 条；elif 链命中即短路——剩余', 8.5, '#334155', 'start',
        maxw=SK_W - 28, tag='qn:l2')
lc.text(SK_X + 14, QN_Y + 68, '判定严格递减，有限步必终止于唯一赋值', 8.5, '#334155', 'start',
        maxw=SK_W - 28, tag='qn:l3')

# ---------------- 列 C（True）：四条硬校验 + 右侧 raise 汇流轨 ----------------
HCHK_X, HCHK_W = 940, 440
HCHK_Y0 = 278
RRAIL_X = 1410

HARD = [
    ('硬校验 ① · ROCm DeepEP 高吞吐 DBO？', 'raise：请 --no-async-scheduling 或换 all2all backend', None),
    ('硬校验 ② · 投机方法不在白名单？', '仅支持 EAGLE / MTP / Draft Model / NGram GPU / DSpark', None),
    ('硬校验 ③ · disable_padded_drafter_batch=True？', '显式 True 下互斥组合直接拒绝（None 分支则静默关）', None),
    ('硬校验 ④ · 执行器不支持 async？', '场景 4：ray 继承基类 False → raise', '场景 4'),
]
hmids = []
for i, (main, sub, tag) in enumerate(HARD):
    y = HCHK_Y0 + i * CHK_STEP
    mid = y + CHK_H / 2
    hmids.append(mid)
    lc.rect(HCHK_X, y, HCHK_W, CHK_H, '#ffffff', lc.C_BEAT_S, rx=6, sw=1.3)
    tag_w = lc.tw(tag, 8, True) if tag else 0
    lc.text(HCHK_X + 12, y + 16, main, 9.5, lc.C_TXT, 'start', True,
            maxw=HCHK_W - 24 - (tag_w + 14 if tag else 0), tag='hd' + str(i))
    lc.text(HCHK_X + 12, y + 31, sub, 8, lc.C_MUTE, 'start', maxw=HCHK_W - 24, tag='hds' + str(i))
    if tag:
        lc.text(HCHK_X + HCHK_W - 10, y + 16, tag, 8, lc.C_MUTE, 'end', True,
                maxw=tag_w + 6, tag='hdt' + str(i))
    # 命中 → 右侧 raise 汇流轨（红）
    lc.seg(HCHK_X + HCHK_W + 2, mid, RRAIL_X - 4, mid, lc.C_ABORT, 1.5, 'ab')

for i in range(len(HARD) - 1):
    y1 = HCHK_Y0 + i * CHK_STEP + CHK_H
    y2 = HCHK_Y0 + (i + 1) * CHK_STEP
    lc.seg(C_CX, y1 + 1, C_CX, y2 - 2, lc.C_BEAT_T, 1.6, 'bt')
lc.text(C_CX + 8, HCHK_Y0 + CHK_H + 14, '过了 → 查下一条', 8, lc.C_MUTE, 'start', maxw=110,
        tag='a:pass1')

# raise 汇流轨
lc.seg(RRAIL_X, hmids[0], RRAIL_X, OUTCOME_Y, lc.C_ABORT, 1.5, 'ab')
lc.text(RRAIL_X - 8, 268, '任一命中 → raise', 8, lc.C_ABORT, 'end', True, maxw=96, tag='rrail:lbl')

# 全过 → 保持 True
HPILL_Y = HCHK_Y0 + 4 * CHK_STEP + 20
lc.seg(C_CX, HCHK_Y0 + 3 * CHK_STEP + CHK_H + 1, C_CX, HPILL_Y - 2, lc.C_BEAT_T, 1.6, 'bt')
lc.text(C_CX + 8, HCHK_Y0 + 3 * CHK_STEP + CHK_H + 16, '全过', 8, lc.C_MUTE, 'start', maxw=40,
        tag='a:allpass')
lc.rect(HCHK_X, HPILL_Y, HCHK_W, 34, lc.C_BEAT_F, lc.C_BEAT_S, rx=6, sw=1.4)
lc.text(HCHK_X + 14, HPILL_Y + 21, '四条全过 → 保持 True（本臂零赋值，用户值原样通过）', 9.5,
        lc.C_BEAT_T, 'start', True, maxw=HCHK_W - 28, tag='pillC')

# ---------------- 结果行 ----------------
OUT_Y, OUT_H = 676, 116
lc.parrow([(A_CX, PILL_Y + 34), (A_CX, 660), (540, 660), (540, OUT_Y)],
          lc.C_BEAT_T, 2.0, 'bt')
lc.parrow([(B_CX, SK_Y + SK_H), (B_CX, 640), (390, 640), (390, OUT_Y)],
          lc.C_MUTE, 2.0, 'std')
lc.parrow([(C_CX, HPILL_Y + 34), (C_CX, 644), (960, 644), (960, OUT_Y)],
          lc.C_BEAT_T, 2.0, 'bt')

# 终值 False
F_X, F_W = 60, 360
lc.rect(F_X, OUT_Y, F_W, OUT_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.6)
lc.text(F_X + 14, OUT_Y + 22, '终值 False', 11.5, lc.C_TXT, 'start', True, maxw=F_W - 28, tag='outF:t')
lc.text(F_X + 14, OUT_Y + 42, '→ 工厂② 选 Scheduler · max_concurrent_batches = 1', 9,
        '#334155', 'start', maxw=F_W - 28, tag='outF:l1')
lc.text(F_X + 14, OUT_Y + 60, '来源：None 排除命中（场景 2 / 3）·', 8.5, lc.C_MUTE, 'start',
        maxw=F_W - 28, tag='outF:l2')
lc.text(F_X + 14, OUT_Y + 76, '显式 False（场景 5）——三岔恰走一路，', 8.5, lc.C_MUTE, 'start',
        maxw=F_W - 28, tag='outF:l3')
lc.text(F_X + 14, OUT_Y + 92, '「来源」是列举，不是先后因果。', 8.5, lc.C_MUTE, 'start',
        maxw=F_W - 28, tag='outF:l4')

# 终值 True
T_X, T_W = 480, 530
lc.rect(T_X, OUT_Y, T_W, OUT_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=7, sw=1.8)
lc.text(T_X + 14, OUT_Y + 22, '终值 True', 11.5, lc.C_BEAT_T, 'start', True, maxw=T_W - 28, tag='outT:t')
lc.text(T_X + 14, OUT_Y + 42, '→ 工厂② 选 AsyncScheduler · max_concurrent_batches = 2', 9,
        '#334155', 'start', maxw=T_W - 28, tag='outT:l1')
lc.text(T_X + 14, OUT_Y + 60, '重叠需要两个并发批（vllm.py:L539-L550）——v0.27.1 服务默认心跳的出生地', 8.5,
        lc.C_MUTE, 'start', maxw=T_W - 28, tag='outT:l2')
lc.text(T_X + 14, OUT_Y + 78, '来源：None 五条全不中（场景 1）· 显式 True 全过——', 8.5,
        lc.C_MUTE, 'start', maxw=T_W - 28, tag='outT:l3')
lc.text(T_X + 14, OUT_Y + 96, '「来源」是列举，不是先后因果。', 8.5, lc.C_MUTE, 'start',
        maxw=T_W - 28, tag='outT:l4')

# raise 中止卡
R_X, R_W = 1070, 370
lc.rect(R_X, OUT_Y, R_W, OUT_H, '#ffffff', lc.C_ABORT, rx=7, sw=1.8)
lc.text(R_X + 14, OUT_Y + 22, '装配当场中止——不静默降级', 11, lc.C_ABORT, 'start', True,
        maxw=R_W - 28, tag='outR:t')
lc.text(R_X + 14, OUT_Y + 44, 'ValueError: `ray` does not support async', 9,
        '#334155', 'start', maxw=R_W - 28, tag='outR:l1')
lc.text(R_X + 14, OUT_Y + 60, 'scheduling yet.', 9, '#334155', 'start', maxw=R_W - 28, tag='outR:l2')
lc.text(R_X + 14, OUT_Y + 82, '场景 4：显式 True + ray 执行器，硬校验 ④ 命中——', 8.5, lc.C_MUTE,
        'start', maxw=R_W - 28, tag='outR:l3')
lc.text(R_X + 14, OUT_Y + 98, '用户显式值要么全过、要么 raise，绝不被改写。', 8.5, lc.C_MUTE,
        'start', maxw=R_W - 28, tag='outR:l4')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = OUT_Y + OUT_H + 26
lx = MX
items = [
    ('beat', '__post_init__ 推导主线（L2 拍片 ⑤ 的机制展开）'),
    ('gpu', '执行器能力反问（工厂① 首调）'),
    ('mut', '命中 → 静默置 False'),
    ('red', '命中 → raise（装配中止）'),
]
for kind, name in items:
    if kind == 'beat':
        lc.rect(lx, LEG_Y - 8, 20, 13, lc.C_BEAT_F, lc.C_BEAT_S, rx=4, sw=1.4)
    elif kind == 'gpu':
        lc.rect(lx, LEG_Y - 8, 20, 13, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.4)
    elif kind == 'mut':
        lc.seg(lx, LEG_Y - 1, lx + 20, LEG_Y - 1, lc.C_MUTE, 1.5, 'std')
    else:
        lc.seg(lx, LEG_Y - 1, lx + 20, LEG_Y - 1, lc.C_ABORT, 1.5, 'ab')
    lc.text(lx + 26, LEG_Y + 2, name, 9, lc.C_TXT, 'start', maxw=280, tag='leg' + name)
    lx += 26 + lc.tw(name, 9) + 22

lc.text(MX, LEG_Y + 26, 'verbatim vllm/config/vllm.py:L1052-L1143 · 三态字段 vllm/config/scheduler.py:L148-L151 · '
        '批并发 vllm.py:L539-L550 · supports 取值 vllm/v1/executor/abstract.py:L364 / uniproc_executor.py:L146 / multiproc_executor.py:L526',
        9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '五场景取自精简版 companion host 实测（supports 取值与真实源码一致）· 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch03-fig-async-tri-state.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
