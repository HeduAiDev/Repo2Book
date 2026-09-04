#!/usr/bin/env python3
"""ch09 机制图 1 · 一拍五段的真引擎泳道时序（figure_spec ch09-fig-five-beats-timeline，模板 swimlane）

放大自 L0 的循环框（loop_box）——即本章 L2 章图 center 拍片行（①-⑤ 一拍五段，
core.py:L584-L614）的时间维展开：L2 画五拍的结构顺序与站号，本图把混相拍（拍 3）放到
五泳道纵向时间轴上，回答「每段何时发生、各占多久、谁跟谁重叠」。架构归属回指 L0/L2
（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：真引擎一拍五段时序实测（钉版 v0.27.1 全链路 + RTX PRO 6000 Blackwell，同步版
step()，混相拍 3）：① 0.045ms → ② 发起 1.341ms（worker launch 1.281ms=kernel 入队，
发起即返回）→ ③ bitmask 0.001ms 且结束时 CUDA event query 前向仍未完（藏窗直接证据）
→ ④a 0.001ms → ④b 0.358ms（入口时前向未完：等 GPU 尾程+采样+D2H）→ ⑤ 0.056ms，
拍全程 1.860ms；guard 拍 0.006ms 零调用——五拍的排布让 ③④a 全部藏进前向窗口。

数字全部取自真引擎实测 trace（explainer m1 spec.numbers：beat 3 各段 wall/wait、
worker launch 1.281、beat 1 对照 2.516/2.466/0.225/2.870、前向完成 1.873/2.873、
五拍批形状与产出、guard 0.006 零事件；外证 8B@H100 ~5ms 明标非本机实测）。
坐标由常量/循环计算；文本全 esc()；时间轴分段非线性（各段时长以标注为准）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

# ---------------- 画布与版式常量 ----------------
W, H = 1660, 930
MX, BXR = 56, 1604

# ---------------- 标题区 ----------------
lc.text(MX, 34, '一拍五段的真引擎时序——③④a 藏进前向窗口，④b 只等 GPU 尾程',
        16.5, lc.C_TXT, 'start', True, maxw=1150, tag='title')
lc.text(MX, 58, '同步版 step() 双 decode 拍（拍 3）实测：① 0.045 → ② 发起 1.341（worker launch 1.281）'
        '→ ③ 0.001 → ④a 0.001 → ④b 0.358 → ⑤ 0.056（单位 ms）· 拍全程 1.860ms '
        '· 钉版 v0.27.1 全链路 + RTX PRO 6000 Blackwell',
        10.5, lc.C_MUTE, 'start', maxw=1290, tag='subtitle')
_ch = '放大自 L2 拍片行『①-⑤ 一拍五段』· L0：循环框'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 泳道几何（5 泳道：忙循环/调度器/executor/worker/GPU） ----------------
LANE_X0, LANE_X1, N_LANE = 210.0, 1300.0, 5
LANE_W = (LANE_X1 - LANE_X0) / N_LANE
LANE_C = [LANE_X0 + LANE_W * (i + 0.5) for i in range(N_LANE)]   # 319/537/755/973/1191
BLK_W = 176.0
HDR_Y, HDR_H = 88, 30
TL_Y0 = 150.0

LANES = [
    ('忙循环', 'EngineCore 单线程', lc.C_ENG_S, lc.C_ENG_F, lc.C_ENG_S),
    ('调度器', '① ③ ⑤ 在此', lc.C_ENG_S, lc.C_ENG_F, lc.C_ENG_S),
    ('executor', '② ④a（进程内组件）', lc.C_BEAT_S, lc.C_BEAT_F, lc.C_ENG_S),
    ('worker', 'GPUModelRunner', lc.C_GPU_S, lc.C_GPU_F, lc.C_GPU_S),
    ('GPU', 'RTX PRO 6000 Blackwell', lc.C_GPU_S, lc.C_GPU_S, '#ffffff'),
]
LANE_TL_END_KEY = 'fwd_e'      # 生命线画到前向完成时刻（全轴最深）

# ---------------- 时间分段（拍 3 实测；dy = max(ms*K, min_px)——非线性分段轴） ----------------
K_MS = 130.0
BANDS = [
    ('pre',    0.006, 12),   # 拍开始 → ①
    ('b1',     0.045, 34),   # ① schedule
    ('g12',    0.003, 12),
    ('b2pre',  0.014, 12),   # executor 进 worker 前
    ('b2w',    1.281, 0),    # worker launch（16 层 kernel 入队）
    ('b2post', 0.046, 12),   # worker 返回后 executor 收尾
    ('g23',    0.005, 12),
    ('b3',     0.001, 18),   # ③ bitmask
    ('g34a',   0.007, 12),
    ('b4a',    0.002, 18),   # ④a future.result
    ('g34b',   0.002, 12),
    ('b4b',    0.358, 78),   # ④b sample_tokens（表头 18 + 三段 20×3）
    ('g45',    0.007, 12),
    ('b5',     0.056, 34),   # ⑤ update
    ('tail',   0.027, 12),   # ⑤ 末 → 拍尾
    ('fwd',    0.013, 14),   # 拍尾 → 前向完成（event 取证）
]
Y = {'t0': TL_Y0}
_y = TL_Y0
for _key, _ms, _minh in BANDS:
    Y[_key + '_s'] = _y
    _y += max(_ms * K_MS, _minh)
    Y[_key + '_e'] = _y
TL_END = _y
BEAT_END = Y['tail_e']          # 拍尾（step() 返回）
FWD_DONE = Y['fwd_e']           # 前向完成时刻（event 取证 1.873ms）

# ---------------- 泳道头 + 生命线 + 左侧时间轴 ----------------
for i, (nm, sub, stroke, fill, tcol) in enumerate(LANES):
    cx = LANE_C[i]
    lc.rect(cx - 100, HDR_Y, 200, HDR_H, fill, stroke, rx=7, sw=1.5)
    lc.text(cx, HDR_Y + 14, nm, 10, tcol, 'middle', True, maxw=190, tag='lane' + nm)
    lc.text(cx, HDR_Y + 26, sub, 7.8, tcol if tcol == '#ffffff' else lc.C_MUTE,
            'middle', maxw=190, tag='lane' + nm + ':sub')
    lc.seg(cx, HDR_Y + HDR_H + 2, cx, TL_END + 6, lc.C_FAINT, 1.0, dash=True)

RULER_X = 190.0
lc.seg(RULER_X, TL_Y0, RULER_X, TL_END, lc.C_MUTE, 1.2)
for _k, _ms, _mh in BANDS:                       # 段边界小刻度（不标数值，时长以块内标注为准）
    lc.seg(RULER_X - 3, Y[_k + '_s'], RULER_X + 3, Y[_k + '_s'], lc.C_MUTE, 1.0)
lc.seg(RULER_X - 3, TL_END, RULER_X + 3, TL_END, lc.C_MUTE, 1.0)
lc.text(MX, 144, '拍内时间（ms）', 8.5, lc.C_MUTE, 'start', maxw=120, tag='ruler:cap')
lc.text(RULER_X - 6, TL_Y0 + 3, '0', 8.5, lc.C_MUTE, 'end', tag='ruler:0')
lc.text(RULER_X - 6, BEAT_END + 3, '拍尾 1.860', 8.5, lc.C_MUTE, 'end', tag='ruler:end')

# ---------------- 元素几何（先算好边，箭头两端都贴元素边） ----------------
SPINE_X0, SPINE_X1 = LANE_C[0] - 7, LANE_C[0] + 7          # 忙循环脊柱 312..326
SCHED_L, SCHED_R = LANE_C[1] - BLK_W / 2, LANE_C[1] + BLK_W / 2
EXEC_L, EXEC_R = LANE_C[2] - BLK_W / 2, LANE_C[2] + BLK_W / 2
WORK_L, WORK_R = LANE_C[3] - BLK_W / 2, LANE_C[3] + BLK_W / 2
BAR_X0, BAR_X1 = LANE_C[4] - 70, LANE_C[4] + 70            # GPU 前向条 1121..1261
PNL_X0, PNL_X1 = 1330.0, 1604.0                            # 右侧窗口实证面板

# ---------------- 箭头（先画，块/条随后覆盖起端） ----------------
def ymid(k):
    return (Y[k + '_s'] + Y[k + '_e']) / 2

# ① 忙循环→调度器
lc.seg(SPINE_X1, ymid('b1'), SCHED_L, ymid('b1'), lc.C_API_S, 1.8, 'dn')
# ② 忙循环→executor（发起）
_y2 = ymid('b2pre')
lc.seg(SPINE_X1, _y2, EXEC_L, _y2, lc.C_API_S, 1.8, 'dn')
lc.text((SPINE_X1 + EXEC_L) / 2, _y2 - 7, '② execute_model(non_block=True)',
        8.5, lc.C_API_S, 'middle', maxw=320, tag='a:call2')
# ② 返回：executor→忙循环（发起即返回，拿 Future）
_y2r = ymid('b2post')
lc.seg(EXEC_L, _y2r, SPINE_X1, _y2r, lc.C_ENG_S, 1.8, 'up')
lc.text(560, _y2r - 8, '返回 Future · 发起即返回（non_block=True）',
        8.5, lc.C_ENG_S, 'middle', maxw=250, tag='a:ret2')
# ③ 忙循环→调度器
lc.seg(SPINE_X1, ymid('b3'), SCHED_L, ymid('b3'), lc.C_API_S, 1.8, 'dn')
# ④a 忙循环→executor
lc.seg(SPINE_X1, ymid('b4a'), EXEC_L, ymid('b4a'), lc.C_API_S, 1.8, 'dn')
# ④b 忙循环→worker（经 executor 转发）
_y4b = Y['b4b_s'] + 9
lc.seg(SPINE_X1, _y4b, WORK_L, _y4b, lc.C_API_S, 1.8, 'dn')
# ④b 返回：worker→忙循环（ModelRunnerOutput）
_y4br = Y['b4b_e'] + 2
lc.seg(WORK_L, _y4br, SPINE_X1, _y4br, lc.C_ENG_S, 1.8, 'up')
lc.text(620, _y4br - 8, 'ModelRunnerOutput（采样+D2H 已在内落地）',
        8.5, lc.C_ENG_S, 'middle', maxw=230, tag='a:ret4b')
# worker→GPU：kernel 入队（16 层字样已在 worker 块内，标签缩短避让 GPU 条顶角）
_yk = Y['b2w_s'] + 7
lc.seg(WORK_R, _yk, BAR_X0, _yk, lc.C_API_S, 1.8, 'dn')
lc.text((WORK_R + BAR_X0) / 2, _yk - 6, 'kernel 入队', 8.2, lc.C_API_S,
        'middle', maxw=62, tag='a:launch')

# ---------------- CPU 侧块 ----------------
# ① schedule
lc.rect(SCHED_L, Y['b1_s'], BLK_W, 34, lc.C_ENG_F, lc.C_ENG_S, rx=4, sw=1.4)
lc.text(LANE_C[1], Y['b1_s'] + 14, '① schedule 0.045ms', 9, lc.C_TXT, 'middle', True,
        maxw=BLK_W - 10, tag='b1')
lc.text(LANE_C[1], Y['b1_s'] + 27, '组批 {A:1, B:1}', 8.2, lc.C_MUTE, 'middle',
        maxw=BLK_W - 10, tag='b1:sub')
# ② executor 发起块（含 worker launch 段的容器标注）
lc.rect(EXEC_L, Y['b2pre_s'], BLK_W, Y['b2post_e'] - Y['b2pre_s'],
        lc.C_BEAT_F, lc.C_BEAT_S, rx=4, sw=1.4)
_e2mid = (Y['b2pre_s'] + Y['b2post_e']) / 2
for _i, _ln in enumerate(['② execute_model', '（non_block=True · 4 拍全 True）',
                          '发起 1.341ms', 'kernel 入队即返回 Future']):
    lc.text(LANE_C[2], _e2mid - 25 + _i * 17, _ln, 9, lc.C_TXT, 'middle',
            maxw=BLK_W - 10, tag='b2:' + str(_i))
# worker launch 块
lc.rect(WORK_L, Y['b2w_s'], BLK_W, Y['b2w_e'] - Y['b2w_s'], lc.C_GPU_F, lc.C_GPU_S,
        rx=4, sw=1.4)
_wmid = (Y['b2w_s'] + Y['b2w_e']) / 2
for _i, _ln in enumerate(['launch 1.281ms', '16 层前向 kernel 入队', '（含 persistent-batch 更新）']):
    lc.text(LANE_C[3], _wmid - 17 + _i * 17, _ln, 9, lc.C_TXT, 'middle',
            maxw=BLK_W - 10, tag='b2w:' + str(_i))
# ③ bitmask
lc.rect(SCHED_L, Y['b3_s'], BLK_W, Y['b3_e'] - Y['b3_s'], lc.C_ENG_F, lc.C_ENG_S,
        rx=4, sw=1.4)
lc.text(LANE_C[1], Y['b3_s'] + 12.5, '③ bitmask 0.001ms', 8.5, lc.C_TXT, 'middle', True,
        maxw=BLK_W - 10, tag='b3')
# ④a result
lc.rect(EXEC_L, Y['b4a_s'], BLK_W, Y['b4a_e'] - Y['b4a_s'], lc.C_BEAT_F, lc.C_BEAT_S,
        rx=4, sw=1.4)
lc.text(LANE_C[2], Y['b4a_s'] + 12.5, '④a result 0.001ms（done→None）', 8.5, lc.C_TXT,
        'middle', True, maxw=BLK_W - 8, tag='b4a')
# ④b sample_tokens（表头 + 三段着色）
B4B_H = Y['b4b_e'] - Y['b4b_s']
lc.rect(WORK_L, Y['b4b_s'], BLK_W, B4B_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=4, sw=1.4)
lc.text(LANE_C[3], Y['b4b_s'] + 11, '④b sample_tokens 0.358ms', 8.5, lc.C_TXT, 'middle',
        True, maxw=BLK_W - 8, tag='b4b')
SUB_H = (B4B_H - 18) / 3
SUBS = [('等 GPU 尾程', 'wait', lc.C_MUTE), ('掩码+采样 kernel', 'gpu', '#ffffff'),
        ('同步 D2H', 'sam', '#ffffff')]
for _i, (_nm, _kind, _tc) in enumerate(SUBS):
    _sy = Y['b4b_s'] + 18 + _i * SUB_H
    _fill = 'url(#wait)' if _kind == 'wait' else (lc.C_GPU_S if _kind == 'gpu' else lc.C_SAM_S)
    _stroke = lc.C_FAINT if _kind == 'wait' else 'none'
    lc.rect(WORK_L + 6, _sy, BLK_W - 12, SUB_H, _fill, _stroke, rx=2, sw=1.0)
    lc.text(LANE_C[3], _sy + SUB_H / 2 + 3, _nm, 8.2, _tc, 'middle', maxw=BLK_W - 20,
            tag='b4b:sub' + str(_i))
# ⑤ update
lc.rect(SCHED_L, Y['b5_s'], BLK_W, Y['b5_e'] - Y['b5_s'], lc.C_ENG_F, lc.C_ENG_S,
        rx=4, sw=1.4)
lc.text(LANE_C[1], Y['b5_s'] + 14, '⑤ update 0.056ms', 9, lc.C_TXT, 'middle', True,
        maxw=BLK_W - 10, tag='b5')
lc.text(LANE_C[1], Y['b5_s'] + 27, 'A [7904] · B [4372] 双双 LENGTH', 8.2, lc.C_MUTE,
        'middle', maxw=BLK_W - 10, tag='b5:sub')

# ---------------- GPU 前向条 + 完成时刻夹逼（盲审修图单 1） ----------------
# 实心条止于 ④b 区间内（完成上确界 = ④b 内 D2H 落地）；条尾画开区间括号；
# 1.873 改为拍尾 event.synchronize 取证读数（完成上界），不再作条尾。
BAR_END = Y['b4b_e']
lc.rect(BAR_X0, Y['b2w_s'], BAR_X1 - BAR_X0, BAR_END - Y['b2w_s'], lc.C_GPU_S,
        'none', rx=3, sw=0)
_bmid = (Y['b2w_s'] + BAR_END) / 2
for _i, _ln in enumerate(['16 层前向 kernel', 'GPU 后台执行', '（default stream）']):
    lc.text(LANE_C[4], _bmid - 16 + _i * 16, _ln, 9, '#ffffff', 'middle',
            maxw=BAR_X1 - BAR_X0 - 8, tag='bar:' + str(_i))
lc.text(LANE_C[4], (Y['b4b_s'] + BAR_END) / 2 + 12, '完成 ∈ (④b 入口, ④b 结束]', 8,
        '#ffffff', 'middle', maxw=BAR_X1 - BAR_X0 - 8, tag='bar:ivl')
# 拍尾取证：虚线延伸至 synchronize 观测时刻（读数 = 完成上界）
lc.seg(LANE_C[4], BAR_END, LANE_C[4], FWD_DONE, lc.C_GPU_S, 1.4, dash=True)
lc.circle(LANE_C[4], FWD_DONE, 3.5, lc.C_KV_S, 1.4, dash=False)
lc.text(LANE_C[4], FWD_DONE + 16, '拍尾 event.synchronize 取证读数 1.873ms（完成上界）', 8.5,
        lc.C_KV_S, 'middle', maxw=250, tag='bar:done')

# ---------------- 忙循环脊柱 + 三个里程碑片 ----------------
lc.rect(SPINE_X0, TL_Y0, SPINE_X1 - SPINE_X0, BEAT_END - TL_Y0, lc.C_ENG_F, lc.C_ENG_S,
        rx=7, sw=1.2)
CHIP_W, CHIP_H = 196, 32
CHIPS = [
    (TL_Y0 + 4, ['step() 拍开始', '（本拍 = 双 decode 拍 3）']),
    (Y['b2post_e'] - CHIP_H - 2, ['② 返回 Future', '发起即返回，CPU 腾出手']),
    (BEAT_END - CHIP_H - 4, ['step() 返回', '拍全程 1.860ms']),
]
for _cy, (_l1, _l2) in CHIPS:
    lc.rect(LANE_C[0] - CHIP_W / 2, _cy, CHIP_W, CHIP_H, '#ffffff', lc.C_ENG_S,
            rx=6, sw=1.3)
    lc.text(LANE_C[0], _cy + 14, _l1, 8.8, lc.C_ENG_S, 'middle', True,
            maxw=CHIP_W - 8, tag='chip:' + _l1[:6])
    lc.text(LANE_C[0], _cy + 26, _l2, 8.2, lc.C_MUTE, 'middle',
            maxw=CHIP_W - 8, tag='chip:' + _l2[:6])

# ---------------- 右侧「窗口实证」面板（藏窗直接证据） ----------------
lc.rect(PNL_X0, TL_Y0, PNL_X1 - PNL_X0, TL_END - TL_Y0 + 29, '#ffffff', lc.C_KV_S,
        rx=8, sw=1.5)
lc.text(PNL_X0 + 12, TL_Y0 + 20, '窗口实证 · CUDA event', 10, lc.C_KV_S, 'start', True,
        maxw=250, tag='pnl:t1')
lc.text(PNL_X0 + 12, TL_Y0 + 34, '非阻塞 query（取证线）', 8.5, lc.C_MUTE, 'start',
        maxw=250, tag='pnl:t2')
ITEMS = [
    (Y['b3_e'], '③ 结束时查询：前向仍未完', '——③④a 藏进前向窗口的直接证据'),
    (Y['b4b_s'], '④b 入口时查询：仍未完', '——④b 等的是 GPU 尾程+采样+D2H'),
]
for _iy, _l1, _l2 in ITEMS:
    lc.seg(BAR_X1, _iy, PNL_X0, _iy, lc.C_KV_S, 1.2, dash=True)
    lc.circle(BAR_X1, _iy, 3.2, lc.C_KV_S, 1.2, dash=False)
    lc.text(PNL_X0 + 12, _iy - 4, _l1, 8.5, lc.C_TXT, 'start', True, maxw=252,
            tag='pnl:' + _l1[:6])
    lc.text(PNL_X0 + 12, _iy + 10, _l2, 8.2, lc.C_MUTE, 'start', maxw=252,
            tag='pnl:' + _l2[:6])
# 条尾开区间括号（左开空心圆骑在 ④b 入口取证线上、右闭粗刻度），x 让开探针线与面板
BRK_X = BAR_X1 + 17
IVL_MID = (Y['b4b_s'] + Y['b4b_e']) / 2
lc.seg(BRK_X, Y['b4b_s'], BRK_X, Y['b4b_e'], lc.C_MUTE, 1.4)
lc.seg(BRK_X - 8, Y['b4b_e'], BRK_X + 8, Y['b4b_e'], lc.C_MUTE, 2.4)
lc.ELEMS.append(((BRK_X - 5, Y['b4b_s'] - 5, BRK_X + 5, Y['b4b_s'] + 5),
                 f'<circle cx="{BRK_X:.1f}" cy="{Y["b4b_s"]:.1f}" r="4.2" '
                 'fill="#ffffff"/>'))
lc.circle(BRK_X, Y['b4b_s'], 3.4, lc.C_MUTE, 1.4, dash=False)
# 夹逼区间面板项（三点：③末 query 未完 + ④b 入口 query 未完 + ④b 内 D2H 落地）
lc.seg(BRK_X + 8, IVL_MID, PNL_X0, IVL_MID, lc.C_MUTE, 1.2, dash=True)
lc.text(PNL_X0 + 12, IVL_MID - 4, '完成 ∈ (④b 入口, ④b 结束]', 8.5, lc.C_TXT, 'start',
        True, maxw=252, tag='pnl:ivl1')
lc.text(PNL_X0 + 12, IVL_MID + 10, '——③末·④b入口 query 未完，④b 内 D2H 落地', 8.2,
        lc.C_MUTE, 'start', maxw=252, tag='pnl:ivl2')
# 拍全程 / guard 小结（面板内，夹逼区间项之下）
SUM_Y = IVL_MID + 17
lc.rect(PNL_X0 + 12, SUM_Y, PNL_X1 - PNL_X0 - 24, 86, lc.C_BEAT_F, lc.C_BEAT_S,
        rx=6, sw=1.3)
lc.text((PNL_X0 + PNL_X1) / 2, SUM_Y + 22, '拍全程 1.860ms', 11, lc.C_BEAT_T, 'middle',
        True, maxw=240, tag='sum:1')
lc.text((PNL_X0 + PNL_X1) / 2, SUM_Y + 42, 'guard 拍 0.006ms · 零事件', 8.5, lc.C_TXT,
        'middle', maxw=240, tag='sum:2')
lc.text((PNL_X0 + PNL_X1) / 2, SUM_Y + 58, '（executor 零调用 · 先于①早退）', 8.2,
        lc.C_MUTE, 'middle', maxw=240, tag='sum:3')

# ---------------- 同场五拍全景条 ----------------
STRIP_HD = TL_END + 35
lc.text(MX, STRIP_HD, '同场五拍全景（同步版，批形状与产出逐拍实测）——本图 = 拍 3 的时间维展开',
        10, lc.C_TXT, 'start', True, maxw=900, tag='strip:hd')
STRIP_Y, CELL_H, CELL_GAP = STRIP_HD + 8, 106, 12
CELL_W = (BXR - MX - 4 * CELL_GAP) / 5
CELLS = [
    ('拍 1 · prefill', True, ['批 {A:3} → A 出 [7189]', '② 发起 2.516ms（launch 2.466ms）',
                              '④b 0.225ms · 全程 2.870ms', '拍尾取证读数 2.873ms（完成上界）']),
    ('拍 2 · 混相批', True, ['批 {A:1, B:4}', '→ A [184] · B [5965]', '迟到的 B 全量 prefill 4 token']),
    ('拍 3 · 双 decode（本图）', True, ['批 {A:1, B:1} → A [7904] · B [4372]', '双双 LENGTH · 全程 1.860ms',
                                       '本图 = 此拍的时间维展开']),
    ('拍 4 · flush', False, ['空批 {}（total=0 仍下发）', '④b 跳过（④a 得非 None）',
                             '冲刷 finished 名单']),
    ('拍 5 · 空转守卫', False, ['has_requests()==False', '0.006ms · 零事件',
                                '（executor 零调用）']),
]
for _i, (_t, _hot, _lines) in enumerate(CELLS):
    _x = MX + _i * (CELL_W + CELL_GAP)
    if _i == 2:
        lc.rect(_x, STRIP_Y, CELL_W, CELL_H, lc.C_BEAT_F, lc.C_ENG_S, rx=6, sw=1.8)
        _bw = 40
        lc.rect(_x + CELL_W - _bw - 6, STRIP_Y + 5, _bw, 17, lc.C_ENG_S, lc.C_ENG_S,
                rx=8, sw=0)
        lc.text(_x + CELL_W - _bw / 2 - 6, STRIP_Y + 17.5, '本图', 8.5, '#ffffff',
                'middle', True, tag='cell:badge')
    elif _hot:
        lc.rect(_x, STRIP_Y, CELL_W, CELL_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=6, sw=1.3)
    else:
        lc.rect(_x, STRIP_Y, CELL_W, CELL_H, '#ffffff', lc.C_MUTE, rx=6, sw=1.2, dash=True)
    _tx = _x + 10 if _i != 2 else _x + 10
    lc.text(_tx, STRIP_Y + 18, _t, 9.2, lc.C_TXT, 'start', True,
            maxw=CELL_W - 24 - (44 if _i == 2 else 0), tag='cell:t' + str(_i))
    for _j, _ln in enumerate(_lines):
        lc.text(_tx, STRIP_Y + 36 + _j * 16, _ln, 8.2, '#334155', 'start',
                maxw=CELL_W - 18, tag='cell:%d:%d' % (_i, _j))
    if _i < 4:
        lc.seg(_x + CELL_W, STRIP_Y + CELL_H / 2, _x + CELL_W + CELL_GAP - 2,
               STRIP_Y + CELL_H / 2, lc.C_MUTE, 1.5, 'std')

# ---------------- 口径注记（launch-bound） ----------------
NOTE_Y = STRIP_Y + CELL_H + 16
NOTE_W = 1000
lc.rect(MX, NOTE_Y, NOTE_W, 62, 'none', lc.C_FAINT, rx=8, sw=1.1, dash=True)
lc.text(MX + 12, NOTE_Y + 16, '口径（launch-bound）：tiny 随机权重 Llama（16 层）上发起主导整拍',
        9.5, lc.C_TXT, 'start', True, maxw=NOTE_W - 24, tag='note:t')
lc.text(MX + 12, NOTE_Y + 33, '· ② 发起 1.341ms（本拍）/ 2.516ms（prefill 拍）主导，GPU 计算段亚毫秒'
        '——模型越重，比例越倒转', 8.8, '#334155', 'start', maxw=NOTE_W - 24, tag='note:1')
lc.text(MX + 12, NOTE_Y + 50, '· 外证（非本机实测）：真实 8B 模型 H100 单步约 5ms（V1 alpha 博客）'
        '——GPU 计算主导、窗口更宽，『发起即返回 + ③ 藏窗』更值钱', 8.8, '#334155',
        'start', maxw=NOTE_W - 24, tag='note:2')

# ---------------- 图例 ----------------
LEG_Y1, LEG_Y2 = NOTE_Y + 80, NOTE_Y + 102
lx = MX


def leg_box(color_fill, color_stroke, name, dash=False, sw=1.4):
    global lx
    lc.rect(lx, LEG_Y1 - 9, 20, 12, color_fill, color_stroke, rx=3, sw=sw, dash=dash)
    lc.text(lx + 26, LEG_Y1 + 1, name, 8.5, lc.C_TXT, 'start', maxw=300, tag='leg:' + name[:8])
    lx += 26 + lc.tw(name, 8.5) + 20


def leg_row2(kind, name):
    global lx
    if kind == 'hatch':
        lc.rect(lx, LEG_Y2 - 9, 20, 12, 'url(#wait)', lc.C_FAINT, rx=3, sw=1.0)
    elif kind == 'sam':
        lc.rect(lx, LEG_Y2 - 9, 20, 12, lc.C_SAM_S, lc.C_SAM_S, rx=3, sw=0)
    elif kind == 'beat':
        lc.rect(lx, LEG_Y2 - 9, 20, 12, lc.C_BEAT_F, lc.C_BEAT_S, rx=3, sw=1.2)
    elif kind == 'kv':
        lc.seg(lx, LEG_Y2 - 3, lx + 22, LEG_Y2 - 3, lc.C_KV_S, 1.3, dash=True)
    elif kind == 'call':
        lc.seg(lx, LEG_Y2 - 3, lx + 26, LEG_Y2 - 3, lc.C_API_S, 2.0)
    else:
        lc.seg(lx, LEG_Y2 - 3, lx + 26, LEG_Y2 - 3, lc.C_ENG_S, 2.0)
    lc.text(lx + 32, LEG_Y2 + 1, name, 8.5, lc.C_TXT, 'start', maxw=300, tag='leg2:' + name[:8])
    lx += 32 + lc.tw(name, 8.5) + 20


leg_box(lc.C_ENG_F, lc.C_ENG_S, '忙循环 / 调度器（EngineCore 进程）')
leg_box(lc.C_BEAT_F, lc.C_BEAT_S, 'executor（进程内组件）')
leg_box(lc.C_GPU_F, lc.C_GPU_S, 'worker（GPU 执行臂）')
leg_box(lc.C_GPU_S, 'none', 'GPU kernel 执行', sw=0)
lx = MX
leg_row2('hatch', '等待（非计算）')
leg_row2('sam', '同步 D2H 拷贝')
leg_row2('kv', 'CUDA event 非阻塞 query（取证）')
leg_row2('beat', '①-⑤ = step() 拍段')
leg_row2('call', '调用 / 发起 →')
leg_row2('ret', '← 返回')

# ---------------- 页脚 ----------------
lc.text(MX, H - 20, '数字取自真引擎实测：容器内钉版 v0.27.1 源树全链路 + NVIDIA RTX PRO 6000 Blackwell'
        '+ tiny 随机权重 Llama（16 层）· 同步版 step() 双 decode 拍 3 · 时间轴分段非线性（各段时长以标注为准）'
        '· 拍序逐字锚 vllm/v1/engine/core.py:L584-L614 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
HATCH = ('<pattern id="wait" width="7" height="7" patternTransform="rotate(45)" '
         'patternUnits="userSpaceOnUse"><rect width="7" height="7" fill="#f1f5f9"/>'
         '<line x1="0" y1="0" x2="0" y2="7" stroke="#94a3b8" stroke-width="2"/></pattern>')
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', '<defs>' + HATCH + '</defs>',
       lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch09-fig-five-beats-timeline.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems, timeline {TL_Y0:.0f}..{TL_END:.0f})')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
