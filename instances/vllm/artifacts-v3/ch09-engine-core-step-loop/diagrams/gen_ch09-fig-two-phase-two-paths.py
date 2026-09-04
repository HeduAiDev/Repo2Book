#!/usr/bin/env python3
"""ch09 机制图 · ④ 收货的两种路径——同步阻塞 vs 异步发起/等待分离（figure_spec
ch09-fig-two-phase-two-paths，模板 swimlane 双栏对照）

放大自 L0 循环框（loop_box）内 ④拍收货窗口——即本章 L2 章图 center ④拍片
『future.result + 条件 sample_tokens』的时序展开：回答『④ 收货到底在等什么、
同步版与异步版（v0.27.1 服务默认）差在哪』。与 m1 五段时间轴图（同一拍全景）
互为缩放关系，架构归属回指 L0/L2。

claim：④ 收货的两种路径（真引擎实测）：同步版 worker.sample_tokens 是阻塞收货——
0.358ms 里含『等 GPU 前向尾程（入口时 CUDA event 未完）+ 掩码/采样 kernel + 同步 D2H』；
异步版发起与等待分离——sample_tokens(non_block=True) 0.314ms 返回
AsyncGPUModelRunnerOutput→executor 包 AsyncOutputFuture 入队，之后 result() 只等
D2H 拷贝事件（0.022ms，其中 event.synchronize 0.012ms），且 batch_queue 的 ⑤
延迟一拍处理上一批——收货等待被推出关键路径。

数字全部取自真引擎实测 trace（explainer m3 spec.numbers：sync 0.358/0.225/0.246、
async 0.314/0.022/0.012/4.146、延迟一拍 [7189]、返回类型两真身）。
坐标由常量/循环计算；文本全 esc()；两栏各按本拍实测时刻布点（分段非线性）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

# ---------------- 画布与版式常量 ----------------
W, H = 1660, 868
MX, BXR = 56, 1604
COL_W = 744
LX, RX = 56.0, 860.0                     # 左/右栏左缘
PAD = 12.0
BLK_W = 178.0

# ---------------- 标题区 ----------------
lc.text(MX, 34, '④ 收货的两种路径——同步版阻塞收货 vs 异步版发起与等待分离',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '起点相同：② 已把前向 kernel 入队、GPU 后台执行；差别不在风格，'
        '在『谁在关键路径上等』（真引擎实测，同一请求场景两版各跑一遍）',
        10.5, lc.C_MUTE, 'start', maxw=1150, tag='subtitle')
_ch = '放大自 L2 ④拍片『future.result+条件 sample_tokens』· L0：循环框'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 共享起点横幅 ----------------
BN_X, BN_W, BN_Y, BN_H = 460, 740, 84, 32
lc.rect(BN_X, BN_Y, BN_W, BN_H, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.6)
lc.text(BN_X + BN_W / 2, BN_Y + 20, '② 前向 kernel 已入队 · GPU 后台执行中——两栏同一时刻起步',
        10.5, lc.C_GPU_S, 'middle', True, maxw=BN_W - 16, tag='banner')

# ---------------- 栏容器 / 栏头 / 泳道 ----------------
CONT_Y, CONT_H_L, CONT_H_R = 128, 438, 514
HDR_Y, HDR_H = 136, 34
LANE_HDR_Y, LANE_HDR_H = 182, 26
TL_Y0 = 224.0
LANE_N = 3
LANE_W = (COL_W - 2 * PAD) / LANE_N


def lane_c(col_x, i):
    return col_x + PAD + LANE_W * (i + 0.5)


lc.rect(LX, CONT_Y, COL_W, CONT_H_L, '#ffffff', lc.C_FAINT, rx=10, sw=1.2, dash=True)
lc.rect(RX, CONT_Y, COL_W, CONT_H_R, '#ffffff', lc.C_FAINT, rx=10, sw=1.2, dash=True)
lc.text(LX + 14, HDR_Y + 14, '左 · 同步版 step()——阻塞收货', 11, lc.C_ENG_S, 'start', True,
        maxw=430, tag='colL')
lc.text(LX + 14, HDR_Y + 28, '拍 3 · 批 {A:1, B:1}（双 decode）· async_scheduling=False（本章主线）',
        8.5, lc.C_MUTE, 'start', maxw=560, tag='colL:sub')
lc.text(RX + 14, HDR_Y + 14, '右 · 异步版 step_with_batch_queue()——发起与等待分离', 11,
        lc.C_ENG_S, 'start', True, maxw=520, tag='colR')
lc.text(RX + 14, HDR_Y + 28, '拍 2 · 批 {A:1, B:4}（混相）· v0.27.1 服务默认',
        8.5, lc.C_MUTE, 'start', maxw=560, tag='colR:sub')

LANE_NAMES = ['EngineCore 侧（executor）', 'worker · GPUModelRunner', 'GPU（Blackwell）']
LANE_STYLES = [(lc.C_BEAT_F, lc.C_BEAT_S, lc.C_ENG_S), (lc.C_GPU_F, lc.C_GPU_S, lc.C_GPU_S),
               (lc.C_GPU_S, lc.C_GPU_S, '#ffffff')]
for _cx_col, _cbottom in ((LX, CONT_Y + CONT_H_L), (RX, CONT_Y + CONT_H_R)):
    for _i, (_nm, (_f, _s, _t)) in enumerate(zip(LANE_NAMES, LANE_STYLES)):
        _cx = lane_c(_cx_col, _i)
        lc.rect(_cx - 112, LANE_HDR_Y, 224, LANE_HDR_H, _f, _s, rx=6, sw=1.3)
        lc.text(_cx, LANE_HDR_Y + 16.5, _nm, 9, _t, 'middle', True, maxw=214, tag='lane:' + _nm[:6])
        lc.seg(_cx, LANE_HDR_Y + LANE_HDR_H + 2, _cx, _cbottom - 8, lc.C_FAINT, 1.0, dash=True)

# ---------------- 左栏时间分段（同步版拍 3；dy = max(ms*K, min_px)） ----------------
KL = 115.0
BANDS_L = [
    ('b2', 1.341, 154),    # ② 发起
    ('g1', 0.007, 12),
    ('b4a', 0.001, 18),    # ④a
    ('g2', 0.002, 10),
    ('b4b', 0.358, 76),    # ④b（表头 16 + 三段 20×3）
    ('g3', 0.007, 12),
    ('b5', 0.056, 34),     # ⑤
    ('ret', 0, 14),        # 返回箭头行
]
YL = {'t0': TL_Y0}
_y = TL_Y0
for _key, _ms, _minh in BANDS_L:
    YL[_key + '_s'] = _y
    _y += max(_ms * KL, _minh)
    YL[_key + '_e'] = _y

# ---------------- 右栏时间分段（异步版拍 2） ----------------
KR = 44.0
BANDS_R = [
    ('b2', 4.146, 182),    # ② 发起（kernel 入队）
    ('g1', 0.017, 12),
    ('b4', 0.314, 64),     # ④ 发起即返回（4 行）
    ('g2', 0, 10),
    ('wrap', 0, 56),       # 包 AsyncOutputFuture 入 batch_queue
    ('g3', 0.016, 12),
    ('result', 0.022, 24), # result()（内 event.synchronize 0.012）
    ('g4', 0.052, 12),
    ('b5', 0.046, 44),     # ⑤ 延迟一拍
]
YR = {'t0': TL_Y0}
_y = TL_Y0
for _key, _ms, _minh in BANDS_R:
    YR[_key + '_s'] = _y
    _y += max(_ms * KR, _minh)
    YR[_key + '_e'] = _y

# 几何锚点
L_EXEC, L_WORK = lane_c(LX, 0), lane_c(LX, 1)
L_GPU = lane_c(LX, 2)
R_EXEC, R_WORK, R_GPU = lane_c(RX, 0), lane_c(RX, 1), lane_c(RX, 2)
L_GPU_X0, L_GPU_X1 = L_GPU - 85, L_GPU + 85          # 左栏前向条 583..753
R_FWD_X0, R_FWD_X1 = R_GPU - 87, R_GPU - 17          # 右栏 default stream 条
R_CP_X0, R_CP_X1 = R_GPU + 7, R_GPU + 77             # 右栏 copy stream 条

# ---------------- 左栏：同步版 ----------------
# ② 发起块
lc.rect(L_EXEC - BLK_W / 2, YL['b2_s'], BLK_W, YL['b2_e'] - YL['b2_s'],
        lc.C_BEAT_F, lc.C_BEAT_S, rx=4, sw=1.4)
_m2 = (YL['b2_s'] + YL['b2_e']) / 2
for _i, _ln in enumerate(['② execute_model', '（non_block=True）', '发起 1.341ms',
                          'kernel 入队即返回', 'Future（内 None · 已 done）']):
    lc.text(L_EXEC, _m2 - 32 + _i * 17, _ln, 9, lc.C_TXT, 'middle', maxw=BLK_W - 8,
            tag='Lb2:' + str(_i))
# ② → GPU：kernel 入队
_yk = TL_Y0 + 16
lc.seg(L_EXEC + BLK_W / 2, _yk, L_GPU_X0, _yk, lc.C_API_S, 1.8, 'dn')
lc.text((L_EXEC + BLK_W / 2 + L_GPU_X0) / 2, _yk - 6, 'kernel 入队（16 层）', 8.2,
        lc.C_API_S, 'middle', maxw=110, tag='La:launch')
# GPU 前向条（贯穿全栏）
lc.rect(L_GPU_X0, TL_Y0, L_GPU_X1 - L_GPU_X0, YL['b5_e'] - TL_Y0, lc.C_GPU_S, 'none',
        rx=3, sw=0)
_lm = (TL_Y0 + YL['b5_e']) / 2
for _i, _ln in enumerate(['前向 kernel', '后台执行', '（default stream）']):
    lc.text(L_GPU, _lm - 14 + _i * 15, _ln, 9, '#ffffff', 'middle', maxw=160, tag='Lbar:' + str(_i))
# ④a
lc.rect(L_EXEC - BLK_W / 2, YL['b4a_s'], BLK_W, YL['b4a_e'] - YL['b4a_s'],
        lc.C_BEAT_F, lc.C_BEAT_S, rx=4, sw=1.4)
lc.text(L_EXEC, YL['b4a_s'] + 12.5, '④a result 0.001ms（done→None）', 8.5, lc.C_TXT,
        'middle', True, maxw=BLK_W - 8, tag='Lb4a')
# ④b（表头 + 三段着色）
lc.rect(L_WORK - BLK_W / 2, YL['b4b_s'], BLK_W, YL['b4b_e'] - YL['b4b_s'],
        lc.C_BEAT_F, lc.C_BEAT_S, rx=4, sw=1.4)
lc.text(L_WORK, YL['b4b_s'] + 11, '④b sample_tokens 0.358ms', 8.5, lc.C_TXT, 'middle',
        True, maxw=BLK_W - 8, tag='Lb4b')
SUB_H = (YL['b4b_e'] - YL['b4b_s'] - 16) / 3
SUBS = [('等 GPU 尾程', 'wait', lc.C_MUTE), ('掩码+采样 kernel', 'gpu', '#ffffff'),
        ('同步 D2H', 'sam', '#ffffff')]
for _i, (_nm, _kind, _tc) in enumerate(SUBS):
    _sy = YL['b4b_s'] + 16 + _i * SUB_H
    _fill = 'url(#wait)' if _kind == 'wait' else (lc.C_GPU_S if _kind == 'gpu' else lc.C_SAM_S)
    _stroke = lc.C_FAINT if _kind == 'wait' else 'none'
    lc.rect(L_WORK - BLK_W / 2 + 6, _sy, BLK_W - 12, SUB_H, _fill, _stroke, rx=2, sw=1.0)
    lc.text(L_WORK, _sy + SUB_H / 2 + 3, _nm, 8.2, _tc, 'middle', maxw=BLK_W - 20,
            tag='Lb4b:sub' + str(_i))
# ④b 调用（④a 块底 → ④b 块左缘，肘形）
lc.parrow([(L_EXEC, YL['b4a_e']), (L_EXEC, YL['b4b_s'] + 9), (L_WORK - BLK_W / 2, YL['b4b_s'] + 9)],
          lc.C_API_S, 1.8, 'dn')
# ④b 等待段 → GPU 条（event 未完，虚线；完整语义见 GPU 条白字，此处短标签避让）
_yw = YL['b4b_s'] + 16 + SUB_H / 2
lc.seg(L_WORK + BLK_W / 2, _yw, L_GPU_X0, _yw, lc.C_FAINT, 1.3, dash=True)
lc.text((L_WORK + BLK_W / 2 + L_GPU_X0) / 2, _yw - 5, 'event 未完', 7.5, lc.C_MUTE,
        'middle', maxw=64, tag='La:wait')
# GPU 条上的 ④b 入口取证刻度
lc.text(L_GPU, YL['b4b_s'] + 8, '④b 入口：event 未完', 8, '#ffffff', 'middle',
        maxw=150, tag='Lbar:probe')
# ④b 返回 → ⑤（肘形：worker 块左缘 → ⑤ 块底缘）
_yr = YL['b4b_e'] + 2
lc.parrow([(L_WORK - BLK_W / 2, _yr), (L_EXEC, _yr), (L_EXEC, YL['b5_e'])],
          lc.C_ENG_S, 1.8, 'up')
lc.text(L_EXEC + 78, _yr - 7, 'ModelRunnerOutput（采样+D2H 已在内落地）', 8.5, lc.C_ENG_S,
        'middle', maxw=200, tag='La:ret4b')
# ⑤ 当拍记账
lc.rect(L_EXEC - BLK_W / 2, YL['b5_s'], BLK_W, YL['b5_e'] - YL['b5_s'],
        lc.C_ENG_F, lc.C_ENG_S, rx=4, sw=1.4)
lc.text(L_EXEC, YL['b5_s'] + 14, '⑤ 紧随其后 · 当拍记账', 8.8, lc.C_TXT, 'middle', True,
        maxw=BLK_W - 10, tag='Lb5')
lc.text(L_EXEC, YL['b5_s'] + 27, '（本拍产物本拍出账）', 8.2, lc.C_MUTE, 'middle',
        maxw=BLK_W - 10, tag='Lb5:sub')
# 左栏脚注：同步版各拍对照
lc.text(LX + PAD, CONT_Y + CONT_H_L + 18, '同步版 ④b 各拍实测：0.225 / 0.246 / 0.358 ms'
        '（prefill / 混相拍同量级）', 8.5, lc.C_MUTE, 'start', maxw=560, tag='Lnote')

# ---------------- 右栏：异步版 ----------------
# ② 发起块
lc.rect(R_EXEC - BLK_W / 2, YR['b2_s'], BLK_W, YR['b2_e'] - YR['b2_s'],
        lc.C_BEAT_F, lc.C_BEAT_S, rx=4, sw=1.4)
_m2 = (YR['b2_s'] + YR['b2_e']) / 2
for _i, _ln in enumerate(['② execute_model', '（non_block=True）', '发起 4.146ms',
                          'kernel 入队即返回']):
    lc.text(R_EXEC, _m2 - 24 + _i * 17, _ln, 9, lc.C_TXT, 'middle', maxw=BLK_W - 8,
            tag='Rb2:' + str(_i))
# ② → GPU：kernel 入队
_yk = TL_Y0 + 16
lc.seg(R_EXEC + BLK_W / 2, _yk, R_FWD_X0, _yk, lc.C_API_S, 1.8, 'dn')
lc.text((R_EXEC + BLK_W / 2 + R_FWD_X0) / 2, _yk - 6, 'kernel 入队', 8.2, lc.C_API_S,
        'middle', maxw=90, tag='Ra:launch')
# GPU default stream 前向条（贯穿）
lc.rect(R_FWD_X0, TL_Y0, R_FWD_X1 - R_FWD_X0, YR['b5_e'] - TL_Y0, lc.C_GPU_S, 'none',
        rx=3, sw=0)
lc.text((R_FWD_X0 + R_FWD_X1) / 2, TL_Y0 + 14, '前向', 8.5, '#ffffff', 'middle', True,
        maxw=60, tag='Rbar:fwd')
# ④ 发起即返回块（worker）
lc.rect(R_WORK - BLK_W / 2, YR['b4_s'], BLK_W, YR['b4_e'] - YR['b4_s'],
        lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.4)
for _i, _ln in enumerate(['④ sample_tokens', '（non_block=True）', '0.314ms 发起即返回',
                          '→ AsyncGPUModelRunnerOutput']):
    lc.text(R_WORK, YR['b4_s'] + 15 + _i * 14, _ln, 8.5, lc.C_TXT, 'middle',
            maxw=BLK_W - 8, tag='Rb4:' + str(_i))
# copy stream 条（构造时 D2H 起飞 + record event；两行短标签避让绿色前向条）
lc.rect(R_CP_X0, YR['b4_e'], R_CP_X1 - R_CP_X0, YR['result_e'] - YR['b4_e'],
        lc.C_SAM_S, 'none', rx=3, sw=0)
lc.text((R_CP_X0 + R_CP_X1) / 2, YR['b4_e'] - 20, 'copy stream：D2H 起飞', 8,
        lc.C_SAM_S, 'middle', maxw=100, tag='Rcp:top1')
lc.text((R_CP_X0 + R_CP_X1) / 2, YR['b4_e'] - 9, '+ record event', 8,
        lc.C_SAM_S, 'middle', maxw=100, tag='Rcp:top2')
# ④ 返回 → 包装（worker 块左缘 → wrap 块右缘）
_yw4 = (YR['wrap_s'] + YR['wrap_e']) / 2
lc.seg(R_WORK - BLK_W / 2, _yw4, R_EXEC + BLK_W / 2, _yw4, lc.C_ENG_S, 1.8, 'up')
lc.text((R_WORK - BLK_W / 2 + R_EXEC + BLK_W / 2) / 2 + 22, _yw4 - 7,
        'AsyncGPUModelRunnerOutput', 8.2, lc.C_ENG_S, 'middle', maxw=140, tag='Ra:ret4')
# 包装块（executor）
lc.rect(R_EXEC - BLK_W / 2, YR['wrap_s'], BLK_W, YR['wrap_e'] - YR['wrap_s'],
        lc.C_BEAT_F, lc.C_BEAT_S, rx=4, sw=1.4)
for _i, _ln in enumerate(['包成 AsyncOutputFuture', '入 batch_queue', '（⑤ 延迟一拍收上一批）']):
    lc.text(R_EXEC, YR['wrap_s'] + 16 + _i * 15, _ln, 8.5, lc.C_TXT, 'middle',
            maxw=BLK_W - 8, tag='Rwrap:' + str(_i))
# result() 块
lc.rect(R_EXEC - BLK_W / 2, YR['result_s'], BLK_W, YR['result_e'] - YR['result_s'],
        lc.C_SAM_F, lc.C_SAM_S, rx=4, sw=1.5)
lc.text(R_EXEC, (YR['result_s'] + YR['result_e']) / 2 + 3, 'result() 0.022ms', 9,
        lc.C_TXT, 'middle', True, maxw=BLK_W - 8, tag='Rresult')
# result() 只等 D2H 事件（虚线品红：copy 条事件 → result 块右缘；0.012 注并入两行标签，
# 不再贴 ⑤ 块边框放下注）
_ywait = YR['result_e'] - 4
lc.seg(R_CP_X0, _ywait, R_EXEC + BLK_W / 2, _ywait, lc.C_SAM_S, 1.4, 'sam', dash=True)
lc.circle(R_CP_X0, _ywait, 3.2, lc.C_SAM_S, 1.3, dash=False)
_wlx = (R_CP_X0 + R_EXEC + BLK_W / 2) / 2
lc.text(_wlx, _ywait - 17, 'result() 只对 D2H 事件 synchronize（不等计算）', 8.2,
        lc.C_SAM_S, 'middle', maxw=210, tag='Ra:waitline1')
lc.text(_wlx, _ywait - 5, '其中 event.synchronize（D2H 事件）0.012ms', 8.2,
        lc.C_SAM_S, 'middle', maxw=200, tag='Ra:waitline2')
# ⑤ 延迟一拍块
lc.rect(R_EXEC - BLK_W / 2, YR['b5_s'], BLK_W, YR['b5_e'] - YR['b5_s'],
        lc.C_ENG_F, lc.C_ENG_S, rx=4, sw=1.4)
for _i, _ln in enumerate(['⑤ 延迟一拍：本拍记上一拍的账', '收 beat 1 的 prefill 货：A [7189]',
                          '（beat 1 无 ⑤ · schedule 填队列优先）']):
    lc.text(R_EXEC, YR['b5_s'] + 14 + _i * 14, _ln, 8.2, lc.C_TXT, 'middle',
            maxw=BLK_W - 8, tag='Rb5:' + str(_i))
# 右栏 GPU 两条流的底部标注
lc.text((R_FWD_X0 + R_FWD_X1) / 2, YR['b5_e'] + 20, 'default stream · 前向 kernel', 8,
        lc.C_MUTE, 'middle', maxw=140, tag='Rbar:sub1')
lc.text((R_CP_X0 + R_CP_X1) / 2, YR['b5_e'] + 34, 'copy stream · D2H', 8, lc.C_SAM_S,
        'middle', maxw=100, tag='Rbar:sub2')

# ---------------- 底部对照条 ----------------
ST_Y, ST_H = 690, 92
lc.rect(MX, ST_Y, BXR - MX, ST_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(MX + 16, ST_Y + 20, '发起 vs 等待——同一场景两版对照：谁在关键路径上等', 10.5,
        lc.C_TXT, 'start', True, maxw=520, tag='strip:t')
LBL_X, SYNC_X, ASYNC_X, NOTE_X = MX + 20, 400.0, 810.0, 1180.0
CELL_W2 = 360.0
ROWS = [
    (ST_Y + 32, '② 发起（kernel 入队）', '同步 1.341ms', '异步 4.146ms',
     lc.C_ENG_F, lc.C_ENG_S),
    (ST_Y + 60, '④ 收货等待', '同步 0.358ms（阻塞收货）', '异步 0.022ms（只等 D2H）',
     lc.C_SAM_F, lc.C_SAM_S),
]
for _ry, _lab, _sc, _ac, _f, _s in ROWS:
    lc.text(LBL_X, _ry + 12, _lab, 9, lc.C_TXT, 'start', True, maxw=170, tag='row:' + _lab[:6])
    lc.rect(SYNC_X, _ry, CELL_W2, 24, _f, _s, rx=5, sw=1.3)
    lc.text(SYNC_X + CELL_W2 / 2, _ry + 16, _sc, 9, lc.C_TXT, 'middle', True,
            maxw=CELL_W2 - 12, tag='cell:' + _sc[:8])
    lc.rect(ASYNC_X, _ry, CELL_W2, 24, _f, _s, rx=5, sw=1.3)
    lc.text(ASYNC_X + CELL_W2 / 2, _ry + 16, _ac, 9, lc.C_TXT, 'middle', True,
            maxw=CELL_W2 - 12, tag='cell:' + _ac[:8])
for _i, _ln in enumerate(['异步版同拍：② 4.146ms vs ④ 0.022ms', '——差近两个数量级；发起/等待分离，',
                          '等待被推出关键路径', '同步 A [7189, 184, 7904] · B [5965, 4372]',
                          '＝ 异步（两版 final tokens 逐 token 相同·产物一致）']):
    lc.text(NOTE_X + 10, ST_Y + 32 + _i * 13, _ln, 8.5, lc.C_TXT if _i < 3 else lc.C_MUTE,
            'start', maxw=400, tag='note:' + str(_i))

# ---------------- 图例 ----------------
LEG_Y = ST_Y + ST_H + 20
lx = MX


def leg(kind, name):
    global lx
    if kind == 'gpu':
        lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_GPU_S, 'none', rx=3, sw=0)
    elif kind == 'sam':
        lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_SAM_S, 'none', rx=3, sw=0)
    elif kind == 'hatch':
        lc.rect(lx, LEG_Y - 9, 20, 12, 'url(#wait)', lc.C_FAINT, rx=3, sw=1.0)
    elif kind == 'cpu':
        lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_BEAT_F, lc.C_BEAT_S, rx=3, sw=1.3)
    elif kind == 'call':
        lc.seg(lx, LEG_Y - 3, lx + 26, LEG_Y - 3, lc.C_API_S, 2.0)
    else:
        lc.seg(lx, LEG_Y - 3, lx + 26, LEG_Y - 3, lc.C_ENG_S, 2.0)
    lc.text(lx + 26, LEG_Y + 1, name, 8.5, lc.C_TXT, 'start', maxw=300, tag='leg:' + name[:8])
    lx += 26 + lc.tw(name, 8.5) + 20


leg('gpu', 'GPU kernel（default stream）')
leg('sam', 'D2H 拷贝 / 事件（收货等待点）')
leg('hatch', '等待（非计算）')
leg('cpu', 'CPU 侧发起 / 包装')
leg('call', '调用 / 发起 →')
leg('ret', '← 返回')

# ---------------- 页脚 ----------------
lc.text(MX, H - 34, '数字取自真引擎实测：容器内钉版 v0.27.1 源树全链路 + NVIDIA RTX PRO 6000 '
        'Blackwell + tiny 随机权重 Llama（16 层）· 同一请求场景两版各跑一遍（左 = 拍 3、右 = 拍 2）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot:1')
lc.text(MX, H - 16, '逐字锚 vllm/v1/executor/uniproc_executor.py:L26-L42（AsyncOutputFuture）'
        '· L91-L106（collective_rpc 两条支）· vllm/v1/worker/gpu_model_runner.py'
        '（AsyncGPUModelRunnerOutput · copy stream D2H）· 两栏各按本拍实测时刻布点（分段非线性，时长以标注为准）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot:2')

# ---------------- 装配输出 ----------------
HATCH = ('<pattern id="wait" width="7" height="7" patternTransform="rotate(45)" '
         'patternUnits="userSpaceOnUse"><rect width="7" height="7" fill="#f1f5f9"/>'
         '<line x1="0" y1="0" x2="0" y2="7" stroke="#94a3b8" stroke-width="2"/></pattern>')
SAM_ARROW = ('<marker id="sam" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
             'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" '
             'fill="#be185d"/></marker>')
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>',
       '<defs>' + HATCH + SAM_ARROW + '</defs>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch09-fig-two-phase-two-paths.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
