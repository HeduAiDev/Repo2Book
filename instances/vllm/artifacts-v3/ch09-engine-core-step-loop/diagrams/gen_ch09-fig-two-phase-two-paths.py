#!/usr/bin/env python3
"""ch09 机制图 · ④ 收货的两种路径——UML 时序图文法 v3（figure_spec
ch09-fig-two-phase-two-paths）

v3 重建（2026-09-05，读者反馈两轮『生硬连接、没有语义』后按 Lead 规格书重写）：
  文法铁律——①每栏 4 条竖直生命线（EngineCore/Worker/GPU/D2H copy stream，顶部名牌）；
  ②统一竖直线性时间轴 + 浅色水平 gridlines 贯穿 4 线，栏内一切元素严格按时间戳 y 对齐
  （左栏 K=210px/ms、右栏 K=84px/ms，右栏恰压缩 2.5×，各栏左缘 1ms 比例尺）；
  ③活动条骑生命线：y=时间戳、高=时长×K（dur×K<8px 的瞬时动作退化为骑线时刻标记，
  标注真实值）；④消息=水平直线（A 线某时刻→B 线同一 y，箭头+动词标签），
  禁折线/肘形/绕行，跨中间生命线的水平线是 UML 标准画法；
  ⑤GPU 完成区间 (1.412,1.770]（左栏）用水平虚线 + 开闭括号标签（同图 A 口径）。

数据全部取自真引擎实测 trace（m1_real.json：左=sync beat 3、右=async 拍 2；
区间端点 t 字段=结束时刻，起点=t−dur）。印在图上的数字仅 spec 集：
1.341/4.146/1.281/0.045/0.001/0.358/0.056/0.314/0.022/0.012/0.225/0.246/
1.860/1.873/(1.412,1.770]/[7189]/批形状。布局时刻不印数。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1660, 800
MX, BXR = 56, 1604
TL_Y0 = 170.0

# ---------------- 生命线（每栏 4 条，同构并排；线距 208） ----------------
LANE_NAMES = ['EngineCore (CPU)', 'Worker (CPU)', 'GPU', 'D2H copy stream']
L_LANES = [118.0, 326.0, 534.0, 742.0]
R_LANES = [930.0, 1138.0, 1346.0, 1554.0]
NAMEPLATE_W = 104
LX0, LX1 = 66.0, 794.0      # 左栏 gridlines 贯穿范围
RX0, RX1 = 878.0, 1606.0

# ---------------- 时间轴（线性；K 单位 px/ms；右栏恰压缩 2.5×） ----------------
K_L, K_R = 210.0, 84.0
T_END_L, T_END_R = 1.873, 4.659


def yl(t):
    return TL_Y0 + t * K_L


def yr(t):
    return TL_Y0 + t * K_R


# ---------------- 数据（m1_real.json；区间=(起点, 终点)，端点=trace 时刻） ----------------
# 左栏 sync beat 3
L_ACTS = [
    # (lane_idx, t0, t1, style, label 行, 标签侧)
    (0, 0.006, 0.051, 'eng', ['① schedule 0.045ms'], 'r-below'),
    (0, 0.054, 1.395, 'eng', ['② execute_model 1.341ms', '（non_block=True 发起）'], 'r'),
    (1, 0.068, 1.349, 'wrk', ['launch 1.281ms', '（16 层 kernel 入队）'], 'r'),
    (1, 1.412, 1.770, 'beat3', None, None),          # ④b 三段（专用绘制）
    (0, 1.777, 1.833, 'eng', ['⑤ update 0.056ms'], 'r'),
    (2, 1.349, 1.770, 'gpu', ['前向 kernel', '（GPU 后台执行）'], 'r'),
    (3, 1.651, 1.770, 'd2h', ['同步 D2H'], 'r'),     # 段界=④b 三段均分（无数字标注）
]
L_MARKS = [  # (lane_idx, t_center, 标签, 标签侧, dy微移±3)
    (0, 1.4005, '③ bitmask 0.001ms', 'left', -3.5),
    (0, 1.409, '④a result 0.001ms', 'right', 3.5),
]
L_MSGS = [  # (x_from, x_to, t, 颜色, 标签, 标签上下, marker)
    (L_LANES[0], L_LANES[1], 0.054, 'call', 'execute_model(non_block=True) →', 'above', 'dn'),
    (L_LANES[1], L_LANES[2], 0.068, 'call', 'launch kernels（16 层入队）→', 'above', 'dn'),
    (L_LANES[1], L_LANES[0], 1.395, 'ret', '← 返回 None（launch 完即交回）', 'above', 'up'),
    (L_LANES[0], L_LANES[1], 1.412, 'call', 'sample_tokens(grammar) →', 'below', 'dn'),
    (L_LANES[1], L_LANES[0], 1.770, 'ret', '← ModelRunnerOutput（D2H 已在内）', 'above', 'up'),
    (L_LANES[1], L_LANES[3], 1.651, 'ret', '同步 D2H（阻塞拷贝）→', 'below', 'std'),
    (L_LANES[3], L_LANES[1], 1.770, 'sam', '← 数据落地', 'below', 'sam'),
]

# 右栏 async 拍 2
R_ACTS = [
    (0, 0.005, 0.078, 'eng', ['① schedule'], 'r-below'),
    (0, 0.081, 4.227, 'eng', ['② execute_model 4.146ms', '（non_block=True 发起）'], 'r'),
    (1, 0.098, 4.166, 'wrk', ['launch（16 层 kernel 入队）'], 'r'),
    (1, 4.260, 4.547, 'wrk3', None, None),           # ④ worker 段（专用绘制）
    (2, 0.098, 4.588, 'gpu', ['前向 kernel（GPU 后台）'], 'r'),
    (3, 4.244, 4.588, 'd2h', ['copy stream D2H', '（构造时起飞→事件置位）'], 'r'),
]
R_MARKS = [
    (0, 4.2385, '③ bitmask', 'left', -3.0),
    (0, 4.582, ['result() 0.022ms', '（内 D2H.get_output 0.012ms）'], 'right', 0.0),
    (0, 4.646, ['⑤ 延迟一拍：本拍记 beat 1 的账', '（A [7189]；schedule 填队列优先）'],
     'left', 4.0),
]
R_MSGS = [
    (R_LANES[0], R_LANES[1], 0.081, 'call', 'execute_model(non_block=True) →', 'above', 'dn'),
    (R_LANES[1], R_LANES[2], 0.098, 'call', 'launch kernels →', 'above', 'dn'),
    (R_LANES[1], R_LANES[0], 4.227, 'ret', '← 返回 None（暂存 worker，采样欠着）', 'above', 'up'),
    (R_LANES[0], R_LANES[1], 4.244, 'call', 'sample_tokens(non_block=True) →', 'below', 'dn'),
    (R_LANES[1], R_LANES[0], 4.558, 'ret', '← AsyncGPUModelRunnerOutput', 'above', 'up'),
    (R_LANES[3], R_LANES[0], 4.588, 'sam', '← 事件就绪（D2H 拷贝事件）', 'below', 'sam'),
]

# ---------------- 标题区 / 顶幅 ----------------
lc.text(MX, 34, '④ 收货的两种路径——同步版阻塞收货 vs 异步版发起与等待分离',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '起点相同：② 已把前向 kernel 入队、GPU 后台执行；差别不在风格，'
        '在『谁在关键路径上等』（真引擎实测，同一请求场景两版各跑一遍）',
        10.5, lc.C_MUTE, 'start', maxw=1150, tag='subtitle')
_ch = '放大自 L2 ④拍片『future.result+条件 sample_tokens』· L0：循环框'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')
BN_X, BN_W, BN_Y, BN_H = 460, 740, 72, 40
lc.rect(BN_X, BN_Y, BN_W, BN_H, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.6)
lc.text(BN_X + BN_W / 2, BN_Y + 14, '② 前向 kernel 已入队 · GPU 后台执行中——两栏同一时刻起步',
        10.5, lc.C_GPU_S, 'middle', True, maxw=BN_W - 16, tag='banner')
lc.text(BN_X + BN_W / 2, BN_Y + 31, '两栏时间轴比例不同——右栏恰压缩 2.5×（1ms 比例尺见各栏左缘；'
        '瞬时动作按最小可见高度画、标注真实值）', 8.5, lc.C_MUTE, 'middle',
        maxw=BN_W - 16, tag='banner:scale')

# ---------------- 生命线 + 名牌 + 栏头 ----------------
NP_Y, NP_H = 138, 22
LANE_STYLES = [(lc.C_ENG_F, lc.C_ENG_S, lc.C_ENG_S), (lc.C_GPU_F, lc.C_GPU_S, lc.C_GPU_S),
               (lc.C_GPU_S, lc.C_GPU_S, '#ffffff'), (lc.C_SAM_F, lc.C_SAM_S, lc.C_SAM_S)]
COL_META = [
    (L_LANES, LX0, LX1, yl, T_END_L,
     '左 · 同步版 step()——阻塞收货（拍 3 · 批 {A:1, B:1} 双 decode）', 'async_scheduling=False'),
    (R_LANES, RX0, RX1, yr, T_END_R,
     '右 · 异步版 step_with_batch_queue()——发起与等待分离（拍 2 · 批 {A:1, B:4} 混相）',
     'v0.27.1 服务默认'),
]
for lanes, gx0, gx1, tf, t_end, col_title, col_note in COL_META:
    for i, (nm, (fill, stroke, tcol)) in enumerate(zip(LANE_NAMES, LANE_STYLES)):
        lc.rect(lanes[i] - NAMEPLATE_W / 2, NP_Y, NAMEPLATE_W, NP_H, fill, stroke, rx=6, sw=1.3)
        lc.text(lanes[i], NP_Y + 14.5, nm, 8.5, tcol, 'middle', True, maxw=NAMEPLATE_W - 8,
                tag='np:' + nm[:6])
        lc.seg(lanes[i], NP_Y + NP_H + 2, lanes[i], tf(t_end) + 8,
               lc.C_FAINT, 1.0, dash=True)
    lc.text(gx0, 122, col_title, 10.5, lc.C_ENG_S, 'start', True, maxw=620, tag='col:t')
    lc.text(gx1, 122, col_note, 8.5, lc.C_MUTE, 'end', maxw=180, tag='col:note')

# ---------------- 1ms 比例尺（各栏左缘） ----------------
for sx, k in ((76.0, K_L), (888.0, K_R)):
    lc.seg(sx, TL_Y0, sx, TL_Y0 + k, lc.C_MUTE, 1.4)
    lc.seg(sx - 5, TL_Y0, sx + 5, TL_Y0, lc.C_MUTE, 1.4)
    lc.seg(sx - 5, TL_Y0 + k, sx + 5, TL_Y0 + k, lc.C_MUTE, 1.4)
    lc.text(sx, TL_Y0 + k + 14, '1ms', 7.5, lc.C_MUTE, 'middle', tag='scale')

# ---------------- gridlines（事件边界贯穿线 + 主刻度线） ----------------
L_EVENTS = [0.051, 0.068, 1.349, 1.395, 1.412, 1.770, 1.777, 1.833, 1.860]
L_MAJOR = [0.5, 1.0, 1.5]
R_EVENTS = [0.078, 0.098, 4.166, 4.227, 4.244, 4.547, 4.588, 4.646, 4.659]
R_MAJOR = [1.0, 2.0, 3.0, 4.0]
for t in L_EVENTS:
    lc.seg(LX0, yl(t), LX1, yl(t), '#e2e8f0', 0.8)
for t in L_MAJOR:
    lc.seg(LX0 - 8, yl(t), LX1, yl(t), '#cbd5e1', 0.9)
    lc.text(104, yl(t) + 3, ('%.1f' % t), 7.5, lc.C_MUTE, 'end', tag='ax' + str(t))
for t in R_EVENTS:
    lc.seg(RX0, yr(t), RX1, yr(t), '#e2e8f0', 0.8)
for t in R_MAJOR:
    lc.seg(RX0 - 8, yr(t), RX1, yr(t), '#cbd5e1', 0.9)
    lc.text(916, yr(t) + 3, ('%.0f' % t), 7.5, lc.C_MUTE, 'end', tag='ax' + str(t))
lc.text(104, yl(1.860) + 3, '拍尾 1.860', 7.5, lc.C_MUTE, 'end', tag='ax:end')

# ---------------- 活动条 / 时刻标记 / 消息 ----------------
ACT_FILL = {'eng': (lc.C_ENG_F, lc.C_ENG_S), 'wrk': (lc.C_GPU_F, lc.C_GPU_S),
            'wrk3': (lc.C_BEAT_F, lc.C_BEAT_S), 'gpu': (lc.C_GPU_S, 'none'),
            'd2h': (lc.C_SAM_S, 'none')}
MSG_COLOR = {'call': lc.C_API_S, 'ret': lc.C_ENG_S, 'sam': lc.C_SAM_S}
MSG_MARKER = {'dn': 'dn', 'up': 'up', 'std': 'std', 'sam': 'sam'}
MIN_MARK = 8.0     # dur×K 低于此 → 时刻标记（方块，中心=时间戳，允许 ±3px 微移）
MIN_BAR = 8.0      # 活动条最小可见高（中心锚定时间戳中点，标注真实值）


def draw_column(lanes, tf, acts, marks, msgs, lane_tag):
    for lane_i, t0, t1, style, label_lines, side in acts:
        x = lanes[lane_i]
        y0, y1 = tf(t0), tf(t1)
        h = y1 - y0
        if style == 'beat3':
            # ④b/④ 活动条：外框 + 条内竖直三段（④b 三段均分，段界无数字）
            y_mid, y_top = (y0 + y1) / 2, y0
            if h < MIN_BAR * 1.5:
                y_top, h = y_mid - MIN_BAR, MIN_BAR * 2
            fill, stroke = ACT_FILL['wrk3']
            lc.rect(x - 6, y_top, 12, h, fill, stroke, rx=2, sw=1.1)
            seg_h = h / 3
            subs = [('url(#wait)', lc.C_FAINT), (lc.C_GPU_S, 'none'), (lc.C_SAM_S, 'none')]
            for si, (sf, ss) in enumerate(subs):
                lc.rect(x - 4.5, y_top + 1 + si * seg_h, 9, seg_h - 1, sf, ss, rx=1.5, sw=0.8)
            lc.text(x + 11, y_top - 6, ('④b sample_tokens 0.358ms' if lane_tag == 'L'
                                        else '④ sample_tokens 0.314ms'), 8, lc.C_TXT, 'start',
                    True, maxw=170, tag='act4' + lane_tag)
            continue
        if h < MIN_BAR:   # 瞬时动作 → 时刻标记
            ym = (y0 + y1) / 2
            fill, stroke = ACT_FILL[style]
            lc.rect(x - 4.5, ym - 4, 9, 8, '#ffffff', stroke, rx=1.5, sw=1.1)
            continue
        fill, stroke = ACT_FILL[style]
        lc.rect(x - 6, y0, 12, h, fill, stroke, rx=2, sw=1.1)
        if label_lines:
            for li, ln in enumerate(label_lines):
                ly = y0 + h / 2 + 5 + li * 12 - (len(label_lines) - 1) * 6
                if side == 'r-below':
                    ly = y1 + 12 + li * 12
                lc.text(x + 11, ly, ln, 8, lc.C_TXT if li == 0 else lc.C_MUTE, 'start',
                        maxw=186, tag='act' + lane_tag + ln[:6])
    for lane_i, tc, label, side, dy in marks:
        x = lanes[lane_i]
        ym = tf(tc) + dy
        lc.rect(x - 4.5, ym - 4, 9, 8, '#ffffff', lc.C_ENG_S, rx=1.5, sw=1.1)
        lines = label if isinstance(label, (list, tuple)) else [label]
        for li, ln in enumerate(lines):
            if side == 'left':
                lc.text(x - 9, ym + 3 + li * 11, ln, 7.8, lc.C_ENG_S, 'end', maxw=150,
                        tag='mk' + ln[:6])
            else:
                lc.text(x + 9, ym + 3 + li * 11, ln, 7.8, lc.C_ENG_S, 'start', maxw=186,
                        tag='mk' + ln[:6])
    for xf, xt, t, kind, label, pos, mk in msgs:
        y = tf(t)
        lc.seg(xf, y, xt, y, MSG_COLOR[kind], 1.7, MSG_MARKER[mk])
        mx = (xf + xt) / 2
        if pos == 'above':
            lc.text(mx, y - 5, label, 7.8, MSG_COLOR[kind], 'middle', maxw=200,
                    tag='msg' + label[:6])
        else:
            lc.text(mx, y + 12, label, 7.8, MSG_COLOR[kind], 'middle', maxw=200,
                    tag='msg' + label[:6])


draw_column(L_LANES, yl, L_ACTS, L_MARKS, L_MSGS, 'L')
draw_column(R_LANES, yr, R_ACTS, R_MARKS, R_MSGS, 'R')

# ---------------- 左栏：GPU 完成区间（水平虚线 + 开闭括号，同图 A 口径） ----------------
IVL_T0, IVL_T1 = 1.412, 1.770
IVL_X0, IVL_X1 = L_LANES[1] + 14, L_LANES[2] - 8      # ④b 条旁 → GPU 生命线
for t in (IVL_T0, IVL_T1):
    lc.seg(IVL_X0, yl(t), IVL_X1, yl(t), lc.C_KV_S, 1.1, dash=True)
lc.circle(IVL_X0, yl(IVL_T0), 3.0, lc.C_KV_S, 1.1, dash=False)          # 开端空心圆
lc.seg(IVL_X0 - 5, yl(IVL_T1), IVL_X0 + 5, yl(IVL_T1), lc.C_KV_S, 2.0)  # 闭端刻度
lc.seg(IVL_X0, yl(IVL_T0), IVL_X0, yl(IVL_T1), lc.C_KV_S, 1.1, dash=True)
lc.text((IVL_X0 + IVL_X1) / 2 + 6, (yl(IVL_T0) + yl(IVL_T1)) / 2 + 3,
        '完成 ∈ (1.412, 1.770]', 8, lc.C_KV_S, 'middle', maxw=130, tag='ivl')
# fwd_done 拍尾取证上界（GPU 线上虚线延伸）
lc.seg(L_LANES[2], yl(IVL_T1), L_LANES[2], yl(1.873), lc.C_GPU_S, 1.3, dash=True)
lc.circle(L_LANES[2], yl(1.873), 3.2, lc.C_KV_S, 1.2, dash=False)
lc.text(L_LANES[2] + 8, yl(1.873) + 8, '拍尾取证上界 1.873ms', 7.8, lc.C_KV_S, 'start',
        maxw=150, tag='fwdub')

# 右栏 D2H 事件置位 → result 标记的呼应线（水平虚线，D2H 条底时刻）
lc.seg(R_LANES[3] - 6, yr(4.588), R_LANES[0] + 14, yr(4.588), lc.C_SAM_S, 1.1, dash=True)

# ---------------- 底部对照条（保留） ----------------
ST_Y, ST_H = 620, 92
lc.rect(MX, ST_Y, BXR - MX, ST_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(MX + 16, ST_Y + 20, '发起 vs 等待——同一场景两版对照：谁在关键路径上等', 10.5,
        lc.C_TXT, 'start', True, maxw=520, tag='strip:t')
LBL_X, SYNC_X, ASYNC_X, NOTE_X = MX + 20, 400.0, 810.0, 1180.0
CELL_W2 = 360.0
ROWS = [
    (ST_Y + 32, '② 发起（kernel 入队）', '同步 1.341ms', '异步 4.146ms', lc.C_ENG_F, lc.C_ENG_S),
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

# 左栏脚注（同步版各拍对照，保留）
lc.text(66, 592, '同步版 ④b 各拍实测：0.225 / 0.246 / 0.358ms（prefill / 混相拍同量级）',
        8.5, lc.C_MUTE, 'start', maxw=560, tag='Lnote')

# ---------------- 图例 ----------------
LEG_Y = ST_Y + ST_H + 20
lx = MX


def leg(kind, name):
    global lx
    if kind == 'eng':
        lc.rect(lx, LEG_Y - 9, 8, 12, lc.C_ENG_F, lc.C_ENG_S, rx=1.5, sw=1.1)
    elif kind == 'wrk':
        lc.rect(lx + 2, LEG_Y - 9, 8, 12, lc.C_GPU_F, lc.C_GPU_S, rx=1.5, sw=1.1)
    elif kind == 'gpu':
        lc.rect(lx + 4, LEG_Y - 9, 8, 12, lc.C_GPU_S, 'none', rx=1.5, sw=0)
    elif kind == 'd2h':
        lc.rect(lx + 6, LEG_Y - 9, 8, 12, lc.C_SAM_S, 'none', rx=1.5, sw=0)
    elif kind == 'hatch':
        lc.rect(lx + 8, LEG_Y - 9, 8, 12, 'url(#wait)', lc.C_FAINT, rx=1.5, sw=1.0)
    elif kind == 'mark':
        lc.rect(lx + 10, LEG_Y - 6, 9, 8, '#ffffff', lc.C_ENG_S, rx=1.5, sw=1.1)
    elif kind == 'call':
        lc.seg(lx + 14, LEG_Y - 3, lx + 40, LEG_Y - 3, lc.C_API_S, 1.7, 'dn')
    elif kind == 'grid':
        lc.seg(lx + 14, LEG_Y - 3, lx + 40, LEG_Y - 3, '#cbd5e1', 0.9)
    else:
        lc.seg(lx + 14, LEG_Y - 3, lx + 40, LEG_Y - 3, lc.C_ENG_S, 1.7, 'up')
    lc.text(lx + 46, LEG_Y + 1, name, 8.5, lc.C_TXT, 'start', maxw=300, tag='leg:' + name[:8])
    lx += 46 + lc.tw(name, 8.5) + 20


leg('eng', 'EngineCore 活动')
leg('wrk', 'Worker 活动')
leg('gpu', 'GPU 前向执行')
leg('d2h', 'D2H 拷贝')
leg('hatch', '等 GPU 尾程（非计算）')
leg('mark', '瞬时动作标记')
leg('grid', 'gridline=同一时刻')
leg('call', '消息·调用 →')
leg('ret', '← 消息·返回')

# ---------------- 页脚 ----------------
lc.text(MX, H - 30, '数字取自真引擎实测：容器内钉版 v0.27.1 源树全链路 + NVIDIA RTX PRO 6000 '
        'Blackwell + tiny 随机权重 Llama（16 层）· 同一请求场景两版各跑一遍（左 = 拍 3、右 = 拍 2）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot:1')
lc.text(MX, H - 12, '逐字锚 vllm/v1/executor/uniproc_executor.py:L26-L42（AsyncOutputFuture）'
        '· L91-L106（collective_rpc 两条支）· vllm/v1/worker/gpu_model_runner.py'
        '（AsyncGPUModelRunnerOutput · copy stream D2H）· 线性时间轴：y=时间戳×比例尺'
        '（右栏恰压缩 2.5×）；瞬时动作最小可见高度、标注真实值',
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
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems, '
      f'K_L={K_L} K_R={K_R} ratio={K_L / K_R:.2f}, '
      f'L axis {TL_Y0:.0f}..{yl(T_END_L):.0f}, R axis {TL_Y0:.0f}..{yr(T_END_R):.0f})')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
