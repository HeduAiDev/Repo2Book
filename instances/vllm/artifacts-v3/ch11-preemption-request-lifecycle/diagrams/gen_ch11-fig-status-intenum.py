#!/usr/bin/env python3
"""ch11 机制图 8 · RequestStatus 单 IntEnum 状态机（figure_spec ch11-fig-status-intenum，模板 state-machine）

放大自 L0 右列『调度 · 显存账本』（kv_column 青色列）上半 Scheduler 框——即本章 L2 章图
south『RequestStatus 单 IntEnum』框（第 1 站，全章地图）的机制展开；
非新架构画法，架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：请求的一生是一枚整数的取值变化：1..6 未完成（含三个阻塞子态与 PREEMPTED 中转），
7..12 终态——is_finished 就是一次『>6』比较，枚举顺序本身是隐式 API。

数字全部取自 figure_spec.numbers（全序值表 1..12 / 分界 PREEMPTED=6 与 FINISHED_STOPPED=7 /
特例映射 WAITING_FOR_STREAMING_REQ(4)→STOP / 注释级约定原文），源出配套精简版 host 实跑 trace。
状态名与注释引文逐字锚 vllm/v1/request.py:L348-L390。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 762
MX, BXR = 60, 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '请求的一生是一枚整数的取值变化：1..6 未完成，7..12 终态——判完成只问『大于 6 吗』',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, 'RequestStatus 单 IntEnum（request.py:L348-L375，enum.auto() 连续赋值）——一次整数比较纳秒级；热循环每秒十万次级的调用只付这一次比较',
        10.5, lc.C_MUTE, 'start', maxw=1040, tag='subtitle')
_ch = '放大自 L2 south『RequestStatus』· L0：调度·显存账本列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_KV_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 阶梯状态格 ----------------
STATES = [
    (1, 'WAITING', ['WAITING']),
    (2, 'WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR', ['WAITING_FOR_', 'STRUCTURED_OUTPUT', '_GRAMMAR']),
    (3, 'WAITING_FOR_REMOTE_KVS', ['WAITING_FOR_', 'REMOTE_KVS']),
    (4, 'WAITING_FOR_STREAMING_REQ', ['WAITING_FOR_', 'STREAMING_REQ']),
    (5, 'RUNNING', ['RUNNING']),
    (6, 'PREEMPTED', ['PREEMPTED']),
    (7, 'FINISHED_STOPPED', ['FINISHED_', 'STOPPED']),
    (8, 'FINISHED_LENGTH_CAPPED', ['FINISHED_LENGTH', '_CAPPED']),
    (9, 'FINISHED_ABORTED', ['FINISHED_', 'ABORTED']),
    (10, 'FINISHED_IGNORED', ['FINISHED_', 'IGNORED']),
    (11, 'FINISHED_ERROR', ['FINISHED_', 'ERROR']),
    (12, 'FINISHED_REPETITION', ['FINISHED_', 'REPETITION']),
]
FIN_REASON = {7: 'STOP', 8: 'LENGTH', 9: 'ABORT', 10: 'LENGTH', 11: 'ERROR', 12: 'REPETITION'}

CELL_W, PITCH, STEP = 100, 107, 13
X0 = 130
BASE_Y = 436            # v=1 格顶（最高值为 BASE_Y - 11*STEP）

def cell_xy(v):
    i = v - 1
    return X0 + i * PITCH, BASE_Y - i * STEP

# 分区底带
BAND_Y0, BAND_Y1 = 250, 500
DIV_X = cell_xy(7)[0] - 5          # 6|7 分界
lc.rect(X0 - 12, BAND_Y0, DIV_X - (X0 - 12), BAND_Y1 - BAND_Y0, lc.C_KV_F, 'none', rx=8, sw=0)
lc.rect(DIV_X, BAND_Y0, X0 + 11 * PITCH + CELL_W + 12 - DIV_X, BAND_Y1 - BAND_Y0, '#fff7ed', 'none', rx=8, sw=0)
lc.text((X0 - 12 + DIV_X) / 2, BAND_Y0 + 20, '未完成（1..6）：等 · 阻塞 · 跑 · 被抢', 10, lc.C_KV_S, 'middle', True,
        maxw=DIV_X - X0 - 20, tag='band1')
lc.text((DIV_X + X0 + 11 * PITCH + CELL_W + 12) / 2, BAND_Y0 + 20, '终态（7..12）：一生结束', 10, lc.C_ABORT,
        'middle', True, maxw=380, tag='band2')

# 状态格
for v, name, lines in STATES:
    x, y = cell_xy(v)
    fin = v >= 7
    stroke = lc.C_ABORT if fin else (lc.C_MUTE if 2 <= v <= 4 else lc.C_KV_S)
    fill = '#ffffff'
    lc.rect(x, y, CELL_W, 60, fill, stroke, rx=6, sw=1.6, dash=(2 <= v <= 4))
    lc.text(x + CELL_W / 2, y + 14, 'v=' + str(v), 8.2, stroke, 'middle', True, tag='vv' + str(v))
    ly0 = y + 28 if len(lines) <= 2 else y + 27
    for j, ln in enumerate(lines):
        lc.text(x + CELL_W / 2, ly0 + j * 11, ln, 7.5, '#334155', 'middle', maxw=CELL_W - 6,
                tag='nm' + str(v) + str(j))
    if v == 4:
        lc.text(x + CELL_W / 2, y + 52, '→STOP 特例', 7, lc.C_ENG_S, 'middle', True, maxw=CELL_W - 8,
                tag='sp4')
    if v == 6:
        lc.text(x + CELL_W / 2, y + 52, '分界：未完成', 7, lc.C_KV_S, 'middle', True, maxw=CELL_W - 8,
                tag='sp6')

# 6|7 粗分界线
lc.seg(DIV_X, BAND_Y0, DIV_X, BAND_Y1, lc.C_ABORT, 3.0, dash=True)
lc.text(DIV_X, BAND_Y0 - 12, 'v > 6 ⇒ is_finished', 10.5, lc.C_ABORT, 'middle', True, tag='div')

# ---------------- 转移弧（四处小旗） ----------------
# ① 调度准入：WAITING → RUNNING（上方长弧）
w1 = cell_xy(1)
r5 = cell_xy(5)
lc.parrow([(w1[0] + CELL_W / 2, w1[1]), (w1[0] + CELL_W / 2, 218), (r5[0] + CELL_W / 2, 218),
           (r5[0] + CELL_W / 2, r5[1])], lc.C_KV_S, 2.0, 'kvm')
lc.text((w1[0] + r5[0]) / 2 + CELL_W / 2, 210, '① 调度准入（WAITING → RUNNING）', 9.5, lc.C_KV_S, 'middle', True,
        maxw=340, tag='t1')
# ② 抢占 / 恢复：RUNNING ↔ PREEMPTED（双弧）
r6 = cell_xy(6)
lc.parrow([(r5[0] + 26, r5[1]), (r5[0] + 26, 262), (r6[0] + 40, 262), (r6[0] + 40, r6[1])],
          lc.C_ENG_S, 2.0, 'up')
lc.parrow([(r6[0] + 70, r6[1]), (r6[0] + 70, 288), (r5[0] + 70, 288), (r5[0] + 70, r5[1])],
          lc.C_KV_S, 2.0, 'kvm')
lc.text((r5[0] + r6[0]) / 2 + 30, 256, '② 抢占 _preempt_request', 9, lc.C_ENG_S, 'middle', True, maxw=250,
        tag='t2a')
lc.text((r5[0] + r6[0]) / 2 + 30, 302, '恢复（resumed，经 WAITING 准入）', 8.4, lc.C_KV_S, 'middle',
        maxw=250, tag='t2b')
# ③ check_stop：RUNNING → FINISHED_*（走格子下方的绕行弧，不穿 PREEMPTED 格）
f7 = cell_xy(7)
lc.parrow([(r5[0] + CELL_W / 2, r5[1] + 60), (r5[0] + CELL_W / 2, 472), (f7[0] + CELL_W / 2, 472),
           (f7[0] + CELL_W / 2, f7[1] + 60)], lc.C_ABORT, 2.0, 'ab')
lc.text((r5[0] + f7[0]) / 2 + CELL_W / 2, 486, '③ check_stop 五连判 → FINISHED_*', 9, lc.C_ABORT,
        'middle', True, maxw=280, tag='t3')
# ④ 外部 abort → FINISHED_ABORTED（上方直落）
f9 = cell_xy(9)
lc.seg(f9[0] + CELL_W / 2, 196, f9[0] + CELL_W / 2, f9[1], lc.C_ABORT, 2.0, 'ab')
lc.text(f9[0] + CELL_W / 2, 188, '④ 外部 abort（finish_requests）→ FINISHED_ABORTED', 9, lc.C_ABORT,
        'middle', True, maxw=380, tag='t4')

# 阻塞子态下划括线
bx0, bx1 = cell_xy(2)[0], cell_xy(4)[0] + CELL_W
by = 500
lc.seg(bx0, by, bx1, by, lc.C_MUTE, 1.4, dash=True)
for bx in (bx0, bx1):
    lc.seg(bx, by - 6, bx, by, lc.C_MUTE, 1.2, dash=True)
lc.text((bx0 + bx1) / 2, by + 14, '三个阻塞子态（skipped_waiting 隔离，见双队列图）', 8.4, lc.C_MUTE, 'middle',
        maxw=360, tag='blk')

# ---------------- FinishReason 映射行 ----------------
FR_Y = 548
lc.text(X0 - 12, FR_Y + 15, 'FinishReason 映射', 9.5, lc.C_MUTE, 'start', True, maxw=140, tag='fr:t')
for v, fr in FIN_REASON.items():
    x, _ = cell_xy(v)
    lc.rect(x + 8, FR_Y, CELL_W - 16, 24, '#ffffff', lc.C_MUTE, rx=4, sw=1.1)
    lc.text(x + CELL_W / 2, FR_Y + 16, fr, 8.8, '#334155', 'middle', True, maxw=CELL_W - 22, tag='fr' + str(v))
lc.text(cell_xy(4)[0] + CELL_W / 2, FR_Y + 16, 'STOP', 8.8, lc.C_ENG_S, 'middle', True, maxw=CELL_W - 22,
        tag='fr4')
lc.text(cell_xy(4)[0] + CELL_W / 2 + 64, FR_Y + 15, '← 挂起时对外仍报 STOP（特例映射）', 8, lc.C_ENG_S, 'start',
        maxw=230, tag='fr4n')

# ---------------- 底部注记 ----------------
AN_Y = 596
lc.rect(MX, AN_Y, 700, 108, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.4)
lc.text(MX + 16, AN_Y + 22, 'is_finished = status > RequestStatus.PREEMPTED（L369-L371）', 10.5, lc.C_KV_S,
        'start', True, maxw=660, tag='an1:t')
for j, ln in enumerate(['· 一次整数比较（纳秒级）；热循环的幂等跳过、abort 的入口过滤每请求每拍至少调一次，',
                        '  千级请求 × 每秒上百拍 = 每秒十万次级（转移点集中四处，如上图小旗）',
                        '· 转移点：调度准入 / 抢占 / check_stop / 外部 abort finish_requests']):
    lc.text(MX + 16, AN_Y + 44 + j * 17, ln, 8.8, '#334155', 'start', maxw=670, tag='an1:l' + str(j))

AN2_X = MX + 724
lc.rect(AN2_X, AN_Y, BXR - AN2_X, 108, '#ffffff', lc.C_ABORT, rx=8, sw=1.3, dash=True)
lc.text(AN2_X + 16, AN_Y + 22, '代价：枚举顺序是隐式 API（全仓无断言保护）', 10.5, lc.C_ABORT, 'start', True,
        maxw=430, tag='an2:t')
for j, ln in enumerate(['注释级约定（L357-L358 原文）：「anything after PREEMPTED will be',
                        'considered as a finished status.」——新状态必须插在 PREEMPTED 的正确一侧，',
                        '插错不报错、只静默改变 is_finished 含义；状态图也非单向 DAG（假终点回 WAITING）']):
    lc.text(AN2_X + 16, AN_Y + 44 + j * 17, ln, 8.8, '#334155', 'start', maxw=BXR - AN2_X - 30,
            tag='an2:l' + str(j))

# ---------------- 页脚 ----------------
lc.text(MX, AN_Y + 132, '逐字锚 vllm/v1/request.py:L348-L390（RequestStatus 12 态 + enum.auto 连续值 + _FINISHED_REASON_MAP）'
        '/ L369-L371（is_finished）/ L357-L358（注释约定）· 值表与 FinishReason 映射取自配套精简版 host 实跑 · 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS,
       '<defs><marker id="kvm" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
       f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_KV_S}"/></marker></defs>']
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch11-fig-status-intenum.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
