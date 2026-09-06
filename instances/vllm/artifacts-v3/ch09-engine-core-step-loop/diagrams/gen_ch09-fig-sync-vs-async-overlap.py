#!/usr/bin/env python3
"""ch09 机制图 · 同步 vs 异步：双面板共享时间轴的重叠对照（figure_spec m11，
模板 swimlane·双面板 swimlane timeline，上下面板同一时间比例尺）

放大自 L0 循环框（loop_box）的跨拍时间维：m1 图（ch09-fig-five-beats-timeline）画
『一拍之内』五段顺序，图 B（ch09-fig-two-phase-two-paths）画 ④ 收货两路径的微观，
本图把连续三拍的 CPU 轨×GPU 轨放到同一时间轴上，回答『拍与拍之间谁在等谁』。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：同一 27B 场景、同一批形 {req-A:1, req-B:1} 的稳态实测——同步版 GPU 每拍断流
一次（0.315/0.260/0.239ms）且 CPU 被 ④b 钉住 23.442ms 干等自家前向（≈22.7ms 斜纹）；
异步版把每拍 ≈2ms 发射窗垫进上一拍 GPU 执行之下（2.082ms，同步版恒 0.0），GPU 三拍
背靠背（缝隙 0.000/0.000/0.001ms），拍距中位 24.861→21.155ms（吞吐 +17.5%）——
重叠不是把谁变快，是不让 GPU 等任何人。

数字全部取自 explainer/traces/m1b_27b.json（27B 真机实测，pin v0.27.1 源树 +
Blackwell，循环内零 synchronize）——gen 脚本直接读该 trace，零转写：
  · sync 稳态窗第 11-13 拍（derived.sync.rows i=11/12/13，第 14 拍起点用于第 3 段拍距）
  · async 稳态窗第 12-14 拍（derived.async.rows i=12/13/14，第 15 拍起点同上），
    上一拍 GPU 条取 raw async_run 第 11 拍（145.767→166.924）
  · 宏观节省 derived.comparison；稳态均值两侧 summary；注脚两极端形态取
    async_run/sync_run beats（2.912/0.019/25.447）与 traces/m1_real.json（3.853/1.860）
坐标由常量/循环从 trace 计算；文本全 esc()；上下两面板同一比例尺 K=20px/ms（10ms=200px），
时间轴刻度＝绝对时刻（ms）。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

# ---------------- trace（唯一数字真相源，零转写） ----------------
TR = json.loads((HERE.parent / 'explainer' / 'traces' / 'm1b_27b.json').read_text(encoding='utf-8'))
DV = TR['derived']


def row(tag, i):
    rs = [r for r in DV[tag]['rows'] if r['i'] == i]
    assert rs, f'{tag} row i={i} missing'
    return rs[0]


SYNC = [row('sync', i) for i in (11, 12, 13)]
SYNC_NEXT = row('sync', 14)                       # 第 3 段拍距终点（下一拍 GPU 起点）
ASYN = [row('async', i) for i in (12, 13, 14)]
ASYN_NEXT = row('async', 15)
PREV = TR['async_run']['beats'][11]               # 上一拍（GPU 条 145.767→166.924）
assert abs(PREV['gpu_launch_start_ms'] - 145.767) < 1e-6 and abs(PREV['gpu_samp_done_ms'] - 166.924) < 1e-6
CMP, SSUM, ASUM = DV['comparison'], DV['sync']['summary'], DV['async']['summary']
assert CMP['sync_beat_period_gpu_ms'] == 24.861 and CMP['async_beat_period_gpu_ms'] == 21.155

# 注脚两极端形态（同源实测；tiny 对照取 m1_real.json 的既有口径数字）
ASYNC_EARLY_RET, ASYNC_EARLY_WAIT, SYNC_SAME_BEAT = 2.912, 0.019, 25.447
TINY_ASYNC, TINY_SYNC = 3.853, 1.860
b_early = [b for b in TR['async_run']['beats'] if abs(b['step_wall_ms'] - ASYNC_EARLY_RET) < 1e-9]
b_sync = [b for b in TR['sync_run']['beats'] if abs(b['step_wall_ms'] - SYNC_SAME_BEAT) < 1e-9]
assert b_early and b_sync, 'prefill 后首拍对照拍未找到'

# ---------------- 画布与版式常量 ----------------
W, H = 1960, 964
MX = 40
K = 20.0                    # px/ms —— 上下两面板同一时间比例尺（10ms = 200px）
PX0 = 200.0                 # 两面板时间轴起点 x（各自窗口的 t0）
LANE_LX = MX                # 左侧泳道名牌 x

C_HATCHBG, C_HATCHLN = '#fff7ed', '#fdba74'       # CPU 阻塞等待斜纹（橙系）
C_GAPS, C_GAPF = '#d97706', '#fde68a'             # GPU 断流（琥珀）
C_SAMP = '#86efac'                                 # GPU 采样+同步 D2H 尾（浅绿）
C_OVS, C_OVF = '#4f46e5', '#eef2ff'               # 重叠高亮（靛蓝虚框）
C_GRID = '#e2e8f0'


def dbl_arrow(x1, x2, y, color, sw=1.6):
    """拍距标尺：双端箭头（marker-start/end=dim），端点由调用方保证落在延伸刻度线上。"""
    s = (f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{color}" '
         f'stroke-width="{sw}" marker-start="url(#dim)" marker-end="url(#dim)"/>')
    lc.ELEMS.append(((min(x1, x2) - 8, y - 8, max(x1, x2) + 8, y + 8), s))


def dot(x, y, r, fill):
    s = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}"/>'
    lc.ELEMS.append(((x - r - 2, y - r - 2, x + r + 2, y + r + 2), s))


def vgrid(x, y0, y1):
    s = f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y1:.1f}" stroke="{C_GRID}" stroke-width="1"/>'
    lc.ELEMS.append(((x - 2, y0, x + 2, y1), s))


def leader(x, y1, y2, color=lc.C_MUTE, sw=1.0):
    lc.seg(x, y1, x, y2, color, sw)


# ---------------- 标题区 ----------------
lc.text(MX, 34, '同步 vs 异步：重叠不是把谁变快，是不让 GPU 等任何人', 17, lc.C_TXT,
        'start', True, maxw=1150, tag='title')
lc.text(MX, 58, '27B 双 decode 稳态（批 {req-A:1, req-B:1}）连续三拍的 CPU×GPU 双轨对照——'
        '上：每拍『发射 → 干等自家前向 → 收尾』GPU 断流一次；下：发射垫进上拍执行、'
        '等待被 GPU 掩护，三拍背靠背（拍距中位 24.861 → 21.155ms，吞吐 +17.5%）',
        10.5, lc.C_MUTE, 'start', maxw=1560, tag='subtitle')
_ch = '放大自 L2 拍片行（跨拍时间维）· L0：循环框'
_cw = lc.chip_w(_ch)
lc.rect(W - MX - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(W - MX - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')


# ---------------- 面板绘制函数 ----------------
def panel(P, tag):
    """P = dict(header, t0, t1, launches[], cpu_beats[], gpu_bars[], ticks[], ...)，
    返回本面板各关键 y。几何常量全部由调用方传入 P 内。"""
    fx0, fx1, fy0, fy1 = P['frame']
    x = lambda t: PX0 + (t - P['t0']) * K                     # noqa: E731
    y_rul, y_rowA, y_rowB = P['y_rul'], P['y_rowA'], P['y_rowB']
    cy0, cy1 = P['cpu_lane']
    gy0, gy1 = P['gpu_lane']
    blk_t, blk_b = P['blk']
    gb_t, gb_b = P['gblk']
    y_st1, y_st2, y_cap, y_dim, y_val = P['y_st1'], P['y_st2'], P['y_cap'], P['y_dim'], P['y_val']

    # 面板外框 + 表头
    lc.rect(fx0, fy0, fx1 - fx0, fy1 - fy0, '#ffffff', lc.C_FAINT, rx=8, sw=1.2)
    lc.text(MX, P['y_hdr'], P['header'], 13, lc.C_TXT, 'start', True,
            maxw=fx1 - MX - 480, tag=tag + ':hdr')
    lc.text(fx1 - 10, P['y_hdr'], '同一时间比例尺：10ms = 200px（与' + P['peer'] + '幅一致）',
            8.5, lc.C_MUTE, 'end', tag=tag + ':scale')

    # 绝对时刻刻度（10ms 一格 → 网格线贯穿两泳道）
    t = P['tick0']
    while t <= P['t1'] + 1e-9:
        vgrid(x(t), y_rul + 2, gy1 + 4)
        lc.seg(x(t), y_rul - 6, x(t), y_rul, lc.C_MUTE, 1.0)
        lc.text(x(t), y_rul - 10, f'{t:.0f}', 8, lc.C_MUTE, 'middle', tag=tag + ':tick')
        t += 10
    lc.text(x(P['t1']) + 12, y_rul - 10, 'ms', 8, lc.C_MUTE, 'start', tag=tag + ':unit')

    # 泳道名牌（框外左侧）
    for (nm, sub), (ly0, ly1) in zip((('CPU 轨', 'EngineCore 单线程'),
                                      ('GPU 轨', 'default stream 串行')),
                                     ((cy0, cy1), (gy0, gy1))):
        ym = (ly0 + ly1) / 2
        lc.text(LANE_LX, ym - 3, nm, 11, lc.C_TXT, 'start', True, maxw=140, tag=tag + ':lane')
        lc.text(LANE_LX, ym + 11, sub, 8.5, lc.C_MUTE, 'start', maxw=140, tag=tag + ':lane:sub')

    # 泳道底色带
    lc.rect(PX0 - 8, cy0, fx1 - PX0 - 6, cy1 - cy0, lc.C_ENG_F, 'none', rx=4, sw=0)
    lc.rect(PX0 - 8, gy0, fx1 - PX0 - 6, gy1 - gy0, lc.C_GPU_F, 'none', rx=4, sw=0)

    # 底层高亮列（重叠靛蓝 / 断流琥珀洗）——先画，块和条盖在其上
    for (ta, tb, kind) in P['underlays']:
        lc.rect(x(ta), cy0, max(x(tb) - x(ta), 1.0), gy1 - cy0,
                C_OVF if kind == 'ov' else C_GAPF, 'none', rx=0, sw=0)
    for (ta, tb) in P['overlap_frames']:        # 重叠虚框描边（跨两轨的列标记）
        lc.rect(x(ta), cy0 - 2, max(x(tb) - x(ta), 2.0), gy1 - cy0 + 4,
                'none', C_OVS, rx=3, sw=1.6, dash=True)

    # GPU 条（GPU 泳道块带；窗口右缘截断，不许画出面板；拍间断流琥珀块在条之间）
    for (ta, tb, tf, ts, label) in P['gpu_bars']:
        tb, ts = min(tb, P['t1']), min(ts, P['t1'])
        if ts > tf + 1e-9:                       # 前向 + 采样尾（浅绿，仅同步版可见宽度）
            lc.rect(x(ta), gb_t, x(ts) - x(ta), gb_b - gb_t, lc.C_GPU_S, 'none', rx=3, sw=0)
            lc.rect(x(tf), gb_t, max(x(ts) - x(tf), 1.5), gb_b - gb_t,
                    C_SAMP, lc.C_GPU_S, rx=3, sw=0.8)
        else:
            lc.rect(x(ta), gb_t, x(tb) - x(ta), gb_b - gb_t, lc.C_GPU_S, 'none', rx=3, sw=0)
        if label:
            lc.text((x(ta) + x(tb)) / 2, (gb_t + gb_b) / 2 + 3, label, 9.5, '#ffffff',
                    'middle', True, maxw=x(tb) - x(ta) - 8, tag=tag + ':gpu:' + label[:6])
    for (ta, tb) in P['gpu_gaps']:
        lc.rect(x(ta), gb_t, max(x(tb) - x(ta), 2.2), gb_b - gb_t,
                C_GAPF, C_GAPS, rx=1.5, sw=1.0)

    # CPU 块：(kind, ta, tb, minw)  kind∈ solid|hatch ；文本另发
    for (kind, ta, tb, minw) in P['cpu_blocks']:
        w = max(x(tb) - x(ta), minw)
        if kind == 'hatch':
            lc.rect(x(ta), blk_t, x(tb) - x(ta), blk_b - blk_t, 'url(#cpuwait)',
                    lc.C_ENG_S, rx=3, sw=1.3)
        else:
            lc.rect(x(ta), blk_t, w, blk_b - blk_t, lc.C_ENG_S, 'none', rx=2, sw=0)

    # 块内文本（在块之上）
    for (cx_t, line, fs, color, bold) in P['block_texts']:
        lc.text(x(cx_t[0]) + (x(cx_t[1]) - x(cx_t[0])) / 2, line[0], line[1], fs, color,
                'middle', bold, maxw=x(cx_t[1]) - x(cx_t[0]) - 6, tag=tag + ':bt:' + line[1][:6])

    # 泳道上沿标注（两行错层 + 引线；居中锚点向内钳制，不许压面板左右框边）
    for (tx, row, s, fs, color, bold, tick) in P['above_labels']:
        xx = x(tx)
        anchor = P.get('ab_anchor', 'middle')
        if anchor == 'middle':
            half = lc.tw(s, fs, bold) / 2 + 4
            xx = min(max(xx, fx0 + 8 + half), fx1 - 8 - half)
        lc.text(xx, P['y_rowA'] if row == 'A' else P['y_rowB'],
                s, fs, color, anchor, bold, maxw=P['ab_maxw'], tag=tag + ':ab:' + s[:8])
        if tick:
            leader(x(tick), (P['y_rowA'] if row == 'A' else P['y_rowB']) + 3, blk_t - 1)

    # 泳道间注（断流数值等，y 固定在 P['y_mid']）
    for (tx, s, anchor) in P['mid_labels']:
        lc.text(x(tx), P['y_mid'], s, 8.5, '#b45309', 'middle' if anchor == 'mid' else anchor,
                True, maxw=200, tag=tag + ':mid:' + s[:8])

    # GPU 条下沿时刻戳（两行错层）
    for (tx, s, row, anchor) in P['stamps']:
        lc.text(x(tx), y_st1 if row == 1 else y_st2, s, 8.5, lc.C_MUTE, anchor,
                maxw=300, tag=tag + ':st:' + s[:10])
    for (gx, gy) in P['dots']:
        dot(gx, gy, 3.0, C_GAPS)

    # 拍距标尺：延伸刻度线（上起 GPU 泳道底）+ 双端箭头 + 数值（拍号并入）
    for lx in P['launches']:
        lc.seg(x(lx), gy1 + 2, x(lx), y_dim + 4, lc.C_FAINT, 1.0)
    lc.text(PX0 + 10, y_cap, '拍距（GPU 起点到下拍 GPU 起点，ms）', 8.5, lc.C_MUTE,
            'start', tag=tag + ':dimcap')
    for (ta, tb, s) in P['dims']:
        dbl_arrow(x(ta) + 3, x(tb) - 3, y_dim, lc.C_MUTE)
        lc.text((x(ta) + x(tb)) / 2, y_val, s, 10.5, lc.C_TXT, 'middle', True,
                maxw=x(tb) - x(ta) - 10, tag=tag + ':dim:' + s[:6])


# ---------------- 数据 → 面板参数 ----------------
def nxt(rows, i, extra):
    """第 i 拍的下一拍（越界时用窗口外的第 4 拍行）。"""
    return rows[i + 1] if i + 1 < len(rows) else extra


def cpu_span(r, seg4_is_wait=False):
    """一拍 CPU 各段绝对时刻：①②③④(④b 或 ④发起+等待)⑤。"""
    t = r['cpu_beat_start_ms']
    s1 = t + r['seg1_schedule_ms']
    s2 = s1 + r['seg2_launch_ms']
    s3 = s2 + r['seg3_bitmask_ms']
    s4 = s3 + r['seg4_sample_ms']
    if seg4_is_wait:
        s4w = s4 + r['④future_wait_ms']
        return {'t0': t, 's1': s1, 's2': s2, 's3': s3, 's4': s4, 's4w': s4w,
                's5': s4w + r['seg5_update_ms']}
    return {'t0': t, 's1': s1, 's2': s2, 's3': s3,
            's4b': s4 + r['seg4_sample_ms'] if False else s3 + r['seg4_sample_ms'],
            's5': s3 + r['seg4_sample_ms'] + r['seg5_update_ms']}


# ================= 上半：同步版 step() =================
S_T0 = SYNC[0]['cpu_beat_start_ms']                       # 171.303
S_T1 = SYNC_NEXT['gpu_launch_start_ms'] + 0.8             # 窗口含第 4 拍起点后少许
SC = [cpu_span(r) for r in SYNC]

syncP = {
    'frame': (190.0, PX0 + (S_T1 - S_T0) * K + 14, 112.0, 384.0),
    't0': S_T0, 't1': S_T1, 'tick0': 175.0,
    'y_hdr': 96, 'y_rul': 132, 'y_rowA': 150, 'y_rowB': 164,
    'cpu_lane': (170.0, 234.0), 'gpu_lane': (254.0, 318.0), 'blk': (180.0, 224.0),
    'gblk': (264.0, 308.0),
    'y_mid': 246, 'y_st1': 332, 'y_st2': 346, 'y_cap': 356, 'y_dim': 360, 'y_val': 378,
    'header': '上｜同步版 step()：发射 → 干等自家前向 → 收尾记账，再发射——GPU 每拍断流一次',
    'peer': '下',
    'underlays': [(SC[i]['s5'] - 0.004, nxt(SYNC, i, SYNC_NEXT)['gpu_launch_start_ms'], 'gap')
                  for i in range(3)],
    'overlap_frames': [],
    'gpu_bars': [(r['gpu_launch_start_ms'], r['gpu_samp_done_ms'], r['gpu_fwd_done_ms'],
                  r['gpu_samp_done_ms'],
                  '本拍前向 kernel' if i == 0 else None)
                 for i, r in enumerate(SYNC)]
                + [(SYNC_NEXT['gpu_launch_start_ms'], S_T1, SYNC_NEXT['gpu_fwd_done_ms'],
                    SYNC_NEXT['gpu_samp_done_ms'], None)],
    'gpu_gaps': [(SYNC[i]['gpu_samp_done_ms'], nxt(SYNC, i, SYNC_NEXT)['gpu_launch_start_ms'])
                 for i in range(3)],
    'cpu_blocks': sum([[
        ('solid', c['t0'], c['s1'], 2.5),                       # ①
        ('solid', c['s1'], c['s2'], 3.0),                       # ②
        ('hatch', c['s3'], c['s5'], 0),                         # ④b 整块（斜纹）
        ('solid', r['gpu_fwd_done_ms'], c['s5'], 2.0),          # ④b 尾段实色（采样+同步 D2H）
        ('solid', c['s5'] - r['seg5_update_ms'], c['s5'], 2.5),  # ⑤
    ] for c, r in zip(SC, SYNC)], []),
    'block_texts': [
        ((SC[0]['s3'], SC[0]['s5']), (196, '④b 收货（阻塞）23.442ms'), 9.5, lc.C_TXT, True),
        ((SC[0]['s3'], SYNC[0]['gpu_fwd_done_ms']), (212, '斜纹＝干等本拍前向尾程 ≈22.7ms（CPU 线程被钉死）'), 8.5, '#7c2d12', False),
        ((SC[1]['s3'], SC[1]['s5']), (205, '④b 22.998'), 9, lc.C_TXT, True),
        ((SC[2]['s3'], SC[2]['s5']), (205, '④b 22.342'), 9, lc.C_TXT, True),
        ((SC[0]['s1'], SC[0]['s2']), (205, '②'), 9, '#ffffff', True),
    ],
    'above_labels': [
        (SC[0]['t0'] + SYNC[0]['seg1_schedule_ms'] / 2, 'A', '① 0.086', 8.5, lc.C_TXT, True, SC[0]['t0'] + 0.04),
        ((SC[0]['s1'] + SC[0]['s2']) / 2, 'B', '② 发射 1.776', 8.5, lc.C_TXT, False, (SC[0]['s1'] + SC[0]['s2']) / 2),
        ((SC[0]['s5'] - SYNC[0]['seg5_update_ms'] / 2), 'B', '⑤ 0.043', 8.5, lc.C_TXT, True, SC[0]['s5'] - SYNC[0]['seg5_update_ms'] / 2),
    ],
    'ab_anchor': 'middle', 'ab_maxw': 220,
    'mid_labels': [((SYNC[i]['gpu_samp_done_ms'] + nxt(SYNC, i, SYNC_NEXT)['gpu_launch_start_ms']) / 2,
                    f"{SYNC[i]['gpu_idle_gap_ms']:.3f}", 'mid') for i in range(3)],
    'stamps': [
        (SYNC[0]['gpu_launch_start_ms'], '171.452', 1, 'start'),
        (SYNC[0]['gpu_fwd_done_ms'], '195.913 前向完', 2, 'end'),
        (SYNC[0]['gpu_samp_done_ms'], '196.587 采样+D2H 完', 2, 'start'),
    ],
    'dots': [],
    'launches': [r['gpu_launch_start_ms'] for r in SYNC] + [SYNC_NEXT['gpu_launch_start_ms']],
    'dims': [(SYNC[i]['gpu_launch_start_ms'], nxt(SYNC, i, SYNC_NEXT)['gpu_launch_start_ms'],
              f"拍{i + 1} {SYNC[i]['beat_period_gpu_ms']:g}")
             for i in range(3)],
}
panel(syncP, 'sync')

# 同步面板右侧注解框（琥珀断流＝GPU 等人）
COX0, COX1, COY0, COY1 = 1746.0, 1920.0, 230.0, 332.0
lc.rect(COX0, COY0, COX1 - COX0, COY1 - COY0, '#fffbeb', C_GAPS, rx=8, sw=1.2)
lc.text(COX0 + 12, COY0 + 20, '琥珀＝GPU 断流', 10, '#b45309', 'start', True,
        maxw=COX1 - COX0 - 24, tag='co:t')
for i, ln in enumerate(['这段 GPU 在等 CPU：',
                        '④b 尾＋⑤ 记账＋拍界＋',
                        '下一拍 ①② 发射头——',
                        '全串行压在关键路径上，',
                        '每拍来一次']):
    lc.text(COX0 + 12, COY0 + 38 + i * 16, ln, 8.8, lc.C_TXT, 'start',
            maxw=COX1 - COX0 - 20, tag='co:%d' % i)
_gap3_x = PX0 + (SYNC[2]['gpu_samp_done_ms'] - S_T0) * K
lc.seg(COX0 - 2, (COY0 + COY1) / 2, _gap3_x + 5, (syncP['cpu_lane'][0] + syncP['gpu_lane'][1]) / 2 + 20,
       C_GAPS, 1.1, dash=True)

# ================= 中央对比带 =================
lc.rect(640, 408, 260, 30, lc.C_ENG_F, lc.C_ENG_S, rx=15, sw=1.6)
lc.text(770, 427, '同步版稳态拍距 24.861 ms', 12, lc.C_ENG_S, 'middle', True,
        maxw=246, tag='mid:l')
lc.rect(1116, 408, 260, 30, lc.C_GPU_F, lc.C_GPU_S, rx=15, sw=1.6)
lc.text(1246, 427, '异步版稳态拍距 21.155 ms', 12, lc.C_GPU_S, 'middle', True,
        maxw=246, tag='mid:r')
lc.seg(904, 423, 1110, 423, lc.C_MUTE, 2.5, 'std')
lc.text(1007, 410, '吞吐 +17.5%', 13, lc.C_TXT, 'middle', True, maxw=180, tag='mid:pct')
lc.text(1007, 441, '拍距中位', 9.5, lc.C_MUTE, 'middle', tag='mid:cap')
lc.text(1020, 466, '拍墙钟 24.829 → 21.241 ms（+16.9%）· 单序列解码 40.3 → 47.1 tok/s · '
        'GPU 断流均值 0.277 → 0.001 ms · 同一 27B 场景 · 批 {req-A:1, req-B:1} · 稳态 35 拍中位',
        10, lc.C_MUTE, 'middle', maxw=1300, tag='mid:sub')

# ================= 下半：异步版 step_with_batch_queue() =================
A_T0 = PREV['gpu_launch_start_ms']                        # 145.767（上一拍 GPU 起点＝窗口左沿）
A_T1 = ASYN_NEXT['gpu_launch_start_ms'] + 0.9             # 窗口含第 4 拍起点后少许
AC = [cpu_span(r, seg4_is_wait=True) for r in ASYN]

asyncP = {
    'frame': (190.0, PX0 + (A_T1 - A_T0) * K + 14, 520.0, 806.0),
    't0': A_T0, 't1': A_T1, 'tick0': 150.0,
    'y_hdr': 504, 'y_rul': 540, 'y_rowA': 560, 'y_rowB': 574,
    'cpu_lane': (580.0, 644.0), 'gpu_lane': (664.0, 728.0), 'blk': (590.0, 634.0),
    'gblk': (674.0, 718.0),
    'y_mid': 656, 'y_st1': 742, 'y_st2': 756, 'y_cap': 766, 'y_dim': 770, 'y_val': 788,
    'header': '下｜异步版 step_with_batch_queue()：先发射本拍 → 收上一拍的货——'
              'CPU 等待被 GPU 执行掩护、GPU 发射被 CPU 等待吸收',
    'peer': '上',
    'underlays': [(AC[i]['t0'], AC[i]['s2'], 'ov') for i in range(3)],
    'overlap_frames': [(AC[0]['t0'], AC[0]['s2'])],
    'gpu_bars': [(PREV['gpu_launch_start_ms'], PREV['gpu_samp_done_ms'],
                  PREV['gpu_fwd_done_ms'], PREV['gpu_samp_done_ms'], '上一拍前向（窗口左起已在执行）')]
                + [(r['gpu_launch_start_ms'], r['gpu_samp_done_ms'], 0, 0,
                    '本拍前向' if i == 0 else None)
                   for i, r in enumerate(ASYN)]
                + [(ASYN_NEXT['gpu_launch_start_ms'], A_T1, 0, 0, None)],
    'gpu_gaps': [],
    'cpu_blocks': sum([[
        ('solid', c['t0'], c['s2'], 3.0),                       # ①+② 发射窗（一体橙）
        ('solid', c['s3'], c['s4'], 3.0),                       # ④ 发起（non_block）
        ('hatch', c['s4'], c['s4w'], 0),                        # ④ 等待（等上拍 D2H 事件）
        ('solid', c['s4w'], c['s5'], 2.5),                      # ⑤ 记上一拍的账
    ] for c in AC], []),
    'block_texts': [
        ((AC[0]['t0'], AC[0]['s2']), (617, '①+②'), 8.5, '#ffffff', True),
        ((AC[0]['s4'], AC[0]['s4w']), (606, '④ 等待 18.552ms'), 9.5, lc.C_TXT, True),
        ((AC[0]['s4'], AC[0]['s4w']), (622, '等上一拍的 D2H 事件——期间 GPU 在算本拍前向'), 8.5, '#7c2d12', False),
        ((AC[1]['s4'], AC[1]['s4w']), (617, '④ 等待 18.469'), 9, lc.C_TXT, True),
        ((AC[2]['s4'], AC[2]['s4w']), (617, '④ 等待 18.461'), 9, lc.C_TXT, True),
    ],
    'above_labels': [],
    'ab_anchor': 'middle', 'ab_maxw': 220,
    'mid_labels': [],
    'stamps': [
        (PREV['gpu_samp_done_ms'], '166.924 采样完 → 166.925 本拍前向起', 1, 'start'),
        (ASYN[0]['gpu_samp_done_ms'], '188.021', 2, 'middle'),
        (ASYN[1]['gpu_samp_done_ms'], '209.154', 2, 'middle'),
        (ASYN[2]['gpu_samp_done_ms'], '230.25', 2, 'end'),
    ],
    'dots': [(PX0 + (t - A_T0) * K, 730.0) for t in
             [PREV['gpu_samp_done_ms']] + [r['gpu_launch_start_ms'] for r in ASYN]
             + [ASYN_NEXT['gpu_launch_start_ms']]],
    'launches': [r['gpu_launch_start_ms'] for r in ASYN] + [ASYN_NEXT['gpu_launch_start_ms']],
    'dims': [(ASYN[i]['gpu_launch_start_ms'], nxt(ASYN, i, ASYN_NEXT)['gpu_launch_start_ms'],
              f"拍{i + 1} {ASYN[i]['beat_period_gpu_ms']:g}")
             for i in range(3)],
}
panel(asyncP, 'async')

# 异步面板泳道上沿标注（发射窗标签从括号右侧起排，避免压框）
_X = lambda t: PX0 + (t - A_T0) * K
lc.text(_X(AC[0]['s2']) + 12, asyncP['y_rowA'],
        '①+② 发射窗 2.082ms——全程垫在上拍 GPU 执行之下（稳态均值 2.02 · 同步版恒 0.0）',
        9, C_OVS, 'start', True, maxw=560, tag='async:ab:win')
lc.text(_X((AC[0]['s3'] + AC[0]['s4']) / 2), asyncP['y_rowB'], '④ 发起 0.315', 8.5,
        lc.C_TXT, 'middle', maxw=140, tag='async:ab:launch')
leader(_X((AC[0]['s3'] + AC[0]['s4']) / 2), asyncP['y_rowB'] + 3, 588)
lc.text(_X((AC[0]['s4w'] + AC[0]['s5']) / 2), asyncP['y_rowB'], '⑤ 0.057 记上一拍的账', 8.5,
        lc.C_TXT, 'middle', maxw=180, tag='async:ab:upd')
leader(_X((AC[0]['s4w'] + AC[0]['s5']) / 2), asyncP['y_rowB'] + 3, 588)
# 边界缝隙一行（稳态三拍逐拍值；紧跟 188.021 戳之后起排）
lc.text(_X(ASYN[0]['gpu_samp_done_ms']) + 46, asyncP['y_st2'],
        '边界缝隙 0.000 / 0.000 / 0.001 ms——背靠背，GPU 从不等 CPU', 8.5,
        '#b45309', 'start', True, maxw=430, tag='async:gapline')
# 稳态 35 拍小结（面板底部）
lc.text(PX0 + 10, 798, '稳态 35 拍：④ 等待均值 18.832ms（其中 D2H 事件等待 18.793ms）· '
        '拍尾本拍前向仍在执行 35/35 拍——CPU 的等待被 GPU 执行掩护', 9, lc.C_TXT, 'start',
        maxw=900, tag='async:note35')

# ================= 图例 =================
LEG_Y = 828
lx = MX


def leg(fill, stroke, name, dash=False, pattern=False):
    global lx
    f = 'url(#cpuwait)' if pattern else fill
    lc.rect(lx, LEG_Y - 9, 20, 12, f, stroke, rx=3, sw=1.2, dash=dash)
    lc.text(lx + 26, LEG_Y + 1, name, 8.5, lc.C_TXT, 'start', maxw=300, tag='leg:' + name[:8])
    lx += 26 + lc.tw(name, 8.5) + 22


leg(lc.C_ENG_S, 'none', 'CPU 执行段（①②④⑤ 发起/记账）')
leg('none', lc.C_ENG_S, 'CPU 阻塞等待（斜纹）', pattern=True)
leg(lc.C_GPU_S, 'none', 'GPU 前向 kernel')
leg(C_SAMP, lc.C_GPU_S, '采样+同步 D2H 尾')
leg(C_GAPF, C_GAPS, 'GPU 断流（琥珀）')
leg(C_OVF, C_OVS, '重叠＝CPU 活动垫在 GPU 执行之下', dash=True)

# ================= 注脚框（两个极端形态） =================
NF_Y0, NF_Y1 = 846.0, 924.0
lc.rect(190, NF_Y0, 1734 - 190 + 0, NF_Y1 - NF_Y0, '#ffffff', lc.C_FAINT, rx=8, sw=1.1, dash=True)
lc.text(204, NF_Y0 + 18, '两个极端形态（同源实测）', 10, lc.C_TXT, 'start', True,
        maxw=300, tag='nf:t')
lc.text(204, NF_Y0 + 36, '· prefill 后首拍：异步 2.912ms 纯填队列早退（④ 等待仅 0.019ms——上拍的货已备好）'
        ' vs 同步同场景 25.447ms', 9, lc.C_TXT, 'start', maxw=1680, tag='nf:1')
lc.text(204, NF_Y0 + 54, '· tiny 反例（16 层 launch-bound，GPU 段亚毫秒、远小于发射）：异步双 decode 拍 '
        '3.853ms 反而慢于同步 1.860ms——重叠的前提＝GPU 执行远大于 CPU 发射',
        9, lc.C_TXT, 'start', maxw=1680, tag='nf:2')
lc.text(204, NF_Y0 + 72, '· 批队列深水（deferred sampling、流水线消泡）见后续章节', 9,
        lc.C_MUTE, 'start', maxw=1680, tag='nf:3')

# ================= 页脚 =================
lc.text(MX, 944, '实测口径：Qwen3.8-27B-FP8 真机（NVIDIA Blackwell）· 被测代码＝钉版 v0.27.1 源树 · '
        '同进程先 sync 后 async（编译缓存共享）· 循环内零 synchronize · ③ bitmask 两版均 0.002ms'
        '（本比例尺下不可见）· ①⑤ 窄段按最小可见宽度绘制（数值以标注为准）· '
        '上下两面板同一时间比例尺（10ms＝200px）· 取自 35 拍稳态窗（同步第 11-13 拍 / 异步第 12-14 拍）',
        8.5, lc.C_FAINT, 'start', maxw=1880, tag='foot')

# ================= 装配输出 =================
HATCH = ('<pattern id="cpuwait" width="7" height="7" patternTransform="rotate(45)" '
         'patternUnits="userSpaceOnUse">'
         f'<rect width="7" height="7" fill="{C_HATCHBG}"/>'
         f'<line x1="0" y1="0" x2="0" y2="7" stroke="{C_HATCHLN}" stroke-width="2"/></pattern>')
DIM = ('<marker id="dim" viewBox="0 0 10 6" refX="9.5" refY="3" markerWidth="6" '
       'markerHeight="4.2" orient="auto-start-reverse">'
       f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_MUTE}"/></marker>')
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>',
       '<defs>' + HATCH + DIM + '</defs>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch09-fig-sync-vs-async-overlap.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
