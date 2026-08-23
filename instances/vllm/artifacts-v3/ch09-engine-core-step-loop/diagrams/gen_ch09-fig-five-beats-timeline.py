#!/usr/bin/env python3
"""ch09 机制图 1 · 一拍五段的双轨时间轴（figure_spec ch09-fig-five-beats-timeline，模板 swimlane）

放大自 L0 的循环框（loop_box）——即本章 L2 章图 center 拍片行（①-⑤ 一拍五段，core.py:L584-L614）
的时间维展开：L2 画五拍的结构顺序与站号，本图把拍 1 放到 CPU/GPU 双轨时间轴上，
回答「每段何时发生、各占多久、谁跟谁重叠」。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：
图右上角指北小签。

claim：一拍五段在 CPU/GPU 双轨时间轴上严格有序（①schedule→②发起前向→③掩码→④采样→⑤记账），
全部 CPU 段合计 ≈0.35ms、不足建模前向 6.041ms 的 6%——五拍的顺序本身就是
『不许任何 CPU 活让 GPU 干等』的性能设计。

数字全部取自 figure_spec.numbers（beat 1 实测：0.003/0.057/6.098/6.125/6.32，全程 6.387ms；
②→③ 间隔 6.041ms；五拍批形状 {A:3}→{A:1,B:4}→{A:1,B:1}→{}flush→空转守卫；
non_block 4 次全 True；beat 3 双 LENGTH、beat 4 0-token flush 不采样）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 746
MX, BXR = 60, 1440
SLATE = '#e2e8f0'   # 与 ch03 机制图同款局部网格灰（slate 家族）

# ---------------- 标题区 ----------------
lc.text(MX, 34, '一拍五段的双轨时间轴——顺序本身就是『不许任何 CPU 活让 GPU 干等』的性能设计',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '① schedule → ② 发起前向 → ③ 语法掩码 → ④ 采样 → ⑤ 记账：全部 CPU 段合计 ≈0.35ms，'
        '不足建模前向 6.041ms 的 6%（拍 1 实测全程 6.387ms）',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片行 ①-⑤ 一拍五段 · L0：循环框'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 时间轴几何（拍 1 实测，真实线性比例） ----------------
TPX0, TPX1 = 260.0, 1340.0
WALL_MS = 6.387
PPM = (TPX1 - TPX0) / WALL_MS           # px per ms


def tx(ms):
    return TPX0 + ms * PPM


# 拍 1 事件时刻（实测）
T_HAS, T_SCHED, T_EXEC, T_BIT, T_SAMP, T_UPD = 0.0, 0.003, 0.057, 6.098, 6.125, 6.32

# ---------------- 双轨泳道 ----------------
CPU_Y0, CPU_H = 118, 96
CPU_Y1 = CPU_Y0 + CPU_H                 # 214
GPU_Y0, GPU_H = 238, 58
GPU_Y1 = GPU_Y0 + GPU_H                 # 296

# 泳道头（左）
lc.rect(MX, CPU_Y0 + CPU_H / 2 - 15, 178, 30, '#ffffff', lc.C_ENG_S, rx=8, sw=1.5)
lc.text(MX + 89, CPU_Y0 + CPU_H / 2 + 4, 'CPU 忙循环（单线程）', 10, lc.C_ENG_S, 'middle', True,
        maxw=170, tag='lane:cpu')
lc.rect(MX, GPU_Y0 + GPU_H / 2 - 15, 178, 30, '#ffffff', lc.C_GPU_S, rx=8, sw=1.5)
lc.text(MX + 89, GPU_Y0 + GPU_H / 2 + 4, 'GPU 执行臂', 10, lc.C_GPU_S, 'middle', True,
        maxw=170, tag='lane:gpu')

# 泳道底板
lc.rect(250, CPU_Y0, 1110, CPU_H, lc.C_ENG_F, lc.C_ENG_S, rx=6, sw=1.2)
lc.rect(250, GPU_Y0, 1110, GPU_H, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.2)

# ---------------- CPU 轨：五段块（真实比例窄条） ----------------
BLK_Y0, BLK_Y1 = CPU_Y0 + 56, CPU_Y1 - 6      # 174..208
MINW = 5.0
blocks = [
    ('①', T_SCHED, T_EXEC),                     # ① schedule 0.003→0.057
    ('②', T_EXEC, T_EXEC + MINW / PPM),         # ② 发起（瞬时：给最小可见宽）
    ('③', T_BIT, T_SAMP),                       # ③ 掩码 6.098→6.125
    ('④', T_SAMP, T_UPD),                       # ④ 采样 6.125→6.32
    ('⑤', T_UPD, WALL_MS),                      # ⑤ 记账 6.32→6.387
]
blk_x = []
for lab, t0, t1 in blocks:
    x0, x1 = tx(t0), max(tx(t0) + MINW, tx(t1))
    blk_x.append((x0, x1, lab))
    lc.rect(x0, BLK_Y0, x1 - x0, BLK_Y1 - BLK_Y0, lc.C_BEAT_F, lc.C_BEAT_S, rx=2, sw=1.3)

# 拍号徽标（两行错开，避免左右两簇内部相撞；下方细线牵到各自块顶）
ROW_A, ROW_B = CPU_Y0 + 22, CPU_Y0 + 42        # 140 / 160
badge_rows = [ROW_A, ROW_B, ROW_A, ROW_B, ROW_A]
for (x0, x1, lab), ry in zip(blk_x, badge_rows):
    cx = (x0 + x1) / 2
    if lab == '①':
        cx = x0 + 6.5        # 微调：让开左侧回环竖线（x=252）
    if lab == '⑤':
        cx = x1 - 6.5        # 微调：让开右侧回环竖线（x=1354）
    lc.rect(cx - 8.5, ry - 8.5, 17, 17, lc.C_BADGE_F, lc.C_ENG_S, rx=8.5, sw=1.1)
    lc.text(cx, ry + 3, lab, 8.5, lc.C_ENG_S, 'middle', True, tag='bdg' + lab)
    lc.seg(cx, ry + 9, cx, BLK_Y0, lc.C_MUTE, 1.0)

# host 建模等待线（② 末 → ③ 始：CPU 在 execute_model 内等建模的一步前向）
WAIT_Y = (BLK_Y0 + BLK_Y1) / 2
lc.seg(blk_x[1][1] + 1, WAIT_Y, blk_x[2][0] - 1, WAIT_Y, lc.C_FAINT, 1.2, dash=True)
lc.text((blk_x[1][1] + blk_x[2][0]) / 2, BLK_Y1 - 4,
        '（host 建模：CPU 在 execute_model 内等前向完成；真实引擎 ② 发起即返回、此刻已腾出手算 ③）',
        8.2, lc.C_MUTE, 'middle', maxw=760, tag='wait:lbl')

# 「下一拍」回环：⑤ 右缘 → 上方横轨 → ① 左缘
RAIL_Y = 96
x5r = blk_x[4][1]
x1l = blk_x[0][0]
lc.parrow([(x5r, WAIT_Y), (1354, WAIT_Y), (1354, RAIL_Y), (252, RAIL_Y), (252, WAIT_Y), (x1l, WAIT_Y)],
          lc.C_MUTE, 1.4, 'std')
lc.text((252 + 1354) / 2, 110, '下一拍（step() 再走一轮 ①→⑤）', 9, lc.C_MUTE, 'middle',
        maxw=400, tag='rail:lbl')

# ---------------- GPU 轨：一步前向长条 ----------------
BAR_Y0, BAR_Y1 = GPU_Y0 + 12, GPU_Y1 - 12      # 250..284
lc.rect(tx(T_EXEC), BAR_Y0, tx(T_BIT) - tx(T_EXEC), BAR_Y1 - BAR_Y0, lc.C_GPU_S, lc.C_GPU_S, rx=3, sw=0)
lc.text((tx(T_EXEC) + tx(T_BIT)) / 2, (BAR_Y0 + BAR_Y1) / 2 + 3.5,
        '前向 6.041ms（建模的一步前向；真实几十 ms，2048-8192 token 预算）',
        10, '#ffffff', 'middle', True, maxw=980, tag='gpu:bar')

# 跨轨箭头：② 发起（下行蓝）/ 结果回手（上行橙）——沿 L0 图例语义
lc.seg(tx(T_EXEC), CPU_Y1 + 2, tx(T_EXEC), BAR_Y0 - 2, lc.C_API_S, 1.8, 'dn')
lc.text(tx(T_EXEC) + 8, GPU_Y0 - 9, '② 发起前向（non_block=True）', 8.5, lc.C_API_S, 'start',
        maxw=210, tag='launch:lbl')
lc.seg(tx(T_BIT), BAR_Y0 - 2, tx(T_BIT), CPU_Y1 + 2, lc.C_ENG_S, 1.8, 'up')
lc.text(tx(T_BIT) - 8, GPU_Y0 - 9, '④ 只等 D2H 拷贝事件（future.result()，不等计算）', 8.5,
        lc.C_ENG_S, 'end', maxw=240, tag='ret:lbl')

# ---------------- 时间轴（ms 刻度，拍 1 实测） ----------------
AX_Y = 312
lc.seg(250, AX_Y, 1360, AX_Y, lc.C_MUTE, 1.2)
ticks = [(0.003, '0.003', 0), (0.057, '0.057', 1), (6.098, '6.098', 0),
         (6.125, '6.125', 1), (6.32, '6.32', 0), (6.387, '6.387', 1)]
for t, s, row in ticks:
    x = tx(t)
    lc.seg(x, AX_Y - 6, x, AX_Y + 6, lc.C_MUTE, 1.2)
    lc.text(x, AX_Y + 18 + row * 16, s, 8.5, lc.C_MUTE, 'middle', tag='tk' + s)
lc.text(64, AX_Y + 18, '拍 1 实测（ms）', 8.5, lc.C_MUTE, 'start', maxw=120, tag='ax:unit')

# ---------------- 逐段账 ----------------
lc.text(MX, 372, '逐段账——拍 1 实测时刻（方法名与顺序逐字自 core.py:L584-L614）', 10, lc.C_TXT,
        'start', True, maxw=700, tag='led:head')
LEDGER = [
    ('①', 'schedule()',
     '（scheduler.py:L439）@0.003→0.057ms（≈0.05ms）——GPU 启动前把所有可能慢、可能触发抢占的决策做完'),
    ('②', 'execute_model(scheduler_output, non_block=True)',
     '（core.py:L596）@0.057ms 发起（host 建模阻塞至前向返回；真实引擎发起即返回——见虚线注；本例 4 次发起全 True）；②→③ 间隔 6.041ms ＝ 建模的一步前向（真实几十 ms，2048-8192 token 预算）'),
    ('③', 'get_grammar_bitmask(scheduler_output)',
     '（core.py:L597）@6.098ms——真实引擎藏进前向窗口算（② 发起就算）；本 host 建模排在返回后，时序约束相同（②发起 ＜ ③ ＜ ④ 应用 ＜ 采样）'),
    ('④', 'future.result() → sample_tokens(grammar_output)',
     '（core.py:L602-L604）@6.125ms（③→④ 0.029ms）——只等 D2H 拷贝事件、不等计算；先应用掩码、再采样'),
    ('⑤', '_process_aborts_queue() + update_from_output(...)',
     '（core.py:L606-L609）@6.32ms→拍末 6.387ms（≈0.07ms）——先批量落地执行期 abort，再记账/判停/回收/按 client 分桶出件'),
]
LED_Y0, LED_STEP = 394, 25
for i, (num, name, fact) in enumerate(LEDGER):
    y = LED_Y0 + i * LED_STEP
    lc.rect(66, y - 11, 17, 17, lc.C_BADGE_F, lc.C_ENG_S, rx=8.5, sw=1.1)
    lc.text(74.5, y + 3, num, 8.5, lc.C_ENG_S, 'middle', True, tag='led' + num)
    lc.text(94, y + 3, name, 9, lc.C_TXT, 'start', True, maxw=460, tag='ln' + num)
    nx = 94 + lc.tw(name, 9, True) + 10
    lc.text(nx, y + 3, fact, 8.7, '#334155', 'start', maxw=BXR - nx, tag='lf' + num)

# ---------------- 结论横幅 ----------------
BN_Y = 516
lc.rect(MX, BN_Y, 1380, 34, lc.C_BEAT_F, lc.C_BEAT_S, rx=7, sw=1.4)
lc.text(MX + 690, BN_Y + 21.5,
        '全部 CPU 段合计 ≈0.35ms，不足建模前向 6.041ms 的 6%——GPU 越快，五拍排布越值钱',
        10.5, lc.C_BEAT_T, 'middle', True, maxw=1360, tag='banner')

# ---------------- 同场五拍全景条 ----------------
lc.text(MX, 572, '同场五拍全景——req-A 的 3 token 一生 + 拍 1 后迟到的 req-B（每格 = 一次 step() 调用）',
        10, lc.C_TXT, 'start', True, maxw=900, tag='strip:head')
CHIPS = [
    ('拍 1 · prefill', True, ['批 {req-A: 3}（prompt 一拍收官）', 'A 出 [7]；client0 收 1 条',
                              '本图上半 = 此拍的时间维展开']),
    ('拍 2 · 混相批', True, ['批 {req-A: 1, req-B: 4}', 'A 出 [8]、B 出 [6]',
                             '迟到的 B 全量 prefill 4 token']),
    ('拍 3 · 双双到顶', True, ['批 {req-A: 1, req-B: 1}', 'A 出 [9]、B 出 [6]',
                               'A LENGTH(3/3)、B LENGTH(2/2) 同拍释放']),
    ('拍 4 · flush（0-token 批）', False, ['finished_ids={req-A, req-B} 随批下发',
                                           'worker 清缓存；不前向、不采样', '（executed=False）']),
    ('拍 5 · 空转守卫', False, ['has_requests()==False', '先于 ① 返回（{}, False）',
                                'executor 零调用']),
]
CH_W, CH_GAP, CH_Y, CH_H = 262, 12, 586, 90
for i, (title, hot, lines) in enumerate(CHIPS):
    x = MX + i * (CH_W + CH_GAP)
    if hot:
        lc.rect(x, CH_Y, CH_W, CH_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=6, sw=1.3)
    else:
        lc.rect(x, CH_Y, CH_W, CH_H, '#ffffff', lc.C_MUTE, rx=6, sw=1.2, dash=True)
    lc.text(x + 12, CH_Y + 20, title, 9.5, lc.C_TXT, 'start', True, maxw=CH_W - 24,
            tag='ch' + str(i) + 't')
    for j, ln in enumerate(lines):
        lc.text(x + 12, CH_Y + 38 + j * 17, ln, 8.2, '#334155', 'start', maxw=CH_W - 22,
                tag='ch' + str(i) + 'l' + str(j))
    if i < len(CHIPS) - 1:
        lc.seg(x + CH_W, CH_Y + CH_H / 2, x + CH_W + CH_GAP - 2, CH_Y + CH_H / 2,
               lc.C_MUTE, 1.5, 'std')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = 700
lx = MX
items = [
    ('eng', 'CPU 忙循环（EngineCore.step 单线程）'),
    ('gpu', 'GPU 前向（建模的一步前向）'),
    ('beat', '一拍的拍段 ①-⑤'),
    ('dash', 'CPU 在 ② 内等前向（host 建模）'),
    ('off', 'executed=False 的拍（不前向）'),
]
for kind, name in items:
    if kind == 'eng':
        lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_ENG_F, lc.C_ENG_S, rx=3, sw=1.4)
    elif kind == 'gpu':
        lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_GPU_S, lc.C_GPU_S, rx=3, sw=0)
    elif kind == 'beat':
        lc.rect(lx, LEG_Y - 9, 20, 12, lc.C_BEAT_F, lc.C_BEAT_S, rx=3, sw=1.2)
    elif kind == 'dash':
        lc.seg(lx, LEG_Y - 3, lx + 24, LEG_Y - 3, lc.C_FAINT, 1.2, dash=True)
        lx -= 4
    else:
        lc.rect(lx, LEG_Y - 9, 20, 12, '#ffffff', lc.C_MUTE, rx=3, sw=1.1, dash=True)
    lc.text(lx + 26, LEG_Y + 1, name, 8.5, lc.C_TXT, 'start', maxw=300, tag='leg' + kind)
    lx += 26 + lc.tw(name, 8.5) + 18

lc.text(MX, 728, '逐字锚 vllm/v1/engine/core.py:L584-L614（step：①→⑤ 顺序与空转早退）· '
        'vllm/v1/core/sched/scheduler.py:L439（schedule）· 时刻与五拍批账取自配套精简版 host 实测'
        '（temperature=0 贪婪 argmax，vocab=16）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch09-fig-five-beats-timeline.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
