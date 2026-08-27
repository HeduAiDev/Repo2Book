#!/usr/bin/env python3
"""ch12 机制图 8 · 队列元素的三元组抽屉柜（figure_spec ch12-fig-queue-triple，模板 layout）

放大自 L0 循环框（loop_box）中 EngineCore 与执行臂之间的队列位——即本章 L2 章图
north 的 batch_queue 组件框（appendleft · pop）的容器内景展开：三层抽屉的三元组
结构与 FIFO 方向。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：队列元素是三元组 (采样 future, SchedulerOutput, exec_future) 的抽屉柜——
appendleft 进 / pop 出配成 FIFO，采样 future 出 None 时靠第三层抽屉里的
exec_future 重抛真异常。

数字全部取自 figure_spec.numbers（三元组层数 3：future / scheduler_output /
exec_future；FIFO 实测 pop 顺序 t7 → t8、output_token_ids [7] → [7,8]；
None ⇒ 重抛 RuntimeError('real worker failure')；容量 deque(maxlen=2)、
稳态每拍 appendleft 一次 + pop 一次）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 866
MX, BXR = 60, 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '队列里躺的是三元组：一次 pop 的三件事——收采样、对账单、验故障',
        16.5, lc.C_TXT, 'start', True, maxw=1010, tag='title')
lc.text(MX, 58, '采样 future 只回答『这一批采出了什么』；它返回 None 时真答案在 exec_future——'
        '绝不吞成 unexpected error', 10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 batch_queue 组件框 · L0：循环框'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左：deque 容器 + 三元组抽屉柜 ----------------
Q_Y, Q_H = 100, 400
Q_X, Q_W = 150, 590
lc.rect(Q_X, Q_Y, Q_W, Q_H, '#ffffff', lc.C_ENG_S, rx=8, sw=1.8)
lc.text(Q_X + 14, Q_Y + 21, 'batch_queue = deque(maxlen=2)', 10.5, lc.C_ENG_S, 'start', True,
        maxw=Q_W - 30, tag='q:t')
lc.text(Q_X + 14, Q_Y + 38, 'appendleft 进 index0（左端）· pop 出末尾（右端，最老批）', 8.5,
        lc.C_MUTE, 'start', maxw=Q_W - 28, tag='q:sub')
lc.text(Q_X + Q_W / 2, Q_Y + 52, '队列元素内景：一个批的三元组抽屉柜（稳态柜里并排两个）',
        8.4, lc.C_MUTE, 'middle', maxw=Q_W - 30, tag='q:cab')

TRIPLE = [
    ('第一层抽屉 · 采样 future', '这一批采出了什么', 'pop 侧 future.result() 收的就是它',
     lc.C_GPU_S, lc.C_GPU_F),
    ('第二层抽屉 · SchedulerOutput', '这批当初排了谁', '下半段 update_from_output 的对账单',
     lc.C_KV_S, lc.C_KV_F),
    ('第三层抽屉 · exec_future', 'execute_model 的原始故障单', '出事时唯一能告诉你哪步炸了的证据',
     lc.C_ABORT, '#fef2f2'),
]
DW = Q_W - 120
DX = Q_X + 60
DR_H, DR_PITCH = 92, 104
for bi, (t, sub, note, stroke, fill) in enumerate(TRIPLE):
    dy = Q_Y + 64 + bi * DR_PITCH
    lc.rect(DX, dy, DW, DR_H, fill, stroke, rx=6, sw=1.4)
    for gx in (DX + 18, DX + DW - 18):          # 抽屉拉手
        lc.seg(gx - 9, dy + 12, gx + 9, dy + 12, stroke, 2.2)
    lc.text(DX + DW / 2, dy + 27, t, 9.6, stroke, 'middle', True, maxw=DW - 30, tag='dw' + t[:6])
    lc.text(DX + DW / 2, dy + 46, sub, 8.8, '#334155', 'middle', maxw=DW - 30, tag='dws' + t[:6])
    lc.text(DX + DW / 2, dy + 64, note, 8.2, lc.C_MUTE, 'middle', maxw=DW - 30, tag='dwn' + t[:6])
# appendleft / pop 方向箭头（容器左右外侧，对准中层抽屉）
AR_Y = Q_Y + 64 + DR_PITCH + DR_H / 2
lc.parrow([(Q_X - 72, AR_Y), (Q_X - 6, AR_Y)], lc.C_ENG_S, 2.4, 'std')
lc.text(Q_X - 72, AR_Y - 12, 'appendleft', 8.6, lc.C_ENG_S, 'middle', True, maxw=80, tag='al')
lc.text(Q_X - 72, AR_Y + 14, '（新批进）', 7.8, lc.C_MUTE, 'middle', maxw=70, tag='al2')
lc.parrow([(Q_X + Q_W + 6, AR_Y), (Q_X + Q_W + 72, AR_Y)], lc.C_ENG_S, 2.4, 'std')
lc.text(Q_X + Q_W + 72, AR_Y - 12, 'pop', 8.6, lc.C_ENG_S, 'middle', True, maxw=60, tag='pp')
lc.text(Q_X + Q_W + 72, AR_Y + 14, '（最老批出）', 7.8, lc.C_MUTE, 'middle', maxw=70, tag='pp2')
lc.text(Q_X + Q_W / 2, Q_Y + Q_H - 12, 'FIFO = LIFO 容器倒着用：序由数据结构保证，无需任何簿记',
        8.4, lc.C_MUTE, 'middle', maxw=Q_W - 30, tag='q:note')

# ---------------- 右：FIFO 实拍 + None⇒重抛 ----------------
R_X = Q_X + Q_W + 116
R_W = BXR - R_X
# -- FIFO 实拍条 --
F_Y, F_H = Q_Y, 196
lc.rect(R_X, F_Y, R_W, F_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.3)
lc.text(R_X + 14, F_Y + 20, 'FIFO 实拍（max_tokens=8、采样行 [7]/[8]/[9] 逐拍交货）',
        9.6, lc.C_TXT, 'start', True, maxw=R_W - 26, tag='f:t')
FIFO_ROWS = [
    ('调用1', 'appendleft 批A 三元组 → return (None, True) 不等结果', '队 [A]'),
    ('调用2', 'appendleft 批B → pop 批A：t7 到账', '队 [B] · token [7]'),
    ('调用3', 'pop 批B：t8 到账（t7 先于 t8，批C 在飞）', '队 [C] · token [7,8]'),
]
for i, (a, b, c) in enumerate(FIFO_ROWS):
    yy = F_Y + 42 + i * 30
    lc.rect(R_X + 16, yy - 11, 52, 18, lc.C_BADGE_F, lc.C_ENG_S, rx=4, sw=1.0)
    lc.text(R_X + 42, yy + 2, a, 8.4, lc.C_ENG_S, 'middle', True, maxw=48, tag='fr' + a)
    lc.text(R_X + 78, yy + 2, b, 8.5, '#334155', 'start', maxw=330, tag='frb' + a)
    lc.text(R_X + R_W - 14, yy + 2, c, 8.4, lc.C_GPU_S, 'end', maxw=150, tag='frc' + a)
lc.text(R_X + 14, F_Y + 152, '先调度的批先取结果——t7 先于 t8 到账', 8.4, lc.C_MUTE,
        'start', maxw=R_W - 26, tag='f:note')
lc.text(R_X + 14, F_Y + 170, '稳态每拍恰一次 appendleft + 一次 pop（深度 2 一进一出）', 8.4,
        lc.C_MUTE, 'start', maxw=R_W - 26, tag='f:note2')

# -- None ⇒ 重抛 --
N_Y = F_Y + F_H + 16
N_H = Q_Y + Q_H - N_Y
lc.rect(R_X, N_Y, R_W, N_H, '#fef2f2', lc.C_ABORT, rx=7, sw=1.4)
lc.text(R_X + 14, N_Y + 20, 'pop 得 None ⇒ 翻第三层抽屉重抛真异常', 9.8, lc.C_ABORT, 'start',
        True, maxw=R_W - 26, tag='n:t')
FAIL_STEPS = [
    ('execute_model 失败', '异常先躺进第三层抽屉（sample future 的结果是 None——失败的信号）'),
    ('pop 侧 future.result() = None', 'core.py:L701-L706：None 意味着 execute_model() 已失败'),
    ('exec_model_fut.result()', '翻出原始异常对象本身，当面重抛 RuntimeError(\'real worker failure\')'),
]
for i, (a, b) in enumerate(FAIL_STEPS):
    yy = N_Y + 42 + i * 40
    lc.rect(R_X + 16, yy - 12, 196, 22, '#ffffff', lc.C_ABORT, rx=4, sw=1.0)
    lc.text(R_X + 114, yy + 3, a, 8.3, lc.C_ABORT, 'middle', True, maxw=188, tag='fsa' + str(i))
    lc.text(R_X + 224, yy + 3, b, 8.3, '#334155', 'start', maxw=R_W - 240, tag='fsb' + str(i))
    if i < 2:
        lc.seg(R_X + 114, yy + 12, R_X + 114, yy + 26, lc.C_ABORT, 1.4, 'std')
lc.text(R_X + 14, N_Y + N_H - 32, '对比：没有第三层抽屉，None 只能翻成 unexpected error 兜底文案，',
        8.4, lc.C_MUTE, 'start', maxw=R_W - 26, tag='n:cmp')
lc.text(R_X + 14, N_Y + N_H - 16, '故障定位要多走一轮日志考古', 8.4, lc.C_MUTE, 'start',
        maxw=R_W - 26, tag='n:cmp2')

# ---------------- 结论横幅 ----------------
BN_Y = Q_Y + Q_H + 18
lc.rect(MX, BN_Y, 1380, 38, lc.C_BEAT_F, lc.C_BEAT_S, rx=7, sw=1.4)
lc.text(MX + 690, BN_Y + 16.5, '一次 pop 的三件事：收采样（future.result）、对账单（SchedulerOutput → update_from_output）、验故障（exec_future 重抛）',
        9.8, lc.C_BEAT_T, 'middle', True, maxw=1360, tag='banner1')
lc.text(MX + 690, BN_Y + 30.5, 'FIFO 保序 + 异常不吞：错误不可能静默丢失——前两张丢了最多丢一顿饭，第三张是出事时的唯一证据',
        8.8, lc.C_BEAT_T, 'middle', maxw=1360, tag='banner2')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BN_Y + 64
lx = MX
lc.rect(lx, LEG_Y - 9, 22, 13, lc.C_GPU_F, lc.C_GPU_S, rx=3, sw=1.2)
lc.text(lx + 28, LEG_Y + 1, '第一层 · 采样 future', 8.5, lc.C_TXT, 'start', maxw=180, tag='leg:1')
lx += 28 + lc.tw('第一层 · 采样 future', 8.5) + 14
lc.rect(lx, LEG_Y - 9, 22, 13, lc.C_KV_F, lc.C_KV_S, rx=3, sw=1.2)
lc.text(lx + 28, LEG_Y + 1, '第二层 · SchedulerOutput', 8.5, lc.C_TXT, 'start', maxw=200, tag='leg:2')
lx += 28 + lc.tw('第二层 · SchedulerOutput', 8.5) + 14
lc.rect(lx, LEG_Y - 9, 22, 13, '#fef2f2', lc.C_ABORT, rx=3, sw=1.2)
lc.text(lx + 28, LEG_Y + 1, '第三层 · exec_future（故障单）', 8.5, lc.C_TXT, 'start', maxw=230, tag='leg:3')
lx += 28 + lc.tw('第三层 · exec_future（故障单）', 8.5) + 14
lc.seg(lx + 4, LEG_Y - 3, lx + 34, LEG_Y - 3, lc.C_ENG_S, 2.2, 'std')
lc.text(lx + 42, LEG_Y + 1, 'appendleft 进 / pop 出（FIFO）', 8.5, lc.C_TXT, 'start', maxw=220, tag='leg:fifo')

lc.text(MX, LEG_Y + 28, '逐字锚 vllm/v1/engine/core.py:L206-L212（建队）/ L681（appendleft 三元组）/ '
        'L696（pop 最老批）/ L701-L706（None ⇒ 重抛）· FIFO/重挂数字取自配套精简版 host 实跑 · 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch12-fig-queue-triple.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
