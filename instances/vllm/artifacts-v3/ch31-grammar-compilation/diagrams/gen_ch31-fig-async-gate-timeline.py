#!/usr/bin/env python3
"""ch30 机制图 · m11 异步编译门时间线（figure_spec ch30-fig-async-gate-timeline，模板 swimlane）

放大自 L0 调度列与采样列之间『编译门』的时序展开——引擎侧 L2 章图站 3-6 的横向
时间轴（时序视图），架构归属回指 L2 章图，不另立第二种架构画法（FIGURE-SYSTEM §3）。
时序图画法守 FIGURE-SYSTEM §0 规约 9（UML 文法）：参与者=竖直生命线（顶部名牌）、
共享时间轴（gridlines 贯穿全部生命线、元素按时间戳 y 对齐、活动条=时长×示意比例）、
消息=水平直线、瞬时动作=骑线时刻标记+真实值标注；时间轴不按比例（footer 声明）。

claim：慢 schema 编译期间引擎照常服务：gr-1 出生即 WAITING_FOR_STRUCTURED_OUTPUT_
GRAMMAR 入侧队，第一拍只批 plain-1，编译完成的第二拍 gr-1 晋升当拍入批——100µs
非阻塞探测从不卡住忙循环。

数字全部取自 figure_spec.numbers（gate_timeline 两拍、timeout=0.0001s、
max_workers=(cpu+1)//2、状态名 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR 不缩写）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1580, 906
MX, BXR = 60, 1540
L_IO, L_POOL, L_SCH = 300.0, 640.0, 990.0     # 三条生命线 x
TRK_X0, TRK_X1 = 1160.0, 1520.0               # 请求状态轨
RULER_X = 190.0
GRID_X0, GRID_X1 = 196.0, 1520.0
C_BOLT = '#d97706'                            # 探测闪电
GR_F, GR_S, GR_T = '#fff7ed', '#fdba74', '#9a3412'   # gr-1（暖）
PLAIN_S = '#94a3b8'                           # plain-1（中性）

DEFS = lc.DEFS + (
    f'<marker id="eng" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
    f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_ENG_S}"/></marker>')


def bolt(cx, cy, k=1.0):
    """小闪电（瞬时探测标记）：k=缩放。"""
    pts = [(0, -7), (3.5, -1.5), (1, -1.5), (4.5, 6), (-0.5, 0.5), (2, 0.5)]
    p = ' '.join(f'{cx + px * k:.1f},{cy + py * k:.1f}' for px, py in pts)
    lc.ELEMS.append(((cx - 2 * k, cy - 8 * k, cx + 5 * k, cy + 7 * k),
                     f'<polygon points="{p}" fill="{C_BOLT}" stroke="none"/>'))


# ---------------- 标题区 ----------------
lc.text(MX, 34, '编译门时间线：没编完的请求不进批、也拖不住别人', 16.5, lc.C_TXT, 'start', True,
        maxw=1020, tag='title')
lc.text(MX, 58, 'gr-1（grammar root ::= "yes" | "no"，编译被受控压住以观察未就绪窗口）出生即阻塞态入侧队；'
               '拍 1 只批 plain-1；编译完成的拍 2 gr-1 晋升当拍入批——每拍探测 100µs 非阻塞，从不卡住忙循环',
        10.5, lc.C_MUTE, 'start', maxw=1370, tag='subtitle')
_ch = '放大自 L0 调度列↔采样列『编译门』 · L2 站 3-6 的时序视图'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 泳道头 + 生命线 ----------------
HDR_Y, HDR_H = 88, 34
LANES = [
    (L_IO, 'IO 线程（输入处理）', 'grammar_init 只在此调用·不占忙循环', lc.C_ENG_F, lc.C_ENG_S, lc.C_ENG_S),
    (L_POOL, '编译线程池', 'max_workers=(cpu+1)//2 · 半 CPU', lc.C_BEAT_F, lc.C_BEAT_S, lc.C_BEAT_T),
    (L_SCH, '调度器忙循环', 'schedule() → … → update 逐拍', lc.C_ENG_F, lc.C_ENG_S, lc.C_ENG_S),
]
TL_END = 660.0
for cx, nm, sub, f_, s_, t_ in LANES:
    lc.rect(cx - 118, HDR_Y, 236, HDR_H, f_, s_, rx=7, sw=1.6)
    lc.text(cx, HDR_Y + 15, nm, 10.5, t_, 'middle', True, maxw=226, tag='lane:' + nm)
    lc.text(cx, HDR_Y + 28, sub, 7.8, lc.C_MUTE, 'middle', maxw=226, tag='lane:s' + nm)
    lc.seg(cx, HDR_Y + HDR_H + 2, cx, TL_END + 6, lc.C_FAINT, 1.0, dash=True)
# 请求状态轨表头
lc.rect(TRK_X0, HDR_Y, TRK_X1 - TRK_X0, HDR_H, '#f8fafc', lc.C_MUTE, rx=7, sw=1.4)
lc.text((TRK_X0 + TRK_X1) / 2, HDR_Y + 15, '请求状态轨', 10.5, lc.C_TXT, 'middle', True, maxw=340, tag='trk:t')
lc.text((TRK_X0 + TRK_X1) / 2, HDR_Y + 28, 'gr-1（暖）/ plain-1（白）随时间戳对齐', 7.8, lc.C_MUTE,
        'middle', maxw=346, tag='trk:s')

# ---------------- 时间轴（四条栅格，不按比例） ----------------
T0, T1, TC, T2 = 170.0, 330.0, 470.0, 610.0     # 进门 / 第一拍 / 编译完成 / 第二拍
GRID = [(T0, '进门'), (T1, '第一拍 schedule()'), (TC, '编译完成（Future 兑现）'), (T2, '第二拍 schedule()')]
for ty, lab in GRID:
    lc.seg(GRID_X0, ty, GRID_X1, ty, '#cbd5e1', 1.0, dash=True)
    lc.text(RULER_X - 6, ty + 3, lab, 8.2, lc.C_MUTE, 'end', maxw=150, tag='t:' + lab)
lc.seg(RULER_X, T0 - 14, RULER_X, TL_END, lc.C_MUTE, 1.2)
for ty, _ in GRID:
    lc.seg(RULER_X - 3, ty, RULER_X + 3, ty, lc.C_MUTE, 1.0)
lc.text(RULER_X - 6, T0 - 18, '时间 →', 8.2, lc.C_MUTE, 'end', maxw=60, tag='ruler:cap')

# ---------------- 事件：进门 ----------------
lc.seg(L_IO, T0, L_POOL, T0, lc.C_ENG_S, 2.2, 'eng')       # submit（水平消息）
lc.text((L_IO + L_POOL) / 2, T0 - 8, 'executor.submit(_create_grammar, gr-1) → Future 挂进请求', 8.6,
        lc.C_ENG_S, 'middle', True, maxw=330, tag='ev:submit')
lc.text(L_IO, T0 + 18, 'add_request 分流：gr-1 带约束→grammar_init；', 7.8, '#334155', 'middle',
        maxw=216, tag='ev:add1')
lc.text(L_IO, T0 + 32, 'plain-1 无约束→直接 waiting', 7.8, '#334155', 'middle', maxw=216, tag='ev:add2')
# 编译活动条（UML 活动条：进门→编译完成）
lc.rect(L_POOL - 9, T0 + 6, 18, TC - T0 - 6, GR_F, GR_S, rx=3, sw=1.2)
lc.text(L_POOL + 16, T0 + 24, '编译 gr-1 的语法', 8.4, GR_T, 'start', True, maxw=250, tag='act:t')
lc.text(L_POOL + 16, T0 + 38, '（受控压住以观察未就绪窗口；', 7.6, lc.C_MUTE, 'start', maxw=250, tag='act:l1')
lc.text(L_POOL + 16, T0 + 51, '  小语法真实编译毫秒级）', 7.6, lc.C_MUTE, 'start', maxw=250, tag='act:l2')

# ---------------- 事件：第一拍（探测未就绪） ----------------
bolt(L_SCH, T1)
lc.text(L_SCH, T1 + 22, '窥侧队队头 gr-1：result(timeout=0.0001s)', 8.4, C_BOLT, 'middle', True,
        maxw=300, tag='ev:p1a')
lc.text(L_SCH, T1 + 36, 'TimeoutError → grammar=None（100µs 即回）', 8, '#334155', 'middle', maxw=300,
        tag='ev:p1b')
lc.text(L_SCH, T1 + 50, 'gr-1 prepend 回侧队·状态不变；本拍批=[plain-1]', 8, '#334155', 'middle',
        maxw=300, tag='ev:p1c')

# ---------------- 事件：编译完成 ----------------
lc.ELEMS.append(((L_POOL - 6, TC - 6, L_POOL + 6, TC + 6),
                 f'<circle cx="{L_POOL}" cy="{TC}" r="5" fill="{GR_T}"/>'))
lc.text(L_POOL + 14, TC + 16, 'Future 兑现：XgrammarGrammar', 8.4, GR_T, 'start', True, maxw=250,
        tag='ev:comp')
lc.text(L_POOL + 14, TC + 30, '（就绪单调：_grammar 一旦替换不再变回 Future）', 7.6, lc.C_MUTE,
        'start', maxw=260, tag='ev:comp2')

# ---------------- 事件：第二拍（就绪晋升） ----------------
bolt(L_SCH, T2)
lc.text(L_SCH, T2 + 22, '窥队头：grammar=XgrammarGrammar（已就绪）', 8.4, C_BOLT, 'middle', True,
        maxw=300, tag='ev:p2a')
lc.text(L_SCH, T2 + 36, '晋升 status=WAITING → 当拍入批 → RUNNING', 8, '#334155', 'middle', maxw=300,
        tag='ev:p2b')
lc.text(L_SCH, T2 + 50, '批=[gr-1] · has_structured_output_requests=True', 8, '#334155', 'middle',
        maxw=300, tag='ev:p2c')

# ---------------- 请求状态轨（随时间戳对齐的 chips） ----------------
CHIPS = [
    (T0, -14, 'gr-1', '出生即 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR → 入侧队'),
    (T0, 26, 'plain-1', '无约束 → 正常 waiting'),
    (T1, -14, 'gr-1', '探测未就绪（100µs 超时）→ prepend 回 skipped_waiting·状态不变'),
    (T1, 26, 'plain-1', '入批 scheduled=[plain-1] → RUNNING → 出 token'),
    (T2, -14, 'gr-1', '就绪晋升：当拍入批 → RUNNING'),
    (T2, 26, 'plain-1', '照常解码（没被拖住）'),
]
for ty, dy, who, txt in CHIPS:
    warm = who == 'gr-1'
    f_, s_ = (GR_F, GR_S) if warm else ('#ffffff', PLAIN_S)
    t_ = GR_T if warm else '#334155'
    lc.rect(TRK_X0 + 4, ty + dy - 12, TRK_X1 - TRK_X0 - 8, 24, f_, s_, rx=6, sw=1.1)
    lc.text((TRK_X0 + TRK_X1) / 2, ty + dy + 3.5, txt, 7.6, t_, 'middle', maxw=TRK_X1 - TRK_X0 - 24,
            tag='chip:' + who + str(int(ty)))

# ---------------- 底部两注 ----------------
NY, NH = 700, 108
lc.rect(MX, NY, 700, NH, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(MX + 16, NY + 20, 'external_launcher（torchrun）模式：回退同步编译', 9.5, lc.C_TXT, 'start', True,
        maxw=660, tag='n1:t')
for j, ln in enumerate(['· 异步会让 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR→WAITING 的迁移时刻在各 rank 不同',
                        '  （每 rank 一个调度器），破坏确定性 → 死锁；该模式下编译回到同步路径',
                        '· 同步分支把异常也包成 Future——两条路径对下游同形']):
    lc.text(MX + 16, NY + 40 + j * 16, ln, 8.2, '#334155', 'start', maxw=668, tag='n1:l' + str(j))
lc.rect(780, NY, BXR - 780, NH, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(796, NY + 20, '编译失败只杀单请求（异常全程值传递，不抛穿忙循环）', 9.5, lc.C_TXT, 'start', True,
        maxw=730, tag='n2:t')
for j, ln in enumerate(['· raise → Future 装异常 → 探测存进 _grammar → 晋升检查记账 → 同拍 finish_requests',
                        '  (FINISHED_ERROR) + 空 token 回执——引擎与其余请求无感',
                        '· 100µs 探测=忙等变体：每拍每请求一次、成本钳在 timeout（忙循环零阻塞）']):
    lc.text(796, NY + 40 + j * 16, ln, 8.2, '#334155', 'start', maxw=BXR - 796 - 12, tag='n2:l' + str(j))

# ---------------- 图例 + 页脚 ----------------
LY = 838
lx0 = MX
lc.rect(lx0, LY - 9, 16, 11, lc.C_ENG_F, lc.C_ENG_S, rx=3, sw=1.4)
lc.text(lx0 + 21, LY + 1, 'EngineCore（IO 线程 / 调度器忙循环）', 8.8, lc.C_TXT, 'start', maxw=280, tag='lg1')
lx0 += 21 + lc.tw('EngineCore（IO 线程 / 调度器忙循环）', 8.8) + 16
lc.rect(lx0, LY - 9, 16, 11, lc.C_BEAT_F, lc.C_BEAT_S, rx=3, sw=1.4)
lc.text(lx0 + 21, LY + 1, '编译线程池（半 CPU）', 8.8, lc.C_TXT, 'start', maxw=180, tag='lg2')
lx0 += 21 + lc.tw('编译线程池（半 CPU）', 8.8) + 16
lc.rect(lx0, LY - 9, 16, 11, GR_F, GR_S, rx=3, sw=1.1)
lc.text(lx0 + 21, LY + 1, 'gr-1（语法请求）', 8.8, lc.C_TXT, 'start', maxw=160, tag='lg3')
lx0 += 21 + lc.tw('gr-1（语法请求）', 8.8) + 16
lc.rect(lx0, LY - 9, 16, 11, '#ffffff', PLAIN_S, rx=3, sw=1.1)
lc.text(lx0 + 21, LY + 1, 'plain-1（无约束请求）', 8.8, lc.C_TXT, 'start', maxw=180, tag='lg4')
lx0 += 21 + lc.tw('plain-1（无约束请求）', 8.8) + 16
bolt(lx0 + 12, LY - 3, 0.9)
lc.text(lx0 + 24, LY + 1, '100µs 非阻塞探测', 8.8, lc.C_TXT, 'start', maxw=140, tag='lg5')
lx0 += 24 + lc.tw('100µs 非阻塞探测', 8.8) + 16
lc.seg(lx0 + 4, LY - 3, lx0 + 34, LY - 3, '#cbd5e1', 1.0, dash=True)
lc.text(lx0 + 40, LY + 1, '时间栅格（不按比例）', 8.8, lc.C_TXT, 'start', maxw=170, tag='lg6')

lc.text(MX, 866, 'vllm/v1/engine/core.py:L969-L991（preprocess_add_request → grammar_init，只在 IO 线程）· '
                 'structured_output/__init__.py:L70-L80（半 CPU 线程池）· L114-L175（executor.submit → Future）· '
                 'scheduler.py:L698-L722 / L2678-L2712（窥队·晋级）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, 882, 'structured_output/request.py:L50-L59（timeout=0.0001s = 100µs 探测）· '
                 '两拍时间线 / 状态名 / 批组成 ＝ 本章受控 trace 实测 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch30-fig-async-gate-timeline.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
