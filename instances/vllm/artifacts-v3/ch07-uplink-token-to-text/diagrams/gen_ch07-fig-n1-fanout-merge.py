#!/usr/bin/env python3
"""ch07 机制图 7 · n>1 扇出与父聚合（figure_spec ch07-fig-n1-fanout-merge，模板 flow）

放大自 L0 蓝色 API 进程带（api_band · 本章 l0_zoom『API 进程上行泳道』）的登记与造输出
工位——即本章 L2 章图 center 拍片 ① 『上行登记』的 n>1 扇出支线 + 拍片 ⑥ 『造输出三道闸』
的父聚合分支展开。架构归属回指 L2/L0（FIGURE-SYSTEM §3.3）。

claim：n=3 的一次请求在 add_request 撕成 3 张 idx_ 前缀子单（末子复用原对象、seed 逐子克隆
42/43/44），n 条流水的输出按 CompletionOutput.index 在唯一的 collector 单槽里配对成托盘
（实测一轮 3 put 合并为 outputs=[(0,"a"),(1,"b"),(2,"c")] 互不覆盖），最后一个子完成才
finished；FINAL_ONLY 则攒齐 3 个一次返回。

数字全部取自 figure_spec.numbers（host 实测 trace + pin 锚点）；坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 852
MX = 60
BXR = 1440
C_BODY = '#334155'
IDX_C = [lc.C_API_S, '#0e7490', '#7c2d12']     # 三档 index 色（图例声明）


def dot(cx, cy, r, fill):
    lc.ELEMS.append(((cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2),
                     f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"/>'))


# ---------------- 标题区 ----------------
lc.text(MX, 34, '一单三杯：n>1 撕成 3 张 idx_ 子单，输出按 index 配对上托盘',
        16.5, lc.C_TXT, 'start', True, maxw=1020, tag='title')
lc.text(MX, 58, '扇出在 add_request、请求过线之前；n 条流水线共享同一个 collector——最后一杯完成才算 finished',
        10.5, lc.C_MUTE, 'start', maxw=980, tag='subtitle')
_ch = '放大自 L2 拍片 ① 上行登记（n>1 支线）+ ⑥ 父聚合 · L0：API 进程上行泳道'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_API_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 段 ①：撕单 + 共享 collector ----------------
lc.text(80, 116, '① 登记段 · 撕单（add_request 内扇出）', 10, lc.C_TXT, 'start', True,
        maxw=380, tag='s1:t')
lc.rect(80, 128, 300, 54, lc.C_API_F, lc.C_API_S, rx=7, sw=1.5)
lc.text(230, 150, '一次请求 · n=3 · seed=42', 10.5, lc.C_TXT, 'middle', True, maxw=280, tag='par:t')
lc.text(230, 170, 'async_llm.py:L401-L418', 8, lc.C_FAINT, 'middle', maxw=280, tag='par:f')
CHILDREN = [('子 0 · id = 0_ext-seed-00000001', 'seed 42 · params n=1（拷贝）'),
            ('子 1 · id = 1_…', 'seed 43 · params n=1（拷贝）'),
            ('子 2 · id = 2_…', 'seed 44 · 复用原对象')]
CARD_W, CARD_H, CARD_Y = 118, 74, 224
card_cx = []
for i, (t1, t2) in enumerate(CHILDREN):
    x = 80 + i * (CARD_W + 12)
    card_cx.append(x + CARD_W / 2)
    lc.rect(x, CARD_Y, CARD_W, CARD_H, '#ffffff', IDX_C[i], rx=6, sw=1.3)
    lc.text(x + CARD_W / 2, CARD_Y + 18, ['子 0', '子 1', '子 2'][i], 10, IDX_C[i], 'middle',
            True, maxw=CARD_W - 10, tag='cd' + str(i))
    lc.text(x + CARD_W / 2, CARD_Y + 36, 'id = ' + ['0_…', '1_…', '2_…'][i], 8, C_BODY, 'middle',
            maxw=CARD_W - 8, tag='cdi' + str(i))
    lc.text(x + CARD_W / 2, CARD_Y + 52, ['seed 42（拷贝）', 'seed 43（拷贝）', 'seed 44（复用原对象）'][i],
            7.5, lc.C_MUTE, 'middle', maxw=CARD_W - 6, tag='cds' + str(i))
    lc.parrow([(230 if i == 1 else 120 + i * 60, 182), (x + CARD_W / 2, CARD_Y - 2)],
              lc.C_API_S, 1.6, 'dn')
lc.text(80, 396, '子 id = 0_ext-seed-00000001（idx_ 前缀 + 内部 id）', 7.5, C_BODY, 'start',
        maxw=370, tag='cd0full')
lc.text(80, 410, '后缀是驱动钉的计数器取可复现值，真服务器为随机 8-hex', 7.5, lc.C_MUTE, 'start',
        maxw=370, tag='cd0n1')
# 子单 → 共享 collector
COL_Y = 312
for i, cx in enumerate(card_cx):
    lc.seg(cx, CARD_Y + CARD_H, cx, COL_Y - 2, IDX_C[i], 1.6, 'std')
lc.rect(80, COL_Y, 378, 66, '#ffffff', lc.C_API_S, rx=7, sw=1.5)
lc.text(269, COL_Y + 24, '共享同一个 RequestOutputCollector（3 子 1 槽）', 10, lc.C_TXT, 'middle',
        True, maxw=360, tag='col:t')
lc.text(269, COL_Y + 44, '在 add_request 里、请求过线之前诞生 · 扇出 = 3 条 ADD 帧（末子复用原对象省 2 次拷贝）',
        8, lc.C_MUTE, 'middle', maxw=360, tag='col:s')

# ---------------- 段 ②：n 条流水 ----------------
lc.text(560, 116, '② 流水段', 10, lc.C_TXT, 'start', True, maxw=200, tag='s2:t')
lc.text(560, 132, '各自做各自的一杯（同一批输出里交错）', 8, lc.C_MUTE, 'start', maxw=200,
        tag='s2:s')
LANE_PROD = ['a', 'b', 'c']
lane_y = []
for i in range(3):
    ly = 244 + i * 56
    lane_y.append(ly)
    lc.text(556, ly + 3, ['子 0', '子 1', '子 2'][i], 8.5, IDX_C[i], 'end', True, maxw=44,
            tag='ln' + str(i))
    lc.rect(562, ly - 16, 116, 32, '#ffffff', IDX_C[i], rx=5, sw=1.2)
    lc.text(620, ly + 3, '轮 1 产出 "' + LANE_PROD[i] + '"', 8.5, C_BODY, 'middle', maxw=108,
            tag='lp' + str(i))

# ---------------- 段 ③：托盘 + 完成时间线 ----------------
TRAY_X, TRAY_W = 800, 640
lc.text(TRAY_X, 116, '③ 合并段 · 托盘（RequestOutput.outputs）', 10, lc.C_TXT, 'start', True,
        maxw=560, tag='s3:t')
lc.rect(TRAY_X, 128, TRAY_W, 106, '#ffffff', lc.C_API_S, rx=7, sw=1.5)
slot_x = [TRAY_X + 30, TRAY_X + 30 + 156, TRAY_X + 30 + 312]
for i in range(3):
    lc.rect(slot_x[i], 158, 150, 46, '#f8fafc', IDX_C[i], rx=5, sw=1.3)
    lc.text(slot_x[i] + 75, 176, 'index ' + str(i), 8, IDX_C[i], 'middle', True, maxw=140,
            tag='sl' + str(i))
    lc.text(slot_x[i] + 75, 194, '(' + str(i) + ', "' + LANE_PROD[i] + '")', 10, lc.C_TXT,
            'middle', True, maxw=140, tag='slv' + str(i))
lc.text(TRAY_X + TRAY_W / 2, 224, '轮 1：3 个 put 合并成 1 个对象 outputs=[(0,"a"),(1,"b"),(2,"c")]——按 CompletionOutput.index 配对，互不覆盖',
        8.3, C_BODY, 'middle', maxw=620, tag='tray:n')
# 流水 → 托盘（肘形汇入）
for i, ly in enumerate(lane_y):
    tgt_y = 181
    lc.parrow([(678, ly), (740 + i * 18, ly), (740 + i * 18, tgt_y), (TRAY_X - 2, tgt_y)],
              IDX_C[i], 1.6, 'std')
lc.text(TRAY_X, 258, '完成时间线（实测逆序 2 → 1 → 0）：每杯完成即转发，只有最后一杯点亮 finished',
        9.5, lc.C_TXT, 'start', True, maxw=620, tag='tl:t')
FIN = [('子 2 流出 "!"', '转发 [(2,"!")]', False),
       ('子 1 流出 "@"', '转发 [(1,"@")]', False),
       ('子 0 流出 "#"（最后一杯）', '转发 [(0,"#")] · finished=true', True)]
for i, (t1, t2, fin) in enumerate(FIN):
    y = 276 + i * 40
    dot(TRAY_X + 8, y - 3, 4, IDX_C[2 - i] if i < 2 else IDX_C[0])
    lc.text(TRAY_X + 20, y, t1, 9, lc.C_TXT, 'start', True, maxw=240, tag='ft' + str(i))
    lc.text(TRAY_X + 250, y, t2, 9, fin and IDX_C[0] or lc.C_MUTE, 'start', True, maxw=340,
            tag='ft2' + str(i))
    if fin:
        lc.text(TRAY_X + 560, y, 'finished', 8.5, '#ea580c', 'end', True, maxw=70, tag='f lamp')

# ---------------- FINAL_ONLY 攒位架对照 ----------------
RK_Y, RK_H = 436, 130
lc.rect(MX, RK_Y, BXR - MX, RK_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(MX + 16, RK_Y + 22, '对照 · FINAL_ONLY：output_aggregator 按 index 攒位——攒齐 3 个才一次端（puts 0/0/1）',
        9.5, lc.C_TXT, 'start', True, maxw=1100, tag='rk:t')
RACKS = [(['None', 'None', 'None'], 'puts=0（攒）'), (['"xx"', 'None', 'None'], 'puts=0（攒）'),
         (['"xx"', '"yy"', 'None'], 'puts=1：攒齐 3 个一次返回'), (['"xx"', '"yy"', '"zz"'], 'finished=true（index 有序 0/1/2）')]
rw, rh, rgap = 56, 34, 8
rx0 = MX + 40
for si, (cells, cap) in enumerate(RACKS):
    bx = rx0 + si * 340
    for ci, cv in enumerate(cells):
        filled = cv != 'None'
        lc.rect(bx + ci * (rw + rgap), RK_Y + 38, rw, rh, '#ffffff' if not filled else lc.C_API_F,
                lc.C_FAINT if not filled else lc.C_API_S, rx=4, sw=1.0, dash=not filled)
        lc.text(bx + ci * (rw + rgap) + rw / 2, RK_Y + 59, cv, 8, lc.C_MUTE if not filled else lc.C_TXT,
                'middle', True, maxw=rw - 4, tag='rk%d%d' % (si, ci))
    if si < 3:
        lc.seg(bx + 3 * (rw + rgap) + 22, RK_Y + 55, bx + 340 - 18, RK_Y + 55,
               lc.C_MUTE, 1.4, 'std')
    lc.text(bx + (3 * rw + 2 * rgap) / 2, RK_Y + 96, cap, 8, lc.C_MUTE, 'middle',
            maxw=300, tag='rkc' + str(si))

# ---------------- 三条补充注 ----------------
NB_Y, NB_H = 592, 84
notes = [('输出写回父外部 id', 'get_outputs 后 external_req_id = 父请求的外部 id（output_processor.py:L326-L332）'),
         ('params 两条路', '带 seed：逐子克隆唯一 seed（42/43/44，对象不共享）；无 seed：缓存复用同一 params 对象'),
         ('防重发', 'get_outputs 对已转发的子先移出 child_requests——already_finished_and_returned，不二次发')]
nw = (BXR - MX - 2 * 20) / 3
for i, (t, l) in enumerate(notes):
    x = MX + i * (nw + 20)
    lc.rect(x, NB_Y, nw, NB_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.1)
    lc.text(x + 12, NB_Y + 20, t, 9, lc.C_TXT, 'start', True, maxw=nw - 24, tag='nb' + str(i))
    lc.text(x + 12, NB_Y + 40, l, 8, C_BODY, 'start', maxw=nw - 24, tag='nbl' + str(i))

# ---------------- 图例 + 页脚 ----------------
LEG_Y = NB_Y + NB_H + 32
lx = MX
for i, name in enumerate(['index 0', 'index 1', 'index 2']):
    lc.rect(lx, LEG_Y - 8, 18, 13, '#ffffff', IDX_C[i], rx=4, sw=1.4)
    lc.text(lx + 24, LEG_Y + 2, name, 9, lc.C_TXT, 'start', maxw=90, tag='leg' + str(i))
    lx += 24 + lc.tw(name, 9) + 18
lc.rect(lx, LEG_Y - 8, 18, 13, '#ffffff', lc.C_MUTE, rx=4, sw=1.1, dash=True)
lc.text(lx + 24, LEG_Y + 2, '攒位空槽（None）', 9, lc.C_TXT, 'start', maxw=160, tag='legn')
lx += 24 + lc.tw('攒位空槽（None）', 9) + 18
lc.text(lx, LEG_Y + 2, '● 完成事件（逆序 2→1→0）', 9, lc.C_TXT, 'start', maxw=240, tag='legd')
lc.text(MX, LEG_Y + 28, '扇出 verbatim vllm/v1/engine/async_llm.py:L401-L418 · ParentRequest vllm/v1/engine/parallel_sampling.py:L36-L126 · '
        'add 配对 vllm/outputs.py:L152-L181 · 扇出/聚合/攒位 host 实测 · 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch07-fig-n1-fanout-merge.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
