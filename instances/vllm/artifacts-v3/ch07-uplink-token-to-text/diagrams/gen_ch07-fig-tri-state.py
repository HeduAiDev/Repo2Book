#!/usr/bin/env python3
"""ch07 机制图 6 · 三态契约 DELTA/CUMULATIVE/FINAL_ONLY（figure_spec ch07-fig-tri-state，模板 swimlane）

放大自 L0 蓝色 API 进程带（api_band · 本章 l0_zoom『API 进程上行泳道』）的造输出工位——
即本章 L2 章图 center 拍片 ⑥ 『造输出三道闸』+ south『why · 三态契约与节流』注的机制展开。
架构归属回指 L2/L0（FIGURE-SYSTEM §3.3）。

claim：同一 token 流（"He"→"ll"→"o" 完成）在三种 output_kind 下产生 3/3/1 条 RequestOutput：
DELTA 每步一条增量、CUMULATIVE 每步一条膨胀快照、FINAL_ONLY 中间步零构造只在完成步给一条
全文——三态终值全等 "Hello"，差别只在中间流量。

数字全部取自 figure_spec.numbers（host 实测 trace + pin 锚点）；坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 796
MX = 60
BXR = 1440
SLOT_CX = [350, 530, 710]        # 三时刻列心
SUM_X0, SUM_X1 = 800, 1090       # 泳道小结区

# ---------------- 标题区 ----------------
lc.text(MX, 34, '同一锅 token，三种上菜节奏：3 条 / 3 条 / 1 条——终值一字不差',
        16.5, lc.C_TXT, 'start', True, maxw=1020, tag='title')
lc.text(MX, 58, '点单时（入口的 stream 参数）选好 output_kind，OutputProcessor 照单裁剪——引擎侧不感知',
        10.5, lc.C_MUTE, 'start', maxw=880, tag='subtitle')
_ch = '放大自 L2 拍片 ⑥ 造输出三道闸 · L0：API 进程上行泳道'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_API_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 输入轴（三时刻列头，共享） ----------------
STEPS = [('轮 1 · [72,101]', '"He"'), ('轮 2 · [108,108]', '"ll"'), ('轮 3 · [111] + LENGTH', '"o" · 完成')]
lc.text(262, 121, '同一 token 流', 9.5, lc.C_MUTE, 'end', True, maxw=90, tag='in:lbl')
for cx, (t1, t2) in zip(SLOT_CX, STEPS):
    lc.rect(cx - 75, 96, 150, 50, lc.C_API_F, lc.C_API_S, rx=7, sw=1.4)
    lc.text(cx, 116, t1, 9, lc.C_TXT, 'middle', True, maxw=140, tag='st1' + t1)
    lc.text(cx, 134, t2, 8.5, lc.C_MUTE, 'middle', maxw=140, tag='st2' + t2)
# 列对齐虚线（极淡，垫底）
for cx in SLOT_CX:
    lc.seg(cx, 150, cx, 606, '#e2e8f0', 1.0, dash=True)

# ---------------- 三条泳道 ----------------
LANES = [
    dict(cy=230, name='DELTA', sub='每步一条增量小碟',
         boxes=[('He (2)', 90), ('ll (2)', 90), ('o (1) finished', 110)],
         s1='合计 3 条 · token 2/2/1', s2='拼接 = "Hello"'),
    dict(cy=384, name='CUMULATIVE', sub='每步一条膨胀快照',
         boxes=[('He (2)', 100), ('Hell (4)', 150), ('Hello (5) finished', 175)],
         s1='合计 3 条 · token 2/4/5', s2='末快照 = "Hello"（每条都是全量）'),
    dict(cy=538, name='FINAL_ONLY', sub='全程沉默，收锅一整盘',
         boxes=[('return None', 140, True), ('return None', 140, True), ('Hello (5) finished', 175)],
         s1='合计 1 条 · 中间 puts=0', s2='只在完成步构造（puts_before_finish=0）'),
]
BH = 46
for ln in LANES:
    cy = ln['cy']
    lc.rect(160, cy - 62, SUM_X1 - 160 + 10, 124, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
    lc.text(262, cy - 12, ln['name'], 11.5, lc.C_API_S, 'end', True, maxw=120, tag='ln' + ln['name'])
    lc.text(262, cy + 8, ln['sub'], 8.5, lc.C_MUTE, 'end', maxw=120, tag='lns' + ln['name'])
    prev_right = None
    for i, b in enumerate(ln['boxes']):
        t, bw = b[0], b[1]
        empty = len(b) > 2 and b[2]
        x0 = SLOT_CX[i] - bw / 2
        if empty:
            lc.rect(x0, cy - BH / 2, bw, BH, '#ffffff', lc.C_FAINT, rx=6, sw=1.2, dash=True)
            lc.text(SLOT_CX[i], cy - 3, t, 9.5, lc.C_MUTE, 'middle', True, maxw=bw - 10,
                    tag='bx' + t + str(i))
            lc.text(SLOT_CX[i], cy + 13, '（零构造 · 槽不沾）', 8, lc.C_FAINT, 'middle',
                    maxw=bw - 10, tag='bxs' + t + str(i))
        else:
            lc.rect(x0, cy - BH / 2, bw, BH, lc.C_API_F, lc.C_API_S, rx=6, sw=1.4)
            lc.text(SLOT_CX[i], cy + 4, t, 9.5, lc.C_TXT, 'middle', True, maxw=bw - 10,
                    tag='bx' + t + str(i))
        if prev_right is not None:
            lc.seg(prev_right + 3, cy, x0 - 4, cy, lc.C_API_S, 1.4, 'std')
        prev_right = x0 + bw
    lc.text(SUM_X0 + 14, cy - 12, ln['s1'], 9, lc.C_TXT, 'start', True, maxw=SUM_X1 - SUM_X0 - 24,
            tag='sm1' + ln['name'])
    lc.text(SUM_X0 + 14, cy + 8, ln['s2'], 8.5, lc.C_MUTE, 'start', maxw=SUM_X1 - SUM_X0 - 24,
            tag='sm2' + ln['name'])
lc.text(860, 308, '↓ CUMULATIVE 快照逐轮膨胀（宽度按 token 数 2 : 4 : 5）', 8, lc.C_MUTE,
        'middle', maxw=280, tag='cum:cap')

# ---------------- 右栏：入口声明 + 省链 + 单槽注 ----------------
RP_X = 1120
lc.rect(RP_X, 96, BXR - RP_X, 214, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(RP_X + 14, 120, '入口点单时声明（引擎不感知）', 10, lc.C_TXT, 'start', True, maxw=280,
        tag='dec:t')
for i, ln in enumerate(['chat stream=True → DELTA',
                        'stream=False → FINAL_ONLY（chat 与 completion 同构）',
                        '离线 LLM 强制 FINAL_ONLY——',
                        '"We only care about the final output"']):
    lc.text(RP_X + 14, 144 + i * 19, ln, 8.5, '#334155' if i != 3 else lc.C_MUTE, 'start',
            maxw=290, tag='dec:l' + str(i))
lc.text(RP_X + 14, 232, 'chat_completion/protocol.py:L722-L724', 8, lc.C_FAINT, 'start',
        maxw=290, tag='dec:f1')
lc.text(RP_X + 14, 248, 'offline_utils.py:L560-L561', 8, lc.C_FAINT, 'start', maxw=290,
        tag='dec:f2')
lc.text(RP_X + 14, 288, '采样参数里的 RequestOutputKind 三态枚举', 8, lc.C_MUTE, 'start',
        maxw=290, tag='dec:f3')

lc.rect(RP_X, 326, BXR - RP_X, 106, lc.C_API_F, lc.C_API_S, rx=8, sw=1.3)
lc.text(RP_X + 14, 350, 'FINAL_ONLY 省的不只是网络', 10, lc.C_TXT, 'start', True, maxw=280,
        tag='sav:t')
lc.text(RP_X + 14, 372, '中间步的 RequestOutput 构造 + collector.put +', 8.5, '#334155',
        'start', maxw=290, tag='sav:l1')
lc.text(RP_X + 14, 388, 'Event 唤醒整条链都不发生——闸门在构造前', 8.5, '#334155', 'start',
        maxw=290, tag='sav:l2')
lc.text(RP_X + 14, 404, '就 return None（output_processor.py:L286-L290）', 8.5, '#334155',
        'start', maxw=290, tag='sav:l3')
lc.text(RP_X + 14, 424, '非流式 HTTP 与离线批处理都走 FINAL_ONLY', 8.5, lc.C_API_S, 'start',
        True, maxw=290, tag='sav:l4')

lc.rect(RP_X, 448, BXR - RP_X, 92, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(RP_X + 14, 470, 'CUMULATIVE 的代价与兜底', 9.5, lc.C_TXT, 'start', True, maxw=280,
        tag='cum:t')
lc.text(RP_X + 14, 490, '中间快照在 collector 侧被单槽压平——每请求', 8.5, '#334155', 'start',
        maxw=290, tag='cum:l1')
lc.text(RP_X + 14, 506, '驻留 O(1) 个对象；但每条快照本身持全量文本', 8.5, '#334155', 'start',
        maxw=290, tag='cum:l2')
lc.text(RP_X + 14, 526, '（滞留字节 O(len)，见单槽邮箱图）', 8.5, lc.C_MUTE, 'start', maxw=290,
        tag='cum:l3')

# ---------------- 终值全等条 ----------------
EQ_Y = 626
lc.rect(MX, EQ_Y, BXR - MX, 56, lc.C_API_F, lc.C_API_S, rx=8, sw=1.4)
lc.text((MX + BXR) / 2, EQ_Y + 24, '三态终值全等："Hello" —— DELTA 拼接 == CUMULATIVE 末快照 == FINAL_ONLY 唯一输出',
        11, lc.C_TXT, 'middle', True, maxw=1100, tag='eq:t')
lc.text((MX + BXR) / 2, EQ_Y + 44, '实测 same_stream_same_texts=true · 三请求同批交错 demux，各自拿到自己的 collector',
        8.8, lc.C_MUTE, 'middle', maxw=1100, tag='eq:s')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = EQ_Y + 86
lx = MX
items = [('solid', '构造出的 RequestOutput'), ('dash', '零构造空位（return None）'),
         ('wide', '盒宽 ∝ token 数（快照膨胀）')]
for kind, name in items:
    if kind == 'solid':
        lc.rect(lx, LEG_Y - 8, 20, 13, lc.C_API_F, lc.C_API_S, rx=4, sw=1.3)
    elif kind == 'dash':
        lc.rect(lx, LEG_Y - 8, 20, 13, '#ffffff', lc.C_FAINT, rx=4, sw=1.1, dash=True)
    else:
        for w_ in (14, 22, 28):
            lc.rect(lx, LEG_Y - 8, w_ - 2, 13, lc.C_API_F, lc.C_API_S, rx=3, sw=1.0)
            lx += w_
        lx -= 4
    lc.text(lx + 26, LEG_Y + 2, name, 9, lc.C_TXT, 'start', maxw=240, tag='leg' + name)
    lx += 26 + lc.tw(name, 9) + 22
lc.text(MX, LEG_Y + 28, '三道闸之第一道 verbatim vllm/v1/engine/output_processor.py:L276-L290 · delta 切片 L388-L423 · '
        '三态计数 host 实测（token 2/2/1 与 2/4/5、puts 3/3/1）· 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch07-fig-tri-state.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
