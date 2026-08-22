#!/usr/bin/env python3
"""ch07 机制图 8 · 单槽邮箱 RequestOutputCollector（figure_spec ch07-fig-single-slot-mailbox，模板 state-machine）

放大自 L0 蓝色 API 进程带（api_band · 本章 l0_zoom『API 进程上行泳道』）的收发工位——
即本章 L2 章图 center 拍片 ⑦ 『单槽邮箱』+ south『why · 为什么不是 asyncio.Queue』注的
机制展开。架构归属回指 L2/L0（FIGURE-SYSTEM §3.3）。

claim：RequestOutputCollector 是每请求恰一个格子的信箱：格子空→put 占格摇铃；格子占→
DELTA 原地并信（实测 4 次投递后槽内仍 1 个对象、一次取走得全部并集 "Hello! world"）/
CUMULATIVE 换最新快照；Exception 无条件抢格——任何积压驻留恒 1 个对象，与 asyncio.Queue
的排队长龙相对。

数字全部取自 figure_spec.numbers（host 实测 trace + pin 锚点）；坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 910
MX = 60
BXR = 1440
C_BODY = '#334155'
ERR = lc.C_ABORT


# ---------------- 标题区 ----------------
lc.text(MX, 34, '一格信箱：格子空就放信摇铃，格子占就并信——任何积压驻留恒 1 个对象',
        16.5, lc.C_TXT, 'start', True, maxw=1020, tag='title')
lc.text(MX, 58, 'RequestOutputCollector：每请求恰一个格子（刻意不用 asyncio.Queue——慢读者的积压不该占内存）',
        10.5, lc.C_MUTE, 'start', maxw=980, tag='subtitle')
_ch = '放大自 L2 拍片 ⑦ 单槽邮箱 · L0：API 进程上行泳道'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_API_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 状态机 ----------------
S1 = (100, 560, 250, 100)      # 空
S2 = (560, 140, 270, 112)      # 占·输出
S3 = (560, 560, 270, 100)      # 占·异常

lc.rect(*S2, '#ffffff', lc.C_API_S, rx=9, sw=1.8)
lc.text(S2[0] + 14, S2[1] + 26, '占 · 1 个 RequestOutput', 11, lc.C_TXT, 'start', True,
        maxw=S2[2] - 28, tag='s2:t')
lc.text(S2[0] + 14, S2[1] + 48, 'Event 已置（摇铃）', 9, lc.C_API_S, 'start', True,
        maxw=S2[2] - 28, tag='s2:l1')
lc.text(S2[0] + 14, S2[1] + 68, '滞留对象可被原地合并', 8.5, lc.C_MUTE, 'start', maxw=S2[2] - 28,
        tag='s2:l2')
lc.text(S2[0] + 14, S2[1] + 92, 'output_processor.py:L45-L96', 8, lc.C_FAINT, 'start',
        maxw=S2[2] - 28, tag='s2:f')

lc.rect(*S1, '#ffffff', lc.C_API_S, rx=9, sw=1.8)
lc.text(S1[0] + 14, S1[1] + 26, '空', 11, lc.C_TXT, 'start', True, maxw=S1[2] - 28, tag='s1:t')
lc.text(S1[0] + 14, S1[1] + 48, '槽 = None · Event 未置', 9, C_BODY, 'start', maxw=S1[2] - 28,
        tag='s1:l1')
lc.text(S1[0] + 14, S1[1] + 68, '消费者 await ready.wait()', 8.5, lc.C_MUTE, 'start',
        maxw=S1[2] - 28, tag='s1:l2')

lc.rect(*S3, '#ffffff', ERR, rx=9, sw=1.8)
lc.text(S3[0] + 14, S3[1] + 26, '占 · Exception', 11, ERR, 'start', True, maxw=S3[2] - 28,
        tag='s3:t')
lc.text(S3[0] + 14, S3[1] + 48, '错误无条件抢格（空槽来 Exception 同样直接占）', 8.5, C_BODY,
        'start', maxw=S3[2] - 24, tag='s3:l1')
lc.text(S3[0] + 14, S3[1] + 68, 'get() 抛出，滞留输出被丢弃——错误优先的取舍', 8.5, C_BODY,
        'start', maxw=S3[2] - 24, tag='s3:l2')

# put(正常)：空 → 占
lc.parrow([(225, S1[1]), (225, 196), (S2[0] - 2, 196)], lc.C_API_S, 1.8, 'dn')
lc.text(240, 186, 'put(正常输出)：占格 + set Event（摇铃）', 8.5, lc.C_API_S, 'start', True,
        maxw=310, tag='a:put')
# get：占 → 空
lc.parrow([(S2[0], 238), (430, 238), (430, 612), (S1[0] + S1[2] + 2, 612)], lc.C_API_S, 1.8, 'std')
lc.text(438, 330, 'get() / get_nowait()', 8.5, lc.C_API_S, 'start', True, maxw=180, tag='a:get')
lc.text(438, 346, '取走 + 清槽 + clear Event', 8.5, lc.C_MUTE, 'start', maxw=180, tag='a:get2')
# 自环：占 → 占（add 原地合并）
lc.parrow([(630, S2[1]), (630, 104), (770, 104), (770, S2[1])], lc.C_API_S, 1.8, 'std')
lc.text(700, 84, 'put + RequestOutput → output.add() 原地合并（槽仍 1 个）', 8.5, lc.C_API_S,
        'middle', True, maxw=380, tag='a:loop')
lc.text(700, 98, 'DELTA 续写增量 · CUMULATIVE 换最新快照 · 不同 index append', 8, lc.C_MUTE,
        'middle', maxw=380, tag='a:loop2')
# put(Exception)：占 → 占（异常）
lc.seg(700, S2[1] + S2[3], 700, S3[1] - 2, ERR, 1.8, 'std')
lc.text(708, 340, 'put(RuntimeError)', 8.5, ERR, 'start', True, maxw=120, tag='a:exc')
lc.text(708, 356, '无条件抢格：顶掉滞留', 8.5, C_BODY, 'start', maxw=130, tag='a:exc2')
# get 抛错：占（异常）→ 空
lc.seg(S3[0], 640, S1[0] + S1[2] + 2, 640, ERR, 1.8, 'std')
lc.text(456, 662, 'get() → raise（错误抛给消费者）', 8.5, ERR, 'middle', maxw=200, tag='a:raise')

# ---------------- 右栏：实测面板 ----------------
RP_X, RP_W = 900, BXR - 900
lc.rect(RP_X, 108, RP_W, 330, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(RP_X + 14, 132, '实测 · DELTA 积压：4 次投递无人取', 10.5, lc.C_TXT, 'start', True,
        maxw=420, tag='ev:t')
PUTS = ['put "He"（token [1]）', 'put "llo"（token [2,3]）', 'put "!"（token [4]）',
        'put " world"（token [5]）']
for i, p in enumerate(PUTS):
    y = 158 + i * 24
    lc.rect(RP_X + 14, y - 13, 250, 20, lc.C_API_F, lc.C_API_S, rx=4, sw=1.0)
    lc.text(RP_X + 22, y, p, 8.5, C_BODY, 'start', maxw=236, tag='ev:p' + str(i))
lc.text(RP_X + RP_W - 16, 194, '4 次 put →', 8.5, lc.C_MUTE, 'end', maxw=90, tag='ev:m')
lc.text(RP_X + RP_W - 16, 220, '槽内仍 1 个对象', 8.5, lc.C_API_S, 'end', True, maxw=140, tag='ev:m2')
lc.text(RP_X + RP_W - 16, 236, '（合并不开第二格）', 8, lc.C_MUTE, 'end', maxw=140, tag='ev:m3')
lc.parrow([(1200, 210), (1200, 272)], lc.C_API_S, 1.6, 'dn')
lc.rect(RP_X + 14, 276, RP_W - 28, 58, lc.C_API_F, lc.C_API_S, rx=7, sw=1.4)
lc.text(RP_X + 26, 298, '一次 get 取走全部并集："Hello! world"（token [1,2,3,4,5]）', 9.5,
        lc.C_TXT, 'start', True, maxw=RP_W - 52, tag='ev:r1')
lc.text(RP_X + 26, 320, '取出即清槽清 Event——再 get_nowait() 为 None · 消费侧内存 O(1)', 8.5,
        lc.C_MUTE, 'start', maxw=RP_W - 52, tag='ev:r2')
lc.text(RP_X + 14, 362, 'index 配对（n>1）：outputs=[(0,"Hello!"),(1,"world")]——不同 index append、'
        '同 index 合并，互不覆盖', 8.5, C_BODY, 'start', maxw=RP_W - 28, tag='ev:idx')
lc.text(RP_X + 14, 384, 'CUMULATIVE：换新非拼接——幸存快照 "Hello"（token [1,2,3]），旧快照被替换',
        8.5, C_BODY, 'start', maxw=RP_W - 28, tag='ev:cum')
lc.text(RP_X + 14, 412, 'vllm/outputs.py:L152-L181（add 配对）', 8, lc.C_FAINT, 'start',
        maxw=RP_W - 28, tag='ev:f')

# 消费拼法
SP_Y = 456
lc.rect(RP_X, SP_Y, RP_W, 108, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(RP_X + 14, SP_Y + 22, '消费拼法（generate 每轮）', 9.5, lc.C_TXT, 'start', True,
        maxw=300, tag='sp:t')
lc.rect(RP_X + 14, SP_Y + 36, 330, 24, '#f8fafc', lc.C_MUTE, rx=4, sw=1.0)
lc.text(RP_X + 24, SP_Y + 52, 'out = q.get_nowait() or await q.get()', 9.5, lc.C_TXT, 'start',
        True, maxw=310, tag='sp:code')
lc.text(RP_X + 14, SP_Y + 80, '非阻塞 drain 优先——avoids task switching under load', 8.5,
        C_BODY, 'start', maxw=RP_W - 28, tag='sp:l1')
lc.text(RP_X + 14, SP_Y + 98, '地雷：依赖 RequestOutput 恒真（未定义 __bool__/__len__）', 8.5,
        ERR, 'start', maxw=RP_W - 28, tag='sp:l2')

# ---------------- 底部对照：无界队列 vs 单槽 ----------------
CT_Y, CT_H = 700, 118
lc.rect(MX, CT_Y, 690, CT_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(MX + 14, CT_Y + 22, 'v0：每请求一条无界 asyncio.Queue', 9.5, lc.C_TXT, 'start', True,
        maxw=400, tag='ct1:t')
qw, qh = 34, 30
for i in range(8):
    x = MX + 14 + i * (qw + 6)
    lc.rect(x, CT_Y + 36, qw, qh + (i * 4), '#e2e8f0', lc.C_MUTE, rx=3, sw=1.0)
lc.text(412, CT_Y + 52, '排队长龙越积越长', 8.5, C_BODY, 'start', maxw=180, tag='ct1:l1')
lc.text(412, CT_Y + 70, 'CUMULATIVE 每条快照持全量文本', 8.5, C_BODY, 'start', maxw=240,
        tag='ct1:l2')
lc.text(412, CT_Y + 88, '——消费者滞后时滞留字节 O(len²) 级', 8.5, ERR, 'start', maxw=240,
        tag='ct1:l3')
lc.rect(770, CT_Y, BXR - 770, CT_H, lc.C_API_F, lc.C_API_S, rx=8, sw=1.5)
lc.text(784, CT_Y + 22, 'v0.27.1：一格信箱（#15156 生产侧单槽）', 9.5, lc.C_TXT, 'start', True,
        maxw=400, tag='ct2:t')
lc.rect(784, CT_Y + 36, 120, 40, '#ffffff', lc.C_API_S, rx=6, sw=1.6)
lc.text(844, CT_Y + 60, '恒 1 个', 10, lc.C_TXT, 'middle', True, maxw=110, tag='ct2:box')
lc.text(920, CT_Y + 52, '任何积压驻留恒 1 个对象（O(1)）', 8.5, C_BODY, 'start', maxw=260,
        tag='ct2:l1')
lc.text(920, CT_Y + 70, '慢读者无论积压多久，内存不涨', 8.5, C_BODY, 'start', maxw=260,
        tag='ct2:l2')
lc.text(920, CT_Y + 88, '#12298（消费侧合并）→ #15156 的两步演进', 8, lc.C_MUTE, 'start',
        maxw=260, tag='ct2:l3')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = CT_Y + CT_H + 34
lx = MX
items = [('state', '信箱状态'), ('err', '错误路径（Exception 抢格）'),
         ('loop', '自环 = 原地合并（不新增格子）')]
for kind, name in items:
    if kind == 'state':
        lc.rect(lx, LEG_Y - 8, 20, 13, '#ffffff', lc.C_API_S, rx=4, sw=1.6)
    elif kind == 'err':
        lc.rect(lx, LEG_Y - 8, 20, 13, '#ffffff', ERR, rx=4, sw=1.6)
    else:
        lc.seg(lx, LEG_Y - 2, lx + 24, LEG_Y - 2, lc.C_API_S, 1.6, 'std')
    lc.text(lx + 30, LEG_Y + 2, name, 9, lc.C_TXT, 'start', maxw=280, tag='leg' + name)
    lx += 30 + lc.tw(name, 9) + 22
lc.text(MX, LEG_Y + 28, 'put 三分支 verbatim vllm/v1/engine/output_processor.py:L62-L96 · 消费循环 vllm/v1/engine/async_llm.py:L596-L606 · '
        '积压/配对/替换/抢格 host 实测 · 行号基线 vLLM v0.27.1', 9, lc.C_FAINT, 'start',
        maxw=BXR - MX, tag='foot1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch07-fig-single-slot-mailbox.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
