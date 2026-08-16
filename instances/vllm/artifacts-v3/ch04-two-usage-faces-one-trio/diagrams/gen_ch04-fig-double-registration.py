#!/usr/bin/env python3
"""ch04 机制图 · 双登记扇出/汇合对称图（explainer m3 figure_spec ch04-fig-double-registration）

放大自 L2 章图 center 组件「add_request→_add_request 双登记」（L0 区域 = API 进程-ZMQ
带交界）——FIGURE-SYSTEM §3.3 正文机制图：架构归属回指 L2/L0，不另立架构画法。
（回指小片连接符用 ASCII 连字符 -：↔ U+2194 在 cairosvg 宿主字体链缺字形、渲染成豆腐块，
盲审 FAIL 已证；- 同片「站 8-9」处已验证可渲染。）

claim：add_request 在一个函数里把同一请求写两本账——先本进程建 RequestState（回程
还原上下文的对账表）、后跨进程发 EngineCoreRequest（带 client_index/external_req_id
两枚章随请求过线）；回程消息只有内部 id、查表命中才还原成 RequestOutput——扇出与
汇合以同一条 ZMQ 过线为对称轴。

构图：左泳道 = API 进程（AsyncLLM），右泳道 = 引擎进程，中间窄竖带 = ZMQ 过线；
上半 = ADD 出向扇出，下半 = EngineCoreOutput 回向汇合。防御分支「查不到=已 abort」
画成汇合侧虚线旁路。数字全部取自 explainer figure_spec.numbers（traces
m3_double_registration.json）。
坐标由常量/循环计算；文本全 esc()；配色走 l0_common 语系（同源强制）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

# ---------------- 画布 ----------------
W, H = 1420, 948
MX = 60

# 泳道几何（左 API / 中 ZMQ 竖带 / 右引擎）
LX0, LX1 = 70, 830            # 左泳道（API 进程）
ZX0, ZX1 = 860, 980           # ZMQ 竖带
RX0, RX1 = 1010, 1390         # 右泳道（引擎进程）
TOPY, BOTY = 100, 858         # 三带统一上下缘
MIDX = (ZX0 + ZX1) // 2       # 对称轴 x = 920


def chip(x_right, y, label, color):
    """右上角回指小片（虚线）。"""
    w = lc.tw(label, 9.5, True) + 14
    x = x_right - w
    lc.rect(x, y, w, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
    lc.text(x + w / 2, y + 14.5, label, 9.5, color, 'middle', True,
            maxw=w - 4, tag='chip:' + label[:12])


def station(x, y, n):
    """站号徽标（本章 L2 站号账本）：小胶囊 + 数字。"""
    lc.rect(x, y, 22, 16, lc.C_BADGE_F, lc.C_ENG_S, rx=7, sw=1.1)
    lc.text(x + 11, y + 12, str(n), 9, lc.C_ENG_S, 'middle', True, maxw=18,
            tag=f'st{n}')


def content(x, y, w, h, title, lines, stroke, fill='#ffffff', sw=1.6, dash=False,
            tfs=11.5, badge=0, tag=''):
    """泳道内内容框：标题 + 若干行；行 3 元组 (text, 样式, fs)。样式 n/m/a=正常/灰/强调。"""
    lc.rect(x, y, w, h, fill, stroke, rx=7, sw=sw, dash=dash)
    tx = x + 14 + (30 if badge else 0)
    if badge:
        station(x + 12, y + 12, badge)
    lc.text(tx, y + 21, title, tfs, lc.C_TXT, 'start', True,
            maxw=x + w - 14 - tx, tag=(tag or title))
    for i, (ln, sty, fs) in enumerate(lines):
        col = {'n': '#334155', 'm': lc.C_MUTE, 'a': stroke, 'w': lc.C_ABORT}[sty]
        bold = sty == 'a'
        lc.text(x + 14, y + 41 + i * 17, ln, fs, col, 'start', bold=bold,
                maxw=w - 26, tag=(tag or title) + ':' + ln[:10])


# ---------------- 标题区 ----------------
lc.text(MX, 36, '双登记：一个请求写两本账，回程才找得到家', 17, lc.C_TXT, 'start', True)
lc.text(MX, 60,
        '_add_request 两行有先后——先本进程建 RequestState（回程还原上下文的对账表），'
        '后跨进程发 EngineCoreRequest（两枚章随请求过线）；回程消息只带内部 id，查表命中才还原成 RequestOutput',
        10.5, lc.C_MUTE, 'start', maxw=1000, tag='subtitle')
chip(1360, 12, '放大自 L2 站 8-9「双登记」· L0 区域：API 进程-ZMQ 交界', lc.C_API_S)

# ---------------- 三带外框 ----------------
lc.rect(LX0, TOPY, LX1 - LX0, BOTY - TOPY, lc.C_API_F, lc.C_API_S, rx=12, sw=2.4)
lc.rect(ZX0, TOPY, ZX1 - ZX0, BOTY - TOPY, lc.C_ZMQ_F, lc.C_ZMQ_S, rx=12, sw=2.2)
lc.rect(RX0, TOPY, RX1 - RX0, BOTY - TOPY, lc.C_ENG_F, lc.C_ENG_S, rx=12, sw=2.4)
lc.text(LX0 + 16, TOPY + 24, 'API 进程 · AsyncLLM（在线面）', 13, lc.C_API_S,
        'start', True, maxw=380, tag='lane-api')
lc.text(LX1 - 14, TOPY + 24, '离线面同构两行 · 不传 queue（llm_engine.py:L272-L277）', 9,
        lc.C_MUTE, 'end', maxw=330, tag='lane-api-sub')
lc.text(MIDX, TOPY + 22, '进程边界', 12, lc.C_ZMQ_S, 'middle', True, maxw=100, tag='zmq-t')
lc.text(MIDX, TOPY + 40, 'ZMQ + msgpack', 9.5, lc.C_ZMQ_S, 'middle', True, maxw=110,
        tag='zmq-s')
# 对称轴注：上半扇出 / 下半汇合
AXMID = (TOPY + BOTY) / 2
lc.seg(ZX0, AXMID, ZX1, AXMID, lc.C_ZMQ_S, 1.0, dash=True)
lc.text(MIDX, AXMID - 8, '上半：出向扇出', 8.5, lc.C_ZMQ_S, 'middle', maxw=112, tag='ax1')
lc.text(MIDX, AXMID + 14, '下半：回向汇合', 8.5, lc.C_ZMQ_S, 'middle', maxw=112, tag='ax2')
lc.text(MIDX, BOTY - 12, '线格式细节 → ch5', 8.5, lc.C_MUTE, 'middle', maxw=110,
        tag='zmq-ch5')
lc.text(RX0 + 16, TOPY + 24, '引擎进程 · EngineCore', 13, lc.C_ENG_S, 'start', True,
        maxw=280, tag='lane-eng')
lc.text(RX1 - 14, TOPY + 24, '不感知使用面', 9, lc.C_MUTE, 'end', maxw=140, tag='lane-eng-sub')

# ================= 左泳道（API 进程） =================
ILX, ILW = LX0 + 20, LX1 - LX0 - 40      # 内框统一 x/宽 = 90..810

# L1 进门三步（站 5-7）
L1Y, L1H = 150, 100
content(ILX, L1Y, ILW, L1H, 'add_request 进门三步（async_llm.py:L283-L418）', [], lc.C_API_S)
rows1 = [
    (5, 'process_inputs：渲染 → EngineCoreRequest（构造点 input_processor.py:L379-L394）'),
    (6, 'assign_request_id 双轨：外部 "chat-abc" + 内部 "chat-abc-8187f9a7"（8 位随机 hex 后缀）'),
    (7, '建 RequestOutputCollector 信箱（DELTA · 离线面无信箱 queue=None）'),
]
for i, (n, ln) in enumerate(rows1):
    yy = L1Y + 41 + i * 18
    station(ILX + 14, yy - 11, n)
    lc.text(ILX + 44, yy, ln, 9, '#334155', 'start', maxw=ILW - 58,
            tag=f'l1r{n}')

# L2 双登记两行（站 8-9）——主角框
L2Y, L2H = 274, 118
content(ILX, L2Y, ILW, L2H,
        '_add_request 双登记两行（L420-L435）——顺序即纪律', [], lc.C_API_S, sw=2.4)
rows2 = [
    (8, '先本进程：output_processor.add_request → RequestState 进 request_states + 外→内 id 映射（L428，注释 this process）', 9.2),
    (9, '后跨进程：engine_core.add_request_async → 盖两枚章 + msgpack 过线（L431，注释 separate process）', 9.2),
]
for i, (n, ln, fs) in enumerate(rows2):
    yy = L2Y + 44 + i * 24
    station(ILX + 14, yy - 11, n)
    lc.text(ILX + 44, yy, ln, fs, '#334155', 'start', maxw=ILW - 58, tag=f'l2r{n}')
lc.text(ILX + 14, L2Y + L2H - 12, '反序则回程可能先于建表到达 → 活请求被当废件丢弃', 8.5,
        lc.C_ABORT, 'start', maxw=ILW - 28, tag='l2warn')

# L3 账本①
L3Y, L3H = 424, 108
content(ILX, L3Y, ILW, L3H, '账本① · request_states（本进程）——回程还原的全部依据', [
    ('"chat-abc-8187f9a7" → RequestState{ external="chat-abc" · 信箱(DELTA) · prompt=[1,2,3] · max_tokens=2 · detokenizer }', 'n', 8.6),
    ('回程消息本身只有内部 id + token + finish_reason——上下文全靠查这张表', 'm', 9),
    ('计数：登记后 1 条 → 终拍清账 0 条（abort 同样清账）', 'a', 9.2),
], lc.C_API_S, tag='ledger1')

# L4 回程对账（站 11）
L4Y, L4H = 558, 128
content(ILX, L4Y, ILW, L4H, '回程对账 · process_outputs——查表命中才还原', [], lc.C_API_S,
        badge=11, tag='l4')
lc.text(ILX + 14, L4Y + 44, '按内部 id 查账本① → 命中：组装 RequestOutput，request_id 换回外部 "chat-abc" → put 信箱',
        9, '#334155', 'start', maxw=ILW - 26, tag='l4a')
lc.text(ILX + 14, L4Y + 63, 'generate yield：token [101]（轮3）→ [102] + finish_reason=length（轮4）· finished 弹表项',
        9, '#334155', 'start', maxw=ILW - 26, tag='l4b')
lc.text(ILX + 14, L4Y + 82, '离线面：queue=None → 收进 list 由 step() 返回（同一个函数两种吃法）',
        9, lc.C_MUTE, 'start', maxw=ILW - 26, tag='l4c')
lc.text(ILX + 14, L4Y + 101, '门牌：output_processor.py:L619-L684 · 全批遍历只许这一处', 8.5,
        lc.C_FAINT, 'start', maxw=ILW - 26, tag='l4d')

# 防御分支旁路（虚线红）
BPY, BPH = 712, 52
lc.rect(ILX, BPY, 430, BPH, 'none', lc.C_ABORT, rx=7, sw=1.4, dash=True)
lc.text(ILX + 14, BPY + 21, '防御分支：查表落空 = 已被 abort/finish 移除 → 跳过，不报错', 9,
        lc.C_ABORT, 'start', maxw=404, tag='bp1')
lc.text(ILX + 14, BPY + 38, '（output_processor.py:L620-L624——不存在第三种状态）', 8.5,
        lc.C_ABORT, 'start', maxw=404, tag='bp2')
lc.seg(200, L4Y + L4H, 200, BPY, lc.C_ABORT, 1.4, 'ab', dash=True)

# 左泳道内部纵向箭头
lc.seg(450, L1Y + L1H, 450, L2Y, lc.C_API_S, 1.8, 'dn')
lc.seg(300, L2Y + L2H, 300, L3Y, lc.C_API_S, 1.8, 'dn')
lc.text(312, L3Y - 6, '先写表', 8.5, lc.C_API_S, 'start', maxw=80, tag='w-l23')
lc.seg(300, L3Y + L3H, 300, L4Y, lc.C_API_S, 1.8, 'dn')
lc.text(312, L4Y - 6, '查表', 8.5, lc.C_API_S, 'start', maxw=80, tag='w-l34')

# 终态注
lc.text((LX0 + LX1) / 2, BOTY - 16, '终拍后：账本① 0 条 · 账本② 0 条 · generate 退出并 close 信箱',
        9, lc.C_MUTE, 'middle', maxw=ILW, tag='final')

# ================= 右泳道（引擎进程） =================
IRX, IRW = RX0 + 20, RX1 - RX0 - 40     # 1030..1370

# R1 input_queue
R1Y, R1H = 250, 112
lc.rect(IRX, R1Y, IRW, R1H, '#ffffff', lc.C_ENG_S, rx=7, sw=1.6)
lc.text(IRX + 14, R1Y + 21, 'input_queue', 11.5, lc.C_TXT, 'start', True, maxw=IRW - 28,
        tag='iq')
cw, cg = 22, 10
bx = IRX + (IRW - (3 * cw + 2 * cg)) / 2
for i in range(3):
    lc.rect(bx + i * (cw + cg), R1Y + 34, cw, 24, '#cbd5e1', lc.C_MUTE, rx=2, sw=1.0)
lc.text(IRX + IRW / 2, R1Y + 74, '轮1 中间态：ADD 已过界、躺在这里等下一拍', 8.8,
        '#334155', 'middle', maxw=IRW - 16, tag='iq-s1')
lc.text(IRX + IRW / 2, R1Y + 92, 'queue.Queue · 保序（busy loop 每拍前排空）', 8.5,
        lc.C_MUTE, 'middle', maxw=IRW - 16, tag='iq-s2')

# R2 账本②
R2Y, R2H = 396, 122
content(IRX, R2Y, IRW, R2H, '账本② · requests（引擎侧）', [
    ('"chat-abc-8187f9a7" → Request', 'n', 9),
    ('章：client_index=0 · external_req_id="chat-abc"', 'n', 8.8),
    ('一拍排空 input_queue 落地（core.py:L1378-L1389）', 'm', 8.8),
    ('计数：轮1 0 条 · 轮2 1 条 · 轮4 0 条', 'a', 9.2),
], lc.C_ENG_S, tag='ledger2')

# R3 输出线程 sockets[client_index]（站 10）
R3Y, R3H = 548, 118
content(IRX, R3Y, IRW, R3H, '输出线程 · sockets[client_index]', [
    ('按章查表：client_index=0 → sockets[0]', 'n', 9),
    ('选回程 PUSH socket（core.py:L1804）', 'n', 8.8),
    ('每前端一条 socket · 多前端回程路由 → ch34 预告', 'm', 8.6),
], lc.C_ENG_S, badge=10, tfs=10.5, tag='sock')

# 右泳道内部纵向箭头
lc.seg(1200, R1Y + R1H, 1200, R2Y, lc.C_ENG_S, 1.8, 'dn')
lc.text(1212, R2Y - 6, '轮2 · 一拍排空', 8.5, lc.C_ENG_S, 'start', maxw=140, tag='w-r12')
lc.seg(1200, R2Y + R2H, 1200, R3Y, lc.C_ENG_S, 1.8, 'dn')
lc.text(1212, R3Y - 6, '产出 (client_index=0, token)', 8.5, lc.C_ENG_S, 'start',
        maxw=150, tag='w-r23')

# ================= ZMQ 竖带：两条过线 =================
# 上半：ADD 出向（API → 引擎）
ADDY = 320
lc.seg(LX1 - 20 + 20, ADDY, IRX, ADDY, lc.C_API_S, 2.2, 'dn')   # (830,320)→(1030,320)
lc.text(MIDX, ADDY - 38, 'ADD · EngineCoreRequest 过线', 9.2, lc.C_API_S, 'middle',
        True, maxw=200, tag='add-l1')
lc.text(MIDX, ADDY - 23, '两枚章：client_index=0', 8.8, lc.C_API_S, 'middle',
        True, maxw=180, tag='add-l2a')
lc.text(MIDX, ADDY - 9, 'external_req_id="chat-abc"', 8.8, lc.C_API_S, 'middle',
        maxw=180, tag='add-l2b')
lc.text(MIDX, ADDY + 16, 'msgpack 编码 · 立即发出，不等循环 tick', 8.5, lc.C_MUTE,
        'middle', maxw=210, tag='add-l3')

# 下半：EngineCoreOutput 回向（引擎 → API）
RETY = 620
lc.seg(IRX, RETY, LX1, RETY, lc.C_ENG_S, 2.2, 'up')             # (1030,620)→(830,620)
lc.text(MIDX, RETY - 24, 'EngineCoreOutput 回程 · 只带内部 id', 9.2, lc.C_ENG_S,
        'middle', True, maxw=200, tag='ret-l1')
lc.text(MIDX, RETY - 8, '"chat-abc-8187f9a7" + token [101]→[102]', 8.8, lc.C_ENG_S,
        'middle', maxw=210, tag='ret-l2')
lc.text(MIDX, RETY + 16, '+ finish_reason=length（轮4 终帧）', 8.5, lc.C_MUTE,
        'middle', maxw=200, tag='ret-l3')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = 880
items = [
    ('swatch', lc.C_API_S, 'API 进程'),
    ('swatch', lc.C_ZMQ_S, '进程边界（ZMQ + msgpack）'),
    ('swatch', lc.C_ENG_S, '引擎进程'),
    ('dash', lc.C_ABORT, '防御分支'),
    ('arrow', lc.C_API_S, 'dn', '请求下行'),
    ('arrow', lc.C_ENG_S, 'up', '输出上行'),
]
lx = MX
for it in items:
    if it[0] == 'swatch':
        lc.rect(lx, LEG_Y - 9, 16, 11, '#ffffff', it[1], rx=3, sw=1.6)
        name = it[2]
    elif it[0] == 'dash':
        lc.seg(lx + 2, LEG_Y - 3, lx + 32, LEG_Y - 3, it[1], 1.6, dash=True)
        name = it[2]
    else:
        lc.seg(lx + 2, LEG_Y - 3, lx + 32, LEG_Y - 3, it[1], 2.0, it[2])
        name = it[3]
    lc.text(lx + 40, LEG_Y + 1, name, 9.5, lc.C_TXT, 'start', maxw=200,
            tag='leg:' + name[:8])
    lx += 40 + lc.tw(name, 9.5) + 22
station(lx + 4, LEG_Y - 8, 8)
lc.text(lx + 34, LEG_Y + 1, '= 本章站号（请求流经顺序）', 9.5, lc.C_TXT, 'start',
        maxw=220, tag='leg-st')
lc.text(MX, LEG_Y + 26, '框内灰字 = 规范源码路径 · 行号基线 vLLM v0.27.1 · 站号与本章 L2 站号账本一致',
        9, lc.C_MUTE, 'start', maxw=1300, tag='footer')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch04-fig-double-registration.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
