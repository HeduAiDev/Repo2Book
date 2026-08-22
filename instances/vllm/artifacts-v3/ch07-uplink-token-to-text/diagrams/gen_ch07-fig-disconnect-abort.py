#!/usr/bin/env python3
"""ch07 机制图 9 · 断连反向 abort 三层接力（figure_spec ch07-fig-disconnect-abort，模板 flow）

放大自 L0 的 API 进程带与紫色 ZMQ 边界带的接缝（api_band + zmq_band 交界，『API 进程上行
泳道』的反向出径）——即本章 L2 章图 center 拍片 ⑨ 『断连反向 abort』+ north『出门 · SSE 流』
的机制展开；ABORT 帧跨边界上行回引擎（架构归属回指 L2/L0）。

claim：断连经三层接力变成两跳 abort：路由层竞速取消 handler → generate 捕
CancelledError → hop1 本进程移状态并投 ABORT 终态（实测落在两跳之间、写回外部 id、
还在等的消费者拿到 finished=true 不挂死）→ hop2 ABORT 帧携内部 id 过线停算——
实测 timeline 严格 hop1 先于 hop2、三张表全清。

数字全部取自 figure_spec.numbers（host 实测 trace + pin 锚点）；坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 856
MX = 60
BXR = 1440
C_BODY = '#334155'
AB = lc.C_ABORT

# ---------------- 标题区 ----------------
lc.text(MX, 34, '客人走了谁喊停：三层接力两跳 abort——先本进程收摊，再过线停算',
        16.5, lc.C_TXT, 'start', True, maxw=1020, tag='title')
lc.text(MX, 58, '断连的 CancelledError 在 generate() 里变成 abort(internal=True)：hop1 移状态并投终态收条（解阻塞），'
        'hop2 ABORT 帧过线停算', 10.5, lc.C_MUTE, 'start', maxw=1020, tag='subtitle')
_ch = '放大自 L2 拍片 ⑨ 断连反向 abort · L0：API 进程上行泳道（反向出径）'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_API_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 泳道 A：HTTP 层 ----------------
lc.rect(MX, 108, 340, 308, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(MX + 16, 132, 'HTTP 层（服务面域 · ch38 展开）', 10, lc.C_TXT, 'start', True, maxw=280,
        tag='la:t')
lc.rect(84, 152, 120, 46, '#ffffff', lc.C_MUTE, rx=6, sw=1.2)
lc.text(144, 180, '客户端', 9.5, lc.C_TXT, 'middle', True, maxw=100, tag='cli')
lc.rect(224, 152, 152, 46, '#ffffff', AB, rx=6, sw=1.3, dash=True)
lc.text(300, 172, '✕ 断开', 9.5, AB, 'middle', True, maxw=140, tag='disc')
lc.text(300, 190, 'http.disconnect', 8, lc.C_MUTE, 'middle', maxw=140, tag='disc:sub')
lc.seg(300, 198, 300, 254, AB, 1.8, 'ab')
lc.rect(84, 258, 292, 118, '#ffffff', lc.C_MUTE, rx=7, sw=1.4)
lc.text(98, 280, 'with_cancellation 竞速', 9.5, lc.C_TXT, 'start', True, maxw=260, tag='race:t')
for i, ln in enumerate(['handler ↔ listen_for_disconnect 双任务',
                        'asyncio.wait(FIRST_COMPLETED)',
                        '断连先到 → cancel handler']):
    lc.text(98, 302 + i * 17, ln, 8.2, C_BODY, 'start', maxw=264, tag='race:l' + str(i))
lc.text(98, 366, 'api_utils.py:L77-L94 · api_router.py:L51-L53', 7.5, lc.C_FAINT, 'start',
        maxw=264, tag='race:f')
# cancel 箭头 → 泳道 B 的 generate
lc.parrow([(376, 300), (408, 300), (408, 206), (436, 206)], AB, 1.8, 'ab', dash=True)

# ---------------- 泳道 B：API 进程 ----------------
lc.rect(420, 108, 560, 460, lc.C_API_F, lc.C_API_S, rx=10, sw=2.0)
lc.text(436, 132, 'API 进程', 10, lc.C_API_S, 'start', True, maxw=200, tag='lb:t')
lc.text(964, 132, 'async_llm.py:L608-L616 → L729-L738', 8, lc.C_FAINT, 'end', maxw=300,
        tag='lb:f')
GEN = (440, 152, 520, 110)
lc.rect(*GEN, '#ffffff', lc.C_API_S, rx=7, sw=1.5)
lc.text(GEN[0] + 14, GEN[1] + 22, 'generate() 协程（每请求一个）', 10, lc.C_TXT, 'start', True,
        maxw=480, tag='gen:t')
for i, ln in enumerate(['捕 CancelledError / GeneratorExit（StreamingResponse 取消传导）',
                        '→ await self.abort(q.request_id, internal=True)——q 持内部 id']):
    lc.text(GEN[0] + 14, GEN[1] + 44 + i * 18, ln, 8.3, C_BODY, 'start', maxw=490,
            tag='gen:l' + str(i))
lc.text(GEN[0] + 14, GEN[1] + 96, '注释原话：断连或生成器被回收时，从这里 abort', 7.5,
        lc.C_MUTE, 'start', maxw=490, tag='gen:n')
lc.seg(700, GEN[1] + GEN[3], 700, 296, lc.C_API_S, 2.0, 'dn')
H1 = (440, 300, 520, 130)
lc.rect(*H1, '#ffffff', lc.C_API_S, rx=7, sw=1.5)
lc.text(H1[0] + 14, H1[1] + 22, 'hop1 · 本进程收摊（先）', 10, lc.C_TXT, 'start', True,
        maxw=400, tag='h1:t')
lc.text(H1[0] + 14, H1[1] + 42, 'OutputProcessor.abort_requests：移 request_states + external_map（各 1→0）',
        8.3, C_BODY, 'start', maxw=490, tag='h1:l1')
lc.rect(H1[0] + 14, H1[1] + 56, 460, 44, lc.C_ENG_F, lc.C_ENG_S, rx=6, sw=1.2, dash=True)
lc.text(H1[0] + 26, H1[1] + 73, 'ABORT 终态收条 → 单槽信箱：finish_reason=abort · finished=true',
        8, '#9a3412', 'start', True, maxw=436, tag='h1:r1')
lc.text(H1[0] + 26, H1[1] + 90, 'request_id 写回外部 id（chatcmpl-dis）——还在 await 的消费者立即解阻塞',
        8, '#9a3412', 'start', maxw=436, tag='h1:r2')
lc.text(H1[0] + 14, H1[1] + 118, 'output_processor.py:L494-L515（移状态+投终态）', 7.5,
        lc.C_FAINT, 'start', maxw=490, tag='h1:f')
lc.seg(700, H1[1] + H1[3], 700, 460, lc.C_API_S, 2.0, 'dn')
lc.text(710, 446, '两跳之间 collector 收到收条（put_between_hops=true）', 7.5, lc.C_ENG_S,
        'start', True, maxw=264, tag='h1:mid')
H2 = (440, 464, 520, 84)
lc.rect(*H2, '#ffffff', lc.C_API_S, rx=7, sw=1.5)
lc.text(H2[0] + 14, H2[1] + 22, 'hop2 · 跨进程停算（后）', 10, lc.C_TXT, 'start',
        True, maxw=400, tag='h2:t')
lc.text(H2[0] + 14, 508, 'engine_core.abort_requests_async → ABORT 帧过线', 8.3, C_BODY,
        'start', maxw=490, tag='h2:l1')
lc.text(H2[0] + 14, 526, '帧内容 = [内部 id chatcmpl-dis-b183c2ef] · core_client.py:L1150-L1152',
        7.5, lc.C_MUTE, 'start', maxw=490, tag='h2:l2')

# ---------------- 泳道 C：ZMQ 边界 → 引擎 ----------------
lc.seg(1000, 112, 1000, 566, lc.C_ZMQ_S, 1.4, dash=True)
lc.rect(1004, 108, 436, 92, lc.C_ZMQ_F, lc.C_ZMQ_S, rx=8, sw=1.8)
lc.text(1020, 134, 'ZMQ 边界（ch5）', 10, lc.C_ZMQ_S, 'start', True, maxw=280, tag='zmq:t')
lc.text(1020, 156, "ABORT b'\\x01' 与 ADD 同一条 socket", 8.3, C_BODY, 'start', maxw=300,
        tag='zmq:l1')
lc.text(1020, 174, 'ROUTER ← DEALER · 只带内部 id', 8.3, C_BODY, 'start', maxw=300,
        tag='zmq:l2')
ENG = (1004, 260, 436, 156)
lc.rect(*ENG, lc.C_ENG_F, lc.C_ENG_S, rx=8, sw=1.8)
lc.text(ENG[0] + 14, ENG[1] + 24, 'EngineCore（busy loop）', 10, lc.C_ENG_S, 'start', True,
        maxw=300, tag='eng:t')
lc.text(ENG[0] + 14, ENG[1] + 46, '引擎侧双投递：input_queue + aborts_queue', 8.3, C_BODY,
        'start', maxw=400, tag='eng:l1')
lc.text(ENG[0] + 14, ENG[1] + 64, '（语义 ch5 已讲——收到即从批次摘除）', 8, lc.C_MUTE, 'start',
        maxw=400, tag='eng:l2')
lc.rect(ENG[0] + 14, ENG[1] + 84, 408, 30, '#ffffff', lc.C_ENG_S, rx=5, sw=1.1, dash=True)
lc.text(ENG[0] + 218, ENG[1] + 103, '收到之后的故事 → ch38 服务面', 8, lc.C_ENG_S, 'middle',
        True, maxw=390, tag='eng:hook')
lc.text(ENG[0] + 14, ENG[1] + 140, 'vllm/v1/engine/core.py', 7.5, lc.C_FAINT, 'start',
        maxw=300, tag='eng:f')
# ABORT 红虚线：hop2 → 过 ZMQ 边界 → 引擎底边
lc.parrow([(960, 506), (1220, 506), (1220, ENG[1] + ENG[3] + 2)], AB, 1.8, 'ab', dash=True)
lc.text(1090, 496, 'ABORT 帧 · [内部 id] 过线', 8.5, AB, 'middle', True, maxw=200, tag='ab:l')

# ---------------- 实测时序条 ----------------
TL_Y, TL_H = 588, 98
lc.rect(MX, TL_Y, BXR - MX, TL_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(MX + 16, TL_Y + 22, '实测时序（host 单次运行；t 为单调钟，只用于顺序断言）', 9.5,
        lc.C_TXT, 'start', True, maxw=520, tag='tl:t')
lc.seg(160, TL_Y + 58, 1380, TL_Y + 58, '#e2e8f0', 2.0)
PINS = [(300, '① hop1 · 本进程', '移状态 + 投 ABORT 终态'),
        (700, '② 两跳之间', 'collector 收到收条（外部 id）'),
        (1100, '③ hop2 · 跨进程', 'ABORT 帧携内部 id 过线')]
for px, t1, t2 in PINS:
    lc.ELEMS.append(((px - 5, TL_Y + 53, px + 5, TL_Y + 63),
                     f'<circle cx="{px}" cy="{TL_Y + 58}" r="4.5" fill="{lc.C_API_S}"/>'))
    lc.text(px, TL_Y + 44, t1, 8.5, lc.C_TXT, 'middle', True, maxw=220, tag='pin:' + t1)
    lc.text(px, TL_Y + 80, t2, 8, lc.C_MUTE, 'middle', maxw=240, tag='pins:' + t1)
lc.text(BXR - 16, TL_Y + 22, '断言：hop2 开始 ≥ hop1 完成 · request_states 1→0 · external_map 1→0 · 已流出 "A"×1',
        8.5, lc.C_MUTE, 'end', maxw=560, tag='tl:asm')

# ---------------- 异步窗口注 ----------------
AW_Y = 702
lc.rect(MX, AW_Y, BXR - MX, 54, lc.C_ENG_F, lc.C_ENG_S, rx=8, sw=1.2, dash=True)
lc.text(MX + 16, AW_Y + 22, '喊停是异步的：ABORT 帧在路上时，引擎正在做的这一步还会做完——废 token 有界但不为零',
        9.5, '#9a3412', 'start', True, maxw=1100, tag='aw:t')
lc.text(MX + 16, AW_Y + 42, '（引擎收到之后如何从批次摘除 → ch38 服务面展开；stop-string 命中的反向 abort 走同一条 hop2 通路）',
        8.5, lc.C_MUTE, 'start', maxw=1100, tag='aw:s')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = 788
lx = MX
items = [('laneB', 'API 进程带'), ('zmq', 'ZMQ 边界带'), ('eng', 'EngineCore 带'),
         ('ab', 'cancel / ABORT（红虚线）'), ('receipt', '虚线暖框 = 终态收条')]
for kind, name in items:
    if kind == 'laneB':
        lc.rect(lx, LEG_Y - 8, 20, 13, lc.C_API_F, lc.C_API_S, rx=4, sw=1.4)
    elif kind == 'zmq':
        lc.rect(lx, LEG_Y - 8, 20, 13, lc.C_ZMQ_F, lc.C_ZMQ_S, rx=4, sw=1.4)
    elif kind == 'eng':
        lc.rect(lx, LEG_Y - 8, 20, 13, lc.C_ENG_F, lc.C_ENG_S, rx=4, sw=1.4)
    elif kind == 'ab':
        lc.seg(lx, LEG_Y - 2, lx + 22, LEG_Y - 2, AB, 1.5, dash=True)
    else:
        lc.rect(lx, LEG_Y - 9, 20, 15, lc.C_ENG_F, lc.C_ENG_S, rx=4, sw=1.1, dash=True)
    lc.text(lx + 28, LEG_Y + 2, name, 9, lc.C_TXT, 'start', maxw=210, tag='leg' + name)
    lx += 28 + lc.tw(name, 9) + 20
lc.text(MX, LEG_Y + 28, '两跳 verbatim vllm/v1/engine/async_llm.py:L608-L616 → L729-L738 · abort_requests vllm/v1/engine/output_processor.py:L494-L515 · '
        '路由层锚点 vllm/entrypoints/serve/utils/api_utils.py:L37-L94 · 时序/清表/收条 host 实测 · 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch07-fig-disconnect-abort.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
