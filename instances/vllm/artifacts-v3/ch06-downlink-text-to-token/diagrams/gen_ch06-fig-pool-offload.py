#!/usr/bin/env python3
"""ch06 机制图 2 · 线程池卸载与 add_request 分流（explainer figure_spec ch06-fig-pool-offload，模板 swimlane）

放大自 L0 蓝色 API 进程带（api_band）的『线程剖面』——即本章 L2 章图 center 拍片 ④
add_request 分流 + south『why · 不许阻塞事件循环』注的机制展开：三条泳道 =
事件循环线程 / renderer tokenize 池 / mm 单工池，分流点在 add_request。
架构归属回指 L2/L0（FIGURE-SYSTEM §3.3）。

claim：add_request 按输入形态分流——已渲染 EngineInput（dict 带 'type'）在事件循环线程
同步快路径（零 tokenizer 调用），raw prompt 下 renderer 线程池跑阻塞的 tokenize/mm
预处理，事件循环全程心跳不停；两个池分工固定：tokenize 池 renderer_num_workers 工
（默认 1）、mm 池恒单工（#38418 保 P0/P1 顺序）。

R3（盲审回修）：泳道标签从「泳道内左上、首个内容框顶上方 4px」改为骑跨泳道顶边的
徽章式标签（白底板后画=切断虚线框线），放泳道顶边右段——左段被 DEC→tokenize 池 /
池→mm 池两条 x=202 下穿箭头占据，标签带放左必被箭头拦腰穿字；各泳道内首个内容框
顶同步下移 FBD（≥一行字高）让出标签带。原版三处标签被内容框顶边切字 = rect-rect
盲区（lint_diagram_geometry 照不到），盲审抓出后本版根治。

R4（盲审回修）：橙色下穿箭头的流标签『带图的路』原 baseline=泳道2 底边线上
（TOK 底+16 恰等于 L2_Y+L2_H=494）——蓝-灰虚线拦腰穿四字（删除线观感）。下移进
泳道2 底边（494）与泳道3 顶边（516）之间的 22px 净空带中央（baseline=带中点+3=508，
字带 [501.2,510]，两侧净空 7.2px/6px、两条虚线边线均不穿字）；标签仍贴橙色箭头
右侧、随箭头跨过泳道边界，语义不变。

数字全部取自 figure_spec.numbers（host 实测 trace + pin 锚点）；线程 id 为运行时值
不进画面，只画归属关系与布尔判定；坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 1020
MX = 60
BXR = 1440
C_BODY = '#334155'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '一名柜员与两个后厨班组：add_request 分流，事件循环全程心跳不停', 16.5,
        lc.C_TXT, 'start', True, maxw=900, tag='title')
lc.text(MX, 58, '已渲染 EngineInput 走同步快路径（零 tokenizer 调用）；raw prompt 的阻塞 tokenize / mm 预处理'
        '下 renderer 线程池——tokenize 池可调、mm 池恒单工', 10.5, lc.C_MUTE, 'start',
        maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ④ add_request 分流 · L0：API 进程下行泳道'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_API_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 泳道框架 ----------------
LANE_GAP = 22                                # 泳道间净空
L1_Y, L1_H = 120, 220                        # 事件循环线程
L2_Y, L2_H = L1_Y + L1_H + LANE_GAP, 132     # renderer tokenize 池
L3_Y, L3_H = L2_Y + L2_H + LANE_GAP, 116     # mm 单工池
L4_Y, L4_H = L3_Y + L3_H + LANE_GAP, 90      # 同步离线面
FBD = 28       # 泳道内首个内容框顶距泳道顶——让出骑跨标签带（≥一行字高，R3 回修）
LANES = [
    (L1_Y, L1_H, '事件循环线程（一名柜员 · 只做轻活）', lc.C_API_S, '#ffffff', False),
    (L2_Y, L2_H, 'renderer tokenize 池 _executor（切配班 · 可多工）', lc.C_API_S, '#f8fafc', True),
    (L3_Y, L3_H, 'mm 单工池 _mm_executor（预处理班 · 恒一人）', lc.C_API_S, '#f8fafc', True),
    (L4_Y, L4_H, '同步离线面（LLM · 不走前台，调用方线程）', lc.C_MUTE, '#ffffff', True),
]
LBL_FS = 11
for ly, lh, lbl, st, fill, dash in LANES:
    lc.rect(MX, ly, BXR - MX, lh, fill, st, rx=10, sw=1.8, dash=dash)
    # R3：骑跨泳道顶边的标签徽章——白底板后画=切断虚线框线；放顶边右段（左段有
    # x=202 下穿箭头，标签带放左必被箭头穿字），右缘内缩 14px。
    pw = lc.tw(lbl, LBL_FS, True) + 20
    px = BXR - 14 - pw
    lc.rect(px, ly - 10, pw, 20, '#ffffff', st, rx=9, sw=1.2)
    lc.text(px + pw / 2, ly + 3.8, lbl, LBL_FS, st, 'middle', True, maxw=pw - 8,
            tag='lane:' + lbl[:6])

# ---------------- 泳道 1：分流 + 快路径 + 出口 ----------------
DEC = (84, 156, 236, 112)          # x, y, w, h
lc.rect(*DEC, '#ffffff', lc.C_API_S, rx=7, sw=1.8)
lc.text(DEC[0] + 14, DEC[1] + 20, 'add_request 分流', 11, lc.C_TXT, 'start', True,
        maxw=DEC[2] - 28, tag='dec:t')
lc.text(DEC[0] + 14, DEC[1] + 38, 'async_llm.py:L352-L380', 8.5, lc.C_FAINT, 'start',
        maxw=DEC[2] - 28, tag='dec:f')
lc.text(DEC[0] + 14, DEC[1] + 58, "isinstance(prompt, dict)", 8.5, C_BODY, 'start',
        maxw=DEC[2] - 24, tag='dec:c1')
lc.text(DEC[0] + 14, DEC[1] + 72, "and 'type' in prompt（L353）", 8.5, C_BODY, 'start',
        maxw=DEC[2] - 24, tag='dec:c2')

FAST = (480, 156, 340, 112)
lc.rect(*FAST, lc.C_API_F, lc.C_API_S, rx=7, sw=1.6)
lc.text(FAST[0] + 14, FAST[1] + 20, 'process_inputs · 同步快路径', 11, lc.C_TXT, 'start',
        True, maxw=FAST[2] - 28, tag='fast:t')
lc.text(FAST[0] + 14, FAST[1] + 40, '· tokenizer 调用 0 次', 9, C_BODY, 'start',
        maxw=FAST[2] - 26, tag='fast:l1')
lc.text(FAST[0] + 14, FAST[1] + 57, '· 跑在事件循环线程上', 9, C_BODY, 'start',
        maxw=FAST[2] - 26, tag='fast:l2')
lc.text(FAST[0] + 14, FAST[1] + 78, "注释：'Rendered EngineInput; no", 8, lc.C_MUTE, 'start',
        maxw=FAST[2] - 26, tag='fast:q1')
lc.text(FAST[0] + 14, FAST[1] + 92, "blocking preprocessing needed.'", 8, lc.C_MUTE, 'start',
        maxw=FAST[2] - 26, tag='fast:q2')

ENG = (1000, 156, 340, 112)
lc.rect(*ENG, '#ffffff', lc.C_API_S, rx=7, sw=1.6)
lc.text(ENG[0] + 14, ENG[1] + 20, 'EngineCoreRequest → 出港', 11, lc.C_TXT, 'start', True,
        maxw=ENG[2] - 28, tag='eng:t')
lc.text(ENG[0] + 14, ENG[1] + 40, '· 快路径同步直出', 9, C_BODY, 'start',
        maxw=ENG[2] - 26, tag='eng:l1')
lc.text(ENG[0] + 14, ENG[1] + 57, '· 池路径 await 取回后再出', 9, C_BODY, 'start',
        maxw=ENG[2] - 26, tag='eng:l2')
lc.text(ENG[0] + 14, ENG[1] + 84, '（双登记 + ADD 过线 = 第 11 站 · 帧序 ch5 已讲）', 8, lc.C_FAINT,
        'start', maxw=ENG[2] - 26, tag='eng:l3')

# DEC → FAST（真：已渲染）
lc.seg(DEC[0] + DEC[2], 212, FAST[0] - 2, 212, lc.C_API_S, 2.0, 'dn')
lc.text((DEC[0] + DEC[2] + FAST[0]) / 2, 200, '真：dict 带 \'type\'', 8.5, lc.C_API_S,
        'middle', maxw=118, tag='a:fast')
lc.text((DEC[0] + DEC[2] + FAST[0]) / 2, 228, '（已渲染）', 8, lc.C_MUTE, 'middle',
        maxw=118, tag='a:fasts')
# FAST → ENG
lc.seg(FAST[0] + FAST[2], 200, ENG[0] - 2, 200, lc.C_API_S, 2.0, 'dn')

# DEC → 下穿 tokenize 池（假：raw prompt）
lc.seg(202, DEC[1] + DEC[3], 202, L2_Y + FBD, lc.C_API_S, 2.0, 'dn')
lc.text(214, 288, '假：raw prompt → await 下池', 8.5, lc.C_API_S, 'start', maxw=210,
        tag='hd:t')
lc.text(214, 304, 'make_async(process_inputs,', 8, C_BODY, 'start', maxw=210, tag='hd:l1')
lc.text(214, 318, 'executor=renderer._executor)', 8, C_BODY, 'start', maxw=210, tag='hd:l2')
lc.text(214, 332, "L79-L81 · 'keep their event loop responsive'", 7.5, lc.C_FAINT, 'start',
        maxw=220, tag='hd:f')

# ---------------- 泳道 1 底部：心跳证据条 ----------------
HB_Y = 296
HB_X0, HB_X1 = 480, 1330
for i in range(19):
    x = HB_X0 + i * (HB_X1 - HB_X0) / 18
    h = 12 if i % 3 == 0 else 7
    lc.seg(x, HB_Y, x, HB_Y - h, lc.C_API_S, 2.2)
lc.text(HB_X0, HB_Y + 22, '循环心跳（间隔 0.01s）持续跳动——0.25s tokenize 期间实测 27 tick ≥ 阈值 20，'
        '循环从未被卡住（与左下池线程窗口并行）', 8.5, lc.C_API_S, 'start', maxw=820,
        tag='hb:t')

# ---------------- 泳道 2：tokenize 池工作 + 池构成 ----------------
TOK = (84, L2_Y + FBD, 346, 88)
lc.rect(*TOK, lc.C_API_F, lc.C_API_S, rx=7, sw=1.6)
lc.text(TOK[0] + 14, TOK[1] + 19, 'process_inputs 全程（阻塞 CPU 活）', 10, lc.C_TXT,
        'start', True, maxw=TOK[2] - 28, tag='tok:t')
lc.text(TOK[0] + 14, TOK[1] + 39, 'tokenize → 校验 → params 补全 → mm 展平', 8.5, C_BODY,
        'start', maxw=TOK[2] - 26, tag='tok:l1')
lc.text(TOK[0] + 14, TOK[1] + 56, '· 实测 0.25s tokenize', 8.5, C_BODY, 'start',
        maxw=TOK[2] - 26, tag='tok:l2')
lc.text(TOK[0] + 14, TOK[1] + 73, "· 注释：'must not block the event loop'", 7.5, lc.C_MUTE,
        'start', maxw=TOK[2] - 26, tag='tok:q')
# 池路径回归事件循环：TOK → 顶部走廊 → ENG 顶边（不穿任何框）
lc.parrow([(TOK[0] + TOK[2] + 2, TOK[1] + TOK[3] // 2), (465, TOK[1] + TOK[3] // 2),
           (465, 104), (ENG[0] + ENG[2] / 2, 104), (ENG[0] + ENG[2] / 2, ENG[1] - 4)],
          lc.C_API_S, 1.8, 'std')
lc.text(770, 96, 'await 取回（池路径）——递交即走，柜员没等', 8.5, lc.C_API_S, 'middle',
        maxw=420, tag='rt:t')

POOL2 = (500, L2_Y + FBD, 940, 88)
lc.rect(*POOL2, '#ffffff', lc.C_API_S, rx=7, sw=1.2, dash=True)
lc.text(POOL2[0] + 14, POOL2[1] + 19, 'ThreadPoolExecutor(renderer_num_workers)——tokenize / decode / '
        'embeds 三处 make_async 共用（base.py:L85-L86）', 9.5, lc.C_TXT, 'start', True,
        maxw=POOL2[2] - 190, tag='p2:t')
WK_Y = POOL2[1] + 38
wk_x0 = POOL2[0] + 14
for i in range(4):
    dash = (i > 0)
    lc.rect(wk_x0 + i * 30, WK_Y, 24, 24, lc.C_API_F if not dash else '#ffffff',
            lc.C_API_S, rx=4, sw=1.3, dash=dash)
    lc.text(wk_x0 + i * 30 + 12, WK_Y + 17, '工', 8.5, lc.C_API_S, 'middle', True,
            tag='wk' + str(i))
lc.text(wk_x0 + 130, WK_Y + 11, '· 默认 1 工（config/model.py:L355）', 8.5, C_BODY, 'start',
        maxw=280, tag='p2:l1')
lc.text(wk_x0 + 130, WK_Y + 27, '· 配 4 → tokenize 池 4 工（实测）', 8.5, C_BODY, 'start',
        maxw=280, tag='p2:l2')
lc.text(POOL2[0] + 460, WK_Y + 11, '线程归属判定：tokenize 线程 = renderer 池线程', 8.5,
        lc.C_API_S, 'start', True, maxw=440, tag='p2:j1')
lc.text(POOL2[0] + 460, WK_Y + 27, '≠ 事件循环线程（同一池线程 id 直接对上）', 8.5,
        lc.C_API_S, 'start', maxw=440, tag='p2:j2')

# ---------------- 泳道 3：mm 支路 + mm 池构成 ----------------
lc.seg(202, TOK[1] + TOK[3], 202, L3_Y + FBD, lc.C_ENG_S, 1.8, 'std')
# R4：标签进泳道2 底边（L2_Y+L2_H=494）与泳道3 顶边（L3_Y=516）之间的净空带中央
# ——原 baseline=494 恰在泳道2 底边线上，虚线拦腰穿字。
MM_LBL_Y = (L2_Y + L2_H + L3_Y) / 2 + 3
lc.text(212, MM_LBL_Y, '带图的路', 8, lc.C_ENG_S, 'start', maxw=90, tag='mm:a')
MM = (84, L3_Y + FBD, 346, 72)
lc.rect(*MM, lc.C_ENG_F, lc.C_ENG_S, rx=7, sw=1.5, dash=True)
lc.text(MM[0] + 14, MM[1] + 18, '_process_multimodal_async', 9.5, lc.C_TXT, 'start', True,
        maxw=MM[2] - 28, tag='mm:t')
lc.text(MM[0] + 14, MM[1] + 36, '（mm 预处理 · base.py:L107-L108）', 8, C_BODY, 'start',
        maxw=MM[2] - 26, tag='mm:f')
lc.text(MM[0] + 14, MM[1] + 54, '跑在 mm 单工池线程 ≠ 事件循环线程', 8.5, C_BODY, 'start',
        maxw=MM[2] - 26, tag='mm:l')
POOL3 = (500, L3_Y + FBD, 940, 72)
lc.rect(*POOL3, '#ffffff', lc.C_API_S, rx=7, sw=1.2, dash=True)
lc.text(POOL3[0] + 14, POOL3[1] + 18, 'ThreadPoolExecutor(1) 恒单工（base.py:L88-L90）', 9.5,
        lc.C_TXT, 'start', True, maxw=POOL3[2] - 28, tag='p3:t')
lc.text(POOL3[0] + 14, POOL3[1] + 37, "注释原话：'must stay single-worker per #38418 (P0/P1 order)'——"
        'mm 预处理同时产 P0/P1 两级缓存键，单工才保同请求键序', 8.5, C_BODY, 'start',
        maxw=POOL3[2] - 26, tag='p3:l1')
lc.text(POOL3[0] + 14, POOL3[1] + 55, '· 配 4 工时 mm 池仍 1 工（两池独立装配，实测）；分开还防长文本 '
        'tokenize 把小图预处理饿在队列里', 8.5, C_BODY, 'start', maxw=POOL3[2] - 26,
        tag='p3:l2')

# ---------------- 泳道 4：同步离线面 ----------------
OFF = (84, L4_Y + FBD, 1256, 48)
lc.rect(*OFF, '#ffffff', lc.C_MUTE, rx=7, sw=1.2, dash=True)
lc.text(OFF[0] + 14, OFF[1] + 20, 'LLM.add_request（离线面，llm_engine.py:L250-L262）：同一条 process_inputs '
        '跑在调用方线程、不经池', 9.5, lc.C_TXT, 'start', True, maxw=OFF[2] - 26, tag='off:t')
lc.text(OFF[0] + 14, OFF[1] + 38, '· renderer_num_workers>1 在离线入口是 no-op（llm.py:L358-L369 显式警告）'
        '——同步面整条泳道无池', 8.5, C_BODY, 'start', maxw=OFF[2] - 26, tag='off:l')

# ---------------- 底部：装配判定 + 外部基准 ----------------
BP_Y, BP_H = L4_Y + L4_H + LANE_GAP, 156
JL = (MX, BP_Y, 660, BP_H)
lc.rect(*JL, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(JL[0] + 16, JL[1] + 22, '线程归属判定（host 实测布尔值；线程 id 为运行时值不印）', 10.5,
        lc.C_TXT, 'start', True, maxw=JL[2] - 32, tag='jl:t')
jl_lines = [
    '· tokenize 线程 = renderer 池线程 ≠ 事件循环线程（同一池线程 id 直接对上）',
    '· mm 作业线程 = mm 池线程 ≠ 事件循环线程',
    '· 快路径 tokenizer 调用 0 次 · process_inputs 跑在事件循环线程',
    '· 离线面 process_inputs 跑在调用方线程',
    "· make_async = partial + run_in_executor（async_utils.py:L28-L45）",
    "   docstring：'The code in this function needs to be thread safe.'",
]
for i, ln in enumerate(jl_lines):
    lc.text(JL[0] + 16, JL[1] + 44 + i * 17, ln, 8.5, C_BODY, 'start', maxw=JL[2] - 30,
            tag='jl:l' + str(i))

BM = (748, BP_Y, 692, BP_H)
lc.rect(*BM, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
_bd = '外部'
lc.rect(BM[0] + BM[2] - 52, BM[1] + 10, 40, 18, lc.C_API_F, lc.C_MUTE, rx=9, sw=1.0)
lc.text(BM[0] + BM[2] - 32, BM[1] + 23, _bd, 9, lc.C_MUTE, 'middle', True, tag='bm:bd')
lc.text(BM[0] + 16, BM[1] + 22, '外部基准 #12287（A100 · Llama-3.2-1B · 6000 请求）', 10.5,
        lc.C_TXT, 'start', True, maxw=BM[2] - 90, tag='bm:t')
lc.text(BM[0] + 16, BM[1] + 52, 'mean TTFT −14%', 15, lc.C_API_S, 'start', True,
        maxw=220, tag='bm:n1')
lc.text(BM[0] + 16, BM[1] + 78, 'p99 TPOT −31%', 15, lc.C_API_S, 'start', True,
        maxw=220, tag='bm:n2')
bm_lines = [
    '· tokenize / 图片预处理是百 ms 级 CPU 活——卡住事件循环 =',
    '   所有并发请求一起等（SSE 全停）',
    '· 与 output_handler 不分块同源的事件循环头阻塞问题，',
    '   在下行侧的另一半解（上行侧 → ch7 的 chunk + sleep(0)）',
]
for i, ln in enumerate(bm_lines):
    lc.text(BM[0] + 260, BM[1] + 48 + i * 17, ln, 8.5, C_BODY, 'start', maxw=BM[2] - 280,
            tag='bm:l' + str(i))

# ---------------- 图例 + 页脚 ----------------
LEG_Y = BP_Y + BP_H + 34
lx = MX
items = [('solid', '实线蓝框 = 事件循环线程上'), ('dashb', '虚线框 = 后台线程池 / 离线调用方线程'),
         ('tick', '竖线刻度 = 心跳 tick')]
for kind, name in items:
    if kind == 'solid':
        lc.rect(lx, LEG_Y - 8, 22, 14, '#ffffff', lc.C_API_S, rx=4, sw=1.4)
    elif kind == 'dashb':
        lc.rect(lx, LEG_Y - 8, 22, 14, '#ffffff', lc.C_MUTE, rx=4, sw=1.2, dash=True)
    else:
        lc.seg(lx + 8, LEG_Y, lx + 8, LEG_Y - 12, lc.C_API_S, 2.2)
        lc.seg(lx + 18, LEG_Y, lx + 18, LEG_Y - 7, lc.C_API_S, 2.2)
    lc.text(lx + 28, LEG_Y + 3, name, 9.5, lc.C_TXT, 'start', maxw=300, tag='leg' + name)
    lx += 28 + lc.tw(name, 9.5) + 24
lc.text(MX, LEG_Y + 26, '心跳 tick 数（27）为 host 单次运行值、每次抖动——引用量级与布尔判定'
        '（loop_never_blocked）；0.25s 为 seam tokenizer 人为延迟（放大真实 tokenize 百 ms 量级）',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 44, '分流 verbatim：vllm/v1/engine/async_llm.py:L352-L380 · 装配：input_processor.py:L79-L81 · '
        '行号基线 vLLM v0.27.1', 9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch06-fig-pool-offload.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
