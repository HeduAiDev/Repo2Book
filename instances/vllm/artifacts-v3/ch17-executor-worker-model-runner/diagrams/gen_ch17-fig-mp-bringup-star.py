#!/usr/bin/env python3
"""ch17 机制图 2 · mp 拉起星形装配时序（figure_spec ch17-fig-mp-bringup-star，模板 flow）

放大自 L0 GPU 执行臂行（gpu_column）的 Executor 块——即 L2 章图 center 拍片 ① Executor
（站 1-3）的拉起时序展开。架构归属回指 L2/L0（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：mp 拉起是一次性的星形装配：建 1 条 SHM 广播 MQ → 逐 local_rank spawn 子进程（每个
子进程内 init_worker→init_device→load_model 全部完成才经 ready_pipe 发 READY、附 response MQ
handle）→ 父进程收齐 READY 才装监控线程与 N 条应答 MQ、futures_queue 起空——实测
world_size=2：拉起 1823.8ms 出 2 个互异 pid 的子进程，READY 时全部 loaded=True 且应答 pid
与 spawn pid 一致，之后 1 次广播 2 个应答。

数字全部取自 figure_spec.numbers（m4_bringup host 实测 trace + pin 锚点，逐字对齐）；坐标由
常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

# 追加 marker：绿（子进程方向）/ 紫（MQ 通道）——沿用 l0_common 配色常量，不另造色值
DEFS = lc.DEFS.replace(
    '</defs>',
    f'<marker id="gp" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_GPU_S}"/></marker>'
    f'<marker id="zq" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_ZMQ_S}"/></marker>'
    '</defs>')

W, H = 1500, 950
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'mp 拉起 = 一次性星形装配：广播电台先于店员存在，培训全部完成才按铃报到',
        16.5, lc.C_TXT, 'start', True, maxw=1070, tag='title')
lc.text(MX, 58, '建 1 条 SHM 广播 MQ → 逐 local_rank spawn 子进程 → 子进程内走完培训链才发 READY（附应答频道）→ '
        '收齐 2 个 READY 才开张——实测 world_size=2：1823.8ms 拉起，之后 1 次广播 2 个应答',
        10.5, lc.C_MUTE, 'start', maxw=1070, tag='subtitle')
_ch = '放大自 L2 拍片 ① Executor（站 1-3）· L0：GPU 执行臂上层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 第一幕：装配时序 =================
# ---- 左栏：父进程竖带（橙） ----
PF_X, PF_Y, PF_W, PF_H = MX, 100, 560, 420
PB_X, PB_W = PF_X + 24, PF_W - 48          # 内框 x=84 w=512
PCX = PB_X + PB_W / 2                      # 内框中线 340
lc.rect(PF_X, PF_Y, PF_W, PF_H, lc.C_ENG_F, lc.C_ENG_S, rx=10, sw=2.0)
lc.text(PF_X + 16, PF_Y + 24, '父进程 — MultiprocExecutor（EngineCore 进程内）', 11.5,
        lc.C_ENG_S, 'start', True, maxw=325, tag='pf:t')
lc.text(PF_X + PF_W - 16, PF_Y + 24, 'vllm/v1/executor/multiproc_executor.py', 8.5,
        lc.C_FAINT, 'end', maxw=200, tag='pf:f')

BOXES = [
    (140, 76, '① 建 rpc_broadcast_mq —— 广播电台（1 条）',
     ['SHM 广播队列：单写多读，一次 enqueue 全读者可见',
      '句柄随 spawn export 给子进程（L151-L201）'], lc.C_ZMQ_S, lc.C_ZMQ_F),
    (236, 64, '② 逐 local_rank spawn 子进程（×2）',
     ['拉起实测 1823.8ms——win32 spawn 冷启动占大头（量级感）',
      '子 pid 35228 / 36668 互异，都 ≠ 父 19260'], lc.C_ENG_S, '#ffffff'),
    (320, 56, '③ wait_for_ready —— 收齐 2 个 READY 才继续',
     ['「Workers must be created before wait_for_ready to avoid deadlock」'],
     lc.C_ENG_S, '#ffffff'),
    (396, 104, '④ 开张装配（全部 READY 之后）',
     ['MultiprocWorkerMonitor 监控线程 ×1',
      'response_mqs ×2 —— MessageQueue(1,1) 每 worker 一条（L585）',
      'futures_queue 空 deque',
      'output_rank = world_size − tp×pcp = 2−2×1 = 0（L509-L523）'],
     lc.C_ENG_S, '#ffffff'),
]
for y, h, title, lines, stroke, fill in BOXES:
    lc.rect(PB_X, y, PB_W, h, fill, stroke, rx=7, sw=1.5)
    lc.text(PB_X + 14, y + 21, title, 10.5, lc.C_TXT, 'start', True, maxw=PB_W - 28,
            tag='pb:' + title[:8])
    for i, ln in enumerate(lines):
        lc.text(PB_X + 14, y + 40 + i * 16, ln, 8.5, '#334155', 'start', maxw=PB_W - 28,
                tag='pbl:' + ln[:10])
# 竖排时序箭头 ①→②→③→④
for (y1, h1, *_), (y2, *_rest) in zip(BOXES, BOXES[1:]):
    lc.seg(PCX, y1 + h1 + 1, PCX, y2 - 2, lc.C_ENG_S, 1.6, 'up')

# ---- 右栏：两个子进程竖带（绿，上下叠放） ----
LANE_X, LANE_W = 680, 760
TRAIN = [
    ('init_worker', 'qualname 解析真 Worker'),
    ('init_device', '选卡 + 分布式初始化'),
    ('load_model', '装载权重（tag=weights）'),
]
TB_W, TB_GAP, TB_H = 170, 20, 64
RD_W = 150
LANES = [
    (100, 'VllmWorker-0 · rank 0 · pid 35228'),
    (320, 'VllmWorker-1 · rank 1 · pid 36668'),
]
for lane_y, lane_title in LANES:
    lc.rect(LANE_X, lane_y, LANE_W, 200, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.0)
    lc.text(LANE_X + 16, lane_y + 24, lane_title, 11, lc.C_GPU_S, 'start', True,
            maxw=LANE_W - 300, tag='ln:' + lane_title[:12])
    lc.text(LANE_X + LANE_W - 16, lane_y + 24, 'worker_main → WorkerProc（L820-L944）', 8.5,
            lc.C_FAINT, 'end', maxw=280, tag='ln:f' + lane_title[9])
    cy = lane_y + 56
    for i, (t, s) in enumerate(TRAIN):
        x = LANE_X + 16 + i * (TB_W + TB_GAP)
        lc.rect(x, cy, TB_W, TB_H, '#ffffff', lc.C_GPU_S, rx=6, sw=1.3)
        lc.text(x + 12, cy + 25, t, 10, lc.C_TXT, 'start', True, maxw=TB_W - 24,
                tag='tb' + str(i) + lane_title[9])
        lc.text(x + 12, cy + 45, s, 8.5, lc.C_MUTE, 'start', maxw=TB_W - 24,
                tag='tbs' + str(i) + lane_title[9])
        if i < len(TRAIN) - 1:
            lc.seg(x + TB_W + 1, cy + TB_H / 2, x + TB_W + TB_GAP - 2, cy + TB_H / 2,
                   lc.C_GPU_S, 1.5, 'gp')
    # 培训链 → READY
    x_rd = LANE_X + 16 + 3 * (TB_W + TB_GAP)
    lc.seg(x_rd - TB_GAP + 1, cy + TB_H / 2, x_rd - 2, cy + TB_H / 2, lc.C_GPU_S, 1.5, 'gp')
    lc.rect(x_rd, cy, RD_W, TB_H, '#ffffff', lc.C_ENG_S, rx=6, sw=1.8)
    lc.text(x_rd + RD_W / 2, cy + 20, 'READY', 11, lc.C_ENG_S, 'middle', True,
            maxw=RD_W - 10, tag='rd' + lane_title[9])
    lc.text(x_rd + RD_W / 2, cy + 37, '培训完成才报到', 8.5, lc.C_MUTE, 'middle',
            maxw=RD_W - 10, tag='rds' + lane_title[9])
    lc.text(x_rd + RD_W / 2, cy + 53, '附 response MQ handle', 7.5, lc.C_MUTE, 'middle',
            maxw=RD_W - 10, tag='rdh' + lane_title[9])
    lc.text(LANE_X + 16, lane_y + 150, '此后：wait_until_ready → worker_busy_loop 服役（dequeue → getattr 派发）',
            8.5, lc.C_MUTE, 'start', maxw=LANE_W - 32, tag='ln:note' + lane_title[9])

# ---- spawn 箭头（父 → 两子；走出右栏走廊） ----
lc.parrow([(PB_X + PB_W, 268), (636, 268), (636, 200), (LANE_X - 2, 200)],
          lc.C_GPU_S, 1.8, 'gp')
lc.parrow([(PB_X + PB_W, 286), (648, 286), (648, 420), (LANE_X - 2, 420)],
          lc.C_GPU_S, 1.8, 'gp')
lc.text(642, 252, 'spawn ×2', 8.5, lc.C_GPU_S, 'start', True, maxw=70, tag='a:spawn')

# ---- READY 回流箭头（各子 → 父 ③；收齐才继续） ----
RD_CX = LANE_X + 16 + 3 * (TB_W + TB_GAP) + RD_W / 2      # READY chip 中线 = 1341
lc.parrow([(RD_CX, 220), (RD_CX, 310), (666, 310), (666, 348),
           (PB_X + PB_W + 2, 348)], lc.C_ENG_S, 1.8, 'up')
lc.parrow([(RD_CX, 440), (RD_CX, 534), (672, 534), (672, 364),
           (PB_X + PB_W + 2, 364)], lc.C_ENG_S, 1.8, 'up')
lc.text(690, 306, 'READY ×2（收齐才继续）', 8.5, lc.C_ENG_S, 'start', True, maxw=150,
        tag='a:ready1')
lc.text(690, 530, 'READY（附 response MQ handle）', 8.5, lc.C_ENG_S, 'start', maxw=180,
        tag='a:ready2')

# ================= 第二幕：稳态控制面（星形不对称） =================
lc.text(MX, 572, '开张之后——每道指令：电台播一次、全体店员都听得到（星形不对称：下行 1 条广播、上行 N 条应答）',
        10, lc.C_TXT, 'start', True, maxw=1380, tag='act2:t')
# 父 chip（上中）
PP_X, PP_Y, PP_W, PP_H = 560, 588, 380, 52
lc.rect(PP_X, PP_Y, PP_W, PP_H, lc.C_ENG_F, lc.C_ENG_S, rx=8, sw=1.8)
lc.text(PP_X + PP_W / 2, PP_Y + 21, '父进程 enqueue 一次（rpc_calls = 1）', 10.5, lc.C_ENG_S,
        'middle', True, maxw=PP_W - 20, tag='pp:t')
lc.text(PP_X + PP_W / 2, PP_Y + 40, '广播 MQ：一次 enqueue 全读者可见', 8.5, lc.C_MUTE,
        'middle', maxw=PP_W - 20, tag='pp:s')
# 两个子 chip（下左/下右）
CH_Y, CH_H = 676, 68
CHIPS2 = [
    (280, 340, 'VllmWorker-0 · rank 0',
     ['worker_busy_loop dequeue → getattr 派发',
      '★ execute_model 只收 output_rank=0 这份（2−2×1=0）'], True),
    (880, 340, 'VllmWorker-1 · rank 1',
     ['同款 busy loop——同一跳广播全收到',
      '应答回自己的频道 MessageQueue(1,1)'], False),
]
for cx0, cw, t, lines, hot in CHIPS2:
    lc.rect(cx0, CH_Y, cw, CH_H, lc.C_GPU_F if hot else '#ffffff', lc.C_GPU_S, rx=8,
            sw=1.8 if hot else 1.4)
    lc.text(cx0 + 14, CH_Y + 21, t, 10.5, lc.C_GPU_S, 'start', True, maxw=cw - 28,
            tag='c2:' + t[:10])
    for i, ln in enumerate(lines):
        lc.text(cx0 + 14, CH_Y + 40 + i * 17, ln, 8.5, '#334155', 'start', maxw=cw - 28,
                tag='c2l:' + ln[:10])
# 下行广播（紫粗，两支）
lc.parrow([(PP_X + 80, PP_Y + PP_H), (450, PP_Y + PP_H), (450, CH_Y - 2)],
          lc.C_ZMQ_S, 3.0, 'zq')
lc.parrow([(PP_X + PP_W - 80, PP_Y + PP_H), (1050, PP_Y + PP_H), (1050, CH_Y - 2)],
          lc.C_ZMQ_S, 3.0, 'zq')
lc.text(750, PP_Y + PP_H + 20, '广播 MQ 一次 enqueue —— 全员可见', 9, lc.C_ZMQ_S, 'middle',
        True, maxw=200, tag='a:bcast')
# 上行应答（橙，rank0 粗 / rank1 细）
lc.parrow([(560, CH_Y), (560, 664), (620, 664), (620, PP_Y + PP_H + 1)],
          lc.C_ENG_S, 2.4, 'up')
lc.parrow([(1000, CH_Y), (1000, 664), (920, 664), (920, PP_Y + PP_H + 1)],
          lc.C_ENG_S, 1.3, 'up')
lc.text(750, 706, '上行应答 ×2 —— MessageQueue(1,1) 每 worker 一条', 8.5, lc.C_MUTE,
        'middle', maxw=240, tag='a:reply')
lc.text(750, 722, 'collective 探针：reply_count=2 · ranks [0,1]', 8.5, lc.C_MUTE,
        'middle', maxw=240, tag='a:reply2')

# ================= 角注小卡 =================
CARD_Y, CARD_H, CARD_W = 756, 112, 440
CARDS = [
    (60, 'READY 的语义 = 培训已完成',
     ['2 个 worker 全部 loaded=True',
      '应答 pid 与 spawn pid 一致（35228 / 36668）',
      '发出点：WorkerProc 构造之后（L886-L893）']),
    (520, 'callable 分支：整函数下发',
     ['经 cloudpickle 序列化整个函数发过去',
      '在子进程 pid 内执行（executed_in_child_pids=true）',
      '（bytes 分支——worker_busy_loop 的另一条腿）']),
    (980, '关停 304.3ms：优雅全退',
     ['death_writer 关闭 → graceful 退出',
      '2 子进程全退 · 广播 MQ 置 None',
      '应答 MQ 归 0（监控线程一并收摊）']),
]
for x, t, lines in CARDS:
    lc.rect(x, CARD_Y, CARD_W, CARD_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
    lc.text(x + 14, CARD_Y + 21, t, 10, lc.C_TXT, 'start', True, maxw=CARD_W - 28,
            tag='cd:' + t[:10])
    for i, ln in enumerate(lines):
        lc.text(x + 14, CARD_Y + 42 + i * 17, ln, 8.5, '#334155', 'start', maxw=CARD_W - 28,
                tag='cdl:' + ln[:10])

# ================= 图例 + 页脚 =================
LEG_Y = 890
lx = MX
lc.rect(lx, LEG_Y - 9, 20, 13, lc.C_ENG_F, lc.C_ENG_S, rx=4, sw=1.4)
lc.text(lx + 26, LEG_Y + 2, '父进程（EngineCore 进程）', 9, lc.C_TXT, 'start', maxw=200,
        tag='leg1')
lx += 26 + lc.tw('父进程（EngineCore 进程）', 9) + 22
lc.rect(lx, LEG_Y - 9, 20, 13, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.4)
lc.text(lx + 26, LEG_Y + 2, 'worker 子进程', 9, lc.C_TXT, 'start', maxw=140, tag='leg2')
lx += 26 + lc.tw('worker 子进程', 9) + 22
lc.rect(lx, LEG_Y - 9, 20, 13, lc.C_ZMQ_F, lc.C_ZMQ_S, rx=4, sw=1.4)
lc.text(lx + 26, LEG_Y + 2, 'MQ 通道（广播电台 / 应答频道）', 9, lc.C_TXT, 'start',
        maxw=260, tag='leg3')
lx += 26 + lc.tw('MQ 通道（广播电台 / 应答频道）', 9) + 22
lc.seg(lx, LEG_Y - 3, lx + 20, LEG_Y - 3, lc.C_ZMQ_S, 3.0, 'zq')
lc.text(lx + 26, LEG_Y + 2, '粗 = 一次广播全员可见', 9, lc.C_TXT, 'start', maxw=180,
        tag='leg4')
lc.text(MX, LEG_Y + 26, 'verbatim vllm/v1/executor/multiproc_executor.py:L151-L201 · L388 · L509-L523 · L585 · L820-L944 · L886-L893',
        9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '实测取自精简版 companion host 实测（真 spawn · 真 READY 握手 · 真 ZMQ loopback 广播 seam——毫秒只取量级感）· '
        '行号基线 vLLM v0.27.1 · 跨节点 peer 装配与失败两路另章展开',
        9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch17-fig-mp-bringup-star.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
