#!/usr/bin/env python3
"""ch17 机制图 1 · 三层各答一问、两轴正交（figure_spec ch17-fig-three-layers-three-questions，模板 layout）

放大自 L0 GPU 执行臂行（gpu_column）的『Executor → Worker → GPUModelRunner』三层——
即 L2 章图 center『三层各答一问』框的问题域切分展开。架构归属回指 L2/L0（FIGURE-SYSTEM
§3.3）：图右上角指北小签。

claim：三层各答一问且两轴正交：Executor 只答『在哪跑』（get_class 6 路分发）、Worker 只答
『设备归谁管』（worker_cls 字符串经平台插件解析、三锚点持有一生）、ModelRunner 只答
『这一拍怎么算』（对前两问无感）——加一种硬件=改一个字符串、换一种编排=换一个 executor
类，互不牵连；代价是一次 execute_model 穿 5 层间接。

数字全部取自 figure_spec.numbers（host 实测 trace + pin 锚点，逐字对齐）；坐标由常量/循环
计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

# 追加绿色 marker（硬件适配轴）——沿用 l0_common 配色常量，不另造色值
DEFS = lc.DEFS.replace(
    '</defs>',
    f'<marker id="gp" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_GPU_S}"/></marker>'
    '</defs>')

W, H = 1500, 990
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '为什么拆三层：各答一问、两轴各改各的——加硬件只改一个字符串，换编排只换一个类',
        16.5, lc.C_TXT, 'start', True, maxw=1070, tag='title')
lc.text(MX, 58, 'Executor 只答「在哪跑」（get_class 6 路分发）· Worker 只答「设备归谁管」（worker_cls 经平台插件解析）· '
        'ModelRunner 只答「这一拍怎么算」——代价：一次 execute_model 穿 5 层间接',
        10.5, lc.C_MUTE, 'start', maxw=1070, tag='subtitle')
_ch = '放大自 L2 center 三层各答一问 · L0：GPU 执行臂上层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 带区常量 ----------------
BAND_X, BAND_W = MX, 1080                 # 带区右缘 = 1140
BAND_R = BAND_X + BAND_W

# ---------------- 带一：Executor（橙） ----------------
B1_Y, B1_H = 100, 192
lc.rect(BAND_X, B1_Y, BAND_W, B1_H, lc.C_ENG_F, lc.C_ENG_S, rx=10, sw=2.0)
lc.text(BAND_X + 16, B1_Y + 24, 'Executor — 只答「在哪跑」', 13, lc.C_ENG_S, 'start', True,
        maxw=420, tag='b1:t')
lc.text(BAND_R - 16, B1_Y + 24, 'vllm/v1/executor/abstract.py · get_class L48-L92', 9,
        lc.C_FAINT, 'end', maxw=420, tag='b1:f')

CHIPS = [
    ('type 子类', 'Executor 子类实例直接用', False),
    ('ray', 'RayDistributedExecutor 等', False),
    ('mp', 'MultiprocExecutor（本章）', True),
    ('uni', 'UniProcExecutor', False),
    ('external_launcher', 'ExecutorWithExternalLauncher', False),
    ('自定义 qualname', 'resolve_obj_by_qualname', False),
]
CHW, CHG, CHY, CHH = 158, 10, B1_Y + 40, 48
chip_cx = []
for i, (t, s, hot) in enumerate(CHIPS):
    x = BAND_X + 16 + i * (CHW + CHG)
    chip_cx.append(x + CHW / 2)
    lc.rect(x, CHY, CHW, CHH, lc.C_ENG_F if hot else '#ffffff', lc.C_ENG_S, rx=5,
            sw=1.8 if hot else 1.2)
    lc.text(x + CHW / 2, CHY + 19, t, 9.5, lc.C_TXT, 'middle', True, maxw=CHW - 10,
            tag='b1:c' + str(i))
    lc.text(x + CHW / 2, CHY + 36, s, 7.5, lc.C_MUTE, 'middle', maxw=CHW - 10,
            tag='b1:cs' + str(i))
# 分发标签 → 工厂框（短箭头逐条汇入）
FAC_Y, FAC_H = CHY + CHH + 22, 52
for cx in chip_cx:
    lc.seg(cx, CHY + CHH + 1, cx, FAC_Y - 2, lc.C_ENG_S, 1.5, 'up')
lc.rect(BAND_X + 16, FAC_Y, BAND_W - 32, FAC_H, '#ffffff', lc.C_ENG_S, rx=6, sw=1.5)
lc.text(BAND_X + 30, FAC_Y + 21, 'get_class 工厂：按 distributed_executor_backend 选出一个 Executor 子类——进程编排轴的分发点',
        10.5, lc.C_TXT, 'start', True, maxw=BAND_W - 60, tag='b1:fac')
lc.text(BAND_X + 30, FAC_Y + 40, '未匹配的值 → else 分支 raise ValueError（Unknown distributed executor backend）· 启动期一次：core.py 在 EngineCore 进程内构造',
        9, lc.C_MUTE, 'start', maxw=BAND_W - 60, tag='b1:fac2')

# ---------------- spawn 边（两带交界的灰虚线） ----------------
lc.seg(BAND_X, 308, 1260, 308, lc.C_MUTE, 1.4, dash=True)
lc.text(BAND_X + 16, 303, 'spawn 边——executor 拉起 worker：mp = spawn 子进程 · uni = 同进程构造', 8.5,
        lc.C_MUTE, 'start', maxw=560, tag='spawn:lbl')

# ---------------- 带二：Worker（绿） ----------------
B2_Y, B2_H = 320, 246
lc.rect(BAND_X, B2_Y, BAND_W, B2_H, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.0)
lc.text(BAND_X + 16, B2_Y + 24, 'WorkerWrapperBase + Worker — 只答「设备归谁管」', 13, lc.C_GPU_S,
        'start', True, maxw=520, tag='b2:t')
lc.text(BAND_R - 16, B2_Y + 24, 'vllm/v1/worker/worker_base.py · gpu_worker.py', 9,
        lc.C_FAINT, 'end', maxw=420, tag='b2:f')

# worker_cls 解析链（左缘入口：字符串经平台插件解析）
WC_Y, WC_H = B2_Y + 40, 54
WC = [
    (76, 190, "worker_cls='auto'", '进场只是一个字符串'),
    (294, 252, '平台插件 check_and_update_config', 'platforms/cuda.py:L308-L313'),
    (574, 330, "'vllm.v1.worker.gpu_worker.Worker'", '解析发生在平台，不在执行器'),
    (930, 194, 'init_worker（延迟解析）', 'qualname → 真 Worker'),
]
for i, (x, w_, t, s) in enumerate(WC):
    lc.rect(x, WC_Y, w_, WC_H, '#ffffff', lc.C_GPU_S, rx=6, sw=1.3)
    lc.text(x + w_ / 2, WC_Y + 21, t, 9.5, lc.C_TXT, 'middle', True, maxw=w_ - 10,
            tag='b2:wc' + str(i))
    lc.text(x + w_ / 2, WC_Y + 39, s, 8.5, lc.C_MUTE, 'middle', maxw=w_ - 10,
            tag='b2:wcs' + str(i))
for i in range(len(WC) - 1):
    x1 = WC[i][0] + WC[i][1]
    x2 = WC[i + 1][0]
    lc.seg(x1 + 1, WC_Y + WC_H / 2, x2 - 2, WC_Y + WC_H / 2, lc.C_GPU_S, 1.5, 'gp')

# 显存三锚点
lc.text(BAND_X + 16, B2_Y + 120, '设备显存的一生——三个锚点（对这台设备的生命周期负责）：', 9.5,
        lc.C_TXT, 'start', True, maxw=560, tag='b2:anchor')
AN_Y, AN_H, AN_W, AN_GAP = B2_Y + 128, 58, 320, 44
ANCHORS = [
    ('load_model', '装载权重——CuMem 池 tag=weights'),
    ('determine_available_memory', 'profile 跑一次定 KV 账'),
    ('initialize_from_config', 'tag=kv_cache 池内分配'),
]
for i, (t, s) in enumerate(ANCHORS):
    x = BAND_X + 16 + i * (AN_W + AN_GAP)
    lc.rect(x, AN_Y, AN_W, AN_H, '#ffffff', lc.C_GPU_S, rx=6, sw=1.3)
    lc.text(x + 12, AN_Y + 23, t, 10, lc.C_TXT, 'start', True, maxw=AN_W - 24,
            tag='b2:an' + str(i))
    lc.text(x + 12, AN_Y + 42, s, 8.5, lc.C_MUTE, 'start', maxw=AN_W - 24,
            tag='b2:ans' + str(i))
    if i < 2:
        lc.seg(x + AN_W + 1, AN_Y + AN_H / 2, x + AN_W + AN_GAP - 2, AN_Y + AN_H / 2,
               lc.C_GPU_S, 1.5, 'gp')
lc.text(BAND_X + 16, B2_Y + 228, 'WorkerBase 自述使命："allows vLLM to cleanly separate implementations '
        'for different hardware"（worker_base.py:L40-L43）', 9, lc.C_MUTE, 'start',
        maxw=BAND_W - 32, tag='b2:quote')

# ---------------- 带三：GPUModelRunner（绿） ----------------
B3_Y, B3_H = 584, 118
lc.rect(BAND_X, B3_Y, BAND_W, B3_H, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.0)
lc.text(BAND_X + 16, B3_Y + 24, 'GPUModelRunner — 只答「这一拍怎么算」', 13, lc.C_GPU_S,
        'start', True, maxw=480, tag='b3:t')
lc.text(BAND_R - 16, B3_Y + 24, 'vllm/v1/worker/gpu_model_runner.py', 9, lc.C_FAINT, 'end',
        maxw=420, tag='b3:f')
SIG_X, SIG_Y, SIG_W, SIG_H = BAND_X + 16, B3_Y + 40, 560, 62
lc.rect(SIG_X, SIG_Y, SIG_W, SIG_H, '#ffffff', lc.C_GPU_S, rx=6, sw=1.3)
lc.text(SIG_X + 14, SIG_Y + 24, 'execute_model(scheduler_output, intermediate_tensors)', 10.5,
        lc.C_TXT, 'start', True, maxw=SIG_W - 28, tag='b3:sig1')
lc.text(SIG_X + 14, SIG_Y + 46, '它只认这份签名——前两问的答案不进参数', 8.5, lc.C_MUTE,
        'start', maxw=SIG_W - 28, tag='b3:sig2')
lc.text(SIG_X + SIG_W + 24, SIG_Y + 22, '对前两问无感：不知道在哪跑、不知道归谁管', 9.5,
        lc.C_TXT, 'start', True, maxw=460, tag='b3:n1')
lc.text(SIG_X + SIG_W + 24, SIG_Y + 42, '不知道自己跑在第几个进程——uni 与 mp 对它无差别', 9.5,
        lc.C_MUTE, 'start', maxw=460, tag='b3:n2')

# ---------------- 右缘：两条正交变化轴 ----------------
AX_X = 1260
# 轴一（进程编排轴，橙）：轴头盒 → spawn 边的交叉点
lc.rect(1200, 86, 120, 26, '#ffffff', lc.C_ENG_S, rx=6, sw=1.5)
lc.text(1260, 104, '进程编排轴', 10.5, lc.C_ENG_S, 'middle', True, maxw=112, tag='ax1:t')
lc.seg(AX_X, 112, AX_X, 300, lc.C_ENG_S, 2.2, 'up')
lc.text(1272, 136, '单机单卡 uni · 单机多卡 mp', 9, lc.C_MUTE, 'start', maxw=168,
        tag='ax1:l1')
lc.text(1272, 152, '多机 ray · 自定义 qualname', 9, lc.C_MUTE, 'start', maxw=168,
        tag='ax1:l2')
lc.text(1272, 172, '换编排 = 换一个', 9.5, lc.C_ENG_S, 'start', True, maxw=168, tag='ax1:l3')
lc.text(1272, 187, 'executor 类', 9.5, lc.C_ENG_S, 'start', True, maxw=168, tag='ax1:l4')
# 交叉点（spawn 边上）
lc.rect(1254, 303, 12, 12, lc.C_TXT, lc.C_TXT, rx=6, sw=1.0)
lc.text(1272, 302, '两条轴只在 spawn 边', 9, lc.C_TXT, 'start', True, maxw=168, tag='cross:l1')
lc.text(1272, 316, '交叉一次，此后互不相见', 9, lc.C_MUTE, 'start', maxw=168, tag='cross:l2')
# 轴二（硬件适配轴，绿）：交叉点 → 带二底边
lc.seg(AX_X, 316, AX_X, B2_Y + B2_H - 2, lc.C_GPU_S, 2.2, 'gp')
lc.text(1272, 348, '硬件适配轴', 10.5, lc.C_GPU_S, 'start', True, maxw=168, tag='ax2:t')
lc.text(1272, 366, 'CUDA · ROCm · CPU · XPU', 9, lc.C_MUTE, 'start', maxw=168, tag='ax2:l1')
lc.text(1272, 381, 'TPU + 开箱外（OOT）插件', 9, lc.C_MUTE, 'start', maxw=168, tag='ax2:l2')
lc.text(1272, 400, '加硬件 = 改 worker_cls', 9.5, lc.C_GPU_S, 'start', True, maxw=168,
        tag='ax2:l3')
lc.text(1272, 415, '一个字符串（平台插件解析）', 9, lc.C_MUTE, 'start', maxw=168, tag='ax2:l4')
# 正交的实测证据小卡
EV_X, EV_Y, EV_W, EV_H = 1272, 436, 168, 100
lc.rect(EV_X, EV_Y, EV_W, EV_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.2, dash=True)
lc.text(EV_X + 10, EV_Y + 18, '实测证据 · world_size=2', 8.5, lc.C_TXT, 'start', True,
        maxw=EV_W - 20, tag='ev:t')
lc.text(EV_X + 10, EV_Y + 36, 'rpc_calls=1 →', 8.5, lc.C_MUTE, 'start', maxw=EV_W - 20,
        tag='ev:l1')
lc.text(EV_X + 10, EV_Y + 50, 'reply_count=2', 8.5, lc.C_MUTE, 'start', maxw=EV_W - 20,
        tag='ev:l2')
lc.text(EV_X + 10, EV_Y + 66, 'ranks [0,1] 全收到', 8.5, lc.C_MUTE, 'start', maxw=EV_W - 20,
        tag='ev:l3')
lc.text(EV_X + 10, EV_Y + 84, '编排与 worker 面互不见面', 8.5, lc.C_MUTE, 'start',
        maxw=EV_W - 20, tag='ev:l4')
lc.text(1164, 596, '（两条轴止步于上两层——', 9, lc.C_MUTE, 'start', maxw=200, tag='axstop1')
lc.text(1164, 611, 'ModelRunner 对两问无感）', 9, lc.C_MUTE, 'start', maxw=200, tag='axstop2')

# ---------------- 代价链：一次 execute_model 穿 5 层 ----------------
COST_LBL_Y = 728
lc.text(MX, COST_LBL_Y, '代价：一次 execute_model 要穿 5 层间接——控制面一跳的 Python 行程（调试栈里多一层字符串解析）',
        9.5, lc.C_TXT, 'start', True, maxw=1380, tag='cost:t')
LAYERS = [
    ('Executor.execute_model', '薄封装 · abstract.py:L221-L227'),
    ('collective_rpc', '广播 · multiproc_executor.py:L354-L416'),
    ('wrapper __getattr__', '字符串透传 · worker_base.py:L333-L334'),
    ('Worker.execute_model', 'gpu_worker.py:L1019'),
    ('model_runner.execute_model', 'gpu_model_runner.py:L4166'),
]
LW, LGAP, LY, LH = 232, 36, COST_LBL_Y + 12, 52
for i, (t, s) in enumerate(LAYERS):
    x = MX + i * (LW + LGAP)
    lc.rect(x, LY, LW, LH, '#ffffff', lc.C_MUTE, rx=6, sw=1.2)
    lc.text(x + 12, LY + 21, t, 9, lc.C_TXT, 'start', True, maxw=LW - 24, tag='cost:c' + str(i))
    lc.text(x + 12, LY + 38, s, 7.5, lc.C_FAINT, 'start', maxw=LW - 24, tag='cost:cs' + str(i))
    if i < len(LAYERS) - 1:
        lc.seg(x + LW + 1, LY + LH / 2, x + LW + LGAP - 2, LY + LH / 2, lc.C_MUTE, 1.5, 'std')

# ---------------- 同一契约两种拓扑（缩略双卡） ----------------
TH_Y, TH_H, TH_W = 806, 96, 680
lc.rect(MX, TH_Y, TH_W, TH_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(MX + 14, TH_Y + 21, '同一契约 · 拓扑一 —— uni：同进程直调', 10, lc.C_TXT, 'start', True,
        maxw=TH_W - 28, tag='th1:t')
lc.text(MX + 14, TH_Y + 42, 'driver_worker 就在本进程 · run_method 同进程直调（serial_utils.py:L486-L514）',
        9, '#334155', 'start', maxw=TH_W - 28, tag='th1:l1')
lc.text(MX + 14, TH_Y + 60, '三分支与 mp 的 worker_busy_loop 同构——一跳即达，无第二进程', 9,
        '#334155', 'start', maxw=TH_W - 28, tag='th1:l2')
lc.text(MX + 14, TH_Y + 82, '世界小到没有 IPC：AsyncOutputFuture 延迟到 result() 才等 D2H', 8.5,
        lc.C_MUTE, 'start', maxw=TH_W - 28, tag='th1:l3')
MP_X = 760
lc.rect(MP_X, TH_Y, TH_W, TH_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(MP_X + 14, TH_Y + 21, '同一契约 · 拓扑二 —— mp：广播 MQ 一跳', 10, lc.C_TXT, 'start', True,
        maxw=TH_W - 28, tag='th2:t')
lc.text(MP_X + 14, TH_Y + 42, 'rpc_broadcast_mq（SHM）一次 enqueue 全读者可见（multiproc_executor.py:L997-L1022）',
        9, '#334155', 'start', maxw=TH_W - 28, tag='th2:l1')
lc.text(MP_X + 14, TH_Y + 60, '实测：rpc_calls=1 → reply_count=2 · ranks [0,1]——编排与 worker 面互不见面',
        9, '#334155', 'start', maxw=TH_W - 28, tag='th2:l2')
lc.text(MP_X + 14, TH_Y + 82, '星形不对称：下行 1 条广播通道、上行 N 条应答通道', 8.5,
        lc.C_MUTE, 'start', maxw=TH_W - 28, tag='th2:l3')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = 928
lx = MX
lc.rect(lx, LEG_Y - 9, 20, 13, lc.C_ENG_F, lc.C_ENG_S, rx=4, sw=1.4)
lc.text(lx + 26, LEG_Y + 2, '进程编排面（Executor）', 9, lc.C_TXT, 'start', maxw=200, tag='leg1')
lx += 26 + lc.tw('进程编排面（Executor）', 9) + 22
lc.rect(lx, LEG_Y - 9, 20, 13, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.4)
lc.text(lx + 26, LEG_Y + 2, '设备执行面（Worker · ModelRunner）', 9, lc.C_TXT, 'start',
        maxw=260, tag='leg2')
lx += 26 + lc.tw('设备执行面（Worker · ModelRunner）', 9) + 22
lc.seg(lx, LEG_Y - 3, lx + 20, LEG_Y - 3, lc.C_MUTE, 1.4, dash=True)
lc.text(lx + 26, LEG_Y + 2, 'spawn 边（两带交界）', 9, lc.C_TXT, 'start', maxw=160, tag='leg3')
lc.text(MX, LEG_Y + 26, 'verbatim vllm/v1/executor/abstract.py:L48-L92 · vllm/platforms/cuda.py:L308-L313 · '
        'vllm/v1/worker/worker_base.py:L40-L43 · vllm/v1/serial_utils.py:L486-L514 · vllm/v1/executor/multiproc_executor.py:L388',
        9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '广播/拓扑实测取自精简版 companion host 实测（真 spawn · 真 READY 握手 · 真 ZMQ 广播 seam）· '
        '行号基线 vLLM v0.27.1 · 三层组件与 13 站走线见本章章图（L2）',
        9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch17-fig-three-layers-three-questions.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
