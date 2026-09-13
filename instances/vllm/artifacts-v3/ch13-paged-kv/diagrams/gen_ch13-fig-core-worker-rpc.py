#!/usr/bin/env python3
"""ch13 机制图 9 · EngineCore 进程 ↔ worker 进程：一拍两次 RPC（template: swimlane/时序）

放大自 L0「EngineCore → GPU 执行臂」之间那一段（FIGURE-SYSTEM §3.3 回指规则；
L0 上它是 A 列 Scheduler 到 B 列 Executor→Worker 的 SchedulerOutput 箭头）。

claim：mp 部署下 EngineCore.step() 一拍两次跨进程 RPC——下行走一条广播 MQ
（一次入队、N 个 worker 各读一份），上行只由 output_rank 回一条；两次 RPC
之间不空等，EngineCore 在 CPU 上算结构化输出掩码；消息体里永远没有 GPU 张量。

事实来源：本章档案 supplement.B_ipc_timeline（topology / step_timeline T0-T7 /
ownership / figure_brief），逐条对 pin vLLM v0.27.1 源码现核（见各锚点注释）。
坐标全部由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

# ============================ 画布与坐标常量 ============================
W = 1500
MX, BXR = 56, 1444

EC_X, WK_X = 452, 1044          # 两根生命线（EngineCore / worker 进程群）
TRACK_HW = 4                     # 生命线轨道半宽
BAR_HW = 13                      # 活动条半宽（骑在生命线上）
GRID_X0, GRID_X1 = EC_X - 14, WK_X + 14
NB_EC = (120, EC_X - 20)         # EngineCore 侧注记框（左）
NB_WK = (WK_X + 22, 1440)        # worker 侧注记框（右）
GUT_X, PILL_W, PILL_H = MX, 52, 22
MID = (EC_X + WK_X) / 2          # 消息标签中心 x = 748
GAP_W = WK_X - EC_X              # 标签可用宽 592

# 角色色即身份（FIGURE-SYSTEM §0.2）：EngineCore 恒橙、worker/GPU 执行臂恒绿。
C_EC, F_EC = lc.C_ENG_S, lc.C_ENG_F
C_WK, F_WK = lc.C_GPU_S, lc.C_GPU_F
C_CH, F_CH = lc.C_ZMQ_S, lc.C_ZMQ_F     # 消息通道（MessageQueue）= 紫（线格式/序列化边界角色）
C_KEY = lc.C_KV_S                       # 块账 / token 账的键色（同 L0 显存账本列）
C_FAINT, C_MUTE, C_TXT = lc.C_FAINT, lc.C_MUTE, lc.C_TXT
C_TRACK = '#e2e8f0'

EXTRA_DEFS = ('<defs>'
              '<marker id="eng" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              'markerHeight="4.6" orient="auto">'
              f'<path d="M0,0 L10,3 L0,6 Z" fill="{C_EC}"/></marker>'
              '<marker id="wkg" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              'markerHeight="4.6" orient="auto">'
              f'<path d="M0,0 L10,3 L0,6 Z" fill="{C_WK}"/></marker>'
              '<marker id="chm" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              'markerHeight="4.2" orient="auto">'
              f'<path d="M0,0 L10,3 L0,6 Z" fill="{C_CH}"/></marker>'
              '</defs>')


def t(x, y, s, fs, fill, anchor='middle', bold=False, maxw=None, tag=''):
    lc.text(x, y, s, fs, fill, anchor, bold, maxw=maxw, tag=tag)


def box(x0, y0, x1, y1, fill, stroke, dash=False, sw=1.3, rx=7):
    lc.rect(x0, y0, x1 - x0, y1 - y0, fill, stroke, rx=rx, sw=sw, dash=dash)


def grid(y):
    """共享时间轴 gridline：贯穿两根生命线（时序规约 §0.2）"""
    lc.seg(GRID_X0, y, GRID_X1, y, '#dfe6ef', 1.2)


def pill(y0, label, sub):
    """左槽：拍号胶囊 + 方向小字"""
    lc.rect(GUT_X, y0 + 10, PILL_W, PILL_H, lc.C_BADGE_F, C_EC, rx=PILL_H / 2, sw=1.1)
    t(GUT_X + PILL_W / 2, y0 + 25, label, 9.5, C_EC, 'middle', True, tag='pill:' + label[:6])
    t(GUT_X + PILL_W / 2, y0 + 46, sub, 8.5, C_MUTE, 'middle', maxw=PILL_W + 40,
      tag='pillsub:' + sub)


def msg_row(y0, h, up, name, mq, payload, carrier, anchor):
    """一条水平消息（时序规约：禁折线、共享时间轴、直穿中间生命线以外的空档）。"""
    y = y0 + h / 2
    grid(y0)
    x1, x2 = (WK_X, EC_X) if up else (EC_X, WK_X)
    lc.seg(x1, y, x2, y, C_EC if not up else C_WK, 2.2, 'eng' if not up else 'wkg')
    t(MID, y - 27, name, 10.5, C_EC if not up else C_WK, 'middle', True, maxw=GAP_W, tag='m:' + name[:16])
    t(MID, y - 12, mq, 9, C_CH, 'middle', maxw=GAP_W, tag='mq:' + mq[:14])
    t(MID, y + 17, payload, 9, '#334155', 'middle', maxw=GAP_W, tag='pl:' + payload[:14])
    t(MID, y + 31, carrier, 8.6, C_MUTE, 'middle', maxw=GAP_W, tag='cr:' + carrier[:14])
    t(MID, y + 45, anchor, 8.2, C_FAINT, 'middle', maxw=GAP_W, tag='anc:' + anchor[:14])
    return y


# ============================ 标题带 ============================
t(MX, 34, '一拍两次 RPC：EngineCore 进程与 worker 进程之间，到底传过什么、谁写谁读',
  16.5, C_TXT, 'start', True, maxw=1000, tag='title')
t(MX, 58, 'mp 部署（MultiprocExecutor）下一个 step 的完整往返：下行一条广播、上行只由 output_rank 回一条，'
          '两次之间夹一段纯 CPU 掩码——消息里永远没有 GPU 张量',
  10.5, C_MUTE, 'start', maxw=1030, tag='subtitle')
_badge = '放大自 L0 · EngineCore → GPU 执行臂之间'
_bw = lc.chip_w(_badge)
lc.rect(BXR - _bw, 12, _bw, 20, '#ffffff', C_MUTE, rx=9, sw=1.1, dash=True)
t(BXR - _bw / 2, 26.5, _badge, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_bw - 4, tag='badge')

# ============================ 部署门（uni vs mp） ============================
GATE_Y, GATE_H = 74, 48
box(MX, GATE_Y, BXR, GATE_Y + GATE_H, '#f8fafc', '#cbd5e1', rx=7, sw=1.2)
t(MX + 14, GATE_Y + 20, '有没有这堵墙，由 distributed_executor_backend 决定：', 10, C_TXT,
  'start', True, maxw=380, tag='gate:t')
t(MX + 14, GATE_Y + 38,
  '单卡（world_size == 1）默认 uni：UniProcExecutor 同进程直呼 driver_worker，零 MQ，本图整条时间轴不成立',
  9, C_MUTE, 'start', maxw=700, tag='gate:uni')
t(MX + 740, GATE_Y + 20, '多卡默认 mp：worker 起成子进程，一拍两次跨进程往返', 9.5, C_EC,
  'start', True, maxw=440, tag='gate:mpt')
t(MX + 740, GATE_Y + 38, '——本图主时间轴画的就是 mp（vllm/config/parallel.py:L910+L955-L956）'
                         '（vllm/v1/executor/abstract.py:L69-L76）',
  9, C_MUTE, 'start', maxw=640, tag='gate:mp')

# ============================ 生命线名牌 ============================
PLATE_Y, PLATE_H = 136, 38
LIFE_Y0 = PLATE_Y + PLATE_H + 6                     # 180
plates = [
    (EC_X, 'EngineCore 进程', 'busy loop · 只做调度 + 执行 · Executor 层在此进程内',
     '① schedule ② execute_model ③ get_grammar_bitmask ④ sample_tokens ⑤ update_from_output', C_EC, F_EC),
    (WK_X, 'worker 进程群（GPU）', '每个 rank 一个子进程 · 回信的只有 output_rank',
     'output_rank = world_size − tp×pcp（最后一个 PP 级的第一个 TP worker；TP=8/PP=4 → 24）', C_WK, F_WK),
]
PLATE_W, PLATE_H2 = 340, 38
for cx, name, sub, sub2, col, fill in plates:
    box(cx - PLATE_W / 2, PLATE_Y, cx + PLATE_W / 2, PLATE_Y + PLATE_H2, fill, col, sw=1.8)
    t(cx, PLATE_Y + 17, name, 11.5, col, 'middle', True, maxw=PLATE_W - 16, tag='plate:' + name[:10])
    t(cx, PLATE_Y + 32, sub, 8.5, C_MUTE, 'middle', maxw=PLATE_W - 12, tag='plate2:' + sub[:12])
# 名牌下的身份小字（挂在名牌两侧，不占名牌）
t(EC_X - PLATE_W / 2 - 10, PLATE_Y + 16, '橙 = EngineCore 进程', 8.5, C_EC, 'end',
  maxw=160, tag='tag:ec')
t(EC_X - PLATE_W / 2 - 10, PLATE_Y + 32, '本书图系恒用角色色', 8.5, C_FAINT, 'end',
  maxw=160, tag='tag:ec2')
t(WK_X + PLATE_W / 2 + 10, PLATE_Y + 16, '绿 = worker / GPU 执行臂', 8.5, C_WK, 'start',
  maxw=170, tag='tag:wk')
t(WK_X + PLATE_W / 2 + 10, PLATE_Y + 32, '一个 rank 一个子进程', 8.5, C_FAINT, 'start',
  maxw=170, tag='tag:wk2')

# ============================ 时间轴行定义 ============================
ROWS = [('T0', 96, 'bar'), ('T1', 112, 'msg'), ('T2', 88, 'band'), ('T3', 88, 'band'),
        ('T4', 104, 'msg'), ('T5', 112, 'msg'), ('T6', 100, 'bar'), ('T7', 116, 'msg'),
        ('收尾', 96, 'bar')]
RY = {}
_y = LIFE_Y0
for _name, _h, _kind in ROWS:
    RY[_name] = (_y, _h)
    _y += _h
LIFE_Y1 = _y

# 生命线轨道（画在最底层：消息箭头端点落在轨道内 = 已连接）
for cx in (EC_X, WK_X):
    box(cx - TRACK_HW, LIFE_Y0, cx + TRACK_HW, LIFE_Y1, C_TRACK, '#cbd5e1', rx=TRACK_HW, sw=0.8)

def note_box(x0, x1, y0, y1, lines, fill='#ffffff', stroke=C_MUTE, dash=False):
    """侧注记框：行距按框高均分（保证框内文字不互压、不越框）"""
    box(x0, y0, x1, y1, fill, stroke, dash=dash, sw=1.2)
    pad = 12.0
    inner0, inner1 = y0 + pad * 0.9, y1 - pad * 0.6
    lead = (inner1 - inner0) / max(len(lines), 1)
    for i, (s, fs, col, bold) in enumerate(lines):
        t(x0 + pad, inner0 + fs * 0.95 + i * lead, s, fs, col, 'start', bold,
          maxw=x1 - x0 - 2 * pad, tag='nb:' + s[:14])


def bar(cx, y0, y1, col, fill):
    """活动条：骑在生命线上（高 = 该活动占的逻辑时段 × 统一行距比例尺）"""
    lc.rect(cx - BAR_HW, y0, 2 * BAR_HW, y1 - y0, fill, col, rx=4, sw=1.5)


# ---------------- T0：EngineCore 进程内，打包 SchedulerOutput ----------------
y0, h = RY['T0']
grid(y0)
pill(y0, 'T0', '进程内')
bar(EC_X, y0 + 4, y0 + h - 8, C_EC, F_EC)
note_box(*NB_EC, y0 + 8, y0 + h - 8, [
    ('① schedule() → SchedulerOutput', 9.5, C_EC, True),
    ('差量协议：新请求全量 NewRequestData / 老请求', 8.5, '#334155', False),
    ('new_block_ids 增量 + new_block_ids_to_zero 清零账', 8.5, '#334155', False),
    ('vllm/v1/engine/core.py:L595 ·', 7.8, C_FAINT, False),
    ('vllm/v1/core/sched/scheduler.py:L1208-L1229', 7.8, C_FAINT, False),
])

# ---------------- T1：第一次 RPC 下行 ----------------
_t1y, _t1h = RY['T1']
msg_row(_t1y, _t1h, False,
        '② execute_model(scheduler_output, non_block=True)',
        '下行 ▸ rpc_broadcast_mq：一次入队、N 个 worker 各读一份（拿 Future 立即返回）',
        '载荷 SchedulerOutput（差量）：新请求全量档案 / 老请求增量电报 / 清零账',
        '载体：SHM 环形缓冲内的 pickle 字节；≥1MiB 张量走 OOB，总量超 16MiB 溢出改走 zmq',
        'vllm/v1/engine/core.py:L596 · vllm/v1/executor/multiproc_executor.py:L321-L331+L388')
note_box(*NB_EC, _t1y + 6, _t1y + _t1h - 6, [
    ('Executor 层就在 EngineCore 进程里', 9.5, C_EC, True),
    ('MultiprocExecutor 持有 SHM 队列本身：下行广播 MQ +', 8.5, '#334155', False),
    ('每 worker 一条应答 MQ + futures_queue——消息的入队、', 8.5, '#334155', False),
    ('出队与配对全发生在这个进程', 8.5, '#334155', False),
    ('vllm/v1/executor/multiproc_executor.py:L103+L157+L585', 7.8, C_FAINT, False),
])

# ---------------- T2+T3：两次 RPC 之间的窗口（掩码 ∥ 前向） ----------------
t2y, t2h = RY['T2']
t3y, t3h = RY['T3']
t4y = RY['T4'][0]
BX0, BX1 = EC_X - BAR_HW, WK_X + BAR_HW
box(BX0, t2y, BX1, t4y, '#fdfaf6', '#fdba74', dash=True, sw=1.2)          # 两次 RPC 之间的窗口
lc.rect(BX0, t3y, BX1 - BX0, t4y - t3y, '#fbe8d5', 'none', rx=0, sw=0)     # 重叠窗口（暖色带）
grid(t2y)
grid(t3y)
bar(EC_X, t2y + 3, t4y - 2, C_EC, F_EC)                                    # ③ 掩码（跨整个窗口）
bar(WK_X, t3y + 3, t4y - 2, C_WK, F_WK)                                    # T3 前向
pill(t2y, 'T2', '进程内')
pill(t3y, 'T3', '进程内')
t(EC_X + 20, t2y + 16, '③ 掩码', 8.5, C_EC, 'start', maxw=80, tag='eclab2')
t(WK_X - 20, t3y + 14, 'T3 前向', 8.5, C_WK, 'end', maxw=80, tag='wklab3')
t(MID, t2y + 38, '③ get_grammar_bitmask：结构化输出掩码在 EngineCore 的 CPU 上算（numpy int32）',
  9.5, C_TXT, 'middle', maxw=GAP_W - 40, tag='b1')
t(MID, t2y + 56, '此时消息在途、worker 在读队列解包预处理——三条线同时在跑',
  8.8, C_MUTE, 'middle', maxw=GAP_W - 40, tag='b2')
t(MID, t3y + 36, '重叠窗口：worker 前向（GPU）已 launch，掩码（CPU）还没算完',
  9.5, C_TXT, 'middle', maxw=GAP_W - 180, tag='b3')
t(MID, t3y + 54, '两边互不等——这就是「两次 RPC 之间不空等」的全部意义',
  8.8, C_MUTE, 'middle', maxw=GAP_W - 180, tag='b4')
note_box(*NB_EC, t2y + 6, t2y + t2h - 6, [
    ('③ 掩码这笔 CPU 活排在这里', 9.5, C_EC, True),
    ('它不占 GPU、也不等消息回来——中间这段空档', 8.5, '#334155', False),
    ('正是给它的', 8.5, '#334155', False),
    ('vllm/v1/engine/core.py:L597 ·', 7.8, C_FAINT, False),
    ('vllm/v1/core/sched/output.py:L286-L291', 7.8, C_FAINT, False),
])
note_box(*NB_WK, t2y + 6, t4y - 6, [
    ('T3 · worker 前向（绿条）之前与之内做的六件事', 9.5, C_WK, True),
    ('worker_busy_loop dequeue → GPUWorker.execute_model', 8.4, '#334155', False),
    ('→ _update_states：block_ids 差量 extend（恢复者整表替换）', 8.4, '#334155', False),
    ('→ input_batch.block_table.append_row 写 CPU 页表行', 8.4, '#334155', False),
    ('→ 新块清零／CoW 块拷贝 → _prepare_inputs（块表先行拷贝）', 8.4, '#334155', False),
    ('→ 前向 launch → 暂存 ExecuteModelState → return None', 8.4, '#334155', False),
    ('vllm/v1/executor/multiproc_executor.py:L1001 · gpu_worker.py:L1019', 7.8, C_FAINT, False),
    ('vllm/v1/worker/gpu_model_runner.py:L4166+L4516-L4535', 7.8, C_FAINT, False),
])

# ---------------- T4：第一次应答上行 ----------------
msg_row(*RY['T4'], True,
        '(SUCCESS, None) —— 第一次应答',
        '上行 ▸ worker_response_mq：每 worker 一条 MessageQueue(1,1)，executor 是唯一读者',
        '载荷 None —— 语义 = 前向已 launch、采样待第二次 RPC，不是「没结果」',
        '载体 pickle 元组；只有 output_rank（= world_size − tp×pcp）回写，其余 worker 算完就扔',
        'vllm/v1/executor/multiproc_executor.py:L950-L967+L1012-L1013 · vllm/v1/engine/core.py:L602')

# ---------------- T5：第二次 RPC 下行 ----------------
msg_row(*RY['T5'], False,
        '④ sample_tokens(grammar_output) —— 第二次 RPC',
        '下行 ▸ 同一条 rpc_broadcast_mq（两次下行复用同一通道；executor 按 futures_queue 配对）',
        '载荷 GrammarOutput：结构化输出请求 id + int32 掩码数组（≥1MiB 走 OOB 共享内存旁路）',
        '载体同 T1 那条广播 MQ 的 pickle 字节；应答仍走各自的 worker_response_mq',
        'vllm/v1/engine/core.py:L604 · vllm/v1/executor/multiproc_executor.py:L333-L343')

# ---------------- T6：worker 进程内兑现采样 ----------------
y0, h = RY['T6']
grid(y0)
pill(y0, 'T6', '进程内')
bar(WK_X, y0 + 5, y0 + h - 8, C_WK, F_WK)
note_box(*NB_WK, y0 + 6, y0 + h - 6, [
    ('T6 · sample_tokens：第二次下行在 worker 侧的兑现', 9.5, C_WK, True),
    ('解包 ExecuteModelState（状态机保证两次调用严格交替）', 8.4, '#334155', False),
    ('→ apply_grammar_bitmask 盖掩码 → _sample → D2H 拷回', 8.4, '#334155', False),
    ('→ 组装 ModelRunnerOutput（token ids 刻意用 list 不用 tensor）', 8.4, '#334155', False),
    ('vllm/v1/worker/gpu_worker.py:L1012-L1015 · gpu_model_runner.py:L4553-L4590', 7.8, C_FAINT, False),
])
note_box(*NB_EC, y0 + 6, y0 + h - 6, [
    ('这段 EngineCore 在等', 9.5, C_EC, True),
    ('future.result() 排干更早的 future 后，阻塞等这条', 8.5, '#334155', False),
    ('应答——忙循环这一段没有别的活（所以掩码才要提前算）', 8.5, '#334155', False),
])

# ---------------- T7：第二次应答上行 ----------------
msg_row(*RY['T7'], True,
        '(SUCCESS, ModelRunnerOutput) —— 第二次应答',
        '上行 ▸ 各自 worker_response_mq → executor dequeue → future.result() 返回',
        '载荷 ModelRunnerOutput：sampled_token_ids（list[list[int]]）+ logprobs …',
        '载体 pickle；原注释：要序列化送给调度进程，对 tensor 太贵所以用 list',
        'vllm/v1/executor/multiproc_executor.py:L394-L416 · vllm/v1/outputs.py:L258-L259+L271')

# ---------------- 收尾：EngineCore 记账 ----------------
y0, h = RY['收尾']
grid(y0)
grid(y0 + h)
pill(y0, '收尾', '进程内')
bar(EC_X, y0 + 5, y0 + h - 8, C_EC, F_EC)
note_box(*NB_EC, y0 + 8, y0 + h - 8, [
    ('⑤ update_from_output（T7 之后，进程内）', 9.5, C_EC, True),
    ('状态推进 + 组装 EngineCoreOutputs → output_queue', 8.5, '#334155', False),
    ('那是 EngineCore→前端那堵墙，不在此图（第 5 / 7 章已讲过）', 8.5, C_MUTE, False),
    ('vllm/v1/engine/core.py:L609-L611+L1435-L1444', 7.8, C_FAINT, False),
])
note_box(*NB_WK, y0 + 8, y0 + h - 8, [
    ('T7 之后 · worker 侧', 9.5, C_WK, True),
    ('应答交回自己的 worker_response_mq 后，回 worker_busy_loop', 8.5, '#334155', False),
    ('阻塞等下一条下行——一拍的终点就是下一拍的起点', 8.5, '#334155', False),
])

# ---------------- 时间轴端点标注 ----------------
t(MX, LIFE_Y1 + 18, '↓ 时间（逻辑顺序轴：T0–T7 等距，不是实测耗时——一拍内谁长谁短不在此图，'
                    '逐拍耗时实测第 9 章已讲）', 8.6, C_MUTE, 'start', maxw=900, tag='axis')


def section(y0, h, title, tcol=C_TXT, fill='#f8fafc', stroke='#cbd5e1', dash=False):
    box(MX, y0, BXR, y0 + h, fill, stroke, dash=dash, sw=1.3)
    t(MX + 14, y0 + 22, title, 10.5, tcol, 'start', True, maxw=BXR - MX - 28,
      tag='sec:' + title[:14])
    return y0 + 22


# ---------------- B1：两条 MQ 与统一载体 ----------------
B1Y = LIFE_Y1 + 40
B1H = 106
section(B1Y, B1H, '消息通道：两条 MQ、一套载体')
LCX, RCX = MX + 14, MX + 720
t(LCX, B1Y + 42, '下行 rpc_broadcast_mq：MessageQueue(world_size, local_world_size)，', 8.6,
  '#334155', 'start', maxw=660, tag='b1l1')
t(LCX, B1Y + 58, '一次 enqueue 全 worker 可读；句柄 export_handle() 随 spawn 传进每个子进程（仅 DP 组 leader 建）',
  8.6, '#334155', 'start', maxw=660, tag='b1l2')
t(RCX, B1Y + 42, '上行 worker_response_mq：每 worker 一条 MessageQueue(1,1)——自己是写者、executor 是唯一读者',
  8.6, '#334155', 'start', maxw=640, tag='b1r1')
t(RCX, B1Y + 58, '句柄放进 READY 报到信，父进程 create_from_handle(handle, 0) 重建读者侧',
  8.6, '#334155', 'start', maxw=640, tag='b1r2')
t(LCX, B1Y + 78, '载体统一：pickle protocol 5；CPU 张量 ≥1MiB 走 out-of-band buffer（_reduce_tensor）；'
                 '总量超 VLLM_MQ_MAX_CHUNK_BYTES_MB=16MiB 标 overflow 改走 zmq XPUB/SUB（本地 ipc://、远端 tcp://）；'
                 'MessageQueue 自身默认 24MiB，默认值注释点名的尺寸依据就是 grammar bitmask（1024 请求）',
  8.6, C_CH, 'start', maxw=BXR - MX - 28, tag='b1c')
t(LCX, B1Y + 94, 'vllm/v1/executor/multiproc_executor.py:L131-L157+L580-L585+L725-L745 · '
                 'vllm/distributed/device_communicators/shm_broadcast.py:L465-L476+L824-L881 · vllm/envs.py:L226-L227',
  7.8, C_FAINT, 'start', maxw=BXR - MX - 28, tag='b1a')

# ---------------- B2：ownership 谁写谁读 ----------------
B2Y = B1Y + B1H + 16
B2H = 88
section(B2Y, B2H, 'ownership：谁写谁读（同一条消息里的东西，权限完全不对称）', tcol=C_KEY,
        fill='#f7fdfe', stroke=C_KEY)
for _i, _s in enumerate([
        'SchedulerOutput：调度器进程独占写、worker 只读——worker 侧各持一份请求档案缓存，所以每拍只发差量',
        'token id 不对称：prompt 随首帧 NewRequestData 全量过线一次；生成的 token 由 worker 自己采、只经 ModelRunnerOutput 回程一次',
        '输入张量不跨进程：worker 用 block_id + num_scheduled_tokens 在本地缓冲重建（block_id 是两个进程唯一的共享键）']):
    t(MX + 14, B2Y + 44 + _i * 17, _s, 8.6, '#334155', 'start', maxw=BXR - MX - 28, tag='b2:%d' % _i)

# ---------------- B3：两个对照（uni / KV connector） ----------------
B3Y = B2Y + B2H + 16
B3H = 104
P1X1 = MX + 660
P2X0 = MX + 680
box(MX, B3Y, P1X1, B3Y + B3H, '#ffffff', C_MUTE, dash=True, sw=1.2)
box(P2X0, B3Y, BXR, B3Y + B3H, '#ffffff', C_MUTE, dash=True, sw=1.2)
t(MX + 14, B3Y + 22, '对照一 · uni 后端：这堵墙根本不存在', 9.8, C_MUTE, 'start', True,
  maxw=P1X1 - MX - 28, tag='p1t')
t(MX + 14, B3Y + 44, 'UniProcExecutor 直接 run_method 调本进程里的 driver_worker：', 8.5,
  '#334155', 'start', maxw=P1X1 - MX - 28, tag='p1a')
t(MX + 14, B3Y + 62, '没有 MQ、没有序列化、没有 output_rank——单卡默认部署就是它（本图是 mp 的世界）',
  8.5, '#334155', 'start', maxw=P1X1 - MX - 28, tag='p1b')
t(MX + 14, B3Y + 82, 'vllm/v1/executor/uniproc_executor.py:L91-L106', 7.8, C_FAINT, 'start',
  maxw=P1X1 - MX - 28, tag='p1c')
t(P2X0 + 14, B3Y + 22, '对照二 · 配了 KV connector：应答改成全 rank 回、再聚合', 9.8, C_MUTE,
  'start', True, maxw=BXR - P2X0 - 28, tag='p2t')
t(P2X0 + 14, B3Y + 44, 'output_rank 置 None → 每个 rank 各回一份；executor 收齐后 aggregate',
  8.5, '#334155', 'start', maxw=BXR - P2X0 - 28, tag='p2a')
t(P2X0 + 14, B3Y + 62, '取 output_rank 那份为主、合并各 rank 的 kv_connector_output', 8.5,
  '#334155', 'start', maxw=BXR - P2X0 - 28, tag='p2b')
t(P2X0 + 14, B3Y + 82, 'vllm/v1/executor/multiproc_executor.py:L375-L382 · '
                       'vllm/distributed/kv_transfer/kv_connector/utils.py:L53-L72',
  7.8, C_FAINT, 'start', maxw=BXR - P2X0 - 28, tag='p2c')

# ---------------- 图例 ----------------
LEG_Y = B3Y + B3H + 30
_lx = MX
SW, SH = 20, 13
for _kind, _fillc, _col, _name in [
        ('bar', F_EC, C_EC, 'EngineCore 进程的活动条'),
        ('bar', F_WK, C_WK, 'worker / GPU 执行臂的活动条'),
        ('bar', F_CH, C_CH, '消息通道（MQ）与载体'),
        ('tint', '#fbe8d5', '#fdba74', '两边同时有活（时间重叠）')]:
    lc.rect(_lx, LEG_Y - 10, SW, SH, _fillc, _col, rx=3, sw=1.4)
    t(_lx + SW + 6, LEG_Y, _name, 9, C_TXT, 'start', maxw=220, tag='lg:' + _name[:8])
    _lx += SW + 6 + lc.tw(_name, 9) + 20
lc.seg(_lx, LEG_Y - 4, _lx + 30, LEG_Y - 4, C_EC, 2.0, 'eng')
t(_lx + 36, LEG_Y, '跨进程消息（水平直线，禁折线）', 9, C_TXT, 'start', maxw=220, tag='lg:msg')
_lx += 36 + lc.tw('跨进程消息（水平直线，禁折线）', 9) + 20
lc.seg(_lx, LEG_Y - 4, _lx + 30, LEG_Y - 4, C_MUTE, 1.4, dash=True)
t(_lx + 36, LEG_Y, '生命线（未点名活动 = 进程仍在，但这一段没写出来）', 9, C_TXT, 'start',
  maxw=340, tag='lg:life')

# ---------------- 图注：分工互指 + 结论 ----------------
CP_Y = LEG_Y + 26
t(MX, CP_Y, '本图只讲「装什么、谁写谁读」——与同题图分工：两段契约的时序语义见第 9 章、'
            'MQ 的装配与传输层见第 17 章、EngineCore→前端那堵墙第 5 / 7 章已讲',
  8.6, C_MUTE, 'start', maxw=BXR - MX, tag='cp1')
t(MX, CP_Y + 17, '一拍两次 RPC：下行一条广播、上行只由 output_rank 回一条，中间夹一段纯 CPU 掩码；'
                 '消息里永远没有 GPU 张量——块账靠 block_id 对，token 账靠 prompt 首帧与 sampled_token_ids 各过一次线',
  9.5, C_TXT, 'start', True, maxw=BXR - MX, tag='cap')

H = CP_Y + 17 + 22
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch13-fig-core-worker-rpc.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
