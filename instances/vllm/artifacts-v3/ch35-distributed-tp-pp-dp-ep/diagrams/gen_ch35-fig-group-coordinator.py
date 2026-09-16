#!/usr/bin/env python3
"""ch35 机制图 · m1 GroupCoordinator 解剖：一个并行维度一个实例（figure_spec ch35-fig-group-coordinator）

放大自 L0 多实例视角 ② GroupCoordinator·双群组（L2 站 4）——
FIGURE-SYSTEM §3.3 正文机制图：架构归属回指 L0/L2，不另立架构画法。

claim：一个并行维度 = 一个 GroupCoordinator 实例：每组同时 new_group 出
device_group（NCCL 等设备后端）与 cpu_group（gloo）双群组按数据类型分流，
world_size>1 才挂平台 device_communicator 挑具体内核，TP 组独享 SHM
mq_broadcaster——TP/PP/DP/EP 就是四个同构实例、每维度一个进程级单例。

数字全部取自 explainer figure_spec.numbers（traces/m01+m02 实测 + pin 锚点）；
坐标由常量/循环计算；文本全 esc()；配色走 l0_common 角色色（同源强制）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1360, 952
MX = 64
BXR = 1296

DEFS = lc.DEFS


def chip(x_right, y, label, color):
    """右上角回指小片（虚线）。"""
    w = lc.tw(label, 9.5, True) + 14
    x = x_right - w
    lc.rect(x, y, w, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
    lc.text(x + w / 2, y + 14.5, label, 9.5, color, 'middle', True,
            maxw=w - 4, tag='chip:' + label[:10])
    return x


def sub_box(x, y, w, h, title, right_note, lines, stroke, fill, tag):
    """解剖面板内的成员框：标题(粗·色) + 右侧灰注 + 若干行。"""
    lc.rect(x, y, w, h, fill, stroke, rx=7, sw=1.6)
    lc.text(x + 13, y + 21, title, 12, lc.C_TXT, 'start', True,
            maxw=w - 26 - lc.tw(right_note, 8.5) - 8 if right_note else w - 26,
            tag=tag + ':t')
    if right_note:
        lc.text(x + w - 12, y + 21, right_note, 8.5, lc.C_FAINT, 'end',
                maxw=w - 26 - lc.tw(title, 12, True) - 40, tag=tag + ':rn')
    for i, ln in enumerate(lines):
        lc.text(x + 13, y + 42 + i * 17, ln, 9.5, '#334155', 'start',
                maxw=w - 24, tag=tag + ':' + ln[:10])


# ---------------- 标题区 ----------------
lc.text(MX, 36, 'GroupCoordinator 解剖：一个并行维度一个实例，双群组按数据类型分流', 16.5,
        lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 62,
        'cpu_group(gloo) 走元数据/对象/barrier，device_group 走张量集合与 P2P；world_size>1 才挂'
        ' device_communicator 挑内核；TP 组独享 SHM mq_broadcaster',
        10.5, lc.C_MUTE, 'start', maxw=1150, tag='subtitle')
chip(BXR, 12, '放大自 L2 站 4 · L0：多实例视角', lc.C_GPU_S)

# ---------------- 左：解剖主面板（以 TP 为例） ----------------
PX, PY, PW, PH = MX, 112, 724, 640
lc.rect(PX, PY, PW, PH, '#ffffff', lc.C_GPU_S, rx=12, sw=2.2)
lc.text(PX + 16, PY + 26, 'GroupCoordinator（以 TP 组为例）——一个并行维度的前台接待员', 13,
        lc.C_GPU_S, 'start', True, maxw=560, tag='panel:t')
lc.text(PX + PW - 14, PY + 26, 'vllm/distributed/parallel_state.py:L394-L533', 9,
        lc.C_FAINT, 'end', tag='panel:file')

# —— 类顶注释表（三套坐标的钥匙）——
IX, IY, IW = PX + 16, PY + 44, 404
lc.rect(IX, IY, IW, 152, '#f8fafc', lc.C_FAINT, rx=6, sw=1.0)
lc.text(IX + 12, IY + 19, '类顶注释表：rank / local_rank / rank_in_group 三套坐标的钥匙', 9.5,
        lc.C_TXT, 'start', True, maxw=IW - 24, tag='inset:t')
cols = [('Process', 62), ('Node', 56), ('Rank', 52), ('Local Rank', 92), ('Rank in Group', 110)]
rows = [(0, 0, 0, 0, 0), (1, 0, 1, 1, 1), (2, 1, 2, 0, 2), (3, 1, 3, 1, 3)]
ty = IY + 38
cx = IX + 14
for name, cw_ in cols:
    lc.text(cx, ty, name, 8.5, lc.C_MUTE, 'start', True, maxw=cw_, tag='tbl:h' + name)
    cx += cw_
for r, row in enumerate(rows):
    ty += 24
    cx = IX + 14
    if r == 2:  # Node 换界行——2 节点例的分界
        lc.seg(IX + 10, ty - 17, IX + IW - 10, ty - 17, lc.C_FAINT, 1.0)
    for v, (name, cw_) in zip(row, cols):
        bold = name in ('Rank', 'Local Rank', 'Rank in Group')
        lc.text(cx, ty, str(v), 9.5, lc.C_TXT if bold else '#334155', 'start', bold,
                maxw=cw_, tag=f'tbl:r{r}{name}')
        cx += cw_

# —— TP=2 实测小盒 ——
QX = IX + IW + 14
QW = PX + PW - 16 - QX
lc.rect(QX, IY, QW, 152, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.2)
lc.text(QX + QW / 2, IY + 19, 'TP=2 每层 all_reduce（实测）', 9.5, lc.C_TXT, 'middle', True,
        maxw=QW - 12, tag='tp2:t')
lc.text(QX + QW / 2, IY + 40, 'rank0 出部分和 1.0', 9.5, '#334155', 'middle', tag='tp2:l1')
lc.text(QX + QW / 2, IY + 58, 'rank1 出部分和 2.0', 9.5, '#334155', 'middle', tag='tp2:l2')
lc.text(QX + QW / 2, IY + 82, '→ all_reduce →', 10, lc.C_GPU_S, 'middle', True, tag='tp2:l3')
lc.text(QX + QW / 2, IY + 103, '两 rank 同得 3.0', 10, lc.C_TXT, 'middle', True, tag='tp2:l4')
lc.text(QX + QW / 2, IY + 128, 'host：device 后端同为 gloo', 8.5, lc.C_MUTE, 'middle',
        maxw=QW - 10, tag='tp2:l5')
lc.text(QX + QW / 2, IY + 143, '（真 pin=NCCL——host seam）', 8.5, lc.C_MUTE, 'middle',
        maxw=QW - 10, tag='tp2:l6')

# —— 三件成员框 ——
SX, SW = PX + 16, PW - 32
sub_box(SX, PY + 216, SW, 88, 'cpu_group（gloo）——元数据/对象通道', 'new_group(backend="gloo")',
        ['· CPU 小活：send_object 装箱单（PP 段间 metadata）· 对象广播 · barrier',
         '· 构造即建：每组 new_group 两次，cpu 侧固定 gloo（parallel_state.py:L455-L470）'],
        lc.C_ZMQ_S, lc.C_ZMQ_F, 'cpu')
sub_box(SX, PY + 318, SW, 88, 'device_group——张量通道（真 pin=NCCL · host 实跑 gloo）', 'new_group(设备后端)',
        ['· 张量集合：all_reduce / all_gatherv / reduce_scatterv（TP/EP 每层的大件）',
         '· 点对点：isend / irecv（PP 段间接力）'],
        lc.C_GPU_S, lc.C_GPU_F, 'dev')
sub_box(SX, PY + 420, SW, 100, 'device_communicator——真正挑内核的下沉点',
        'world_size>1 才挂（L505-L517）',
        ['· use_device_communicator 且 ws>1 才解析：current_platform.get_device_communicator_cls()',
         '· 七级回退链对调用方透明——挑内核是它的事，模型层只管喊 all_reduce（下节展开）'],
        lc.C_ENG_S, '#ffffff', 'dc')
# mq_broadcaster——TP 独享件
sub_box(SX, PY + 534, SW, 86, 'mq_broadcaster · MessageQueue（SHM 环形缓冲）',
        '仅 TP 组创建（has_mq=true）',
        ['· broadcast_object 的快路：TP 组内小对象广播绕开 gloo 线路',
         '· PP/DP/EP 组无此件（实测 recorded：仅出现 tp:mq 一个键）'],
        lc.C_ZMQ_S, '#ffffff', 'mq')

# ---------------- 右：四个同构实例 ----------------
RX, RW = PX + PW + 28, BXR - (PX + PW + 28)
lc.text(RX + RW / 2, PY + 10, '四个同构实例——每维度一个', 12.5, lc.C_TXT, 'middle', True,
        tag='rcol:t')
lc.text(RX + RW / 2, PY + 28, '长相完全相同、各管自己那摊 rank；进程级单例 get_*_group()',
        9, lc.C_MUTE, 'middle', maxw=RW, tag='rcol:s')
dims = [
    ('TP · get_tp_group()', 'tp:0', '独享 mq_broadcaster（has_mq=true）', True),
    ('PP · get_pp_group()', 'pp:0', '段间接力 isend/irecv 走它', False),
    ('DP · get_dp_group()', 'dp:0', '负载小票 / wave 共识的 dp 线路', False),
    ('EP · get_ep_group()', 'ep:0', '专家重排 all_gatherv / reduce_scatterv', False),
]
RBH, RPITCH = 96, 112
for i, (t, name, ln, has_mq) in enumerate(dims):
    y = PY + 48 + i * RPITCH
    lc.rect(RX, y, RW, RBH, '#ffffff', lc.C_GPU_S, rx=7, sw=1.6)
    lc.text(RX + 13, y + 22, t, 12, lc.C_TXT, 'start', True, maxw=RW - 120, tag='rc%d:t' % i)
    if has_mq:
        bw = lc.tw('SHM mq', 8.5, True) + 12
        lc.rect(RX + RW - bw - 8, y + 7, bw, 17, lc.C_ZMQ_F, lc.C_ZMQ_S, rx=8, sw=1.0)
        lc.text(RX + RW - bw / 2 - 8, y + 19.5, 'SHM mq', 8.5, lc.C_ZMQ_S, 'middle', True,
                maxw=bw - 2, tag='rc:bdg')
    lc.text(RX + 13, y + 44, 'unique_name "%s" · world 内多组并存' % name, 9.5, '#334155',
            'start', maxw=RW - 24, tag='rc%d:l1' % i)
    lc.text(RX + 13, y + 63, ln, 9.5, '#334155', 'start', maxw=RW - 24, tag='rc%d:l2' % i)
    lc.text(RX + 13, y + 82, 'vllm/distributed/parallel_state.py', 8.5, lc.C_FAINT, 'start',
            maxw=RW - 24, tag='rc%d:f' % i)

# ---------------- 底部建组账 ----------------
BY = PY + PH + 22
lc.rect(MX, BY, BXR - MX, 78, '#f8fafc', lc.C_MUTE, rx=8, sw=1.1)
lc.text(MX + 16, BY + 22, '8 GPU 例建组账（实测）：TP 4 组 · PCP 8 组 · PP 4 组 · DP 4 组 · EP 2 组 '
        '→ 22 个组 × 2 双群组 = 44 个 ProcessGroup', 10.5, lc.C_TXT, 'start', True,
        maxw=BXR - MX - 32, tag='acct:l1')
lc.text(MX + 16, BY + 44, '「一个并行维度一个实例」的配额代价：ProcessGroup 是稀缺资源，维度多了有配额压力'
        '——双群组是结构性翻倍（每组必配 cpu+device 两条线）', 9.5, '#334155', 'start',
        maxw=BXR - MX - 32, tag='acct:l2')
lc.text(MX + 16, BY + 62, '一个并行维度 = 一个前台接待员：同时管 cpu 线路（gloo，喊话/传纸条/对表）和 '
        'device 线路（NCCL，搬大件张量）——搬什么货走哪条线他来分', 9.5, lc.C_MUTE, 'start',
        maxw=BXR - MX - 32, tag='acct:l3')

# ---------------- 图例 + 页脚 ----------------
LY = BY + 78 + 26
swatches = [
    (lc.C_ZMQ_S, 'cpu 线路（gloo·元数据/对象）'),
    (lc.C_GPU_S, 'device 线路（真 pin=NCCL·张量）'),
    (lc.C_ENG_S, '内核下沉点 device_communicator'),
]
lx0 = MX
for color, name in swatches:
    lc.rect(lx0, LY - 9, 16, 11, '#ffffff', color, rx=3, sw=1.6)
    lc.text(lx0 + 21, LY + 1, name, 9.5, lc.C_TXT, 'start', tag='leg:' + name[:6])
    lx0 += 21 + lc.tw(name, 9.5) + 22
lc.text(MX, LY + 24,
        '行号基线 vLLM v0.27.1（6e448d0ea）· 实测值标 host（单机 8 进程 gloo——集合通信图语义等价；'
        '真 pin 多卡后端 NCCL）· 框内灰字 = 规范源码路径',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='footer')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch35-fig-group-coordinator.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
