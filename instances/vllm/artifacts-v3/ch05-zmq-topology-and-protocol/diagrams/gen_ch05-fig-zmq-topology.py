#!/usr/bin/env python3
"""ch05 机制图 · ZMQ 全拓扑：四扇门与一次性握手（explainer m1 figure_spec ch05-fig-zmq-topology）

放大自 L0 紫色 ZMQ 边界带（= 本章 L2 站 1-3 的三半边装配）——
FIGURE-SYSTEM §3.3 正文机制图：架构归属回指 L0/L2，不另立架构画法。

claim：进出不对称——下行 client ROUTER(bind)→engine DEALER(connect×每前端一条,
identity=engine_index 2 字节小端)，上行 engine PUSH(connect×每前端一条)→client
PULL(bind)，全部 socket HWM=0，且每条 DEALER 必须先发一条 ready 认亲帧 ROUTER
才认识它；旁边另有一条一次性握手旁路（HELLO→地址集→READY）。

数字全部取自 explainer figure_spec.numbers（topology_probe 实测 + pin 锚点）；
坐标由常量/循环计算；文本全 esc()；配色走 l0_common 语系（同源强制）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1280, 912
MX = 96
BXR = 1184                      # 内容右缘（进程框右缘）

# 紫色箭头 marker（ready / 握手步）——l0_common 语系内的补色
DEFS = lc.DEFS.replace('</defs>',
                       '<marker id="zq" viewBox="0 0 10 6" refX="9" refY="3" '
                       'markerWidth="6.5" markerHeight="4.6" orient="auto">'
                       f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_ZMQ_S}"/></marker>'
                       '</defs>')


def chip(x_right, y, label, color):
    """右上角回指小片（虚线）。"""
    w = lc.tw(label, 9.5, True) + 14
    x = x_right - w
    lc.rect(x, y, w, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
    lc.text(x + w / 2, y + 14.5, label, 9.5, color, 'middle', True,
            maxw=w - 4, tag='chip:' + label[:12])
    return x


def socket_box(x, y, w, title, lines, stroke, badges=(), dash=False, tag=''):
    """socket 端点框：标题(粗) + 徽标排 + 若干行 + 底部灰路径。高固定 96。"""
    lc.rect(x, y, w, 96, '#ffffff', stroke, rx=7, sw=1.6, dash=dash)
    bx = x + w - 8
    for b in badges:
        bw = lc.tw(b, 8.5, True) + 12
        bx -= bw
        lc.rect(bx, y + 6, bw, 17, lc.C_ZMQ_F, lc.C_ZMQ_S, rx=8, sw=1.0)
        lc.text(bx + bw / 2, y + 18.5, b, 8.5, lc.C_ZMQ_S, 'middle', True,
                maxw=bw - 2, tag='bdg:' + b)
    lc.text(x + 13, y + 21, title, 12, lc.C_TXT, 'start', True,
            maxw=bx - x - 20, tag=(tag or title))
    for i, (ln, mut) in enumerate(lines):
        lc.text(x + 13, y + 42 + i * 17, ln, 9, lc.C_MUTE if mut else '#334155',
                'start', maxw=w - 24, tag=(tag or title) + ':' + ln[:10])


# ---------------- 几何骨架（全部箭头端点由这些常量推出） ----------------
FY, FH = 96, 180                 # 前端进程框（头部在上、socket 行贴下）
SOCK_Y = FY + 66                 # 前端 socket 框顶（框高 96 → 底 = SOCK_Y+96）
ZY = FY + FH + 36                # 紫 带顶
ZH = 170                         # 紫 带高
EY = ZY + ZH + 36                # 引擎进程框顶
EH = 180                         # 引擎进程框高（socket 行贴上、头部沉底——与前端镜像，
                                 #   六条连线得以框边到框边、全程不穿任何头部文字）
ESOCK_Y = EY + 12                # 引擎 socket 框顶（底 = ESOCK_Y+96）
FE_SOCK_B = SOCK_Y + 96          # 前端 socket 框底 = 六条连线统一顶端
UP_X, READY_X, DN_X = 645, 960, 1035
HS_X0, HS_PITCH = 300, 22        # 握手走廊三线 x（右移避开带标题 112..283）

# ---------------- 标题区 ----------------
lc.text(MX, 36, 'ZMQ 全拓扑：进出的四扇门不对称——下行带信封定向，上行匿名单向扇入', 16.5,
        lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 60,
        '下行 ROUTER→DEALER 每条消息首帧 = 目标引擎 2 字节 identity 信封；上行 PUSH→PULL 百川归海；'
        '全部数据 socket HWM=0；每条 DEALER 先发 ready 认亲帧，ROUTER 才认识它',
        10.5, lc.C_MUTE, 'start', maxw=1088, tag='subtitle')
chip(1184, 12, '放大自 L2 站 1-3 装配 · L0：紫色 ZMQ 边界带', lc.C_ZMQ_S)

# ---------------- 上：前端进程 ----------------
lc.rect(MX, FY, BXR - MX, FH, lc.C_API_F, lc.C_API_S, rx=12, sw=2.2)
lc.text(MX + 16, FY + 24, '前端进程 · client 半边（API 进程，零 GPU）', 13, lc.C_API_S,
        'start', True, tag='fe:title')
lc.text(BXR - 16, FY + 24, 'vllm/v1/engine/core_client.py · L527-L607', 9, lc.C_FAINT,
        'end', tag='fe:file')
lc.text(MX + 16, FY + 46, 'zmq.Context(io_threads=2) · asyncio 版包 zmq.asyncio.Context（L527-L528）',
        9.5, lc.C_MUTE, 'start', tag='fe:ctx')

socket_box(MX + 16, SOCK_Y, 250, '握手 ROUTER (bind)',
           [('一次性 · 拓扑发现专用', False),
            ('launch_core_engines 持有', False),
            ('vllm/v1/engine/utils.py:L1171-L1173', True)],
           lc.C_MUTE, dash=True, tag='hs-router')
socket_box(494, SOCK_Y, 306, 'PULL (bind)',
           [('output_socket · 收全部引擎输出（扇入）', False),
            ('bind 缺省即 bind（bind 规则 L308）', False),
            ('core_client.py:L596-L598', True)],
           lc.C_API_S, badges=('HWM=0',), tag='pull')
socket_box(878, SOCK_Y, 306, 'ROUTER (bind)',
           [('input_socket · 按信封寻址目标 DEALER', False),
            ('tcp:0 占位 → bind 后 LAST_ENDPOINT 回填', False),
            ('core_client.py:L589-L595 · L600-L607', True)],
           lc.C_API_S, badges=('HWM=0',), tag='router')

# ---------------- 中：紫色 ZMQ 带 ----------------
lc.rect(MX, ZY, BXR - MX, ZH, lc.C_ZMQ_F, lc.C_ZMQ_S, rx=12, sw=2.0)
lc.text(MX + 16, ZY + 22, '进程边界 · ZMQ + msgpack', 12, lc.C_ZMQ_S, 'start', True,
        tag='band:title')

# —— 左：一次性握手旁路 ①HELLO↑ ②地址集↓ ③READY↑（三线在「边界连线」节统一画，
#     框边到框边：握手 ROUTER 框底 ↔ 握手 DEALER 框顶；标注放三线右侧整体让位）——
hs_labels = [
    ('① HELLO「我在」↑', ZY + 60),
    ('② 地址集 EngineZmqAddresses「收发室门牌」↓', ZY + 112),
    ('③ READY「配置核对无误」↑', ZY + 154),
]
for lab, ly in hs_labels:
    lc.text(HS_X0 + 2 * HS_PITCH + 9, ly, lab, 9, lc.C_ZMQ_S, 'start', maxw=340,
            tag='hs:' + lab[:8])

# —— 中：上行 / 右：下行 + ready 认亲帧（线在「边界连线」节统一画，此处只放标注）——
lc.text(UP_X - 9, ZY + 134, '上行 · EngineCoreOutputs 整批回程', 9.5, lc.C_ENG_S,
        'end', True, maxw=200, tag='up:l1')
lc.text(UP_X - 9, ZY + 152, 'PUSH→PULL 匿名单向扇入', 9, lc.C_MUTE, 'end',
        maxw=200, tag='up:l2')
lc.text(DN_X + 9, ZY + 134, '下行 · ADD / ABORT / UTILITY', 9.5, lc.C_API_S, 'start',
        True, maxw=180, tag='dn:l1')
lc.text(DN_X + 9, ZY + 152, '首帧 = identity 信封 2B', 9, lc.C_MUTE, 'start',
        maxw=180, tag='dn:l2')
lc.text(READY_X - 8, ZY + 126, 'ready 认亲帧先行（虚线上行）', 9.5, lc.C_ZMQ_S, 'end',
        True, maxw=280, tag='rd:l1')
lc.text(READY_X - 8, ZY + 144, 'DEALER 首条消息 = EngineCoreReadyResponse', 9, lc.C_MUTE,
        'end', maxw=280, tag='rd:l2')
lc.text(READY_X - 8, ZY + 162, 'ROUTER 实收 2 帧：信封 2B + 载荷（host 实测 31B）', 9,
        lc.C_MUTE, 'end', maxw=280, tag='rd:l3')

# —— 中右：HWM 注块（带内）——
HWX = 812
lc.text(HWX, ZY + 48, '全对 HWM=0（RCVHWM / SNDHWM=0）', 10, lc.C_TXT, 'middle', True,
        maxw=240, tag='hwm:l1')
lc.text(HWX, ZY + 66, '引擎永不被慢前端阻塞 GPU', 9, '#334155', 'middle', maxw=240,
        tag='hwm:l2')
lc.text(HWX, ZY + 82, '大内存机器 0.5GB 内核缓冲', 9, '#334155', 'middle', maxw=240,
        tag='hwm:l3')
lc.text(HWX, ZY + 98, 'network_utils.py:L310-L316', 8.5, lc.C_FAINT, 'middle', maxw=240,
        tag='hwm:l4')

# ---------------- 下：EngineCore 进程 ----------------
lc.rect(MX, EY, BXR - MX, EH, lc.C_ENG_F, lc.C_ENG_S, rx=12, sw=2.2)
# 头部沉底（socket 行贴上）——六条连线从引擎框顶直达 socket 框顶边，全程不穿头部文字
lc.text(MX + 16, EY + EH - 46, 'EngineCore 进程 · 引擎半边（IO 线程 ×2 + busy loop）', 13,
        lc.C_ENG_S, 'start', True, tag='eng:title')
lc.text(BXR - 16, EY + EH - 46, 'vllm/v1/engine/core.py · L1034 · L1661-L1766', 9,
        lc.C_FAINT, 'end', tag='eng:file')
lc.text(MX + 16, EY + EH - 24, 'busy loop 永不直接碰 socket——输入/输出 IO 线程各一条', 9.5,
        lc.C_MUTE, 'start', tag='eng:ctx')

socket_box(MX + 16, ESOCK_Y, 250, '握手 DEALER (connect)',
           [('同一 identity · _perform_handshake', False),
            ('startup_handshake 收地址集', False),
            ('vllm/v1/engine/core.py:L1213-L1233', True)],
           lc.C_MUTE, dash=True, tag='hs-dealer')
socket_box(494, ESOCK_Y, 306, 'PUSH (connect)',
           [('每前端 1 条（many-to-many #17546）', False),
            ('linger=4000：死讯先于关 socket 送出', False),
            ('core.py:L1758-L1763 · L1761-L1766', True)],
           lc.C_ENG_S, badges=('HWM=0',), tag='push')
socket_box(878, ESOCK_Y, 306, 'DEALER (connect)',
           [('每前端 1 条 · 同一 identity=engine_index', False),
            ("identity = 2 字节小端 b'\\x00\\x00'（L1034）", False),
            ('core.py:L1661-L1667', True)],
           lc.C_ENG_S, badges=('HWM=0',), tag='dealer')

# ---------------- 边界连线（socket 框全部画完后才画：全程可见，端到端贴框边） ----------------
# 上行 PUSH→PULL：PUSH 框顶 → PULL 框底，箭头指入 PULL
lc.seg(UP_X, ESOCK_Y, UP_X, FE_SOCK_B, lc.C_ENG_S, 3.2, 'up')
# 下行 ROUTER→DEALER：ROUTER 框底 → DEALER 框顶，箭头指入 DEALER
lc.seg(DN_X, FE_SOCK_B, DN_X, ESOCK_Y, lc.C_API_S, 3.2, 'dn')
# ready 认亲帧（虚线上行）：DEALER 框顶 → ROUTER 框底，箭头指入 ROUTER
lc.seg(READY_X, ESOCK_Y, READY_X, FE_SOCK_B, lc.C_ZMQ_S, 1.8, 'zq', dash=True)
# 一次性握手走廊三线：握手 ROUTER 框底 ↔ 握手 DEALER 框顶（①↑ ②↓ ③↑，框边到框边）
for dr, k in (('up', 0), ('down', 1), ('up', 2)):
    x = HS_X0 + k * HS_PITCH
    if dr == 'up':                                   # 引擎 → 前端：箭头在顶端
        lc.seg(x, ESOCK_Y, x, FE_SOCK_B, lc.C_ZMQ_S, 1.6, 'zq', dash=True)
    else:                                            # 前端 → 引擎：箭头在底端
        lc.seg(x, FE_SOCK_B, x, ESOCK_Y, lc.C_ZMQ_S, 1.6, 'zq', dash=True)

# ---------------- why 注（虚线框） ----------------
WY = EY + EH + 24
WH = 118
lc.rect(MX, WY, BXR - MX, WH, 'none', lc.C_FAINT, rx=8, sw=1.1, dash=True)
lc.text(MX + 16, WY + 20, 'why · 进出为什么不对称', 10, lc.C_TXT, 'start', True,
        tag='why:t')
lc.text(MX + 16, WY + 40,
        '输入是「多选一」的定向问题（M 个引擎发给哪一个）——PUSH/PULL 匿名无寻址能力，必须 ROUTER 信封；'
        '输出纯扇入（百川归海），匿名单向管道最便宜',
        9, '#334155', 'start', maxw=BXR - MX - 32, tag='why:l1')
lc.text(MX + 16, WY + 58,
        "代价：DP=1 部署也在付信封帧开销；DEALER 必须先发言认亲——'required before the front-end ROUTER "
        "socket can send input messages back to us'（core.py:L1688-L1693）",
        9, '#334155', 'start', maxw=BXR - MX - 32, tag='why:l2')
lc.text(MX + 16, WY + 76,
        '演进：#15906 首改 ROUTER 次日即 revert；#17546 随 many-to-many 需求成熟才立住——'
        '这条带信封的路什么时候才真的需要寻址？ch34 回收',
        9, '#334155', 'start', maxw=BXR - MX - 32, tag='why:l3')
lc.text(MX + 16, WY + 98,
        'socket 记账：client 每前端 1 条 ROUTER(bind) + 1 条 PULL(bind)；engine 每前端 1 条 DEALER(connect) '
        '+ 1 条 PUSH(connect)；另有 1 对一次性握手 ROUTER/DEALER（startup 期即弃）',
        9, lc.C_MUTE, 'start', maxw=BXR - MX - 32, tag='why:l4')

# ---------------- 图例 + 页脚 ----------------
LY = WY + WH + 26
items = [
    ('arrow', lc.C_API_S, 'dn', 3.2, '下行 ROUTER→DEALER'),
    ('arrow', lc.C_ENG_S, 'up', 3.2, '上行 PUSH→PULL'),
    ('dash', lc.C_ZMQ_S, 'zq', 1.8, 'ready / 握手（一次性控制流）'),
]
lx = MX
for kind, color, mk, sw, name in items:
    lc.seg(lx + 2, LY - 3, lx + 32, LY - 3, color, sw, mk, dash=(kind == 'dash'))
    lc.text(lx + 40, LY + 1, name, 9.5, lc.C_TXT, 'start', maxw=220,
            tag='leg:' + name[:8])
    lx += 40 + lc.tw(name, 9.5) + 22
bw = lc.tw('HWM=0', 8.5, True) + 12
lc.rect(lx + 4, LY - 10, bw, 17, lc.C_ZMQ_F, lc.C_ZMQ_S, rx=8, sw=1.0)
lc.text(lx + 4 + bw / 2, LY + 1, 'HWM=0', 8.5, lc.C_ZMQ_S, 'middle', True, maxw=bw - 2,
        tag='leg:hwm')
lc.text(lx + 12 + bw, LY + 1, '= RCVHWM/SNDHWM=0 · 框内灰字 = 规范源码路径', 9.5,
        lc.C_TXT, 'start', maxw=420, tag='leg:hwm2')
lc.text(MX, LY + 24,
        '行号基线 vLLM v0.27.1 · 实测值标 host（win32 回环 tcp——HWM/linger/io_threads 是 socket 选项、与传输无关，'
        'pin 的 Linux ipc:// 同值）· bind 规则：PUSH/SUB/XSUB 默认 connect、其余默认 bind（network_utils.py:L308）',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='footer')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch05-fig-zmq-topology.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
