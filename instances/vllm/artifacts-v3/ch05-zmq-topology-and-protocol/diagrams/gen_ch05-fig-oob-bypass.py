#!/usr/bin/env python3
"""ch05 机制图 · OOB 旁路专线（explainer m6 figure_spec ch05-fig-oob-bypass）

放大自 L0 紫色 ZMQ 边界带的『旁路』——即本章 L2 south 组件「OOB 旁路 · torch_shm」
的机制展开：双进程带之间，紫色 ZMQ 带之外另画一条绕行管道（mp.Queue 共享内存专线），
张量走专线、句柄走紫带。

claim：多模态大/CUDA 张量不过 socket——share_memory_ 后走 torch.mp.Queue 共享内存
专线，msgpack 主帧只放 (sender_id, message_id, tensor_id) 提货句柄（16KB 张量：
OOB 1 帧 vs 走 ZMQ 2 帧）；接收端 drain-and-buffer 乱序重组 + 水位线过期清理。

数字全部取自 explainer figure_spec.numbers（oob trace 实测 + pin 锚点）；
坐标由常量/循环计算；文本全 esc()；配色走 l0_common 语系（同源强制）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1340, 1010
MX, BXR = 100, 1290

# 绿色箭头 marker（张量本体专线）
DEFS = lc.DEFS.replace('</defs>',
                       '<marker id="gn" viewBox="0 0 10 6" refX="9" refY="3" '
                       'markerWidth="7" markerHeight="5" orient="auto">'
                       f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_GPU_S}"/></marker>'
                       '</defs>')


def chip(x_right, y, label, color):
    w = lc.tw(label, 9.5, True) + 14
    x = x_right - w
    lc.rect(x, y, w, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
    lc.text(x + w / 2, y + 14.5, label, 9.5, color, 'middle', True,
            maxw=w - 4, tag='chip:' + label[:12])
    return x


def box(x, y, w, h, title, lines, stroke, fill='#ffffff', dash=False, tfs=11,
        lfs=8.5, tag='', file=None):
    lc.rect(x, y, w, h, fill, stroke, rx=7, sw=1.6, dash=dash)
    lc.text(x + 12, y + 20, title, tfs, stroke if fill != '#ffffff' else lc.C_TXT,
            'start', True, maxw=w - 22, tag=(tag or title))
    for i, ln in enumerate(lines):
        lc.text(x + 12, y + 40 + i * 15.5, ln, lfs, '#334155', 'start', maxw=w - 22,
                tag=(tag or title) + ':' + ln[:10])
    if file:
        lc.text(x + 12, y + h - 10, file, 8.5, lc.C_FAINT, 'start', maxw=w - 22,
                tag=(tag or title) + ':file')


# ---------------- 标题区 ----------------
lc.text(MX, 36, 'OOB 旁路专线：大张量不过 socket——句柄过线、本体走共享内存', 16.5,
        lc.C_TXT, 'start', True, maxw=900, tag='title')
lc.text(MX, 60,
        'share_memory_ 后 torch.multiprocessing.Queue 直达引擎进程；紫带上那条消息瘦成 1 帧'
        '（16KB 张量：OOB 1 帧 vs 走 ZMQ 2 帧）——为视频类『单次大块、cache 不命中』负载特化（#32104）',
        10.5, lc.C_MUTE, 'start', maxw=1140, tag='subtitle')
chip(BXR, 12, '放大自 L2 south「OOB 旁路 · torch_shm」· L0：紫色 ZMQ 边界带', lc.C_ZMQ_S)

# ---------------- 几何骨架 ----------------
FE_Y, FE_H = 100, 150            # 前端进程框
EN_Y, EN_H = 690, 150            # 引擎进程框
BAND_X, BAND_W = 340, 500        # 紫 带（x 340..840）
BAND_Y, BAND_H = 340, 300
PIPE_X = 230                     # 绿色专线（带左绕行）
MSG_X = 620                      # 紫带内消息箭头

# ---------------- 双进程框 ----------------
box(MX, FE_Y, 740, FE_H, '前端进程 · TensorIpcSender',
    ['mm processor 产出大 / CUDA 张量',
     'tensor.share_memory_()——存储页标记跨进程共享',
     'new_message()：message_id 翻号、tensor_id 归零递增',
     '句柄 {sender_id（8 位 hex）, message_id, tensor_id} 进主帧'],
    lc.C_API_S, lc.C_API_F, tag='sender', file='vllm/v1/engine/tensor_ipc.py:L69-L105')
box(MX, EN_Y, 740, EN_H, 'EngineCore 进程 · TensorIpcReceiver',
    ['MsgpackDecoder(oob_tensor_provider=receiver)',
     '按句柄取货：drain-and-buffer——排空队列、沿途缓冲',
     '乱序到货也各归各位（同消息内乱序 + 跨消息按序）',
     '水位线推进：迟到的旧张量直接作废（防泄漏）'],
    lc.C_ENG_S, lc.C_ENG_F, tag='receiver', file='vllm/v1/engine/tensor_ipc.py:L110-L178')

# ---------------- 紫色 ZMQ 带（消息通道） ----------------
lc.rect(BAND_X, BAND_Y, BAND_W, BAND_H, lc.C_ZMQ_F, lc.C_ZMQ_S, rx=10, sw=2.0)
lc.text(BAND_X + 16, BAND_Y + 22, '进程边界 · ZMQ + msgpack', 11, lc.C_ZMQ_S, 'start',
        True, tag='band:t')
lc.seg(MSG_X, FE_Y + FE_H, MSG_X, EN_Y, lc.C_API_S, 2.2, 'dn')
lc.text(MSG_X - 12, BAND_Y + 96, '下行消息照走 ROUTER→DEALER', 9, lc.C_API_S, 'end',
        True, maxw=200, tag='msg:l1')
lc.text(MSG_X - 12, BAND_Y + 114, '提货句柄 dict 随主帧过线', 8.5, lc.C_MUTE, 'end',
        maxw=200, tag='msg:l2')

# 帧数对比（带内右侧）
lc.text(739, BAND_Y + 24, '同一张 16KB 张量', 8.5, lc.C_MUTE, 'middle', maxw=150,
        tag='cmp:t')
lc.rect(650, BAND_Y + 34, 178, 58, '#ffffff', lc.C_GPU_S, rx=6, sw=1.4)
lc.text(739, BAND_Y + 54, 'OOB 在场：1 帧', 9.5, lc.C_GPU_S, 'middle', True, maxw=162,
        tag='cmp:a1')
lc.text(739, BAND_Y + 72, '主帧 141B——只含句柄', 8.5, '#334155', 'middle', maxw=162,
        tag='cmp:a2')
lc.rect(650, BAND_Y + 104, 178, 58, '#ffffff', lc.C_MUTE, rx=6, sw=1.3, dash=True)
lc.text(739, BAND_Y + 124, 'OOB 缺席：2 帧', 9.5, lc.C_MUTE, 'middle', True, maxw=162,
        tag='cmp:b1')
lc.text(739, BAND_Y + 142, '主帧 + aux 16384B 过 socket', 8.5, '#334155', 'middle',
        maxw=162, tag='cmp:b2')

# ---------------- 绿色专线（绕行管道） ----------------
lc.seg(PIPE_X, FE_Y + FE_H, PIPE_X, EN_Y, lc.C_GPU_S, 7.0, 'gn')
lc.text(PIPE_X - 10, BAND_Y + 84, 'torch.multiprocessing.Queue', 9.5, lc.C_GPU_S,
        'end', True, maxw=180, tag='pipe:l1')
lc.text(PIPE_X - 10, BAND_Y + 102, '共享内存专线（零拷贝直达）', 8.5, lc.C_GPU_S,
        'end', maxw=180, tag='pipe:l2')
lc.text(PIPE_X - 10, BAND_Y + 120, 'launch_core_engines 创建', 8, lc.C_MUTE, 'end',
        maxw=180, tag='pipe:l3')
lc.text(PIPE_X - 10, BAND_Y + 138, 'utils.py:L1078-L1085', 8, lc.C_FAINT, 'end',
        maxw=180, tag='pipe:l4')
lc.rect(PIPE_X + 14, BAND_Y + 96, 88, 44, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.3)
lc.text(PIPE_X + 58, BAND_Y + 113, '张量本体', 8.5, lc.C_GPU_S, 'middle', True,
        maxw=76, tag='pipe:c1')
lc.text(PIPE_X + 58, BAND_Y + 130, '16384B', 8.5, lc.C_GPU_S, 'middle', maxw=76,
        tag='pipe:c2')

# ---------------- 右侧：接收端 drain-and-buffer ----------------
PX, PY, PW, PH = 880, 340, 410, 480
lc.rect(PX, PY, PW, PH, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(PX + 16, PY + 24, '接收端 · drain-and-buffer 乱序重组', 11, lc.C_TXT, 'start',
        True, maxw=340, tag='rp:t')

# 队列 + 排空说明
lc.rect(PX + 4, PY + 46, 130, 58, '#ffffff', lc.C_MUTE, rx=7, sw=1.3)
lc.text(PX + 69, PY + 66, 'mp.Queue', 9.5, lc.C_TXT, 'middle', True, maxw=110,
        tag='rp:q')
bw2, gap2 = 10, 6
bx0 = PX + 4 + (130 - (3 * bw2 + 2 * gap2)) / 2
for i in range(3):
    lc.rect(bx0 + i * (bw2 + gap2), PY + 76, bw2, 18, '#cbd5e1', lc.C_MUTE, rx=1.5,
            sw=1.0)
lc.text(PX + 150, PY + 66, '排空队列找目标张量', 8.5, '#334155', 'start', maxw=240,
        tag='rp:d1')
lc.text(PX + 150, PY + 84, '沿途张量按句柄缓冲', 8.5, '#334155', 'start', maxw=240,
        tag='rp:d2')

# 缓冲架（格位 = (message_id, tensor_id)）
CELL_W, CELL_H = 72, 40
GRID_X0, GRID_Y0 = PX + 96, PY + 146
lc.text(GRID_X0, PY + 138, 'tensor 0', 8, lc.C_MUTE, 'middle', tag='rp:c0')
lc.text(GRID_X0 + CELL_W, PY + 138, 'tensor 1', 8, lc.C_MUTE, 'middle', tag='rp:c1')
cells = [  # (row, col, numel, 取货序)
    (0, 0, 'numel 4', '②'),
    (0, 1, 'numel 5', '①'),
    (1, 0, 'numel 6', '③'),
    (1, 1, None, None),
]
for r in range(2):
    lc.text(GRID_X0 - 10, GRID_Y0 + r * CELL_H + 26, f'message {r + 1}', 8, lc.C_MUTE,
            'end', maxw=86, tag=f'rp:r{r}')
    for c in range(2):
        cx0, cy0 = GRID_X0 + c * CELL_W, GRID_Y0 + r * CELL_H
        val = next((v for rr, cc, v, _ in cells if rr == r and cc == c), None)
        if val:
            lc.rect(cx0, cy0, CELL_W, CELL_H, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.2)
            lc.text(cx0 + CELL_W / 2, cy0 + 26, val, 9, lc.C_TXT, 'middle', True,
                    maxw=CELL_W - 6, tag='rp:v' + val[-1])
            badge = next((b for rr, cc, _, b in cells if rr == r and cc == c), None)
            lc.text(cx0 + CELL_W - 12, cy0 + 15, badge, 9, lc.C_ENG_S, 'middle', True,
                    tag='rp:b' + badge)
        else:
            lc.rect(cx0, cy0, CELL_W, CELL_H, 'none', lc.C_FAINT, rx=4, sw=1.0, dash=True)
            lc.text(cx0 + CELL_W / 2, cy0 + 26, '—', 9, lc.C_FAINT, 'middle', tag='rp:empty')
lc.parrow([(PX + 69, PY + 104), (PX + 69, PY + 122), (GRID_X0 + CELL_W / 2, PY + 122),
           (GRID_X0 + CELL_W / 2, PY + 144)], lc.C_MUTE, 1.6, 'std')

# 按句柄取货
lc.parrow([(GRID_X0 + 36, PY + 226), (GRID_X0 + 36, PY + 248)], lc.C_MUTE, 1.6, 'std')
lc.rect(PX + 4, PY + 250, 250, 50, '#ffffff', lc.C_GPU_S, rx=6, sw=1.3)
lc.text(PX + 119, PY + 268, '按句柄取货：请求 (1,1) → (1,0) → (2,0)', 8.5, lc.C_TXT,
        'middle', True, maxw=234, tag='rp:pk1')
lc.text(PX + 119, PY + 288, '返回 numel [5, 4, 6]——全对号', 8.5, '#334155', 'middle',
        maxw=234, tag='rp:pk2')

# 迟到旧张量 → 丢弃
lc.parrow([(PX + 134, PY + 104), (PX + 380, PY + 104), (PX + 380, PY + 360),
           (PX + 256, PY + 360)], lc.C_ABORT, 1.5, 'ab', dash=True)
lc.text(PX + 372, PY + 122, '水位线已过', 8, lc.C_ABORT, 'end', maxw=90, tag='rp:st0')
lc.rect(PX + 4, PY + 336, 250, 62, 'none', lc.C_ABORT, rx=6, sw=1.3, dash=True)
lc.text(PX + 16, PY + 354, '迟到旧张量 → 丢弃', 9, lc.C_ABORT, 'start', True, maxw=226,
        tag='rp:st1')
lc.text(PX + 16, PY + 371, "'Ignoring stale tensor'", 8.5, '#334155', 'start',
        maxw=226, tag='rp:st2')
lc.text(PX + 16, PY + 388, '（1 条警告 · 新消息照常）', 8, lc.C_MUTE, 'start', maxw=226,
        tag='rp:st3')

lc.text(PX + 16, PY + 420, '水位线（current_message_id）推进后，旧 message 的迟到张量直接作废——防泄漏',
        8.5, lc.C_MUTE, 'start', maxw=PW - 32, tag='rp:ft')
lc.text(PX + 16, PY + 436, '句柄编号线：(1,0)(1,1) 同一消息、(2,0) 下一消息', 8.5,
        lc.C_MUTE, 'start', maxw=PW - 32, tag='rp:ft2')
lc.text(PX + 16, PY + 456, '跨进程 e2e：4096 元素张量经真实 client → 引擎往返实测', 8.5,
        lc.C_MUTE, 'start', maxw=PW - 32, tag='rp:ft3')

# ---------------- 底部两注 ----------------
NY = 868
lc.rect(MX, NY, 740, 74, 'none', lc.C_FAINT, rx=8, sw=1.1, dash=True)
lc.text(MX + 14, NY + 18, '启用面与限制（窄而明确）', 9.5, lc.C_TXT, 'start', True,
        tag='n1:t')
lc.text(MX + 14, NY + 38, "仅 multimodal_config.mm_tensor_ipc=='torch_shm' 启用 · 单队列只打 rank 0 · "
        "DP>1 不支持 · mm processor cache 开启即失效", 8.5, '#334155', 'start',
        maxw=712, tag='n1:l1')
lc.text(MX + 14, NY + 56, '（vllm/v1/engine/tensor_ipc.py:L46-L49——为视频类『单次大块、cache 不命中』负载特化）',
        8.5, lc.C_MUTE, 'start', maxw=712, tag='n1:l2')
lc.rect(PX, NY, PW, 74, 'none', lc.C_FAINT, rx=8, sw=1.1, dash=True)
lc.text(PX + 14, NY + 18, '动机（#32104）：为什么要专线', 9.5, lc.C_TXT, 'start', True,
        tag='n2:t')
lc.text(PX + 14, NY + 38, '视频 HW 解码张量在 API server 的 VRAM——走 ZMQ 要', 8.5,
        '#334155', 'start', maxw=382, tag='n2:l1')
lc.text(PX + 14, NY + 56, 'GPU→CPU→socket→CPU→GPU 两跳拷贝；通用路径仍是多帧零拷贝', 8.5,
        '#334155', 'start', maxw=382, tag='n2:l2')

# ---------------- 图例 + 页脚 ----------------
LY = 966
items = [
    ('thick', lc.C_GPU_S, 'gn', '张量本体专线（torch_shm）'),
    ('arrow', lc.C_API_S, 'dn', '下行消息（ROUTER→DEALER）'),
    ('swatch', lc.C_ZMQ_S, lc.C_ZMQ_F, 'ZMQ 消息通道'),
    ('dash', lc.C_ABORT, 'ab', '迟到丢弃'),
]
lx = MX
for kind, color, mk, name in items:
    if kind == 'thick':
        lc.seg(lx + 2, LY - 3, lx + 32, LY - 3, color, 5.0, mk)
    elif kind == 'arrow':
        lc.seg(lx + 2, LY - 3, lx + 32, LY - 3, color, 2.2, mk)
    elif kind == 'dash':
        lc.seg(lx + 2, LY - 3, lx + 32, LY - 3, color, 1.6, mk, dash=True)
    else:
        lc.rect(lx, LY - 9, 16, 11, lc.C_ZMQ_F, color, rx=3, sw=1.6)
    lc.text(lx + 40, LY + 1, name, 9.5, lc.C_TXT, 'start', maxw=220,
            tag='leg:' + name[:8])
    lx += 40 + lc.tw(name, 9.5) + 22
lc.text(MX, LY + 24, '行号基线 vLLM v0.27.1 · 乱序 / 过期 / 帧数均为 host 实测（进程内 mp spawn 队列同真实路径）· '
        '框内灰字 = 规范源码路径', 9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='footer')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch05-fig-oob-bypass.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
