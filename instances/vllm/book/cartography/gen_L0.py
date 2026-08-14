#!/usr/bin/env python3
"""L0 全书唯一权威架构图 —— v3 图系的根（excalidraw 范式重画）。

一张图 = 一个请求的一生：3 user → API 双泳道 → AsyncMpClient 中间带 → EngineCore 三框。
范式骨架（L0-paradigm.md）：方法名节点链 + 泳道对偶 + 中间带显式化 + 引擎简略 + zoom 索引。
"""
import sys
from pathlib import Path
import xml.sax.saxutils as xs

FONT = 'Microsoft YaHei, SimHei, Noto Sans CJK SC, PingFang SC, sans-serif'
C_TXT, C_MUTE = '#0f172a', '#64748b'
C_API_S, C_API_F = '#2563eb', '#dbeafe'      # API 进程 = 蓝
C_ENG_S, C_ENG_F = '#ea580c', '#fed7aa'      # EngineCore 进程 = 橙
C_ZMQ_S, C_ZMQ_F = '#7c3aed', '#e9d5ff'      # ZMQ 边界 = 紫
C_ZOOM = '#94a3b8'                            # zoom 索引虚线框 = 灰

W = 1400
L = []

def esc(s): return xs.escape(str(s))

def tw(s, fs, bold=False):
    """估算文本宽度（CJK 按 fs，ASCII 按 0.58×fs）"""
    n = sum(1 for c in str(s) if ord(c) > 0x2E80)
    return (n * fs + (len(str(s)) - n) * fs * 0.58) * (1.07 if bold else 1.0)

def text(x, y, s, fs=11, fill=C_TXT, anchor='middle', bold=False):
    b = ' font-weight="bold"' if bold else ''
    L.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{fs}" '
             f'fill="{fill}" text-anchor="{anchor}"{b}>{esc(s)}</text>')

def box(x, y, w, h, fill, stroke, r=6, sw=1.6, dash=False):
    d = ' stroke-dasharray="6,4"' if dash else ''
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
             f'rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

def arrow(x1, y1, x2, y2, color=C_MUTE, sw=1.8, marker='std'):
    L.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
             f'stroke="{color}" stroke-width="{sw}" marker-end="url(#{marker})"/>')

def node(x, y, w, h, title, sub='', fill='#ffffff', stroke='#94a3b8'):
    """方法名节点（标题+副标题）"""
    box(x, y, w, h, fill, stroke, r=5, sw=1.3)
    text(x + w / 2, y + (h / 2 if not sub else h / 2 - 3), title, 10, C_TXT, bold=True)
    if sub:
        text(x + w / 2, y + h / 2 + 10, sub, 8, C_MUTE)

def queue_icon(x, y):
    """队列图标 ▐█▌▐█▌▐█▌"""
    for i in range(3):
        bx = x + i * 10
        L.append(f'<rect x="{bx}" y="{y}" width="6" height="14" fill="{C_MUTE}" rx="1"/>')

# ========== 画布与标记 ==========
L.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 1050">')
L.append('<defs>'
         '<marker id="std" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="5" markerHeight="4" orient="auto">'
         f'<path d="M0,0 L10,3 L0,6 Z" fill="{C_MUTE}"/></marker>'
         '<marker id="dn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto">'
         f'<path d="M0,0 L10,3 L0,6 Z" fill="{C_API_S}"/></marker>'
         '<marker id="up" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto">'
         f'<path d="M0,0 L10,3 L0,6 Z" fill="{C_ENG_S}"/></marker>'
         '</defs>')
L.append(f'<rect width="{W}" height="1050" fill="#fafafa"/>')

# 标题
text(24, 32, 'vLLM v1 架构图（L0）：一个请求的一生', 16, C_TXT, anchor='start', bold=True)
text(24, 52, '方法名级节点链 · 双泳道对偶 · 中间带显式化 · zoom 索引为后续 L1/L2 留锚', 10, C_MUTE, anchor='start')

MX = 24
y = 72

# ========== 三个 user 并排 ==========
user_w, user_gap = 120, 30
user_y = y
for i in range(3):
    ux = MX + 350 + i * (user_w + user_gap)
    box(ux, user_y, user_w, 36, '#f1f5f9', C_MUTE, r=4, sw=1.2)
    text(ux + user_w / 2, user_y + 22, f'user{i+1}', 10, C_TXT, bold=True)
    # SSE 双向标注
    text(ux + user_w / 2, user_y + 52, 'sse', 8.5, C_MUTE)
    arrow(ux + user_w / 2, user_y + 36, ux + user_w / 2, user_y + 60, C_MUTE, 1.2)

y = user_y + 70

# ========== API 进程（双泳道对偶）==========
api_h = 280
box(MX, y, W - 48, api_h, C_API_F, C_API_S, r=8, sw=2.2)
text(MX + 16, y + 20, 'API 进程（frontend · 零 GPU）', 13, C_API_S, anchor='start', bold=True)
text(MX + W - 72, y + 20, 'HTTP/JSON/SSE 的 CPU 活全在这里', 9, C_MUTE, anchor='end')

# 中线分左右泳道
mid_x = MX + (W - 48) / 2
L.append(f'<line x1="{mid_x}" y1="{y + 34}" x2="{mid_x}" y2="{y + api_h - 10}" '
         f'stroke="{C_MUTE}" stroke-width="1.2" stroke-dasharray="4,3"/>')

# 右泳道（下行输入链）
rx = MX + 60
rw = 220
ry = y + 44
node(rx, ry, rw, 38, 'render', 'tokenize 在这里！文本不过 IPC')
arrow(rx + rw / 2, ry + 38, rx + rw / 2, ry + 50, C_API_S, 1.6, 'dn')
ry += 50
node(rx, ry, rw, 38, 'InputProcessor', 'request_id 双轨 + EngineCoreRequest')
arrow(rx + rw / 2, ry + 38, rx + rw / 2, ry + 50, C_API_S, 1.6, 'dn')
ry += 50
node(rx, ry, rw, 38, 'assign_request_id', '外部 id + 8 位 hex 内部 id')
arrow(rx + rw / 2, ry + 38, rx + rw / 2, ry + 50, C_API_S, 1.6, 'dn')
ry += 50
node(rx, ry, rw, 34, 'add_request', '双登记：本进程建表 + 跨进程发')
api_dn_cx = rx + rw / 2  # 下行中心线 x

# 左泳道（上行输出链）
lx = mid_x + 40
lw = 220
ly = y + 44
# 三个收集器并排
coll_w = (lw - 20) / 3
for i in range(3):
    cx = lx + i * (coll_w + 10)
    box(cx, ly, coll_w, 32, '#f8fafc', C_MUTE, r=4, sw=1.1)
    text(cx + coll_w / 2, ly + 20, f'Coll{i+1}', 8.5, C_TXT, bold=True)
text(lx + lw / 2, ly + 42, '单槽 + Event（刻意不用 Queue）', 7.5, C_MUTE)
arrow(lx + lw / 2, ly + 58, lx + lw / 2, ly + 70, C_ENG_S, 1.6, 'up')
ly += 70
node(lx, ly, lw, 38, 'process_outputs', 'demux 到 RequestOutputCollector')
arrow(lx + lw / 2, ly + 38, lx + lw / 2, ly + 50, C_ENG_S, 1.6, 'up')
ly += 50
node(lx, ly, lw, 38, 'output_handler', 'chunk 切片 + sleep(0) 让事件循环喘气')
arrow(lx + lw / 2, ly + 38, lx + lw / 2, ly + 50, C_ENG_S, 1.6, 'up')
ly += 50
node(lx, ly, lw, 34, 'PULL socket', '上行接收端')
api_up_cx = lx + lw / 2  # 上行中心线 x

# user 连到泳道
for i in range(3):
    ux_center = MX + 350 + user_w / 2 + i * (user_w + user_gap)
    # 进：user → 右泳道顶部
    arrow(ux_center, y + 70 - 10, rx + 30 + i * 60, y + 44, C_API_S, 1.3, 'dn')
    # 出：左泳道顶部 → user
    arrow(lx + 30 + i * 60, y + 44, ux_center, y + 70 - 10, C_ENG_S, 1.3, 'up')

y += api_h

# ========== AsyncMpClient 中间带（三段式的第 2 段显式化）==========
zmq_h = 80
zmq_y = y + 8
box(MX, zmq_y, W - 48, zmq_h, C_ZMQ_F, C_ZMQ_S, r=6, sw=1.8)
text(MX + 16, zmq_y + 18, 'AsyncMpClient（中间带）', 12, C_ZMQ_S, anchor='start', bold=True)
# 进程边界竖标（左侧大字）
text(MX + 28, zmq_y + zmq_h / 2 + 18, '进程边界', 11, C_ZMQ_S, anchor='start', bold=True)
text(MX + 28, zmq_y + zmq_h / 2 + 32, 'zmq+msgpack', 9, C_MUTE, anchor='start')

# 双通道节点
ch_w = 180
ch_y = zmq_y + 22
# 右通道：add_request_async
node(MX + 200, ch_y, ch_w, 32, 'add_request_async', 'ROUTER→DEALER')
# 左通道：get_output_async
node(MX + 200 + ch_w + 100, ch_y, ch_w, 32, 'get_output_async', 'engine PUSH→client PULL')

# 两条贯穿长箭头
# 下行：API 下行中心 → AsyncMpClient 右通道 → 引擎 input
arrow(api_dn_cx, y, api_dn_cx, zmq_y, C_API_S, 2.4, 'dn')
arrow(MX + 200 + ch_w / 2, zmq_y + 54, MX + 200 + ch_w / 2, zmq_y + zmq_h + 4, C_API_S, 2.4, 'dn')
text(api_dn_cx + 8, y + 30, 'EngineCoreRequest', 9, C_API_S, anchor='start')
text(api_dn_cx + 8, y + 44, '（只有 token ids）', 8, C_MUTE, anchor='start')

# 上行：引擎 output → AsyncMpClient 左通道 → API 上行中心
arrow(api_up_cx, zmq_y, api_up_cx, y, C_ENG_S, 2.4, 'up')
arrow(MX + 200 + ch_w + 100 + ch_w / 2, zmq_y + zmq_h + 4, MX + 200 + ch_w + 100 + ch_w / 2, zmq_y + 54, C_ENG_S, 2.4, 'up')
text(api_up_cx - 8, y + 30, 'EngineCoreOutputs', 9, C_ENG_S, anchor='end')
text(api_up_cx - 8, y + 44, '（每步整批聚合）', 8, C_MUTE, anchor='end')

y = zmq_y + zmq_h + 12

# ========== EngineCore 进程（简略三框 + 循环五段 + zoom 索引）==========
eng_h = 420
box(MX, y, W - 48, eng_h, C_ENG_F, C_ENG_S, r=8, sw=2.2)
text(MX + 16, y + 20, 'EngineCore 进程（busy loop）', 13, C_ENG_S, anchor='start', bold=True)
text(MX + W - 72, y + 20, 'GPU 上下文只存在于此进程及 worker', 9, C_MUTE, anchor='end')

# 三框布局：input_queue（右）+ 循环（中）+ output_queue（左）
qw = 140  # 队列框宽
loop_y = y + 42

# 右框：input_queue + DEALER
qx_r = MX + 40
box(qx_r, loop_y, qw, 56, '#ffffff', C_MUTE, r=5, sw=1.3)
text(qx_r + qw / 2, loop_y + 16, 'input_queue', 10, C_TXT, bold=True)
text(qx_r + qw / 2, loop_y + 32, 'DEALER', 9, C_MUTE)
queue_icon(qx_r + qw / 2 - 13, loop_y + 40)

# 中框：循环五段（横排）
loop_x = qx_r + qw + 30
loop_w = 760
box(loop_x, loop_y, loop_w, 160, '#ffffff', C_ENG_S, r=6, sw=1.6)
text(loop_x + 12, loop_y + 18, 'EngineCore.step() 逐拍循环（五段）', 11, C_ENG_S, anchor='start', bold=True)

steps = [
    ('① schedule', 'RUNNING 先于 WAITING'),
    ('② execute_model', 'non_block→Future'),
    ('③ get_grammar_bitmask', 'CPU 藏进 GPU 窗口'),
    ('④ sample_tokens', 'logits+bitmask'),
    ('⑤ update_from_output', '状态推进+free 块')
]
step_w = (loop_w - 40 - 4 * 20) / 5
step_y = loop_y + 32
for i, (title, sub) in enumerate(steps):
    sx = loop_x + 20 + i * (step_w + 20)
    box(sx, step_y, step_w, 50, '#fef3c7', '#f59e0b', r=4, sw=1.1)
    text(sx + step_w / 2, step_y + 18, title, 9, C_TXT, bold=True)
    text(sx + step_w / 2, step_y + 36, sub, 7.5, C_MUTE)
    if i < 4:
        arrow(sx + step_w, step_y + 25, sx + step_w + 20, step_y + 25, C_ENG_S, 1.4)

# 左框：output_queue + PUSH
qx_l = loop_x + loop_w + 30
box(qx_l, loop_y, qw, 56, '#ffffff', C_MUTE, r=5, sw=1.3)
text(qx_l + qw / 2, loop_y + 16, 'output_queue', 10, C_TXT, bold=True)
text(qx_l + qw / 2, loop_y + 32, 'PUSH', 9, C_MUTE)
queue_icon(qx_l + qw / 2 - 13, loop_y + 40)

# 循环流：input_queue → 循环 → output_queue
arrow(qx_r + qw, loop_y + 28, loop_x, loop_y + 28, C_ENG_S, 1.8)
arrow(loop_x + loop_w, loop_y + 28, qx_l, loop_y + 28, C_ENG_S, 1.8)
# 回程：循环底部回到顶部
arrow(loop_x + loop_w / 2, step_y + 50, loop_x + loop_w / 2, step_y + 64, C_MUTE, 1.4)
arrow(loop_x + loop_w / 2, step_y + 74, loop_x + loop_w / 2, step_y + 64, C_MUTE, 1.4)
text(loop_x + loop_w / 2 + 8, step_y + 69, '下一拍', 8, C_MUTE, anchor='start')

# 三个 zoom 索引虚线小框（循环框下方）
zoom_y = step_y + 90
zoom_w = (loop_w - 60) / 3
zoom_h = 64
zooms = [
    ('调度↔KV 对账', 'scheduler ↔ KV 块池'),
    ('GPU 执行臂', 'Executor→Worker→Runner'),
    ('采样与出口', 'Sampler 9 步管线')
]
for i, (title, sub) in enumerate(zooms):
    zx = loop_x + 20 + i * (zoom_w + 20)
    box(zx, zoom_y, zoom_w, zoom_h, '#fafafa', C_ZOOM, r=5, sw=1.2, dash=True)
    text(zx + zoom_w / 2, zoom_y + 24, title, 10, C_TXT, bold=True)
    text(zx + zoom_w / 2, zoom_y + 44, sub, 8.5, C_MUTE)
    text(zx + zoom_w / 2, zoom_y + zoom_h - 8, f'→ L1/L2 放大', 7.5, C_ZOOM)

# 图例
legend_y = y + eng_h + 24
text(MX, legend_y, '读图：请求自左上 user 进入，沿蓝箭头下行穿 API 双泳道→中间带→引擎；', 9.5, C_MUTE, anchor='start')
text(MX, legend_y + 16, '引擎循环五段（schedule→execute→bitmask→sample→update）；新 token 沿橙箭头上行回 API 拼字流出。', 9.5, C_MUTE, anchor='start')
text(MX, legend_y + 32, '第一设计原则：GPU 是最贵的员工，一切 CPU 活都不能让它等。第二原则：显存是共享账本，一切调度先对账。', 9.5, C_MUTE, anchor='start')

H = legend_y + 50
L[0] = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">'
L.append('</svg>')

svg = '\n'.join(L)
out = Path(__file__).parent / 'L0-architecture.svg'
out.write_text(svg, encoding='utf-8')
print(f'OK {out.absolute()} (viewBox 0 0 {W} {H})')
