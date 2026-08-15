#!/usr/bin/env python3
"""L0 全书唯一权威架构图 —— Fable5 重画版（2026-08-15）

完全重新设计的布局：更紧凑的垂直流、改进的视觉层次、优化的空间利用。
保留范式要素但重新排布：3 user / API 双泳道 / 中间带 / 引擎五段 / zoom 索引。
"""
import sys
from pathlib import Path
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

# 色彩方案
FONT = 'Microsoft YaHei, SimHei, Noto Sans CJK SC, PingFang SC, sans-serif'
C_TXT, C_MUTE = '#1e293b', '#64748b'
C_API_S, C_API_F = '#0ea5e9', '#e0f2fe'      # API = 天蓝
C_ENG_S, C_ENG_F = '#f59e0b', '#fef3c7'      # Engine = 琥珀
C_ZMQ_S, C_ZMQ_F = '#8b5cf6', '#ede9fe'      # ZMQ = 紫罗兰
C_ZOOM = '#94a3b8'                            # zoom 索引 = 石板灰
C_DOWN, C_UP = '#2563eb', '#dc2626'          # 下行蓝 / 上行红

W = 1450
L = []

def tw(s, fs=11, bold=False):
    """CJK 字符按 fs 宽度，ASCII 按 0.55×fs，粗体 ×1.08"""
    n_cjk = sum(1 for c in str(s) if ord(c) > 0x2E80)
    n_ascii = len(str(s)) - n_cjk
    base = n_cjk * fs + n_ascii * fs * 0.55
    return base * (1.08 if bold else 1.0)

def text(x, y, txt, fs=11, fill=C_TXT, anchor='middle', weight='normal'):
    w = f' font-weight="{weight}"' if weight != 'normal' else ''
    L.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{fs}" '
             f'fill="{fill}" text-anchor="{anchor}"{w}>{esc(txt)}</text>')

def rect(x, y, w, h, fill, stroke, rx=6, sw=1.5, dash=False):
    d = ' stroke-dasharray="5,3"' if dash else ''
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
             f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

def line(x1, y1, x2, y2, color=C_MUTE, sw=1.5, marker=None, dash=False):
    m = f' marker-end="url(#{marker})"' if marker else ''
    d = ' stroke-dasharray="5,3"' if dash else ''
    L.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
             f'stroke="{color}" stroke-width="{sw}"{d}{m}/>')

def box_node(x, y, w, h, title, sub='', fill='#ffffff', stroke=C_MUTE, rx=5):
    """方法名节点：标题 + 可选副标题"""
    rect(x, y, w, h, fill, stroke, rx=rx, sw=1.2)
    if sub:
        text(x + w/2, y + h/2 - 4, title, 10, C_TXT, weight='bold')
        text(x + w/2, y + h/2 + 9, sub, 8, C_MUTE)
    else:
        text(x + w/2, y + h/2 + 4, title, 10, C_TXT, weight='bold')

def queue_bars(x, y, w=24, h=12):
    """队列图标：三条竖杠"""
    gap = (w - 18) / 2
    for i in range(3):
        bx = x + i * (6 + gap)
        L.append(f'<rect x="{bx}" y="{y}" width="6" height="{h}" fill="{C_MUTE}" rx="1"/>')

# ========== SVG 初始化 ==========
L.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 980">')
L.append('<defs>'
         '<marker id="std" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="5" markerHeight="3.5" orient="auto">'
         f'<path d="M0,0 L10,3 L0,6 Z" fill="{C_MUTE}"/></marker>'
         '<marker id="dn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4.5" orient="auto">'
         f'<path d="M0,0 L10,3 L0,6 Z" fill="{C_DOWN}"/></marker>'
         '<marker id="up" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4.5" orient="auto">'
         f'<path d="M0,0 L10,3 L0,6 Z" fill="{C_UP}"/></marker>'
         '</defs>')
rect(0, 0, W, 980, '#fafafa', 'none')

# 标题区
text(28, 28, 'vLLM v1 架构（L0）：一个请求的一生', 17, C_TXT, 'start', 'bold')
text(28, 48, '三段式进程解耦 · 双泳道对偶 · 循环五段 · 显存共享账本', 10, C_MUTE, 'start')

MX, MY = 28, 68

# ========== 顶部：3 users 横排 ==========
user_w, user_gap = 110, 28
user_y = MY
users_start_x = W/2 - (3*user_w + 2*user_gap)/2  # 居中
for i in range(3):
    ux = users_start_x + i * (user_w + user_gap)
    rect(ux, user_y, user_w, 32, '#f1f5f9', C_MUTE, rx=4, sw=1.1)
    text(ux + user_w/2, user_y + 20, f'user{i+1}', 9.5, C_TXT, weight='bold')
    # sse 双向
    text(ux + user_w/2, user_y + 44, 'sse', 8, C_MUTE)
    line(ux + user_w/2, user_y + 32, ux + user_w/2, user_y + 50, C_MUTE, 1.1, 'std')

api_y = user_y + 62

# ========== API 进程（紧凑双泳道）==========
api_h = 255
rect(MX, api_y, W - 56, api_h, C_API_F, C_API_S, rx=7, sw=2.0)
text(MX + 14, api_y + 18, 'API 进程（frontend · 零 GPU）', 12, C_API_S, 'start', 'bold')
text(MX + W - 70, api_y + 18, 'HTTP/SSE 的 CPU 活全在此进程', 8.5, C_MUTE, 'end')

# 泳道分界线
swim_x = MX + (W - 56) / 2
line(swim_x, api_y + 30, swim_x, api_y + api_h - 8, C_MUTE, 1.0, dash=True)

# 右泳道（下行输入）
rw = 200
rx = MX + 52
nodes_right = [
    ('render', 'tokenize 在这里'),
    ('InputProcessor', 'id 双轨 + EngineCoreRequest'),
    ('assign_request_id', '外部 id + 8 位 hex'),
    ('add_request', '双登记：本地表 + 跨进程')
]
ry = api_y + 36
node_h = 34
node_gap = 14
for title, sub in nodes_right:
    box_node(rx, ry, rw, node_h, title, sub, '#ffffff', C_API_S)
    if title != 'add_request':
        line(rx + rw/2, ry + node_h, rx + rw/2, ry + node_h + node_gap, C_DOWN, 1.8, 'dn')
    ry += node_h + node_gap
api_dn_x = rx + rw/2
api_dn_y = api_y + api_h - 4

# 左泳道（上行输出）
lw = 200
lx = swim_x + 42
ly = api_y + 36
# 三收集器横排
coll_w = (lw - 20) / 3
for i in range(3):
    cx = lx + i * (coll_w + 10)
    rect(cx, ly, coll_w, 28, '#f8fafc', C_MUTE, rx=3, sw=1.0)
    text(cx + coll_w/2, ly + 18, f'Coll{i+1}', 8, C_TXT, weight='bold')
text(lx + lw/2, ly + 36, '单槽+Event 刻意不用 Queue', 7, C_MUTE)
line(lx + lw/2, ly + 48, lx + lw/2, ly + 60, C_UP, 1.8, 'up')

nodes_left = [
    ('process_outputs', 'demux 到 Collector'),
    ('output_handler', 'chunk+sleep(0) 让循环喘气'),
    ('PULL socket', '上行接收端')
]
ly += 60
for title, sub in nodes_left:
    box_node(lx, ly, lw, node_h, title, sub, '#ffffff', C_ENG_S)
    if title != 'PULL socket':
        line(lx + lw/2, ly + node_h, lx + lw/2, ly + node_h + node_gap, C_UP, 1.8, 'up')
    ly += node_h + node_gap
api_up_x = lx + lw/2
api_up_y = api_y + api_h - 4

# user 连线到泳道
for i in range(3):
    u_cx = users_start_x + user_w/2 + i * (user_w + user_gap)
    # 下行
    line(u_cx, user_y + 50, rx + 24 + i * 60, api_y + 36, C_DOWN, 1.2, 'dn')
    # 上行
    line(lx + 24 + i * 60, api_y + 36, u_cx, user_y + 50, C_UP, 1.2, 'up')

zmq_y = api_y + api_h + 10

# ========== AsyncMpClient 中间带 ==========
zmq_h = 72
rect(MX, zmq_y, W - 56, zmq_h, C_ZMQ_F, C_ZMQ_S, rx=6, sw=1.8)
text(MX + 14, zmq_y + 16, 'AsyncMpClient（中间带）', 11, C_ZMQ_S, 'start', 'bold')
# 进程边界标签（左侧竖排）
text(MX + 22, zmq_y + zmq_h/2 + 14, '进程边界', 10.5, C_ZMQ_S, 'start', 'bold')
text(MX + 22, zmq_y + zmq_h/2 + 28, 'zmq+msgpack', 8.5, C_MUTE, 'start')

# 双通道节点
ch_w = 170
ch_x_r = MX + 180
ch_x_l = MX + 180 + ch_w + 90
ch_y = zmq_y + 20
box_node(ch_x_r, ch_y, ch_w, 30, 'add_request_async', 'ROUTER→DEALER', '#ffffff', C_ZMQ_S)
box_node(ch_x_l, ch_y, ch_w, 30, 'get_output_async', 'PUSH→PULL', '#ffffff', C_ZMQ_S)

# 贯穿箭头（下行）
line(api_dn_x, api_dn_y, api_dn_x, zmq_y, C_DOWN, 2.5, 'dn')
line(ch_x_r + ch_w/2, ch_y + 30, ch_x_r + ch_w/2, zmq_y + zmq_h, C_DOWN, 2.5, 'dn')
text(api_dn_x + 10, api_y + api_h + 22, 'EngineCoreRequest', 8.5, C_DOWN, 'start')
text(api_dn_x + 10, api_y + api_h + 35, '（只有 token ids）', 7.5, C_MUTE, 'start')

# 贯穿箭头（上行）
line(api_up_x, zmq_y, api_up_x, api_up_y, C_UP, 2.5, 'up')
line(ch_x_l + ch_w/2, zmq_y + zmq_h, ch_x_l + ch_w/2, ch_y + 30, C_UP, 2.5, 'up')
text(api_up_x - 10, api_y + api_h + 22, 'EngineCoreOutputs', 8.5, C_UP, 'end')
text(api_up_x - 10, api_y + api_h + 35, '（每步整批聚合）', 7.5, C_MUTE, 'end')

eng_y = zmq_y + zmq_h + 10

# ========== EngineCore 进程（三框布局）==========
eng_h = 380
rect(MX, eng_y, W - 56, eng_h, C_ENG_F, C_ENG_S, rx=7, sw=2.0)
text(MX + 14, eng_y + 18, 'EngineCore 进程（busy loop）', 12, C_ENG_S, 'start', 'bold')
text(MX + W - 70, eng_y + 18, 'GPU 上下文只在此进程', 8.5, C_MUTE, 'end')

# 三框：input_queue（右）/ 循环（中）/ output_queue（左）
qw = 130
loop_y = eng_y + 36

# 右框：input_queue
qx_r = MX + 36
rect(qx_r, loop_y, qw, 50, '#ffffff', C_MUTE, rx=5, sw=1.2)
text(qx_r + qw/2, loop_y + 15, 'input_queue', 9.5, C_TXT, weight='bold')
text(qx_r + qw/2, loop_y + 30, 'DEALER', 8.5, C_MUTE)
queue_bars(qx_r + qw/2 - 12, loop_y + 35)

# 中框：循环五段
loop_x = qx_r + qw + 26
loop_w = 750
rect(loop_x, loop_y, loop_w, 150, '#ffffff', C_ENG_S, rx=6, sw=1.6)
text(loop_x + 10, loop_y + 16, 'EngineCore.step() 逐拍循环', 10.5, C_ENG_S, 'start', 'bold')

steps = [
    ('① schedule', 'RUNNING 先于 WAITING'),
    ('② execute_model', 'non_block→Future'),
    ('③ get_grammar_bitmask', 'CPU 窗口'),
    ('④ sample_tokens', 'logits+bitmask'),
    ('⑤ update_from_output', '状态推进+free 块')
]
step_w = (loop_w - 30 - 4 * 18) / 5
step_y = loop_y + 28
for i, (title, sub) in enumerate(steps):
    sx = loop_x + 15 + i * (step_w + 18)
    rect(sx, step_y, step_w, 46, '#fde68a', '#f59e0b', rx=4, sw=1.1)
    text(sx + step_w/2, step_y + 18, title, 9, C_TXT, weight='bold')
    text(sx + step_w/2, step_y + 34, sub, 7.5, C_MUTE)
    if i < 4:
        line(sx + step_w, step_y + 23, sx + step_w + 18, step_y + 23, C_ENG_S, 1.5, 'std')

# 左框：output_queue
qx_l = loop_x + loop_w + 26
rect(qx_l, loop_y, qw, 50, '#ffffff', C_MUTE, rx=5, sw=1.2)
text(qx_l + qw/2, loop_y + 15, 'output_queue', 9.5, C_TXT, weight='bold')
text(qx_l + qw/2, loop_y + 30, 'PUSH', 8.5, C_MUTE)
queue_bars(qx_l + qw/2 - 12, loop_y + 35)

# 循环流动箭头
line(qx_r + qw, loop_y + 25, loop_x, loop_y + 25, C_ENG_S, 1.8, 'std')
line(loop_x + loop_w, loop_y + 25, qx_l, loop_y + 25, C_ENG_S, 1.8, 'std')
# 回环
cy = step_y + 46
line(loop_x + loop_w/2, cy, loop_x + loop_w/2, cy + 12, C_MUTE, 1.4)
line(loop_x + loop_w/2, cy + 20, loop_x + loop_w/2, cy + 12, C_MUTE, 1.4, 'std')
text(loop_x + loop_w/2 + 10, cy + 16, '下一拍', 8, C_MUTE, 'start')

# 三个 zoom 索引框
zoom_y = step_y + 80
zoom_w = (loop_w - 40) / 3
zoom_h = 58
zooms = [
    ('调度↔KV 对账', 'KV 块池·前缀缓存'),
    ('GPU 执行臂', 'Executor→Worker→Runner'),
    ('采样与出口', 'Sampler 9 步管线')
]
for i, (title, sub) in enumerate(zooms):
    zx = loop_x + 15 + i * (zoom_w + 20)
    rect(zx, zoom_y, zoom_w, zoom_h, '#fafafa', C_ZOOM, rx=4, sw=1.1, dash=True)
    text(zx + zoom_w/2, zoom_y + 22, title, 9.5, C_TXT, weight='bold')
    text(zx + zoom_w/2, zoom_y + 38, sub, 8, C_MUTE)
    text(zx + zoom_w/2, zoom_h - 8 + zoom_y, '→ L1/L2 放大', 7, C_ZOOM)

# 图例
legend_y = eng_y + eng_h + 20
text(MX, legend_y, '读图：请求自顶部 users 进入，蓝箭头下行穿 API 双泳道→中间带→引擎输入队列；', 9, C_MUTE, 'start')
text(MX, legend_y + 15, '引擎循环五段（schedule→execute→bitmask→sample→update）生成新 token；', 9, C_MUTE, 'start')
text(MX, legend_y + 30, '红箭头上行回 API 进程 detokenize 后流式返回。第一原则：GPU 不等 CPU。第二原则：显存共享账本。', 9, C_MUTE, 'start')

H = legend_y + 48
L[0] = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">'
L.append('</svg>')

svg_text = '\n'.join(L)
out = Path(__file__).parent / 'L0-architecture.svg'
out.write_text(svg_text, encoding='utf-8')
print(f'✓ {out} (viewBox 0 0 {W} {H})')
