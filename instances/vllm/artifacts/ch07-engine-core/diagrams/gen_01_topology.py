#!/usr/bin/env python3
"""01-ipc-socket-topology: ZMQ socket 拓扑总图。"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(s)

w, h = 1000, 640
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append('<defs>')
L.append('<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker>')
L.append('<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>')
L.append('<marker id="ao" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker>')
L.append('<marker id="am" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#9333ea"/></marker>')
L.append('</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

def box(x, y, bw, bh, fill, stroke, rx=8):
    L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

def txt(x, y, s, size=15, anchor="middle", fill="#1e293b", weight="normal"):
    L.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{esc(s)}</text>')

# 进程框
box(30, 70, 380, 480, "#eff6ff", "#1d4ed8")
txt(220, 56, "前端进程  (AsyncLLM / LLM)", 17, weight="bold", fill="#1d4ed8")
box(560, 70, 410, 480, "#f0fdf4", "#15803d")
txt(765, 56, "EngineCoreProc 进程", 17, weight="bold", fill="#15803d")

# client sockets
box(70, 120, 300, 70, "#dbeafe", "#2563eb")
txt(220, 150, "input_socket = ROUTER", 15, weight="bold")
txt(220, 172, "bind  tcp://127.0.0.1:*", 13, fill="#475569")

box(70, 420, 300, 70, "#dbeafe", "#2563eb")
txt(220, 450, "output_socket = PULL", 15, weight="bold")
txt(220, 472, "MsgpackDecoder.decode", 13, fill="#475569")

# engine sockets + threads + queues + busy loop
box(600, 120, 330, 64, "#dcfce7", "#16a34a")
txt(765, 147, "DEALER  (connect, identity)", 14, weight="bold")
txt(765, 167, "process_input_sockets 线程", 12, fill="#475569")

box(600, 250, 150, 56, "#fef9c3", "#ca8a04")
txt(675, 274, "input_queue", 13, weight="bold")
txt(675, 294, "(queue.Queue)", 11, fill="#475569")

box(780, 250, 150, 56, "#fef9c3", "#ca8a04")
txt(855, 274, "output_queue", 13, weight="bold")
txt(855, 294, "(queue.Queue)", 11, fill="#475569")

box(640, 350, 250, 56, "#bbf7d0", "#15803d")
txt(765, 375, "run_busy_loop  (主线程)", 14, weight="bold")
txt(765, 395, "消费 input → step → 产出 output", 11, fill="#475569")

box(600, 430, 330, 64, "#dcfce7", "#16a34a")
txt(765, 457, "PUSH", 14, weight="bold")
txt(765, 477, "process_output_sockets 线程", 12, fill="#475569")

# 请求 ROUTER -> DEALER (蓝)
L.append(f'<line x1="370" y1="145" x2="600" y2="148" stroke="#2563eb" stroke-width="2.5" marker-end="url(#ar)"/>')
txt(485, 132, "① 请求  type+帧", 13, fill="#2563eb", weight="bold")

# 握手 ready DEALER -> ROUTER (绿虚线, 先发)
L.append(f'<line x1="600" y1="168" x2="372" y2="178" stroke="#16a34a" stroke-width="2.5" stroke-dasharray="7 4" marker-end="url(#ag)"/>')
txt(486, 200, "⓪ ready 帧（DEALER 必先发）", 12, fill="#16a34a", weight="bold")

# DEALER -> input_queue
L.append(f'<line x1="675" y1="184" x2="675" y2="250" stroke="#16a34a" stroke-width="2" marker-end="url(#ag)"/>')
# input_queue -> busy loop
L.append(f'<line x1="690" y1="306" x2="720" y2="350" stroke="#16a34a" stroke-width="2" marker-end="url(#ag)"/>')
# busy loop -> output_queue
L.append(f'<line x1="820" y1="350" x2="850" y2="306" stroke="#16a34a" stroke-width="2" marker-end="url(#ag)"/>')
# output_queue -> PUSH
L.append(f'<line x1="855" y1="306" x2="855" y2="430" stroke="#16a34a" stroke-width="2" marker-end="url(#ag)"/>')

# 输出 PUSH -> PULL (灰)
L.append(f'<line x1="600" y1="462" x2="372" y2="455" stroke="#94a3b8" stroke-width="2.5" marker-end="url(#ao)"/>')
txt(486, 432, "③ 输出  EngineCoreOutputs", 13, fill="#64748b", weight="bold")

# OOB tensor channel
box(120, 560, 200, 56, "#f3e8ff", "#9333ea")
txt(220, 584, "TensorIpcSender", 13, weight="bold", fill="#7e22ce")
txt(220, 604, "share_memory_()", 11, fill="#475569")
box(660, 560, 220, 56, "#f3e8ff", "#9333ea")
txt(770, 584, "TensorIpcReceiver", 13, weight="bold", fill="#7e22ce")
txt(770, 604, "drain-and-buffer", 11, fill="#475569")
L.append(f'<line x1="320" y1="588" x2="660" y2="588" stroke="#9333ea" stroke-width="2.5" stroke-dasharray="2 3" marker-end="url(#am)"/>')
txt(490, 578, "torch.mp.Queue 共享内存（OOB 大张量旁路，不进 ZMQ 帧）", 12, fill="#7e22ce", weight="bold")

L.append('</svg>')
open("01-ipc-socket-topology.svg", "w").write('\n'.join(L))
print("wrote 01-ipc-socket-topology.svg")
