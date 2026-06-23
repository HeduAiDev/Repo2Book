#!/usr/bin/env python3
"""ch17-future-fifo-pairing: FutureWrapper + futures_queue FIFO drain vs response MQ FIFO."""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

w, h = 1160, 720
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append('<defs>'
         '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
         '<marker id="ap" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

def box(x, y, bw, bh, fill, stroke, lines, fs=13, tw="bold", tc="#0f172a", sw=1.5):
    L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    n = len(lines)
    cy = y + bh/2 - (n-1)*(fs+3)/2 + fs/2 - 2
    for i, t in enumerate(lines):
        fw = tw if i == 0 else "normal"
        fc = tc if i == 0 else "#475569"
        L.append(f'<text x="{x+bw/2}" y="{cy+i*(fs+3)}" text-anchor="middle" font-size="{fs}" font-weight="{fw}" fill="{fc}">{esc(t)}</text>')

def txt(x, y, t, fs=13, anchor="start", fill="#334155", tw="normal"):
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" font-weight="{tw}" fill="{fill}">{esc(t)}</text>')

txt(40, 40, "FutureWrapper + futures_queue：FIFO 配对让回复对上请求", 21, "start", "#0f172a", "bold")

cols = [("#2563eb", "#dbeafe", "#1d4ed8"),
        ("#16a34a", "#dcfce7", "#15803d"),
        ("#ea580c", "#ffedd5", "#c2410c")]

# Top: caller fires 3 non_block RPCs
txt(40, 88, "① 调用方依次 collective_rpc(..., non_block=True)，每次生成 FutureWrapper 并 appendleft 进 deque", 13.5, "start", "#334155", "bold")
fx0, fbw, fgap = 70, 230, 40
for i in range(3):
    fx = fx0 + i*(fbw+fgap)
    s, f, t = cols[i]
    box(fx, 105, fbw, 50, f, s, [f"RPC#{i+1} → FutureWrapper#{i+1}"], 13, "bold", t)

# deque (futures_queue) — appendleft means newest at left, oldest at right (pop side)
dy = 210
txt(40, dy-12, "② futures_queue（deque）：appendleft 入左端，pop 从右端取 → 右端 = 最早发出的请求", 13.5, "start", "#334155", "bold")
dx0 = 360
cellw = 150
labels = ["#3 (左/新)", "#2", "#1 (右/旧)"]
order = [3, 2, 1]  # left to right
for i, n in enumerate(order):
    s, f, t = cols[n-1]
    box(dx0 + i*cellw, dy, cellw-8, 50, f, s, [f"Future#{n}"], 13, "bold", t)
txt(dx0-10, dy+30, "appendleft →", 12, "end", "#7c3aed", "bold")
txt(dx0+3*cellw+4, dy+30, "← pop", 12, "start", "#7c3aed", "bold")

# arrows from top futures into deque positions
for i in range(3):
    fx = fx0 + i*(fbw+fgap) + fbw/2
    # future #(i+1) -> its deque slot
    n = i+1
    slot = order.index(n)
    tx = dx0 + slot*cellw + (cellw-8)/2
    L.append(f'<line x1="{fx}" y1="155" x2="{tx}" y2="{dy-4}" stroke="#7c3aed" stroke-width="1.4" marker-end="url(#ap)" stroke-dasharray="4,3" opacity="0.8"/>')

# response MQ (FIFO) on the right-ish, vertical
my0 = 320
txt(40, my0-12, "③ 底层 response MQ 是 FIFO：回复顺序 = 发出顺序（worker 按收到次序回写）", 13.5, "start", "#334155", "bold")
mqx = 850
for i, n in enumerate([1, 2, 3]):  # FIFO: #1 first out
    s, f, t = cols[n-1]
    box(mqx, my0 + i*56, 220, 46, f, s, [f"reply #{n}"], 12.5, "bold", t)
txt(mqx+110, my0+3*56+28, "↑ dequeue 顺序：#1 → #2 → #3", 12, "middle", "#475569", "bold")

# result() drain illustration (bottom-left)
ry0 = 410
txt(40, ry0, "④ 对 Future#3 调 result()：while not done → pop 出队尾，先把排在前面的排空", 13.5, "start", "#334155", "bold")
steps = [
    ("pop #1 → _wait_for_response()", "取 reply#1，set_result，Future#1 done", cols[0]),
    ("pop #2 → _wait_for_response()", "取 reply#2，set_result，Future#2 done", cols[1]),
    ("此时队列只剩 #3 自己", "取 reply#3，set_result，返回给调用方", cols[2]),
]
sy = ry0 + 24
for i, (a, b, c) in enumerate(steps):
    s, f, t = c
    box(70, sy + i*70, 460, 56, f, s, [a, b], 12.5, "bold", t)
    # arrow to the matching MQ cell
    L.append(f'<line x1="530" y1="{sy + i*70 + 28}" x2="{mqx-6}" y2="{my0 + i*56 + 23}" stroke="{s}" stroke-width="1.6" marker-end="url(#a)" opacity="0.7"/>')

txt(40, h-30, "结论：第 k 个发出的请求恒配第 k 个回复 —— 无需给每个 RPC 配 id 或独立通道，两个 FIFO 同序即可。", 13.5, "start", "#7c3aed", "bold")

svg = '\n'.join(L) + '\n</svg>\n'
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch17-worker-and-executor/diagrams/ch17-future-fifo-pairing.svg", "w", encoding="utf-8").write(svg)
print("wrote 03")
