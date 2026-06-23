#!/usr/bin/env python3
"""ch17-control-plane-topology: star control plane — 1 Executor ⇄ N WorkerProc."""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

w, h = 1180, 760
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append('<defs>'
         '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
         '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
         '<marker id="ab" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker>'
         '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

def box(x, y, bw, bh, fill, stroke, lines, fs=14, tw="normal", tc="#0f172a", sw=1.5):
    L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    n = len(lines)
    cy = y + bh/2 - (n-1)*(fs+4)/2 + fs/2 - 2
    for i, t in enumerate(lines):
        fw = tw if i == 0 else "normal"
        fc = tc if i == 0 else "#475569"
        L.append(f'<text x="{x+bw/2}" y="{cy+i*(fs+4)}" text-anchor="middle" font-size="{fs}" '
                 f'font-weight="{fw}" fill="{fc}">{esc(t)}</text>')

def txt(x, y, t, fs=14, anchor="start", fill="#334155", tw="normal"):
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" font-weight="{tw}" fill="{fill}">{esc(t)}</text>')

txt(40, 42, "控制平面拓扑：1 个 MultiprocExecutor 进程 ⇄ N 个 WorkerProc 子进程", 21, "start", "#0f172a", "bold")

# Executor process box (top)
ex, ey, ew, eh = 300, 70, 580, 160
box(ex, ey, ew, eh, "#eef2ff", "#6366f1", [], 14, "bold", sw=2)
txt(ex+16, ey+26, "MultiprocExecutor（执行器进程）", 15, "start", "#3730a3", "bold")
# inner components
box(ex+24, ey+44, 250, 46, "#ffffff", "#6366f1", ["rpc_broadcast_mq", "共享内存广播 MQ"], 12.5, "bold")
box(ex+300, ey+44, 250, 46, "#ffffff", "#6366f1", ["response_mqs[ ]", "每 worker 一条应答 MQ"], 12.5, "bold")
box(ex+24, ey+100, 250, 44, "#ffffff", "#6366f1", ["futures_queue (deque)"], 12.5, "bold")
box(ex+300, ey+100, 250, 44, "#fff7ed", "#ea580c", ["MultiprocWorkerMonitor 线程"], 12.5, "bold")

# Worker subprocesses (bottom row)
ranks = [0, 1, 2, 3]
wy, wbw, wbh = 540, 220, 150
gap = (w - 80 - len(ranks)*wbw) / (len(ranks)-1)
wxs = []
for i, r in enumerate(ranks):
    wx = 40 + i*(wbw+gap)
    wxs.append(wx)
    out = (r == 0)  # output_rank = rank 0 in this single-PP example
    fill = "#ecfdf5" if out else "#f8fafc"
    stroke = "#10b981" if out else "#64748b"
    box(wx, wy, wbw, wbh, fill, stroke, [], 13, "bold", sw=2 if out else 1.5)
    tag = f"WorkerProc  rank={r}"
    if out:
        tag += "  ★output_rank"
    txt(wx+wbw/2, wy+24, tag, 12.5, "middle", "#065f46" if out else "#334155", "bold")
    box(wx+16, wy+38, wbw-32, 40, "#ffffff", stroke, ["worker_busy_loop"], 12)
    box(wx+16, wy+86, wbw-32, 46, "#ffffff", stroke, ["Worker (GPU)", "1 张卡"], 11.5)

# broadcast arrow: one enqueue -> all workers (thick)
bsx = ex+24+125  # center of broadcast mq
bsy = ey+44+46
for wx in wxs:
    L.append(f'<line x1="{bsx}" y1="{bsy}" x2="{wx+wbw/2-30}" y2="{wy-4}" '
             f'stroke="#2563eb" stroke-width="3" marker-end="url(#ab)" opacity="0.85"/>')
txt(bsx-150, 330, "一次 enqueue，N 个 reader 各看到一份（O(1) 次发送）", 13, "start", "#1d4ed8", "bold")

# response arrows: each worker -> response_mqs (thin); only rank0 highlighted
rmx = ex+300+125
rmy = ey+44+46
for i, wx in enumerate(wxs):
    out = (ranks[i] == 0)
    col = "#dc2626" if out else "#94a3b8"
    mk = "ar" if out else "ag"
    sw = 2.4 if out else 1.4
    dash = '' if out else ' stroke-dasharray="4,3"'
    L.append(f'<line x1="{wx+wbw/2+30}" y1="{wy-4}" x2="{rmx+(i-1.5)*8}" y2="{rmy}" '
             f'stroke="{col}" stroke-width="{sw}" marker-end="url(#{mk})"{dash}/>')
txt(rmx+150, 360, "execute_model 只收 output_rank 一条（红），", 12.5, "start", "#b91c1c", "bold")
txt(rmx+150, 380, "其余应答 MQ 在该指令下不回写（灰虚）", 12.5, "start", "#64748b")

# sentinel dashed lines: each worker -> monitor
mx = ex+300+125
my = ey+100+44
for wx in wxs:
    L.append(f'<line x1="{wx+wbw/2}" y1="{wy-4}" x2="{mx}" y2="{my+2}" '
             f'stroke="#ea580c" stroke-width="1.3" stroke-dasharray="2,4" opacity="0.7"/>')
txt(40, 470, "虚橙线 = sentinel：子进程一旦死亡，multiprocessing.connection.wait 立即唤醒监控线程", 12.5, "start", "#9a3412")

svg = '\n'.join(L) + '\n</svg>\n'
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch17-worker-and-executor/diagrams/ch17-control-plane-topology.svg", "w", encoding="utf-8").write(svg)
print("wrote 01")
