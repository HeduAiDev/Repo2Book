#!/usr/bin/env python3
"""ch17-worker-lifecycle-timeline: two-swimlane sequence, Executor(parent) vs WorkerProc(child)."""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

w, h = 1140, 880
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append('<defs>'
         '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
         '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
         '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

def txt(x, y, t, fs=14, anchor="start", fill="#334155", tw="normal"):
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" font-weight="{tw}" fill="{fill}">{esc(t)}</text>')

txt(40, 40, "Worker 子进程生命周期：出生 → READY 握手 → 服役 → 死亡", 21, "start", "#0f172a", "bold")

# two lifelines
px, cx = 300, 840   # parent x, child x
top, bot = 80, 840
L.append(f'<rect x="{px-150}" y="{top-30}" width="300" height="40" rx="6" fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>')
txt(px, top-3, "Executor（父进程）", 15, "middle", "#3730a3", "bold")
L.append(f'<rect x="{cx-150}" y="{top-30}" width="300" height="40" rx="6" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.5"/>')
txt(cx, top-3, "WorkerProc（子进程）", 15, "middle", "#15803d", "bold")
L.append(f'<line x1="{px}" y1="{top+14}" x2="{px}" y2="{bot}" stroke="#94a3b8" stroke-width="1.5"/>')
L.append(f'<line x1="{cx}" y1="{top+14}" x2="{cx}" y2="{bot}" stroke="#94a3b8" stroke-width="1.5"/>')

def activ(x, y0, y1, fill, stroke):
    L.append(f'<rect x="{x-14}" y="{y0}" width="28" height="{y1-y0}" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')

def msg(y, x0, x1, label, col="#475569", mk="a", dash=False, fs=12.5, lab_anchor="middle"):
    d = ' stroke-dasharray="5,4"' if dash else ''
    L.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{col}" stroke-width="1.6" marker-end="url(#{mk})"{d}/>')
    lx = (x0+x1)/2
    txt(lx, y-7, label, fs, lab_anchor, col, "bold")

def note(x, y, ww, lines, fill="#fefce8", stroke="#ca8a04"):
    hh = 16 + len(lines)*17
    L.append(f'<rect x="{x}" y="{y}" width="{ww}" height="{hh}" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
    for i, t in enumerate(lines):
        txt(x+10, y+20+i*17, t, 11.5, "start", "#713f12")

# sequence
y = 130
msg(y, px, cx, "make_worker_process：起进程 + ready/death pipe", "#475569", "a")
y += 46
activ(cx, y-6, y+150, "#dcfce7", "#16a34a")
note(cx+24, y-6, 260, ["worker_main：装 SIGTERM/SIGINT 信号处理",
                       "WorkerWrapperBase.init_worker",
                       "  └ 按 worker_cls qualname 解析真实 Worker",
                       "init_device（建设备+分布式组+model_runner）",
                       "load_model"])
y += 150
msg(y, cx, px, "ready_pipe.send({status: READY, handle})", "#16a34a", "ag")
y += 40
activ(px, y-6, y+60, "#e0e7ff", "#6366f1")
note(px-290, y-6, 300, ["wait_for_ready：收齐所有 worker 的 READY",
                        "（任一 worker 未就绪就不发 RPC，避免死锁）"], "#eef2ff", "#6366f1")
y += 60
msg(y, px, px-200, "start_worker_monitor 起监控线程", "#ea580c", "a")
y += 50

# steady state box
L.append(f'<rect x="{px-180}" y="{y}" width="{cx-px+360}" height="150" rx="8" fill="#f8fafc" stroke="#0ea5e9" stroke-width="1.5" stroke-dasharray="6,4"/>')
txt(px-170, y+22, "稳态：worker_busy_loop 循环服役", 14, "start", "#0369a1", "bold")
yy = y+50
msg(yy, px, cx, "rpc_broadcast_mq.enqueue((method, args, output_rank))", "#2563eb", "a", fs=12)
yy += 44
note(cx+24, yy-14, 250, ["dequeue → getattr(self.worker, method)",
                         "或 cloudpickle.loads(callable)",
                         "执行 → handle_output 回写"], "#eff6ff", "#3b82f6")
yy += 56
msg(yy, cx, px, "worker_response_mq.enqueue((SUCCESS, output))", "#16a34a", "ag", fs=12)
y += 150 + 30

# failure branches
txt(40, y+18, "两条失败支线：", 14, "start", "#0f172a", "bold")
y += 36
msg(y, cx, px, "方法抛异常 → enqueue((FAILURE, str(e))) → collective_rpc 抛 RuntimeError", "#dc2626", "ar", fs=12, lab_anchor="middle")
y += 44
L.append(f'<line x1="{cx}" y1="{y}" x2="{px-200}" y2="{y}" stroke="#dc2626" stroke-width="1.6" stroke-dasharray="3,3" marker-end="url(#ar)"/>')
txt((cx+px-200)/2, y-7, "进程被 OS 杀死（OOM/segfault）→ sentinel 唤醒 monitor → failure_callback", 12, "middle", "#b91c1c", "bold")
y += 40
msg(y, px, cx, "shutdown：关 death_writer → graceful → SIGTERM → SIGKILL", "#475569", "a", fs=12)

svg = '\n'.join(L) + '\n</svg>\n'
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch17-worker-and-executor/diagrams/ch17-worker-lifecycle-timeline.svg", "w", encoding="utf-8").write(svg)
print("wrote 02")
