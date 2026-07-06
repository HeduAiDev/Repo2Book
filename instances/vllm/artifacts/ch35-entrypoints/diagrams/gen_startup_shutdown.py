#!/usr/bin/env python3
"""ch32 图2：服务器启动序与两条关停路径（信号 / watchdog）。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 940, 760
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '<marker id="arR" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append('<text x="30" y="34" font-size="18" font-weight="bold" fill="#0f172a">启动序（run_server）</text>')

steps = [
    ("setup_server", "先绑 socket + 装 SIGTERM handler（早于引擎，避端口竞争）", "#fef9c3", "#ca8a04"),
    ("build_async_engine_client", "AsyncLLM.from_vllm_config 起异步引擎（ch04 三段式）", "#dcfce7", "#16a34a"),
    ("build_and_serve", "get_supported_tasks → build_app → init_app_state", "#e0f2fe", "#0284c7"),
    ("serve_http (launcher)", "起 server_task + watchdog_task + handle_shutdown", "#e0f2fe", "#0284c7"),
    ("lifespan 启动期", "拉起 _force_log 后台任务 + freeze_gc_heap", "#f1f5f9", "#94a3b8"),
]
x0 = 60
bw = 820
bh = 50
gap = 22
y = 52
bottoms = []
for title, desc, fill, st in steps:
    L.append(f'<rect x="{x0}" y="{y}" width="{bw}" height="{bh}" rx="8" '
             f'fill="{fill}" stroke="{st}" stroke-width="1.6"/>')
    L.append(f'<text x="{x0 + 16}" y="{y + 21}" font-size="15" font-weight="bold" '
             f'fill="#0f172a" font-family="monospace">{esc(title)}</text>')
    L.append(f'<text x="{x0 + 16}" y="{y + 40}" font-size="13" fill="#334155">{esc(desc)}</text>')
    bottoms.append(y + bh)
    y += bh + gap

cx = x0 + bw / 2
for i in range(len(steps) - 1):
    L.append(f'<line x1="{cx}" y1="{bottoms[i]}" x2="{cx}" y2="{bottoms[i] + gap}" '
             f'stroke="#475569" stroke-width="1.8" marker-end="url(#ar)"/>')

# serving steady state
serv_y = y + 6
# arrow from last step (lifespan) bottom to steady-state box top
L.append(f'<line x1="{cx}" y1="{bottoms[-1]}" x2="{cx}" y2="{serv_y}" '
         f'stroke="#475569" stroke-width="1.8" marker-end="url(#ar)"/>')
L.append(f'<rect x="{x0}" y="{serv_y}" width="{bw}" height="42" rx="8" '
         f'fill="#ffffff" stroke="#475569" stroke-width="1.6" stroke-dasharray="6 4"/>')
L.append(f'<text x="{cx}" y="{serv_y + 26}" text-anchor="middle" font-size="14" '
         f'font-weight="bold" fill="#0f172a">… await server_task：稳态服务请求 …</text>')

# shutdown title
sy = serv_y + 80
L.append(f'<text x="30" y="{sy}" font-size="18" font-weight="bold" fill="#b91c1c">两条关停路径</text>')

# two columns
col_y = sy + 18
colw = 395
gapc = 30
lx = x0
rx = x0 + colw + gapc

# left: signal path
L.append(f'<rect x="{lx}" y="{col_y}" width="{colw}" height="156" rx="10" '
         f'fill="#fef2f2" stroke="#b91c1c" stroke-width="1.6"/>')
L.append(f'<text x="{lx + 16}" y="{col_y + 26}" font-size="14" font-weight="bold" '
         f'fill="#b91c1c">A. 信号触发（SIGINT / SIGTERM）</text>')
sig_lines = [
    "signal_handler → shutdown_event.set()",
    "handle_shutdown 醒来：",
    "engine.shutdown(timeout)  (run_in_executor)",
    "server.should_exit = True",
    "cancel server_task / watchdog_task",
]
for i, t in enumerate(sig_lines):
    fam = 'monospace' if ('=' in t or '(' in t or 'shutdown' in t.lower()) else 'sans-serif'
    L.append(f'<text x="{lx + 24}" y="{col_y + 50 + i * 21}" font-size="12.5" '
             f'fill="#334155" font-family="{fam}">{esc(t)}</text>')

# right: watchdog path
L.append(f'<rect x="{rx}" y="{col_y}" width="{colw}" height="156" rx="10" '
         f'fill="#fff7ed" stroke="#ea580c" stroke-width="1.6"/>')
L.append(f'<text x="{rx + 16}" y="{col_y + 26}" font-size="14" font-weight="bold" '
         f'fill="#ea580c">B. watchdog 兜底（引擎暗死）</text>')
wd_lines = [
    "watchdog_loop 每 5s 醒一次：",
    "terminate_if_errored 检查",
    "engine.errored and not is_running",
    "→ server.should_exit = True",
    "（流式 200 已发，异常吞进生成器）",
]
for i, t in enumerate(wd_lines):
    fam = 'monospace' if ('=' in t or 'errored' in t or 'terminate' in t or 'watchdog' in t) else 'sans-serif'
    L.append(f'<text x="{rx + 24}" y="{col_y + 50 + i * 21}" font-size="12.5" '
             f'fill="#334155" font-family="{fam}">{esc(t)}</text>')

# converge
conv_y = col_y + 156 + 30
mcx = W / 2
L.append(f'<line x1="{lx + colw / 2}" y1="{col_y + 156}" x2="{lx + colw / 2}" y2="{conv_y - 14}" '
         f'stroke="#b91c1c" stroke-width="1.7"/>')
L.append(f'<line x1="{rx + colw / 2}" y1="{col_y + 156}" x2="{rx + colw / 2}" y2="{conv_y - 14}" '
         f'stroke="#ea580c" stroke-width="1.7"/>')
L.append(f'<line x1="{lx + colw / 2}" y1="{conv_y - 14}" x2="{rx + colw / 2}" y2="{conv_y - 14}" '
         f'stroke="#94a3b8" stroke-width="1.7"/>')
L.append(f'<line x1="{mcx}" y1="{conv_y - 14}" x2="{mcx}" y2="{conv_y}" '
         f'stroke="#94a3b8" stroke-width="1.8" marker-end="url(#ar)"/>')

fin_h = 44
L.append(f'<rect x="{x0}" y="{conv_y}" width="{bw}" height="{fin_h}" rx="8" '
         f'fill="#f1f5f9" stroke="#475569" stroke-width="1.6"/>')
L.append(f'<text x="{mcx}" y="{conv_y + 27}" text-anchor="middle" font-size="13.5" fill="#0f172a">'
         f'{esc("退出 build_async_engine_client 上下文 → async_llm.shutdown → await shutdown_task → sock.close()")}</text>')

L.append('</svg>')
open("ch32-startup-shutdown.svg", "w").write('\n'.join(L))
print("wrote ch32-startup-shutdown.svg")
