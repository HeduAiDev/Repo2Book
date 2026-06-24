#!/usr/bin/env python3
"""两个入口，一套核心：离线同步 vs 在线异步，汇聚到同一 step / process_outputs。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1000, 660
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def text(x, y, s, size=14, anchor="start", weight="normal", fill="#1e293b"):
    L.append(
        f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" '
        f'text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{esc(s)}</text>'
    )


def box(x, y, w, h, fill, stroke, lines, size=13, tcolor="#1e293b", weight="normal", rx=8):
    L.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    )
    cy = y + h / 2 - (len(lines) - 1) * (size + 2) / 2 + size / 2 - 1
    for i, ln in enumerate(lines):
        text(x + w / 2, cy + i * (size + 2), ln, size=size, anchor="middle",
             weight=weight, fill=tcolor)


def arrow(x1, y1, x2, y2, color="#475569", dash=False, width=2):
    d = ' stroke-dasharray="6 4"' if dash else ''
    L.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{width}"{d} marker-end="url(#a)"/>'
    )


text(W / 2, 36, "两个入口，一套核心", size=22, anchor="middle", weight="bold")

# ── Left column: offline LLM ──
lx = 40
lw = 380
box(lx, 70, lw, 32, "#eff6ff", "#93c5fd", ["离线 · 同进程 · 无 IPC · FINAL_ONLY"], size=13, weight="bold", tcolor="#1d4ed8")
box(lx + 40, 120, 300, 50, "white", "#3b82f6", ["LLM.generate() / chat()"], size=14, weight="bold")
box(lx + 40, 190, 300, 58, "#dbeafe", "#2563eb",
    ["_run_engine：裸循环", "while has_unfinished_requests: step()"], size=12, weight="bold")
box(lx + 40, 268, 300, 50, "white", "#3b82f6",
    ["收集 finished，按 request_id 排序返回"], size=12)
arrow(lx + 190, 170, lx + 190, 190)
arrow(lx + 190, 248, lx + 190, 268)

# ── Right column: online AsyncLLM ──
rx = 580
rw = 380
box(rx, 70, rw, 32, "#f0fdf4", "#86efac", ["在线 · 跨进程 ZMQ · 流式 DELTA"], size=13, weight="bold", tcolor="#15803d")
box(rx + 40, 120, 300, 50, "white", "#16a34a",
    ["OpenAIServingChat → AsyncLLM.generate()"], size=12, weight="bold")
box(rx + 40, 190, 300, 58, "#dcfce7", "#16a34a",
    ["后台 output_handler + ZMQ", "拉一批 → process_outputs → 各请求队列"], size=12, weight="bold")
box(rx + 40, 268, 300, 50, "white", "#16a34a",
    ["async for → SSE / JSON 流式回客户端"], size=12)
arrow(rx + 190, 170, rx + 190, 190)
arrow(rx + 190, 248, rx + 190, 268)

# ── Converge to shared core ──
cy = 400
box(290, cy, 420, 56, "#fef3c7", "#d97706",
    ["同一个 EngineCore.step()", "schedule → execute → sample → update"], size=14, weight="bold", tcolor="#92400e")
arrow(lx + 190, 318, 400, cy, color="#64748b")
arrow(rx + 190, 318, 600, cy, color="#64748b")

box(250, cy + 110, 500, 78, "#fee2e2", "#dc2626",
    ["同一个 OutputProcessor.process_outputs()",
     "用 req_state.queue is None 分流：",
     "None → 返回 list（离线）  ·  非 None → put 进队列（在线）"], size=13, weight="bold", tcolor="#991b1b")
arrow(500, cy + 56, 500, cy + 110, color="#64748b")

# split arrows back out (labels)
text(500, 640, "一套调度/执行/采样/去 token 化，两种驱动方式", size=14, anchor="middle", weight="bold", fill="#334155")

L.append('</svg>')
with open(__file__.replace("gen_two_entrypoints.py", "02-two-entrypoints.svg"), "w") as f:
    f.write('\n'.join(L))
print("wrote 02-two-entrypoints.svg")
