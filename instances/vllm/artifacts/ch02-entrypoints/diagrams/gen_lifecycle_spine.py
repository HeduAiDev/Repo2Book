#!/usr/bin/env python3
"""一个请求的一生：端到端主线（三段式）横向时序泳道。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1180, 720
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '<marker id="ar" viewBox="0 0 10 6" refX="1" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M10,0 L0,3 L10,6 Z" fill="#475569"/></marker>'
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


# ── Title ──
text(W / 2, 34, "一个请求的一生：端到端主线（三段式异步）", size=20, anchor="middle", weight="bold")

# ── Three lane bands ──
lane_x = 20
lane_w = W - 40
lanes = [
    (62, 300, "#eff6ff", "#bfdbfe", "API 进程 · 单事件循环上的多协程"),
    (372, 110, "#fef9c3", "#fde68a", "进程边界 · ZMQ（msgpack 序列化）"),
    (492, 200, "#f0fdf4", "#bbf7d0", "引擎进程 · run_busy_loop 同步循环"),
]
for y, h, fill, stroke, label in lanes:
    L.append(
        f'<rect x="{lane_x}" y="{y}" width="{lane_w}" height="{h}" rx="10" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    )
    # ZMQ lane label moved down to avoid collision with process_outputs box text
    label_dy = 42 if "ZMQ" in label else 22
    text(lane_x + 14, y + label_dy, label, size=14, weight="bold", fill="#334155")

# ── API process lane content ──
# Row 1: add_request chain (top)
y_add = 110
box(40, y_add, 150, 56, "white", "#3b82f6",
    ["render", "PromptType→EngineInput"], size=12)
box(220, y_add, 168, 56, "#dbeafe", "#2563eb",
    ["InputProcessor (Stage 1)", "process_inputs()"], size=12, weight="bold")
box(418, y_add, 150, 56, "white", "#3b82f6",
    ["assign_request_id", "+8 位随机后缀"], size=12)
box(598, y_add, 150, 56, "#dbeafe", "#2563eb",
    ["add_request", "建 collector / 扇出"], size=12)

arrow(190, y_add + 28, 220, y_add + 28)
arrow(388, y_add + 28, 418, y_add + 28)
arrow(568, y_add + 28, 598, y_add + 28)
text(305, y_add - 8, "EngineCoreRequest 在这里诞生", size=11, anchor="middle", fill="#1d4ed8")

# Row 2: output_handler (producer)  +  generate (consumer)
y_oh = 250
box(40, y_oh, 200, 64, "#dbeafe", "#2563eb",
    ["output_handler（后台 task）", "= 三段式的搬运工 / 生产者"], size=12, weight="bold")
box(40, y_oh + 96, 200, 60, "#fee2e2", "#dc2626",
    ["process_outputs (Stage 3)", "去 token 化 + 停止 + logprobs"], size=11, weight="bold")
box(300, y_oh + 50, 168, 56, "#fef3c7", "#d97706",
    ["RequestOutputCollector", "每请求信箱 put/get"], size=11, weight="bold")
box(560, y_oh + 50, 188, 60, "#dbeafe", "#2563eb",
    ["generate()（消费者协程）", "get_nowait() or await get()"], size=11, weight="bold")
box(560, y_oh + 130, 188, 44, "#e0e7ff", "#4f46e5",
    ["yield RequestOutput → 用户"], size=12, weight="bold")

# output_handler -> process_outputs
arrow(140, y_oh + 64, 140, y_oh + 96)
# process_outputs -> collector (put)
arrow(240, y_oh + 126, 300, y_oh + 86, color="#d97706")
text(270, y_oh + 100, "put()", size=11, anchor="middle", fill="#b45309")
# collector -> generate (get)
arrow(468, y_oh + 78, 560, y_oh + 78)
text(514, y_oh + 70, "get()", size=11, anchor="middle", fill="#475569")
# generate -> yield
arrow(654, y_oh + 110, 654, y_oh + 130)

# add_request -> engine (down across ZMQ)
box(40, y_add + 70, 168, 30, "none", "none", [""], rx=0)
arrow(682, y_add + 56, 682, 372, color="#a16207")
text(700, y_add + 130, "add_request_async", size=11, fill="#a16207")
text(700, y_add + 148, "EngineCoreRequest →ZMQ", size=10, fill="#a16207")

# ── ZMQ lane content ──
box(300, 392, 200, 70, "#fef08a", "#ca8a04",
    ["AsyncMPClient", "add_request_async（入向）", "get_output_async（出向）"], size=11, weight="bold")

# ── Engine lane content ──
y_eng = 532
box(40, y_eng, 188, 70, "#dcfce7", "#16a34a",
    ["run_busy_loop", "取入队 → step → 出队", "（独立 OS 进程的心跳）"], size=11, weight="bold")
# step four sub-steps
steps = ["schedule", "execute_model", "sample_tokens", "update_from_output"]
sx = 260
sw = 178
for i, st in enumerate(steps):
    x = sx + i * (sw + 12)
    box(x, y_eng + 8, sw, 54, "white", "#15803d", [f"{i+1}. {st}"], size=13, weight="bold")
    if i < 3:
        arrow(x + sw, y_eng + 35, x + sw + 12, y_eng + 35, color="#15803d")
# bracket label for step
text(260 + 2 * (sw + 12), y_eng - 6, "EngineCore.step() —— 一拍：连续批里 prefill + decode 混跑", size=12, anchor="middle", weight="bold", fill="#166534")

# engine -> ZMQ (output up)
arrow(450, y_eng, 450, 462, color="#a16207")
text(468, y_eng - 30, "EngineCoreOutputs →ZMQ", size=10, fill="#a16207")

# ZMQ -> output_handler (up into API)
arrow(300, 414, 240, 290, color="#a16207")
text(250, 360, "get_output_async", size=11, anchor="end", fill="#a16207")
text(250, 376, "拉一批", size=10, anchor="end", fill="#a16207")

# ── per-station 放大角标 (right margin notes) ──
notes = [
    (y_add + 4, "Stage 1 → 第 5 章放大", "#2563eb"),
    (y_oh + 4, "搬运/解耦 → 第 4·8 章", "#2563eb"),
    (y_oh + 158, "去 token 化 → 第 9·10 章", "#dc2626"),
    (392, "IPC → 第 7 章", "#ca8a04"),
    (y_eng - 22, "四步 → 第 11·13·19·27 章", "#16a34a"),
]
for yy, s, c in notes:
    text(W - 28, yy, s, size=11, anchor="end", fill=c, weight="bold")

L.append('</svg>')
with open(__file__.replace("gen_lifecycle_spine.py", "01-lifecycle-spine.svg"), "w") as f:
    f.write('\n'.join(L))
print("wrote 01-lifecycle-spine.svg")
