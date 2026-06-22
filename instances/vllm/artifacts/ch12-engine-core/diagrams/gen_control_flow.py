#!/usr/bin/env python3
"""step_with_batch_queue 控制流：两态（填管道 / 取结果）+ deferred 支线。"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

w, h = 1000, 720
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>'
         '<marker id="ar" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="8" markerHeight="6" orient="auto">'
         '<path d="M0,0 L10,3.5 L0,7 Z" fill="#475569"/></marker>'
         '<marker id="arg" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="8" markerHeight="6" orient="auto">'
         '<path d="M0,0 L10,3.5 L0,7 Z" fill="#9333ea"/></marker>'
         '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

def text(x, y, s, size=13, anchor="start", weight="normal", fill="#1e293b"):
    L.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" '
             f'text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{esc(s)}</text>')

def box(x, y, bw, bh, fill, stroke, lines, lab_size=13, lab_fill="#1e293b", lab_weight="normal"):
    L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    n = len(lines)
    y0 = y + bh/2 - (n-1)*9 + 4
    for i, ln in enumerate(lines):
        text(x + bw/2, y0 + i*18, ln, size=lab_size, anchor="middle", fill=lab_fill, weight=lab_weight)

def arrow(x1, y1, x2, y2, color="#475569", marker="ar", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2"{d} marker-end="url(#{marker})"/>')

text(24, 34, "step_with_batch_queue() 一次调用的控制流", size=18, weight="bold")

cx = 300
# Entry
box(cx-110, 56, 220, 40, "#e2e8f0", "#64748b", ["进入：assert len(queue) < size"], lab_weight="bold")
arrow(cx, 96, cx, 122)

# decision: has_requests?
box(cx-110, 122, 220, 44, "#fef9c3", "#ca8a04", ["scheduler.has_requests() ?"], lab_weight="bold")

# ===== State A: 填管道 =====
text(60, 200, "态 A：填管道（优先）", size=15, weight="bold", fill="#1d4ed8")
ax = 90
box(ax, 212, 230, 40, "#dbeafe", "#3b82f6", ["schedule() + execute_model(non_block)"])
arrow(ax+115, 252, ax+115, 276)
box(ax, 276, 230, 56, "#dbeafe", "#3b82f6",
    ["model_executed = ", "total_num_scheduled_tokens > 0"])
arrow(ax+115, 332, ax+115, 356)
# pending? decision
box(ax, 356, 230, 44, "#fef9c3", "#ca8a04", ["pending_structured_output_tokens ?"])

# left: immediate sample
box(ax-60, 432, 150, 56, "#dbeafe", "#3b82f6",
    ["否 → get_grammar_bitmask", "+ sample_tokens(non_block)"], lab_size=12)
arrow(ax+30, 400, ax+25, 432)
# right: defer (支线)
box(ax+150, 432, 175, 56, "#f3e8ff", "#9333ea",
    ["是 → 存 deferred_scheduler_output", "（不入队，留到态 B）"], lab_size=12, lab_fill="#6b21a8")
arrow(ax+200, 400, ax+225, 432)

# appendleft
box(ax-10, 516, 200, 40, "#dbeafe", "#3b82f6", ["appendleft((future, so, exec_f))"], lab_weight="bold")
arrow(ax+65, 488, ax+88, 516)
arrow(cx-30, 166, ax+115, 212, color="#3b82f6")
text(ax+30, 192, "有请求", size=12, fill="#1d4ed8")

# fill-priority decision after appendleft
box(ax-30, 580, 250, 56, "#fef9c3", "#ca8a04",
    ["model_executed and len<size", "and not queue[-1][0].done() ?"], lab_size=12)
arrow(ax+90, 556, ax+95, 580)
# return None,True
box(ax-30, 656, 130, 40, "#dcfce7", "#16a34a", ["是 → return", "(None, True)"], lab_size=12, lab_fill="#15803d", lab_weight="bold")
arrow(ax+30, 636, ax+35, 656)
text(ax-30, 650, "只填不取，转下一圈", size=11, fill="#15803d")

# ===== State B: 取结果 =====
text(660, 200, "态 B：取结果（阻塞）", size=15, weight="bold", fill="#b45309")
bx = 640
# triggers in
text(560, 600, "落到态 B 的三个触发：", size=12, weight="bold", fill="#b45309")
text(560, 620, "① 队满  ② 无更多请求  ③ 队尾批已 done()", size=12, fill="#b45309")
# from fill-priority "否" arrow to B
arrow(ax+215, 608, bx+90, 312, color="#ca8a04", dash="5,4")
# from has_requests "否+空队" → return None,False handled separately; from "否" path goes to B
arrow(cx+30, 166, bx+90, 252, color="#ca8a04", dash="5,4")
text(cx+90, 200, "无请求但队非空", size=11, fill="#b45309")

box(bx, 252, 190, 40, "#fed7aa", "#ea580c", ["future,so,exec_f = queue.pop()"], lab_weight="bold")
arrow(bx+95, 292, bx+95, 312)
box(bx, 312, 190, 56, "#fed7aa", "#ea580c", ["model_output = future.result()", "（None → exec_f.result() 抛异常）"], lab_size=11)
arrow(bx+95, 368, bx+95, 388)
box(bx, 388, 190, 40, "#fed7aa", "#ea580c", ["update_from_output(...)"], lab_weight="bold")
arrow(bx+95, 428, bx+95, 448)
# deferred recovery
box(bx-30, 448, 250, 70, "#f3e8ff", "#9333ea",
    ["若有 deferred：(spec→take_draft)", "get_grammar_bitmask + sample_tokens", "→ appendleft 重新入队"],
    lab_size=11, lab_fill="#6b21a8")
arrow(bx+95, 518, bx+95, 538)
box(bx, 538, 190, 40, "#dcfce7", "#16a34a", ["return (outputs, model_executed)"], lab_weight="bold", lab_fill="#15803d")

# deferred 支线连线：态A defer → 态B deferred recovery
arrow(ax+325, 460, bx-30, 470, color="#9333ea", marker="arg", dash="6,4")
text((ax+325+bx-30)/2, 452, "deferred 支线", size=11, anchor="middle", fill="#9333ea", weight="bold")

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch12-engine-core/diagrams/control-flow.svg", "w").write('\n'.join(L))
print("wrote control-flow.svg")
