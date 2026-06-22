#!/usr/bin/env python3
"""deque 三元组结构小图：appendleft 进左端、pop 出右端，右端=最旧。"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

w, h = 1020, 360
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>'
         '<marker id="ar" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="9" markerHeight="7" orient="auto">'
         '<path d="M0,0 L10,3.5 L0,7 Z" fill="#475569"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

def text(x, y, s, size=13, anchor="start", weight="normal", fill="#1e293b"):
    L.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" '
             f'text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{esc(s)}</text>')

text(24, 36, "batch_queue：deque[ (sample future, SchedulerOutput, exec future) ]，maxlen = batch_queue_size", size=15, weight="bold")

# three slots
slot_w, slot_h = 220, 150
y0 = 90
gap = 24
x0 = 140
labels = ["最新（左端）", "中间", "最旧（右端）"]
colors = [("#dbeafe", "#3b82f6"), ("#fef3c7", "#f59e0b"), ("#d1fae5", "#10b981")]
for i in range(3):
    sx = x0 + i * (slot_w + gap)
    fill, stroke = colors[i]
    L.append(f'<rect x="{sx}" y="{y0}" width="{slot_w}" height="{slot_h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    text(sx + slot_w/2, y0 - 10, labels[i], size=12, anchor="middle", fill=stroke, weight="bold")
    # three fields
    fields = ["future  (sample 结果)", "SchedulerOutput", "exec_future  (execute_model)"]
    for j, fld in enumerate(fields):
        fy = y0 + 24 + j * 40
        L.append(f'<rect x="{sx+14}" y="{fy}" width="{slot_w-28}" height="30" rx="4" fill="white" stroke="{stroke}" stroke-width="1"/>')
        text(sx + slot_w/2, fy + 20, fld, size=12, anchor="middle", fill="#334155")

# appendleft arrow into left end
lx = x0
text(lx - 130, y0 + slot_h/2 - 8, "appendleft", size=14, anchor="start", weight="bold", fill="#1d4ed8")
text(lx - 130, y0 + slot_h/2 + 12, "新批进左端", size=11, anchor="start", fill="#1d4ed8")
L.append(f'<line x1="{lx-40}" y1="{y0+slot_h/2}" x2="{lx-6}" y2="{y0+slot_h/2}" stroke="#1d4ed8" stroke-width="2.5" marker-end="url(#ar)"/>')

# pop arrow out of right end
rx = x0 + 3*(slot_w+gap) - gap
text(rx + 24, y0 + slot_h/2 - 8, "pop", size=14, anchor="start", weight="bold", fill="#047857")
text(rx + 24, y0 + slot_h/2 + 12, "最旧批出右端", size=11, anchor="start", fill="#047857")
L.append(f'<line x1="{rx+6}" y1="{y0+slot_h/2}" x2="{rx+20}" y2="{y0+slot_h/2}" stroke="#047857" stroke-width="2.5" marker-end="url(#ar)"/>')

# fill-priority note on rightmost
note_y = y0 + slot_h + 36
text(rx, note_y, "填管道判定看它：queue[-1][0].done() —— 最旧批是否算完", size=13, anchor="end", fill="#047857", weight="bold")

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch12-engine-core/diagrams/deque-struct.svg", "w").write('\n'.join(L))
print("wrote deque-struct.svg")
