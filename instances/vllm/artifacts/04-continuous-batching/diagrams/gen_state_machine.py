"""
Generate request state machine diagram for Continuous Batching scheduler.
Shows the lifecycle: WAITING -> RUNNING -> PREEMPTED -> WAITING (loop) or FINISHED_*.
"""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(str(s))

W, H = 900, 520
L = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
    '<defs>',
    '  <marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">',
    '    <path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/>',
    '  </marker>',
    '  <marker id="ar_red" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">',
    '    <path d="M0,0 L10,3 L0,6 Z" fill="#f59e0b"/>',
    '  </marker>',
    '  <marker id="ar_green" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">',
    '    <path d="M0,0 L10,3 L0,6 Z" fill="#22c55e"/>',
    '  </marker>',
    '</defs>',
    f'<rect width="{W}" height="{H}" fill="#fafbfc"/>',
]

TITLE_FILL = "#1e293b"
LABEL_FILL = "#334155"
SUBTITLE = "#64748b"
WHITE = "#ffffff"

def add_text(x, y, text, fill=LABEL_FILL, size=13, anchor="start", bold=False):
    fw = 'font-weight="bold"' if bold else ''
    L.append(f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-family="sans-serif" text-anchor="{anchor}" {fw}>{esc(text)}</text>')

# Box drawing helper
def state_box(x, y, w, h, fill, label, subtitle=None):
    r = 12
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" rx="{r}" stroke="#cbd5e1" stroke-width="1.5"/>')
    add_text(x + w/2, y + h/2 - 2, label, WHITE if fill != "#f8fafc" else LABEL_FILL, 15, "middle", True)
    if subtitle:
        add_text(x + w/2, y + h/2 + 18, subtitle, WHITE if fill != "#f8fafc" else SUBTITLE, 11, "middle")

# Arrow drawing
def arrow(x1, y1, x2, y2, marker="ar", curved=False):
    if curved:
        mx = (x1 + x2) / 2
        L.append(f'<path d="M{x1},{y1} Q{mx},{y1-20} {x2},{y2}" fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#{marker})"/>')
    else:
        L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#64748b" stroke-width="2" marker-end="url(#{marker})"/>')

# ═══════════════════════════ Layout ═══════════════════════════
add_text(W/2, 38, "Request State Machine", TITLE_FILL, 20, "middle", True)
add_text(W/2, 62, "each request traverses this lifecycle inside the scheduler", SUBTITLE, 13, "middle")

# State boxes
box_w, box_h = 180, 56

# WAITING — top center
wx, wy = W/2 - box_w/2, 100
state_box(wx, wy, box_w, box_h, "#f8fafc", "WAITING", "waiting in FCFS queue")

# RUNNING — center
rx, ry = W/2 - box_w/2, 230
state_box(rx, ry, box_w, box_h, "#3b82f6", "RUNNING", "active, holding KV cache blocks")

# PREEMPTED — left middle
px, py = 100, 230
state_box(px, py, box_w, box_h, "#fef3c7", "PREEMPTED", "KV cache freed, requeued")

# FINISHED_STOPPED — right bottom
fsx, fsy = W - 260, 390
state_box(fsx, fsy, 150, 56, "#dcfce7", "FINISHED", "EOS / max_tokens")

# WAITING -> RUNNING (admitted)
arrow(wx + box_w/2, wy + box_h, rx + box_w/2, ry, "ar_green")
add_text(wx + box_w/2 + 15, (wy + box_h + ry)/2, "admitted:\nKV alloc OK\nbudget OK", "#22c55e", 11, "start")

# RUNNING -> PREEMPTED (OOM)
arrow(rx, ry + box_h/2, px + box_w, py + box_h/2, "ar_red")
add_text((rx + px + box_w)/2, py - 15, "KV cache OOM\npreempted", "#f59e0b", 11, "middle")

# PREEMPTED -> WAITING (cycle)
arrow(px + box_w/2, py, wx - 60, wy + box_h, curved=True)
add_text(px + box_w/2 - 40, 120, "requeue\n(insert at head)", SUBTITLE, 10, "middle")

# RUNNING -> FINISHED
arrow(rx + box_w, ry + box_h/2, fsx, fsy + box_h/2, "ar_green")
add_text((rx + box_w + fsx)/2 + 15, ry + box_h/2 - 10, "EOS / max_tokens\n→ free KV cache", "#22c55e", 11, "start")

# Phase annotations
add_text(450, 185, "Phase 2 entry", "#22c55e", 12, "start")
add_text(450, 200, "(waiting → running)", SUBTITLE, 10, "start")

add_text(315, 280, "Phase 1 OOM → preempt", "#f59e0b", 12, "start")
add_text(315, 295, "(pop lowest priority)", SUBTITLE, 10, "start")

# Legend at bottom
legend_y = 490
add_text(30, legend_y, "Phase 1 (schedule): running requests first — preempt on OOM", LABEL_FILL, 12)
add_text(30, legend_y + 20, "Phase 2 (schedule): admit waiting → running — stop on OOM", LABEL_FILL, 12)

# Add request tracking note
add_text(W/2, legend_y, "\"no prefill phase, no decode phase — just num_computed_tokens\"", SUBTITLE, 12, "middle", True)

L.append('</svg>')

svg_content = '\n'.join(L)
with open('/mnt/e/Laboratory/vllm-from-scratch/instances/vllm/artifacts/04-continuous-batching/diagrams/state_machine.svg', 'w') as f:
    f.write(svg_content)
print("State machine SVG written.")
