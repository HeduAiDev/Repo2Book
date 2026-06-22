#!/usr/bin/env python3
"""Pipeline timeline comparison: depth-1 step (bubbles) vs depth-N batch_queue (filled)."""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

w, h = 960, 540
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# colors per batch
BC = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
BC_L = ["#dbeafe", "#d1fae5", "#fef3c7", "#fee2e2"]
stages = ["stage 0", "stage 1", "stage 2"]

def text(x, y, s, size=14, anchor="start", weight="normal", fill="#1e293b"):
    L.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" '
             f'text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{esc(s)}</text>')

def cell(x, y, cw, ch, fill, stroke, label="", lab_fill="#1e293b"):
    L.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="4" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    if label:
        text(x + cw/2, y + ch/2 + 5, label, size=13, anchor="middle", fill=lab_fill, weight="bold")

# Layout params
left = 130          # left margin for stage labels
col_w = 78         # width of one time slot
row_h = 46
gap = 10

# ---- Title top ----
text(20, 32, "PP=3：流水线时间轴对比（横轴=时间槽，纵轴=pipeline stage）", size=17, weight="bold")

# ===== Upper: depth-1 (no batch queue) =====
top_y = 70
text(20, top_y, "深度 1（普通 step）：单批串行流过 3 个 stage —— 任一时刻只有 1/3 硬件在干活", size=14, weight="bold", fill="#b91c1c")
grid_y = top_y + 20
n_slots = 9
# stage rows
for r, st in enumerate(stages):
    ry = grid_y + r * (row_h + gap)
    text(left - 12, ry + row_h/2 + 5, st, size=13, anchor="end", fill="#475569")
# Each batch b occupies stage r at time slot t = b + r (diagonal). 3 batches.
for t in range(n_slots):
    tx = left + t * col_w
    for r in range(3):
        ry = grid_y + r * (row_h + gap)
        # which batch is in stage r at slot t? batch b such that b + r == t for the serial schedule
        # serial: batch 0 occupies slot0@s0, slot1@s1, slot2@s2 ; batch1 only starts after batch0 fully done (slot3)
        b = None
        if t // 3 < 3:
            base = (t // 3) * 3
            if t - base == r:
                b = t // 3
        if b is not None and b < 3:
            cell(tx + 4, ry, col_w - 8, row_h, BC_L[b], BC[b], f"B{b}", BC[b])
        else:
            L.append(f'<rect x="{tx+4}" y="{ry}" width="{col_w-8}" height="{row_h}" rx="4" '
                     f'fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3,3"/>')
# bubble annotation
text(left + 1.5*col_w, grid_y + 3*(row_h+gap) + 16, "空白格 = 气泡（stage 闲置）", size=12, fill="#b91c1c")

# ===== Lower: depth-N batch queue =====
bot_y = grid_y + 3*(row_h+gap) + 56
text(20, bot_y, "深度 3（batch_queue）：3 个批错位填进各 stage —— 稳态每个 stage 都满载，无气泡", size=14, weight="bold", fill="#047857")
grid_y2 = bot_y + 20
for r, st in enumerate(stages):
    ry = grid_y2 + r * (row_h + gap)
    text(left - 12, ry + row_h/2 + 5, st, size=13, anchor="end", fill="#475569")
# pipelined: batch b enters stage 0 at slot b, stage r at slot b+r
for t in range(n_slots):
    tx = left + t * col_w
    for r in range(3):
        ry = grid_y2 + r * (row_h + gap)
        b = t - r   # batch index occupying stage r at slot t
        if 0 <= b < 6:
            cb = b % 4
            cell(tx + 4, ry, col_w - 8, row_h, BC_L[cb], BC[cb], f"B{b}", BC[cb])
        else:
            L.append(f'<rect x="{tx+4}" y="{ry}" width="{col_w-8}" height="{row_h}" rx="4" '
                     f'fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3,3"/>')
# steady-state bracket
sx0 = left + 2*col_w
sx1 = left + 6*col_w
by = grid_y2 + 3*(row_h+gap) + 8
L.append(f'<line x1="{sx0}" y1="{by}" x2="{sx1}" y2="{by}" stroke="#047857" stroke-width="2"/>')
text((sx0+sx1)/2, by + 18, "稳态：3 个 stage 同时满载（填管道优先把队列填到 batch_queue_size=3）", size=12, anchor="middle", fill="#047857")

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch12-engine-core/diagrams/pipeline-timeline.svg", "w").write('\n'.join(L))
print("wrote pipeline-timeline.svg")
