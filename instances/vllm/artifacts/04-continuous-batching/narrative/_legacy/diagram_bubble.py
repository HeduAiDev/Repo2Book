#!/usr/bin/env python3
"""Static Batching vs Continuous Batching timeline comparison diagram."""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(s)

w, h = 1000, 620
L = []
L.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {0} {1}">'.format(w, h))
L.append('<defs>')
L.append('<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto">')
L.append('<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/>')
L.append('</marker>')
L.append('</defs>')
L.append('<rect width="{0}" height="{1}" fill="#f8fafc"/>'.format(w, h))

# Title
L.append('<text x="500" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#1e293b">')
L.append(esc('Static Batching vs Continuous Batching 对比'))
L.append('</text>')

# Layout parameters
left_x = 60
right_x = 510
col_w = 400
row_top = 70

# Helper functions
def draw_panel_label(x, y, text):
    L.append('<text x="{}" y="{}" text-anchor="middle" font-size="16" font-weight="bold" fill="#334155">{}</text>'.format(x, y, esc(text)))

def draw_bubble(x, y, w, h, label=""):
    L.append('<rect x="{}" y="{}" width="{}" height="{}" rx="3" fill="none" stroke="#dc2626" stroke-width="2" stroke-dasharray="6,3"/>'.format(x, y, w, h))
    if label:
        L.append('<text x="{}" y="{}" text-anchor="middle" font-size="13" font-weight="bold" fill="#dc2626">{}</text>'.format(x + w // 2, y + h // 2 + 4, esc(label)))

# ═══ LEFT PANEL: Static Batching ═══
draw_panel_label(left_x + col_w // 2, row_top, 'Static Batching')

prefill_y = row_top + 30
L.append('<text x="{}" y="{}" font-size="13" font-weight="bold" fill="#1e40af">{}</text>'.format(left_x, prefill_y, esc('Phase 1: Prefill (所有请求一起处理)')))
prefill_y += 8

prefill_bar_y = prefill_y + 10
prefill_bar_h = 30

# Background bar for prefill
L.append('<rect x="{}" y="{}" width="{}" height="{}" rx="4" fill="#bfdbfe" stroke="#3b82f6" stroke-width="1"/>'.format(left_x, prefill_bar_y, col_w, prefill_bar_h))

# Individual request bars within prefill
num_requests = 8
req_w = col_w / num_requests
for i in range(num_requests):
    req_x = left_x + i * req_w
    if i >= 2:
        # short prompt - narrower
        short_w = req_w * 0.3
        L.append('<rect x="{}" y="{}" width="{}" height="{}" rx="2" fill="#3b82f6"/>'.format(req_x, prefill_bar_y, short_w, prefill_bar_h))
        L.append('<text x="{}" y="{}" font-size="7" fill="white" text-anchor="middle">{}</text>'.format(req_x + short_w // 2, prefill_bar_y + prefill_bar_h // 2 + 3, esc('短')))
        # Bubble area
        L.append('<rect x="{}" y="{}" width="{}" height="{}" rx="2" fill="#fca5a5" opacity="0.6"/>'.format(req_x + short_w, prefill_bar_y, req_w - short_w, prefill_bar_h))
    else:
        # long prompt
        L.append('<rect x="{}" y="{}" width="{}" height="{}" rx="2" fill="#2563eb"/>'.format(req_x, prefill_bar_y, req_w, prefill_bar_h))
        L.append('<text x="{}" y="{}" font-size="7" fill="white" text-anchor="middle">{}</text>'.format(req_x + req_w // 2, prefill_bar_y + prefill_bar_h // 2 + 3, esc('长')))

prefill_end_y = prefill_bar_y + prefill_bar_h

# Bubble annotation for prefill
draw_bubble(left_x + col_w * 0.3, prefill_bar_y, col_w * 0.7, prefill_bar_h, esc('BUBBLE: 短请求等待长请求'))

# Decode phase label
decode_y = prefill_end_y + 35
L.append('<text x="{}" y="{}" font-size="13" font-weight="bold" fill="#15803d">{}</text>'.format(left_x, decode_y, esc('Phase 2: Decode (每步1 token/请求)')))
decode_y += 8

# Decode timeline
num_decode_steps = 10
decode_bar_y = decode_y + 10
decode_bar_h = 8
decode_step_gap = 3

# GPU capacity indicator
gpu_cap_x = left_x + col_w + 10
gpu_cap_total_h = num_decode_steps * (decode_bar_h + decode_step_gap)
L.append('<rect x="{}" y="{}" width="80" height="{}" rx="3" fill="#e2e8f0" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4,2"/>'.format(gpu_cap_x, decode_bar_y, gpu_cap_total_h))
L.append('<text x="{}" y="{}" font-size="9" fill="#64748b">{}</text>'.format(gpu_cap_x + 2, decode_bar_y + 12, esc('GPU capacity')))
L.append('<text x="{}" y="{}" font-size="9" fill="#64748b">{}</text>'.format(gpu_cap_x + 2, decode_bar_y + 24, esc('2048 tok/step')))
L.append('<text x="{}" y="{}" font-size="9" fill="#dc2626">{}</text>'.format(gpu_cap_x + 2, decode_bar_y + 50, esc('实际只用 8 tok/step')))

# Decode steps
for step in range(num_decode_steps):
    step_w = col_w
    step_h = decode_bar_h
    step_y = decode_bar_y + step * (step_h + decode_step_gap)
    # Background (idle / bubble)
    L.append('<rect x="{}" y="{}" width="{}" height="{}" rx="1" fill="#fee2e2"/>'.format(left_x, step_y, step_w, step_h))
    # Actual decode tokens (tiny - about 0.4% of capacity)
    actual_w = 8  # fixed small width for visual
    L.append('<rect x="{}" y="{}" width="{}" height="{}" rx="1" fill="#22c55e"/>'.format(left_x, step_y, actual_w, step_h))
    # Step label
    L.append('<text x="{}" y="{}" font-size="6" fill="#991b1b">{}</text>'.format(left_x + actual_w + 3, step_y + step_h // 2 + 2, esc('IDLE')))

# Bubble annotation for decode
decode_bubble_y = decode_bar_y + num_decode_steps * (decode_bar_h + decode_step_gap) + 5
draw_bubble(left_x, decode_bar_y, col_w, num_decode_steps * (decode_bar_h + decode_step_gap), esc('BUBBLE: GPU 利用率 < 1%'))

# Static total steps
total_static_y = decode_bubble_y + 35
L.append('<text x="{}" y="{}" font-size="14" font-weight="bold" fill="#dc2626">{}</text>'.format(left_x, total_static_y, esc('总步数: ~2304 steps  |  GPU利用率: ~36%')))

# ═══ RIGHT PANEL: Continuous Batching ═══
draw_panel_label(right_x + col_w // 2, row_top, 'Continuous Batching')

cb_start_y = row_top + 30
L.append('<text x="{}" y="{}" font-size="13" font-weight="bold" fill="#1e40af">{}</text>'.format(right_x, cb_start_y, esc('Interleaved: Prefill Chunks + Decode 混合')))
cb_start_y += 8

# Token budget visualization
budget_y = cb_start_y + 10
budget_h = 24
L.append('<rect x="{}" y="{}" width="{}" height="{}" rx="4" fill="#f1f5f9" stroke="#64748b" stroke-width="2"/>'.format(right_x, budget_y, col_w, budget_h))
L.append('<text x="{}" y="{}" font-size="11" fill="#64748b">{}</text>'.format(right_x + 5, budget_y + 16, esc('Token Budget: 2048 tokens/step')))

# Steps showing interleaved prefill + decode
steps_data = [
    (0.55, 0.45, 'Step 1: 短prefill(8x128) + 长prefill(1024)'),
    (0.05, 0.95, 'Step 2: 8xdecode(8) + 长prefill(2040)'),
    (0.05, 0.95, 'Step 3: 8xdecode(8) + 长prefill(2040)'),
    (0.55, 0.35, 'Step 4: 短prefill+decode + 长prefill'),
    (0.05, 0.60, 'Step 5: decode + 新请求prefill'),
    (0.55, 0.30, 'Step 6: 混合prefill+decode'),
    (0.05, 0.80, 'Step 7: 多请求decode'),
    (0.30, 0.70, 'Step 8: prefill+decode'),
]
step_bar_y = budget_y + budget_h + 15
step_bar_h = 18
step_gap = 4

for idx, (decode_frac, prefill_frac, label) in enumerate(steps_data):
    sy = step_bar_y + idx * (step_bar_h + step_gap)
    # Background
    L.append('<rect x="{}" y="{}" width="{}" height="{}" rx="2" fill="#e2e8f0"/>'.format(right_x, sy, col_w, step_bar_h))
    # Decode portion (green)
    decode_w = int(col_w * decode_frac)
    if decode_w > 0:
        L.append('<rect x="{}" y="{}" width="{}" height="{}" rx="2" fill="#22c55e"/>'.format(right_x, sy, decode_w, step_bar_h))
    # Prefill portion (blue) — after decode
    prefill_x = right_x + decode_w
    prefill_w = int(col_w * prefill_frac)
    if prefill_w > 0:
        if decode_w + prefill_w > col_w:
            prefill_w = col_w - decode_w
        L.append('<rect x="{}" y="{}" width="{}" height="{}" rx="2" fill="#3b82f6"/>'.format(prefill_x, sy, prefill_w, step_bar_h))
    # Rest is unused
    used_w = decode_w + prefill_w
    if used_w < col_w:
        L.append('<rect x="{}" y="{}" width="{}" height="{}" rx="2" fill="#fef3c7"/>'.format(right_x + used_w, sy, col_w - used_w, step_bar_h))
        L.append('<text x="{}" y="{}" font-size="6" fill="#d97706">{}</text>'.format(right_x + used_w + 3, sy + step_bar_h // 2 + 2, esc('空闲')))
    # Step label
    L.append('<text x="{}" y="{}" font-size="8" fill="#1e293b">{}</text>'.format(right_x + 3, sy + step_bar_h // 2 + 3, esc(label)))

# Green "FULL UTILIZATION" banner
full_util_y = step_bar_y + len(steps_data) * (step_bar_h + step_gap) + 10
L.append('<rect x="{}" y="{}" width="{}" height="28" rx="6" fill="#16a34a"/>'.format(right_x, full_util_y, col_w, 28))
L.append('<text x="{}" y="{}" text-anchor="middle" font-size="16" font-weight="bold" fill="white">{}</text>'.format(right_x + col_w // 2, full_util_y + 20, esc('GPU 几乎满负荷运作 (FULL UTILIZATION)')))

# CB total steps
total_cb_y = full_util_y + 40
L.append('<text x="{}" y="{}" font-size="14" font-weight="bold" fill="#16a34a">{}</text>'.format(right_x, total_cb_y, esc('总步数: ~400-800 steps  |  GPU利用率: ~95%+')))

# Arrow from left to right
arrow_y = total_cb_y + 30
L.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>'.format(left_x + col_w + 15, arrow_y, right_x - 10, arrow_y))
L.append('<text x="{}" y="{}" text-anchor="middle" font-size="13" font-weight="bold" fill="#475569">{}</text>'.format((left_x + col_w + 15 + right_x - 10) // 2, arrow_y - 8, esc('Continuous Batching 消除 Bubble')))

# Legend
legend_y = h - 60
L.append('<text x="50" y="{}" font-size="13" font-weight="bold" fill="#1e293b">{}</text>'.format(legend_y, esc('图例:')))
legend_items = [
    ('#2563eb', 'Prefill (prompt 处理)'),
    ('#22c55e', 'Decode (逐 token 生成)'),
    ('#fca5a5', 'Bubble (GPU 空闲)'),
    ('#fef3c7', '少量空闲 budget'),
]
lx = 100
for color, label_text in legend_items:
    L.append('<rect x="{}" y="{}" width="16" height="16" rx="2" fill="{}"/>'.format(lx, legend_y - 12, color))
    L.append('<text x="{}" y="{}" font-size="12" fill="#334155">{}</text>'.format(lx + 22, legend_y, esc(label_text)))
    lx += 180

# Footer
L.append('<text x="500" y="{}" text-anchor="middle" font-size="11" fill="#94a3b8">{}</text>'.format(h - 15, esc('vLLM-from-Scratch 第4章 Continuous Batching --- Bubble 对比')))

L.append('</svg>')
svg = '\n'.join(L)
print(svg)
