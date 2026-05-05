"""
Generate bubble comparison diagram: Static Batching vs Continuous Batching.
Two-panel timeline showing GPU utilization per request.
"""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(str(s))

W, H = 1200, 520
L = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
    '<defs>',
    '  <marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto">',
    '    <path d="M0,0 L10,3 L0,6 Z" fill="#ef4444"/>',
    '  </marker>',
    '  <pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">',
    '    <line x1="0" y1="0" x2="0" y2="8" stroke="#fca5a5" stroke-width="3"/>',
    '  </pattern>',
    '</defs>',
    f'<rect width="{W}" height="{H}" fill="#fafbfc"/>',
]

# ── fonts / styles ──
TITLE_FILL = "#1e293b"
SUBTITLE_FILL = "#475569"
LABEL_FILL = "#334155"
BLUE = "#3b82f6"
GREEN = "#22c55e"
RED = "#ef4444"
LIGHT_RED = "#fecaca"
GRAY = "#e2e8f0"
DARK_GRAY = "#94a3b8"
WHITE = "#ffffff"
BORDER = "#cbd5e1"

def add_text(x, y, text, fill=LABEL_FILL, size=13, anchor="start", bold=False):
    fw = 'font-weight="bold"' if bold else ''
    L.append(f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-family="sans-serif" text-anchor="{anchor}" {fw}>{esc(text)}</text>')

def add_rect(x, y, w, h, fill, rx=0):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" rx="{rx}" stroke="{BORDER}" stroke-width="0.5"/>')

# ═══════════════════════════════════════════════════════════════
# Layout constants
# ═══════════════════════════════════════════════════════════════
LEFT_MARGIN = 60
RIGHT_EDGE = 540
PANEL2_START = 640
PANEL2_EDGE = 1140
BAR_W = RIGHT_EDGE - LEFT_MARGIN  # 480
PANEL2_BAR_W = PANEL2_EDGE - PANEL2_START  # 500

ROW1_Y = 110
ROW2_Y = 180
ROW3_Y = 250
ROW_H = 40
ROW_GAP = 70

# ═══════════════════════════════════════════════════════════════
# LEFT PANEL: Static Batching
# ═══════════════════════════════════════════════════════════════
L.append(f'<rect x="20" y="15" width="560" height="{H-30}" fill="none" stroke="{BORDER}" stroke-width="1.5" rx="6"/>')
add_text(305, 45, "Static Batching", TITLE_FILL, 18, "middle", True)
add_text(305, 65, "prefill/dedode are separate phases — short requests wait for the longest", SUBTITLE_FILL, 12, "middle")

# For static batching: timeline 0-900
# scale: BAR_W / 900 ≈ 0.533 px per token
SCALE = BAR_W / 900.0

# Time axis
add_text(LEFT_MARGIN, 95, "0", DARK_GRAY, 11, "middle")
add_text(RIGHT_EDGE, 95, "900", DARK_GRAY, 11, "middle")
add_text(LEFT_MARGIN + BAR_W/2, 95, "Time (steps)", DARK_GRAY, 11, "middle")

# Grid lines at phase boundary (prefill→decode at step 800)
px_800 = LEFT_MARGIN + 800 * SCALE
L.append(f'<line x1="{px_800}" y1="105" x2="{px_800}" y2="310" stroke="{DARK_GRAY}" stroke-width="1" stroke-dasharray="6,4"/>')
add_text(px_800, 325, "phase", DARK_GRAY, 10, "middle")
add_text(px_800, 338, "boundary", DARK_GRAY, 10, "middle")

# Phase labels
prefill_mid = LEFT_MARGIN + 400 * SCALE
decode_mid = LEFT_MARGIN + 850 * SCALE
add_text(prefill_mid, 320, "PREFILL", "#3b82f680", 11, "middle")
add_text(decode_mid, 320, "DECODE", "#22c55e80", 11, "middle")

# r1: prefill 0→800, decode 800→820, idle 820→900
def x_pos(step):
    return LEFT_MARGIN + step * SCALE

# r1 row
add_text(LEFT_MARGIN - 10, ROW1_Y + ROW_H/2 + 5, "r1", LABEL_FILL, 12, "end", True)
add_rect(x_pos(0), ROW1_Y, 800*SCALE, ROW_H, BLUE)          # prefill 800
add_rect(x_pos(800), ROW1_Y, 20*SCALE, ROW_H, GREEN)         # decode 20
add_rect(x_pos(820), ROW1_Y, 80*SCALE, ROW_H, GRAY)          # idle
add_text(x_pos(400), ROW1_Y + ROW_H/2 + 5, "prefill (800)", WHITE, 11, "middle")
add_text(x_pos(810), ROW1_Y + ROW_H/2 + 5, "20", WHITE, 9, "middle")

# r2 row
add_text(LEFT_MARGIN - 10, ROW2_Y + ROW_H/2 + 5, "r2", LABEL_FILL, 12, "end", True)
add_rect(x_pos(0), ROW2_Y, 200*SCALE, ROW_H, BLUE)          # prefill 200
add_rect(x_pos(200), ROW2_Y, 600*SCALE, ROW_H, LIGHT_RED)    # BUBBLE 600
add_rect(x_pos(200), ROW2_Y, 600*SCALE, ROW_H, "url(#hatch)", 0)
add_rect(x_pos(800), ROW2_Y, 50*SCALE, ROW_H, GREEN)         # decode 50
add_rect(x_pos(850), ROW2_Y, 50*SCALE, ROW_H, GRAY)          # idle
add_text(x_pos(100), ROW2_Y + ROW_H/2 + 5, "prefill (200)", WHITE, 10, "middle")

# r3 row
add_text(LEFT_MARGIN - 10, ROW3_Y + ROW_H/2 + 5, "r3", LABEL_FILL, 12, "end", True)
add_rect(x_pos(0), ROW3_Y, 50*SCALE, ROW_H, BLUE)           # prefill 50
add_rect(x_pos(50), ROW3_Y, 750*SCALE, ROW_H, LIGHT_RED)     # BUBBLE 750
add_rect(x_pos(50), ROW3_Y, 750*SCALE, ROW_H, "url(#hatch)", 0)
add_rect(x_pos(800), ROW3_Y, 100*SCALE, ROW_H, GREEN)        # decode 100
add_text(x_pos(25), ROW3_Y + ROW_H/2 + 5, "50", WHITE, 9, "middle")

# BUBBLE labels with arrows
bubble_r2_x = x_pos(500)
add_text(bubble_r2_x, ROW2_Y - 8, "BUBBLE", RED, 12, "middle", True)
L.append(f'<line x1="{bubble_r2_x}" y1="{ROW2_Y-2}" x2="{bubble_r2_x}" y2="{ROW2_Y+5}" stroke="{RED}" stroke-width="2" marker-end="url(#ar)"/>')

bubble_r3_x = x_pos(425)
add_text(bubble_r3_x, ROW3_Y - 8, "BUBBLE", RED, 12, "middle", True)
L.append(f'<line x1="{bubble_r3_x}" y1="{ROW3_Y-2}" x2="{bubble_r3_x}" y2="{ROW3_Y+5}" stroke="{RED}" stroke-width="2" marker-end="url(#ar)"/>')

# GPU utilization summary for static
util_y = 370
add_text(LEFT_MARGIN, util_y, "GPU utilization:", LABEL_FILL, 12, "start", True)
util_bar_x = LEFT_MARGIN + 130
util_bar_w = BAR_W - 130
# actual util ≈ 45.2% for 3 requests = (1220 tokens) / (900*3 slots)
add_rect(util_bar_x, util_y - 3, util_bar_w, 22, GRAY, 3)  # background
work_w = util_bar_w * 0.452
add_rect(util_bar_x, util_y - 3, work_w, 22, GREEN, 3)      # useful work
add_text(util_bar_x + work_w/2, util_y + 12, "45%", WHITE, 10, "middle", True)
add_text(util_bar_x + util_bar_w + 10, util_y + 12, "", LABEL_FILL, 10, "start")

# ═══════════════════════════════════════════════════════════════
# RIGHT PANEL: Continuous Batching
# ═══════════════════════════════════════════════════════════════
L.append(f'<rect x="{PANEL2_START - 20}" y="15" width="540" height="{H - 30}" fill="none" stroke="{BORDER}" stroke-width="1.5" rx="6"/>')
add_text(PANEL2_START + 250, 45, "Continuous Batching", TITLE_FILL, 18, "middle", True)
add_text(PANEL2_START + 250, 65, "prefill & decode mixed in every step — budget fills dynamically", SUBTITLE_FILL, 12, "middle")

# For continuous: we show a stylized timeline, not to exact scale
# 5 steps shown: Step 1, Step 2, Step 3, Step 4-103 (compressed)
# Each step is a narrow column showing which requests are active

# We'll show 5 step columns + a continuation
CB_BAR_W = PANEL2_BAR_W
cb_left = PANEL2_START
STEP_PAD = 8
# First 4 steps get proportional space, then a continuation region

# Add time axis
add_text(cb_left, 95, "Step 1", DARK_GRAY, 11, "middle")
add_text(cb_left + CB_BAR_W, 95, "Step 103", DARK_GRAY, 11, "middle")

# Budget labels inside steps
budget_y = ROW3_Y + ROW_H + 25

# Step 1: r1 only (512 tokens, prefill)
s1_x = cb_left
s1_w = CB_BAR_W * 0.15  # ~15% of total
step_rects = [(s1_x, s1_w, [(ROW1_Y, BLUE, "prefill\nchunk 512")])]

# Step 2: r1 (288 remaining prefill) + r2 (200 prefill) + r3 (24 prefill)
s2_x = s1_x + s1_w + STEP_PAD
s2_w = CB_BAR_W * 0.15
step_rects.append((s2_x, s2_w, [(ROW1_Y, "#60a5fa", "288"),
                                 (ROW2_Y, BLUE, "200"),
                                 (ROW3_Y, BLUE, "24")]))

# Step 3: all decode + r3 finishes prefill
s3_x = s2_x + s2_w + STEP_PAD
s3_w = CB_BAR_W * 0.15
step_rects.append((s3_x, s3_w, [(ROW1_Y, GREEN, "1"),
                                 (ROW2_Y, GREEN, "1"),
                                 (ROW3_Y, "#60a5fa", "26")]))

# Steps 4-103: all decode
s4_x = s3_x + s3_w + STEP_PAD
s4_w = CB_BAR_W * 0.55  # remaining ~55%
step_rects.append((s4_x, s4_w, [(ROW1_Y, GREEN, "decode ×20"),
                                 (ROW2_Y, GREEN, "decode ×50"),
                                 (ROW3_Y, GREEN, "decode ×100")]))

# Draw step backgrounds and bars
for sx, sw, rows in step_rects:
    # Step background
    L.append(f'<rect x="{sx}" y="105" width="{sw}" height="160" fill="#f1f5f9" stroke="{BORDER}" stroke-width="1" rx="2"/>')
    for ry, color, label in rows:
        ry_center = ry + (ROW_H - 30) / 2 if len(rows) > 1 else ry
        actual_ry = ry + 5 if len(rows) == 1 else ry
        actual_rh = ROW_H - 10 if len(rows) == 1 else 30
        L.append(f'<rect x="{sx+3}" y="{actual_ry}" width="{sw-6}" height="{actual_rh}" fill="{color}" rx="3"/>')
        if sw > 50:
            add_text(sx + sw/2, actual_ry + actual_rh/2 + 5, label, WHITE, 9, "middle")

# Row labels for right panel
add_text(cb_left - 10, ROW1_Y + ROW_H/2 + 5, "r1", LABEL_FILL, 12, "end", True)
add_text(cb_left - 10, ROW2_Y + ROW_H/2 + 5, "r2", LABEL_FILL, 12, "end", True)
add_text(cb_left - 10, ROW3_Y + ROW_H/2 + 5, "r3", LABEL_FILL, 12, "end", True)

# Budget summary for each step
step_data = [
    (s1_x, s1_w, "512/512"),
    (s2_x, s2_w, "512/512"),
    (s3_x, s3_w, "28/512"),
    (s4_x, s4_w, "3/512 avg"),
]
for sx, sw, label in step_data:
    add_text(sx + sw/2, budget_y, label, DARK_GRAY, 10, "middle")

add_text(cb_left, budget_y, "Budget:", LABEL_FILL, 10, "end")

# GPU utilization for CB
cb_util_y = 370
add_text(cb_left, cb_util_y, "GPU utilization:", LABEL_FILL, 12, "start", True)
cb_util_bar_x = cb_left + 130
cb_util_bar_w = CB_BAR_W - 130
add_rect(cb_util_bar_x, cb_util_y - 3, cb_util_bar_w, 22, GRAY, 3)
cb_work_w = cb_util_bar_w * 0.93
add_rect(cb_util_bar_x, cb_util_y - 3, cb_work_w, 22, GREEN, 3)
add_text(cb_util_bar_x + cb_work_w/2, cb_util_y + 12, "93%+", WHITE, 10, "middle", True)

# ── Legend ──
legend_y = 430
legend_items = [
    (20, BLUE, "Prefill"),
    (150, GREEN, "Decode"),
    (280, LIGHT_RED, "Bubble (idle)"),
    (440, GRAY, "Finished / idle"),
]
for lx, color, label in legend_items:
    add_rect(lx, legend_y, 16, 16, color, 3)
    add_text(lx + 22, legend_y + 13, label, LABEL_FILL, 12)

# ── Key insight text ──
add_text(20, 490, "Static: GPU waits for the longest request in each phase.", RED, 12, "start")
add_text(20, 508, "Continuous: prefill and decode interleaved — budget fills every step.", GREEN, 12, "start")

L.append('</svg>')

svg_content = '\n'.join(L)
with open('/mnt/e/Laboratory/vllm-from-scratch/instances/vllm/artifacts/04-continuous-batching/diagrams/bubble_comparison.svg', 'w') as f:
    f.write(svg_content)
print("SVG written. Validating...")
