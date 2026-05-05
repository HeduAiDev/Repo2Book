"""
Generate budget allocation diagram: how 512 token budget fills across 3 steps.
Shows the mix of prefill/decode tokens for each request per step.
"""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(str(s))

W, H = 960, 520
L = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
    '<defs>',
    '  <marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto">',
    '    <path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/>',
    '  </marker>',
    '</defs>',
    f'<rect width="{W}" height="{H}" fill="#fafbfc"/>',
]

TITLE_FILL = "#1e293b"
LABEL_FILL = "#334155"
SUBTITLE = "#64748b"
DARK_GRAY = "#94a3b8"
BLUE = "#3b82f6"
BLUE_LIGHT = "#93c5fd"
GREEN = "#22c55e"
GREEN_LIGHT = "#86efac"
WHITE = "#ffffff"
BORDER = "#cbd5e1"
BG_STEP = "#f8fafc"

def add_text(x, y, text, fill=LABEL_FILL, size=13, anchor="start", bold=False):
    fw = 'font-weight="bold"' if bold else ''
    L.append(f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-family="sans-serif" text-anchor="{anchor}" {fw}>{esc(text)}</text>')

# ═══════════════════ Layout ═══════════════════
add_text(W/2, 35, "Token Budget Allocation: First 3 Steps", TITLE_FILL, 19, "middle", True)
add_text(W/2, 56, "B = 512 max tokens per step. Prefill tokens fill leftover budget after decode.", SUBTITLE, 12, "middle")

# Three step columns
col_w = 240
col_h = 340
col_gap = 40
start_x = 60

for si, (step_label, items, budget_note) in enumerate([
    # Step 1
    ("Step 1", [
        ("r1", "512 tokens", "prefill chunk", 512/512, BLUE),
    ], "512/512 used"),
    # Step 2
    ("Step 2", [
        ("r1", "288 tokens", "remaining prefill", 288/512, "#60a5fa"),
        ("r2", "200 tokens", "full prefill", 200/512, BLUE),
        ("r3", "24 tokens", "prefill chunk", 24/512, BLUE),
    ], "512/512 used"),
    # Step 3
    ("Step 3", [
        ("r1", "1 token", "decode", 1/512, GREEN),
        ("r2", "1 token", "decode", 1/512, GREEN),
        ("r3", "26 tokens", "remaining prefill", 26/512, "#60a5fa"),
    ], "28/512 used"),
]):
    cx = start_x + si * (col_w + col_gap)

    # Column background
    L.append(f'<rect x="{cx}" y="70" width="{col_w}" height="{col_h}" fill="{BG_STEP}" stroke="{BORDER}" stroke-width="1.5" rx="8"/>')

    # Step header
    add_text(cx + col_w/2, 95, step_label, TITLE_FILL, 16, "middle", True)

    # Budget bar (full width)
    bar_x = cx + 20
    bar_w = col_w - 40
    bar_y = 110
    bar_h = 18
    total_w = 0
    bar_segments = []
    for _, _, _, frac, color in items:
        seg_w = int(bar_w * frac)
        bar_segments.append((seg_w, color))
        total_w += seg_w
    # Unused portion
    unused_w = bar_w - total_w
    if unused_w > 0:
        bar_segments.append((unused_w, "#e2e8f0"))

    # Draw bar segments
    seg_x = bar_x
    for seg_w, color in bar_segments:
        L.append(f'<rect x="{seg_x}" y="{bar_y}" width="{seg_w}" height="{bar_h}" fill="{color}" rx="2"/>')
        seg_x += seg_w

    add_text(bar_x + bar_w/2, bar_y + bar_h + 16, budget_note, DARK_GRAY, 11, "middle")

    # Request details below
    ry = bar_y + 50
    for req_id, tokens, phase, frac, color in items:
        # Color indicator
        L.append(f'<rect x="{cx + 15}" y="{ry + 4}" width="10" height="10" fill="{color}" rx="2"/>')
        add_text(cx + 32, ry + 14, f"{req_id}: {tokens}", LABEL_FILL, 13, "start", True)
        add_text(cx + 32, ry + 30, phase, SUBTITLE, 11, "start")
        ry += 55

    # Step arrow (except last)
    if si < 2:
        ax = cx + col_w + 5
        ay = 70 + col_h/2
        L.append(f'<line x1="{ax}" y1="{ay}" x2="{ax + col_gap - 12}" y2="{ay}" stroke="{DARK_GRAY}" stroke-width="2" marker-end="url(#ar)"/>')

# Bottom insight
add_text(60, 440, "Key insight: Step 2 mixes prefill (r1, r2, r3) tokens in one forward pass — this is continuous batching.", GREEN_LIGHT, 13, "start")
add_text(60, 460, "Step 3 mixes r3's remaining prefill with r1/r2 decode tokens. No phase segregation.", GREEN_LIGHT, 13, "start")

# Legend
add_text(60, 498, "Prefill", LABEL_FILL, 12, "start")
L.append(f'<rect x="120" y="488" width="14" height="14" fill="{BLUE}" rx="2"/>')
add_text(160, 498, "Prefill (chunked)", LABEL_FILL, 12, "start")
L.append(f'<rect x="300" y="488" width="14" height="14" fill="#60a5fa" rx="2"/>')
add_text(440, 498, "Decode", LABEL_FILL, 12, "start")
L.append(f'<rect x="500" y="488" width="14" height="14" fill="{GREEN}" rx="2"/>')
add_text(560, 498, "Unused budget", LABEL_FILL, 12, "start")
L.append(f'<rect x="680" y="488" width="14" height="14" fill="#e2e8f0" stroke="{BORDER}" stroke-width="0.5" rx="2"/>')

L.append('</svg>')

svg_content = '\n'.join(L)
with open('/mnt/e/Laboratory/vllm-from-scratch/instances/vllm/artifacts/04-continuous-batching/diagrams/budget_allocation.svg', 'w') as f:
    f.write(svg_content)
print("Budget allocation SVG written.")
