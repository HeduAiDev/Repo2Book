#!/usr/bin/env python3
"""连续批处理时间线：每拍混跑多个不同阶段的请求。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 920, 490
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3"'
    ' markerWidth="6" markerHeight="4" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def t(x, y, s, size=13, anchor="middle", weight="normal", fill="#1e293b"):
    L.append(
        f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" '
        f'text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">'
        f'{esc(s)}</text>'
    )


def cell(x, y, w, h, bg, border, lines, size=13, tcolor="#1e293b", rx=7):
    L.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{bg}" stroke="{border}" stroke-width="1.8"/>'
    )
    if lines:
        total_h = len(lines) * (size + 3)
        cy = y + h / 2 - total_h / 2 + size
        for i, ln in enumerate(lines):
            t(x + w / 2, cy + i * (size + 3), ln, size=size,
              anchor="middle", fill=tcolor)


# ── Title ──────────────────────────────────────────────────────────
t(W / 2, 32, "连续批处理：每拍混跑多个不同阶段的请求",
  size=18, weight="bold", fill="#0f172a")

# ── Grid geometry ──────────────────────────────────────────────────
LABEL_W = 108      # left label column width
STEP_COLS = 4      # number of time steps
COL_W = 190        # each step column width
ROW_H = 74         # each request row height
GAP = 28           # gap between cells (wider to give room for between-row annotations)
GRID_X = LABEL_W + 10   # grid starts here (x)
HEADER_Y = 52      # top of header row
GRID_Y = HEADER_Y + 60  # top of first data row (increased to give space for annotation labels)

steps = ["Step  t−1", "Step  t", "Step  t+1", "Step  t+2"]
reqs  = ["Req A", "Req B", "Req C"]

# ── Phase definitions ───────────────────────────────────────────────
# (bg, border, text_lines, text_color)
PREFILL  = ("#fef3c7", "#d97706", ["prefill"], "#92400e")
DECODE   = ("#dbeafe", "#2563eb", ["decode"],  "#1e40af")
DONE     = ("#f1f5f9", "#94a3b8", ["完成"],     "#64748b")
NONE     = ("#f8fafc", "#e2e8f0", ["—"],        "#94a3b8")

# Grid[row][col] = phase  (row=0→A, 1→B, 2→C; col=0→t-1, 1→t, 2→t+1, 3→t+2)
grid = [
    [NONE,    PREFILL, DECODE,  DECODE],   # Req A
    [DECODE,  DECODE,  DECODE,  DONE  ],   # Req B  (finishes after t+1)
    [NONE,    NONE,    NONE,    PREFILL],  # Req C
]

# ── Column headers ──────────────────────────────────────────────────
for ci, sname in enumerate(steps):
    cx = GRID_X + ci * (COL_W + GAP) + COL_W / 2
    t(cx, HEADER_Y, sname, size=14, weight="bold", fill="#334155")

# ── Row labels ─────────────────────────────────────────────────────
for ri, rname in enumerate(reqs):
    ry = GRID_Y + ri * (ROW_H + GAP) + ROW_H / 2 + 5
    t(LABEL_W / 2, ry, rname, size=14, weight="bold", fill="#1e293b")

# ── Cells ──────────────────────────────────────────────────────────
for ri in range(len(reqs)):
    for ci in range(STEP_COLS):
        bg, border, lines, tcolor = grid[ri][ci]
        cx = GRID_X + ci * (COL_W + GAP)
        cy = GRID_Y + ri * (ROW_H + GAP)
        cell(cx, cy, COL_W, ROW_H, bg, border, lines, size=14, tcolor=tcolor)

# ── "enters batch" annotation arrows for Req A col t and Req C col t+2 ──
# Req A enters at col t (ci=1, ri=0)
ax = GRID_X + 1 * (COL_W + GAP)
ay = GRID_Y + 0 * (ROW_H + GAP)  # top of the cell
# small downward arrow from above cell
L.append(
    f'<line x1="{ax + COL_W/2}" y1="{ay - 18}" '
    f'x2="{ax + COL_W/2}" y2="{ay - 3}" '
    f'stroke="#d97706" stroke-width="2" marker-end="url(#a)"/>'
)
t(ax + COL_W / 2, ay - 22, "新请求入队", size=11, fill="#b45309", weight="bold")

# Req C enters at col t+2 (ci=3, ri=2)
cx3 = GRID_X + 3 * (COL_W + GAP)
cy3 = GRID_Y + 2 * (ROW_H + GAP)  # top of the cell
L.append(
    f'<line x1="{cx3 + COL_W/2}" y1="{cy3 - 18}" '
    f'x2="{cx3 + COL_W/2}" y2="{cy3 - 3}" '
    f'stroke="#d97706" stroke-width="2" marker-end="url(#a)"/>'
)
t(cx3 + COL_W / 2, cy3 - 22, "新请求入队", size=11, fill="#b45309", weight="bold")

# ── Req B "finished" callout ─────────────────────────────────────
bx = GRID_X + 2 * (COL_W + GAP) + COL_W / 2
by = GRID_Y + 1 * (ROW_H + GAP) + ROW_H + 4
t(bx, by + 13, "↓ t+1 后结束", size=11, fill="#64748b")

# ── Bottom note ────────────────────────────────────────────────────
note_y = GRID_Y + len(reqs) * (ROW_H + GAP) + 18
t(W / 2, note_y,
  "每拍 EngineCore.step() 同时推进这整批——prefill 和 decode 混在一次前向里算",
  size=13, fill="#334155")

# ── Legend ─────────────────────────────────────────────────────────
legend_y = note_y + 30
items = [
    ("#fef3c7", "#d97706", "prefill（处理 prompt token）"),
    ("#dbeafe", "#2563eb", "decode（逐 token 推理）"),
    ("#f1f5f9", "#94a3b8", "完成 / 未入队"),
]
lx = 220
for bg, border, label in items:
    L.append(
        f'<rect x="{lx}" y="{legend_y - 14}" width="24" height="18" '
        f'rx="4" fill="{bg}" stroke="{border}" stroke-width="1.5"/>'
    )
    t(lx + 32, legend_y, label, size=12, anchor="start", fill="#334155")
    lx += 210

L.append('</svg>')
out = __file__.replace("gen_continuous_batching.py", "03-continuous-batching.svg")
with open(out, "w") as f:
    f.write('\n'.join(L))
print(f"wrote {out}")
