#!/usr/bin/env python3
"""ch03 figure: compute_hash pipeline + before/after fingerprint across 3 scenarios.
template: before-after (retrofit — figure was missing, this is the first figure for
mechanism ch03-compute-hash). All coordinates computed from constants/loops."""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(s)


# ---- pipeline strip (top) ----
PIPE_STEPS = [
    "factors = [版本, model_h, cache_h,\nparallel_h, scheduler_h, compilation_h, kernel_h]",
    "str(factors).encode()",
    "SHA-256",
    "hexdigest()[:10]\n= 指纹",
]

# ---- 3 scenario columns ----
# each: title, list of (factor_name, before, after, changed), fp_before, fp_after, fp_changed, outcome, outcome_color
COLUMNS = [
    {
        "title": "改 TP 1 → 2",
        "rows": [
            ("parallel", "028ab3525e", "54dfeba4d6", True),
            ("scheduler", "ac26a305e1", "ac26a305e1", False),
            ("compilation", "27caeac125", "27caeac125", False),
        ],
        "fp_before": "5f197a5c93", "fp_after": "7f9de5602c", "fp_changed": True,
        "outcome": "重编译", "outcome_fill": "#fee2e2", "outcome_stroke": "#dc2626",
    },
    {
        "title": "改 max_num_seqs 256 → 512",
        "rows": [
            ("parallel", "028ab3525e", "028ab3525e", False),
            ("scheduler", "ac26a305e1", "ac26a305e1", False),
            ("compilation", "27caeac125", "27caeac125", False),
        ],
        "fp_before": "5f197a5c93", "fp_after": "5f197a5c93", "fp_changed": False,
        "outcome": "命中缓存", "outcome_fill": "#dcfce7", "outcome_stroke": "#16a34a",
    },
    {
        "title": "改优化级 O2 → O0",
        "rows": [
            ("parallel", "028ab3525e", "028ab3525e", False),
            ("scheduler", "ac26a305e1", "ac26a305e1", False),
            ("compilation", "27caeac125", "89294b0e76", True),
        ],
        "fp_before": "5f197a5c93", "fp_after": "17488b3f00", "fp_changed": True,
        "outcome": "重编译", "outcome_fill": "#fee2e2", "outcome_stroke": "#dc2626",
    },
]

HL_FILL, HL_STROKE = "#fef3c7", "#d97706"
UNCH_FILL, UNCH_STROKE = "#f1f5f9", "#64748b"

COL_W = 360
COL_GAP = 46
PAD = 40
W = PAD * 2 + COL_W * 3 + COL_GAP * 2
PIPE_TOP = 66
PIPE_BOX_W, PIPE_BOX_H = 250, 56
PIPE_GAP = 46
ANCHOR_Y = PIPE_TOP + PIPE_BOX_H + 74
ANCHOR_W, ANCHOR_H = 420, 54
COL_TOP = ANCHOR_Y + ANCHOR_H + 66
ROW_H, ROW_GAP = 34, 10
FP_ROW_H = 40
OUTCOME_H = 34
COL_TITLE_H = 26

col_body_h = COL_TITLE_H + 8 + len(COLUMNS[0]["rows"]) * (ROW_H + ROW_GAP) + 10 + FP_ROW_H + 16 + OUTCOME_H
H = COL_TOP + col_body_h + 56

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="19" font-weight="bold" '
          f'fill="#0f172a">compute_hash：改哪个因子才会翻新指纹、触发重编译？</text>')


def box(x, y, w, h, lines, fill, stroke, fs=13, bold_last=False):
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    n = len(lines)
    for i, t in enumerate(lines):
        ty = y + h / 2 + (i - (n - 1) / 2) * (fs + 5) + fs / 3
        fw = ' font-weight="bold"' if (bold_last and i == n - 1) else ''
        L.append(f'<text x="{x + w / 2:.1f}" y="{ty:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs}"{fw} fill="#0f172a">{esc(t)}</text>')


def arrow(x1, y1, x2, y2, label=None, lx=None, ly=None):
    L.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
              f'stroke="#475569" stroke-width="1.7" marker-end="url(#a)"/>')
    if label:
        L.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="11" font-weight="bold" fill="#334155">{esc(label)}</text>')


# ---- pipeline strip ----
n_pipe = len(PIPE_STEPS)
pipe_total_w = n_pipe * PIPE_BOX_W + (n_pipe - 1) * PIPE_GAP
pipe_x0 = (W - pipe_total_w) / 2
for i, step in enumerate(PIPE_STEPS):
    px = pipe_x0 + i * (PIPE_BOX_W + PIPE_GAP)
    lines = step.split("\n")
    fill = "#e0e7ff" if i == n_pipe - 1 else "#eef2ff"
    stroke = "#4338ca" if i == n_pipe - 1 else "#6366f1"
    box(px, PIPE_TOP, PIPE_BOX_W, PIPE_BOX_H, lines, fill, stroke, fs=12.5)
    if i < n_pipe - 1:
        arrow(px + PIPE_BOX_W, PIPE_TOP + PIPE_BOX_H / 2, px + PIPE_BOX_W + PIPE_GAP - 4,
              PIPE_TOP + PIPE_BOX_H / 2)

# ---- baseline anchor ----
anchor_x = (W - ANCHOR_W) / 2
box(anchor_x, ANCHOR_Y, ANCHOR_W, ANCHOR_H,
    ["baseline: TP=1, max_num_seqs=256, O2  →  指纹 = 5f197a5c93"],
    "#f0fdf4", "#16a34a", fs=13, bold_last=False)
anchor_cx = anchor_x + ANCHOR_W / 2
anchor_by = ANCHOR_Y + ANCHOR_H

# ---- 3 columns ----
col_xs = [PAD + c * (COL_W + COL_GAP) for c in range(3)]
for c, col in enumerate(COLUMNS):
    cx0 = col_xs[c]
    col_cx = cx0 + COL_W / 2
    # arrow from anchor down to this column
    arrow(anchor_cx, anchor_by, col_cx, COL_TOP - 6)
    # title
    L.append(f'<text x="{col_cx:.1f}" y="{COL_TOP + COL_TITLE_H - 8:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" '
              f'fill="#0f172a">{esc(col["title"])}</text>')
    y = COL_TOP + COL_TITLE_H + 8
    row_w = COL_W
    for name, before, after, changed in col["rows"]:
        fill = HL_FILL if changed else UNCH_FILL
        stroke = HL_STROKE if changed else UNCH_STROKE
        L.append(f'<rect x="{cx0:.1f}" y="{y:.1f}" width="{row_w:.1f}" height="{ROW_H:.1f}" rx="6" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
        arrow_glyph = "→" if changed else "≡"
        line = f"{name}: {before} {arrow_glyph} {after}"
        L.append(f'<text x="{cx0 + row_w / 2:.1f}" y="{y + ROW_H / 2 + 4.5:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" fill="#0f172a">{esc(line)}</text>')
        y += ROW_H + ROW_GAP
    y += 6
    # fingerprint row
    fp_fill = HL_FILL if col["fp_changed"] else "#dbeafe"
    fp_stroke = HL_STROKE if col["fp_changed"] else "#2563eb"
    L.append(f'<rect x="{cx0:.1f}" y="{y:.1f}" width="{row_w:.1f}" height="{FP_ROW_H:.1f}" rx="7" '
              f'fill="{fp_fill}" stroke="{fp_stroke}" stroke-width="1.8"/>')
    fp_glyph = "→" if col["fp_changed"] else "≡"
    fp_line = f'指纹 {col["fp_before"]} {fp_glyph} {col["fp_after"]}'
    L.append(f'<text x="{cx0 + row_w / 2:.1f}" y="{y + FP_ROW_H / 2 + 5:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
              f'fill="#0f172a">{esc(fp_line)}</text>')
    y += FP_ROW_H + 16
    # outcome badge
    ob_w = 132
    L.append(f'<rect x="{col_cx - ob_w / 2:.1f}" y="{y:.1f}" width="{ob_w:.1f}" height="{OUTCOME_H:.1f}" '
              f'rx="17" fill="{col["outcome_fill"]}" stroke="{col["outcome_stroke"]}" stroke-width="1.8"/>')
    L.append(f'<text x="{col_cx:.1f}" y="{y + OUTCOME_H / 2 + 5:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{col["outcome_stroke"]}">{esc(col["outcome"])}</text>')

L.append(f'<text x="{PAD}" y="{H - 18}" font-family="sans-serif" font-size="12" fill="#64748b">'
          f'{esc("scheduler 因子 ac26a305e1 三例恒定：max_num_seqs 不入 scheduler 因子，只影响调度行为不影响计算图。")}</text>')
L.append('</svg>')

with open("ch03-compute-hash-fingerprint.svg", "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("wrote ch03-compute-hash-fingerprint.svg")
