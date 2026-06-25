"""05-stop-string-window: stop-string sliding window search illustration.
stop_str = "ab" (len=2)
Step 1: output_text="xyzc", new_char_count=1, start=1-1-2=-2, window="zc", no match
Step 2: output_text="xyzcab", new_char_count=2, start=1-2-2=-3, window="cab", match "ab"
"""
import xml.sax.saxutils as xs
import os


def esc(s):
    return xs.escape(str(s))


W, H = 920, 530
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>')
L.append(
    '<marker id="arr" viewBox="0 0 10 6" refX="9" refY="3" '
    'markerWidth="6" markerHeight="4" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
)
L.append(
    '<marker id="arr_blue" viewBox="0 0 10 6" refX="9" refY="3" '
    'markerWidth="6" markerHeight="4" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker>'
)
L.append('</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def txt(x, y, s, fs=13, col="#0f172a", anchor="middle", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-family="{esc(fam)}" font-size="{fs}" '
        f'fill="{col}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>'
    )


def rect(x, y, w, h, fill, stroke="#94a3b8", sw=1.5, rx=3):
    L.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}" rx="{rx}"/>'
    )


def bracket(x1, x2, y, col="#2563eb", label=""):
    """Draw a bottom-open bracket from x1 to x2 at y, with optional label below."""
    bh = 14  # bracket arm height
    L.append(
        f'<line x1="{x1}" y1="{y}" x2="{x1}" y2="{y+bh}" '
        f'stroke="{col}" stroke-width="2"/>'
    )
    L.append(
        f'<line x1="{x2}" y1="{y}" x2="{x2}" y2="{y+bh}" '
        f'stroke="{col}" stroke-width="2"/>'
    )
    L.append(
        f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" '
        f'stroke="{col}" stroke-width="2"/>'
    )
    if label:
        mid = (x1 + x2) / 2
        txt(mid, y + bh + 15, label, 12, col, "middle", "bold")


# ── Title ──────────────────────────────────────────────────────────────────
txt(W / 2, 32, "停止串窗口搜索：只扫新增字符覆盖所有跨步命中", 17, "#0f172a", "middle", "bold")
txt(W / 2, 54, 'stop_str = "ab"（长 2）　搜索起点 = 1 − new_char_count − stop_string_len', 13, "#475569")

# ── Legend ─────────────────────────────────────────────────────────────────
lx, ly, lbox = 60, 78, 18
for fill, label in [
    ("#e2e8f0", "已扫旧字符"),
    ("#bbf7d0", "本步新增字符"),
    ("#fee2e2", "命中停止串"),
]:
    rect(lx, ly, lbox, lbox, fill, "#94a3b8", 1)
    txt(lx + lbox + 6, ly + 13, label, 12, "#334155", "start")
    lx += 120

# dashed box legend for search window
L.append(
    f'<rect x="{lx}" y="{ly}" width="{lbox}" height="{lbox}" fill="none" '
    f'stroke="#2563eb" stroke-width="2" stroke-dasharray="4,2" rx="2"/>'
)
txt(lx + lbox + 6, ly + 13, "搜索窗口", 12, "#2563eb", "start")

# ── Helper: draw one step ──────────────────────────────────────────────────
CELL = 56   # cell width
CH = 42     # cell height
LEFT = 60   # left margin for chars


def draw_step(
    y0,
    step_num,
    chars,          # list of char strings
    num_old,        # how many are "old" (gray)
    window_start,   # index of first char in search window
    match_start,    # index of match (-1 = no match)
    match_len,      # length of match
    formula,        # formula string
    result_text,    # result label
    result_ok,      # True=match, False=no match
):
    """Draw one time-step row."""
    n = len(chars)
    total_w = n * CELL

    # Step label
    txt(LEFT - 8, y0 + 22, f"Step {step_num}", 14, "#334155", "end", "bold")

    # output_text label
    quoted = '"' + "".join(chars) + '"'
    txt(LEFT, y0 - 6, f"output_text = {quoted}", 12, "#64748b", "start")

    # Draw cells
    for i, ch in enumerate(chars):
        cx = LEFT + i * CELL
        if match_start != -1 and match_start <= i < match_start + match_len:
            fill = "#fee2e2"
            stroke = "#ef4444"
            sw = 2.5
        elif i >= len(chars) - (len(chars) - num_old):
            fill = "#bbf7d0"
            stroke = "#22c55e"
            sw = 2.0
        else:
            fill = "#e2e8f0"
            stroke = "#94a3b8"
            sw = 1.5
        rect(cx, y0, CELL, CH, fill, stroke, sw)
        txt(cx + CELL / 2, y0 + CH / 2 + 6, ch, 18, "#0f172a", "middle", "bold", mono=True)

    # index labels below cells
    for i in range(n):
        cx = LEFT + i * CELL + CELL / 2
        txt(cx, y0 + CH + 14, str(i), 11, "#94a3b8", "middle")

    # Search window bracket (blue dashed overlay rect)
    wx = LEFT + window_start * CELL
    ww = (n - window_start) * CELL
    L.append(
        f'<rect x="{wx}" y="{y0}" width="{ww}" height="{CH}" fill="none" '
        f'stroke="#2563eb" stroke-width="2.5" stroke-dasharray="6,3" rx="3"/>'
    )

    # Formula label right of cells
    fx = LEFT + total_w + 20
    txt(fx, y0 + 18, formula, 12, "#2563eb", "start", "bold", mono=True)

    # Result label
    rcol = "#15803d" if result_ok else "#94a3b8"
    rsym = "✓ " if result_ok else "✗ "
    txt(fx, y0 + 36, rsym + result_text, 13, rcol, "start", "bold")


# ── Step 1 ──────────────────────────────────────────────────────────────────
# output_text = "xyzc", new_char_count=1, start=1-1-2=-2 → index len-2=2 → "z"
# window covers chars at index 2,3 = "zc", no "ab" found
draw_step(
    y0=115,
    step_num=1,
    chars=["x", "y", "z", "c"],
    num_old=3,
    window_start=2,   # index 2 = "z" → 1-1-2=-2 → last 2 chars
    match_start=-1,
    match_len=0,
    formula="起点 = 1 − 1 − 2 = −2  →  index 2",
    result_text="窗口=\"zc\"，未命中",
    result_ok=False,
)

# annotation: new_char_count
txt(LEFT + 3 * CELL + CELL / 2, 115 + CH + 28, "new_char_count=1", 11, "#22c55e", "middle", "bold")

# ── Divider ─────────────────────────────────────────────────────────────────
L.append(f'<line x1="40" y1="240" x2="{W-40}" y2="240" stroke="#e2e8f0" stroke-width="1.5" stroke-dasharray="6,4"/>')

# ── Step 2 ──────────────────────────────────────────────────────────────────
# output_text = "xyzcab", new_char_count=2, start=1-2-2=-3 → last 3 chars "cab"
# "ab" found at index 4
draw_step(
    y0=265,
    step_num=2,
    chars=["x", "y", "z", "c", "a", "b"],
    num_old=4,
    window_start=3,   # 1-2-2=-3 → last 3 chars: index 3,4,5 = "cab"
    match_start=4,
    match_len=2,
    formula="起点 = 1 − 2 − 2 = −3  →  index 3",
    result_text='窗口="cab"，命中 "ab"（pos 4）',
    result_ok=True,
)

txt(LEFT + 4 * CELL + CELL, 265 + CH + 28, "new_char_count=2", 11, "#22c55e", "middle", "bold")

# ── Key insight box ──────────────────────────────────────────────────────────
bx, by, bw, bh2 = 40, 400, W - 80, 100
rect(bx, by, bw, bh2, "#f8fafc", "#94a3b8", 1.5, 6)
txt(bx + bw / 2, by + 22, "为什么这个窗口足够？", 13, "#334155", "middle", "bold")
txt(bx + bw / 2, by + 44,
    "停止串首次命中 → 其尾部必然落在本步新增字符里（否则上一步就该命中了）",
    12, "#475569", "middle")
txt(bx + bw / 2, by + 62,
    "→ 只需覆盖「新增字符 + 前面 stop_len−1 个旧字符」= 窗口大小 stop_len + new_char_count − 1",
    12, "#475569", "middle")
txt(bx + bw / 2, by + 80,
    "→ 搜索起点从末尾往回 (new_char_count + stop_len − 1) 位 = Python 负索引 1 − new_char_count − stop_len",
    12, "#475569", "middle")

L.append('</svg>')

out_dir = os.path.dirname(os.path.abspath(__file__))
svg_path = os.path.join(out_dir, "05-stop-string-window.svg")
with open(svg_path, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print(f"wrote {svg_path}")
