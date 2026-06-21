"""n-gt-1-aggregation: n=3 parent aggregation, streaming vs FINAL_ONLY."""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


w, h = 1240, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')


def text(x, y, txt, fs=13, col="#0f172a", anchor="middle", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{fs}" '
        f'fill="{col}" text-anchor="{anchor}" font-weight="{weight}">{esc(txt)}</text>'
    )


def box(x, y, bw, bh, fill, stroke, lines, fs=13, rx=6, mono=False):
    L.append(
        f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    )
    n = len(lines)
    total = n * (fs + 3)
    sy = y + bh / 2 - total / 2 + fs
    for i, ln in enumerate(lines):
        text(x + bw / 2, sy + i * (fs + 3), ln,
             fs if i == 0 else fs - 1,
             "#0f172a" if i == 0 else "#475569",
             "middle", "bold" if i == 0 else "normal", mono)


def arrow(x1, y1, x2, y2, col="#475569"):
    L.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
        f'stroke-width="2" marker-end="url(#a)"/>'
    )


text(w / 2, 30, "n = 3 父聚合：ParentRequest.get_outputs", 18, "#0f172a", "middle", "bold")
text(w / 2, 50, "finished = not child_requests（子集合空即整体完成）", 13, "#94a3b8")

# divider
L.append(f'<line x1="{w/2}" y1="72" x2="{w/2}" y2="{h-20}" stroke="#e2e8f0" stroke-width="2"/>')

# LEFT: streaming
lx = 40
text(260, 96, "流式（非 FINAL_ONLY）：逐子转发", 15, "#2563eb", "middle", "bold")
child_y = 130
for i in range(3):
    cy = child_y + i * 70
    box(lx, cy, 150, 54, "#dbeafe", "#2563eb",
        [f"子 {i}", "当前 delta"])
    arrow(lx + 150, cy + 27, lx + 270, cy + 27)
    box(lx + 270, cy, 160, 54, "#f1f5f9", "#64748b",
        ["转发给客户端", f"index={i}"])
text(260, child_y + 3 * 70 + 16,
     "子完成 → child_requests 递减；已返回的不重发", 12, "#475569", "middle")
text(260, child_y + 3 * 70 + 40,
     "（already_finished_and_returned 保护）", 12, "#94a3b8", "middle")

# RIGHT: FINAL_ONLY aggregator
rx = w / 2 + 40
text(rx + 220, 96, "FINAL_ONLY：output_aggregator 攒齐 n 个", 15, "#a21caf", "middle", "bold")
agg_x = rx + 30
agg_y = 140
cell_w = 110
states = [
    ("子0 完成", ["[子0]", "—", "—"], "child={1,2} → 返回 []", False),
    ("子2 完成", ["[子0]", "—", "[子2]"], "child={1} → 返回 []", False),
    ("子1 完成", ["[子0]", "[子1]", "[子2]"], "child={} → 整体返回", True),
]
sy = agg_y
for title, cells, childline, done in states:
    text(agg_x, sy - 6, title, 13, "#0f172a", "start", "bold")
    for j, c in enumerate(cells):
        filled = c != "—"
        fill = "#fae8ff" if filled else "white"
        stroke = "#a21caf" if filled else "#cbd5e1"
        cx = agg_x + j * cell_w
        L.append(
            f'<rect x="{cx}" y="{sy}" width="{cell_w-8}" height="40" rx="4" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )
        text(cx + (cell_w - 8) / 2, sy + 25, c, 12, "#0f172a", "middle",
             "bold" if filled else "normal", True)
    cc = "#16a34a" if done else "#94a3b8"
    text(agg_x + 3 * cell_w + 6, sy + 25, childline, 12, cc, "start",
         "bold" if done else "normal", False)
    sy += 78
text(rx + 220, sy + 6, "三格齐（child_requests 空）才一次性返回全部 n 个", 12, "#a21caf", "middle")

L.append('</svg>')
open("n-gt-1-aggregation.svg", "w", encoding="utf-8").write('\n'.join(L))
print("wrote n-gt-1-aggregation.svg")
