"""DP topology + load balance: N front-ends <-> coordinator <-> M engines."""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


w, h = 1000, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#047857"/></marker>'
    '<marker id="ab" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#1d4ed8"/></marker>'
    '<marker id="ao" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append('<text x="20" y="32" font-size="20" font-weight="bold" fill="#0f172a" '
         'font-family="sans-serif">DP 拓扑：前端 ⇄ Coordinator ⇄ Engine，三类消息流</text>')

# columns
front_x, coord_x, eng_x = 60, 430, 760
bw = 180
fronts = ["前端 0\nDPLBAsyncMPClient", "前端 1\nDPLBAsyncMPClient"]
engines = ["engine 0", "engine 1", "engine 2"]


def node(x, y, bw, bh, lines, color, fill):
    L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8" '
             f'fill="{fill}" stroke="{color}" stroke-width="2"/>')
    for i, ln in enumerate(lines):
        fw = "bold" if i == 0 else "normal"
        fs = 14 if i == 0 else 12
        col = color if i == 0 else "#475569"
        L.append(f'<text x="{x + bw // 2}" y="{y + 22 + i * 18}" font-size="{fs}" '
                 f'text-anchor="middle" fill="{col}" font-weight="{fw}" '
                 f'font-family="sans-serif">{esc(ln)}</text>')


# front-end nodes
f_y = [110, 320]
for i, fy in enumerate(f_y):
    node(front_x, fy, bw, 60, fronts[i].split("\n"), "#1d4ed8", "#eff6ff")

# coordinator (center, tall)
coord_y, coord_h = 150, 230
node(coord_x, coord_y, bw, coord_h,
     ["DPCoordinator", "聚合 [waiting,running]", "跟踪 current_wave", "engines_running",
      "暂停时广播", "START_DP_WAVE"],
     "#7c3aed", "#f5f3ff")

# engine nodes
e_y = [80, 250, 420]
for i, ey in enumerate(e_y):
    node(eng_x, ey, bw, 64,
         [engines[i], "DPEngineCoreProc", "dp_group all-reduce"],
         "#b45309", "#fffbeb")

# arrows: engine -> coord (stats + wave_complete), green up-link
for ey in e_y:
    L.append(f'<line x1="{eng_x}" y1="{ey + 40}" x2="{coord_x + bw + 4}" '
             f'y2="{coord_y + 60}" stroke="#047857" stroke-width="1.6" '
             'marker-end="url(#ag)"/>')
L.append(f'<text x="{(coord_x + bw + eng_x) // 2 + 10}" y="{coord_y + 30}" '
         'font-size="11.5" text-anchor="middle" fill="#047857" '
         'font-family="sans-serif">stats(waiting,running)</text>')
L.append(f'<text x="{(coord_x + bw + eng_x) // 2 + 10}" y="{coord_y + 46}" '
         'font-size="11.5" text-anchor="middle" fill="#047857" '
         'font-family="sans-serif">+ wave_complete</text>')

# arrows: coord -> engine (START_DP_WAVE), orange down-link
for ey in e_y:
    L.append(f'<line x1="{coord_x + bw}" y1="{coord_y + 140}" x2="{eng_x - 4}" '
             f'y2="{ey + 50}" stroke="#b45309" stroke-width="1.6" '
             'stroke-dasharray="5 3" marker-end="url(#ao)"/>')
L.append(f'<text x="{(coord_x + bw + eng_x) // 2 + 10}" y="{coord_y + 185}" '
         'font-size="11.5" text-anchor="middle" fill="#b45309" '
         'font-family="sans-serif">START_DP_WAVE</text>')

# arrows: coord -> front (counts,wave,running broadcast), blue
for fy in f_y:
    L.append(f'<line x1="{coord_x}" y1="{coord_y + 100}" x2="{front_x + bw + 4}" '
             f'y2="{fy + 30}" stroke="#1d4ed8" stroke-width="1.6" '
             'marker-end="url(#ab)"/>')
L.append(f'<text x="{(front_x + bw + coord_x) // 2}" y="{coord_y + 78}" '
         'font-size="11.5" text-anchor="middle" fill="#1d4ed8" '
         'font-family="sans-serif">广播 (counts,wave,running)</text>')

# arrows: front -> coord (FIRST_REQ), gray dashed
for fy in f_y:
    L.append(f'<line x1="{front_x + bw}" y1="{fy + 45}" x2="{coord_x - 4}" '
             f'y2="{coord_y + 200}" stroke="#475569" stroke-width="1.4" '
             'stroke-dasharray="4 3" marker-end="url(#a)"/>')
L.append(f'<text x="{(front_x + bw + coord_x) // 2}" y="{coord_y + 215}" '
         'font-size="11.5" text-anchor="middle" fill="#475569" '
         'font-family="sans-serif">FIRST_REQ（暂停时唤醒）</text>')

# load-balance score table overlaid bottom-left under front 1
tx, ty = front_x, 410
L.append(f'<rect x="{tx}" y="{ty}" width="320" height="120" rx="8" '
         'fill="#f8fafc" stroke="#cbd5e1"/>')
L.append(f'<text x="{tx + 12}" y="{ty + 22}" font-size="13" fill="#0f172a" '
         'font-weight="bold" font-family="sans-serif">前端选 engine：score = waiting·4 + running</text>')
rows = [
    ("engine", "waiting", "running", "score"),
    ("0", "2", "1", "9"),
    ("1", "1", "5", "9"),
    ("2", "0", "0", "0  ← 选它"),
]
colx = [tx + 14, tx + 100, tx + 185, tx + 250]
for ri, row in enumerate(rows):
    yy = ty + 44 + ri * 19
    pick = "←" in row[3]
    for ci, cell in enumerate(row):
        col = "#b91c1c" if pick else ("#334155" if ri else "#64748b")
        fw = "bold" if (ri == 0 or pick) else "normal"
        L.append(f'<text x="{colx[ci]}" y="{yy}" font-size="12" fill="{col}" '
                 f'font-weight="{fw}" font-family="sans-serif">{esc(cell)}</text>')

L.append('</svg>')
open("instances/vllm/artifacts/ch21-async-engine/diagrams/dp-topology-and-loadbalance.svg",
     "w", encoding="utf-8").write('\n'.join(L))
print("wrote dp-topology-and-loadbalance.svg")
