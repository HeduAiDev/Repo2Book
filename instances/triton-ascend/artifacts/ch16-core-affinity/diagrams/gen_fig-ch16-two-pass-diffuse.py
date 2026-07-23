#!/usr/bin/env python3
"""state-table 模板:两遍 diffuse 迭代到不动点。横轴=阶段(Pass1 R1..R4,兜底,Pass2),
纵轴=代表节点(dot,%a,load_a,%pa,%d,store,const);U/V/C 三色格,竖分隔线切三段。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK = "#0f172a"
GRAY = "#64748b"
CUBE = "#1e40af"
CUBE_BG = "#dbeafe"
VEC = "#15803d"
VEC_BG = "#dcfce7"
UND = "#64748b"
UND_BG = "#f1f5f9"

TITLE = "两遍 diffuse:迭代到不动点(kernel G,16 节点,threshold=16×5=80)"
SUBTITLE = "Pass1 worklist 迭代 → 残留 UNDETERMINED 兜底判 VECTOR_ONLY → Pass2 再确认不动点(本例无变化)"

COLS = ["Pass1\nR1", "Pass1\nR2", "Pass1\nR3", "Pass1\nR4", "兜底\nL474-478", "Pass2\nL480"]
ROWS = ["dot", "%a", "load_a", "%pa", "%d", "store", "const/%z/return"]

# grid[row][col] = "C"/"V"/"U"
GRID = {
    "dot":             ["C", "C", "C", "C", "C", "C"],
    "%a":              ["U", "C", "C", "C", "C", "C"],
    "load_a":          ["U", "U", "C", "C", "C", "C"],
    "%pa":             ["U", "U", "U", "C", "C", "C"],
    "%d":              ["U", "V", "V", "V", "V", "V"],
    "store":           ["V", "V", "V", "V", "V", "V"],
    "const/%z/return":  ["U", "U", "U", "U", "V", "V"],
}
# 本轮相对上一列发生变化的格(用于加粗描边强调)
CHANGED = {
    "dot": [0],
    "%a": [1],
    "load_a": [2],
    "%pa": [3],
    "%d": [1],
    "store": [0],
    "const/%z/return": [4],
}
COLOR = {"C": (CUBE_BG, CUBE), "V": (VEC_BG, VEC), "U": (UND_BG, UND)}
NAME = {"C": "CUBE_ONLY", "V": "VECTOR_ONLY", "U": "UNDETERMINED"}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 170, 128, 46, 44, 118, 30
W = PAD * 2 + LABEL_W + COL_W * len(COLS)
H = TOP + HEADER_H + ROW_H * len(ROWS) + 150
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROWS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="{INK}">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12.5" fill="{GRAY}">'
     f'{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    lines = name.split("\n")
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-6}" height="{HEADER_H-6}" rx="4" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.2"/>')
    n = len(lines)
    y0 = TOP + (HEADER_H - 6) / 2 - (n - 1) * 8 + 4
    for k, line in enumerate(lines):
        L.append(f'<text x="{x+(COL_W-6)/2}" y="{y0+k*16}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="white" '
                  f'font-weight="bold">{esc(line)}</text>')

for i, row in enumerate(ROWS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-14}" y="{ry+ROW_H/2+5}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    for j in range(len(COLS)):
        cx = col_x[j]
        val = GRID[row][j]
        bg, stroke = COLOR[val]
        changed = j in CHANGED[row]
        sw = 2.6 if changed else 1.2
        L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-6}" height="{ROW_H-8}" rx="4" '
                  f'fill="{bg}" stroke="{stroke}" stroke-width="{sw}"/>')
        L.append(f'<text x="{cx+(COL_W-6)/2}" y="{ry+ROW_H/2+6}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="15" font-weight="bold" '
                  f'fill="{stroke}">{esc(val)}</text>')

# section divider lines: Pass1(0-3) | 兜底(4) | Pass2(5)
grid_top = TOP + HEADER_H
grid_bot = row_y[-1] + ROW_H
for boundary in (4,):
    x = col_x[boundary]
    L.append(f'<line x1="{x}" y1="{grid_top}" x2="{x}" y2="{grid_bot}" stroke="#0f172a" '
              'stroke-width="2.4" stroke-dasharray="2,0"/>')

# legend
ly = grid_bot + 34
legend = [("C", "CUBE_ONLY"), ("V", "VECTOR_ONLY"), ("U", "UNDETERMINED")]
lx = PAD
for key, name in legend:
    bg, stroke = COLOR[key]
    L.append(f'<rect x="{lx}" y="{ly}" width="20" height="20" rx="4" fill="{bg}" '
              f'stroke="{stroke}" stroke-width="1.6"/>')
    L.append(f'<text x="{lx+28}" y="{ly+15}" font-family="sans-serif" font-size="12.5" '
              f'fill="{INK}">{esc(key + " = " + name)}</text>')
    lx += 30 + 11 * len(key + " = " + name) + 24

foot_y = ly + 40
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" fill="{GRAY}">'
          f'{esc("粗描边=本列相对上一列发生变化;竖分隔线左=Pass1(4 轮到不动点)、右=兜底后 Pass2(2 次 diffuse() 调用,DAG.cpp:L472/L480)")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="12" fill="{GRAY}">'
          f'{esc("兜底把 3 个残留 UNDETERMINED 节点(const/%z/return)设为 VECTOR_ONLY(DAG.cpp:L474-478);Pass2 本例无变化,即刻稳定")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch16-two-pass-diffuse.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
