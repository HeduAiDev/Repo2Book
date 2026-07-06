#!/usr/bin/env python3
"""fig36-5-value-rerank: state-table 模板(转置:列=树节点)。
每列一个节点,行=depth/token/局部置信度 c/路径价值 V=∏c/是否被 top-m 选中。
数字来自 explainer.json fig36-5 numbers(traces/eagle2_tree.json)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "EAGLE-2 动态树的 value 与重排"
SUBTITLE = "路径价值 V = 该节点到根路径上置信度之积；重排取全树 top-m=4（浅层优先打破平局）"
NODES = ["根\nd0_t4", "d1_t3", "d1_t5", "d2_t1a", "d2_t0a", "d2_t1b", "d2_t0b"]
ROW_LABELS = ["depth", "token", "confidence c", "value V=∏c", "top-m 选中"]
CELLS = {
    "depth":        ["0", "1", "1", "2", "2", "2", "2"],
    "token":        ["4", "3", "5", "1", "0", "1", "0"],
    "confidence c": ["1.0", "0.354", "0.212", "0.304", "0.291", "0.354", "0.342"],
    "value V=∏c":   ["1.0", "0.354", "0.212", "0.108", "0.103", "0.075", "0.073"],
    "top-m 选中":    ["yes", "yes", "yes", "yes", "no", "no", "no"],
}
SELECTED_ROW = "top-m 选中"
STATUS = {"top-m 选中": ["selected", "selected", "selected", "selected", "reject", "reject", "reject"]}
COLOR = {"selected": ("#dcfce7", "#15803d"), "reject": ("#fee2e2", "#b91c1c")}
CALLOUT_COL = 5  # d2_t1b: 局部 c 更高却落选

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 130, 148, 44, 40, 100, 30
w = PAD * 2 + LABEL_W + COL_W * len(NODES)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 70
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(NODES))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-4}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+18}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(NODES):
    x = col_x[j]
    lines = name.split("\n")
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    ny = TOP + (HEADER_H-6)/2 - (len(lines)-1)*7 + 4
    for k, line in enumerate(lines):
        L.append(f'<text x="{x+(COL_W-8)/2}" y="{ny+k*14}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="white" '
                  f'font-weight="bold">{esc(line)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(NODES)):
        cx = col_x[j]
        text = CELLS[row][j]
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        disp = {"yes": "✓ 选中", "no": "✗ 落选"}.get(text, text)
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="{text_fill}" '
                  f'{weight_attr}>{esc(disp)}</text>')

callout_x = col_x[CALLOUT_COL] + (COL_W-8)/2
callout_y = row_y[-1] + ROW_H + 26
L.append(f'<line x1="{callout_x}" y1="{row_y[-1]+ROW_H}" x2="{callout_x}" y2="{callout_y-14}" '
          'stroke="#b91c1c" stroke-width="1.5" stroke-dasharray="3,3" marker-end="url(#a)"/>')
L.append(f'<text x="{PAD}" y="{callout_y+4}" font-family="sans-serif" font-size="12" '
          f'fill="#b91c1c">{esc("d2_t1b 局部 c=0.354 更高，但路径 V=0.075 更低 → 落选：排序看全局 V，不看局部 c")}</text>')
foot_y = callout_y + 26
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("绿=top-m 选中（4 个）,红=落选；选中集对父指针封闭 → 连通子树")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig36-5-value-rerank.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
