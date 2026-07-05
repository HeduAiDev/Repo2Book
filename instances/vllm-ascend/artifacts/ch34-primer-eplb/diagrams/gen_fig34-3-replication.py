#!/usr/bin/env python3
"""fig34-3-replication — state-table 模板：贪心复制两轮的状态追踪。
列=轮次，行=选中专家/加副本前后平均热度/副本数/剩余名额。
"摊薄后平均热度"行高亮（每轮减半），改造自 example-softmax-trace.py。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "贪心复制冗余专家：两轮状态追踪"
SUBTITLE = "num_redundant=2 个冗余名额，每轮把名额发给『当前单副本平均热度最高』的专家"
COLS = ["轮次 1 — 选中 expert 3", "轮次 2 — 选中 expert 4"]
ROW_LABELS = ["加副本前平均热度", "副本数 k→k+1", "摊薄后平均热度", "剩余冗余名额"]
CELLS = {
    "加副本前平均热度": ["60.0", "55.0"],
    "副本数 k→k+1":     ["0 → 1", "0 → 1"],
    "摊薄后平均热度":     ["60.0 / 2 = 30.0", "55.0 / 2 = 27.5"],
    "剩余冗余名额":       ["2 → 1", "1 → 0"],
}
HIGHLIGHT_ROW = "摊薄后平均热度"
STATUS = {"摊薄后平均热度": ["changed", "changed"]}
COLOR = {"changed": ("#fee2e2", "#b91c1c")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 190, 260, 54, 40, 108, 34
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 70
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="4" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+5}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" fill="{text_fill}" '
                  f'{weight_attr}>{esc(CELLS[row][j])}</text>')

table_bottom = row_y[-1] + ROW_H
box_y = table_bottom + 24
box_h = 46
L.append(f'<rect x="{PAD}" y="{box_y}" width="{w-PAD*2}" height="{box_h}" rx="8" '
          'fill="#fff7ed" stroke="#d97706" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+16}" y="{box_y+28}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#92400e">'
          f'结果：replicas_of={{3:[8], 4:[9]}} — 2 个冗余名额全部给了最热的专家 3、4，各减半</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig34-3-replication.svg")
out.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {out}")
