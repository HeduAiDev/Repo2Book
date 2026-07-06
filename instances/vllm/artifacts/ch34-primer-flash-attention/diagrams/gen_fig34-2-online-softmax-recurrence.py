#!/usr/bin/env python3
"""state-table 模板:online-softmax 单遍递推,x=[1,3,2,5],四轮 (m,d) 演化。
关键:第 4 轮 max 从 3 跳到 5,旧 d 被 exp(3-5)=0.1353 缩小——高亮该列。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "online-softmax 单遍递推 — x = [1, 3, 2, 5]"
SUBTITLE = "每列处理一个新元素后的 (m,d);末值 m=5, d=1.2034 与三遍 safe-softmax 逐位相等"
COLS = ["j=1  x=1", "j=2  x=3", "j=3  x=2", "j=4  x=5"]
ROW_LABELS = ["m: 旧→新", "rescale=exp(m_old-m_new)", "d_before", "d_new"]
CELLS = {
    "m: 旧→新":     ["-inf → 1", "1 → 3", "3 → 3\n(不变)", "3 → 5\n(跳升)"],
    "rescale=exp(m_old-m_new)": ["n/a\n(首元素)", "0.1353", "1.0", "0.1353"],
    "d_before": ["0", "1.0", "1.1353", "1.5032"],
    "d_new":    ["1.0", "1.1353", "1.5032", "1.2034"],
}
HIGHLIGHT_ROW = "rescale=exp(m_old-m_new)"
STATUS = {"rescale=exp(m_old-m_new)": ["stable", "changed", "stable", "changed"]}
COLOR = {"stable": ("#ecfdf5", "#047857"), "changed": ("#fee2e2", "#b91c1c")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 210, 190, 60, 34, 96, 30
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 80
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    is_last = j == len(COLS) - 1
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              f'fill="{"#dc2626" if is_last else "#3b82f6"}" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        lines = CELLS[row][j].split("\n")
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        n = len(lines)
        y0 = ry + ROW_H / 2 - (n - 1) * 8 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+(COL_W-8)/2}" y="{y0+k*16}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12" fill="{text_fill}" '
                      f'{weight_attr}>{esc(line)}</text>')

    if i < len(ROW_LABELS) - 1:
        pass

final_y = row_y[-1] + ROW_H + 30
L.append(f'<rect x="{PAD}" y="{final_y}" width="{w-2*PAD}" height="34" rx="6" '
          'fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.5"/>')
FINAL = "末值 m=5, d=1.2034  ==  三遍 safe-softmax 参照 d_V=1.2034(逐元素差 0.0)"
L.append(f'<text x="{w/2}" y="{final_y+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#1e3a8a">{esc(FINAL)}</text>')

foot_y = h - 16
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("绿=rescale≈1.0(max 未变),红=rescale<1.0(max 跳升,旧累计被缩小)")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig34-2-online-softmax-recurrence.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
