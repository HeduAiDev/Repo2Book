#!/usr/bin/env python3
"""state-table 模板:逐 token 接受/拒绝判定表(玩具分布 p,q over {A,B,C,D})。
高亮 token C(q=0.3>p=0.1,被过度提议,只以 p/q=0.333 存活);其余行(q<=p)恒接受。
数字全部来自 explainer/traces/accept_reject.json。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "投机采样逐 token 接受判定 — 接受概率 = min(1, p(x)/q(x))"
SUBTITLE = "玩具分布 p=[0.5,0.3,0.1,0.1], q=[0.4,0.2,0.3,0.1] over vocab {A,B,C,D}"
COLS = ["p(x)", "q(x)", "q<=p ?", "接受概率 min(1,p/q)"]
ROW_LABELS = ["A", "B", "C", "D"]
CELLS = {
    "A": ["0.5", "0.4", "yes", "1.0"],
    "B": ["0.3", "0.2", "yes", "1.0"],
    "C": ["0.1", "0.3", "no", "0.333"],
    "D": ["0.1", "0.1", "yes", "1.0"],
}
HIGHLIGHT = {"A": "ok", "B": "ok", "C": "warn", "D": "ok"}
COLOR = {"ok": ("#ecfdf5", "#047857"), "warn": ("#fef2f2", "#b91c1c")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 70, 190, 52, 34, 96, 34
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 46
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
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    status = HIGHLIGHT[row]
    fill, stroke = COLOR[status]
    L.append(f'<rect x="{PAD}" y="{ry+4}" width="{LABEL_W-10}" height="{ROW_H-8}" rx="4" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{PAD+(LABEL_W-10)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="15" font-weight="bold" '
              f'fill="{stroke}">{esc(row)}</text>')
    for j in range(len(COLS)):
        cx = col_x[j]
        val = CELLS[row][j]
        is_accept_col = (j == len(COLS) - 1)
        text_fill = stroke if (is_accept_col and status == "warn") else "#374151"
        weight = 'font-weight="bold" ' if (is_accept_col) else ''
        L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" '
                  f'fill="none" stroke="#e2e8f0"/>')
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" fill="{text_fill}" '
                  f'{weight}>{esc(val)}</text>')

foot1_y = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 26
L.append(f'<text x="{PAD}" y="{foot1_y}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#0f172a">beta = Sum_x min(p,q) = 0.8'
          f'  —— 蒙特卡洛 N=400000 次抽样经验接受频率 = 0.8</text>')
foot2_y = foot1_y + 20
L.append(f'<text x="{PAD}" y="{foot2_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">绿=q&lt;=p 恒被接受;红=C 被过度提议(q=0.3&gt;p=0.1),仅以 p/q=0.333 存活</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig33-accept-reject-per-token.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
