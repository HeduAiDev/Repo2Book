#!/usr/bin/env python3
"""state-table 模板改造:n=4 bitonic sort 用 3 次数据无关 CAS 把 [3,1,2,0] 收敛为 [0,1,2,3]。
数据来自 traces/algorithms.json bitonic_sort。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "排序网络定死,与数值无关:n=4 用 3 次 CAS 收敛"
SUBTITLE = "n_dims=log2(4)=2,顶层 2 个阶段铺 1+2=3 次 compare-and-swap(实测)"
COLS = ["初始", "CAS#1 (i=1)", "CAS#2 (i=0)", "CAS#3 (i=1)"]
STATE = [
    ("初始", [3, 1, 2, 0]),
    ("CAS#1 (i=1)", [1, 3, 2, 0]),
    ("CAS#2 (i=0)", [1, 0, 2, 3]),
    ("CAS#3 (i=1)", [0, 1, 2, 3]),
]
# 每一步相对上一步变化的下标(用来高亮改变的格子)
CHANGED_IDX = [set(), {0, 1}, {0, 1}, {0, 1}]

CELL_W, CELL_GAP, GROUP_GAP, PAD, TOP = 46, 6, 60, 40, 130
N = 4
group_w = CELL_W * N + CELL_GAP * (N - 1)
w = PAD * 2 + group_w * len(STATE) + GROUP_GAP * (len(STATE) - 1)
h = TOP + 90 + 60

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{40}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{60}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

group_x = [PAD + i * (group_w + GROUP_GAP) for i in range(len(STATE))]

for gi, (label, arr) in enumerate(STATE):
    gx = group_x[gi]
    L.append(f'<text x="{gx + group_w/2}" y="{TOP - 14}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#0f172a">{esc(label)}</text>')
    for k in range(N):
        cx = gx + k * (CELL_W + CELL_GAP)
        cy = TOP
        changed = k in CHANGED_IDX[gi]
        fill = "#fee2e2" if changed else "#e2e8f0"
        stroke = "#b91c1c" if changed else "#64748b"
        tw = "bold" if changed else "normal"
        tfill = "#b91c1c" if changed else "#0f172a"
        L.append(f'<rect x="{cx}" y="{cy}" width="{CELL_W}" height="{CELL_W}" rx="6" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if changed else 1}"/>')
        L.append(f'<text x="{cx+CELL_W/2}" y="{cy+CELL_W/2+6}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="16" font-weight="{tw}" '
                  f'fill="{tfill}">{esc(str(arr[k]))}</text>')
        L.append(f'<text x="{cx+CELL_W/2}" y="{cy+CELL_W+16}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10" fill="#94a3b8">idx{k}</text>')
    if gi < len(STATE) - 1:
        ay = TOP + CELL_W / 2
        x1 = gx + group_w + 6
        x2 = group_x[gi + 1] - 6
        L.append(f'<line x1="{x1}" y1="{ay}" x2="{x2}" y2="{ay}" stroke="#64748b" '
                  f'stroke-width="2" marker-end="url(#a)"/>')

foot_y = TOP + CELL_W + 50
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">红=本次 CAS 改变的位置;每次调用只由块长(编译期常量)决定谁跟谁比,与数值无关</text>')
L.append(f'<text x="{PAD}" y="{foot_y+18}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">3 次调用后 [3,1,2,0] -&gt; [0,1,2,3];调用数 = n_dims(n_dims+1)/2 = 2x3/2 = 3</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch09-bitonic-n4.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
