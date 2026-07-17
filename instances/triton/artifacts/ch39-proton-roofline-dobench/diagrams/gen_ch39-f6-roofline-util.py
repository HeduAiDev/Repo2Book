#!/usr/bin/env python3
"""ch39-f6-roofline-util: roofline util 判据(条形图 + state-table 混合)。
claim: util = max(compute屋顶, memory屋顶) / 实测时间——两个 ideal_time 谁大谁是瓶颈,决定优化方向。
数据来自 pin 真 viewer 喂 third_party/proton/test/example_cuda.json 的输出。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

KERNELS = [
    {"name": "foo0", "actual": 204.8, "compute": 50.6, "memory": 24.9, "util": 0.247},
    {"name": "foo1", "actual": 204.8, "compute": 30.3, "memory": 9.9, "util": 0.148},
]
SCALE_MAX = 220.0  # µs, axis upper bound
BAR_W = 420
ROW_H = 34
PAD = 40
LABEL_W = 150
HEAD_GAP = 30   # 面板标题与第一根条之间的留白(避免标注文字与标题相撞)

def bar_x(v):
    return LABEL_W + (v / SCALE_MAX) * BAR_W

BARSPEC = [
    ("actual", "实测时间", "#f97316"),
    ("compute", "compute 屋顶 ideal", "#3b82f6"),
    ("memory", "memory 屋顶 ideal", "#10b981"),
]
panel_h = HEAD_GAP + len(BARSPEC) * ROW_H
PANEL_GAP = 50

W = PAD * 2 + LABEL_W + BAR_W + 90
H = PAD + 40 + len(KERNELS) * (panel_h + PANEL_GAP) + 300

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="28" font-family="sans-serif" font-size="16" font-weight="bold" '
     f'fill="#1e40af">roofline: util = max(compute 屋顶 ideal, memory 屋顶 ideal) / 实测时间</text>',
     f'<text x="{PAD}" y="48" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'third_party/proton/test/example_cuda.json(仓库自带,无 GPU 跑通 pin 真 viewer)</text>']

y = PAD + 70
for k in KERNELS:
    which = "compute" if k["compute"] > k["memory"] else "memory"
    L.append(f'<text x="{PAD}" y="{y-10}" font-family="sans-serif" font-size="14" '
             f'font-weight="bold" fill="#0f172a">{esc(k["name"])} —— util = {k["util"]:.3f}'
             f'（{which}-bound）</text>')
    bars_top = y + HEAD_GAP
    for i, (key, label, color) in enumerate(BARSPEC):
        by = bars_top + i * ROW_H
        v = k[key]
        bw = bar_x(v) - LABEL_W
        L.append(f'<text x="{LABEL_W-14}" y="{by+ROW_H*0.6}" text-anchor="end" '
                 f'font-family="sans-serif" font-size="12" fill="#374151">{esc(label)}</text>')
        L.append(f'<rect x="{LABEL_W}" y="{by+4}" width="{BAR_W}" height="{ROW_H-14}" '
                 f'fill="#f1f5f9" stroke="#e2e8f0"/>')
        L.append(f'<rect x="{LABEL_W}" y="{by+4}" width="{bw}" height="{ROW_H-14}" fill="{color}"/>')
        L.append(f'<text x="{LABEL_W+bw+8}" y="{by+ROW_H*0.6}" font-family="sans-serif" '
                 f'font-size="12" font-weight="bold" fill="{color}">{v}µs</text>')
        if key == which:  # 瓶颈屋顶标注:只贴在该行数字右侧,不跨行画线
            L.append(f'<text x="{LABEL_W+bw+70}" y="{by+ROW_H*0.6}" font-family="sans-serif" '
                     f'font-size="11" font-weight="bold" fill="#1d4ed8">← 瓶颈屋顶(此处封顶)</text>')
    y += panel_h + PANEL_GAP

# --- numeric table (state-table style) ---
table_y = y - PANEL_GAP + 55
COLS = ["kernel", "实测时间(µs)", "compute 屋顶\nideal(µs)", "memory 屋顶\nideal(µs)", "max→瓶颈", "util = max/实测"]
FRAC = [0.09, 0.15, 0.17, 0.17, 0.27, 0.15]
col_x, acc = [], PAD
for f in FRAC:
    col_x.append(acc)
    acc += f * (W - 2*PAD)
COL_WS = [f * (W - 2*PAD) for f in FRAC]
HEADER_H = 40
ROWH2 = 44
L.append(f'<text x="{PAD}" y="{table_y-14}" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#1e40af">逐项数据(与 viewer 输出一致)</text>')
for j, c in enumerate(COLS):
    cx, cw = col_x[j], COL_WS[j]
    L.append(f'<rect x="{cx}" y="{table_y}" width="{cw-4}" height="{HEADER_H}" '
             f'fill="#3b82f6" stroke="#1e3a5f"/>')
    lines = c.split("\n")
    y0 = table_y + HEADER_H/2 - (len(lines)-1)*7 + 4
    for k, line in enumerate(lines):
        L.append(f'<text x="{cx+(cw-4)/2}" y="{y0+k*14}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="11" fill="white" '
                 f'font-weight="bold">{esc(line)}</text>')

rows = []
for k in KERNELS:
    which = "compute" if k["compute"] > k["memory"] else "memory"
    roof, other = max(k["compute"], k["memory"]), min(k["compute"], k["memory"])
    rows.append([k["name"], f'{k["actual"]}', f'{k["compute"]}', f'{k["memory"]}',
                 f'{which} ({roof}>{other})', f'{k["util"]:.3f}'])

for i, row in enumerate(rows):
    ry = table_y + HEADER_H + i*ROWH2
    fill = "#f8fafc" if i % 2 == 0 else "white"
    for j, val in enumerate(row):
        cx, cw = col_x[j], COL_WS[j]
        L.append(f'<rect x="{cx}" y="{ry}" width="{cw-4}" height="{ROWH2-4}" '
                 f'fill="{fill}" stroke="#e2e8f0"/>')
        fs = 10.5 if j == 4 else 11.5
        L.append(f'<text x="{cx+(cw-4)/2}" y="{ry+(ROWH2-4)/2+4}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="{fs}" fill="#1e293b">{esc(val)}</text>')

foot_y = table_y + HEADER_H + len(rows)*ROWH2 + 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" fill="#64748b">'
         f'两 kernel 的 util 均低于 0.25——离算力屋顶还远,优化方向指向「喂满 tensor core」而非省访存。</text>')
L.append('</svg>')
out = Path(__file__).with_name("ch39-f6-roofline-util.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
