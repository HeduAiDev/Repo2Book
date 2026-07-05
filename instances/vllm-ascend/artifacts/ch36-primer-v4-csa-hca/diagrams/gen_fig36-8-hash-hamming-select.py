#!/usr/bin/env python3
"""state-table 模板改造:KVComp 用汉明距离 top-k 选块,must_select 强制并入
首块(sink)与最近块(recent)。列=6 个候选块,行=汉明距离/top-k命中/强制选中/最终入选。
数字来自 traces/kvcomp_hash.json。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "KVComp 汉明选块:top-k 命中 ∪ must_select 强制,并集保 sink+recent 不漏"
SUBTITLE = "dim_in=8, hash_bits=8, top_k=2, num_blocks=6, must_select_blocks=[0,-1](首块+末块)"

COLS = ["块0", "块1", "块2", "块3", "块4", "块5"]
ROW_LABELS = ["汉明距离", "汉明 top-2 命中", "must-select 强制", "最终入选"]
CELLS = {
    "汉明距离":        ["5", "3", "5", "0", "4", "4"],
    "汉明 top-2 命中":  ["否", "是", "否", "是", "否", "否"],
    "must-select 强制": ["是(sink)", "否", "否", "否", "否", "是(recent)"],
    "最终入选":        ["是", "是", "否", "是", "否", "是"],
}
STATUS_ROW = "最终入选"
STATUS = {
    "最终入选": ["sel", "sel", "drop", "sel", "drop", "sel"],
    "汉明距离": ["hi" if v in ("5",) else ("lo" if v == "0" else None) for v in CELLS["汉明距离"]],
}
COLOR = {"sel": ("#ecfdf5", "#047857"), "drop": ("#f1f5f9", "#94a3b8"),
         "hi": ("#fee2e2", "#b91c1c"), "lo": ("#ecfdf5", "#047857")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 190, 128, 46, 36, 100, 30
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 100 + PAD
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
    sel = CELLS["最终入选"][j] == "是"
    fill = "#059669" if sel else "#94a3b8"
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              f'fill="{fill}" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        text = CELLS[row][j]
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        fs = "12.5" if len(text) <= 3 else "11"
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs}" fill="{text_fill}" '
                  f'{weight_attr}>{esc(text)}</text>')

callout_y = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 26
box_w = w - PAD * 2
L.append(f'<rect x="{PAD}" y="{callout_y}" width="{box_w}" height="66" rx="6" '
          'fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+16}" y="{callout_y+22}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#92400e">块3 汉明距 0(与 query 指纹几乎重合)自然入 top-2;块0 距 5、块5 距 4 本会被淘汰</text>')
L.append(f'<text x="{PAD+16}" y="{callout_y+42}" font-family="sans-serif" font-size="11.5" '
          f'fill="#92400e">但 must_select=[0,-1] 把 sink 首块与 recent 末块强制拉回 —— 最终入选 = 汉明 top-2 ∪ 强制块</text>')
L.append(f'<text x="{PAD+16}" y="{callout_y+60}" font-family="sans-serif" font-size="11.5" '
          f'fill="#92400e">选块只需 XOR + popcount 位运算(每块指纹 1 字节)替代逐维浮点内积,近似检索的粗糙不会漏掉关键块</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig36-8-hash-hamming-select.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
