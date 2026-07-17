#!/usr/bin/env python3
"""flow 模板:别名传播图。v1/v2 是唯二真 alloc(第0列);v3/v4/v5 是 scf.for 迭代参数
经 join 传播别名集(第1列,v4 由两条边汇合);v6 是 scf.yield 返回值(第2列)。
全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "别名传播:local_alloc 造新 buffer, scf.for/scf.yield 只传播别名集"

# (col, row, name, op, alias_set_str, fill, stroke)
NODES = {
    "v1": (0, 0, "v1", "local_alloc", "{v1}", "#dcfce7", "#15803d"),
    "v2": (0, 2, "v2", "local_alloc", "{v2}", "#dcfce7", "#15803d"),
    "v3": (1, 0, "v3", "scf.for 迭代参数", "{v1}", "#dbeafe", "#1e40af"),
    "v4": (1, 1, "v4", "scf.for 迭代参数", "{v1, v2}", "#fee2e2", "#b91c1c"),
    "v5": (1, 2, "v5", "scf.for 迭代参数", "{v2}", "#dbeafe", "#1e40af"),
    "v6": (2, 0, "v6", "scf.yield(返回 v3)", "{v1}", "#ede9fe", "#6d28d9"),
}
EDGES = [("v1", "v3"), ("v1", "v4"), ("v2", "v4"), ("v2", "v5"), ("v3", "v6")]

BOX_W, BOX_H = 168, 78
COL_GAP, ROW_GAP, PAD, TOP = 130, 34, 40, 90
n_cols = 3
n_rows = 3
w = PAD * 2 + BOX_W * n_cols + COL_GAP * (n_cols - 1)
h = TOP + BOX_H * n_rows + ROW_GAP * (n_rows - 1) + PAD + 46

def pos(col, row):
    x = PAD + col * (BOX_W + COL_GAP)
    y = TOP + row * (BOX_H + ROW_GAP)
    return x, y

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-12}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>']

# column headers
COL_LABEL = ["alloc(唯二真 buffer)", "scf.for 迭代参数(join 传播)", "scf.yield"]
for c in range(n_cols):
    x, _ = pos(c, 0)
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11.5" fill="#64748b">{esc(COL_LABEL[c])}</text>')

# edges first (under nodes)
for src, dst in EDGES:
    sc, sr, *_ = NODES[src]
    dc, dr, *_ = NODES[dst]
    sx, sy = pos(sc, sr)
    dx, dy = pos(dc, dr)
    x1, y1 = sx + BOX_W, sy + BOX_H / 2
    x2, y2 = dx, dy + BOX_H / 2
    mx = (x1 + x2) / 2
    L.append(f'<path d="M {x1} {y1} C {mx} {y1}, {mx} {y2}, {x2} {y2}" '
              'fill="none" stroke="#334155" stroke-width="1.5" marker-end="url(#a)" opacity="0.85"/>')

# nodes
for key, (col, row, name, op, alias, fill, stroke) in NODES.items():
    x, y = pos(col, row)
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="9" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{x+16}" y="{y+24}" font-family="sans-serif" font-size="15" '
              f'font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<text x="{x+16}" y="{y+44}" font-family="sans-serif" font-size="11" '
              f'fill="#334155">{esc(op)}</text>')
    L.append(f'<text x="{x+16}" y="{y+64}" font-family="sans-serif" font-size="13" '
              f'font-weight="bold" fill="{stroke}">{esc("别名集 " + alias)}</text>')

foot_y = h - 14
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc("v4 同时挂在 v1、v2 两条边下,join=并集,别名集扩到 {v1, v2};其余节点各只随一条来源边,别名集基数为 1。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-alias-scf-for.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
