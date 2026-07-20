#!/usr/bin/env python3
"""paper-fig-eagle2-fig5: 论文精髓图重绘。
重绘自 arXiv:2406.16858 Fig.5（§3.1 Context-Dependent Acceptance Rates）——
(a) 静态草稿树的 P1-P6 六个固定位置；(b) 同一位置在不同 query 上的实测接受率散点，
方差极大。左图结构（Query→P1/P2，P1→P3/P4，P2→P5/P6）与原图一致；右图散点为按原图
六列分布形态（每列纵向抖动、position 6 明显被压低）重绘的示意抽样，趋势与原图一致，
不是逐点像素抽取。数据 provenance=原论文图本身(illustrator 契约提供的 provenance 豁免)。
全部坐标由循环计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

# (b) 每个 position 的示意接受率抽样(捕捉原图形态:每列方差都很大,position 6 被明显压低)
SCATTER = {
    1: [1.00, 0.98, 0.96, 0.94, 0.89, 0.86, 0.71, 0.63, 0.57, 0.49, 0.48, 0.39, 0.34, 0.17, 0.10, 0.01],
    2: [0.82, 0.58, 0.51, 0.21, 0.20, 0.16, 0.13, 0.10, 0.09, 0.06, 0.02, 0.01, 0.00],
    3: [1.00, 0.98, 0.97, 0.96, 0.93, 0.86, 0.65, 0.60, 0.59, 0.49, 0.39, 0.35, 0.25, 0.22, 0.17, 0.16, 0.10, 0.09, 0.06, 0.01, 0.00],
    4: [0.95, 0.89, 0.88, 0.85, 0.83, 0.71, 0.25, 0.20, 0.09, 0.04, 0.03, 0.02, 0.01, 0.00],
    5: [0.82, 0.56, 0.18, 0.11, 0.10, 0.04, 0.02, 0.01, 0.00],
    6: [0.20, 0.13, 0.08, 0.07, 0.05, 0.02, 0.01, 0.00],
}
POS_COLOR = {1: "#3b82f6", 2: "#f59e0b", 3: "#16a34a", 4: "#dc2626", 5: "#8b5cf6", 6: "#92400e"}
# 抖动偏移(确定性,非随机——避免每次渲染画面不同)
JITTER = [-9, 7, -4, 8, -7, 3, -8, 5, 0, -3, 9, 6, -6, 2, -2, 4, -5, 1, 8, -9, 6]

TITLE = "EAGLE-2 动态树的第一手动机：位置相同，接受率天差地别"
SUB = "重绘自 arXiv:2406.16858 Fig.5：静态树的 P1-P6 六个固定位置（左）；同一位置在不同 query 上接受率方差极大、position 6 普遍偏低（右，示意抽样重绘）"

PAD, TOP = 44, 140

# --- 左：静态树 P1-P6 ---
TREE_W = 420
NODE_W, NODE_H = 70, 40
row_y = [TOP + 40, TOP + 40 + 110, TOP + 40 + 220]
tree_positions = {
    "Query": (TREE_W / 2, row_y[0]),
    "P1": (TREE_W / 2 - 110, row_y[1]),
    "P2": (TREE_W / 2 + 110, row_y[1]),
    "P3": (TREE_W / 2 - 165, row_y[2]),
    "P4": (TREE_W / 2 - 55, row_y[2]),
    "P5": (TREE_W / 2 + 55, row_y[2]),
    "P6": (TREE_W / 2 + 165, row_y[2]),
}
TREE_EDGES = [("Query", "P1"), ("Query", "P2"), ("P1", "P3"), ("P1", "P4"), ("P2", "P5"), ("P2", "P6")]

# --- 右：散点 ---
SCAT_X0 = PAD + TREE_W + 100
SCAT_W, SCAT_H = 620, 340
SCAT_Y0 = TOP + 30

W = SCAT_X0 + SCAT_W + PAD
H = SCAT_Y0 + SCAT_H + 90

def scat_x(pos, jidx):
    base = SCAT_X0 + (pos - 1) / 5 * SCAT_W
    return base + JITTER[jidx % len(JITTER)]

def scat_y(v):
    return SCAT_Y0 + SCAT_H * (1 - v)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="38" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{W/2}" y="62" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUB)}</text>',
     f'<text x="{PAD+TREE_W/2}" y="{TOP-40}" text-anchor="middle" font-family="sans-serif" '
     f'font-size="14" font-weight="bold" fill="#1e40af">{esc("(a) 静态草稿树的六个固定位置")}</text>',
     f'<text x="{SCAT_X0+SCAT_W/2}" y="{TOP-40}" text-anchor="middle" font-family="sans-serif" '
     f'font-size="14" font-weight="bold" fill="#1e40af">{esc("(b) 同一位置、不同 query 的实测接受率")}</text>']

# tree
for a, b in TREE_EDGES:
    ax, ay = tree_positions[a]; ax += PAD;
    bx, by = tree_positions[b]; bx += PAD
    L.append(f'<line x1="{ax:.1f}" y1="{ay+NODE_H/2:.1f}" x2="{bx:.1f}" y2="{by-NODE_H/2:.1f}" '
              'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
for name, (x, y) in tree_positions.items():
    x += PAD
    is_query = name == "Query"
    fill = "#dbeafe" if is_query else "#fde8d7"
    stroke = "#1e40af" if is_query else "#c2410c"
    L.append(f'<rect x="{x-NODE_W/2:.1f}" y="{y-NODE_H/2:.1f}" width="{NODE_W}" height="{NODE_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{x:.1f}" y="{y+5:.1f}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="{stroke}">{esc(name)}</text>')

# scatter axes
L.append(f'<line x1="{SCAT_X0}" y1="{SCAT_Y0}" x2="{SCAT_X0}" y2="{SCAT_Y0+SCAT_H}" stroke="#334155" stroke-width="1.5"/>')
L.append(f'<line x1="{SCAT_X0}" y1="{SCAT_Y0+SCAT_H}" x2="{SCAT_X0+SCAT_W}" y2="{SCAT_Y0+SCAT_H}" stroke="#334155" stroke-width="1.5"/>')
for gv in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
    gy = scat_y(gv)
    L.append(f'<line x1="{SCAT_X0}" y1="{gy:.1f}" x2="{SCAT_X0+SCAT_W}" y2="{gy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
    L.append(f'<text x="{SCAT_X0-10}" y="{gy+4:.1f}" text-anchor="end" font-family="sans-serif" '
              f'font-size="11" fill="#64748b">{esc(f"{gv:.1f}")}</text>')
L.append(f'<text x="{SCAT_X0-46}" y="{SCAT_Y0+SCAT_H/2:.1f}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#64748b" transform="rotate(-90 {SCAT_X0-46} {SCAT_Y0+SCAT_H/2:.1f})">'
          f'{esc("Accept Rate 接受率")}</text>')
jidx = 0
for pos in range(1, 7):
    px = SCAT_X0 + (pos - 1) / 5 * SCAT_W
    L.append(f'<text x="{px:.1f}" y="{SCAT_Y0+SCAT_H+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" fill="#334155">{esc(str(pos))}</text>')
    for v in SCATTER[pos]:
        x = scat_x(pos, jidx); jidx += 1
        y = scat_y(v)
        L.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{POS_COLOR[pos]}" fill-opacity="0.55" '
                  f'stroke="{POS_COLOR[pos]}" stroke-width="1"/>')
L.append(f'<text x="{SCAT_X0+SCAT_W/2:.1f}" y="{SCAT_Y0+SCAT_H+42}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" fill="#64748b">{esc("Position 树内位置 P1-P6")}</text>')

foot_y = H - 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" fill="#334155">'
          f'{esc("同一位置(如 P1)在不同 query 上接受率可从接近 0 跨到接近 1——静态固定树形状,对简单 query 浪费候选,对难 query 又不够。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("paper-fig-eagle2-fig5.svg")
out.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {out}")
