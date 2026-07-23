#!/usr/bin/env python3
"""fig-m2-flatten: before-after 模板（改造为树→叶子对比）。
左：andi 掩码树 broadcast(a&b)&c；右：collectAndLeaves 递归拍平出的 3 个叶子。
broadcast(andi) 按分配律下推成两个 broadcast 因子。全坐标由常量计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

W, H = 980, 480
NODE_W, NODE_H = 150, 40
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>']

L.append(f'<text x="30" y="34" font-family="sans-serif" font-size="17" font-weight="bold" '
         f'fill="#0f172a">{esc("collectAndLeaves：andi 掩码树递归拍平成叶子集合")}</text>')
L.append(f'<text x="30" y="54" font-family="sans-serif" font-size="12" fill="#64748b">'
         f'{esc("DiscreteMaskAccessConversionPass.cpp:L78-L102")}</text>')

# ---- Left panel: input tree ----
LEFT_CX = 250
TOP = 110

def node(cx, cy, label, fill, stroke, w=NODE_W, h=NODE_H, bold=False):
    L.append(f'<rect x="{cx-w/2}" y="{cy-h/2}" width="{w}" height="{h}" rx="7" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    fw = 'font-weight="bold" ' if bold else ''
    L.append(f'<text x="{cx}" y="{cy+5}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="12.5" {fw}fill="#0f172a">{esc(label)}</text>')

def edge(x1, y1, x2, y2):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
             'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

L.append(f'<text x="{LEFT_CX}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="#1e40af">{esc("输入：1 棵 andi 树（含 1 个 broadcast）")}</text>')

# root: andi(broadcast(a&b), c)
root_y = TOP
node(LEFT_CX, root_y, "andi(_, c)", "#dbeafe", "#1e3a8a", bold=True)

bc_y = root_y + 80
bc_x = LEFT_CX - 90
c_x = LEFT_CX + 90
node(bc_x, bc_y, "broadcast(andi(a,b))", "#fef3c7", "#b45309", w=200)
node(c_x, bc_y, "c", "#e2e8f0", "#334155")
edge(LEFT_CX - 30, root_y + NODE_H/2, bc_x + 10, bc_y - NODE_H/2 + 6)
edge(LEFT_CX + 30, root_y + NODE_H/2, c_x - 10, bc_y - NODE_H/2 + 6)

inner_y = bc_y + 80
a_x = bc_x - 55
b_x = bc_x + 55
node(a_x, inner_y, "a", "#e2e8f0", "#334155", w=90)
node(b_x, inner_y, "b", "#e2e8f0", "#334155", w=90)
edge(bc_x - 20, bc_y + NODE_H/2, a_x + 5, inner_y - NODE_H/2 + 6)
edge(bc_x + 20, bc_y + NODE_H/2, b_x - 5, inner_y - NODE_H/2 + 6)
L.append(f'<text x="{bc_x}" y="{bc_y+NODE_H/2+22}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#b45309">{esc("andi(a,b)")}</text>')

# ---- Middle: transform arrow ----
mid_x = W / 2
mid_y = (TOP + inner_y) / 2 + 20
edge(LEFT_CX + 260, mid_y, LEFT_CX + 380, mid_y)
L.append(f'<text x="{mid_x-10}" y="{mid_y-14}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#334155">{esc("collectAndLeaves()")}</text>')
L.append(f'<text x="{mid_x-10}" y="{mid_y+20}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#64748b">{esc("broadcast(andi) 分配律下推")}</text>')
L.append(f'<text x="{mid_x-10}" y="{mid_y+36}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#64748b">{esc("→ 2 个 broadcast 因子各自递归")}</text>')

# ---- Right panel: 3 leaves ----
RIGHT_CX = W - 240
L.append(f'<text x="{RIGHT_CX}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="#15803d">{esc("输出：3 个叶子（一次后序遍历各 push 一次）")}</text>')

LEAVES = ["bc(a)", "bc(b)", "c"]
leaf_gap = 90
leaf_y0 = TOP + 20
for i, lf in enumerate(LEAVES):
    ly = leaf_y0 + i * leaf_gap
    node(RIGHT_CX, ly, lf, "#dcfce7", "#15803d", w=180)
    L.append(f'<text x="{RIGHT_CX+120}" y="{ly+5}" font-family="sans-serif" font-size="12" '
             f'fill="#166534">{esc(f"leaves[{i}]")}</text>')

# ---- Caption ----
foot_y = H - 40
L.append(f'<rect x="30" y="{foot_y-30}" width="{W-60}" height="52" rx="8" '
         'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
L.append(f'<text x="46" y="{foot_y-8}" font-family="sans-serif" font-size="12.5" '
         f'fill="#334155">{esc("3 个 & 因子拍平成 3 个叶子；藏在 broadcast 里的 & 先按分配律下推，一个因子都不漏，")}</text>')
L.append(f'<text x="46" y="{foot_y+10}" font-family="sans-serif" font-size="12.5" '
         f'fill="#334155">{esc("好让每个因子被单独判连续 / 离散。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m2-flatten.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
