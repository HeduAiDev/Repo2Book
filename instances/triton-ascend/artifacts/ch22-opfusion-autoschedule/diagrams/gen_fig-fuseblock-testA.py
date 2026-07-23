#!/usr/bin/env python3
"""flow 模板：fuseBlock 在 @testA（ShallowCV，静态 7x7）上把 9 个重要算子沿 6 条候选边
并查集合并，出组过滤后只留 1 个含 matmul 的融合组（5 op），另 2 个纯 vector 分量被踢回。
节点/边取自 traces/fuseblock_testA.md（对应 FileCheck 断言）。全坐标计算，零魔数。"""
import math
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

def text_w(s, size):
    """粗略估算文本像素宽：CJK 全宽按 size 计，ASCII/半角按 0.58*size 计。"""
    total = 0.0
    for ch in s:
        total += size if ord(ch) > 0x2e80 else size * 0.58
    return total

TITLE = "fuseBlock 在 @testA 上的并查集合并"
SUBTITLE = "9 个重要算子、6 条候选边（依赖方向）；ShallowCV 下 matmul↔vector 全通过，出组只留含 matmul 的组"
FOOTER = ("9 个重要算子、6 条候选边 → 并查集并出 3 个连通分量（1 组保留、2 组被踢出）；"
          "checkGroupRequirements 只放行含 matmul 的组，最终 1 个融合组（5 op）。")

# 节点：id -> (label, kind)  kind: vector | cube
NODES = {
    "n3":  ("%3\nceil", "vector"),
    "n5":  ("%5\nadd", "vector"),
    "n7":  ("%7\nlog", "vector"),
    "n9":  ("%9\nmatmul", "cube"),
    "n11": ("%11\nceil", "vector"),
    "n13": ("%13\nbcast", "vector"),
    "n17": ("%17\nbcast", "vector"),
    "n15": ("%15\nabs", "vector"),
    "n19": ("%19\ntranspose", "vector"),
}
EDGES = [("n3", "n5"), ("n5", "n7"), ("n7", "n9"), ("n7", "n11"),
         ("n13", "n17"), ("n15", "n19")]

COLOR = {"vector": ("#dbeafe", "#1d4ed8", "#1e3a8a"),
         "cube": ("#fef3c7", "#b45309", "#78350f")}

BOX_W, BOX_H = 96, 52
COL_GAP = 130
PAD = 46
LABEL_GAP = 42   # 组框顶部到内部节点上边缘的留白（要能放下组标签）
GROUP_GAP = 46   # 相邻组之间的垂直间距

chain_x0 = PAD + 40
col_x = [chain_x0 + i * COL_GAP for i in range(4)]  # n3,n5,n7,{n9/n11}

# --- 纵向布局：逐组累加，保证组间/组与标题间不重叠 ---
title_bottom = 62
fanout = 40  # n9/n11 相对主行的上下偏移
# 主行（n3/n5/n7）y：需保证 n9 所在的组框顶部（本组最高点 - 半框 - LABEL_GAP）
# 与副标题之间留出至少一行文字的间距，避免组标签压到副标题。
accepted_row = title_bottom + fanout + BOX_H / 2 + LABEL_GAP + 26
n9_y = accepted_row - fanout
n11_y = accepted_row + fanout
accepted_bottom = n11_y + BOX_H / 2 + 16  # 组框底边

reject1_row = accepted_bottom + GROUP_GAP + LABEL_GAP + BOX_H / 2
reject1_bottom = reject1_row + BOX_H / 2 + 16

reject2_row = reject1_bottom + GROUP_GAP + LABEL_GAP + BOX_H / 2
reject2_bottom = reject2_row + BOX_H / 2 + 16

pos = {
    "n3": (col_x[0], accepted_row), "n5": (col_x[1], accepted_row),
    "n7": (col_x[2], accepted_row),
    "n9": (col_x[3], n9_y), "n11": (col_x[3], n11_y),
    "n13": (col_x[0], reject1_row), "n17": (col_x[1], reject1_row),
    "n15": (col_x[0], reject2_row), "n19": (col_x[1], reject2_row),
}

legend_y = reject2_bottom + 54
foot_y = legend_y + 40
diagram_w = col_x[3] + BOX_W / 2 + PAD + 40
subtitle_w = PAD + text_w(SUBTITLE, 12) + PAD
footer_w = PAD + text_w(FOOTER, 11) + PAD
w = max(diagram_w, subtitle_w, footer_w)
h = foot_y + 30

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="26" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="48" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

def group_box(ids):
    xs_ = [pos[i][0] for i in ids]
    ys_ = [pos[i][1] for i in ids]
    x0 = min(xs_) - BOX_W / 2 - 16
    x1 = max(xs_) + BOX_W / 2 + 16
    y0 = min(ys_) - BOX_H / 2 - LABEL_GAP + 16
    y1 = max(ys_) + BOX_H / 2 + 16
    return x0, y0, x1 - x0, y1 - y0

gx, gy, gw, gh = group_box(["n3", "n5", "n7", "n9", "n11"])
L.append(f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" rx="14" '
          'fill="#ecfdf5" stroke="#047857" stroke-width="2.5" stroke-dasharray="7,4"/>')
L.append(f'<text x="{gx+10}" y="{gy-12}" font-family="sans-serif" font-size="12" '
          'font-weight="bold" fill="#047857">保留 → @testA_0（5 op，含 matmul）</text>')

for grp, label in [(["n13", "n17"], "matmulCount=0 → 踢出"),
                     (["n15", "n19"], "matmulCount=0 → 踢出")]:
    gx2, gy2, gw2, gh2 = group_box(grp)
    L.append(f'<rect x="{gx2}" y="{gy2}" width="{gw2}" height="{gh2}" rx="14" '
              'fill="#fef2f2" stroke="#b91c1c" stroke-width="2" stroke-dasharray="6,4"/>')
    L.append(f'<text x="{gx2+10}" y="{gy2-12}" font-family="sans-serif" font-size="11.5" '
              f'font-weight="bold" fill="#b91c1c">{esc(label)}</text>')

# 边（先画线，再画节点盖住线端）
for u, v in EDGES:
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    ux, uy = dx / dist, dy / dist
    sx1, sy1 = x1 + ux * BOX_W / 2, y1 + uy * BOX_H / 2
    sx2, sy2 = x2 - ux * BOX_W / 2, y2 - uy * BOX_H / 2
    L.append(f'<line x1="{sx1:.1f}" y1="{sy1:.1f}" x2="{sx2:.1f}" y2="{sy2:.1f}" '
              'stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')

# 节点
for nid, (label, kind) in NODES.items():
    x, y = pos[nid]
    fill, stroke, tf = COLOR[kind]
    lines = label.split("\n")
    L.append(f'<rect x="{x-BOX_W/2}" y="{y-BOX_H/2}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    y0 = y - (len(lines) - 1) * 8
    for k, line in enumerate(lines):
        weight = 'font-weight="bold" ' if k == 0 else ''
        L.append(f'<text x="{x}" y="{y0+k*16+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" fill="{tf}" '
                  f'{weight}>{esc(line)}</text>')

# 图例
L.append(f'<rect x="{PAD}" y="{legend_y}" width="18" height="18" rx="3" fill="#dbeafe" stroke="#1d4ed8" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+26}" y="{legend_y+14}" font-family="sans-serif" font-size="11.5" fill="#334155">vector 算子</text>')
L.append(f'<rect x="{PAD+130}" y="{legend_y}" width="18" height="18" rx="3" fill="#fef3c7" stroke="#b45309" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+156}" y="{legend_y+14}" font-family="sans-serif" font-size="11.5" fill="#334155">cube 算子（matmul）</text>')

L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc(FOOTER)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-fuseblock-testA.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
