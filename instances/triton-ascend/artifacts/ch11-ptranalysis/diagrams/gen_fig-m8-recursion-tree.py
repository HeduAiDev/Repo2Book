#!/usr/bin/env python3
"""fig-m8-recursion-tree: visitOperand 从 %27=addptr 展开的递归树（flow 模板）。
ptr 侧(%26 splat -> %arg1) 浅、depth=2；offset 侧在 %25 处再分两支：
%23 支(depth=5)与 %24 支(depth=6，全树最深，经 broadcast->addi->rem->addi->make_range)。
muli(%21)与 addi(%12)均为二元算子，各自还有一个标量 splat 操作数（%20=splat(%arg4)、
%11=splat(%9)），单独一条 splat 叶子车道（LANE_D）画出，与 %25 的双孩子模式一致。
全链三次 splat（%11/%20/%26）均在树上可见。全坐标由常量布局计算，零手写数据魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "从 %27=addptr 展开的 visitOperand 递归树"
SUBTITLE = "ptr 侧两跳到 base 指针；offset 侧在 %25 分两支，最深经 %24 六跳到 make_range 叶；muli(%21)/addi(%12) 各自还有一个标量 splat 操作数（%20/%11）——子树各自求解后由 addState 焊接"

BOX_W, BOX_H = 132, 46
HGAP, VGAP = 34, 62
PAD, TOP = 40, 108

# 纵向车道的 x 中心（常量布局，非数据）
LANE_A_X = 150     # ptr 侧
LANE_B_X = 560     # offset 侧 %23 支
LANE_C_X = 940     # offset 侧 %24 支
LANE_D_X = 1160    # 标量 splat 叶子专用车道（muli/addi 的第二个操作数）
DEPTH_COL_W = 110  # 右侧 depth 标注独立车道，避免与 lane D 节点框重叠
W = LANE_D_X + BOX_W / 2 + DEPTH_COL_W + PAD

# 每个节点: (label, lane_x, depth, kind)
NODES = {
    "root": ("%27\naddptr", (LANE_A_X + (LANE_B_X + LANE_C_X) / 2) / 2, 0, "root"),
    "n26": ("%26\nsplat", LANE_A_X, 1, "ptr"),
    "n25": ("%25\naddi", (LANE_B_X + LANE_C_X) / 2, 1, "offset"),
    "arg1": ("%arg1\n（base 指针叶）", LANE_A_X, 2, "leaf"),
    "n23": ("%23\nbroadcast", LANE_B_X, 2, "offset"),
    "n24": ("%24\nbroadcast", LANE_C_X, 2, "offset"),
    "n21": ("%21\nmuli", LANE_B_X, 3, "offset"),
    "n22": ("%22\nexpand_dims", LANE_C_X, 3, "offset"),
    "n19": ("%19\nexpand_dims", LANE_B_X, 4, "offset"),
    "n13": ("%13\nremsi", LANE_C_X, 4, "offset"),
    "n20": ("%20\nsplat(%arg4)\n（标量 splat，叶）", LANE_D_X, 4, "leaf"),
    "n14": ("%14\nmake_range\n（叶）", LANE_B_X, 5, "leaf"),
    "n12": ("%12\naddi", LANE_C_X, 5, "offset"),
    "n10": ("%10\nmake_range\n（叶）", LANE_C_X, 6, "leaf"),
    "n11": ("%11\nsplat(%9)\n（标量 splat，叶）", LANE_D_X, 6, "leaf"),
}
EDGES = [
    ("root", "n26"), ("root", "n25"),
    ("n26", "arg1"),
    ("n25", "n23"), ("n25", "n24"),
    ("n23", "n21"), ("n21", "n19"), ("n19", "n14"), ("n21", "n20"),
    ("n24", "n22"), ("n22", "n13"), ("n13", "n12"), ("n12", "n10"), ("n12", "n11"),
]

COLOR = {
    "root": ("#1e3a5f", "#0f172a", "white"),
    "ptr": ("#dbeafe", "#2563eb", "#1e3a8a"),
    "offset": ("#fef3c7", "#d97706", "#78350f"),
    "leaf": ("#dcfce7", "#16a34a", "#14532d"),
}

max_depth = max(d for _, _, d, _ in NODES.values())
h = TOP + (max_depth + 1) * (BOX_H + VGAP) + 90

def node_xy(key):
    _, x, depth, _ = NODES[key]
    y = TOP + depth * (BOX_H + VGAP)
    return x, y

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 边（先画，压在节点下层）
for a, b in EDGES:
    ax, ay = node_xy(a)
    bx, by = node_xy(b)
    y1 = ay + BOX_H
    y2 = by
    if abs(ax - bx) < 1:
        L.append(f'<line x1="{ax}" y1="{y1}" x2="{bx}" y2="{y2}" '
                  'stroke="#94a3b8" stroke-width="1.5" marker-end="url(#a)"/>')
    else:
        mid_y = (y1 + y2) / 2
        L.append(f'<path d="M {ax} {y1} L {ax} {mid_y} L {bx} {mid_y} L {bx} {y2}" '
                  'fill="none" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#a)"/>')

# 节点
for key, (label, x, depth, kind) in NODES.items():
    y = TOP + depth * (BOX_H + VGAP)
    fill, stroke, text_fill = COLOR[kind]
    bw = BOX_W if kind != "root" else BOX_W + 20
    L.append(f'<rect x="{x-bw/2}" y="{y}" width="{bw}" height="{BOX_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    lines = label.split("\n")
    y0 = y + BOX_H / 2 - (len(lines) - 1) * 8 + 4
    for k, line in enumerate(lines):
        L.append(f'<text x="{x}" y="{y0+k*14}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="11" font-weight="bold" fill="{text_fill}">{esc(line)}</text>')

# 深度标注侧栏（右侧标出每一深度对应的跳数，独立车道不与节点框重叠）
depth_label_x = W - PAD
for d in range(max_depth + 1):
    y = TOP + d * (BOX_H + VGAP) + BOX_H / 2 + 4
    L.append(f'<text x="{depth_label_x}" y="{y}" text-anchor="end" font-family="sans-serif" '
              f'font-size="10" fill="#94a3b8">depth {d}</text>')

# 底部小结
foot_y = h - 46
L.append(f'<rect x="{PAD}" y="{foot_y-20}" width="{W-2*PAD}" height="50" rx="8" '
          'fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+16}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'font-weight="bold" fill="#3730a3">'
          f'{esc("根节点 %27 addptr；ptr 侧深度 2（%26 splat→%arg1）；offset 侧最深路径 6（%25→%24→%22→%13→%12→make_range %10）")}</text>')
L.append(f'<text x="{PAD+16}" y="{foot_y+18}" font-family="sans-serif" font-size="11" '
          f'fill="#4338ca">{esc("muli(%21)/addi(%12) 各是二元算子，另一操作数是标量 splat（%20/%11）——全链 3 次 splat(%11/%20/%26) 均可见")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m8-recursion-tree.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
